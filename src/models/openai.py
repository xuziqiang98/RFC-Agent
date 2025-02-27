from typing import Dict, Any, Optional, AsyncGenerator
import json
import aiohttp
from src.models.base import BaseModel

class OpenAIModel(BaseModel):
    """BaseModel 接口的 OpenAI 兼容 API 实现。
    
    此类提供了与 OpenAI 兼容的 API 交互方法，用于文本生成。
    它支持使用聊天补全端点进行普通生成和流式生成两种方法。
    """
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """使用 OpenAI 兼容的 API 生成完整响应。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Returns：
            str: 模型生成的完整响应
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "messages": messages,
                **kwargs
            }
            
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """使用 OpenAI 兼容的 API 生成流式响应。
        
        此方法通过 OpenAI 兼容的 API 以流式方式生成文本响应。它会逐步返回生成的文本片段，
        而不是等待整个响应完成。这对于需要实时显示生成内容的场景特别有用。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Yields：
            str: 模型生成的响应片段
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                **kwargs
            }
            
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                            try:
                                chunk = json.loads(line)
                                # 修改这里的逻辑，不再依赖 finish_reason
                                if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                                    content = chunk["choices"][0]["delta"].get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue