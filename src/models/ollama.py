from typing import Dict, Any, Optional, AsyncGenerator
import aiohttp
from src.models.base import BaseModel

class OllamaModel(BaseModel):
    """BaseModel 接口的 Ollama API 实现。
    
    此类提供了与 Ollama API 交互的方法，用于文本生成。
    它实现了普通生成和流式生成两种方法。
    """
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """使用 Ollama API 生成完整响应。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Returns：
            str: 模型生成的完整响应
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                **kwargs
            }
            if system:
                payload["system"] = system
                
            async with session.post(
                f"{self.api_base}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["response"]
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """使用 Ollama API 生成流式响应。
        
        此方法通过 Ollama API 以流式方式生成文本响应。它会逐步返回生成的文本片段，
        而不是等待整个响应完成。这对于需要实时显示生成内容的场景特别有用。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Yields：
            str: 模型生成的响应片段
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                **kwargs
            }
            if system:
                payload["system"] = system
                
            async with session.post(
                f"{self.api_base}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        chunk = await response.json()
                        if chunk.get("done", False):
                            break
                        if "response" in chunk:
                            yield chunk["response"]