from typing import Dict, Any, Optional, AsyncGenerator, List
import json
from openai import OpenAI, AsyncOpenAI
from src.models.llms.base import BaseModel

class OpenAIModel(BaseModel):
    """OpenAI API 的模型实现类。
    
    该类实现了 BaseModel 接口，提供与 OpenAI 兼容的 API 交互功能。支持同步和流式
    两种方式生成文本，并维护多轮对话上下文。使用官方 SDK 进行 API 调用，确保了
    接口调用的可靠性和兼容性。
    
    Attributes:
        client (AsyncOpenAI): OpenAI 异步客户端实例，用于进行 API 调用。
    """
    
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        """初始化 OpenAI 模型实例。
        
        Args:
            model_name: 要使用的模型名称，如 'gpt-4'。
            api_base: API 服务的基础 URL。
            api_key: OpenAI API 密钥，用于认证。
            **kwargs: 额外的模型配置参数。
        """
        super().__init__(model_name, api_base, api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    async def generate(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Dict[str, Any]
    ) -> str:
        """生成单轮回复。
        
        使用 OpenAI API 生成对给定提示的回复。支持通过 system 参数设置系统提示，
        以指导模型的行为。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 可选的系统提示，用于设置模型的行为规范。
            **kwargs: 传递给 API 的额外参数。
        
        Returns:
            模型生成的回复文本。
        
        Raises:
            OpenAIError: 当 API 调用失败时抛出。
        """
        messages = []
        
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_stream(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """以流式方式生成回复。
        
        使用 OpenAI API 流式生成回复，逐步返回生成的文本片段。适用于需要实时
        显示生成内容的场景。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 可选的系统提示，用于设置模型的行为规范。
            **kwargs: 传递给 API 的额外参数。
        
        Yields:
            生成的文本片段。每个片段都是完整回复的一部分。
        
        Raises:
            OpenAIError: 当 API 调用失败时抛出。
        """
        messages = []
            
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    async def chat(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Dict[str, Any]
    ) -> str:
        """进行多轮对话。
        
        使用对话历史记录生成上下文相关的回复。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 系统提示，用于设置模型的行为规范。仅在首次对话时使用。
            **kwargs: 传递给 API 的额外参数。
        
        Returns:
            str: 模型生成的回复文本。
        
        Raises:
            OpenAIError: API 调用失败时抛出。
        """
        
        # 消息列表为空时，添加系统提示
        if not self._conversation_history:
            self._conversation_history.append({"role": "system", "content": system})
        
        # 添加当前用户输入
        user_message = {"role": "user", "content": prompt}
        self._conversation_history.append(user_message)
        
        # 调用 API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._conversation_history,
            **kwargs
        )
        
        # 保存助手的回复到历史
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
        self._conversation_history.append(assistant_message)
        
        return response.choices[0].message.content
    
    async def chat_stream(
        self,
        prompt: str,
        system: str = "",
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """以流式方式进行多轮对话。
        
        使用对话历史记录流式生成上下文相关的回复。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 系统提示，用于设置模型的行为规范。仅在首次对话时使用。
            **kwargs: 传递给 API 的额外参数。
        
        Yields:
            str: 生成的文本片段，每个片段都是完整回复的一部分。
        
        Raises:
            OpenAIError: API 调用失败时抛出。
        """
        
        # 消息列表为空时，添加系统提示
        if not self._conversation_history:
            self._conversation_history.append({"role": "system", "content": system})
        
        # 添加当前用户输入
        user_message = {"role": "user", "content": prompt}
        self._conversation_history.append(user_message)
        
        # 用于收集完整响应
        full_response = []
        
        # 调用流式 API
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._conversation_history,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                yield content
        
        # 保存完整回复到历史
        assistant_message = {
            "role": "assistant",
            "content": "".join(full_response)
        }
        self._conversation_history.append(assistant_message)