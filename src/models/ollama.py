from typing import Dict, Any, Optional, AsyncGenerator, List
from ollama import AsyncClient
from src.models.base import BaseModel

class OllamaModel(BaseModel):
    """Ollama API 的模型实现类。
    
    该类实现了 BaseModel 接口，提供与 Ollama API 的交互功能。支持单轮生成和多轮对话，
    每种模式都提供同步和流式两种响应方式。使用官方 SDK 进行 API 调用，确保了接口调用
    的可靠性和兼容性。
    
    Attributes:
        client: Ollama 异步客户端实例，用于进行 API 调用。
    """
    
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        """初始化 Ollama 模型实例。
        
        Args:
            model_name: 要使用的模型名称，如 'llama2' 或 'mistral'。
            api_base: Ollama 服务的基础 URL。
            api_key: API 密钥（Ollama 当前不需要）。
            **kwargs: 额外的模型配置参数。
        """
        super().__init__(model_name, api_base, api_key, **kwargs)
        self.client = AsyncClient(host=self.api_base)
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """生成单轮回复。
        
        使用 Ollama API 生成对给定提示的回复。支持通过 system 参数设置系统提示，
        以指导模型的行为。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 可选的系统提示，用于设置模型的行为规范。
            **kwargs: 传递给 API 的额外参数，如 temperature、top_p 等。
        
        Returns:
            模型生成的回复文本。
        
        Raises:
            OllamaAPIError: 当 API 调用失败时抛出。
        """
        response = await self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            system=system,
            stream=False,
            **kwargs
        )
        return response['message']['content']
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """以流式方式生成单轮回复。
        
        使用 Ollama API 流式生成回复，逐步返回生成的文本片段。适用于需要实时
        显示生成内容的场景，可以提供更好的用户体验。
        
        Args:
            prompt: 用户输入的提示文本。
            system: 可选的系统提示，用于设置模型的行为规范。
            **kwargs: 传递给 API 的额外参数，如 temperature、top_p 等。
        
        Yields:
            生成的文本片段。每个片段都是完整回复的一部分。
        
        Raises:
            OllamaAPIError: 当 API 调用失败时抛出。
        """
        async for chunk in self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            system=system,
            stream=True,
            **kwargs
        ):
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Dict[str, Any]
    ) -> str:
        """进行多轮对话。
        
        使用完整的对话历史生成回复，支持上下文理解和多轮交互。每条消息都包含角色
        和内容信息，用于构建对话上下文。
        
        Args:
            messages: 对话历史列表，每条消息包含 'role' 和 'content' 字段。
                role 可以是 'system'、'user' 或 'assistant'。
            **kwargs: 传递给 API 的额外参数，如 temperature、top_p 等。
        
        Returns:
            模型生成的回复文本。
        
        Raises:
            OllamaAPIError: 当 API 调用失败时抛出。
        """
        response = await self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            **kwargs
        )
        return response['message']['content']
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """以流式方式进行多轮对话。
        
        使用完整的对话历史流式生成回复，支持上下文理解和实时输出。适用于需要即时
        反馈的交互式对话场景。
        
        Args:
            messages: 对话历史列表，每条消息包含 'role' 和 'content' 字段。
                role 可以是 'system'、'user' 或 'assistant'。
            **kwargs: 传递给 API 的额外参数，如 temperature、top_p 等。
        
        Yields:
            生成的文本片段。每个片段都是完整回复的一部分。
        
        Raises:
            OllamaAPIError: 当 API 调用失败时抛出。
        """
        async for chunk in self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        ):
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']