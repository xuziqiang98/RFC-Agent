from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator

class BaseModel(ABC):
    """所有语言模型的基类。
    
    这个抽象类定义了所有模型实现必须遵循的接口。
    它提供了一个统一的方式来与不同类型的语言模型（如 Ollama、OpenAI）进行交互。
    
    Args：
        model_name (str): 要使用的模型名称
        api_base (str): API 端点的基础 URL
        api_key (Optional[str]): 认证所需的 API 密钥（如果需要）
        **kwargs: 额外的模型特定配置选项
    """
    
    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.kwargs = kwargs
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> str:
        """根据输入的提示生成响应。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Returns：
            str: 模型生成的响应
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """根据输入的提示生成流式响应。
        
        Args：
            prompt (str): 用户的输入提示
            system (Optional[str]): 用于指导模型行为的系统消息
            **kwargs: 额外的生成参数
            
        Yields：
            str: 模型生成的响应片段
        """
        pass