from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
import copy

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
        self._conversation_history: List[Dict[str, str]] = []
    
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

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Dict[str, Any]
    ) -> str:
        """进行多轮对话并生成响应。
        
        Args：
            messages: 对话历史消息列表，每条消息包含 'role' 和 'content' 字段
            **kwargs: 额外的生成参数
            
        Returns：
            str: 模型生成的响应
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """进行多轮对话并生成流式响应。
        
        Args：
            messages: 对话历史消息列表，每条消息包含 'role' 和 'content' 字段
            **kwargs: 额外的生成参数
            
        Yields：
            str: 模型生成的响应片段
        """
        pass    
    
    def clear_history(self) -> None:
        """清除对话历史记录。
        
        此方法会清空当前实例的对话历史记录，使其回到初始状态。
        这在需要开始新的对话时特别有用。
        """
        self._conversation_history.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取当前对话的历史记录。
        
        Returns:
            对话历史消息列表，每条消息包含 'role' 和 'content' 字段。
        """
        
        return copy.deepcopy(self._conversation_history)  # 使用深拷贝替代浅拷贝