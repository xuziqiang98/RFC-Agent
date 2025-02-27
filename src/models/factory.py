from typing import Optional
from src.models.openai import OpenAIModel
from src.models.ollama import OllamaModel
from src.models.base import BaseModel

class LLMFactory:
    """LLM 模型工厂类，用于创建不同类型的语言模型实例。
    
    该类实现了工厂模式，集中管理所有 LLM 模型的创建逻辑。通过统一的接口创建
    不同类型的模型实例，使得模型创建过程与具体实现解耦。
    
    Attributes:
        _models: 字典，存储支持的模型类型及其对应的类引用。
            key 为模型类型字符串，value 为对应的模型类。
    """
    
    _models = {
        "openai": OpenAIModel,
        "ollama": OllamaModel
    }
    
    @classmethod
    def create(cls, 
               model_type: str, 
               model_name: str, 
               api_base: str, 
               api_key: Optional[str] = "") -> BaseModel:
        """创建指定类型的 LLM 模型实例。

        Args:
            model_type: 模型类型，支持 'openai' 或 'ollama'。
            model_name: 模型名称，如 'gpt-3.5-turbo' 或 'deepseek-r1:32b'。
            api_base: API 基础 URL。
            api_key: API 密钥，仅 OpenAI 类型模型需要。默认为空字符串。

        Returns:
            BaseModel: 创建的模型实例。

        Raises:
            ValueError: 当指定的模型类型不受支持时抛出。
        """
        model_class = cls._models.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        if model_type == "openai":
            return model_class(model_name=model_name, api_base=api_base, api_key=api_key)
        return model_class(model_name=model_name, api_base=api_base)