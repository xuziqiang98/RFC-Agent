from typing import Optional
from src.prompts.base import BasePrompt

class SystemPrompt(BasePrompt):
    """系统提示词类，用于定义和管理模型的基础行为提示。
    
    该类继承自 BasePrompt，用于管理大语言模型的系统级提示词。系统提示词在对话开始时
    设置，用于定义模型的基本行为、角色定位和回答规范。支持使用默认提示词或自定义提示词。
    
    Attributes:
        DEFAULT_PROMPT (str): 默认的系统提示词，定义了 RFC 文档阅读助手的基本行为规范。
            包含了对客观性、引用、格式和专业性的基本要求。
    """
    
    DEFAULT_PROMPT = f"""
        你是一个AI Agent，可以从RFC文档中挖掘信息和约束，并根据对应的具体实现分析分析代码与规范的不一致性错误。
    """

    def __init__(self, content: Optional[str] = None):
        """初始化系统提示词实例。
        
        创建一个新的系统提示词实例。如果提供了自定义内容，会将其附加到默认提示词之后。
        如果没有提供自定义内容，则仅使用默认提示词。
        
        Args:
            content: 可选的自定义提示词内容。如果提供，将会被添加到默认提示词之后。
                默认为 None，表示仅使用默认提示词。
        
        Example:
            >>> prompt = SystemPrompt()  # 使用默认提示词
            >>> prompt = SystemPrompt("额外关注安全相关内容")  # 添加自定义内容
        """
        super().__init__(f"{self.DEFAULT_PROMPT}\n{content}")