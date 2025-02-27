from typing import Optional
from src.prompts.base import BasePrompt

class CustomPrompt(BasePrompt):
    """用户自定义提示词，用于在系统提示词之后添加额外的指导。
    
    自定义提示词可以根据具体场景补充特定的指导和约束。
    """
    
    def __init__(self, content: str):
        """初始化自定义提示词。
        
        Args:
            content: 自定义提示词内容。
        """
        super().__init__(content)