from typing import Optional

class BasePrompt:
    """提示词基类，定义了基本的提示词接口。
    
    Attributes:
        content: 提示词内容。
    """
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self) -> str:
        return self.content