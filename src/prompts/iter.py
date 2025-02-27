from typing import Dict
from src.prompts.base import BasePrompt

class IterPrompt(BasePrompt):
    """预设提示词集合，存储多个预定义的提示词模板。"""
    
    PROMPTS = {
        "summary": "请总结这段 RFC 文档的主要内容。",
        "explain": "请解释这段 RFC 文档中的专业术语和概念。",
        "compare": "请比较这段 RFC 文档与之前版本的主要区别。",
        "security": "请分析这段 RFC 文档中的安全相关考虑。",
        "example": "请给出这段 RFC 文档描述的协议或机制的具体使用示例。"
    }
    
    def __init__(self, prompt_key: str = "summary"):
        """初始化预设提示词。
        
        Args:
            prompt_key: 预设提示词的键名，默认使用 summary。
        """
        if prompt_key not in self.PROMPTS:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        super().__init__(self.PROMPTS[prompt_key])
    
    @classmethod
    def get_prompt(cls, prompt_key: str) -> str:
        """获取指定的预设提示词。
        
        Args:
            prompt_key: 预设提示词的键名。
        
        Returns:
            对应的提示词内容。
        """
        return cls.PROMPTS.get(prompt_key, cls.PROMPTS["summary"])