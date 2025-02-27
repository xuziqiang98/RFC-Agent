from typing import Any, Dict, List, Optional

class OutputParser:
    """大语言模型输出解析器。
    
    该类负责解析和标准化大语言模型的输出，将其转换为应用程序可以使用的结构化数据。
    
    Attributes:
        _cache: 解析结果缓存字典。
    """
    
    def __init__(self):
        """初始化解析器。"""
        self._cache: Dict[str, Any] = {}
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析模型输出文本。

        Args:
            text: 待解析的模型输出文本。

        Returns:
            Dict[str, Any]: 解析后的结构化数据。

        Raises:
            ValueError: 当输入文本格式不正确时抛出。
        """
        pass
    
    def parse_stream(
        self,
        chunk: str,
        stream_id: Optional[str] = None,
        **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析模型的流式输出数据。

        该方法用于处理大语言模型的流式响应，将每个文本块解析为结构化数据。
        支持使用 stream_id 跟踪和关联同一会话的多个数据块。

        Args:
            chunk: 当前接收到的文本块。
            stream_id: 可选的流标识符，用于关联同一会话的多个数据块。
            **kwargs: 额外的解析参数。

        Returns:
            Dict[str, Any]: 解析后的结构化数据，包含：
                - text: 解析后的文本内容
                - metadata: 解析元数据
                - stream_id: 流标识符（如果提供）

        Raises:
            ValueError: 当输入文本格式不正确时抛出。
            RuntimeError: 当流处理出现错误时抛出。
        """
        pass
    
    def clear_cache(self) -> None:
        """清除解析缓存。"""
        self._cache.clear()