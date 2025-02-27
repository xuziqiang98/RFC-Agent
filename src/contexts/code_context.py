import ast
from pathlib import Path
from typing import Dict, Optional, List
from src.configs.common_configs import PathConfig

class CodeContext:
    """代码实现上下文管理器。
    
    该类负责读取和解析具体的协议实现代码，将其转换为 AST 树形结构，
    便于后续的代码分析和规范对比。
    
    Attributes:
        _cache (Dict[str, ast.AST]): AST 解析结果的缓存。
        _impl_dir (Path): 协议实现代码的根目录。
    """
    
    def __init__(self, impl_dir: Optional[Path] = None):
        """初始化代码上下文管理器。
        
        Args:
            impl_dir: 协议实现代码的根目录路径，默认为项目根目录下的 impls 目录。
        """
        self._cache: Dict[str, ast.AST] = {}
        self._impl_dir = impl_dir or PathConfig().impls
    
    def load_implementation(self, protocol: str) -> ast.AST:
        """加载指定协议的实现代码并解析为 AST。
        
        Args:
            protocol: 协议名称，如 'http', 'websocket' 等。
            
        Returns:
            ast.AST: 解析后的 AST 树。
            
        Raises:
            FileNotFoundError: 当指定的协议实现代码不存在时抛出。
            SyntaxError: 当代码解析失败时抛出。
        """
        pass
    
    def get_ast_node(self, protocol: str, node_path: List[str]) -> ast.AST:
        """获取 AST 树中指定路径的节点。
        
        Args:
            protocol: 协议名称。
            node_path: 节点路径列表，如 ['class_name', 'method_name']。
            
        Returns:
            ast.AST: 指定路径的 AST 节点。
            
        Raises:
            KeyError: 当指定的节点路径不存在时抛出。
        """
        pass
    
    def clear_cache(self) -> None:
        """清除 AST 解析缓存。"""
        self._cache.clear()