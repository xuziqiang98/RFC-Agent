from typing import List, Optional, Dict, Any
import os
import sqlite3
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from src.rag.index import FAISSIndexManager
from src.rag.utils import (
    get_doc_from_path,
    needs_processing,
    split_documents,
    get_file_signature,
    init_database
)
from src.configs.common_configs import PathConfig
from src.models.embeddings.base import BaseEmbedding


class RFCChain:
    """RFC文档处理责任链。
    
    该类负责RFC文档的获取、检查、切块、向量化和持久化存储的完整流程。
    
    Attributes:
        db_path (Path): SQLite数据库路径
        rfc_docs_path (str): RFC文档存储路径
        embedding_model (BaseEmbedding): 嵌入模型实例
        index_manager (FAISSIndexManager): FAISS索引管理器
        chunk_size (int): 文档切块大小
        chunk_overlap (int): 文档切块重叠大小
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        rfc_docs_path: Optional[str] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """初始化RFC文档处理责任链。
        
        Args:
            db_path: SQLite数据库路径，默认使用配置中的路径
            rfc_docs_path: RFC文档存储路径
            embedding_model: 嵌入模型实例
            chunk_size: 文档切块大小
            chunk_overlap: 文档切块重叠大小
        """
        path_config = PathConfig()
        self.db_path = db_path or path_config.dbs / "metadata.db"
        self.rfc_docs_path = rfc_docs_path or str(path_config.rfcs)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 确保数据库初始化
        init_database(self.db_path)
        
        # 如果提供了嵌入模型，初始化索引管理器
        if self.embedding_model:
            self.index_manager = FAISSIndexManager(self.embedding_model)
    
    async def process(self) -> None:
        """执行完整的RFC文档处理流程。
        
        流程包括：
        1. 获取所有RFC文档
        2. 检查哪些文档需要处理
        3. 对文档进行切块
        4. 向量化文档
        5. 持久化存储
        
        Raises:
            ValueError: 当未提供嵌入模型时抛出
        """
        if not self.embedding_model:
            raise ValueError("必须提供嵌入模型以完成向量化步骤")
        
        # 1. 获取所有RFC文档
        all_documents = self._get_all_documents()
        if not all_documents:
            print("未找到RFC文档")
            return
        
        # 2. 检查哪些文档需要处理
        docs_to_process = self._filter_unprocessed_documents(all_documents)
        if not docs_to_process:
            print("所有文档已处理，无需更新")
            return
        
        print(f"需要处理 {len(docs_to_process)} 个文档")
        
        # 3. 对文档进行切块
        chunked_docs = self._chunk_documents(docs_to_process)
        
        # 4 & 5. 向量化并持久化存储
        await self._vectorize_and_store(chunked_docs)  # 添加 await
        
        print("RFC文档处理完成")
    
    def _get_all_documents(self) -> List[Document]:
        """获取指定路径下的所有RFC文档。
        
        Returns:
            List[Document]: 文档对象列表
        """
        return get_doc_from_path(self.rfc_docs_path)
    
    def _filter_unprocessed_documents(self, documents: List[Document]) -> List[Document]:
        """筛选出需要处理的文档。
        
        Args:
            documents: 所有文档列表
            
        Returns:
            List[Document]: 需要处理的文档列表
        """
        docs_to_process = []
        
        for doc in documents:
            source = doc.metadata.get("source")
            if not source:
                continue
                
            file_path = os.path.join(self.rfc_docs_path, f"{source}.txt")
            
            # 检查文件是否需要处理
            if needs_processing(self.db_path, file_path):
                docs_to_process.append(doc)
        
        return docs_to_process
    
    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        """将文档切分为更小的块。
        
        Args:
            documents: 需要切块的文档列表
            
        Returns:
            List[Document]: 切块后的文档列表
        """
        return split_documents(
            documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    async def _vectorize_and_store(self, documents: List[Document]) -> None:
        """向量化文档并持久化存储。
        
        Args:
            documents: 需要向量化的文档列表
        """
        # 按文档源分组
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)
        
        # 为每个源文档创建向量存储
        for source, docs in docs_by_source.items():
            # 创建向量存储
            vector_store = await self.index_manager.create_from_documents(docs)
            
            # 保存索引
            index_path = os.path.join(self.rfc_docs_path, "indices", source)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.index_manager.save_index(vector_store, index_path)
            
            # 更新数据库记录
            self._update_document_record(source, len(docs))
    
    def _update_document_record(self, source: str, chunk_count: int) -> None:
        """更新文档处理记录。
        
        Args:
            source: 文档源名称
            chunk_count: 文档块数量
        """
        file_path = os.path.join(self.rfc_docs_path, f"{source}.txt")
        file_sig = get_file_signature(file_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查记录是否存在
        cursor.execute(
            "SELECT id FROM documents WHERE filename = ?", 
            (file_sig['filename'],)
        )
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有记录
            cursor.execute(
                """UPDATE documents 
                   SET file_hash = ?, chunk_count = ?, last_modified = ?, 
                       processed_time = CURRENT_TIMESTAMP 
                   WHERE filename = ?""",
                (file_sig['hash'], chunk_count, file_sig['last_modified'], 
                 file_sig['filename'])
            )
        else:
            # 插入新记录
            cursor.execute(
                """INSERT INTO documents 
                   (filename, file_hash, chunk_count, last_modified) 
                   VALUES (?, ?, ?, ?)""",
                (file_sig['filename'], file_sig['hash'], 
                 chunk_count, file_sig['last_modified'])
            )
        
        conn.commit()
        conn.close()
    