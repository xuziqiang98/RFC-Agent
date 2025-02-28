from typing import List, Optional, Dict, Any, Union
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.models.embeddings.base import BaseEmbedding  # 修改这里：embedding -> embeddings

class FAISSRetriver:
    """FAISS 索引管理器，用于创建和管理文档的向量索引。"""
    
    def __init__(self, vector_store: FAISS):
        """
        初始化 FAISS 索引管理器。
        
        Args:
            vector_store: FAISS对象，存储文档的向量表示
        """
        self.vector_store = vector_store

    async def similarity_search_with_score(
        self,
        query: float,
        k: int = 4,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        在向量存储中执行相似度搜索，返回 top-k 相似文档及其分数。

        Args:
            query: 查询文本的嵌入向量
            k: 返回的最相似文档数量
            **kwargs: 传递给 similarity_search 的额外参数

        Returns:
            List[tuple[Document, float]]: 文档和相似度分数的元组列表，按相似度降序排列
        """
        
        # 使用向量进行搜索并返回分数
        return self.vector_store.similarity_search_with_score_by_vector(
            embedding=query,
            k=k,
            **kwargs
        )

    async def get_top_k_similar(
        self,
        query: float,
        k: int = 4,
        return_scores: bool = True
    ) -> List[Document] | List[tuple[Document, float]]:
        """
        获取与查询最相似的 top-k 文档。

        Args:
            query: 查询文本的嵌入向量
            k: 返回的最相似文档数量
            return_scores: 是否返回相似度分数

        Returns:
            如果 return_scores 为 True，返回文档和分数的元组列表
            否则仅返回文档列表
        """
        results = await self.similarity_search_with_score(query, k)
        
        if return_scores:
            return results
        return [doc for doc, _ in results]

