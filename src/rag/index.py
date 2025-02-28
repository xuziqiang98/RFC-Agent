from typing import List, Optional, Dict, Any, Union
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.models.embeddings.base import BaseEmbedding

class FAISSIndexManager:
    """FAISS 索引管理器，用于创建和管理文档的向量索引。"""
    
    def __init__(self, embedding_model: BaseEmbedding):
        """
        初始化 FAISS 索引管理器。
        
        Args:
            embedding_model: 实现了 BaseEmbedding 接口的嵌入模型实例
        """
        self.embedding_model = embedding_model
    
    async def create_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> FAISS:
        """
        从文本列表创建 FAISS 索引。
        
        Args:
            texts: 要建立索引的文本列表
            metadatas: 与文本对应的元数据列表
            **kwargs: 传递给 FAISS.from_texts 的额外参数
        
        Returns:
            FAISS: 构建好的向量存储对象
        """
        # 获取文本的嵌入向量
        embeddings = await self.embedding_model.embed_documents(texts)
        
        # 创建文本和嵌入向量的元组列表
        text_embedding_pairs = list(zip(texts, embeddings))
        
        # 创建 FAISS 索引
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embedding_model,
            metadatas=metadatas,
            **kwargs
        )
        
        return vector_store
    
    async def create_from_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> FAISS:
        """
        从 Document 对象列表创建 FAISS 索引。
        
        Args:
            documents: LangChain Document 对象列表
            **kwargs: 传递给 FAISS.from_documents 的额外参数
        
        Returns:
            FAISS: 构建好的向量存储对象
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        return await self.create_from_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs
        )
    
    def save_index(self, vector_store: FAISS, save_path: str) -> None:
        """
        将 FAISS 索引保存到指定路径。
        
        Args:
            vector_store: 要保存的 FAISS 索引对象
            save_path: 保存路径，应该是一个目录路径
        """
        vector_store.save_local(save_path)
    
    def check_document_exists(self, doc_name: str, index_path: str) -> bool:
        """
        检查指定文档是否已存在于本地索引中。
        
        Args:
            index_path: 索引文件路径
            doc_name: 文档名称，如果提供则同时检查文档名
            
        Returns:
            bool: 如果文档存在返回True，否则返回False
        """
        try:
            if not os.path.exists(index_path):
                return False
                
            vector_store = self.load_index(index_path)
            existing_docs = vector_store.docstore.search({})
            
            for doc in existing_docs:
                if doc.metadata.get("source") == doc_name:
                    return True
            return False
            
        except Exception as e:
            print(f"检查文档存在性时发生错误: {e}")
            return False

    def load_index(self, load_path: str) -> FAISS:
        """
        从指定路径加载 FAISS 索引。
        
        Args:
            load_path: 加载路径，应该是保存索引的目录路径
        
        Returns:
            FAISS: 加载的向量存储对象
        """
        # 创建一个自定义的 Embeddings 类来包装我们的嵌入模型
        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model
            
            async def embed_documents(self, texts):
                return await self.model.embed_documents(texts)
            
            async def embed_query(self, text):
                return await self.model.embed_query(text)
        
        embeddings = CustomEmbeddings(self.embedding_model)
        return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
