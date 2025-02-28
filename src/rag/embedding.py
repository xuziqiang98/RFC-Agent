from typing import List, Optional, Dict, Any, Union
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def create_faiss_index(
    texts: List[str],
    embedding_model: str = "text-embedding-ada-002",
    embedding_type: str = "openai",
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    faiss_save_path: Optional[str] = None,
    **kwargs
) -> FAISS:
    """
    将文本列表向量化并使用FAISS库构建索引
    
    Args:
        texts: 要向量化的文本列表
        embedding_model: 使用的嵌入模型名称
        embedding_type: 嵌入类型，可选 'openai' 或 'huggingface'
        embedding_kwargs: 传递给嵌入模型的额外参数
        metadatas: 与文本对应的元数据列表，每个元素是一个字典
        faiss_save_path: 保存FAISS索引的路径，如果为None则不保存
        **kwargs: 传递给FAISS.from_texts的额外参数
    
    Returns:
        构建好的FAISS向量存储对象
    
    Raises:
        ValueError: 当embedding_type不是支持的类型时
    """
    # 初始化默认参数
    if embedding_kwargs is None:
        embedding_kwargs = {}
    
    # 选择嵌入模型
    if embedding_type.lower() == "openai":
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            **embedding_kwargs
        )
    elif embedding_type.lower() == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            **embedding_kwargs
        )
    else:
        raise ValueError(f"不支持的嵌入类型: {embedding_type}，目前支持 'openai' 和 'huggingface'")
    
    # 创建FAISS索引
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        **kwargs
    )
    
    # 如果提供了保存路径，则保存索引
    if faiss_save_path:
        vector_store.save_local(faiss_save_path)
    
    return vector_store

def create_faiss_index_from_documents(
    documents: List[Document],
    embedding_model: str = "text-embedding-ada-002",
    embedding_type: str = "openai",
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    faiss_save_path: Optional[str] = None,
    **kwargs
) -> FAISS:
    """
    从Document对象列表创建FAISS索引
    
    Args:
        documents: LangChain Document对象列表
        embedding_model: 使用的嵌入模型名称
        embedding_type: 嵌入类型，可选 'openai' 或 'huggingface'
        embedding_kwargs: 传递给嵌入模型的额外参数
        faiss_save_path: 保存FAISS索引的路径，如果为None则不保存
        **kwargs: 传递给FAISS.from_documents的额外参数
    
    Returns:
        构建好的FAISS向量存储对象
    """
    # 初始化默认参数
    if embedding_kwargs is None:
        embedding_kwargs = {}
    
    # 选择嵌入模型
    if embedding_type.lower() == "openai":
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            **embedding_kwargs
        )
    elif embedding_type.lower() == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            **embedding_kwargs
        )
    else:
        raise ValueError(f"不支持的嵌入类型: {embedding_type}，目前支持 'openai' 和 'huggingface'")
    
    # 从文档创建FAISS索引
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        **kwargs
    )
    
    # 如果提供了保存路径，则保存索引
    if faiss_save_path:
        vector_store.save_local(faiss_save_path)
    
    return vector_store

def load_faiss_index(
    faiss_path: str,
    embedding_model: str = "text-embedding-ada-002",
    embedding_type: str = "openai",
    embedding_kwargs: Optional[Dict[str, Any]] = None
) -> FAISS:
    """
    加载已保存的FAISS索引
    
    Args:
        faiss_path: FAISS索引的保存路径
        embedding_model: 使用的嵌入模型名称
        embedding_type: 嵌入类型，可选 'openai' 或 'huggingface'
        embedding_kwargs: 传递给嵌入模型的额外参数
    
    Returns:
        加载的FAISS向量存储对象
    """
    # 初始化默认参数
    if embedding_kwargs is None:
        embedding_kwargs = {}
    
    # 选择嵌入模型
    if embedding_type.lower() == "openai":
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            **embedding_kwargs
        )
    elif embedding_type.lower() == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            **embedding_kwargs
        )
    else:
        raise ValueError(f"不支持的嵌入类型: {embedding_type}，目前支持 'openai' 和 'huggingface'")
    
    # 加载FAISS索引
    vector_store = FAISS.load_local(
        folder_path=faiss_path,
        embeddings=embeddings
    )
    
    return vector_store