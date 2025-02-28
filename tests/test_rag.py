import path_setup
import os
import pytest
import tempfile
import shutil
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from src.models.embeddings.base import BaseEmbedding
from src.rag.index import FAISSIndexManager
from src.rag.retriver import FAISSRetriver
from src.env import Env
from src.models.embeddings.factory import EmbeddingFactory


@pytest.fixture
def embedding_model():
    """Create a model instance based on environment configuration."""
    env = Env()
    model_type = getattr(env, "EMBEDDING_MODEL_TYPE", "openai")
    model_name = getattr(env, "EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
    api_base = getattr(env, "EMBEDDING_MODEL_BASE", "https://api.openai.com/v1")
    api_key = getattr(env, "EMBEDDING_API_KEY", "")
    
    # breakpoint()
    
    return EmbeddingFactory.create(
        model_type=model_type,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key
    )


@pytest.fixture
def index_manager(embedding_model):
    """创建一个FAISSIndexManager实例"""
    return FAISSIndexManager(embedding_model)


@pytest.fixture
def temp_dir():
    """创建一个临时目录用于保存和加载索引"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 测试结束后清理临时目录
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
async def test_create_from_texts(index_manager, embedding_model):
    """测试从文本列表创建FAISS索引"""
    # 准备测试数据
    texts = ["这是第一个测试文本", "这是第二个测试文本"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    
    # 监视embed_documents方法的调用
    with patch.object(embedding_model, 'embed_documents', wraps=embedding_model.embed_documents) as mock_embed:
        # 调用被测试的方法
        vector_store = await index_manager.create_from_texts(texts, metadatas)
        
        # 验证embed_documents被正确调用
        mock_embed.assert_called_once_with(texts)
        
        # 验证返回的是FAISS实例
        assert isinstance(vector_store, FAISS)
        
        # 验证索引包含正确数量的文档
        assert len(vector_store.docstore._dict) == len(texts)


@pytest.mark.asyncio
async def test_create_from_documents(index_manager):
    """测试从Document对象列表创建FAISS索引"""
    # 准备测试数据
    documents = [
        Document(page_content="这是第一个测试文档", metadata={"source": "doc1"}),
        Document(page_content="这是第二个测试文档", metadata={"source": "doc2"})
    ]
    
    # 调用被测试的方法
    vector_store = await index_manager.create_from_documents(documents)
    
    # 验证返回的是FAISS实例
    assert isinstance(vector_store, FAISS)
    
    # 验证索引包含正确数量的文档
    assert len(vector_store.docstore._dict) == len(documents)


@pytest.mark.asyncio
async def test_save_and_load_index(index_manager, temp_dir):
    """测试保存和加载FAISS索引"""
    # 准备测试数据并创建索引
    texts = ["这是测试文本1", "这是测试文本2"]
    vector_store = await index_manager.create_from_texts(texts)
    
    # 保存索引
    save_path = os.path.join(temp_dir, "test_index")
    index_manager.save_index(vector_store, save_path)
    
    # 验证索引文件已创建
    assert os.path.exists(save_path)
    assert os.path.exists(os.path.join(save_path, "index.faiss"))
    
    # 加载索引
    loaded_vector_store = index_manager.load_index(save_path)
    
    # 验证加载的索引是FAISS实例
    assert isinstance(loaded_vector_store, FAISS)


@pytest.fixture
def mock_vector_store():
    """创建一个模拟的FAISS向量存储"""
    mock_store = MagicMock(spec=FAISS)
    
    # 模拟similarity_search_with_score_by_vector方法
    mock_store.similarity_search_with_score_by_vector.return_value = [
        (Document(page_content="文档1", metadata={"source": "test1"}), 0.8),
        (Document(page_content="文档2", metadata={"source": "test2"}), 0.6)
    ]
    
    return mock_store


@pytest.fixture
def retriever(mock_vector_store):
    """创建一个FAISSRetriver实例"""
    # 修复FAISSRetriver初始化参数的类型注解问题
    return FAISSRetriver(mock_vector_store)


@pytest.mark.asyncio
async def test_similarity_search_with_score(retriever, mock_vector_store):
    """测试相似度搜索功能"""
    # 准备测试数据
    query_vector = [0.1, 0.2, 0.3, 0.4]
    k = 2
    
    # 调用被测试的方法
    results = await retriever.similarity_search_with_score(query_vector, k)
    
    # 验证向量存储的方法被正确调用
    mock_vector_store.similarity_search_with_score_by_vector.assert_called_once_with(
        embedding=query_vector,
        k=k
    )
    
    # 验证返回结果
    assert len(results) == 2
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)


@pytest.mark.asyncio
async def test_get_top_k_similar_with_scores(retriever):
    """测试获取Top-K相似文档（带分数）"""
    # 准备测试数据
    query_vector = [0.1, 0.2, 0.3, 0.4]
    
    # 修复get_top_k_similar方法中的错误参数
    with patch.object(retriever, 'similarity_search_with_score') as mock_search:
        mock_search.return_value = [
            (Document(page_content="文档1"), 0.8),
            (Document(page_content="文档2"), 0.6)
        ]
        
        # 调用被测试的方法，返回带分数的结果
        results = await retriever.get_top_k_similar(query_vector, k=2, return_scores=True)
        
        # 验证similarity_search_with_score被正确调用
        mock_search.assert_called_once_with(query_vector, 2)
        
        # 验证返回结果
        assert len(results) == 2
        assert isinstance(results[0][0], Document)
        assert isinstance(results[0][1], float)


@pytest.mark.asyncio
async def test_get_top_k_similar_without_scores(retriever):
    """测试获取Top-K相似文档（不带分数）"""
    # 准备测试数据
    query_vector = [0.1, 0.2, 0.3, 0.4]
    
    # 修复get_top_k_similar方法中的错误参数
    with patch.object(retriever, 'similarity_search_with_score') as mock_search:
        mock_search.return_value = [
            (Document(page_content="文档1"), 0.8),
            (Document(page_content="文档2"), 0.6)
        ]
        
        # 调用被测试的方法，返回不带分数的结果
        results = await retriever.get_top_k_similar(query_vector, k=2, return_scores=False)
        
        # 验证similarity_search_with_score被正确调用
        mock_search.assert_called_once_with(query_vector, 2)
        
        # 验证返回结果
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        
        breakpoint()