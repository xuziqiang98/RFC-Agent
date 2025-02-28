import path_setup
import os
import pytest
import tempfile
import shutil
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain.schema import Document
from src.chains.rfc_chain import RFCChain
from src.models.embeddings.factory import EmbeddingFactory
from src.rag.index import FAISSIndexManager
from src.configs.common_configs import PathConfig
from src.env import Env
    
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
def temp_dir():
    """创建临时目录用于测试"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_path_config(temp_dir):
    """模拟PathConfig配置"""
    with patch('src.chains.rfc_chain.PathConfig') as mock:
        mock.return_value.dbs = Path(temp_dir) / 'dbs'
        mock.return_value.rfcs = Path(temp_dir) / 'rfcs'
        os.makedirs(mock.return_value.dbs)
        os.makedirs(mock.return_value.rfcs)
        yield mock

@pytest.fixture
def rfc_chain(embedding_model, mock_path_config):
    """创建RFCChain实例"""
    return RFCChain(embedding_model=embedding_model)

@pytest.fixture
def sample_documents():
    """创建示例文档用于测试"""
    return [
        Document(
            page_content="RFC 1234 测试文档内容",
            metadata={"source": "rfc1234"}
        ),
        Document(
            page_content="RFC 5678 测试文档内容",
            metadata={"source": "rfc5678"}
        )
    ]

# 添加导入
from src.models.embeddings.base import BaseEmbedding

@pytest.mark.asyncio
async def test_init_rfc_chain(rfc_chain):
    """测试RFCChain初始化"""
    # 修改为检查基类类型
    assert isinstance(rfc_chain.embedding_model, BaseEmbedding)
    assert isinstance(rfc_chain.index_manager, FAISSIndexManager)
    assert rfc_chain.chunk_size == 1000
    assert rfc_chain.chunk_overlap == 200

@pytest.mark.asyncio
async def test_get_all_documents(rfc_chain, sample_documents):
    """测试获取所有RFC文档"""
    with patch('src.chains.rfc_chain.get_doc_from_path') as mock_get_docs:
        mock_get_docs.return_value = sample_documents
        docs = rfc_chain._get_all_documents()
        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)

@pytest.mark.asyncio
async def test_filter_unprocessed_documents(rfc_chain, sample_documents):
    """测试筛选未处理的文档"""
    with patch('src.chains.rfc_chain.needs_processing') as mock_needs_processing:
        mock_needs_processing.return_value = True
        docs = rfc_chain._filter_unprocessed_documents(sample_documents)
        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)

@pytest.mark.asyncio
async def test_chunk_documents(rfc_chain, sample_documents):
    """测试文档切块"""
    with patch('src.chains.rfc_chain.split_documents') as mock_split:
        mock_split.return_value = [
            Document(page_content="切块1", metadata={"source": "rfc1234"}),
            Document(page_content="切块2", metadata={"source": "rfc1234"}),
            Document(page_content="切块3", metadata={"source": "rfc5678"})
        ]
        chunks = rfc_chain._chunk_documents(sample_documents)
        assert len(chunks) == 3
        assert all(isinstance(chunk, Document) for chunk in chunks)

@pytest.mark.asyncio
async def test_vectorize_and_store(rfc_chain, sample_documents, mock_path_config):
    """测试文档向量化和存储"""
    # 创建测试文件
    for doc in sample_documents:
        file_path = Path(mock_path_config.return_value.rfcs) / f"{doc.metadata['source']}.txt"
        file_path.write_text(doc.page_content)
    
    # 模拟索引管理器的create_from_documents方法
    mock_vector_store = MagicMock()
    with patch.object(rfc_chain.index_manager, 'create_from_documents') as mock_create:
        mock_create.return_value = mock_vector_store
        
        # 模拟save_index方法
        with patch.object(rfc_chain.index_manager, 'save_index') as mock_save:
            # 模拟数据库操作
            with patch('sqlite3.connect'):
                await rfc_chain._vectorize_and_store(sample_documents)
                
                # 验证create_from_documents被调用
                assert mock_create.call_count > 0
                
                # 验证save_index被调用
                assert mock_save.call_count > 0

@pytest.mark.asyncio
async def test_process_with_empty_documents(rfc_chain):
    """测试处理空文档列表的情况"""
    with patch.object(rfc_chain, '_get_all_documents') as mock_get_docs:
        mock_get_docs.return_value = []
        await rfc_chain.process()  # 添加 await
        # 验证没有继续处理
        mock_get_docs.assert_called_once()

@pytest.mark.asyncio
async def test_process_with_no_unprocessed_documents(rfc_chain, sample_documents):
    """测试没有未处理文档的情况"""
    with patch.object(rfc_chain, '_get_all_documents') as mock_get_docs:
        mock_get_docs.return_value = sample_documents
        with patch.object(rfc_chain, '_filter_unprocessed_documents') as mock_filter:
            mock_filter.return_value = []
            await rfc_chain.process()  # 添加 await
            # 验证流程正确终止
            mock_get_docs.assert_called_once()
            mock_filter.assert_called_once_with(sample_documents)

@pytest.mark.asyncio
async def test_process_complete_flow(rfc_chain, sample_documents):
    """测试完整处理流程"""
    with patch.object(rfc_chain, '_get_all_documents') as mock_get_docs:
        mock_get_docs.return_value = sample_documents
        with patch.object(rfc_chain, '_filter_unprocessed_documents') as mock_filter:
            mock_filter.return_value = sample_documents
            with patch.object(rfc_chain, '_chunk_documents') as mock_chunk:
                chunked_docs = [
                    Document(page_content="切块1", metadata={"source": "rfc1234"}),
                    Document(page_content="切块2", metadata={"source": "rfc5678"})
                ]
                mock_chunk.return_value = chunked_docs
                with patch.object(rfc_chain, '_vectorize_and_store') as mock_vectorize:
                    await rfc_chain.process()  # 添加 await
                    
                    # 验证完整流程的调用
                    mock_get_docs.assert_called_once()
                    mock_filter.assert_called_once_with(sample_documents)
                    mock_chunk.assert_called_once_with(sample_documents)
                    mock_vectorize.assert_called_once_with(chunked_docs)