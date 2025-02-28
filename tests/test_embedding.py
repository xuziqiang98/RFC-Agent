import path_setup
import os
import pytest
import asyncio
import numpy as np
from src.models.embedding.openai import OpenAIEmbedding
from src.models.embedding.ark import ArkEmbedding
from src.env import Env
from src.models.embedding.factory import EmbeddingFactory

@pytest.fixture
def model_instance():
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

@pytest.mark.asyncio
async def test_embed_query(model_instance):
    """测试单个文本嵌入功能。"""
    text = "这是一个测试文本，用于生成嵌入向量。"
    embedding = await model_instance.embed_query(text)
    
    # 验证嵌入向量是否为列表类型
    assert isinstance(embedding, list)
    # 验证嵌入向量长度是否符合预期
    assert len(embedding) == model_instance.get_dimension()
    # 验证嵌入向量元素是否为浮点数
    assert all(isinstance(value, float) for value in embedding)

@pytest.mark.asyncio
async def test_embed_documents(model_instance):
    """测试多个文档文本嵌入功能。"""
    texts = [
        "这是第一个测试文本。",
        "这是第二个测试文本，内容不同。",
        "这是第三个测试文本，用于测试多文档嵌入。"
    ]
    embeddings = await model_instance.embed_documents(texts)
    
    # 验证返回结果是否为列表
    assert isinstance(embeddings, list)
    # 验证返回的嵌入向量数量是否与输入文本数量相同
    assert len(embeddings) == len(texts)
    # 验证每个嵌入向量的维度是否正确
    for embedding in embeddings:
        assert len(embedding) == model_instance.get_dimension()
        assert all(isinstance(value, float) for value in embedding)

@pytest.mark.asyncio
async def test_embedding_similarity(model_instance):
    """测试嵌入向量的相似度计算。"""
    text1 = "人工智能是计算机科学的一个分支。"
    text2 = "机器学习是人工智能的一个子领域。"
    text3 = "足球是一项团队运动。"
    
    # 获取嵌入向量
    embedding1 = await model_instance.embed_query(text1)
    embedding2 = await model_instance.embed_query(text2)
    embedding3 = await model_instance.embed_query(text3)
    
    # 计算余弦相似度
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # 计算相似度
    sim_1_2 = cosine_similarity(embedding1, embedding2)
    sim_1_3 = cosine_similarity(embedding1, embedding3)
    sim_2_3 = cosine_similarity(embedding2, embedding3)
    
    # 相关主题的相似度应该高于不相关主题
    assert sim_1_2 > sim_1_3
    assert sim_1_2 > sim_2_3

    
