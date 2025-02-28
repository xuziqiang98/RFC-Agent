import path_setup
import os
import pytest
import asyncio
from src.models.openai import OpenAIModel
from src.models.ollama import OllamaModel
from src.env import Env
from src.models.factory import LLMFactory

@pytest.fixture
def model_instance():
    """Create a model instance based on environment configuration."""
    env = Env()
    model_type = getattr(env, "DEFAULT_MODEL_TYPE", "ollama")
    model_name = getattr(env, "OPENAI_MODEL_NAME", "deepseek-r1:32b")
    api_base = getattr(env, "OPENAI_API_BASE", "http://localhost:11434")
    api_key = getattr(env, "OPENAI_API_KEY", "")
    
    # breakpoint()
    
    return LLMFactory.create(
        model_type=model_type,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key
    )
    
@pytest.mark.asyncio
async def test_chat(model_instance):
    """Test the chat functionality with multiple messages."""
    # 系统提示
    system_prompt = "你是一个有用的AI助手。"
    
    # 第一轮对话
    first_response = await model_instance.chat(
        prompt="你好，请介绍一下自己。",
        system=system_prompt
    )
    assert isinstance(first_response, str)
    assert len(first_response) > 0
    
    # 验证历史记录是否正确保存
    assert len(model_instance._conversation_history) == 3
    
    # breakpoint()
    
    # 第二轮对话，测试上下文保持
    second_response = await model_instance.chat(
        prompt="我们刚才聊了什么？"
    )
    assert isinstance(second_response, str)
    assert len(second_response) > 0
    
    assert len(model_instance._conversation_history) == 5
    
    # breakpoint()
    
    # 清除对话历史
    model_instance.clear_history()
    assert len(model_instance._conversation_history) == 0


@pytest.mark.asyncio
async def test_chat_stream(model_instance):
    """Test the streaming chat functionality."""
    # 系统提示
    system_prompt = "你是一个有用的AI助手。"
    
    # 测试流式响应
    chunks = []
    async for chunk in model_instance.chat_stream(
        prompt="请用三句话描述人工智能的发展历程。",
        system=system_prompt
    ):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    
    # 验证收到了多个文本块
    assert len(chunks) > 0
    
    # 验证历史记录是否正确保存
    assert len(model_instance._conversation_history) == 3  # 系统消息、用户消息和助手回复
    assert model_instance._conversation_history[0]["role"] == "system"
    assert model_instance._conversation_history[1]["role"] == "user"
    assert model_instance._conversation_history[2]["role"] == "assistant"
    
    # 测试多轮流式对话
    chunks = []
    async for chunk in model_instance.chat_stream(
        prompt="继续补充一点关于AI未来发展的看法。"
    ):
        chunks.append(chunk)
    
    # 验证收到了响应
    assert len(chunks) > 0
    
    # 验证历史记录长度增加
    assert len(model_instance._conversation_history) == 5  # 新增用户消息和助手回复

