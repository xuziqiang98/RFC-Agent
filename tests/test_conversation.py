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
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more about it."}
    ]
    
    response = await model_instance.chat(messages)
    assert isinstance(response, str)
    assert len(response) > 0
    
    breakpoint()
    
    # Test context understanding
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "What were we talking about?"})
    
    context_response = await model_instance.chat(messages)
    assert isinstance(context_response, str)
    assert len(context_response) > 0
    # 修改断言，移除可能的 Markdown 标记
    cleaned_response = context_response.lower().replace('*', '')
    assert "python" in cleaned_response
    
    breakpoint()

@pytest.mark.asyncio
async def test_chat_stream(model_instance):
    """Test the streaming chat functionality."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    chunks = []
    async for chunk in model_instance.chat_stream(messages):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    
    # Combine all chunks and verify the result
    full_response = "".join(chunks)
    assert len(full_response) > 0
    
    breakpoint()
    
    # Test streaming with context
    messages.append({"role": "assistant", "content": full_response})
    messages.append({"role": "user", "content": "Now count from 6 to 10."})
    
    context_chunks = []
    async for chunk in model_instance.chat_stream(messages):
        assert isinstance(chunk, str)
        context_chunks.append(chunk)
    
    context_response = "".join(context_chunks)
    assert len(context_response) > 0
    assert any(str(num) in context_response for num in range(6, 11))
    
    breakpoint()
