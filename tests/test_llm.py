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
async def test_generate(model_instance):
    """Test the synchronous generation functionality."""
    response = await model_instance.generate("hello")
    assert isinstance(response, str)
    assert len(response) > 0
    breakpoint()

@pytest.mark.asyncio
async def test_generate_stream(model_instance):
    """Test the streaming generation functionality."""
    chunks = []
    async for chunk in model_instance.generate_stream("hello"):
        assert isinstance(chunk, str)
        chunks.append(chunk)
    
    # Combine all chunks and verify the result
    full_response = "".join(chunks)
    assert len(full_response) > 0
    breakpoint()