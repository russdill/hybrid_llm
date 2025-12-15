"""Fixtures for testing."""
import sys
import pytest
from unittest.mock import AsyncMock, patch
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component




@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir."""
    yield

@pytest.fixture
def mock_ollama_client():
    """Mock the Ollama AsyncClient."""
    with patch("ollama.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        # Mock chat response
        mock_client.chat.return_value = {
            "model": "test-model",
            "message": {"role": "assistant", "content": "Test response"},
            "done": True
        }
        
        # Mock generate response (for pre-warmer/filler)
        mock_client.generate.return_value = {
            "response": "Test response",
            "done": True
        }
        
        # Mock tags/list
        mock_client.list.return_value = {
            "models": [
                {"name": "test-model", "size": 1000},
                {"name": "llama3", "size": 2000}
            ]
        }
        
        yield mock_client

@pytest.fixture
def mock_llm_api():
    """Mock the LLM API helpers."""
    with patch("homeassistant.helpers.llm.LLMContext"), \
         patch("homeassistant.helpers.llm.async_get_apis") as mock_get_apis, \
         patch("homeassistant.helpers.llm.async_get_api") as mock_get_api:
         
        mock_api = AsyncMock()
        mock_api.id = "test_api"
        mock_api.name = "Test API"
        mock_api.tools = []
        mock_api.async_call_tool = AsyncMock(return_value={"success": True, "result": "Tool executed"})
        
        mock_get_apis.return_value = [mock_api]
        mock_get_api.return_value = mock_api
        
        yield mock_get_api

@pytest.fixture(autouse=True)
async def setup_dependencies(hass):
    """Setup dependencies to avoid failures in dependencies."""
    # Setup conversation component
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "media_player", {})
    
    return
