"""Test the Hybrid Conversation Agent Failure Modes."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import ollama
from homeassistant.core import HomeAssistant, Context
from homeassistant.components import conversation
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hybrid_llm import DOMAIN
from custom_components.hybrid_llm.const import CONF_URL
from custom_components.hybrid_llm.conversation import HybridConversationAgent


@pytest.fixture
def agent(hass):
    """Return a initialized agent."""
    entry = MockConfigEntry(
        domain=DOMAIN, 
        data={}, 
        options={
            "url": "http://mock-ollama",
            "model": "mock-model",
        }
    )
    hass.data.setdefault(DOMAIN, {"cache": {}})
    entry.add_to_hass(hass)
    return HybridConversationAgent(hass, entry)


@pytest.fixture
def mock_chat_log():
    """Create a mock ChatLog."""
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.llm_api = None
    chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="Hello"),
    ]
    chat_log.unresponded_tool_results = False
    chat_log.async_provide_llm_data = AsyncMock()
    
    async def mock_add_delta_stream(agent_id, stream):
        async for delta in stream:
            yield MagicMock(content=delta.get("content", ""))
    
    chat_log.async_add_delta_content_stream = mock_add_delta_stream
    
    return chat_log


async def test_ollama_connection_error(hass: HomeAssistant, agent, mock_ollama_client, mock_llm_api, mock_chat_log) -> None:
    """Test handling of Ollama connection errors."""
    
    # Filler succeeds but chat fails
    mock_ollama_client.generate.return_value = {"response": ""}

    # Force connection error on chat
    mock_ollama_client.chat.side_effect = ollama.RequestError("Connection Refused")
    
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="fail_test",
        device_id=None,
        satellite_id=None,
        agent_id="test_agent",
        language="en"
    )
    
    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default

        with pytest.raises(HomeAssistantError, match="problem talking to the Ollama server"):
            await agent._async_handle_message(user_input, mock_chat_log)

    
async def test_ollama_500_error(hass: HomeAssistant, agent, mock_ollama_client, mock_llm_api, mock_chat_log) -> None:
    """Test graceful handling of Ollama 500 errors."""
    
    mock_ollama_client.generate.return_value = {"response": ""}

    # Force 500 error
    mock_ollama_client.chat.side_effect = ollama.ResponseError("Internal Server Error")
    
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="fail_500",
        device_id=None,
        satellite_id=None,
        agent_id="test_agent",
        language="en"
    )
    
    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default

        with pytest.raises(HomeAssistantError, match="problem talking to the Ollama server"):
            await agent._async_handle_message(user_input, mock_chat_log)
