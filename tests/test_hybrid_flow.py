"""Test the Hybrid Conversation Agent flow logic."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from homeassistant.components import conversation

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hybrid_llm import DOMAIN
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
    chat_log.content = []
    chat_log.unresponded_tool_results = False
    chat_log.async_provide_llm_data = AsyncMock()
    
    async def mock_add_delta_stream(agent_id, stream):
        async for delta in stream:
            pass
        yield {"content": ""} # Yield something to make it a generator
    chat_log.async_add_delta_content_stream = mock_add_delta_stream
    return chat_log

@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", [
    {
        "text": "Turn on the kitchen light",
        "intent_match": True,
        "expect_llm": False,
        "response_speech": "Turned on kitchen light"
    },
    {
        "text": "Turn on kitchin light",  # Fuzzy match simulated
        "intent_match": True,
        "expect_llm": False,
        "response_speech": "Turned on kitchen light"
    },
    {
        "text": "What is the capital of France?",
        "intent_match": False,
        "expect_llm": True,
        "response_speech": "Paris"
    },
    {
        "text": "Make it weird in here", # Complex command, no native intent
        "intent_match": False,
        "expect_llm": True,
        "response_speech": "Okay, making it weird."
    }
])
async def test_hybrid_flow_routing(
    hass, 
    agent, 
    mock_ollama_client, 
    mock_chat_log, 
    test_case
):
    """Test routing between Native Intents and LLM based on input."""
    
    user_input = conversation.ConversationInput(
        text=test_case["text"],
        context=MagicMock(),
        conversation_id="test_flow",
        device_id="dev1",
        language="en",
        agent_id="test_agent",
        satellite_id=None,
    )

    # Mock default agent logic
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        
        if test_case["intent_match"]:
            # Simulate successful native matching
            mock_default.async_debug_recognize.return_value = {
                "match": True, 
                "fuzzy_match": False
            }
            
            mock_process_result = MagicMock()
            mock_process_result.response.speech = {"plain": {"speech": test_case["response_speech"]}}
            mock_default.async_process.return_value = mock_process_result
        else:
            # Simulate no match
            mock_default.async_debug_recognize.return_value = {"match": False}
            
        mock_get_default.return_value = mock_default

        # Mock LLM logic
        mock_ollama_client.generate.return_value = {"response": "Thinking..."}
        
        async def mock_chat_stream(*args, **kwargs):
            yield {"message": {"content": test_case["response_speech"]}}
        
        mock_ollama_client.chat.return_value = mock_chat_stream()

        # Mock result parsing from chat log (for LLM path)
        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_chat_result = MagicMock()
            mock_chat_result.response.speech = {"plain": {"speech": test_case["response_speech"]}}
            mock_get_result.return_value = mock_chat_result

            result = await agent._async_handle_message(user_input, mock_chat_log)

            # Assertions
            if test_case["expect_llm"]:
                assert mock_ollama_client.chat.called, f"LLM should have been called for '{test_case['text']}'"
                # For no match, we just don't call process
                mock_default.async_process.assert_not_called()
            else:
                assert not mock_ollama_client.chat.called, f"LLM should NOT have been called for '{test_case['text']}'"
                # For successful match, we call process
                mock_default.async_process.assert_called_once()
            
            assert result.response.speech["plain"]["speech"] == test_case["response_speech"]
