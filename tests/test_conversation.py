"""Test the Hybrid Conversation Agent."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from homeassistant.core import HomeAssistant, Context
from homeassistant.components import conversation
from homeassistant.helpers import intent
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hybrid_llm import DOMAIN
from custom_components.hybrid_llm.conversation import HybridConversationAgent


@pytest.fixture
def mock_chat_log():
    """Create a mock ChatLog."""
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.llm_api = None
    chat_log.content = []
    chat_log.unresponded_tool_results = False
    
    # Mock async_provide_llm_data
    chat_log.async_provide_llm_data = AsyncMock()
    
    # Mock async_add_delta_content_stream to collect deltas
    collected_content = []
    
    async def mock_add_delta_stream(agent_id, stream):
        async for delta in stream:
            if content := delta.get("content"):
                collected_content.append(content)
            yield MagicMock(content="".join(collected_content))
    
    chat_log.async_add_delta_content_stream = mock_add_delta_stream
    chat_log._collected_content = collected_content
    
    return chat_log


@pytest.mark.asyncio
async def test_native_intent_fast_path(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test that native intents are handled by the default agent, bypassing Ollama."""
    agent = HybridConversationAgent(hass, MockConfigEntry(options={}))
    
    user_input = conversation.ConversationInput(
        text="turn on light",
        context=MagicMock(),
        conversation_id="123",
        device_id="dev1",
        language="en",
        agent_id="homeassistant",
        satellite_id=None,
    )
    
    # Mock default agent that matches a native intent
    # Mock default agent that matches a native intent
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        
        # Mock async_recognize_intent (peek check)
        mock_recognize_result = MagicMock()
        mock_default.async_recognize_intent.return_value = mock_recognize_result
        
        # Mock internal_async_process (execution)
        mock_native_result = MagicMock()
        mock_native_result.response.intent = MagicMock()  # Indicates a native intent matched
        mock_native_result.response.response_type = intent.IntentResponseType.ACTION_DONE
        mock_native_result.response.speech = {"plain": {"speech": "Turned on the light"}}
        mock_default.internal_async_process.return_value = mock_native_result
        
        mock_get_default.return_value = mock_default
        
        result = await agent._async_handle_message(user_input, mock_chat_log)
        
        # Should have checked intent and then processed it
        mock_default.async_recognize_intent.assert_called_once_with(user_input)
        
        # Verify internal_async_process called with temp ID
        assert mock_default.internal_async_process.called
        args, _ = mock_default.internal_async_process.call_args
        called_input = args[0]
        assert called_input.text == user_input.text
        assert called_input.conversation_id != user_input.conversation_id
        assert called_input.conversation_id.startswith("_temp_")
        
        # Verify result has original ID restored
        assert result.conversation_id == user_input.conversation_id
        
        # Should return native intent result
        assert result.response.intent is not None
        assert result.response.speech["plain"]["speech"] == "Turned on the light"
        
        # Should NOT have called Ollama since native intent matched
        assert not mock_ollama_client.chat.called


@pytest.mark.asyncio
async def test_llm_fallback_when_no_native_intent(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test that LLM is used when no native intent matches."""
    agent = HybridConversationAgent(hass, MockConfigEntry(options={}))
    
    user_input = conversation.ConversationInput(
        text="what's the weather",
        context=MagicMock(),
        conversation_id="123",
        device_id="dev1",
        language="en",
        agent_id="homeassistant",
        satellite_id=None,
    )
    
    # Setup mock for chat_log.async_provide_llm_data
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="what's the weather"),
    ]
    
    # Mock Ollama responses
    mock_ollama_client.generate.return_value = {"response": ""}
    
    async def mock_chat_stream(*args, **kwargs):
        yield {"message": {"content": "I don't have weather info"}}
    
    mock_ollama_client.chat.return_value = mock_chat_stream()
    
    # Mock default agent that does NOT match any native intent
    # Mock default agent that does NOT match any native intent
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        
        # Mock async_recognize_intent returning None (no match)
        mock_default.async_recognize_intent.return_value = None
        
        mock_get_default.return_value = mock_default
        
        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.intent = None
            mock_result.response.speech = {"plain": {"speech": "I don't have weather info"}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(user_input, mock_chat_log)
            
            # Should have called Ollama since no native intent matched
            assert mock_ollama_client.chat.called
            
            assert result.response.speech["plain"]["speech"] == "I don't have weather info"


@pytest.mark.asyncio
async def test_ollama_process_streaming(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test streaming response through ChatLog."""
    entry = MockConfigEntry(data={}, options={"url": "http://ollama"})
    agent = HybridConversationAgent(hass, entry)
    agent.hass.data[DOMAIN] = {"cache": {}}
    
    # Mock Filler
    mock_ollama_client.generate.return_value = {"response": "Checking"}

    # Mock Main Chat
    async def mock_chat_stream(*args, **kwargs):
        yield {"message": {"content": "Hello "}}
        yield {"message": {"content": "World"}}
    
    mock_ollama_client.chat.return_value = mock_chat_stream()

    # Mock chat_log.content with system and user messages
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="hello"),
    ]

    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default

        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.speech = {"plain": {"speech": "Checking... Hello World"}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(
                conversation.ConversationInput(
                    text="hello",
                    context=MagicMock(),
                    conversation_id="123",
                    device_id="dev1",
                    language="en",
                    agent_id="test_agent",
                    satellite_id=None,
                ),
                mock_chat_log
            )
    
        # Verify filler was generated
        assert mock_ollama_client.generate.called
        
        # Verify chat was called
        assert mock_ollama_client.chat.called
        
        # Verify result was built from chat_log
        assert mock_get_result.called


@pytest.mark.asyncio
async def test_speech_is_string_not_generator(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Regression test: Verify speech is a string (not generator)."""
    entry = MockConfigEntry(data={}, options={"url": "http://ollama"})
    agent = HybridConversationAgent(hass, entry)
    agent.hass.data[DOMAIN] = {"cache": {}}

    mock_ollama_client.generate.return_value = {"response": "Checking"}
    
    async def mock_chat_stream(*args, **kwargs):
        yield {"message": {"content": "Hello"}}

    mock_ollama_client.chat.return_value = mock_chat_stream()
    
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="test"),
    ]

    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default

        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.speech = {"plain": {"speech": "Checking... Hello"}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(
                conversation.ConversationInput(
                    text="test",
                    context=MagicMock(),
                    conversation_id="123",
                    device_id=None,
                    language="en",
                    agent_id="test_agent",
                    satellite_id=None,
                ),
                mock_chat_log
            )
    
        speech = result.response.speech["plain"]["speech"]
        
        # Critical check: speech must be a string so .strip() works
        assert isinstance(speech, str), f"Speech must be a string, got {type(speech)}"
        assert speech.strip()  # Must not raise AttributeError


@pytest.mark.asyncio
async def test_tool_use_loop(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test tool calling loop."""
    entry = MockConfigEntry(data={}, options={
        "url": "http://ollama",
        "llm_hass_api": "assist"
    })
    agent = HybridConversationAgent(hass, entry)
    agent.hass.data[DOMAIN] = {"cache": {}}

    mock_ollama_client.generate.return_value = {"response": "Working"}

    # First call: tool call, second call: response
    call_count = 0
    async def mock_chat_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "test_tool",
                            "arguments": {"param": "value"}
                        }
                    }]
                }
            }
        else:
            yield {"message": {"content": "Done"}}

    mock_ollama_client.chat.side_effect = mock_chat_stream

    # Set up chat_log with LLM API for tool support
    mock_api = MagicMock()
    mock_api.tools = []
    mock_chat_log.llm_api = mock_api
    mock_chat_log.unresponded_tool_results = False
    
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="do something"),
    ]

    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default

        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.speech = {"plain": {"speech": "Working... Done"}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(
                conversation.ConversationInput(
                    text="do something",
                    context=MagicMock(),
                    conversation_id="123",
                    device_id="dev1",
                    language="en",
                    agent_id="test_agent",
                    satellite_id=None,
                ),
                mock_chat_log
            )
        
            assert result.response.speech["plain"]["speech"] == "Working... Done"


@pytest.mark.asyncio
async def test_filler_called_only_once_in_tool_loop(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test that filler is generated only once even when tool loop runs multiple iterations."""
    agent = HybridConversationAgent(hass, MockConfigEntry(options={}))
    
    user_input = conversation.ConversationInput(
        text="turn on kitchen light",
        context=MagicMock(),
        conversation_id="123",
        device_id="dev1",
        language="en",
        agent_id="test_agent",
        satellite_id=None,
    )
    
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="You are helpful."),
        MagicMock(spec=conversation.UserContent, content="turn on kitchen light"),
    ]
    
    # Set up mock LLM API with a tool
    mock_tool = MagicMock()
    mock_tool.name = "HassTurnOn"
    mock_tool.description = "Turn on a device"
    mock_tool.parameters = MagicMock()
    mock_tool.parameters.schema = {"type": "object", "properties": {}}
    mock_api = MagicMock()
    mock_api.tools = [mock_tool]
    mock_chat_log.llm_api = mock_api
    
    # First call: return tool call, then set unresponded_tool_results=True
    # Second call: return final response
    call_count = [0]
    
    async def mock_chat_with_tool_loop(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First iteration: tool call
            mock_chat_log.unresponded_tool_results = True
            yield {"message": {"tool_calls": [{"function": {"name": "HassTurnOn", "arguments": {}}}]}}
        else:
            # Second iteration: final response
            mock_chat_log.unresponded_tool_results = False
            yield {"message": {"content": "Light turned on."}}
    
    mock_ollama_client.chat.side_effect = mock_chat_with_tool_loop
    
    # Track filler generate calls
    filler_calls = []
    original_generate = mock_ollama_client.generate
    
    async def track_generate(*args, **kwargs):
        filler_calls.append(kwargs.get("prompt", args[0] if args else ""))
        return {"response": "Checking lights..."}
    
    mock_ollama_client.generate.side_effect = track_generate
    
    # Mock default agent to NOT match any intent (force LLM path)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default
        
        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.speech = {"plain": {"speech": "Light turned on."}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(user_input, mock_chat_log)
    
    # Verify filler was called exactly once, not twice
    assert len(filler_calls) == 1, f"Filler should be called once, but was called {len(filler_calls)} times"
    # Verify chat was called twice (once for tool call, once for final response)
    assert call_count[0] == 2, f"Chat should be called twice for tool loop, but was called {call_count[0]} times"

async def test_disable_fuzzy_matching(hass: HomeAssistant, mock_ollama_client, mock_chat_log) -> None:
    """Test disabling fuzzy matching toggles the agent property."""
    from custom_components.hybrid_llm.const import CONF_DISABLE_FUZZY_MATCHING
    from custom_components.hybrid_llm.conversation import HybridConversationAgent
    
    agent = HybridConversationAgent(hass, MockConfigEntry(options={CONF_DISABLE_FUZZY_MATCHING: True}))
    
    # Mock default agent
    with patch(
        "homeassistant.components.conversation.async_get_agent"
    ) as mock_get_default:
        mock_default = AsyncMock()
        mock_default.fuzzy_matching = True # Initial state
        
        # Verify fuzzy_matching is False WHEN async_recognize_intent is called
        # Verify fuzzy_matching is False WHEN async_recognize_intent is called
        async def verify_fuzzy_during_call(*args, **kwargs):
            assert mock_default.fuzzy_matching is False
            return MagicMock() # Simulate MATCH
            
        mock_default.async_recognize_intent.side_effect = verify_fuzzy_during_call
        
        # Mock internal execution to succeed
        mock_exec_result = MagicMock()
        mock_exec_result.response.response_type = intent.IntentResponseType.ACTION_DONE
        mock_default.internal_async_process.return_value = mock_exec_result

        mock_get_default.return_value = mock_default
        
        user_input = conversation.ConversationInput(
            text="Turn on light",
            context=MagicMock(),
            conversation_id="test-convo",
            device_id=None,
            language="en",
            agent_id="homeassistant",
            satellite_id=None,
        )
        
        # Call handle_message (expected to hit native fast path)
        await agent._async_handle_message(user_input, mock_chat_log)
        
        # Verify call happened
        mock_default.async_recognize_intent.assert_called_once()
        # Verify it was restored
        assert mock_default.fuzzy_matching is True
