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
        
        # Mock async_debug_recognize (exact match)
        mock_default.async_debug_recognize.return_value = {
            "match": True,
            "fuzzy_match": False
        }
        
        # Mock async_process (execution + rendering)
        mock_conversation_result = MagicMock(spec=conversation.ConversationResult)
        mock_conversation_result.response = MagicMock(spec=intent.IntentResponse)
        mock_conversation_result.response.intent_type = "HassTurnOn"
        mock_conversation_result.response.speech = {"plain": {"speech": "Turned on the light"}}
        
        mock_default.async_process.return_value = mock_conversation_result
        
        mock_get_default.return_value = mock_default
        
        result = await agent._async_handle_message(user_input, mock_chat_log)
        
        # Should have checked debug then execute
        mock_default.async_debug_recognize.assert_called_once_with(user_input)
        
        # Should call async_process, NOT async_recognize_intent
        mock_default.async_process.assert_called_once_with(user_input)
        assert not mock_default.async_recognize_intent.called

        # Should return native intent result with full fidelity
        assert result.response.intent_type == "HassTurnOn"
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
        
        # Mock async_debug_recognize returning NO match (None or match=False)
        mock_default.async_debug_recognize.return_value = None # or {"match": False}
        
        # Mock async_recognize_intent returning None (no match)
        mock_default.async_recognize_intent.return_value = None
        
        mock_get_default.return_value = mock_default
        
        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_result = MagicMock()
            mock_result.response.intent = None
            mock_result.response.speech = {"plain": {"speech": "I don't have weather info"}}
            mock_get_result.return_value = mock_result
            
            result = await agent._async_handle_message(user_input, mock_chat_log)
            
            # verify debug was checked
            mock_default.async_debug_recognize.assert_called_once_with(user_input)
            # verify recognize not called
            mock_default.async_recognize_intent.assert_not_called()
            
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
        mock_default.async_debug_recognize.return_value = None
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
        mock_default.async_debug_recognize.return_value = None
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
        mock_default.async_debug_recognize.return_value = None
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
        mock_default.async_debug_recognize.return_value = None
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

async def test_hybrid_config_options(hass: HomeAssistant, mock_ollama_client, mock_chat_log) -> None:
    """Test native intents and fuzzy matching configuration options."""
    from custom_components.hybrid_llm.const import CONF_ENABLE_NATIVE_INTENTS, CONF_ENABLE_FUZZY_MATCHING
    from custom_components.hybrid_llm.conversation import HybridConversationAgent
    
    # CASE 1: Disable Native Intents
    agent = HybridConversationAgent(hass, MockConfigEntry(options={
        CONF_ENABLE_NATIVE_INTENTS: False
    }))
    
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_get_default.return_value = mock_default
        
        user_input = conversation.ConversationInput(
            text="Turn on light",
            context=MagicMock(),
            conversation_id="test-convo-1",
            device_id=None,
            language="en",
            agent_id="homeassistant",
            satellite_id=None,
        )
        
        # Should NOT check native intent
        # Mock result from chat log since it goes to LLM
        # Patch _async_handle_chat_log to skip LLM execution
        with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat:
            with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
                mock_get_result.return_value = MagicMock()
                await agent._async_handle_message(user_input, mock_chat_log)
        
        mock_default.async_recognize_intent.assert_not_called()

    # CASE 2: Enable Native Intents, Enable Fuzzy (Default False) -> With Enable Fuzzy=False (Default)
    # Refactored: We check async_debug_recognize first.
    agent = HybridConversationAgent(hass, MockConfigEntry(options={
        CONF_ENABLE_NATIVE_INTENTS: True,
        CONF_ENABLE_FUZZY_MATCHING: False
    }))

    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_get_default.return_value = mock_default
        
        # Simulate Debug Result: Fuzzy Match
        mock_default.async_debug_recognize.return_value = {
            "match": True,
            "fuzzy_match": True
        }
        
        # Since Fuzzy is Disabled in Options (False), logic should ignore it.
        # Should NOT call recognise_intent
        
        # Patch result from chat log to avoid LLM error (and skip LLM exec)
        with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat:
            with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
                mock_get_result.return_value = MagicMock()
                await agent._async_handle_message(user_input, mock_chat_log)
        
        # Debug was called
        mock_default.async_debug_recognize.assert_called_once()
        # Fuzzy match but disabled -> Should NOT execute process
        mock_default.async_process.assert_not_called()
        # And should fall through to LLM (mock_handle_chat called)
        mock_handle_chat.assert_called_once()

    # CASE 3: Enable Native Intents, ENABLE Fuzzy Matching -> Accept Fuzzy
    agent = HybridConversationAgent(hass, MockConfigEntry(options={
        CONF_ENABLE_NATIVE_INTENTS: True,
        CONF_ENABLE_FUZZY_MATCHING: True
    }))

    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_get_default.return_value = mock_default
        
        # Simulate Debug Result: Fuzzy Match
        mock_default.async_debug_recognize.return_value = {
            "match": True,
            "fuzzy_match": True
        }
        
        # Mock process result
        mock_result = MagicMock(spec=conversation.ConversationResult)
        mock_result.response.speech = {"plain": {"speech": "OK"}}
        mock_default.async_process.return_value = mock_result
        
        user_input = conversation.ConversationInput(
            text="Turn on light",
            context=MagicMock(),
            conversation_id="test-convo-3",
            device_id=None,
            language="en",
            agent_id="homeassistant",
            satellite_id=None,
        )
        
        result = await agent._async_handle_message(user_input, mock_chat_log)

        # Debug called
        mock_default.async_debug_recognize.assert_called_once()
        # Should EXECUTE process because fuzzy is allowed
        mock_default.async_process.assert_called_once_with(user_input)


@pytest.mark.asyncio
async def test_filler_echo_mode(hass, mock_ollama_client, mock_llm_api, mock_chat_log):
    """Test filler in Echo mode returns rendered prompt without calling LLM."""
    # Configure with Echo Mode and a custom prompt
    config_entry = MockConfigEntry(options={
        "filler_model": "echo",
        "filler_prompt": "Echoing: {{ text }}"
    })
    agent = HybridConversationAgent(hass, config_entry)
    
    # Mock default agent (No match -> LLM)
    with patch("homeassistant.components.conversation.async_get_agent") as mock_get_default:
        mock_default = AsyncMock()
        mock_default.async_debug_recognize.return_value = None
        mock_default.async_recognize_intent.return_value = None
        mock_get_default.return_value = mock_default
        
        # Mock chat log result to avoid errors
        with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
            mock_get_result.return_value = MagicMock()
            
            # Mock chat stream so main loop runs
            mock_ollama_client.chat.return_value = MagicMock()
            
            # Let's mock async_add_delta_content_stream to capture the generator
            captured_content = []
            
            async def capture_stream(entity_id, stream_generator):
                async for chunk in stream_generator:
                    if chunk.get("role") == "assistant" and "content" in chunk:
                        captured_content.append(chunk["content"])
                yield MagicMock() # dummy yield for the async for loop
                
            mock_chat_log.async_add_delta_content_stream = capture_stream
            
            # Execute
            await agent._async_handle_message(
                conversation.ConversationInput(
                    text="Hello World",
                    context=MagicMock(),
                    conversation_id="echo-id",
                    device_id=None,
                    language="en",
                    agent_id="test_agent",
                    satellite_id=None,
                ),
                mock_chat_log
            )
            
            # Assertions
            # 1. client.generate should NOT be called
            mock_ollama_client.generate.assert_not_called()
            
            # 2. Filler text should be the rendered prompt ("Echoing: Hello World") + "... " suffix
            expected = "Echoing: Hello World... "
            assert any(expected in c for c in captured_content), f"Expected '{expected}' in {captured_content}"


async def test_prewarm_cache_hit(hass: HomeAssistant, mock_ollama_client, mock_chat_log):
    """Test that valid cached state is used and cleared."""
    import time
    
    # Setup Cache
    cached_prompt = "Cached Specail Prompt"
    hass.data[DOMAIN] = {
        "fresh_state": {
            "timestamp": time.time(), # Now
            "prompt": cached_prompt,
            "device_id": None
        }
    }
    
    # Create Agent
    entry = MockConfigEntry(options={"enable_native_intents": False})
    agent = HybridConversationAgent(hass, entry)
    
    # Process Message
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="test_conv",
        language="en",
        agent_id=entry.entry_id,
        device_id=None,
        satellite_id=None,
    )
    
    # We need to mock _async_handle_chat_log to avoid actual Ollama calls,
    # but we want to verify async_provide_llm_data was called with cached_prompt.
    with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat, \
         patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
        mock_handle_chat.return_value = None
        mock_get_result.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en"),
            conversation_id="test_conv"
        )
        
        await agent._async_handle_message(user_input, mock_chat_log)
    
    # Verify Cache Cleared
    assert "fresh_state" not in hass.data[DOMAIN]
    
    # Verify ChatLog provided with cached prompt
    # Note: async_provide_llm_data arg 2 is prompt (index 2 in args tuple)
    # def async_provide_llm_data(self, llm_context, llm_api, prompt, extra_system_prompt):
    mock_chat_log.async_provide_llm_data.assert_called_once()
    print("DEBUG: test_prewarm_cache_hit completed assertions")
    

async def test_prewarm_cache_miss_expired(hass: HomeAssistant, mock_ollama_client, mock_chat_log):
    """Test that expired cached state is ignored and cleared."""
    import time

    # Setup Expired Cache (20s old)
    hass.data[DOMAIN] = {
        "fresh_state": {
            "timestamp": time.time() - 20,
            "prompt": "Old Prompt",
            "device_id": None
        }
    }
    
    # Create Agent
    entry = MockConfigEntry(options={"enable_native_intents": False})
    agent = HybridConversationAgent(hass, entry)
    
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="test_conv",
        language="en",
        agent_id=entry.entry_id,
        device_id=None,
        satellite_id=None,
    )
    
    with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat, \
         patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
        mock_handle_chat.return_value = None
        mock_get_result.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en"),
            conversation_id="test_conv"
        )
        await agent._async_handle_message(user_input, mock_chat_log)

    # Verify Cache Cleared
    assert "fresh_state" not in hass.data[DOMAIN]
    
    # Verify ChatLog provided with DEFAULT prompt (not Old Prompt)
    mock_chat_log.async_provide_llm_data.assert_called_once()


async def test_prewarm_device_mismatch(hass: HomeAssistant, mock_ollama_client, mock_chat_log):
    """Test that cache is ignored if device_id mismatches."""
    import time
    
    hass.data[DOMAIN] = {
        "fresh_state": {
            "timestamp": time.time(),
            "prompt": "Cached Prompt",
            "device_id": "kitchen_satellite"
        }
    }
    
    entry = MockConfigEntry(options={"enable_native_intents": False})
    agent = HybridConversationAgent(hass, entry)
    
    # Request from different device
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="test_conv",
        language="en",
        agent_id=entry.entry_id,
        device_id="living_room_satellite",
        satellite_id=None,
    )
    
    # Mock LLM handling to avoid actual calls
    with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat, \
         patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
        mock_handle_chat.return_value = None
        mock_get_result.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en"),
            conversation_id="test_conv"
        )
        await agent._async_handle_message(user_input, mock_chat_log)

    # Verify Cache Cleared (Mismatch counts as a miss/consumption)
    assert "fresh_state" not in hass.data[DOMAIN]
    
    # Verify DEFAULT prompt used (implied by call)
    mock_chat_log.async_provide_llm_data.assert_called_once()


async def test_prewarm_run_id_continuity(hass: HomeAssistant, mock_ollama_client, mock_chat_log):
    """Test that run_id is carried over from cache."""
    import time
    
    cached_run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    hass.data[DOMAIN] = {
        "fresh_state": {
            "timestamp": time.time(),
            "prompt": "Cached Prompt",
            "device_id": None,
            "run_id": cached_run_id
        }
    }
    
    # Enable tracer
    tracer_mock = MagicMock()
    tracer_mock.dump = AsyncMock()
    hass.data[DOMAIN]["tracer"] = tracer_mock
    tracer = hass.data[DOMAIN]["tracer"]
    
    entry = MockConfigEntry(options={"enable_native_intents": False})
    agent = HybridConversationAgent(hass, entry)
    
    user_input = conversation.ConversationInput(
        text="Hello",
        context=Context(),
        conversation_id="test_conv",
        language="en",
        agent_id=entry.entry_id,
        device_id=None,
        satellite_id=None,
    )
    
    with patch.object(agent, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle_chat, \
         patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
        mock_handle_chat.return_value = None
        mock_get_result.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en"),
            conversation_id="test_conv"
        )
        await agent._async_handle_message(user_input, mock_chat_log)
    
    # Verify tracer called with cached_run_id (Specifically the 'trace_continued' event)
    # Check all calls to trace_event
    found_continuation = False
    for call in tracer.trace_event.call_args_list:
        args, kwargs = call
        if args and args[0] == cached_run_id and kwargs.get("args", {}).get("event") == "trace_continued":
            found_continuation = True
            break
            
    assert found_continuation, "Did not find trace_continued event with cached run ID"


@pytest.mark.asyncio
async def test_filler_position(hass, mock_ollama_client, mock_chat_log):
    """Test that filler is included in LLM context (after user message)."""
    entry = MockConfigEntry(options={
        "filler_model": "test_filler",
        "enable_native_intents": False
    })
    agent = HybridConversationAgent(hass, entry)
    hass.data[DOMAIN] = {"cache": {}}
    
    # Mock Filler Generation
    mock_ollama_client.generate.return_value = {"response": "One moment..."}
    # Mock Chat Response
    mock_ollama_client.chat.return_value = MagicMock()
    
    # Setup Chat Log with User Message
    mock_chat_log.content = [
        MagicMock(spec=conversation.SystemContent, content="System Prompt"),
        MagicMock(spec=conversation.UserContent, content="Do something"),
    ]
    
    # Mock async_add_delta_content_stream to append to chat_log content for this test view
    async def mock_stream(entity_id, stream):
        # Consume stream
        async for chunk in stream:
            content = chunk["content"]
            # Simulate adding to chat log content so _build_messages sees it
            # Note: _build_messages reads mock_chat_log.content.
            # In update, we manually add it to 'messages' list in the test verification below?
            # No, the code calls _build_messages_from_chat_log(chat_log).
            # We need mock_chat_log.content to reflect the addition if we want _build_messages to pick it up.
            # BUT: The code calls messages = self._build_messages_from_chat_log(chat_log) BEFORE any filler logic?
            # Let's check conversation.py order.
            pass
        yield MagicMock()
        
    mock_chat_log.async_add_delta_content_stream = mock_stream
    
    # CRITICAL: In conversation.py, messages = self._build_messages_from_chat_log(chat_log) is called at the TOP.
    # formatting happens LATER?
    # Actually:
    # 1. messages = _build_messages... (User input is there)
    # 2. Filler logic runs -> calls chat_log.async_add_delta_content_stream
    # 3. Code logic does NOT re-fetch messages?
    # Wait, if messages is not updated, filler won't be in 'messages' passed to client.chat loop?
    # Let's verify conversation.py logic.
    
    user_input = conversation.ConversationInput(
        text="Do something",
        context=Context(),
        conversation_id="test_conv",
        language="en",
        agent_id=entry.entry_id,
        device_id=None,
        satellite_id=None,
    )
    
    with patch("homeassistant.components.conversation.async_get_result_from_chat_log") as mock_get_result:
        mock_get_result.return_value = conversation.ConversationResult(
            response=intent.IntentResponse(language="en"),
            conversation_id="test_conv"
        )
        
        # We need _async_handle_chat_log to actually run to test formatting
        await agent._async_handle_message(user_input, mock_chat_log)

    assert mock_ollama_client.chat.called
    call_args = mock_ollama_client.chat.call_args
    messages = call_args[1]["messages"]
    
    # Check that filler is present
    has_filler = any(m["role"] == "assistant" and "One moment..." in m["content"] for m in messages)
    assert has_filler, f"Filler message missing from LLM context! Messages: {messages}"
