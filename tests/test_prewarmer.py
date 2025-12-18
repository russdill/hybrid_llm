
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_URL, CONF_MODEL
from custom_components.hybrid_llm import DOMAIN

from pytest_homeassistant_custom_component.common import MockConfigEntry

@pytest.fixture
def mock_ollama_client():
    with patch("custom_components.hybrid_llm.ollama.AsyncClient") as mock_client:
        client_instance = AsyncMock()
        client_instance.chat.return_value = {
            "total_duration": 1000000000,
            "load_duration": 100000000,
            "prompt_eval_duration": 100000000,
            "prompt_eval_count": 10
        }
        client_instance.generate.return_value = {
            "response": "Test Response",
            "total_duration": 1000000000,
            "load_duration": 100000000
        }
        mock_client.return_value = client_instance
        yield client_instance

async def test_trigger_prewarm_service(hass: HomeAssistant, mock_ollama_client):
    """Test that the trigger_prewarm service freezes state and calls Ollama."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={
            CONF_URL: "http://test-url",
            CONF_MODEL: "test-model",
        }
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Call the service
    await hass.services.async_call(DOMAIN, "trigger_prewarm", service_data={"device_id": "test_device"}, blocking=True)
    
    # Verify State Frozen
    assert "fresh_state" in hass.data[DOMAIN]
    fresh_state = hass.data[DOMAIN]["fresh_state"]
    assert "prompt" in fresh_state
    assert "timestamp" in fresh_state
    
    # Verify Ollama Call (Chat API with 0 tokens)
    mock_ollama_client.chat.assert_called_once()
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["options"]["num_predict"] == 0
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"] == fresh_state["prompt"]

async def test_satellite_event_trigger(hass: HomeAssistant, mock_ollama_client):
    """Test that a satellite turning ON triggers the pre-warmer."""
    # Use assist_satellite domain
    satellite_entity = "assist_satellite.kitchen"
    
    # Create the entity state so async_entity_ids finds it
    hass.states.async_set(satellite_entity, "idle")
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={
            CONF_URL: "http://test-url",
            CONF_MODEL: "test-model",
        }
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    # Mock Resolver
    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
        mock_registry = MagicMock()
        mock_entry = MagicMock()
        mock_entry.device_id = "satellite_device_id"
        mock_registry.async_get.return_value = mock_entry
        mock_er_get.return_value = mock_registry
        
        # Turn on Satellite (Transition to listening)
        hass.states.async_set(satellite_entity, "listening")
        await hass.async_block_till_done()
    
    # Verify Pre-warm triggered
    assert "fresh_state" in hass.data[DOMAIN]
    assert hass.data[DOMAIN]["fresh_state"]["device_id"] == "satellite_device_id"
    mock_ollama_client.chat.assert_called_once()

async def test_assist_satellite_trigger(hass: HomeAssistant, mock_ollama_client):
    """Test that an assist_satellite transitioning to listening triggers pre-warmer."""
    satellite_entity = "assist_satellite.living_room"
    
    # Create the entity state
    hass.states.async_set(satellite_entity, "idle")
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={
            CONF_URL: "http://test-url",
            CONF_MODEL: "test-model",
        }
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    # Mock Resolver
    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
        mock_registry = MagicMock()
        mock_entry = MagicMock()
        mock_entry.device_id = "satellite_device_id"
        mock_registry.async_get.return_value = mock_entry
        mock_er_get.return_value = mock_registry
        
        # Transition to Listening
        hass.states.async_set(satellite_entity, "listening")
        await hass.async_block_till_done()
    
    # Verify Pre-warm triggered
    assert "fresh_state" in hass.data[DOMAIN]
    assert hass.data[DOMAIN]["fresh_state"]["device_id"] == "satellite_device_id"
    mock_ollama_client.chat.assert_called_once()
