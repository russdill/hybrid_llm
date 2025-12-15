"""Test the Hybrid LLM init file."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from homeassistant.core import HomeAssistant

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hybrid_llm.const import DOMAIN, CONF_URL, CONF_MODEL

async def test_setup_entry(hass: HomeAssistant, mock_ollama_client, mock_llm_api) -> None:
    """Test setting up the integration entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={
            CONF_URL: "http://test-ollama",
            CONF_MODEL: "test-model"
        }
    )
    entry.add_to_hass(hass)
    
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    # Check if entry was setup successfully (it returns True/False)
    # The MockConfigEntry.add_to_hass handles registration.
    # hass.config_entries.async_setup calls the component setup.
    
    assert DOMAIN in hass.data
    assert hass.data[DOMAIN]["config"][CONF_URL] == "http://test-ollama"

async def test_prewarmer_event_listener(hass: HomeAssistant, mock_ollama_client, mock_llm_api) -> None:
    """Test that the event listener triggers the pre-warmer."""
    # Setup Component
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={
            CONF_URL: "http://mock-ollama",
            CONF_MODEL: "test-model"
        }
    )
    # entry.add_to_hass(hass) # We manually setup

    # Manually call setup to ensure logic runs in test env
    # Note: async_forward_entry_setups fails if state is not loaded.
    # We patch it to avoid that check/failure in unit test isolation.
    
    # Setup normally via HA
    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups", return_value=True):
        from custom_components.hybrid_llm import async_setup_entry
        await async_setup_entry(hass, entry)
        await hass.async_block_till_done()
    
    assert DOMAIN in hass.data

    # Simulate Event
    event_data = {
        "pipeline_execution_id": "test_run_123",
        "device_id": "test_device"
    }

    # We mock prewarm_ollama to verify it's called
    with patch("custom_components.hybrid_llm.prewarm_ollama", new_callable=AsyncMock) as mock_prewarm:
        hass.bus.async_fire("assist_pipeline_pipeline_start", event_data)
        await hass.async_block_till_done()
        
        # Verify Pre-warm called
        assert mock_prewarm.call_count == 1
        args = mock_prewarm.call_args[0]
        assert args[0] == "test_run_123" # run_id
        assert "You are a voice assistant" in args[1] # prompt (default)
        assert args[2] == "http://mock-ollama" # url
        assert args[3] == "test-model" # model
