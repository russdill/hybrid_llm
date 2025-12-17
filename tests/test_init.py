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



async def test_tracer_config(hass: HomeAssistant, mock_ollama_client, mock_llm_api) -> None:
    """Test that tracer depends on config."""
    from custom_components.hybrid_llm.const import CONF_ENABLE_TRACER
    
    # CASE 1: Default (False)
    entry1 = MockConfigEntry(domain=DOMAIN, data={}, options={})
    entry1.add_to_hass(hass)
    
    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups", return_value=True):
         await hass.config_entries.async_setup(entry1.entry_id)
         await hass.async_block_till_done()

    assert DOMAIN in hass.data
    assert "tracer" not in hass.data[DOMAIN] # Should be absent
    
    # Teardown
    await hass.config_entries.async_unload(entry1.entry_id)

    # CASE 2: Enabled (True)
    entry2 = MockConfigEntry(domain=DOMAIN, data={}, options={
        CONF_ENABLE_TRACER: True
    })
    entry2.add_to_hass(hass)
    
    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups", return_value=True):
         await hass.config_entries.async_setup(entry2.entry_id)
         await hass.async_block_till_done()

    assert "tracer" in hass.data[DOMAIN] 
    assert hass.data[DOMAIN]["tracer"] is not None
