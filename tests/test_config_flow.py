"""Test the Hybrid LLM config flow."""
from unittest.mock import patch, AsyncMock
import asyncio
import pytest
from homeassistant import config_entries, data_entry_flow
from homeassistant.core import HomeAssistant

# Assuming pytest-homeassistant-custom-component is available
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.hybrid_llm.const import (
    DOMAIN,
    CONF_URL,
    CONF_MODEL,
    CONF_FILLER_MODEL,
)

async def test_form(hass: HomeAssistant) -> None:
    """Test we get the form."""
    # Mock 'conversation' component as loaded to avoid dependency errors
    hass.config.components.add("conversation")
    
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["errors"] is None

    with patch(
        "custom_components.hybrid_llm.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "title": "My Hybrid LLM",
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY
    assert result2["title"] == "My Hybrid LLM"
    assert result2["data"] == {}
    assert len(mock_setup_entry.mock_calls) == 1

async def test_options_flow(hass: HomeAssistant) -> None:
    """Test options flow happy path (models installed)."""
    from custom_components.hybrid_llm.const import CONF_ENABLE_NATIVE_INTENTS, CONF_ENABLE_FUZZY_MATCHING, CONF_ENABLE_TRACER
    hass.config.components.add("conversation")
    
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={}
    )
    config_entry.add_to_hass(hass)

    # 1. Init Options
    with patch("custom_components.hybrid_llm.config_flow._get_installed_models", return_value=["test-model", "test-filler"]):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)

    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["step_id"] == "init"

    # 2. Submit Form (Models exist)
    with patch("custom_components.hybrid_llm.async_reload_entry") as mock_reload, \
         patch("custom_components.hybrid_llm.config_flow._get_installed_models", return_value=["test-model", "test-filler"]):
         result2 = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_URL: "http://test-url",
                CONF_MODEL: "test-model",
                CONF_FILLER_MODEL: "test-filler",
                CONF_ENABLE_NATIVE_INTENTS: True,
                CONF_ENABLE_FUZZY_MATCHING: False,
                CONF_ENABLE_TRACER: False,
            }
        )
        
    assert result2["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY
    assert result2["data"][CONF_URL] == "http://test-url"
    assert result2["data"][CONF_MODEL] == "test-model"


async def test_options_flow_download(hass: HomeAssistant) -> None:
    """Test options flow with download trigger."""
    hass.config.components.add("conversation")
    
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={},
        options={}
    )
    config_entry.add_to_hass(hass)

    # 1. Init
    with patch("custom_components.hybrid_llm.config_flow._get_installed_models", return_value=[]):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)

    # 2. Submit Form (Models MISSING) -> triggers Download
    download_event = asyncio.Event()
    
    async def mock_pull_with_wait(*args, **kwargs):
        await download_event.wait()
        
    with patch("custom_components.hybrid_llm.config_flow._get_installed_models", return_value=[]), \
         patch("custom_components.hybrid_llm.config_flow._pull_model", side_effect=mock_pull_with_wait) as mock_pull:
         
         result2 = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_URL: "http://test-url",
                CONF_MODEL: "new-model",
                CONF_FILLER_MODEL: "new-filler"
            }
        )
    
         # Should transition to download task
         assert result2["type"] == data_entry_flow.FlowResultType.SHOW_PROGRESS
         assert result2["step_id"] == "download"
         assert result2["progress_action"] == "download"
         
         # 3. Simulate Download Completion
         download_event.set() # Release the task
         await hass.async_block_till_done() 
        
         # Call again to see result (Poll)
         result3 = await hass.config_entries.options.async_configure(
             result["flow_id"]
         )
         
         # Note: Depending on HA version/test harness, it might auto-advance to finish
         # if the progress is done. In our case, it seems to go straight to create_entry.
         assert result3["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY
         assert result3["data"][CONF_MODEL] == "new-model"
