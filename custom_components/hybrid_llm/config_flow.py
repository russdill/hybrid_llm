"""Config flow for Hybrid Local Voice Backend."""
from typing import Any
import voluptuous as vol
import logging

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.const import CONF_URL
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    BooleanSelector,
    TemplateSelector,
)


from .const import (
    DOMAIN,
    CONF_KEEP_ALIVE,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_NUM_CTX,
    CONF_MAX_HISTORY,
    CONF_LLM_HASS_API,
    CONF_THINK,
    CONF_FILLER_MODEL,
    CONF_FILLER_PROMPT,
    CONF_WAIT_FOR_FILLER,
    CONF_ENABLE_NATIVE_INTENTS,
    CONF_ENABLE_FUZZY_MATCHING,
    DEFAULT_URL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_NUM_CTX,
    DEFAULT_MAX_HISTORY,
    DEFAULT_FILLER_MODEL,
    DEFAULT_FILLER_PROMPT,
    DEFAULT_WAIT_FOR_FILLER,
    DEFAULT_ENABLE_NATIVE_INTENTS,
    DEFAULT_ENABLE_FUZZY_MATCHING,
    CONF_ENABLE_TRACER,
    DEFAULT_ENABLE_TRACER,
    CONF_ENABLE_PREWARM,
    DEFAULT_ENABLE_PREWARM,
    FILLER_MODEL_ECHO,
    COMMON_MODELS,
)

_LOGGER = logging.getLogger(__name__)

import ollama

async def _get_installed_models(url: str) -> list[str]:
    """Fetch installed models from Ollama."""
    try:
        client = ollama.AsyncClient(host=url, verify=False)
        response = await client.list()
        # response is dict with 'models' list
        # each model object has 'model' key (not 'name')
        return [m["model"] for m in response.get("models", [])]
    except Exception as e:
        _LOGGER.warning(f"Failed to fetch installed models: {e}")
    
    return []

def _get_model_choices(installed: list[str]) -> list[dict]:
    """Build the list of model choices for the selector."""
    options = []
    
    # Add installed models first
    for model in installed:
        options.append({"label": f"{model} (installed)", "value": model})
    
    # Add common models if not already in list
    for model in COMMON_MODELS:
        if model not in installed:
            options.append({"label": model, "value": model})
            
    return options

async def _pull_model(hass, url: str, model: str):
    """Pull a model from Ollama."""
    try:
        client = ollama.AsyncClient(host=url, verify=False)
        # Pull returns an async generator if stream=True
        async for progress in await client.pull(model=model, stream=True):
            pass # Keep task alive
    except Exception as e:
         _LOGGER.error(f"Failed to pull model: {e}")
         raise


class HybridLLMConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Hybrid LLM."""

    VERSION = 1

    def __init__(self):
        """Initialize."""
        self.config_data = {}
        self.download_task = None

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({
                    vol.Required("title", default="Hybrid LLM"): str,
                })
            )

        return self.async_create_entry(title=user_input["title"], data={})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return HybridLLMOptionsFlow(config_entry)


class HybridLLMOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.entry = config_entry
        self.options_buffer = {} # Store user input while downloading
        self.download_task = None
        self.missing_models = []
        self.model_to_download = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        """Manage the options."""
        errors = {}
        options = self.entry.options
        current_url = user_input.get(CONF_URL, options.get(CONF_URL, DEFAULT_URL)) if user_input else options.get(CONF_URL, DEFAULT_URL)

        if user_input is not None:
             # Check if we need to download models
             installed = await _get_installed_models(current_url)
             
             missing_models = []
             main_model = user_input.get(CONF_MODEL)
             filler_model = user_input.get(CONF_FILLER_MODEL)
             
             if main_model and main_model not in installed:
                 missing_models.append(main_model)
                 
             if filler_model and filler_model not in installed:
                 missing_models.append(filler_model)
             
             # If multiple missing, we pick one to download (simplified flow)
             if missing_models:
                 self.missing_models = missing_models
                 self.model_to_download = self.missing_models.pop(0)
                 self.options_buffer = user_input
                 return await self.async_step_download()

             return self.async_create_entry(title="", data=user_input)

        # Retrieve installed models for populating choices
        installed = await _get_installed_models(current_url)
        model_choices = _get_model_choices(installed)
        
        # Filler model choices: Echo + installed models
        filler_model_choices = [{"label": "Echo (Use prompt as filler)", "value": FILLER_MODEL_ECHO}] + model_choices

        # Get LLM APIs
        from homeassistant.helpers import llm
        llm_apis = llm.async_get_apis(self.hass)
        api_options = [{"label": api.name, "value": api.id} for api in llm_apis]

        return self.async_show_form(
            step_id="init",
            errors=errors,
            data_schema=vol.Schema({
                # Connection (Top as requested)
                vol.Required(
                    CONF_URL, 
                    description={"suggested_value": options.get(CONF_URL, DEFAULT_URL)}
                ): str,

                vol.Required(
                    CONF_ENABLE_PREWARM,
                    default=options.get(CONF_ENABLE_PREWARM, DEFAULT_ENABLE_PREWARM)
                ): BooleanSelector(),

                # Native Intents and Fuzzy Matching
                vol.Required(
                     CONF_ENABLE_NATIVE_INTENTS,
                     default=options.get(CONF_ENABLE_NATIVE_INTENTS, DEFAULT_ENABLE_NATIVE_INTENTS)
                ): BooleanSelector(),
                vol.Required(
                     CONF_ENABLE_FUZZY_MATCHING,
                     default=options.get(CONF_ENABLE_FUZZY_MATCHING, DEFAULT_ENABLE_FUZZY_MATCHING)
                ): BooleanSelector(),

                # Filler Model (Before Main Model)
                vol.Required(
                    CONF_FILLER_MODEL,
                    description={"suggested_value": options.get(CONF_FILLER_MODEL, DEFAULT_FILLER_MODEL)}
                ): SelectSelector(SelectSelectorConfig(
                    options=filler_model_choices,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True
                )),
                vol.Optional(
                    CONF_FILLER_PROMPT,
                    description={"suggested_value": options.get(CONF_FILLER_PROMPT, DEFAULT_FILLER_PROMPT)}
                ): TemplateSelector(),
                vol.Required(
                    CONF_WAIT_FOR_FILLER,
                    default=options.get(CONF_WAIT_FOR_FILLER, DEFAULT_WAIT_FOR_FILLER)
                ): BooleanSelector(),

                # Main Model
                vol.Required(
                    CONF_MODEL,
                    description={"suggested_value": options.get(CONF_MODEL, DEFAULT_MODEL)}
                ): SelectSelector(SelectSelectorConfig(
                    options=model_choices,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True
                )),
                vol.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": options.get(CONF_PROMPT, DEFAULT_PROMPT)}
                ): TemplateSelector(),
                vol.Optional(
                    CONF_NUM_CTX,
                    description={"suggested_value": options.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)}
                ): NumberSelector(NumberSelectorConfig(
                    min=1024, max=128000, step=1024, mode=NumberSelectorMode.BOX
                )),
                vol.Optional(
                    CONF_MAX_HISTORY,
                    description={"suggested_value": options.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)}
                ): int,
                vol.Optional(
                    CONF_KEEP_ALIVE,
                    description={"suggested_value": options.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)}
                ): str,
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={"suggested_value": options.get(CONF_LLM_HASS_API, [])}
                ): SelectSelector(SelectSelectorConfig(
                    options=api_options,
                    multiple=True,
                )),
                vol.Optional(
                    CONF_THINK,
                    description={"suggested_value": options.get(CONF_THINK, False)}
                ): BooleanSelector(),
                
                vol.Required(
                    CONF_ENABLE_TRACER,
                    default=options.get(CONF_ENABLE_TRACER, DEFAULT_ENABLE_TRACER)
                ): BooleanSelector(),
            })
        )

    async def async_step_download(self, user_input=None):
        """Wait for cached download."""
        if not self.download_task:
            url = self.options_buffer.get(CONF_URL, DEFAULT_URL)
            self.download_task = self.hass.async_create_task(
                _pull_model(self.hass, url, self.model_to_download)
            )
            
        if not self.download_task.done():
             return self.async_show_progress(
                step_id="download",
                progress_action="download",
                progress_task=self.download_task,
                description_placeholders={"model": self.model_to_download}
            )
        
        # Task done
        try:
            await self.download_task
        except Exception as e:
            _LOGGER.error(f"Download failed: {e}")
            return self.async_abort(reason="download_failed")
            
        # Success -> Check if there is another model missing (Sequential Loop)
        self.download_task = None
        if hasattr(self, "missing_models") and self.missing_models:
             self.model_to_download = self.missing_models.pop(0)
             # Trigger next download immediately
             return await self.async_step_download()
        
        return self.async_show_progress_done(next_step_id="finish")

    async def async_step_finish(self, user_input=None):
        """Finish the flow."""
        return self.async_create_entry(title="", data=self.options_buffer)

