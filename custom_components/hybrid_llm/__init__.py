
import logging
import ollama
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, Event, Context
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_START
from homeassistant.helpers import intent
from homeassistant.util import ulid
from .const import (
    DOMAIN,
    CONF_URL,
    CONF_MODEL,
    CONF_PROMPT,
    DEFAULT_URL,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    CONF_ENABLE_TRACER,
    DEFAULT_ENABLE_TRACER,
    CONF_LLM_HASS_API,
    CONF_NUM_CTX,
    DEFAULT_NUM_CTX,
)
from .helpers import convert_tool_to_ollama

_LOGGER = logging.getLogger(__name__)

async def prewarm_ollama(run_id: str, prompt: str, url: str, model: str):
    """Send a pre-warm request to Ollama."""
    try:
        client = ollama.AsyncClient(host=url, verify=False)
        await client.generate(
            model=model,
            prompt=prompt,
            keep_alive="5m",
            options={"num_predict": 1}
        )
        _LOGGER.debug(f"Pre-warm success for {run_id}")
    except Exception as e:
        _LOGGER.warning(f"Failed to reach Ollama: {e}")

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Hybrid Local Voice Backend from a config entry."""
    hass.data.setdefault(DOMAIN, {"cache": {}})
    
    # Store in hass.data
    hass.data[DOMAIN]["config"] = entry.options

    # Retrieve the configured system prompt
    system_prompt_template = entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
    url = entry.options.get(CONF_URL, DEFAULT_URL)
    main_model = entry.options.get(CONF_MODEL, DEFAULT_MODEL)
    llm_api_id = entry.options.get(CONF_LLM_HASS_API)
    num_ctx = entry.options.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)

    # Check for Tracer Config
    enable_tracer = entry.options.get(CONF_ENABLE_TRACER, DEFAULT_ENABLE_TRACER)
    tracer = None
    
    if enable_tracer:
        # Create and store tracer
        from .tracer import PerformanceTracer
        tracer = PerformanceTracer(hass.config.path("traces"))
        hass.data[DOMAIN]["tracer"] = tracer
   
    # --- Pre-warmer Logic ---
    async def async_trigger_prewarm(device_id: str | None = None):
        """Pre-warm the model with a cached system prompt."""
        if not device_id:
            _LOGGER.debug("Skipping Pre-warm: No device_id provided for context targeting.")
            return

        _LOGGER.debug(f"Triggering Pre-warm for device: {device_id}")
        
        # 1. Generate Context/Prompt (State Freeze)
        from homeassistant.helpers import template, llm
        
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=None,
            language="en", # Default to en for pre-warm
            assistant=None, 
            device_id=device_id
        )
        
        # 1a. Get Tools (if configured)
        tools = None
        if llm_api_id:
            try:
                # Use llm helper to get the API implementation
                api = await llm.async_get_api(hass, llm_api_id, llm_context)
                if api and api.tools:
                    tools = [convert_tool_to_ollama(tool) for tool in api.tools]
                    _LOGGER.debug(f"Pre-warm included {len(tools)} tools from API '{llm_api_id}'")
            except Exception as e:
                _LOGGER.warning(f"Failed to load tools for pre-warm: {e}")

        try:
            prompt_template = template.Template(system_prompt_template, hass)
            system_prompt = prompt_template.async_render(
                parse_result=False,
                variables={"llm_context": llm_context}
            )
        except Exception as e:
            _LOGGER.warning(f"Failed to render prompt template during pre-warm: {e}")
            system_prompt = "You are a helpful assistant."

        # 2. Store in Cache (Fresh State)
        import time
        hass.data[DOMAIN]["fresh_state"] = {
            "timestamp": time.time(),
            "prompt": system_prompt,
            "device_id": device_id
        }
        
        # 3. Fire-and-forget Pre-warm request
        # CRITICAL: Use chat() with system message to match KV Cache key
        # If we used generate(), the prefix wouldn't match the chat template.
        async def send_prewarm():
            run_id = ulid.ulid()
            # Update fresh_state with the run_id
            hass.data[DOMAIN]["fresh_state"]["run_id"] = run_id
            
            if tracer:
                tracer.start_trace(run_id)
                tracer.trace_event(run_id, "LLM Pre-warm", "B", "hybrid_llm", args={"device_id": device_id})

            try:
                client = ollama.AsyncClient(host=url, verify=False)
                # Send system prompt as a chat message with 0 prediction tokens
                # This forces Ollama to process and cache the system prompt.
                response = await client.chat(
                    model=main_model,
                    messages=[{"role": "system", "content": system_prompt}],
                    tools=tools,
                    options={"num_predict": 0, "num_ctx": num_ctx}, 
                    keep_alive="5m"
                )
                
                # Log detailed stats
                total_duration = response.get("total_duration", 0) / 10**9
                load_duration = response.get("load_duration", 0) / 10**9
                prompt_eval_duration = response.get("prompt_eval_duration", 0) / 10**9
                prompt_eval_count = response.get("prompt_eval_count", 0)
                
                _LOGGER.debug(
                    f"Pre-warm complete. Total: {total_duration:.3f}s, "
                    f"Load: {load_duration:.3f}s, "
                    f"Prompt Eval: {prompt_eval_duration:.3f}s ({prompt_eval_count} tokens)"
                )
            except Exception as e:
                _LOGGER.warning(f"Pre-warm HTTP request failed: {e}")
            
            if tracer:
                 tracer.trace_event(run_id, "LLM Pre-warm", "E", "hybrid_llm")
                 await tracer.dump(hass, run_id, clear_buffer=False)

        hass.async_create_task(send_prewarm())

    # Register Service
    async def handle_trigger_prewarm(call):
        """Handle hybrid_llm.trigger_prewarm service call."""
        device_id = call.data.get("device_id")
        await async_trigger_prewarm(device_id)

    hass.services.async_register(DOMAIN, "trigger_prewarm", handle_trigger_prewarm)
    
    # Register Satellite Monitors
    from homeassistant.helpers.event import async_track_state_change_event
    from homeassistant.const import STATE_ON
    from homeassistant.helpers import device_registry as dr, entity_registry as er
    

    
    # Dynamic Discovery: Listen to all state changes for assist_satellite domain
    # This covers entities that exist now AND ones that come later (discovery/init order safe)
    async def handle_satellite_state_change(event: Event):
        # Filter for assist_satellite domain
        entity_id = event.data.get("entity_id")
        if not entity_id or not entity_id.startswith("assist_satellite."):
            return

        new_state = event.data.get("new_state")
        if not new_state:
            return

        # Assist Satellite states: 'listening'
        # Wyoming/Binary Sensor: 'on'
        active_states = {STATE_ON, "listening"}
        
        if new_state.state in active_states:
            _LOGGER.debug(f"Satellite {entity_id} woke up (state: {new_state.state})")
            
            # Resolve Device ID
            device_id = None
            ent_reg = er.async_get(hass)
            entity_entry = ent_reg.async_get(entity_id)
            if entity_entry and entity_entry.device_id:
                    device_id = entity_entry.device_id
            
            await async_trigger_prewarm(device_id)

    # Listen to ALL state changes on the bus, but filter inside
    from homeassistant.const import EVENT_STATE_CHANGED
    entry.async_on_unload(hass.bus.async_listen(EVENT_STATE_CHANGED, handle_satellite_state_change))

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    return True

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, [Platform.CONVERSATION])

