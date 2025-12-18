
import logging
import ollama
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, Event
from homeassistant.const import Platform, CONF_URL

from homeassistant.util import ulid
from .const import (
    DOMAIN,

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
    CONF_ENABLE_PREWARM,
    DEFAULT_ENABLE_PREWARM,
    CONF_FILLER_MODEL,
    FILLER_MODEL_ECHO,
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
        
        # Generate run_id in outer scope so it's accessible to tracer.dump
        run_id = ulid.ulid()
        hass.data[DOMAIN]["fresh_state"]["run_id"] = run_id

        # 3. Fire-and-forget Pre-warm request
        # CRITICAL: Use chat() with system message to match KV Cache key
        # If we used generate(), the prefix wouldn't match the chat template.
        async def send_prewarm():
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
                
                # Log detailed stats (Safety check for None)
                total_duration = (response.get("total_duration") or 0) / 10**9
                load_duration = (response.get("load_duration") or 0) / 10**9
                prompt_eval_duration = (response.get("prompt_eval_duration") or 0) / 10**9
                prompt_eval_count = response.get("prompt_eval_count") or 0
                
                _LOGGER.debug(f"Pre-warm complete. Total: {total_duration:.3f}s, Load: {load_duration:.3f}s, Prompt Eval: {prompt_eval_duration:.3f}s ({prompt_eval_count} tokens)")

                if tracer:
                    tracer.trace_event(run_id, "LLM Pre-warm", "E", "hybrid_llm")

            except Exception as e:
                _LOGGER.warning(f"Failed to reach Ollama during pre-warm: {e}")
                if tracer:
                    tracer.trace_event(run_id, "LLM Pre-warm", "E", "hybrid_llm", args={"error": str(e)})

        # Pre-warm Filler Model (Concurrent)
        async def send_filler_prewarm():
            filler_model = entry.options.get(CONF_FILLER_MODEL)
            if not filler_model or filler_model == FILLER_MODEL_ECHO:
                return

            try:
                client = ollama.AsyncClient(host=url, verify=False)
                # Use generate() for filler (matches conversation.py)
                response = await client.generate(
                    model=filler_model,
                    prompt="",
                    options={"num_predict": 0},
                    keep_alive="5m"
                )
                
                total_duration = (response.get("total_duration") or 0) / 10**9
                load_duration = (response.get("load_duration") or 0) / 10**9
                _LOGGER.debug(f"Filler pre-warm complete ({filler_model}). Total: {total_duration:.3f}s, Load: {load_duration:.3f}s")
                
            except Exception as e:
                _LOGGER.warning(f"Failed to pre-warm filler model: {e}")

        # Run both pre-warms currently
        import asyncio
        await asyncio.gather(send_filler_prewarm(), send_prewarm())
        
        if tracer:
            await tracer.dump(hass, run_id, clear_buffer=False)

    # Register Service
    async def handle_trigger_prewarm(call):
        """Handle hybrid_llm.trigger_prewarm service call."""
        device_id = call.data.get("device_id")
        await async_trigger_prewarm(device_id)

    enable_prewarm = entry.options.get(CONF_ENABLE_PREWARM, DEFAULT_ENABLE_PREWARM)
    
    if enable_prewarm:
        hass.services.async_register(DOMAIN, "trigger_prewarm", handle_trigger_prewarm)
        
        # Register Satellite Monitors
        from homeassistant.helpers import entity_registry as er
        
        async def satellite_state_listener(event: Event):
            """Listen for all state changes to capture late-loading satellites."""
            entity_id = event.data.get("entity_id")
            new_state = event.data.get("new_state")
            
            if not entity_id or not new_state:
                return

            # Check domain (Strictly assist_satellite)
            if not entity_id.startswith("assist_satellite."):
                return
            
            # Helper to check if entity belongs to assist_satellite domain (more robust check)
            domain = entity_id.split(".", 1)[0]
            if domain != "assist_satellite":
                return

            if not new_state:
                return

            # Assist Satellite states: 'listening'
            active_states = {"listening"}
            
            if new_state.state in active_states:
                 _LOGGER.debug(f"Satellite {entity_id} woke up (state: {new_state.state})")
                 
                 # Resolve Device ID
                 er_entry = er.async_get(hass).async_get(entity_id)
                 if er_entry and er_entry.device_id:
                     await async_trigger_prewarm(er_entry.device_id)

        # Listen to ALL state changes to catch dynamically added satellites
        from homeassistant.const import EVENT_STATE_CHANGED
        entry.async_on_unload(
            hass.bus.async_listen(EVENT_STATE_CHANGED, satellite_state_listener)
        )
    else:
        _LOGGER.debug("Pre-warming is disabled in configuration.")

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    return True

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, [Platform.CONVERSATION])

