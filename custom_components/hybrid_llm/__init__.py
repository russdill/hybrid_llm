
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
)

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

    # Check for Tracer Config
    enable_tracer = entry.options.get(CONF_ENABLE_TRACER, DEFAULT_ENABLE_TRACER)
    tracer = None
    
    if enable_tracer:
        # Create and store tracer
        from .tracer import PerformanceTracer
        tracer = PerformanceTracer(hass.config.path("traces"))
        hass.data[DOMAIN]["tracer"] = tracer

    async def handle_pipeline_start(event: Event):
        """Handle pipeline start event to freeze state and pre-warm."""
        data = event.data
        pipeline_run_id = data.get("pipeline_execution_id")
        
        if not pipeline_run_id:
            return

        # Start Trace if enabled
        if tracer:
            tracer.start_trace(pipeline_run_id)
            tracer.trace_event(pipeline_run_id, "Pipeline Run", "B", "pipeline")

        _LOGGER.debug(f"Pipeline started: {pipeline_run_id}. Freezing state...")
        
        # 1. Generate Prompt (State Freeze)
        from homeassistant.helpers import template, llm
        
        # Create Context
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=None, 
            language=data.get("language", "en"),
            assistant=None, 
            device_id=data.get("device_id")
        )

        try:
            # Render Template
            prompt_template = template.Template(system_prompt_template, hass)
            system_prompt = prompt_template.async_render(
                parse_result=False,
                variables={"llm_context": llm_context}
            )
        except Exception as e:
            _LOGGER.warning(f"Failed to render prompt template: {e}")
            system_prompt = "You are a helpful assistant."
        
        # 2. Store in Cache
        hass.data[DOMAIN]["cache"][pipeline_run_id] = system_prompt
        
        # 3. Fire-and-forget Pre-warm request
        # Instrument Pre-warm
        async def trace_prewarm():
            if tracer:
                tracer.trace_event(pipeline_run_id, "LLM Pre-warm", "B", "hybrid_llm")
            
            await prewarm_ollama(pipeline_run_id, system_prompt, url, main_model)
            
            if tracer:
                tracer.trace_event(pipeline_run_id, "LLM Pre-warm", "E", "hybrid_llm")

        hass.async_create_task(trace_prewarm())

    async def handle_pipeline_end(event: Event):
        """Handle pipeline end."""
        run_id = event.data.get("pipeline_execution_id")
        if run_id and tracer:
            tracer.trace_event(run_id, "Pipeline Run", "E", "pipeline")
            tracer.dump(run_id)

    async def handle_stage_start(event: Event):
        """Trace stage start."""
        run_id = event.data.get("pipeline_execution_id")
        stage = event.data.get("stage")
        if run_id and stage and tracer:
            tracer.trace_event(run_id, f"Stage: {stage}", "B", "pipeline")

    async def handle_stage_end(event: Event):
        """Trace stage end."""
        run_id = event.data.get("pipeline_execution_id")
        stage = event.data.get("stage")
        if run_id and stage and tracer:
             tracer.trace_event(run_id, f"Stage: {stage}", "E", "pipeline")

    # Listen for Assist Pipeline events
    entry.async_on_unload(hass.bus.async_listen("assist_pipeline_pipeline_start", handle_pipeline_start))
    
    if tracer:
        entry.async_on_unload(hass.bus.async_listen("assist_pipeline_pipeline_finished", handle_pipeline_end))
        entry.async_on_unload(hass.bus.async_listen("assist_pipeline_stage_start", handle_stage_start))
        entry.async_on_unload(hass.bus.async_listen("assist_pipeline_stage_finish", handle_stage_end))
    
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    return True

async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, [Platform.CONVERSATION])

