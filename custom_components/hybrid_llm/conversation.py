"""Hybrid Conversation Agent for Home Assistant."""
from typing import Literal, AsyncGenerator, Any
import logging
import ollama
import json

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm, template, intent
from homeassistant.util import ulid
from homeassistant.const import MATCH_ALL
from homeassistant.exceptions import HomeAssistantError, TemplateError

from .const import (
    DOMAIN,
    CONF_URL,
    CONF_KEEP_ALIVE,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_NUM_CTX,
    CONF_MAX_HISTORY,
    CONF_LLM_HASS_API,
    CONF_THINK,
    CONF_FILLER_MODEL,
    CONF_FILLER_PROMPT,
    DEFAULT_URL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_NUM_CTX,
    DEFAULT_MAX_HISTORY,
    DEFAULT_FILLER_MODEL,
    CONF_FILLER_PROMPT,
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
    DEFAULT_ENABLE_NATIVE_INTENTS,
    DEFAULT_ENABLE_FUZZY_MATCHING,
    FILLER_MODEL_ECHO,
)
from .helpers import convert_tool_to_ollama

_LOGGER = logging.getLogger(__name__)

# Maximum number of tool call iterations to prevent infinite loops
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up the conversation agent."""
    agent = HybridConversationAgent(hass, config_entry)
    async_add_entities([agent])


class HybridConversationAgent(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Hybrid Local Voice Backend Conversation Agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self.history: dict[str, list[dict]] = {}
        self._client: ollama.AsyncClient | None = None
        self._client_url: str | None = None

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @property
    def supported_features(self) -> conversation.ConversationEntityFeature:
        """Return supported features."""
        enable_control = self.entry.options.get(CONF_LLM_HASS_API)
        if enable_control and enable_control != "none":
            return conversation.ConversationEntityFeature.CONTROL
        return conversation.ConversationEntityFeature(0)

    async def async_added_to_hass(self) -> None:
        """Register as conversation agent."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def _get_client(self, url: str) -> ollama.AsyncClient:
        """Get or create Ollama client in a non-blocking way."""
        if self._client is None or self._client_url != url:
            # Client creation can block on SSL context load unless verify=False
            self._client = ollama.AsyncClient(host=url, verify=False)
            self._client_url = url
        return self._client

    async def async_will_remove_from_hass(self) -> None:
        """Unregister as conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_check_native_intent(self, user_input) -> conversation.ConversationResult | None:
        """Check for native intent match using debug interface to filter fuzzy matches."""
        default_agent = conversation.async_get_agent(self.hass)
        if not default_agent or default_agent == self:
            return None

        # Use async_debug_recognize to peek at potential match details
        debug_result = await default_agent.async_debug_recognize(user_input)
        
        if not debug_result or not debug_result.get("match"):
            return None
            
        is_fuzzy = debug_result.get("fuzzy_match", False)
        
        # Check if fuzzy matching is enabled (default False since Phase 19)
        enable_fuzzy = self.entry.options.get(CONF_ENABLE_FUZZY_MATCHING, DEFAULT_ENABLE_FUZZY_MATCHING)
        
        if is_fuzzy and not enable_fuzzy:
            _LOGGER.debug("Ignoring fuzzy match for '%s' because fuzzy matching is disabled.", user_input.text)
            return None
            
        # If we got here, it's a valid match we want to execute.
        # Delegate directly to the native agent's process method.
        # This handles intent execution (turning on lights) AND response generation (templated speech).
        return await default_agent.async_process(user_input)

    async def _async_handle_message(
        self, user_input: conversation.ConversationInput, chat_log: conversation.ChatLog
    ) -> conversation.ConversationResult:
        """Process a sentence - called by base class async_process."""
        # Get tracer
        tracer = self.hass.data.get(DOMAIN, {}).get("tracer")
        
        # 1. Load context settings & Cache Check (Optimized Order)
        settings = self.entry.options
        prompt_template = settings.get(CONF_PROMPT, DEFAULT_PROMPT)
        
        fresh_state = self.hass.data.get(DOMAIN, {}).get("fresh_state")
        run_id = None
        
        # Check for fresh state (Pre-warmer hit)
        if fresh_state:
            import time
            age = time.time() - fresh_state["timestamp"]
            cached_device_id = fresh_state.get("device_id")
            
            # Verify age and device compatibility
            # Strict match: device_id MUST match.
            device_match = (cached_device_id == user_input.device_id)
            
            if age < 45.0 and device_match:
                _LOGGER.debug(f"Pre-warmer HIT! Using cached state (age: {age:.2f}s, run_id: {fresh_state.get('run_id')})")
                prompt_template = fresh_state["prompt"]
                
                # Adopt cached run_id
                if "run_id" in fresh_state:
                    run_id = fresh_state["run_id"]
                    if tracer:
                        # Log continuation event
                        tracer.trace_event(run_id, "Conversation Phase", "B", "pipeline", 
                            args={"event": "trace_continued", "conversation_id": user_input.conversation_id}
                        )
            else:
                reason = "Expired" if age >= 15.0 else f"Device/Context Mismatch ({cached_device_id} vs {user_input.device_id})"
                _LOGGER.debug(f"Pre-warmer MISS ({reason}: {age:.2f}s)")
            
            # Clear cache
            del self.hass.data[DOMAIN]["fresh_state"]

        # If no run_id from cache, start new trace
        if not run_id:
            run_id = ulid.ulid()
            if tracer:
                tracer.start_trace(run_id)
                tracer.trace_event(run_id, "Conversation Phase", "B", "pipeline", 
                    args={"conversation_id": user_input.conversation_id, "text": user_input.text})

        try:
            # 0. Fast Path: Try native intents first (turn on light, etc.)
            try:
                # Check if Native Intents are enabled
                enable_native = self.entry.options.get(CONF_ENABLE_NATIVE_INTENTS, DEFAULT_ENABLE_NATIVE_INTENTS)
                
                if enable_native:
                    # Trace Native Check
                    if tracer:
                         tracer.trace_event(run_id, "Native Intent Check", "B", "hybrid_llm")
                         
                    result = await self._async_check_native_intent(user_input)
                    
                    if tracer:
                         tracer.trace_event(run_id, "Native Intent Check", "E", "hybrid_llm")

                    if result:
                        if tracer:
                            # If we matched native, we are done.
                            tracer.trace_event(run_id, "Conversation Phase", "E", "pipeline", args={"outcome": "native_intent"})
                            await tracer.dump(self.hass, run_id)
                        return result
            except Exception as err:
                _LOGGER.debug("Native intent check failed: %s", err)

            extra_system_prompt = user_input.extra_system_prompt

            
            # Get LLM API setting - convert empty list to None
            llm_hass_api = settings.get(CONF_LLM_HASS_API)
            if not llm_hass_api:  # Handles None, empty list, empty string
                llm_hass_api = None
            
            try:
                await chat_log.async_provide_llm_data(
                    user_input.as_llm_context(DOMAIN),
                    llm_hass_api,
                    prompt_template,
                    extra_system_prompt,
                )
            except conversation.ConverseError as err:
                if tracer:
                    tracer.trace_event(run_id, "Conversation Phase", "E", "pipeline", args={"error": str(err)})
                    await tracer.dump(self.hass, run_id)
                return err.as_conversation_result()

            # 2. Generate Response via Ollama
            await self._async_handle_chat_log(chat_log, settings, user_input, tracer, run_id)

            if tracer:
                # Inspect chat_log to find the assistant response we just added
                assistant_response_text = "ERROR: No response found"
                if chat_log.content and isinstance(chat_log.content[-1], conversation.AssistantContent):
                    assistant_response_text = chat_log.content[-1].content
                
                tracer.trace_event(run_id, "Conversation Phase", "E", "pipeline", args={
                    "outcome": "llm_response",
                    "input": user_input.text,
                    "response": assistant_response_text
                })
                await tracer.dump(self.hass, run_id)

            return conversation.async_get_result_from_chat_log(user_input, chat_log)
            
        except Exception as e:
            if tracer:
                tracer.trace_event(run_id, "Conversation Phase", "E", "pipeline", args={"exception": str(e)})
                await tracer.dump(self.hass, run_id)
            raise e

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        settings: dict[str, Any],
        user_input: conversation.ConversationInput,
        tracer=None,
        run_id: str | None = None,
    ) -> None:
        """Generate an answer for the chat log using Ollama."""
        url = settings.get(CONF_URL, DEFAULT_URL)
        model = settings.get(CONF_MODEL, DEFAULT_MODEL)
        filler_model = settings.get(CONF_FILLER_MODEL, DEFAULT_FILLER_MODEL)
        filler_prompt_template = settings.get(CONF_FILLER_PROMPT, DEFAULT_FILLER_PROMPT)
        keep_alive = settings.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)
        num_ctx = settings.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)
        max_history = settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)

        client = await self._get_client(url)

        # Build message history from chat_log
        messages = self._build_messages_from_chat_log(chat_log, max_history)
        
        # Debug: Log the messages being sent
        _LOGGER.debug("Chat log content types: %s", [type(c).__name__ for c in chat_log.content])

        # Prepare tools if LLM API is available
        tools = None
        if chat_log.llm_api:
            tools = [
                convert_tool_to_ollama(tool)
                for tool in chat_log.llm_api.tools
            ]

        # Generate and stream filler BEFORE the main loop (only once)
        if tracer:
            tracer.trace_event(run_id, "Filler Inference", "B", "hybrid_llm")
            
        filler_text = await self._generate_filler(client, filler_model, filler_prompt_template, user_input)
        
        if tracer:
            tracer.trace_event(run_id, "Filler Inference", "E", "hybrid_llm")
            
        if filler_text:
            async def _filler_stream():
                yield {"role": "assistant", "content": filler_text}
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id, _filler_stream()
            ):
                pass
            
            # Append filler to messages so LLM sees it as recent history
            # Order: User Input -> Assistant (Filler)
            messages.append({"role": "assistant", "content": filler_text})

        _LOGGER.debug("Messages to Ollama: %s", messages)

        # Main conversation loop
        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Debug: Log the full request payload
            _LOGGER.debug(
                "Ollama chat request [iteration %d] - model: %s, messages: %s, tools: %s",
                _iteration, model, messages, tools
            )
            
            if tracer:
                tracer.trace_event(run_id, f"Main LLM Inference {_iteration}", "B", "hybrid_llm")
            
            try:
                response_generator = await client.chat(
                    model=model,
                    messages=list(messages),
                    tools=tools,
                    stream=True,
                    keep_alive=f"{keep_alive}s" if isinstance(keep_alive, int) else keep_alive,
                    options={"num_ctx": num_ctx},
                    think=settings.get(CONF_THINK),
                )
            except (ollama.RequestError, ollama.ResponseError) as err:
                 if tracer:
                     tracer.trace_event(run_id, f"Main LLM Inference {_iteration}", "E", "hybrid_llm")
                 _LOGGER.error("Unexpected error talking to Ollama server: %s", err)
                 raise HomeAssistantError(
                     f"Sorry, I had a problem talking to the Ollama server: {err}"
                 ) from err

            # Stream response through chat_log
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._transform_stream(response_generator, run_id, tracer, _iteration),
            ):
                # Content is collected by chat_log automatically
                pass
            
            if tracer:
                tracer.trace_event(run_id, f"Main LLM Inference {_iteration}", "E", "hybrid_llm")

            # Check if there are unresolved tool calls
            if not chat_log.unresponded_tool_results:
                break

            # Rebuild messages with tool results for next iteration
            messages = self._build_messages_from_chat_log(chat_log, max_history)

    async def _generate_filler(
        self,
        client: ollama.AsyncClient,
        filler_model: str,
        filler_prompt_template: str,
        user_input: conversation.ConversationInput,
    ) -> str | None:
        """Generate a filler phrase using a fast, small model."""
        # Render the prompt template with Jinja2
        try:
            filler_prompt = template.Template(filler_prompt_template, self.hass).async_render(
                {
                    "text": user_input.text,
                    "ha_name": self.hass.config.location_name,
                },
                parse_result=False,
            )
        except TemplateError as e:
            _LOGGER.warning("Filler prompt template error: %s", e)
            filler_prompt = f"User said: '{user_input.text}'. Checking..."
        
        _LOGGER.debug("Filler LLM request - model: %s, prompt: %s", filler_model, filler_prompt)
        
        # Echo Mode: If configured to "echo", return the rendered prompt immediately
        if filler_model == FILLER_MODEL_ECHO:
            _LOGGER.debug("Filler in Echo Mode: returning prompt as filler")
            # Only return if there is text (trim whitespace)
            prompt_text = filler_prompt.strip()
            return f"{prompt_text}... " if prompt_text else None

        try:
            filler_response = await client.generate(
                model=filler_model,
                prompt=filler_prompt,
                stream=False,
                keep_alive="5m"
            )
            filler_text = filler_response.get("response", "").strip()
        
            if filler_text:
                total_duration = filler_response.get("total_duration", 0) / 10**9
                load_duration = filler_response.get("load_duration", 0) / 10**9
                prompt_eval_duration = filler_response.get("prompt_eval_duration", 0) / 10**9
                
                _LOGGER.debug(
                    f"Filler LLM response: {filler_text} "
                    f"(Total: {total_duration:.3f}s, Load: {load_duration:.3f}s, Prompt Eval: {prompt_eval_duration:.3f}s)"
                )
                return f"{filler_text}... "
                
            return None
        except Exception as e:
            _LOGGER.warning(f"Filler failed: {e}")
            return None

    async def _transform_stream(
        self,
        response_generator,
        run_id: str = None,
        tracer = None,
        iteration: int = 0
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Transform Ollama stream to HA delta format."""
        full_response = []
        first_token = True
        
        async for chunk in response_generator:
            if first_token and tracer and run_id:
                tracer.trace_event(run_id, f"First Token {iteration}", "i", "hybrid_llm")
                first_token = False
                
            message = chunk.get("message", {})
            delta: conversation.AssistantContentDeltaDict = {}

            if content := message.get("content"):
                delta["content"] = content
                full_response.append(content)
                if tracer and run_id:
                     tracer.trace_event(run_id, f"Stream Chunk {iteration}", "i", "hybrid_llm", args={"content": content})

            if tool_calls := message.get("tool_calls"):
                _LOGGER.debug("Ollama tool call: %s", tool_calls)
                delta["tool_calls"] = [
                    llm.ToolInput(
                        id=f"tool_{i}",
                        tool_name=tc["function"]["name"],
                        tool_args=tc["function"].get("arguments", {}),
                    )
                    for i, tc in enumerate(tool_calls)
                ]

            if delta:
                yield delta
            
            if chunk.get("done"):
                total_duration = chunk.get("total_duration", 0) / 10**9
                load_duration = chunk.get("load_duration", 0) / 10**9
                prompt_eval_duration = chunk.get("prompt_eval_duration", 0) / 10**9
                eval_duration = chunk.get("eval_duration", 0) / 10**9
                prompt_eval_count = chunk.get("prompt_eval_count", 0)
                eval_count = chunk.get("eval_count", 0)
                
                _LOGGER.debug(
                    f"Ollama Response Final. Total: {total_duration:.3f}s, "
                    f"Load: {load_duration:.3f}s, "
                    f"Prompt Eval: {prompt_eval_duration:.3f}s ({prompt_eval_count} toks), "
                    f"Eval: {eval_duration:.3f}s ({eval_count} toks)"
                )
                
                if tracer and run_id:
                     tracer.trace_event(run_id, f"Main LLM Stats {iteration}", "i", "hybrid_llm", args={
                         "total_duration": total_duration,
                         "load_duration": load_duration,
                         "prompt_eval_duration": prompt_eval_duration,
                         "eval_duration": eval_duration,
                         "prompt_eval_count": prompt_eval_count,
                         "eval_count": eval_count
                     })
        
        # Log the complete response at the end
        _LOGGER.debug("Ollama complete response: %s", "".join(full_response))

    def _build_messages_from_chat_log(
        self, chat_log: conversation.ChatLog, max_history: int
    ) -> list[dict]:
        """Convert ChatLog content to Ollama message format."""
        messages = []

        for content in chat_log.content:
            if isinstance(content, conversation.SystemContent):
                messages.append({"role": "system", "content": content.content})
            elif isinstance(content, conversation.UserContent):
                messages.append({"role": "user", "content": content.content})
            elif isinstance(content, conversation.AssistantContent):
                msg = {"role": "assistant", "content": content.content or ""}
                if content.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.tool_name,
                                "arguments": tc.tool_args,
                            }
                        }
                        for tc in content.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(content, conversation.ToolResultContent):
                messages.append({
                    "role": "tool",
                    "content": json.dumps(content.tool_result),
                })

        # Trim to max_history (keep system prompt)
        if max_history > 0 and len(messages) > max_history * 2 + 1:
            messages = [messages[0]] + messages[-(max_history * 2):]

        return messages


