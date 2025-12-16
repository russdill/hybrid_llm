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
            # Client creation can block on SSL context load
            self._client = await self.hass.async_add_executor_job(ollama.AsyncClient, url)
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
            
        # If we got here, it's a valid match we want to execute
        return await default_agent.async_recognize_intent(user_input)

    async def _async_handle_message(
        self, user_input: conversation.ConversationInput, chat_log: conversation.ChatLog
    ) -> conversation.ConversationResult:
        """Process a sentence - called by base class async_process."""
        # 0. Fast Path: Try native intents first (turn on light, etc.)
        try:
            # Check if Native Intents are enabled
            enable_native = self.entry.options.get(CONF_ENABLE_NATIVE_INTENTS, DEFAULT_ENABLE_NATIVE_INTENTS)
            
            if enable_native:
                result = await self._async_check_native_intent(user_input)
                if result:
                    return result

        except Exception as err:
            _LOGGER.debug("Native intent check failed: %s", err)

        # 1. Load Config
        settings = self.entry.options
        
        # Get LLM API setting - convert empty list to None
        llm_hass_api = settings.get(CONF_LLM_HASS_API)
        if not llm_hass_api:  # Handles None, empty list, empty string
            llm_hass_api = None
        
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_hass_api,
                settings.get(CONF_PROMPT, DEFAULT_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # 2. Generate Response via Ollama
        await self._async_handle_chat_log(chat_log, settings, user_input)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        settings: dict[str, Any],
        user_input: conversation.ConversationInput,
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
        _LOGGER.debug("Messages to Ollama: %s", messages)

        # Prepare tools if LLM API is available
        tools = None
        if chat_log.llm_api:
            tools = [
                self._convert_tool_to_ollama(tool)
                for tool in chat_log.llm_api.tools
            ]

        # Generate and stream filler BEFORE the main loop (only once)
        filler_text = await self._generate_filler(client, filler_model, filler_prompt_template, user_input)
        if filler_text:
            async def _filler_stream():
                yield {"role": "assistant", "content": filler_text}
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id, _filler_stream()
            ):
                pass

        # Main conversation loop
        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Debug: Log the full request payload
            _LOGGER.debug(
                "Ollama chat request [iteration %d] - model: %s, messages: %s, tools: %s",
                _iteration, model, messages, tools
            )
            
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
                _LOGGER.error("Unexpected error talking to Ollama server: %s", err)
                raise HomeAssistantError(
                    f"Sorry, I had a problem talking to the Ollama server: {err}"
                ) from err

            # Stream response through chat_log
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id,
                self._transform_stream(response_generator),
            ):
                # Content is collected by chat_log automatically
                pass

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
            _LOGGER.debug("Filler LLM response: %s", filler_text)
            return f"{filler_text}... " if filler_text else None
        except Exception as e:
            _LOGGER.warning(f"Filler failed: {e}")
            return None

    async def _transform_stream(
        self,
        response_generator,
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Transform Ollama stream to HA delta format."""
        full_response = []
        async for chunk in response_generator:
            message = chunk.get("message", {})
            delta: conversation.AssistantContentDeltaDict = {}

            if content := message.get("content"):
                delta["content"] = content
                full_response.append(content)

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

    def _convert_tool_to_ollama(self, tool) -> dict:
        """Convert a Home Assistant Tool definition to Ollama JSON Schema."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters.schema if hasattr(tool.parameters, 'schema') else tool.parameters,
            }
        }
