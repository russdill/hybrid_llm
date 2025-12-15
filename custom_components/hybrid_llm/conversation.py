"""Hybrid Conversation Agent for Home Assistant."""
from typing import Literal, AsyncGenerator, Any
import logging
import asyncio
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
    CONF_DISABLE_FUZZY_MATCHING,
    DEFAULT_URL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_NUM_CTX,
    DEFAULT_MAX_HISTORY,
    DEFAULT_FILLER_MODEL,
    DEFAULT_FILLER_PROMPT,
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
        self._lock = asyncio.Lock()

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

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a sentence - called by base class async_process."""
        # 0. Fast Path: Try native intents first (turn on light, etc.)
        try:
            default_agent = conversation.async_get_agent(self.hass)
            if default_agent and default_agent != self:
                disable_fuzzy = self.entry.options.get(CONF_DISABLE_FUZZY_MATCHING, False)
                result = None

                if disable_fuzzy:
                    async with self._lock:
                        # Safely toggle fuzzy matching on the shared agent
                        old_fuzzy = getattr(default_agent, "fuzzy_matching", True)
                        try:
                            default_agent.fuzzy_matching = False
                            result = await default_agent.async_recognize_intent(user_input)
                        finally:
                            default_agent.fuzzy_matching = old_fuzzy
                else:
                    # passing strict_intents_only=False to allow fuzzy matching if configured
                    result = await default_agent.async_recognize_intent(user_input)
                
                # If we have a match
                if result:
                   # 2. EXECUTE with TEMP ID (avoids polluting log if fails)
                   temp_input = conversation.ConversationInput(
                       text=user_input.text,
                       context=user_input.context,
                       conversation_id=f"_temp_{ulid.ulid_now()}", 
                       device_id=user_input.device_id,
                       language=user_input.language,
                       agent_id=user_input.agent_id,
                       satellite_id=user_input.satellite_id,
                   )
                   processed_result = await default_agent.internal_async_process(temp_input)
                   
                   # 3. Check Success
                   if processed_result.response.response_type != intent.IntentResponseType.ERROR:
                       # Fix ID so it looks like it processed the original input
                       processed_result.conversation_id = user_input.conversation_id
                       _LOGGER.debug("Native intent matched and executed: %s", processed_result.response.intent)
                       return processed_result
                   
                   # If execution resulted in error, we fall through to LLM (ignoring the error log)
                   _LOGGER.debug("Native intent matched but execution failed, falling back to LLM")
        except Exception as e:
            _LOGGER.debug("Native intent check failed: %s", e)

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
