
from typing import Final

DOMAIN: Final = "hybrid_llm"

CONF_KEEP_ALIVE: Final = "keep_alive"
CONF_MODEL: Final = "model"
CONF_PROMPT: Final = "prompt"
CONF_NUM_CTX: Final = "num_ctx"
CONF_MAX_HISTORY: Final = "max_history"
CONF_LLM_HASS_API: Final = "llm_hass_api"
CONF_THINK: Final = "think"
CONF_FILLER_MODEL: Final = "filler_model"
CONF_FILLER_PROMPT: Final = "filler_prompt"
CONF_WAIT_FOR_FILLER: Final = "wait_for_filler"
CONF_ENABLE_NATIVE_INTENTS: Final = "enable_native_intents"
CONF_ENABLE_FUZZY_MATCHING: Final = "enable_fuzzy_matching"
CONF_ENABLE_TRACER: Final = "enable_tracer"
CONF_ENABLE_PREWARM: Final = "enable_prewarm"

from homeassistant.helpers import llm

DEFAULT_URL: Final = "http://localhost:11434"
DEFAULT_KEEP_ALIVE: Final = "5m"
DEFAULT_MODEL: Final = "llama3"
DEFAULT_PROMPT: Final = llm.DEFAULT_INSTRUCTIONS_PROMPT
DEFAULT_NUM_CTX: Final = 4096
DEFAULT_MAX_HISTORY: Final = 10
DEFAULT_FILLER_MODEL: Final = "qwen2.5:0.5b"
DEFAULT_FILLER_PROMPT: Final = "User said: '{{ text }}'. Respond with a short phrase like 'Checking kitchen lights'. Max 4 words."
DEFAULT_WAIT_FOR_FILLER: Final = False
DEFAULT_ENABLE_PREWARM: Final = True

FILLER_MODEL_ECHO: Final = "echo"

DEFAULT_ENABLE_NATIVE_INTENTS: Final = True
DEFAULT_ENABLE_FUZZY_MATCHING: Final = False
DEFAULT_ENABLE_TRACER: Final = False


COMMON_MODELS: Final = [
    "alfred",
    "all-minilm",
    "aya-expanse",
    "aya",
    "bakllava",
    "bespoke-minicheck",
    "bge-large",
    "bge-m3",
    "codebooga",
    "codegeex4",
    "codegemma",
    "codellama",
    "codeqwen",
    "codestral",
    "codeup",
    "command-r-plus",
    "command-r",
    "dbrx",
    "deepseek-coder-v2",
    "deepseek-coder",
    "deepseek-llm",
    "deepseek-v2.5",
    "deepseek-v2",
    "dolphin-llama3",
    "dolphin-mistral",
    "dolphin-mixtral",
    "dolphin-phi",
    "dolphincoder",
    "duckdb-nsql",
    "everythinglm",
    "falcon",
    "falcon2",
    "firefunction-v2",
    "gemma",
    "gemma2",
    "glm4",
    "goliath",
    "granite-code",
    "granite3-dense",
    "granite3-guardian",
    "granite3-moe",
    "hermes3",
    "internlm2",
    "llama-guard3",
    "llama-pro",
    "llama2-chinese",
    "llama2-uncensored",
    "llama2",
    "llama3-chatqa",
    "llama3-gradient",
    "llama3-groq-tool-use",
    "llama3.1",
    "llama3.2",
    "llama3",
    "llava-llama3",
    "llava-phi3",
    "llava",
    "magicoder",
    "mathstral",
    "meditron",
    "medllama2",
    "megadolphin",
    "minicpm-v",
    "mistral-large",
    "mistral-nemo",
    "mistral-openorca",
    "mistral-small",
    "mistral",
    "mistrallite",
    "mixtral",
    "moondream",
    "mxbai-embed-large",
    "nemotron-mini",
    "nemotron",
    "neural-chat",
    "nexusraven",
    "nomic-embed-text",
    "notus",
    "notux",
    "nous-hermes",
    "nous-hermes2-mixtral",
    "nous-hermes2",
    "nuextract",
    "open-orca-platypus2",
    "openchat",
    "openhermes",
    "orca-mini",
    "orca2",
    "paraphrase-multilingual",
    "phi",
    "phi3.5",
    "phi3",
    "phind-codellama",
    "qwen",
    "qwen2-math",
    "qwen2.5-coder",
    "qwen2.5",
    "qwen2",
    "reader-lm",
    "reflection",
    "samantha-mistral",
    "shieldgemma",
    "smollm",
    "smollm2",
    "snowflake-arctic-embed",
    "solar-pro",
    "solar",
    "sqlcoder",
    "stable-beluga",
    "stable-code",
    "stablelm-zephyr",
    "stablelm2",
    "starcoder",
    "starcoder2",
    "starling-lm",
    "tinydolphin",
    "tinyllama",
    "vicuna",
    "wizard-math",
    "wizard-vicuna-uncensored",
    "wizard-vicuna",
    "wizardcoder",
    "wizardlm-uncensored",
    "wizardlm",
    "wizardlm2",
    "xwinlm",
    "yarn-llama2",
    "yarn-mistral",
    "yi-coder",
    "yi",
    "zephyr",
]
