<p align="center">
  <img src="logo.png" alt="Hybrid LLM Logo" width="200">
</p>

# Hybrid Local Voice Backend

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=russdill&repository=hybrid_llm&category=integration)

A high-performance, hybrid conversation agent for Home Assistant that combines:
1.  **Local LLM Streaming** (via Ollama)
2.  **Context Pre-warming** (State Freeze at Wake Word)
3.  **Smart Filler Audio** (Immediate "Checking that..." feedback)

## Installation

### Easy Install (My Home Assistant)
If you have HACS installed:

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=russdill&repository=hybrid_llm&category=integration)

### Manual HACS Install
1.  Go to HACS -> Integrations -> 3 dots -> Custom repositories.
2.  Add the URL of this repository.
3.  Category: **Integration**.
4.  Click **Add**, then install "Hybrid Local Voice Backend".
5.  Restart Home Assistant.

## Configuration

Go to **Settings -> Devices & Services -> Add Integration** and search for **Hybrid LLM**.

### Main Settings
| Option | Description | Default |
| :--- | :--- | :--- |
| **Ollama URL** | URL of your Ollama server. | `http://localhost:11434` |
| **Enable Native Intents** | If checked, Home Assistant's built-in intent recognition (HassTurnOn, etc.) will be checked *before* using the LLM. This provides a fast path for simple commands. | `True` |
| **Enable Fuzzy Matching** | If checked, native intents will use fuzzy string matching. **Recommendation:** Leave unchecked (`False`) to let the LLM handle inexact phrasing, as it is smarter than the standard fuzzy matcher. | `False` |
| **Main Model** | The large LLM (e.g., `llama3`, `mxbai-embed-large` etc) used for complex queries and tool usage. | `llama3` |

### Filler Settings
The "Filler" is the text spoken immediately while the main LLM is thinking (to reduce perceived latency).

| Option | Description |
| :--- | :--- |
| **Filler Model** | Select a small, fast model (e.g., `qwen2:0.5b`) to generate dynamic filler, OR select **Echo (Use prompt as filler)** to skip the LLM call entirely. |
| **Filler Prompt** | A **Jinja2** template used to generate the filler text. |

#### Filler Templates
The **Filler Prompt** supports Jinja2 templates. You can use:
- `{{ text }}`: The user's spoken text.
- `{{ ha_name }}`: The name of your Home Assistant installation.

**Example 1: Dynamic Echo (Randomized)**
In **Echo** mode, you can use Jinja's `random` filter to vary the response without using an LLM:
```jinja
{{ [
  "Checking on " + text + "...",
  "One second, looking up " + text + "...",
  "On it...",
  "Just a moment..."
] | random }}
```

**Example 2: LLM Context**
If using a real model (e.g., `qwen2:0.5b`), use a prompt that instructs the model to be brief:
```text
User said: '{{ text }}'. Echo the subject with a 'checking' phrase. Max 4 words.
```

## Features

- **Sub-200ms Perceived Latency**: Plays filler audio immediately while the big model thinks.
- **Accurate Context**: Freezes entity states the moment you start speaking.
- **Tool Use**: Supports controlling your home via Home Assistant tools.
