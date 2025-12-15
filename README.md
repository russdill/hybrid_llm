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

1.  Go to **Settings -> Devices & Services -> Add Integration**.
2.  Search for **Hybrid LLM**.
3.  Enter your Ollama URL (default: `http://localhost:11434`) and Model Name.

## Features

- **Sub-200ms Perceived Latency**: Plays filler audio immediately while the big model thinks.
- **Accurate Context**: Freezes entity states the moment you start speaking.
- **Tool Use**: Supports controlling your home via Home Assistant tools.
