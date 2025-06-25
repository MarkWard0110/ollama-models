# Ollama Models

A collection of tools for managing and analyzing Ollama models.

# Ollama Models – Usage Guide

Ollama Models provides a suite of command-line tools for managing, analyzing, and optimizing models from the Ollama ecosystem.

## Getting Started

Depending on your goals and the current state of your Ollama server, you can begin in one of two ways:

### 1. Start Fresh: Define Your Desired Model Configuration
- Run `ollama-models model edit` to interactively select the models and tags you want to use.
- When you later run `ollama-models model apply`, the tool will:
  - Pull any models that are missing from your Ollama server.
  - Remove any models from Ollama that are not in your selected configuration.
- This is ideal if you want to curate a specific set of models and keep your Ollama environment in sync with your chosen configuration.

### 2. Preserve Your Current Ollama Setup
- Run `ollama-models model init` to record your current Ollama model configuration into a config file.
- This lets you capture your existing setup and manage it going forward, without making immediate changes to your Ollama server.
- Use this if you want to start managing your current environment as-is, and make changes incrementally.

**Model Sources and Customization:**
- The `ollama-models model fetch` command retrieves only the default models available from Ollama.com.
- You can add additional models—including user-uploaded or Hugging Face models—by editing your `selected_tags.conf` file. The tool will support keeping your Ollama server configured with these custom model references, alongside the defaults.

**Tip:**
- The tool includes a starter `ollama_models.json` file with a snapshot of available models. You can fetch the latest model data at any time using `ollama-models model fetch --output models.json`. This creates a new file and does not overwrite the original.
- Fetching model data relies on scraping Ollama.com. If the website changes, fetching may fail until the tool is updated.

## Installation

It is recommended to install `ollama-models` in a virtual environment or globally, and use it from a separate working directory (not the source code folder). This keeps your Ollama configuration and model files organized and under version control (e.g., with Git), separate from the tool's source code.

**Install from source:**

First, clone the repository (anywhere on your system):

```
git clone https://github.com/MarkWard0110/ollama-models.git
```

Then, from your working environment (such as your virtual environment or system Python), install the package by referencing the path to the cloned repo. For example:

```
pip install /path/to/ollama-models
```

Replace `/path/to/ollama-models` with the actual path where you cloned the repository.

After installation, create a new directory for your Ollama configuration and managed files:

```
mkdir my-ollama-config
cd my-ollama-config
```

You can now run the `ollama-models` CLI from this directory. All files generated (e.g., `models.json`, `config.json`) will be kept here, making it easy to manage with Git or other tools.

## CLI Overview

The main entry point is the `ollama-models` CLI, which supports two primary command groups:

### 1. Model Management (`model`)

- **fetch**: Scrape and update the local model database from Ollama’s website.
- **edit**: Launch an interactive tag selector to customize model tags and configurations.
- **apply**: Apply your selected tag configuration to your local Ollama instance.
- **init**: Initialize your tag configuration from the Ollama API.

#### Example:

```
ollama-models model fetch --output models.json
ollama-models model edit --models-file models.json
ollama-models model apply --config-file config.json
```

### 2. Context Analysis (`context`)

- **usage**: Generate a CSV report of context usage for all (or a specific) model.
- **probe**: Probe and record the maximum context size that fits in VRAM for each model.
  - The `--max-vram` argument allows you to set an upper limit for VRAM usage during probing. This is useful if you:
    - Need to reserve VRAM for other applications running on your system.
    - Have multiple GPUs and want to restrict probing to the capacity of a single GPU for optimal performance (avoiding model sharding across devices).
    - Want to load multiple models into VRAM and need to determine the maximum context size for each within a shared VRAM budget.

> **Note:** For best results, run context analysis tools (usage/probe) when your Ollama server is *not* being used for other tasks. This ensures accurate measurements and avoids interfering with running models or workloads.

#### Example:

```
ollama-models context usage --output usage.csv
ollama-models context probe --output max_context.csv --max-vram 8
```

**Tip:** For more details on any command or its arguments, use the `--help` flag with any CLI command or subcommand. For example:
```
ollama-models --help
ollama-models model --help
ollama-models context probe --help
```
This will display available options, arguments, and usage examples for each command.

## Features

- **Automated scraping** of Ollama’s model library
- **Interactive tag selection** for fine-grained model configuration
- **Context window and memory probing** for hardware-aware deployment
- **Extensive logging** and error handling for robust automation

## Configuration Files

- `selected_tags.conf`: The main configuration file for your selected models and tags. This plain text file (one model/tag per line) is created and updated by the tool and determines which models Ollama will keep installed. You can edit this file directly or manage it through the CLI.
- `ollama_models.json`: The default models database included with the package. Used as a starting point or fallback if no local models file is present.
- `models.json`: A user-generated or updated models database, typically created by running `ollama-models model fetch`. Contains the latest scraped model data from Ollama.com.
- `context_usage.csv`: Output file generated by the `context usage` command, containing context usage statistics.
- `context_probe.csv`: Output file generated by the `context probe` command, containing maximum context size probe results.

## Advanced

- Modular Python codebase for easy extension
- Integrates with Ollama’s API and local environment

---

For more information, see the source code and comments in each module.
