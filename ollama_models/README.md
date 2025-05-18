# Ollama Models

A collection of tools for managing and analyzing Ollama models.

## Installation

```bash
# Install from source
pip install -e .
```

## Usage

```bash
# Show help
ollama-models --help

# Show help for model commands
ollama-models model --help

# Show help for context commands
ollama-models context --help
```

## Commands

### Model Management

#### Fetch

Fetch and update the model database by scraping Ollama's website.

```bash
ollama-models model fetch [--output file.json]
```

#### Edit

Interactive tool for selecting specific model tags.

```bash
ollama-models model edit [--models-file file.json] [--config-file tags.conf]
```

#### Apply

Apply selected models to your Ollama instance.

```bash
ollama-models model apply [--config-file tags.conf]
```

#### Init

Initialize selected tags from your running Ollama instance.

```bash
ollama-models model init [--config-file tags.conf]
```

### Context Analysis

#### Usage

Generate a report on context usage for Ollama models.

```bash
ollama-models context usage [--output file.csv] [--model model-name]
```

#### Probe

Probe for maximum context sizes that fit in VRAM.

```bash
ollama-models context probe [--output file.csv] [--model model-name]
```

## Configuration

By default, the CLI connects to the Ollama API at `http://localhost:11434`. You can specify a different base URL with the `--api` option:

```bash
ollama-models --api http://ollama.example.com:11434 model fetch
```

## Development

### Prerequisites

```bash
pip install -r requirements.txt
```

### Testing

```bash
# Run tests
python -m unittest discover -s tests
```

## License

MIT
