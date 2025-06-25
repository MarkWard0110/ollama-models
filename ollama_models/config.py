"""
Configuration management for the Ollama Models CLI.
"""
import os
from pathlib import Path
import json

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.environ.get("OLLAMA_MODELS_DATA_DIR", str(PROJECT_ROOT))
CURRENT_DIR = os.getcwd()

# Default filenames
MODELS_JSON_FILENAME = "ollama_models.json"
CONFIG_FILENAME = "ollama_models.conf"
CONTEXT_USAGE_FILENAME = "context_usage.csv"
MAX_CONTEXT_FILENAME = "context_probe.csv"
HOST_CONFIG_FILENAME = "ollama_host.conf"

# Default file paths - now checking the current directory first
DEFAULT_MODELS_JSON = os.environ.get(
    "OLLAMA_MODELS_JSON", 
    os.path.join(CURRENT_DIR, MODELS_JSON_FILENAME)
)

DEFAULT_CONFIG_FILE = os.environ.get(
    "OLLAMA_MODELS_CONFIG", 
    os.path.join(CURRENT_DIR, CONFIG_FILENAME)
)

DEFAULT_CONTEXT_USAGE_CSV = os.environ.get(
    "OLLAMA_MODELS_CONTEXT_USAGE", 
    os.path.join(CURRENT_DIR, CONTEXT_USAGE_FILENAME)
)

DEFAULT_MAX_CONTEXT_CSV = os.environ.get(
    "OLLAMA_MODELS_MAX_CONTEXT", 
    os.path.join(CURRENT_DIR, MAX_CONTEXT_FILENAME)
)

DEFAULT_HOST_CONFIG_FILE = os.environ.get(
    "OLLAMA_HOST_CONFIG", 
    os.path.join(CURRENT_DIR, HOST_CONFIG_FILENAME)
)

# API configuration
DEFAULT_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
API_TIMEOUT = int(os.environ.get("OLLAMA_API_TIMEOUT", "1200")) 

def get_file_path(filename, default_dir=DATA_DIR):
    """
    Get an absolute file path for the given filename.
    
    Args:
        filename (str): The filename
        default_dir (str): The default directory if not absolute
        
    Returns:
        str: Absolute file path
    """
    if os.path.isabs(filename):
        return filename
    return os.path.join(default_dir, filename)

def load_api_base_from_config(config_path):
    """Load the Ollama API base URL from a config file (JSON or simple text)."""
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            # Try JSON first
            try:
                data = json.load(f)
                return data.get('api_base')
            except json.JSONDecodeError:
                f.seek(0)
                # Fallback: treat as plain text (single line with URL)
                line = f.readline().strip()
                if line:
                    return line
    except Exception:
        pass
    return None
