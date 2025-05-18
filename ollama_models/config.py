"""
Configuration management for the Ollama Models CLI.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.environ.get("OLLAMA_MODELS_DATA_DIR", str(PROJECT_ROOT))

# Default file paths
DEFAULT_MODELS_JSON = os.environ.get(
    "OLLAMA_MODELS_JSON", 
    os.path.join(DATA_DIR, "ollama_models.json")
)

DEFAULT_CONFIG_FILE = os.environ.get(
    "OLLAMA_MODELS_CONFIG", 
    os.path.join(DATA_DIR, "selected_tags.conf")
)

DEFAULT_CONTEXT_USAGE_CSV = os.environ.get(
    "OLLAMA_MODELS_CONTEXT_USAGE", 
    os.path.join(DATA_DIR, "context_usage.csv")
)

DEFAULT_MAX_CONTEXT_CSV = os.environ.get(
    "OLLAMA_MODELS_MAX_CONTEXT", 
    os.path.join(DATA_DIR, "max_context.csv")
)

# API configuration
DEFAULT_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
API_TIMEOUT = int(os.environ.get("OLLAMA_API_TIMEOUT", "600"))  # 10 minutes default

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
