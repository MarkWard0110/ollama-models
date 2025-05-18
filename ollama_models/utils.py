"""
Utility functions for the Ollama Models CLI.
"""
import requests
import os
import csv
import logging
from ollama_models.config import DEFAULT_API_BASE, API_TIMEOUT

# Default API base URL
API_BASE = DEFAULT_API_BASE

def fetch_installed_models():
    """
    Fetch installed models from the Ollama API.
    
    Returns:
        list: List of installed models
    
    Raises:
        ConnectionError: If the API is unreachable
    """
    try:
        resp = requests.get(f"{API_BASE}/api/tags", timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])
    except requests.RequestException as e:
        logger = logging.getLogger("ollama_models.utils")
        logger.error(f"Error fetching models from {API_BASE}: {str(e)}")
        if isinstance(e, requests.ConnectionError):
            raise ConnectionError(f"Could not connect to Ollama API at {API_BASE}. Is Ollama running?")
        return []

def fetch_max_context_size(model_name):
    """
    Fetch the maximum context size for a given model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        int: Maximum context size for the model
    """
    logger = logging.getLogger("ollama_models.utils")
    try:
        resp = requests.post(f"{API_BASE}/api/show", json={"model": model_name}, timeout=API_TIMEOUT)
        resp.raise_for_status()
        info = resp.json().get("model_info", {})
        for key, value in info.items():
            if key.endswith(".context_length"):
                return value
        logger.debug(f"No context_length property found for {model_name}. Using default: 2048")
        return 2048
    except requests.RequestException as e:
        logger.warning(f"Error fetching context size for {model_name}: {str(e)}")
        return 2048

def try_model_call(model_name, context_size):
    """
    Test if a model can handle a given context size.
    
    Args:
        model_name (str): Name of the model
        context_size (int): Context size to test
        
    Returns:
        bool: True if the model can handle the context size, False otherwise
    """
    logger = logging.getLogger("ollama_models.utils")
    
    # Try a chat completion first
    payload_chat = {
        "model": model_name,
        "prompt": "What is the capital of France?",
        "stream": False,
        "options": {
            "num_ctx": context_size,
            "max_tokens": 64,
        }
    }
    try:
        logger.debug(f"Testing {model_name} with context size {context_size} via chat API")
        resp = requests.post(f"{API_BASE}/api/chat", json=payload_chat, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.debug(f"Chat API test failed for {model_name} with context {context_size}: {str(e)}")
        
        # If chat fails, try embeddings
        embed_payload = {"model": model_name, "input": "Test"}
        try:
            logger.debug(f"Testing {model_name} with context size {context_size} via embed API")
            resp = requests.post(f"{API_BASE}/api/embed", json=embed_payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.debug(f"Embed API test failed for {model_name} with context {context_size}: {str(e)}")
            return False

def fetch_memory_usage(model_name):
    """
    Fetch memory usage for a given model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (total_size, vram_size) in bytes
    """
    logger = logging.getLogger("ollama_models.utils")
    try:
        logger.debug(f"Fetching memory usage for {model_name}")
        resp = requests.get(f"{API_BASE}/api/ps", timeout=API_TIMEOUT)
        resp.raise_for_status()
        
        models_data = resp.json().get("models", [])
        for m in models_data:
            if m.get("model") == model_name:
                size = m.get("size", 0)
                vram = m.get("size_vram", 0)
                logger.debug(f"Memory usage for {model_name}: total={size}, vram={vram}")
                return size, vram
                
        logger.warning(f"Model {model_name} not found in process list")
        return 0, 0
    except requests.RequestException as e:
        logger.error(f"Error fetching memory usage: {str(e)}")
        return 0, 0

def format_size(num_bytes):
    """
    Format bytes into a human-readable format.
    
    Args:
        num_bytes (int): Number of bytes
        
    Returns:
        str: Human-readable size string
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f}{units[idx]}"

def set_api_base(url):
    """Set the API base URL"""
    global API_BASE
    API_BASE = url
