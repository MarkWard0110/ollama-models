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

# Force the logger for 'ollama_models.utils' to DEBUG level at the top of the file for troubleshooting
logger = logging.getLogger("ollama_models.utils")
logger.setLevel(logging.INFO)

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

def try_model_call(model_name, context_size, isLoad=False):
    """
    Test if a model can handle a given context size and return metrics.
    
    Args:
        model_name (str): Name of the model
        context_size (int): Context size to test
        
    Returns:
        dict: {
            'success': bool,
            'tokens_per_second': float or None,
            'decode_tokens_per_second': float or None,
            'total_duration': float or None (seconds),
            'total_duration_human': str or None,
            'eval_count': int or None,
            'prompt_eval_count': int or None,
            'eval_duration': float or None (ns),
            'prompt_eval_duration': float or None (ns),
            'load_duration': float or None (ns),
            'raw_response': dict or None
        }
    """
    import time
    logger = logging.getLogger("ollama_models.utils")
    payload_chat = {
        "model": model_name,
        "stream": False,
        "options": {
            "num_ctx": context_size,
            "num_predict": 512,
        }
    }
    if not isLoad:
        payload_chat["messages"] = [
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ]        

    try:
        logger.debug(f"Testing {model_name} with context size {context_size} via chat API")
        start = time.time()
        resp = requests.post(f"{API_BASE}/api/chat", json=payload_chat, timeout=API_TIMEOUT)
        resp.raise_for_status()
        end = time.time()
        data = resp.json()
        # Extract metrics if present
        eval_count = data.get('eval_count')
        prompt_eval_count = data.get('prompt_eval_count')
        eval_duration = data.get('eval_duration')
        prompt_eval_duration = data.get('prompt_eval_duration')
        load_duration = data.get('load_duration')
        total_duration = data.get('total_duration')
        # Fallback to wall time if not present
        if total_duration is None:
            total_duration = int((end - start) * 1e9)  # ns
        # Compute metrics
        tokens_per_second = None
        decode_tokens_per_second = None
        if total_duration and prompt_eval_count is not None and eval_count is not None:
            tokens_per_second = (prompt_eval_count + eval_count) / (total_duration / 1e9)
        if eval_duration and eval_count is not None:
            decode_tokens_per_second = eval_count / (eval_duration / 1e9)
        # Human-friendly duration
        def human_time(ns):
            s = ns / 1e9
            if s < 1:
                return f"{int(ns/1e6)} ms"
            elif s < 60:
                return f"{s:.2f} s"
            else:
                m = int(s // 60)
                sec = s % 60
                return f"{m}m {sec:.1f}s"
        total_duration_human = human_time(total_duration) if total_duration else None
        return {
            'success': True,
            'tokens_per_second': tokens_per_second,
            'decode_tokens_per_second': decode_tokens_per_second,
            'total_duration': total_duration / 1e9 if total_duration else None,
            'total_duration_human': total_duration_human,
            'eval_count': eval_count,
            'prompt_eval_count': prompt_eval_count,
            'eval_duration': eval_duration,
            'prompt_eval_duration': prompt_eval_duration,
            'load_duration': load_duration,
            'raw_response': data
        }
    except requests.RequestException as e:
        logger.debug(f"Chat API test failed for {model_name} with context {context_size}: {str(e)}")
        # If chat fails, try embeddings
        embed_payload = {
            "model": model_name
            }
        if not isLoad:
            embed_payload["prompt"] = "test"

        try:
            logger.debug(f"Testing {model_name} with context size {context_size} via embed API")
            resp = requests.post(f"{API_BASE}/api/embed", json=embed_payload, timeout=API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return {
                'success': True,
                'tokens_per_second': None,
                'decode_tokens_per_second': None,
                'total_duration': None,
                'total_duration_human': None,
                'eval_count': None,
                'prompt_eval_count': None,
                'eval_duration': None,
                'prompt_eval_duration': None,
                'load_duration': None,
                'raw_response': data
            }
        except requests.RequestException as e:
            logger.debug(f"Embed API test failed for {model_name} with context {context_size}: {str(e)}")
            return {
                'success': False,
                'tokens_per_second': None,
                'decode_tokens_per_second': None,
                'total_duration': None,
                'total_duration_human': None,
                'eval_count': None,
                'prompt_eval_count': None,
                'eval_duration': None,
                'prompt_eval_duration': None,
                'load_duration': None,
                'raw_response': None
            }

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

def fetch_ollama_version():
    """
    Fetch the Ollama version from the API.
    
    Returns:
        str: The Ollama version string or "unknown" if it cannot be determined
    """
    logger = logging.getLogger("ollama_models.utils")
    try:
        logger.debug("Fetching Ollama version")
        resp = requests.get(f"{API_BASE}/api/version", timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        version = data.get("version", "unknown")
        logger.info(f"Detected Ollama version: {version}")
        return version
    except requests.RequestException as e:
        logger.error(f"Error fetching Ollama version: {str(e)}")
        return "unknown"
