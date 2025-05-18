"""
Initialize model configuration from Ollama API.
"""
import os
import requests
import logging

logger = logging.getLogger("ollama_models.core.initializer")

def init_from_api(config_file, api_base):
    """
    Initialize model configuration from Ollama API.
    
    Args:
        config_file (str): Path to config file
        api_base (str): Base URL for the Ollama API
        
    Returns:
        tuple: (success, models)
    """
    try:
        # Fetch tags from Ollama API
        logger.info(f"Fetching models from Ollama API: {api_base}")
        response = requests.get(f"{api_base}/api/tags")
        response.raise_for_status()
        data = response.json()
        model_data = data.get("models", [])
        
        models = []
        for item in model_data:
            if isinstance(item, dict) and item.get("model"):
                models.append(item["model"])
        
        # Update selected_tags.conf with retrieved models
        with open(config_file, "w") as f:
            for model in sorted(set(models)):
                f.write(f"{model}\n")
                
        logger.info(f"Successfully wrote {len(models)} models to {config_file}")
        return True, models
    except Exception as e:
        logger.error(f"Error initializing from API: {e}")
        return False, []
