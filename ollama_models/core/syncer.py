"""
Sync models to an Ollama instance.
"""
import os
import requests
import logging

logger = logging.getLogger("ollama_models.core.syncer")

def load_config(config_file):
    """
    Load selected models from config file.
    
    Args:
        config_file (str): Path to config file
        
    Returns:
        set: Set of selected models
    """
    selected = set()
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    selected.add(line)
    return selected

def sync_ollama(config_file, api_base):
    """
    Sync models to an Ollama instance.
    
    Args:
        config_file (str): Path to config file
        api_base (str): Base URL for the Ollama API
        
    Returns:
        tuple: (success, new_models, removed_models)
    """
    logger.info("Loading selected models from config...")
    selected = load_config(config_file)
    logger.info(f"Found {len(selected)} models in config.")

    logger.info("Fetching currently installed models from Ollama...")
    try:
        resp = requests.get(f"{api_base}/api/tags", timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"Error fetching Ollama tags: {e}")
        return False, [], []

    installed_models = set(m["name"] for m in data.get("models", []))
    logger.info(f"Found {len(installed_models)} installed models.")
    
    logger.info("Determining new and removed models...")
    new_models = selected - installed_models
    removed_models = installed_models - selected

    success = True
    
    # Pull new models
    for model in sorted(selected): # using the full list of selected models to ensure we pull updates
        logger.info(f"Pulling model: {model}")
        try:
            resp = requests.post(f"{api_base}/api/pull", json={"model": model}, timeout=10800)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Error pulling {model}: {e}")
            success = False

    # Delete models to be removed
    for model in sorted(removed_models):
        logger.info(f"Deleting model: {model}")
        try:
            resp = requests.delete(f"{api_base}/api/delete", json={"model": model}, timeout=3600)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Error deleting {model}: {e}")
            success = False

    logger.info(f"New models: {sorted(new_models)}")
    logger.info(f"Deleted models: {sorted(removed_models)}")

    return success, new_models, removed_models
