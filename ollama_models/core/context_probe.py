"""
Context probe analysis for finding maximum context sizes for Ollama models.
"""
import os
import csv
import logging
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size
)
from ollama_models.config import API_TIMEOUT

logger = logging.getLogger("ollama_models.core.context_probe")

def fits_in_vram(model_name, context_size):
    """
    Check if a model fits in VRAM at a given context size.
    
    Args:
        model_name (str): Name of the model
        context_size (int): Size of the context window
        
    Returns:
        bool: True if the model fits, False otherwise
    """
    if not try_model_call(model_name, context_size):
        logger.info(f"Failed model call for {model_name} at context size {context_size}.")
        return False
    size, size_vram = fetch_memory_usage(model_name)
    size_hr = format_size(size)
    size_vram_hr = format_size(size_vram)
    logger.info(f"Memory usage for {model_name} at {context_size}: total={size_hr}, VRAM={size_vram_hr}")
    return (size_vram >= size)

def find_max_fit_in_vram(model_name, max_ctx):
    """
    Find maximum context size that fits in VRAM for a model.
    Reports progress during the search for better monitoring.
    
    Args:
        model_name (str): Name of the model
        max_ctx (int): Maximum context size to test
        
    Returns:
        int: Maximum context size that fits in VRAM
    """
    logger.info(f"Finding max context size fitting in VRAM for {model_name}...")
    best_fit = 0
    start = 2048
    if not fits_in_vram(model_name, start):
        logger.info(f"{model_name} cannot fit in VRAM even at 2048.")
        return 0
    
    logger.info(f"Initial test successful at context size {start}.")
    high = start
    
    # Exponential search phase
    while high <= max_ctx and fits_in_vram(model_name, high):
        best_fit = high  # Update best_fit during exponential search
        logger.info(f"Model {model_name} fits at context size {high}, trying larger size...")
        high *= 2
        
    # Binary search phase
    logger.info(f"Starting binary search between {high // 2} and {min(high, max_ctx)}...")
    left, right = high // 2, min(high, max_ctx)
    while left <= right:
        mid = (left + right) // 2
        logger.info(f"Testing context size {mid}...")
        if fits_in_vram(model_name, mid):
            best_fit = mid
            logger.info(f"Success at {mid}, trying larger size...")
            left = mid + 1
        else:
            logger.info(f"Failed at {mid}, trying smaller size...")
            right = mid - 1
            
    logger.info(f"Highest context size fitting in VRAM for {model_name}: {best_fit}")
    return best_fit

def probe_max_context(output_file, model_name=None):
    """
    Find and save the maximum context size that fits in VRAM for models.
    The function now saves each model's context fit as soon as it's found,
    making the process resumable if interrupted.
    
    Args:
        output_file (str): Path to output CSV file
        model_name (str, optional): Specific model to process
        
    Returns:
        list: List of probe data rows
    """
    fit_models = set()
    fit_rows = []

    # Read existing fit data (skip header)
    if os.path.isfile(output_file):
        with open(output_file, "r", newline="") as ff:
            r = csv.reader(ff)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    fit_rows.append(row)
                if len(row) >= 1:
                    fit_models.add(row[0])

    # Get models to process
    if model_name:
        models = [{"name": model_name}]
    else:
        models = fetch_installed_models()

    # Function to write current fit data to file
    def write_fit_data():
        sorted_rows = sorted(fit_rows, key=lambda row: row[0])
        with open(output_file, 'w', newline="") as fit_file:
            fit_writer = csv.writer(fit_file)
            fit_writer.writerow(["model_name", "max_context_size", "is_model_max"])
            for row in sorted_rows:
                fit_writer.writerow(row)

    for m in models:
        name = m.get("name")
        if name in fit_models and not model_name:
            logger.info(f"Skipping {name}: already has a max_context entry.")
            continue
            
        logger.info(f"Processing model: {name}")
        max_ctx = fetch_max_context_size(name)
        logger.info(f"Maximum reported context size: {max_ctx}")
        
        best_fit = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            is_model_max = (best_fit == max_ctx)
            logger.info(f"Max context size fully in VRAM for {name} is {best_fit}")
            
            # Update or add row for this model
            existing_row_index = -1
            for i, row in enumerate(fit_rows):
                if row[0] == name:
                    existing_row_index = i
                    break
                    
            if existing_row_index >= 0:
                fit_rows[existing_row_index] = [name, best_fit, is_model_max]
            else:
                fit_rows.append([name, best_fit, is_model_max])
                fit_models.add(name)
                
            # Save progress immediately after processing each model
            logger.info(f"Saving progress for {name}...")
            write_fit_data()
            logger.info(f"Progress saved.")

    return fit_rows
