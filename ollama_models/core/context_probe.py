"""
Context probe analysis for finding maximum context sizes for Ollama models.
"""
import os
import csv
import logging
import pathlib
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size,
    fetch_ollama_version
)
from ollama_models.config import API_TIMEOUT

logger = logging.getLogger("ollama_models.core.context_probe")

def fits_in_vram(model_name, context_size):
    """
    Check if a model fits in VRAM at a given context size and collect metrics.
    
    Args:
        model_name (str): Name of the model
        context_size (int): Size of the context window
        
    Returns:
        tuple: (fits: bool, metrics: dict)
    """
    result = try_model_call(model_name, context_size)
    if not result['success']:
        logger.info(f"Failed model call for {model_name} at context size {context_size}.")
        return False, result
    size, size_vram = fetch_memory_usage(model_name)
    size_hr = format_size(size)
    size_vram_hr = format_size(size_vram)
    logger.info(f"Memory usage for {model_name} at {context_size}: total={size_hr}, VRAM={size_vram_hr}")
    return (size_vram >= size), result

def find_max_fit_in_vram(model_name, max_ctx):
    """
    Find maximum context size that fits in VRAM for a model, using pure binary search.
    Logs the number of tries.
    
    Args:
        model_name (str): Name of the model
        max_ctx (int): Maximum context size to test
    
    Returns:
        int: Maximum context size that fits in VRAM
    """
    logger.info(f"Finding max context size fitting in VRAM for {model_name} (pure binary search)...")
    tries = []  # (context_size, fits, mem_used, vram_used)
    min_ctx = 2048
    low = min_ctx
    high = max_ctx
    best_fit = 0
    best_metrics = None
    # First, check if the minimum fits
    fits, metrics = fits_in_vram(model_name, min_ctx)
    mem_used, vram_used = fetch_memory_usage(model_name)
    tries.append((min_ctx, fits, mem_used, vram_used))
    if not fits:
        logger.info(f"{model_name} cannot fit in VRAM even at {min_ctx}.")
        logger.info(f"Tried: {tries}")
        logger.info(f"Total tries: {len(tries)}")
        return 0, None
    best_fit = min_ctx
    best_metrics = metrics
    # Binary search
    while low < high:
        mid = (low + high + 1) // 2  # bias upward to find the highest fit
        logger.info(f"Binary search test at context size {mid}...")
        fits, metrics = fits_in_vram(model_name, mid)
        mem_used, vram_used = fetch_memory_usage(model_name)
        tries.append((mid, fits, mem_used, vram_used))
        if fits:
            best_fit = mid
            best_metrics = metrics
            low = mid
        else:
            high = mid - 1
    logger.info(f"Highest context size fitting in VRAM for {model_name}: {best_fit}")
    logger.info(f"Probe tries: {[(c, f) for c, f, _, _ in sorted(tries)]}")
    logger.info(f"Total tries: {len(tries)}")
    return best_fit, best_metrics

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
    # Get the Ollama version and format the output file
    ollama_version = fetch_ollama_version()
    
    # Process the output file path to include the Ollama version
    path_obj = pathlib.Path(output_file)
    version_output_file = str(path_obj.with_stem(f"{path_obj.stem}_{ollama_version}"))
    
    logger.info(f"Using Ollama version {ollama_version} for context probe")
    logger.info(f"Output will be saved to {version_output_file}")
    
    fit_models = set()
    fit_rows = []

    # Read existing fit data (skip header)
    if os.path.isfile(version_output_file):
        with open(version_output_file, "r", newline="") as ff:
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
        models = fetch_installed_models()    # Function to write current fit data to file
    def write_fit_data():
        sorted_rows = sorted(fit_rows, key=lambda row: row[0])
        with open(version_output_file, 'w', newline="") as fit_file:
            fit_writer = csv.writer(fit_file)
            fit_writer.writerow([
                "model_name", "max_context_size", "is_model_max",
                "memory_allocated", "tokens_per_second", "decode_tokens_per_second", "total_duration", "total_duration_human"
            ])
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
        
        best_fit, best_metrics = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            is_model_max = (best_fit == max_ctx)
            logger.info(f"Max context size fully in VRAM for {name} is {best_fit}")
            
            # Fetch memory usage for the best-fit context size
            size, _ = fetch_memory_usage(name)
            size_hr = format_size(size)
            
            # Update or add row for this model
            existing_row_index = -1
            for i, row in enumerate(fit_rows):
                if row[0] == name:
                    existing_row_index = i
                    break
                    
            row_data = [
                name, best_fit, is_model_max,
                size_hr,
                best_metrics.get('tokens_per_second') if best_metrics else None,
                best_metrics.get('decode_tokens_per_second') if best_metrics else None,
                best_metrics.get('total_duration') if best_metrics else None,
                best_metrics.get('total_duration_human') if best_metrics else None
            ]
            if existing_row_index >= 0:
                fit_rows[existing_row_index] = row_data
            else:
                fit_rows.append(row_data)
                fit_models.add(name)
                
            # Save progress immediately after processing each model
            logger.info(f"Saving progress for {name}...")
            write_fit_data()
            logger.info(f"Progress saved.")

    return fit_rows
