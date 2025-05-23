"""
Context probe analysis for finding maximum context sizes for Ollama models.
"""
import os
import csv
import logging
import pathlib
import time
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size,
    fetch_ollama_version
)
from ollama_models.config import API_TIMEOUT

logger = logging.getLogger("ollama_models.core.context_probe")

def fits_in_vram(model_name, context_size, isLoad=True):
    """
    Check if a model fits in VRAM at a given context size and collect metrics.
    
    Args:
        model_name (str): Name of the model
        context_size (int): Size of the context window
        
    Returns:
        tuple: (fits: bool, metrics: dict)
    """
    result = try_model_call(model_name, context_size, isLoad=isLoad)
    if not result['success']:
        logger.info(f"Failed model call for {model_name} at context size {context_size}.")
        return False, result
    size, size_vram = fetch_memory_usage(model_name)
    size_hr = format_size(size)
    size_vram_hr = format_size(size_vram)
    logger.info(f"Memory usage for {model_name} at {context_size}: total={size_hr}, VRAM={size_vram_hr}")
    return (size_vram >= size), result

def find_max_fit_in_vram(model_name, max_ctx, granularity=32):
    """
    Find maximum context size that fits in VRAM for a model using a prediction-driven (secant) search.
    Optimized: If min_ctx == max_ctx, probe only once.
    """
    logger.info(f"Finding max context size fitting in VRAM for {model_name} (prediction-driven secant search)...")
    tries = []  # (context_size, fits, mem_used, vram_used)
    min_ctx = 2048
    # Special case: min_ctx == max_ctx
    if min_ctx == max_ctx:
        fits, metrics = fits_in_vram(model_name, min_ctx, isLoad=True)
        mem, vram = fetch_memory_usage(model_name)
        tries.append((min_ctx, fits, mem, vram))
        return _log_and_return_max_fit(min_ctx if fits else 0, metrics if fits else None, tries)
    # Step 1: Find initial bracket
    fits_low, metrics_low = fits_in_vram(model_name, min_ctx, isLoad=True)
    mem_low, vram_low = fetch_memory_usage(model_name)
    tries.append((min_ctx, fits_low, mem_low, vram_low))
    if not fits_low:
        return _log_and_return_max_fit(0, None, tries)
    fits_high, metrics_high = fits_in_vram(model_name, max_ctx, isLoad=True)
    mem_high, vram_high = fetch_memory_usage(model_name)
    tries.append((max_ctx, fits_high, mem_high, vram_high))
    if fits_high:
        return _log_and_return_max_fit(max_ctx, metrics_high, tries)
    # Bracket: low (fits), high (fails)
    low, high = min_ctx, max_ctx
    mem_low_val, mem_high_val = vram_low, vram_high
    best_metrics = metrics_low
    vram_capacity = vram_high  # Use the VRAM at fail as the capacity estimate
    while high - low > granularity:
        # Fit linear model: mem(c) ≈ a·c + b
        if high == low:
            c_guess = low
        else:
            a = (mem_high_val - mem_low_val) / (high - low)
            if a == 0:
                c_guess = (low + high) // 2
            else:
                b = mem_low_val - a * low
                c_guess = int((vram_capacity - b) / a)
                # Clamp strictly inside (low, high)
                if c_guess <= low or c_guess >= high:
                    c_guess = (low + high) // 2
        logger.info(f"Probing at context size {c_guess} (low={low}, high={high})...")
        fits_guess, metrics_guess = fits_in_vram(model_name, c_guess, isLoad=True)
        mem_guess, vram_guess = fetch_memory_usage(model_name)
        tries.append((c_guess, fits_guess, mem_guess, vram_guess))
        if fits_guess:
            low = c_guess
            mem_low_val = vram_guess
            best_metrics = metrics_guess
        else:
            high = c_guess
            mem_high_val = vram_guess
    return _log_and_return_max_fit(low, best_metrics, tries)

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
        start_time = time.time()
        max_ctx = fetch_max_context_size(name)
        logger.info(f"Maximum reported context size: {max_ctx}")
        best_fit, best_metrics = find_max_fit_in_vram(name, max_ctx)
        elapsed = time.time() - start_time
        elapsed_human = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        logger.info(f"Time taken to probe {name}: {elapsed:.2f} seconds ({elapsed_human})")
        if best_fit >= 2048:
            best_metrics = try_model_call(model_name, best_fit, isLoad=False) # get metrics by using a chat request
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

def _log_and_return_max_fit(max_ctx, last_fit_metrics, tries):
    logger.info(f"The model successfully fits the maximum reported context size in VRAM: {max_ctx}.")
    logger.info(f"Tried: {tries}")
    logger.info(f"Total tries: {len(tries)}")
    return max_ctx, last_fit_metrics
