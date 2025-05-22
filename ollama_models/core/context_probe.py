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
    Find maximum context size that fits in VRAM for a model, using bidirectional exponential search to bracket, then pure binary search to converge.
    This approach adapts the starting point and direction for optimal bracketing.
    """
    logger.info(f"Finding max context size fitting in VRAM for {model_name} (bidirectional exponential + binary search)...")
    tries = []  # (context_size, fits, mem_used, vram_used)
    min_ctx = 2048
    # Start in the middle of the range
    start = max(min_ctx, max_ctx // 2)
    last_fit = None
    last_fail = None
    last_fit_metrics = None
    last_fail_metrics = None
    fits, metrics = fits_in_vram(model_name, start)
    mem_used, vram_used = fetch_memory_usage(model_name)
    tries.append((start, fits, mem_used, vram_used))
    # Optimization: If start == max_ctx and it fits, return immediately
    if fits and start == max_ctx:
        last_fit = (start, mem_used, vram_used)
        last_fit_metrics = metrics
        return _log_and_return_max_fit(max_ctx, last_fit_metrics, tries)
    if fits:
        # Exponential search upward
        last_fit = (start, mem_used, vram_used)
        last_fit_metrics = metrics
        ctx = min(start * 2, max_ctx)
        while ctx <= max_ctx:
            if ctx == start:
                break  # Prevent redundant try
            fits, metrics = fits_in_vram(model_name, ctx)
            mem_used, vram_used = fetch_memory_usage(model_name)
            tries.append((ctx, fits, mem_used, vram_used))
            if fits:
                last_fit = (ctx, mem_used, vram_used)
                last_fit_metrics = metrics
                if ctx == max_ctx:
                    break
                ctx = min(ctx * 2, max_ctx)
            else:
                last_fail = (ctx, mem_used, vram_used)
                last_fail_metrics = metrics
                break
        else:
            last_fail = None
    else:
        # Exponential search downward
        last_fail = (start, mem_used, vram_used)
        last_fail_metrics = metrics
        ctx = max(start // 2, min_ctx)
        while ctx >= min_ctx:
            fits, metrics = fits_in_vram(model_name, ctx)
            mem_used, vram_used = fetch_memory_usage(model_name)
            tries.append((ctx, fits, mem_used, vram_used))
            if fits:
                last_fit = (ctx, mem_used, vram_used)
                last_fit_metrics = metrics
                break
            elif ctx == min_ctx:
                last_fit = None
                last_fit_metrics = None
                break
            ctx = max(ctx // 2, min_ctx)
    if last_fit is None or last_fit_metrics is None:
        logger.info(f"{model_name} cannot fit in VRAM even at {min_ctx}.")
        logger.info(f"Tried: {tries}")
        logger.info(f"Total tries: {len(tries)}")
        return 0, None
    if last_fail is None or last_fail_metrics is None:
        # Never failed, so max_ctx is the answer
        return _log_and_return_max_fit(max_ctx, last_fit_metrics, tries)
    # Step 2: Pure binary search between last fit and first fail
    low_ctx = last_fit[0]
    high_ctx = last_fail[0]
    best_fit = low_ctx
    best_metrics = last_fit_metrics
    while high_ctx - low_ctx > 1:
        mid_ctx = (low_ctx + high_ctx) // 2
        logger.info(f"Binary search test at context size {mid_ctx}...")
        fits, metrics = fits_in_vram(model_name, mid_ctx)
        mem_used, vram_used = fetch_memory_usage(model_name)
        tries.append((mid_ctx, fits, mem_used, vram_used))
        if fits:
            low_ctx = mid_ctx
            best_fit = mid_ctx
            best_metrics = metrics
        else:
            high_ctx = mid_ctx
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

def _log_and_return_max_fit(max_ctx, last_fit_metrics, tries):
    logger.info(f"The model successfully fits the maximum reported context size in VRAM: {max_ctx}.")
    logger.info(f"Tried: {tries}")
    logger.info(f"Total tries: {len(tries)}")
    return max_ctx, last_fit_metrics
