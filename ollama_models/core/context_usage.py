"""
Context usage analysis for Ollama models.
"""
import os
import csv
import logging
import math
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size
)
from ollama_models.config import API_TIMEOUT, DEFAULT_MAX_CONTEXT_CSV

logger = logging.getLogger("ollama_models.core.context_usage")

def is_power_of_two(n):
    """
    Check if a number is a power of 2.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if the number is a power of 2, False otherwise
    """
    return n > 0 and (n & (n - 1)) == 0

def save_progress(output_file, usage_rows):
    """
    Save the current progress to the output file.
    
    Args:
        output_file (str): Path to output CSV file
        usage_rows (list): List of usage data rows
    """
    try:
        # Sort and write usage file
        sorted_rows = sorted(usage_rows, key=lambda row: row[0])
        with open(output_file, "w", newline="") as usage_file:
            usage_writer = csv.writer(usage_file)
            usage_writer.writerow([
                "model_name", "context_size", "memory_allocated",
                "tokens_per_second", "decode_tokens_per_second", "total_duration", "total_duration_human"
            ])
            for row in sorted_rows:
                usage_writer.writerow(row)
        logger.debug(f"Saved progress to {output_file} with {len(usage_rows)} entries")
    except Exception as e:
        logger.error(f"Error saving progress: {str(e)}")

def generate_usage_report(output_file, model_name=None):
    """
    Generate a context usage report for Ollama models.
    
    Args:
        output_file (str): Path to output CSV file
        model_name (str, optional): Specific model to process
        
    Returns:
        list: List of usage data rows
    """
    usage_set = set()
    usage_rows = []

    # Read existing usage data (skip header)
    if os.path.isfile(output_file):
        with open(output_file, "r", newline="") as uf:
            r = csv.reader(uf)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    usage_rows.append(row)
                if len(row) >= 2:
                    usage_set.add((row[0], int(row[1])))
        logger.info(f"Found existing usage data with {len(usage_rows)} entries")

    # Get models to process
    if model_name:
        models = [{"name": model_name}]
    else:
        models = fetch_installed_models()

    for m in models:
        name = m.get("name")
        max_ctx = fetch_max_context_size(name)
        logger.info(f"Processing model: {name}")
        logger.info(f"Maximum reported context size: {max_ctx}")
        
        # Process standard power-of-2 sizes
        ctx = 2048
        while ctx <= max_ctx:
            if (name, ctx) in usage_set:
                logger.info(f"Skipping model {name} at context = {ctx}: already tested.")
                ctx *= 2
                continue
                
            measure_usage(output_file, usage_set, usage_rows, name, ctx)
            ctx *= 2

        if not is_power_of_two(max_ctx):
            if (name, max_ctx) in usage_set:
                logger.info(f"Skipping model {name} at context = {max_ctx}: already tested.")
            else:
                measure_usage(output_file, usage_set, usage_rows, name, max_ctx)

    # Final save and sort of usage file
    save_progress(output_file, usage_rows)
            
    return usage_rows

def measure_usage(output_file, usage_set, usage_rows, name, ctx):
    result = try_model_call(name, ctx)
    if result['success']:
        size, size_vram = fetch_memory_usage(name)
        size_hr = format_size(size)
        size_vram_hr = format_size(size_vram)
        logger.info(f"Measured at context = {ctx}, total allocated: {size_hr}, VRAM: {size_vram_hr}")
        usage_rows.append([
            name, ctx, size_hr,
            result.get('tokens_per_second'),
            result.get('decode_tokens_per_second'),
            result.get('total_duration'),
            result.get('total_duration_human')
        ])
        usage_set.add((name, ctx))
        # Save progress after each successful test
        save_progress(output_file, usage_rows)
    else:
        logger.info(f"Failed chat/embed call for {name} at context size {ctx}")
