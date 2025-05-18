"""
Context usage analysis for Ollama models.
"""
import os
import csv
import logging
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size
)
from ollama_models.config import API_TIMEOUT

logger = logging.getLogger("ollama_models.core.context_usage")

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
        
        ctx = 2048
        while ctx <= max_ctx:
            if (name, ctx) in usage_set:
                logger.info(f"Skipping model {name} at 2^n={ctx}: already tested.")
                ctx *= 2
                continue
                
            success = try_model_call(name, ctx)
            if success:
                size, size_vram = fetch_memory_usage(name)
                size_hr = format_size(size)
                size_vram_hr = format_size(size_vram)
                logger.info(f"Measured at 2^n = {ctx}, total allocated: {size_hr}, VRAM: {size_vram_hr}")
                usage_rows.append([name, ctx, size_hr])
                usage_set.add((name, ctx))
            else:
                logger.info(f"Failed chat/embed call for {name} at 2^n size {ctx}")
            ctx *= 2

    # Sort and write usage file
    usage_rows.sort(key=lambda row: row[0])
    with open(output_file, "w", newline="") as usage_file:
        usage_writer = csv.writer(usage_file)
        usage_writer.writerow(["model_name", "context_size", "memory_allocated"])
        for row in usage_rows:
            usage_writer.writerow(row)
            
    return usage_rows
