"""
Context probe analysis for finding maximum context sizes for Ollama models.
"""
import os
import csv
import logging
import pathlib
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from ollama_models.utils import (
    fetch_installed_models, fetch_max_context_size,
    try_model_call, fetch_memory_usage, format_size,
    fetch_ollama_version
)
from ollama_models.config import API_TIMEOUT

logger = logging.getLogger("ollama_models.core.context_probe")

class SearchAlgorithm(Enum):
    """Available search algorithms for context probing."""
    PURE_BINARY_MAX_FIRST_G01 = "pure_binary_max_first_g01"

@dataclass
class SearchMetrics:
    """Metrics collected during the search process."""
    algorithm: SearchAlgorithm
    total_tries: int
    total_time: float
    coarse_tries: int = 0
    fine_tries: int = 0
    precision_confidence: Optional[float] = None  # Higher is better (100% = perfect confidence)
    
@dataclass
class ProbeResult:
    """Result from a context probe operation."""
    max_context: int
    model_metrics: Optional[Dict[str, Any]]
    search_metrics: SearchMetrics
    tries: List[Tuple[int, bool, int, int]]  # (context_size, fits, mem_used, vram_used)

def fits_in_vram(model_name, context_size, max_vram=0, isLoad=True):
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
    
    if max_vram <= 0:
        return (size_vram >= size), result
    else:
        return (size_vram >= size and size_vram <= max_vram), result

def find_max_fit_in_vram(model_name: str, max_ctx: int, algorithm: SearchAlgorithm, max_vram=0) -> ProbeResult:
    """
    Find maximum context size that fits in VRAM using specified search algorithm.
    
    Args:
        model_name: Name of the model
        max_ctx: Maximum context size reported by the model
        algorithm: Search algorithm to use
        granularity: Search precision. If None, will be calculated dynamically (adaptive only)
        
    Returns:
        ProbeResult containing max context, metrics, and search details
    """
    start_time = time.time()
    
    
    if algorithm == SearchAlgorithm.PURE_BINARY_MAX_FIRST_G01:
        result = _pure_binary_search_max_first(model_name, max_ctx, 1, SearchAlgorithm.PURE_BINARY_MAX_FIRST_G01, max_vram=max_vram)
    else:
        raise ValueError(f"Unknown search algorithm: {algorithm}")
    
    # Update timing
    result.search_metrics.total_time = time.time() - start_time
    
    _log_search_results(model_name, result)
    return result

def _pure_binary_search_max_first(model_name: str, max_ctx: int, granularity: int, algorithm: SearchAlgorithm, max_vram=0) -> ProbeResult:
    """
    Pure binary search implementation that checks max context first.
    """
    logger.info(f"Finding max context size for {model_name} using binary search max first (granularity={granularity})...")
    tries = []
    min_ctx = 2048
     
    # Initial bounds testing
     
    fits_high, metrics_high = fits_in_vram(model_name, max_ctx, isLoad=True, max_vram=max_vram)
    mem_high, vram_high = fetch_memory_usage(model_name)
    tries.append((max_ctx, fits_high, mem_high, vram_high))
    if fits_high:        
        search_metrics = SearchMetrics(
            algorithm=algorithm,
            total_tries=1,
            total_time=0.0,
            precision_confidence=100.0  # 100% confidence when exact max context fits
        )
        return ProbeResult(
            max_context=max_ctx,
            model_metrics=metrics_high,
            search_metrics=search_metrics,
            tries=tries
        )
    
    fits_low, metrics_low = fits_in_vram(model_name, min_ctx, isLoad=True, max_vram=max_vram)
    mem_low, vram_low = fetch_memory_usage(model_name)
    tries.append((min_ctx, fits_low, mem_low, vram_low))
    
    if not fits_low:
        search_metrics = SearchMetrics(
            algorithm=algorithm,
            total_tries=2,
            total_time=0.0
        )
        return ProbeResult(
            max_context=0,
            model_metrics=None,
            search_metrics=search_metrics,
            tries=tries
        )
       
    # Pure binary search
    low, high = min_ctx, max_ctx
    best_metrics = metrics_low
    
    while high - low > granularity:
        mid = (low + high) // 2
        
        logger.info(f"Binary search max first at {mid} (low={low}, high={high}, gap={high-low})...")
        fits_mid, metrics_mid = fits_in_vram(model_name, mid, isLoad=True, max_vram=max_vram)
        mem_mid, vram_mid = fetch_memory_usage(model_name)
        tries.append((mid, fits_mid, mem_mid, vram_mid))
        
        if fits_mid:
            low = mid
            best_metrics = metrics_mid
        else:
            high = mid    # Calculate precision metrics
    error_percentage = granularity / low * 100 if low > 0 else 0
    confidence = 100.0 - error_percentage  # Higher is better
    
    search_metrics = SearchMetrics(
        algorithm=algorithm,
        total_tries=len(tries),
        total_time=0.0,
        precision_confidence=confidence  # Higher is better (100% = perfect)
    )
    
    return ProbeResult(
        max_context=low,
        model_metrics=best_metrics,
        search_metrics=search_metrics,
        tries=tries
    )

def _log_search_results(model_name: str, result: ProbeResult) -> None:
    """Log detailed search results and performance metrics."""
    metrics = result.search_metrics
    
    logger.info(f"=== Search Results for {model_name} ===")
    logger.info(f"Algorithm: {metrics.algorithm.value}")
    logger.info(f"Max context size in VRAM: {result.max_context}")
    logger.info(f"Total search time: {metrics.total_time:.2f} seconds")
    logger.info(f"Total tries: {metrics.total_tries}")
    
    if metrics.precision_confidence:
        logger.info(f"Search confidence: {metrics.precision_confidence:.2f}%")
    
    logger.info(f"Tried contexts: {[t[0] for t in result.tries]}")
    logger.info("=== End Search Results ===")

def probe_max_context(output_file: str, algorithm: SearchAlgorithm, model_name: Optional[str] = None, max_vram=0) -> List[List[str]]:
    """
    Find and save the maximum context size that fits in VRAM for models.
    The function now saves each model's context fit as soon as it's found,
    making the process resumable if interrupted.
    
    Args:
        output_file: Path to output CSV file
        model_name: Specific model to process
        algorithm: Search algorithm to use
        
    Returns:
        List of probe data rows
    """
    # Get the Ollama version and format the output file
    ollama_version = fetch_ollama_version()
    
    # Process the output file path to include the Ollama version
    path_obj = pathlib.Path(output_file)
    version_output_file = str(path_obj.with_stem(f"{path_obj.stem}_{ollama_version}"))
    
    logger.info(f"Using Ollama version {ollama_version} for context probe")
    logger.info(f"Output will be saved to {version_output_file}")
    logger.info(f"Using search algorithm: {algorithm.value}")
    
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
        models = fetch_installed_models()
        
    # Function to write current fit data to file
    def write_fit_data():
        sorted_rows = sorted(fit_rows, key=lambda row: row[0])        
        with open(version_output_file, 'w', newline="") as fit_file:
            fit_writer = csv.writer(fit_file)
            fit_writer.writerow([
                "model_name", "max_context_size", "is_model_max",
                "memory_allocated", "input_tokens_per_second", "output_tokens_per_second", 
                "total_duration", "total_duration_human", "search_algorithm",
                "search_time", "total_tries", "precision_confidence"
            ])
            for row in sorted_rows:
                fit_writer.writerow(row)

    for m in models:
        name = m.get("name")
        if not name:  # Skip if name is None
            logger.warning(f"Skipping model with no name: {m}")
            continue
            
        if name in fit_models and not model_name:
            logger.info(f"Skipping {name}: already has a max_context entry.")
            continue
            
        logger.info(f"Processing model: {name}")
        start_time = time.time()
        max_ctx = fetch_max_context_size(name)
        logger.info(f"Maximum reported context size: {max_ctx}")
        
        # Use the new algorithm-based search
        result = find_max_fit_in_vram(name, max_ctx, algorithm, max_vram=max_vram)
        
        elapsed = time.time() - start_time
        elapsed_human = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        logger.info(f"Time taken to probe {name}: {elapsed:.2f} seconds ({elapsed_human})")
        
        if result.max_context >= 2048:
            # Get fresh metrics with a chat request for the final result
            fresh_metrics = try_model_call(name, result.max_context, isLoad=False)
            is_model_max = (result.max_context == max_ctx)
            logger.info(f"Max context size fully in VRAM for {name} is {result.max_context}")
            
            # Fetch memory usage for the best-fit context size
            size, _ = fetch_memory_usage(name)
            size_hr = format_size(size)
            
            # Update or add row for this model
            existing_row_index = -1
            for i, row in enumerate(fit_rows):
                if row[0] == name:
                    existing_row_index = i
                    break
                    
            # Use fresh metrics for performance data, but result.model_metrics as fallback
            metrics_to_use = fresh_metrics if fresh_metrics and fresh_metrics.get('success') else result.model_metrics
            
            row_data = [
                name, 
                result.max_context, 
                is_model_max,
                size_hr,
                metrics_to_use.get('input_tokens_per_second') if metrics_to_use else None,
                metrics_to_use.get('output_tokens_per_second') if metrics_to_use else None,
                metrics_to_use.get('total_duration') if metrics_to_use else None,
                metrics_to_use.get('total_duration_human') if metrics_to_use else None,
                result.search_metrics.algorithm.value,                f"{result.search_metrics.total_time:.2f}",
                result.search_metrics.total_tries,
                f"{result.search_metrics.precision_confidence:.2f}%"
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
