#!/usr/bin/env python3
"""
Test script for the refactored context probe module with multiple search algorithms.
Demonstrates algorithm selection, comparison, and consistent logging across multiple models.

Configuration:
- MODEL_FILTER: List of specific models to test (empty = test all models)
- Logging level and format can be adjusted below
"""
import logging
import random
import csv
from typing import Dict, List, Tuple, Optional, Any
from ollama_models.core.context_probe import (
    SearchAlgorithm, 
    find_max_fit_in_vram, 
    probe_max_context,
    ProbeResult
)
from ollama_models.utils import fetch_installed_models, fetch_max_context_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model filter configuration
# Add model names to this list to test only specific models
# Leave empty to test all installed models
# 
# QUICK START EXAMPLES:
# To test only small models:   MODEL_FILTER = ["llama3.2:1b-instruct-fp16", "gemma3:1b-it-fp16"]
# To test only one model:      MODEL_FILTER = ["gemma3:27b-it-q8_0"]
# To test flat memory models:  MODEL_FILTER = ["gemma3:27b-it-q8_0", "gemma3:27b-it-q4_K_M"]
# To test all models:          MODEL_FILTER = []
#
# Run list_available_models() first to see what models you have installed
MODEL_FILTER = [
    # === QUICK TEST MODELS (Small, fast) ===
    # "llama3.2:1b-instruct-fp16",  # ~1GB VRAM
    # "gemma3:1b-it-fp16",          # ~1GB VRAM
    # "qwen2.5-coder:1.5b-instruct-fp16", # ~1.5GB VRAM
    # "qwen3:0.6b-fp16",            # <1GB VRAM
    
    # === MEDIUM MODELS (Moderate testing) ===
    # "llama3.2:3b-instruct-fp16",  # ~2-3GB VRAM
    # "gemma3:4b-it-fp16",          # ~3GB VRAM
    # "qwen3:4b-fp16",              # ~3GB VRAM
    # "phi4-mini:3.8b-fp16",        # ~3GB VRAM
    
    # === LARGE MODELS (Thorough testing) ===
    # "llama3.1:8b-instruct-fp16",  # ~5GB VRAM
    # "qwen3:8b-fp16",              # ~5GB VRAM
    # "phi4:14b-fp16",              # ~8GB VRAM
    # "qwen3:14b-fp16",             # ~8GB VRAM
    
    # === VERY LARGE MODELS (Extensive VRAM required) ===
    # "llama3.3:70b-instruct-q3_K_M", # ~25GB+ VRAM
    # "qwen3:32b-q4_K_M",             # ~16GB+ VRAM
    
    # === PROBLEMATIC MODELS (Known for flat memory curves) ===
    # "gemma3:27b-it-q8_0",           # The model mentioned in the original issue
    # "gemma3:27b-it-q4_K_M",         # Quantized version
    
    # === QUANTIZATION COMPARISON (Same model, different quantization) ===
    # "llama3.1:8b-instruct-fp16",    # Full precision
    # "llama3.1:8b-instruct-q8_0",    # 8-bit quantization
    # "llama3.1:8b-instruct-q4_K_M",  # 4-bit quantization
    # "llama3.1:8b-instruct-q2_K",    # 2-bit quantization
]

# Advanced filter options
# These options provide alternative ways to filter models
# They are applied after the MODEL_FILTER

# Filter by model size category
# Options: "SMALL", "MEDIUM", "LARGE", "XLARGE", None (no size filtering)
FILTER_BY_SIZE = None     

# Filter by parameter count range
# Format: (min_billions, max_billions)
# Examples: (0, 3)  - Models <3B parameters
#           (3, 8)  - Models 3-8B parameters
#           (8, 20) - Models 8-20B parameters
#           (20, None) - Models >20B parameters
FILTER_BY_PARAMS = None  # Set to a tuple (min_billions, max_billions) or None

# Skip models that would require significant VRAM
SKIP_LARGE_MODELS = False  # Set to True to skip LARGE and XLARGE models

# Limit the total number of models to test
# MAX_MODELS_TO_TEST = None  # Set to a number to limit total models tested
MAX_MODELS_TO_TEST = 2

# Maximum runtime (in minutes) - will attempt to estimate and skip models that would exceed this
# Only applies when testing multiple models in sequence
MAX_RUNTIME_MINUTES = None  # Set to a number of minutes to limit total runtime

def list_available_models():
    """List all available models for reference with parameter counts and size estimates."""
    from ollama_models.utils import fetch_parameter_count
    
    models = fetch_installed_models()
    if models:
        logger.info("Available models:")
        logger.info("="*80)
        logger.info(f"{'#':4} {'Model Name':50} {'Parameters':12} {'Size Category':15}")
        logger.info("-"*80)
        
        for i, model in enumerate(models, 1):
            name = model['name']
            
            try:
                # Get parameter count and size category
                param_count, formatted_size = fetch_parameter_count(name)
                size_category = estimate_model_size_category(name)
                
                logger.info(f"{i:4} {name:50} {formatted_size:12} {size_category:15}")
            except Exception as e:
                logger.info(f"{i:4} {name:50} {'Unknown':12} {'Error':15} - {str(e)}")
                
        logger.info("="*80)
        logger.info(f"Total models found: {len(models)}")
        
        # Show size distribution
        size_counts = {}
        param_totals = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0, "XLARGE": 0, "UNKNOWN": 0}
        param_counts = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0, "XLARGE": 0, "UNKNOWN": 0}
        
        for model in models:
            name = model['name']
            category = estimate_model_size_category(name)
            size_counts[category] = size_counts.get(category, 0) + 1
            
            # Accumulate parameter counts for average calculation
            try:
                param_count, _ = fetch_parameter_count(name)
                param_totals[category] += int(param_count)
                param_counts[category] += 1
            except Exception:
                pass
                
        # Calculate average sizes and display stats
        logger.info("\nSize distribution:")
        logger.info(f"{'Category':10} {'Count':7} {'Avg Parameters':15}")
        logger.info("-"*40)
        
        for size, count in sorted(size_counts.items()):
            avg_params = "N/A"
            if param_counts[size] > 0:
                avg = param_totals[size] / param_counts[size]
                if avg >= 1e9:
                    avg_params = f"{avg/1e9:.1f}B"
                else:
                    avg_params = f"{avg/1e6:.1f}M"
                    
            logger.info(f"{size:10} {count:7} {avg_params:15}")
            
    else:
        logger.info("No models found")
    return models

def estimate_model_size_category(model_name):
    """
    Estimate model size category based on parameter count.
    Uses the fetch_parameter_count function to get accurate parameter size.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Size category - "SMALL", "MEDIUM", "LARGE", "XLARGE"
    """
    from ollama_models.utils import fetch_parameter_count
    
    try:
        param_count, formatted_size = fetch_parameter_count(model_name)
        
        # Categorize based on parameter count
        if param_count < 3e9:  # Less than 3B parameters
            return "SMALL"
        elif param_count < 8e9:  # 3B to 8B parameters
            return "MEDIUM"
        elif param_count < 20e9:  # 8B to 20B parameters
            return "LARGE"
        else:  # 20B+ parameters
            return "XLARGE"
            
    except Exception as e:
        logger.warning(f"Error determining parameter count for {model_name}: {e}")
       
def apply_advanced_filters(models):
    """Apply advanced filtering options to model list."""
    filtered = models[:]
      # Apply size-based filtering
    if FILTER_BY_SIZE:
        size_filter = FILTER_BY_SIZE  # Already uppercase in the config
        logger.info(f"Applying size filter: {size_filter}")
        filtered = [m for m in filtered if estimate_model_size_category(m['name']) == size_filter]
        logger.info(f"After size filtering: {len(filtered)} models")
    
    # Apply parameter count-based filtering
    if FILTER_BY_PARAMS is not None:
        min_billions, max_billions = FILTER_BY_PARAMS
        min_params = min_billions * 1e9 if min_billions is not None else 0
        max_params = max_billions * 1e9 if max_billions is not None else float('inf')
        
        logger.info(f"Applying parameter filter: {min_billions}B to {max_billions if max_billions else 'unlimited'}B")
        
        def is_in_param_range(model_name):
            from ollama_models.utils import fetch_parameter_count
            try:
                param_count, _ = fetch_parameter_count(model_name)
                return min_params <= param_count <= max_params
            except Exception as e:
                logger.warning(f"Error checking parameter count for {model_name}: {e}")
                return False
                
        filtered = [m for m in filtered if is_in_param_range(m['name'])]
        logger.info(f"After parameter filtering: {len(filtered)} models")
    
    # Skip large models if requested
    if SKIP_LARGE_MODELS:
        logger.info("Skipping large models (LARGE and XLARGE)")
        filtered = [m for m in filtered if estimate_model_size_category(m['name']) not in ['LARGE', 'XLARGE']]
        logger.info(f"After skipping large models: {len(filtered)} models")
    
    # Apply max models limit
    if MAX_MODELS_TO_TEST and len(filtered) > MAX_MODELS_TO_TEST:
        logger.info(f"Limiting to first {MAX_MODELS_TO_TEST} models")
        random.shuffle(filtered)
        filtered = filtered[:MAX_MODELS_TO_TEST]
    
    # Apply runtime estimation-based filtering
    if MAX_RUNTIME_MINUTES and len(filtered) > 1:
        logger.info(f"Maximum runtime limit set to {MAX_RUNTIME_MINUTES} minutes")
        # We'll sort models by estimated size to test small models first
        filtered.sort(key=lambda m: _get_size_score(estimate_model_size_category(m['name'])))
        
        # Rough estimate: small=2min, medium=5min, large=10min, xlarge=20min per model
        def estimate_model_runtime_minutes(model):
            category = estimate_model_size_category(model['name'])
            if category == 'SMALL': return 2
            if category == 'MEDIUM': return 5
            if category == 'LARGE': return 10
            if category == 'XLARGE': return 20
            return 5  # Default for unknown
        
        # Only keep models within estimated runtime
        total_estimated_runtime = 0
        filtered_by_runtime = []
        
        for model in filtered:
            model_runtime = estimate_model_runtime_minutes(model)
            if total_estimated_runtime + model_runtime <= MAX_RUNTIME_MINUTES:
                filtered_by_runtime.append(model)
                total_estimated_runtime += model_runtime
            else:
                logger.info(f"Skipping {model['name']} as it would exceed runtime limit")
                
        logger.info(f"Estimated total runtime: {total_estimated_runtime} minutes")
        logger.info(f"After runtime filtering: {len(filtered_by_runtime)} models")
        filtered = filtered_by_runtime
    
    return filtered

def _get_size_score(size_category):
    """Helper to get a numeric score for size categories for sorting."""
    if size_category == 'SMALL': return 1
    if size_category == 'MEDIUM': return 2
    if size_category == 'LARGE': return 3
    if size_category == 'XLARGE': return 4
    return 2  # Default for UNKNOWN

def get_filtered_models():
    """Get list of models to test based on all filter configurations."""
    all_models = fetch_installed_models()
    if not all_models:
        logger.error("No models found installed")
        return []
    
    logger.info(f"Found {len(all_models)} total installed models")
    
    # Apply MODEL_FILTER (specific model names)
    if MODEL_FILTER:
        logger.info(f"Applying model name filter: {MODEL_FILTER}")
        available_names = {model["name"] for model in all_models}
        filtered_models = []
        
        for filter_name in MODEL_FILTER:
            if filter_name in available_names:
                # Find the full model info
                for model in all_models:
                    if model["name"] == filter_name:
                        filtered_models.append(model)
                        break
            else:
                logger.warning(f"Model '{filter_name}' in filter list not found in installed models")
        
        if not filtered_models:
            logger.error("No models found matching the filter criteria")
            logger.info(f"Available models: {[m['name'] for m in all_models]}")
            return []
        
        logger.info(f"After name filtering: {len(filtered_models)} models")
        models_to_test = filtered_models
    else:
        logger.info("No model name filter specified, using all installed models")
        models_to_test = all_models
    
    # Apply advanced filters
    models_to_test = apply_advanced_filters(models_to_test)
    
    if not models_to_test:
        logger.error("No models remaining after applying all filters")
        return []
    
    # randomize the order of models to test
    random.shuffle(models_to_test)
    logger.info(f"Randomized model order for testing")

    # Log final selection
    logger.info(f"Final model selection: {len(models_to_test)} models")
    for i, model in enumerate(models_to_test, 1):
        size_cat = estimate_model_size_category(model['name'])
        logger.info(f"  {i:2d}. {model['name']} [{size_cat}]")
    
    return models_to_test

def test_algorithm_selection():
    """Test individual algorithm selection and execution on filtered models."""
    logger.info("=== Testing Algorithm Selection ===")
    
    # Get filtered test models
    models = get_filtered_models()
    if not models:
        return []
    
    results = []
    
    for i, model_info in enumerate(models, 1):
        test_model = model_info["name"]
        max_ctx = fetch_max_context_size(test_model)
        
        logger.info(f"\n[{i}/{len(models)}] Testing model: {test_model}")
        logger.info(f"Max context: {max_ctx}")
        
        try:
            # Test Pure Binary Search
            logger.info(f"--- Testing Pure Binary Search on {test_model} ---")
            result_pure = find_max_fit_in_vram(
                test_model, 
                max_ctx, 
                SearchAlgorithm.PURE_BINARY,
                granularity=64  # Fixed granularity for pure binary
            )
            
            # Test Adaptive Binary Search  
            logger.info(f"--- Testing Adaptive Binary Search on {test_model} ---")
            result_adaptive = find_max_fit_in_vram(
                test_model,
                max_ctx,
                SearchAlgorithm.ADAPTIVE_BINARY
                # granularity will be calculated dynamically
            )
            
            # Store results
            model_results = {
                'model': test_model,
                'max_ctx': max_ctx,
                'pure_binary': result_pure,
                'adaptive_binary': result_adaptive
            }
            results.append(model_results)
            
            # Log individual results
            logger.info(f"--- Results for {test_model} ---")
            logger.info(f"Pure Binary: {result_pure.max_context} tokens in {result_pure.search_metrics.total_time:.2f}s ({result_pure.search_metrics.total_tries} tries)")
            logger.info(f"Adaptive Binary: {result_adaptive.max_context} tokens in {result_adaptive.search_metrics.total_time:.2f}s ({result_adaptive.search_metrics.total_tries} tries)")
            
        except Exception as e:
            logger.error(f"Error testing {test_model}: {e}")
            continue
    
    # Summary of all results
    logger.info("\n=== Algorithm Selection Summary ===")
    for result in results:
        logger.info(f"{result['model']}:")
        logger.info(f"  Pure Binary: {result['pure_binary'].max_context} tokens, {result['pure_binary'].search_metrics.total_time:.1f}s")
        logger.info(f"  Adaptive Binary: {result['adaptive_binary'].max_context} tokens, {result['adaptive_binary'].search_metrics.total_time:.1f}s")
    
    return results

def test_algorithm_comparison():
    """Test the algorithm comparison framework on filtered models."""
    logger.info("\n=== Testing Algorithm Comparison Framework ===")
    
    # Get filtered test models
    models = get_filtered_models()
    if not models:
        return []
    
    all_results = []
    
    for i, model_info in enumerate(models, 1):
        test_model = model_info["name"]
        max_ctx = fetch_max_context_size(test_model)
        
        logger.info(f"\n[{i}/{len(models)}] Comparing algorithms for: {test_model}")
        
        try:
            # Run comparison
            results = compare_algorithms(test_model, max_ctx)
            all_results.append((test_model, results))
            
            # Create detailed comparison report for each model
            report_file = f"algorithm_comparison_{test_model.replace(':', '_')}.csv"
            create_algorithm_comparison_report(test_model, max_ctx, report_file)
            logger.info(f"Detailed comparison report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error comparing algorithms for {test_model}: {e}")
            continue
    
    # Create combined summary report
    if all_results:
        logger.info("\n=== Algorithm Comparison Summary ===")
        for test_model, results in all_results:
            logger.info(f"\n{test_model}:")
            for algorithm, result in results.items():
                metrics = result.search_metrics
                efficiency = result.max_context / metrics.total_time if metrics.total_time > 0 else 0
                logger.info(f"  {algorithm.value:15} | {result.max_context:6} tokens | {metrics.total_time:6.2f}s | {metrics.total_tries:3} tries | {efficiency:7.0f} tokens/sec")
    
    return all_results

def test_probe_with_different_algorithms():
    """Test the main probe function with different algorithms on filtered models."""
    logger.info("\n=== Testing Probe Function with Different Algorithms ===")
    
    models = get_filtered_models()
    if not models:
        return
    
    for i, model_info in enumerate(models, 1):
        test_model = model_info["name"]
        safe_model_name = test_model.replace(':', '_')
        
        logger.info(f"\n[{i}/{len(models)}] Testing probe functions for: {test_model}")
        
        try:
            # Test with Pure Binary Search
            logger.info(f"--- Probing {test_model} with Pure Binary Search ---")
            probe_max_context(
                output_file=f"test_pure_binary_{safe_model_name}_results.csv",
                model_name=test_model,
                algorithm=SearchAlgorithm.PURE_BINARY
            )
            
            # Test with Adaptive Binary Search
            logger.info(f"--- Probing {test_model} with Adaptive Binary Search ---")
            probe_max_context(
                output_file=f"test_adaptive_binary_{safe_model_name}_results.csv", 
                model_name=test_model,
                algorithm=SearchAlgorithm.ADAPTIVE_BINARY
            )
            
            logger.info(f"Probe results for {test_model} saved to CSV files")
            
        except Exception as e:
            logger.error(f"Error probing {test_model}: {e}")
            continue

def compare_algorithms(model_name: str, max_ctx: int, algorithms: Optional[List[SearchAlgorithm]] = None) -> Dict[SearchAlgorithm, ProbeResult]:
    """
    Compare multiple search algorithms on the same model.
    
    Args:
        model_name: Name of the model to test
        max_ctx: Maximum context size reported by the model
        algorithms: List of algorithms to compare. If None, tests all available algorithms.
        
    Returns:
        Dictionary mapping algorithms to their results
    """
    if algorithms is None:
        algorithms = [SearchAlgorithm.PURE_BINARY, SearchAlgorithm.ADAPTIVE_BINARY]
    
    logger.info(f"=== Algorithm Comparison for {model_name} ===")
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Testing {algorithm.value}...")
        result = find_max_fit_in_vram(model_name, max_ctx, algorithm)
        results[algorithm] = result
        
        # Brief summary for comparison
        logger.info(f"{algorithm.value}: {result.max_context} tokens in {result.search_metrics.total_time:.1f}s ({result.search_metrics.total_tries} tries)")
    
    # Comparison summary
    logger.info("=== Comparison Summary ===")
    for algorithm, result in results.items():
        efficiency = result.max_context / result.search_metrics.total_time if result.search_metrics.total_time > 0 else 0
        logger.info(f"{algorithm.value}: {result.max_context} tokens, {result.search_metrics.total_time:.1f}s, {result.search_metrics.total_tries} tries, {efficiency:.0f} tokens/sec")
    
    return results


def create_algorithm_comparison_report(model_name: str, max_ctx: int, output_file: str) -> None:
    """
    Create a detailed comparison report of all available search algorithms.
    
    Args:
        model_name: Name of the model to test
        max_ctx: Maximum context size reported by the model  
        output_file: Path to save the comparison report
    """
    results = compare_algorithms(model_name, max_ctx)
      # Create detailed comparison report
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries', 
            'Precision Confidence (%)', 'Tokens/Second', 'Coarse Tries', 'Fine Tries',
            'Flat Memory Detections', 'Dynamic Granularity', 'Estimated Max Fit'
        ])
        for algorithm, result in results.items():
            metrics = result.search_metrics
            efficiency = result.max_context / metrics.total_time if metrics.total_time > 0 else 0
            
            writer.writerow([
                algorithm.value,
                result.max_context,
                f"{metrics.total_time:.2f}",
                metrics.total_tries,
                f"{metrics.precision_confidence:.2f}%" if metrics.precision_confidence is not None else "N/A",
                f"{efficiency:.0f}",
                metrics.coarse_tries if metrics.coarse_tries else "N/A",
                metrics.fine_tries if metrics.fine_tries else "N/A", 
                metrics.flat_memory_detections if metrics.flat_memory_detections else "N/A",
                metrics.dynamic_granularity if metrics.dynamic_granularity else "N/A",
                metrics.estimated_max_fit if metrics.estimated_max_fit else "N/A"
            ])
    
    logger.info(f"Algorithm comparison report saved to {output_file}")

def test_performance_comparison():
    """Compare the performance characteristics of different algorithms across filtered models."""
    logger.info("\n=== Performance Comparison Test ===")
    
    models = get_filtered_models()
    if not models:
        return []
    
    performance_data = []
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info["name"]
        max_ctx = fetch_max_context_size(model_name)
        
        logger.info(f"\n[{i}/{len(models)}] Testing performance on {model_name}")
        
        try:
            # Compare algorithms
            results = compare_algorithms(model_name, max_ctx)
            
            for algorithm, result in results.items():                
                metrics = result.search_metrics
                efficiency = result.max_context / metrics.total_time if metrics.total_time > 0 else 0
                
                performance_data.append({
                    'model': model_name,
                    'algorithm': algorithm.value,
                    'max_context': result.max_context,
                    'search_time': metrics.total_time,
                    'total_tries': metrics.total_tries,
                    'efficiency': efficiency,
                    'precision': metrics.precision_confidence
                })
                
        except Exception as e:
            logger.error(f"Error testing performance for {model_name}: {e}")
            continue
    
    # Log performance summary
    logger.info("\n=== Performance Summary ===")
    logger.info("Model | Algorithm | Tokens | Time | Tries | Efficiency")
    logger.info("-" * 70)
    for data in performance_data:
        logger.info(f"{data['model']:20} | {data['algorithm']:15} | {data['max_context']:6} | {data['search_time']:6.2f}s | {data['total_tries']:5} | {data['efficiency']:7.0f} tok/s")
    
    # Calculate algorithm averages
    if performance_data:
        logger.info("\n=== Algorithm Performance Averages ===")
        algorithms = set(data['algorithm'] for data in performance_data)
        for algorithm in algorithms:
            alg_data = [data for data in performance_data if data['algorithm'] == algorithm]
            avg_time = sum(data['search_time'] for data in alg_data) / len(alg_data)
            avg_tries = sum(data['total_tries'] for data in alg_data) / len(alg_data)
            avg_efficiency = sum(data['efficiency'] for data in alg_data) / len(alg_data)
            
            logger.info(f"{algorithm:15} | Avg Time: {avg_time:6.2f}s | Avg Tries: {avg_tries:5.1f} | Avg Efficiency: {avg_efficiency:7.0f} tok/s")
    
    return performance_data

def show_test_configuration():
    """Display a summary of the test configuration."""
    logger.info("="*80)
    logger.info("ALGORITHM REFACTORING TEST CONFIGURATION")
    logger.info("="*80)
    
    # MODEL_FILTER status
    if MODEL_FILTER:
        logger.info(f"✓ MODEL_FILTER: Testing {len(MODEL_FILTER)} specific models:")
        for i, model in enumerate(MODEL_FILTER, 1):
            logger.info(f"  {i}. {model}")
    else:
        logger.info("✓ MODEL_FILTER: Empty - will test all available models (subject to other filters)")
    
    # Advanced filters
    logger.info("\nAdvanced Filters:")
    if FILTER_BY_SIZE:
        logger.info(f"✓ FILTER_BY_SIZE: Only testing {FILTER_BY_SIZE} models")
    else:
        logger.info("✗ FILTER_BY_SIZE: Not active")
        
    if SKIP_LARGE_MODELS:
        logger.info("✓ SKIP_LARGE_MODELS: Skipping LARGE and XLARGE models")
    else:
        logger.info("✗ SKIP_LARGE_MODELS: Not active")
        
    if MAX_MODELS_TO_TEST:
        logger.info(f"✓ MAX_MODELS_TO_TEST: Limited to {MAX_MODELS_TO_TEST} models")
    else:
        logger.info("✗ MAX_MODELS_TO_TEST: No limit")
        
    if MAX_RUNTIME_MINUTES:
        logger.info(f"✓ MAX_RUNTIME_MINUTES: Limited to {MAX_RUNTIME_MINUTES} minutes estimated runtime")
    else:
        logger.info("✗ MAX_RUNTIME_MINUTES: No runtime limit")
    
    logger.info("="*80)

def main():
    """Run all tests for the refactored context probe module."""
    logger.info("Starting Algorithm Refactoring Tests")
    
    # Show test configuration
    show_test_configuration()
    
    try:
        # List available models
        logger.info("\n" + "="*60)
        all_models = list_available_models()
        
        if not all_models:
            logger.error("No models found. Please install models first.")
            return        # Test 1: Individual algorithm selection
        logger.info("\n" + "="*60)
        algorithm_results = test_algorithm_selection()
        
        # Test 2: Algorithm comparison framework
        logger.info("\n" + "="*60)
        comparison_results = test_algorithm_comparison()
        
        # Test 3: Probe function with different algorithms
        logger.info("\n" + "="*60)
        test_probe_with_different_algorithms()
        
        # Test 4: Performance comparison
        logger.info("\n" + "="*60)
        performance_data = test_performance_comparison()
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("=== All Tests Completed Successfully ===")
        
        models_tested = len(algorithm_results) if algorithm_results else 0
        logger.info(f"Models tested: {models_tested}")
        
        if performance_data:
            total_tests = len(performance_data)
            logger.info(f"Total algorithm tests: {total_tests}")
            
            # Find best performing algorithm overall
            algorithms = set(data['algorithm'] for data in performance_data)
            best_algorithm = None
            best_efficiency = 0
            
            for algorithm in algorithms:
                alg_data = [data for data in performance_data if data['algorithm'] == algorithm]
                avg_efficiency = sum(data['efficiency'] for data in alg_data) / len(alg_data)
                if avg_efficiency > best_efficiency:
                    best_efficiency = avg_efficiency
                    best_algorithm = algorithm
            
            if best_algorithm:
                logger.info(f"Best performing algorithm: {best_algorithm} ({best_efficiency:.0f} tokens/sec average)")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception("Detailed error traceback:")
        return False
        
    return True

if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test algorithm refactoring with multiple search algorithms")
    parser.add_argument("--list-models", action="store_true", help="Only list available models and exit")
    parser.add_argument("--only", choices=["selection", "comparison", "probe", "performance"], 
                        help="Run only a specific test")
    
    args = parser.parse_args()
    
    # Just list models if requested
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Run specific test or full suite
    try:
        if args.only:
            show_test_configuration()
            all_models = list_available_models()
            
            if args.only == "selection":
                test_algorithm_selection()
            elif args.only == "comparison":
                test_algorithm_comparison()
            elif args.only == "probe":
                test_probe_with_different_algorithms()
            elif args.only == "performance":
                test_performance_comparison()
            
            logger.info("Single test completed successfully")
        else:
            # Run the full test suite
            success = main()
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user. Exiting gracefully.")
        sys.exit(130)  # Standard exit code for SIGINT
