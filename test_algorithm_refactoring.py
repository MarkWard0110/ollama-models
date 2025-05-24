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
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ollama_models.core.context_probe import (
    SearchAlgorithm, 
    find_max_fit_in_vram, 
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
MODEL_FILTER = [
    # === QUICK TEST MODELS (Small, fast) ===
    #"llama3.2:1b-instruct-fp16",  # ~1GB VRAM
    #"qwen3:0.6b-fp16",            # ~1GB VRAM
    # "qwen2.5-coder:1.5b-instruct-fp16", # ~1.5GB VRAM
    
    # === MEDIUM MODELS (Moderate testing) ===
    # "llama3.2:3b-instruct-fp16",  # ~2-3GB VRAM
    # "gemma3:4b-it-fp16",          # ~3GB VRAM
    
    # === LARGE MODELS (Thorough testing) ===
    # "llama3.1:8b-instruct-fp16",  # ~5GB VRAM
    # "qwen3:8b-fp16",              # ~5GB VRAM
]

# Advanced filter options
FILTER_BY_SIZE = None     
FILTER_BY_PARAMS: Optional[Tuple[Optional[float], Optional[float]]] = None  # Tuple[Optional[float], Optional[float]] or None
SKIP_LARGE_MODELS = False  
MAX_MODELS_TO_TEST = 1
MAX_RUNTIME_MINUTES = None

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
            
    else:
        logger.info("No models found")
    return models

def estimate_model_size_category(model_name: str) -> str:
    """
    Estimate model size category based on parameter count.
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
        return "UNKNOWN"

def apply_advanced_filters(models: List[Dict]) -> List[Dict]:
    """Apply advanced filtering options to model list."""
    filtered = models[:]
    
    # Apply size-based filtering
    if FILTER_BY_SIZE:
        size_filter = FILTER_BY_SIZE
        logger.info(f"Applying size filter: {size_filter}")
        filtered = [m for m in filtered if estimate_model_size_category(m['name']) == size_filter]
        logger.info(f"After size filtering: {len(filtered)} models")
      # Apply parameter count-based filtering
    if FILTER_BY_PARAMS is not None and isinstance(FILTER_BY_PARAMS, (tuple, list)) and len(FILTER_BY_PARAMS) == 2:
        min_billions, max_billions = FILTER_BY_PARAMS
        min_params = min_billions * 1e9 if min_billions is not None else 0
        max_params = max_billions * 1e9 if max_billions is not None else float('inf')
        
        logger.info(f"Applying parameter filter: {min_billions}B to {max_billions if max_billions else 'unlimited'}B")
        
        def is_in_param_range(model_name: str) -> bool:
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
    
    return filtered

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

class ProbeDataCollector:
    """Collects probe results for multiple analysis modes."""
    
    def __init__(self, output_base_dir: Optional[str] = None):
        self.model_results: Dict[str, Dict[SearchAlgorithm, ProbeResult]] = {}
        self.tested_models: List[str] = []
        
        # Set up organized output directory structure
        if output_base_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base_dir = f"test_results_{timestamp}"
        
        self.output_base_dir = Path(output_base_dir)
        self.setup_output_directories()
        
    def setup_output_directories(self):
        """Create organized directory structure for outputs."""
        directories = [
            self.output_base_dir,
            self.output_base_dir / "algorithm_comparisons",
            self.output_base_dir / "probe_outputs", 
            self.output_base_dir / "performance_reports",
            self.output_base_dir / "summaries"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Output directories created in: {self.output_base_dir}")
        
    def get_output_path(self, analysis_type: str, filename: str) -> Path:
        """Get the full path for an output file based on analysis type."""
        type_mapping = {
            'comparison': 'algorithm_comparisons',
            'probe': 'probe_outputs',
            'performance': 'performance_reports',
            'summary': 'summaries'
        }
        
        subdirectory = type_mapping.get(analysis_type, '')
        if subdirectory:
            return self.output_base_dir / subdirectory / filename
        else:
            return self.output_base_dir / filename
        
    def add_result(self, model_name: str, algorithm: SearchAlgorithm, result: ProbeResult):
        """Add a probe result to the collection."""
        if model_name not in self.model_results:
            self.model_results[model_name] = {}
            self.tested_models.append(model_name)
        self.model_results[model_name][algorithm] = result
        
    def get_models(self) -> List[str]:
        """Get list of tested models."""
        return self.tested_models.copy()
        
    def get_result(self, model_name: str, algorithm: SearchAlgorithm) -> Optional[ProbeResult]:
        """Get a specific probe result."""
        return self.model_results.get(model_name, {}).get(algorithm)
        
    def get_all_results_for_model(self, model_name: str) -> Dict[SearchAlgorithm, ProbeResult]:
        """Get all algorithm results for a specific model."""
        return self.model_results.get(model_name, {}).copy()

def run_all_probes(models_to_test: List[Dict], algorithms: Optional[List[SearchAlgorithm]] = None) -> ProbeDataCollector:
    """
    Run probes for all models and algorithms, collecting results for later analysis.
    
    Args:
        models_to_test: List of model info dictionaries
        algorithms: List of algorithms to test (defaults to all available)
        
    Returns:
        ProbeDataCollector containing all results
    """
    if algorithms is None:
        algorithms = [SearchAlgorithm.PURE_BINARY_G32, SearchAlgorithm.PURE_BINARY_G01, SearchAlgorithm.ADAPTIVE_BINARY]
    
    collector = ProbeDataCollector()
    
    logger.info("=== Running All Probes ===")
    logger.info(f"Testing {len(models_to_test)} models with {len(algorithms)} algorithms")
    
    total_tests = len(models_to_test) * len(algorithms)
    current_test = 0
    
    for model_info in models_to_test:
        model_name = model_info["name"]
        max_ctx = fetch_max_context_size(model_name)
        
        logger.info(f"\nTesting model: {model_name} (max_ctx: {max_ctx})")
        
        for algorithm in algorithms:
            current_test += 1
            logger.info(f"[{current_test}/{total_tests}] Running {algorithm.value} on {model_name}")
            
            try:
                result = find_max_fit_in_vram(model_name, max_ctx, algorithm)
                collector.add_result(model_name, algorithm, result)
                  # Brief progress update
                metrics = result.search_metrics
                logger.info(f"  ✓ {result.max_context} tokens in {metrics.total_time:.1f}s")
                
            except Exception as e:
                logger.error(f"  ✗ Error testing {algorithm.value} on {model_name}: {e}")
                continue
    
    logger.info(f"\nProbe collection complete: {len(collector.tested_models)} models tested")
    return collector

def analyze_algorithm_selection(collector: ProbeDataCollector):
    """Analyze results from the perspective of algorithm selection."""
    logger.info("\n=== Algorithm Selection Analysis ===")
    
    for model_name in collector.get_models():
        results = collector.get_all_results_for_model(model_name)
        
        logger.info(f"\n{model_name}:")
        for algorithm, result in results.items():
            if result:
                metrics = result.search_metrics
                logger.info(f"  {algorithm.value}: {result.max_context} tokens in {metrics.total_time:.2f}s ({metrics.total_tries} tries)")
                
        # Save individual algorithm selection report for each model
        safe_model_name = model_name.replace(':', '_')
        selection_filename = f"algorithm_selection_{safe_model_name}.csv"
        selection_path = collector.get_output_path('summary', selection_filename)
        _save_algorithm_selection_report(model_name, results, str(selection_path))
        logger.info(f"Algorithm selection report for {model_name} saved to {selection_path}")
        
    # Create a consolidated algorithm selection summary
    consolidated_selection_path = collector.get_output_path('summary', 'algorithm_selection_summary.csv')
    _save_consolidated_algorithm_selection(collector, str(consolidated_selection_path))
    logger.info(f"Consolidated algorithm selection summary saved to {consolidated_selection_path}")

def analyze_algorithm_comparison(collector: ProbeDataCollector):
    """Analyze results from the perspective of algorithm comparison."""
    logger.info("\n=== Algorithm Comparison Analysis ===")
    
    # Create comparison reports for each model
    for model_name in collector.get_models():
        results = collector.get_all_results_for_model(model_name)
        if not results:
            continue
        
        logger.info(f"\n--- Comparison for {model_name} ---")
        logger.info(f"{'Algorithm':<20} | {'Tokens':<8} | {'Time':<8} | {'Tries':<6}")
        logger.info("-" * 58)
        
        for algorithm, result in results.items():
            if result:
                metrics = result.search_metrics
                logger.info(f"{algorithm.value:<20} | {result.max_context:<8} | {metrics.total_time:<8.2f} | {metrics.total_tries:<6}")
        
        # Save detailed comparison report to organized directory
        safe_model_name = model_name.replace(':', '_')
        report_filename = f"algorithm_comparison_{safe_model_name}.csv"
        report_path = collector.get_output_path('comparison', report_filename)
        _save_comparison_report(model_name, results, str(report_path))
        logger.info(f"Detailed report saved to {report_path}")
        
    # Create a summary comparison report across all models
    summary_path = collector.get_output_path('summary', 'algorithm_comparison_summary.csv')
    _save_summary_comparison_report(collector, str(summary_path))
    logger.info(f"Summary comparison report saved to {summary_path}")

def analyze_probe_outputs(collector: ProbeDataCollector):
    """Analyze results from the perspective of probe function outputs."""
    logger.info("\n=== Probe Output Analysis ===")
    
    for model_name in collector.get_models():
        results = collector.get_all_results_for_model(model_name)
        safe_model_name = model_name.replace(':', '_')
        
        # Generate CSV outputs similar to the original probe function
        for algorithm, result in results.items():
            if result:
                output_filename = f"test_{algorithm.value.lower().replace(' ', '_')}_{safe_model_name}_results.csv"
                output_path = collector.get_output_path('probe', output_filename)
                _save_probe_output(model_name, algorithm, result, str(output_path))
                logger.info(f"Probe output for {model_name} ({algorithm.value}) saved to {output_path}")
                
    # Create a consolidated probe results file
    consolidated_path = collector.get_output_path('summary', 'consolidated_probe_results.csv')
    _save_consolidated_probe_results(collector, str(consolidated_path))
    logger.info(f"Consolidated probe results saved to {consolidated_path}")

def analyze_performance_comparison(collector: ProbeDataCollector):
    """Analyze results from the perspective of performance comparison."""
    logger.info("\n=== Performance Comparison Analysis ===")
    
    performance_data = []
    
    # Collect all performance metrics
    for model_name in collector.get_models():
        results = collector.get_all_results_for_model(model_name)
        for algorithm, result in results.items():
            if result:
                metrics = result.search_metrics
                
                performance_data.append({
                    'model': model_name,
                    'algorithm': algorithm.value,
                    'max_context': result.max_context,
                    'search_time': metrics.total_time,
                    'total_tries': metrics.total_tries,
                    'precision': metrics.precision_confidence
                })
      # Performance summary
    logger.info("\n--- Individual Results ---")
    logger.info(f"{'Model':<25} | {'Algorithm':<15} | {'Tokens':<8} | {'Time':<8} | {'Tries':<6}")
    logger.info("-" * 75)
    
    for data in performance_data:
        logger.info(f"{data['model']:<25} | {data['algorithm']:<15} | {data['max_context']:<8} | {data['search_time']:<8.2f} | {data['total_tries']:<6}")
      # Algorithm averages
    if performance_data:
        logger.info("\n--- Algorithm Performance Averages ---")
        algorithms = set(data['algorithm'] for data in performance_data)
        
        logger.info(f"{'Algorithm':<15} | {'Avg Time':<10} | {'Avg Tries':<10}")
        logger.info("-" * 45)
        
        for algorithm in algorithms:
            alg_data = [data for data in performance_data if data['algorithm'] == algorithm]
            avg_time = sum(data['search_time'] for data in alg_data) / len(alg_data)
            avg_tries = sum(data['total_tries'] for data in alg_data) / len(alg_data)
            
            logger.info(f"{algorithm:<15} | {avg_time:<10.2f} | {avg_tries:<10.1f}")
    
    # Save detailed performance report
    performance_path = collector.get_output_path('performance', 'detailed_performance_analysis.csv')
    _save_detailed_performance_report(performance_data, str(performance_path))
    logger.info(f"Detailed performance report saved to {performance_path}")
    
    return performance_data

def _save_comparison_report(model_name: str, results: Dict[SearchAlgorithm, ProbeResult], output_file: str):
    """Save detailed algorithm comparison report to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)        
        writer.writerow([
            'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries', 
            'Precision Confidence (%)', 'Coarse Tries', 'Fine Tries',
            'Flat Memory Detections', 'Dynamic Granularity', 'Estimated Max Fit'
        ])
        
        for algorithm, result in results.items():
            if result:
                metrics = result.search_metrics
                
                writer.writerow([
                    algorithm.value,
                    result.max_context,
                    f"{metrics.total_time:.2f}",
                    metrics.total_tries,
                    f"{metrics.precision_confidence:.2f}%" if metrics.precision_confidence is not None else "N/A",
                    metrics.coarse_tries if metrics.coarse_tries else "N/A",
                    metrics.fine_tries if metrics.fine_tries else "N/A", 
                    metrics.flat_memory_detections if metrics.flat_memory_detections else "N/A",
                    metrics.dynamic_granularity if metrics.dynamic_granularity else "N/A",
                    metrics.estimated_max_fit if metrics.estimated_max_fit else "N/A"
                ])

def _save_probe_output(model_name: str, algorithm: SearchAlgorithm, result: ProbeResult, output_file: str):
    """Save probe output in the format expected by the original probe function."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Max Context', 'Search Time', 'Total Tries', 'Precision'])
        
        metrics = result.search_metrics
        writer.writerow([
            model_name,
            algorithm.value,
            result.max_context,
            f"{metrics.total_time:.2f}",
            metrics.total_tries,
            f"{metrics.precision_confidence:.2f}%" if metrics.precision_confidence is not None else "N/A"
        ])

def _save_summary_comparison_report(collector: ProbeDataCollector, output_file: str):
    """Save a summary comparison report across all models."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries', 
            'Precision Confidence (%)', 'Coarse Tries', 'Fine Tries',
            'Flat Memory Detections', 'Dynamic Granularity', 'Estimated Max Fit'
        ])
        
        for model_name in collector.get_models():
            results = collector.get_all_results_for_model(model_name)
            for algorithm, result in results.items():
                if result:
                    metrics = result.search_metrics
                    
                    writer.writerow([
                        model_name,
                        algorithm.value,
                        result.max_context,
                        f"{metrics.total_time:.2f}",
                        metrics.total_tries,
                        f"{metrics.precision_confidence:.2f}%" if metrics.precision_confidence is not None else "N/A",
                        metrics.coarse_tries if metrics.coarse_tries else "N/A",
                        metrics.fine_tries if metrics.fine_tries else "N/A", 
                        metrics.flat_memory_detections if metrics.flat_memory_detections else "N/A",
                        metrics.dynamic_granularity if metrics.dynamic_granularity else "N/A",
                        metrics.estimated_max_fit if metrics.estimated_max_fit else "N/A"
                    ])

def _save_consolidated_probe_results(collector: ProbeDataCollector, output_file: str):
    """Save consolidated probe results across all models and algorithms."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Max Context', 'Search Time', 'Total Tries', 'Precision'])
        
        for model_name in collector.get_models():
            results = collector.get_all_results_for_model(model_name)
            for algorithm, result in results.items():
                if result:
                    metrics = result.search_metrics
                    writer.writerow([
                        model_name,
                        algorithm.value,
                        result.max_context,
                        f"{metrics.total_time:.2f}",
                        metrics.total_tries,
                        f"{metrics.precision_confidence:.2f}%" if metrics.precision_confidence is not None else "N/A"
                    ])

def _save_detailed_performance_report(performance_data: List[Dict], output_file: str):
    """Save detailed performance analysis to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries', 
            'Precision Confidence (%)'
        ])
        
        for data in performance_data:
            writer.writerow([
                data['model'],
                data['algorithm'],
                data['max_context'],
                f"{data['search_time']:.2f}",
                data['total_tries'],
                f"{data['precision']:.2f}%" if data['precision'] is not None else "N/A"
            ])

def _save_algorithm_selection_report(model_name: str, results: Dict[SearchAlgorithm, ProbeResult], output_file: str):
    """Save algorithm selection report for a specific model to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries'])
        
        for algorithm, result in results.items():
            if result:
                metrics = result.search_metrics
                
                writer.writerow([
                    model_name,
                    algorithm.value,
                    result.max_context,
                    f"{metrics.total_time:.2f}",
                    metrics.total_tries
                ])

def _save_consolidated_algorithm_selection(collector: ProbeDataCollector, output_file: str):
    """Save consolidated algorithm selection summary across all models."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Algorithm', 'Max Context', 'Search Time (s)', 'Total Tries', 'Recommended'])
        
        for model_name in collector.get_models():
            results = collector.get_all_results_for_model(model_name)
            
            # Determine which algorithm is recommended (fastest for this model)
            best_algorithm = None
            best_time = float('inf')
            
            for algorithm, result in results.items():
                if result and result.search_metrics.total_time < best_time:
                    best_time = result.search_metrics.total_time
                    best_algorithm = algorithm.value
            
            for algorithm, result in results.items():
                if result:
                    metrics = result.search_metrics
                    is_recommended = "Yes" if algorithm.value == best_algorithm else "No"
                    
                    writer.writerow([
                        model_name,
                        algorithm.value,
                        result.max_context,
                        f"{metrics.total_time:.2f}",
                        metrics.total_tries,
                        is_recommended
                    ])

# ...existing code...

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
            return False
            
        # Get filtered models to test
        models_to_test = get_filtered_models()
        if not models_to_test:
            logger.error("No models match the filter criteria.")
            return False
        
        # Run all probes once
        logger.info("\n" + "="*60)
        collector = run_all_probes(models_to_test)
        
        # Analyze results in different ways
        logger.info("\n" + "="*60)
        analyze_algorithm_selection(collector)
        
        logger.info("\n" + "="*60)
        analyze_algorithm_comparison(collector)
        
        logger.info("\n" + "="*60)
        analyze_probe_outputs(collector)
        
        logger.info("\n" + "="*60)
        performance_data = analyze_performance_comparison(collector)
          # Final summary
        logger.info("\n" + "="*60)
        logger.info("=== All Tests Completed Successfully ===")
        
        models_tested = len(collector.get_models())
        logger.info(f"Models tested: {models_tested}")
        
        if performance_data:
            total_tests = len(performance_data)
            logger.info(f"Total algorithm tests: {total_tests}")
            
            # Find best performing algorithm overall (fastest average time)
            algorithms = set(data['algorithm'] for data in performance_data)
            best_algorithm = None
            best_avg_time = float('inf')
            
            for algorithm in algorithms:
                alg_data = [data for data in performance_data if data['algorithm'] == algorithm]
                avg_time = sum(data['search_time'] for data in alg_data) / len(alg_data)
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_algorithm = algorithm
            
            if best_algorithm:
                logger.info(f"Best performing algorithm: {best_algorithm} ({best_avg_time:.2f}s average time)")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception("Detailed error traceback:")
        return False
        
    return True

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test algorithm refactoring with multiple search algorithms")
    parser.add_argument("--list-models", action="store_true", help="Only list available models and exit")
    parser.add_argument("--only", choices=["selection", "comparison", "probe", "performance"], 
                        help="Run only a specific analysis")
    parser.add_argument("--exclude", choices=["selection", "comparison", "probe", "performance"], 
                        action="append", help="Exclude specific analyses")
    
    args = parser.parse_args()
    
    # Just list models if requested
    if args.list_models:
        list_available_models()
        sys.exit(0)
    
    # Run specific analysis or full suite
    try:
        show_test_configuration()
        all_models = list_available_models()
        
        if not all_models:
            logger.error("No models found. Please install models first.")
            sys.exit(1)
            
        models_to_test = get_filtered_models()
        if not models_to_test:
            logger.error("No models match the filter criteria.")
            sys.exit(1)
        
        # Run probes once
        collector = run_all_probes(models_to_test)
        
        # Determine which analyses to run
        excluded = set(args.exclude or [])
        
        if args.only:
            # Run only the specified analysis
            if args.only == "selection" and "selection" not in excluded:
                analyze_algorithm_selection(collector)
            elif args.only == "comparison" and "comparison" not in excluded:
                analyze_algorithm_comparison(collector)
            elif args.only == "probe" and "probe" not in excluded:
                analyze_probe_outputs(collector)
            elif args.only == "performance" and "performance" not in excluded:
                analyze_performance_comparison(collector)
        else:
            # Run all analyses except excluded ones
            if "selection" not in excluded:
                analyze_algorithm_selection(collector)
            if "comparison" not in excluded:
                analyze_algorithm_comparison(collector)
            if "probe" not in excluded:
                analyze_probe_outputs(collector)
            if "performance" not in excluded:
                analyze_performance_comparison(collector)
        
        logger.info("Analysis completed successfully")
        sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user. Exiting gracefully.")
        sys.exit(130)  # Standard exit code for SIGINT
