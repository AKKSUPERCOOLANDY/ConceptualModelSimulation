import polars as pl
import numpy as np
import argparse
import os
from omegaconf import OmegaConf
from tabulate import tabulate

from explora_backend.core.generate_models import GenerateModels
from explora_backend.utils.logger_config import log
from explora_backend.utils.data_processing import DataPreprocessing
from explora_backend.integrations.redis_manager import RedisManager
from explora_backend.utils.polars_utils import PolarsUtils
from explora_backend.utils.model_utils import ModelUtils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate initial models and test each one multiple times')
    parser.add_argument('--iterations', type=int, default=10, help='Number of times to test each model')
    parser.add_argument('--config_path', type=str, default="explora_backend/config/main.yaml", help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config_path)
    log.info(f"Using configuration with gemini.temperature: {cfg.gemini.temperature}")
    
    # Load dataset
    data_file = "data/breast.csv"
    if not os.path.exists(data_file):
        data_file = "./data/breast.csv"
        if not os.path.exists(data_file):
            # Try to find the file
            potential_paths = [
                "data/breast.csv",
                "./data/breast.csv",
                "../data/breast.csv",
                "data/breast-cancer.csv",
                "./data/breast-cancer.csv",
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    data_file = path
                    break
    
    log.info(f"Loading dataset from: {data_file}")
    df = pl.read_csv(data_file)
    
    # Setup Redis manager and retrieve dataset info
    redis_manager = RedisManager(cfg.cache)
    info_list = redis_manager.retrieve_or_call(DataPreprocessing.get_dataset_info, (df, cfg), cfg)
    task, target_var = info_list
    
    # Get classes and bad columns
    class_bad_list = redis_manager.retrieve_or_call(DataPreprocessing.get_classes_and_bad_columns, (df, target_var, cfg), cfg)
    clean_targets, dirty_targets, bad_columns = class_bad_list
    
    # Use full dataset
    clean_df = df
    
    # Generate initial models (same as run.py does)
    log.info("Generating initial 5 models...")
    models = GenerateModels.generate_initial_models(clean_df, task, target_var, clean_targets, cfg)
    
    if not models:
        log.error("Failed to generate models. Exiting.")
        return
    
    log.info(f"Generated {len(models)} models")
    
    # Test each model multiple times
    model_results = {}
    
    for model_num in sorted(models.keys()):
        model_data = models[model_num]
        model_name = model_data.get('model_name', f"Model {model_num}")
        log.info(f"\n\n{'='*80}")
        log.info(f"TESTING MODEL {model_num}: {model_name}")
        log.info(f"{'='*80}")
        
        # Display model info
        log.info(f"Model description: {model_data.get('description', 'No description')[:500]}")
        if 'hypothesis' in model_data:
            log.info(f"Hypothesis: {model_data['hypothesis']}")
        
        # Run tests multiple times
        results = []
        log.info(f"Running {args.iterations} test iterations for Model {model_num}...")
        
        for i in range(args.iterations):
            log.info(f"  Run {i+1}/{args.iterations}")
            metrics = GenerateModels.test_model(clean_df, target_var, clean_targets, model_num, model_name, cfg)
            
            if metrics is not None:
                accuracy = metrics['metrics']['accuracy_score']
                results.append({
                    'run': i+1,
                    'accuracy': accuracy,
                    'macro_f1': metrics['metrics']['macro_f1'],
                    'false_negatives': metrics['metrics']['false_negatives'],
                    'false_positives': metrics['metrics']['false_positives']
                })
                log.info(f"  Run {i+1} completed with accuracy: {accuracy:.4f}")
            else:
                log.warning(f"  Run {i+1} failed to produce metrics")
        
        # Sort and record results
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        model_results[model_num] = {
            'name': model_name,
            'results': sorted_results,
            'stats': calculate_stats(results) if results else None
        }
        
        # Print results in sorted order
        log.info("\n" + "-"*80)
        log.info(f"RESULTS FOR MODEL {model_num} ({model_name}) SORTED BY ACCURACY:")
        log.info("-"*80)
        
        for i, result in enumerate(sorted_results):
            log.info(f"RANK {i+1}: Run {result['run']} - Accuracy: {result['accuracy']:.4f}, F1: {result['macro_f1']:.4f}, FN: {result['false_negatives']}, FP: {result['false_positives']}")
        
        # Calculate and print statistics
        if results:
            stats = model_results[model_num]['stats']
            log.info("\n" + "-"*80)
            log.info(f"STATISTICS FOR MODEL {model_num}:")
            log.info(f"Average Accuracy: {stats['mean']:.4f}")
            log.info(f"Standard Deviation: {stats['std']:.4f}")
            log.info(f"Min Accuracy: {stats['min']:.4f}")
            log.info(f"Max Accuracy: {stats['max']:.4f}")
            log.info("-"*80)
        else:
            log.warning("No results to analyze")
    
    # Final comparison of all models
    print_final_comparison(model_results)

def calculate_stats(results):
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['macro_f1'] for r in results]
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'min': min(accuracies),
        'max': max(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }

def print_final_comparison(model_results):
    log.info("\n\n" + "="*100)
    log.info("FINAL COMPARISON OF ALL MODELS")
    log.info("="*100)
    
    # Prepare table data
    table_data = []
    headers = ["Model #", "Model Name", "Avg Accuracy", "Std Dev", "Min Acc", "Max Acc", "Avg F1", "Stability"]
    
    for model_num, data in sorted(model_results.items()):
        stats = data['stats']
        if stats:
            # Calculate a stability score (lower std dev = more stable)
            stability = 1.0 - (stats['std'] / 0.1)  # Normalize: 0.1 std dev or higher = 0 stability
            stability = max(0, min(1, stability))  # Clamp between 0-1
            
            table_data.append([
                model_num,
                data['name'],
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{stats['f1_mean']:.4f}",
                f"{stability:.2f}"
            ])
    
    # Sort by average accuracy
    table_data.sort(key=lambda x: float(x[2]), reverse=True)
    
    # Print table
    log.info(tabulate(table_data, headers=headers, tablefmt="grid"))
    log.info("\nNote: Stability score (0-1) indicates consistency of results, with 1 being perfectly consistent")

if __name__ == "__main__":
    main() 