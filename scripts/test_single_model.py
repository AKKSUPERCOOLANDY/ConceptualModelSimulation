import sys
import argparse
import polars as pl
from omegaconf import OmegaConf
import numpy as np
import os

from explora_backend.core.generate_models import GenerateModels
from explora_backend.utils.data_processing import DataPreprocessing
from explora_backend.utils.logger_config import log
from explora_backend.utils.model_utils import ModelUtils
from explora_backend.integrations.redis_manager import RedisManager

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a single model multiple times')
    parser.add_argument('model_num', type=int, help='Model number to test')
    parser.add_argument('--model_name', type=str, default="Unnamed Model", help='Name of the model')
    parser.add_argument('--iterations', type=int, default=10, help='Number of times to test the model')
    parser.add_argument('--config_path', type=str, default="explora_backend/config/main.yaml", help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    cfg = OmegaConf.load(args.config_path)
    log.info(f"Using configuration with gemini.temperature: {cfg.gemini.temperature}")
    
    # Load dataset - search for the correct path
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
    
    # Use full dataset for testing
    clean_df = df
    
    # Run multiple tests
    results = []
    log.info(f"Testing Model {args.model_num} ({args.model_name}) for {args.iterations} iterations...")
    
    for i in range(args.iterations):
        log.info(f"Run {i+1}/{args.iterations}")
        metrics = GenerateModels.test_model(clean_df, target_var, clean_targets, args.model_num, args.model_name, cfg)
        
        if metrics is not None:
            accuracy = metrics['metrics']['accuracy_score']
            results.append({
                'run': i+1,
                'accuracy': accuracy,
                'macro_f1': metrics['metrics']['macro_f1'],
                'false_negatives': metrics['metrics']['false_negatives'],
                'false_positives': metrics['metrics']['false_positives']
            })
            log.info(f"Run {i+1} completed with accuracy: {accuracy:.4f}")
        else:
            log.warning(f"Run {i+1} failed to produce metrics")
    
    # Sort results by accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Print results in sorted order
    log.info("\n" + "="*80)
    log.info(f"RESULTS FOR MODEL {args.model_num} ({args.model_name}) SORTED BY ACCURACY:")
    log.info("="*80)
    
    for i, result in enumerate(sorted_results):
        log.info(f"RANK {i+1}: Run {result['run']} - Accuracy: {result['accuracy']:.4f}, F1: {result['macro_f1']:.4f}, FN: {result['false_negatives']}, FP: {result['false_positives']}")
    
    # Calculate and print statistics
    if results:
        accuracies = [r['accuracy'] for r in results]
        log.info("\n" + "-"*80)
        log.info(f"STATISTICS:")
        log.info(f"Average Accuracy: {np.mean(accuracies):.4f}")
        log.info(f"Standard Deviation: {np.std(accuracies):.4f}")
        log.info(f"Min Accuracy: {min(accuracies):.4f}")
        log.info(f"Max Accuracy: {max(accuracies):.4f}")
        log.info("-"*80)
    else:
        log.warning("No results to analyze")

if __name__ == "__main__":
    main() 