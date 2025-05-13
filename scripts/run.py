import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os
import numpy as np
from omegaconf import OmegaConf
from explora_backend.core.generate_models import GenerateModels
from explora_backend.utils.logger_config import log
from explora_backend.integrations.redis_manager import RedisManager
from explora_backend.utils.data_processing import DataPreprocessing
from explora_backend.utils.model_utils import ModelUtils
from explora_backend.utils.polars_utils import PolarsUtils
import re

class Run:
    def run_explora(cfg, df: pl.DataFrame):
        redis_manager = RedisManager(cfg.cache)
        info_list = redis_manager.retrieve_or_call(DataPreprocessing.get_dataset_info, (df, cfg), cfg)
        if not isinstance(info_list, (list, tuple)) or len(info_list) != 2:
            log.error("Expected list [task, target_col], got %s", info_list)
            return None
        task, target_var = info_list
        class_bad_list = redis_manager.retrieve_or_call(DataPreprocessing.get_classes_and_bad_columns, (df, target_var, cfg),cfg)
        if not isinstance(class_bad_list, (list, tuple)) or len(class_bad_list) != 3:
            log.error("Expected list [clean_targets, dirty_targets, bad_columns], got %s", class_bad_list)
            return None
        clean_targets, dirty_targets, bad_columns = class_bad_list
        # clean_df = DataPreprocessing.get_clean_df(df, clean_targets, bad_columns)
        clean_df = df
        try:
            all_models = []
            sorted_all_models = []
            cfg['all_models'] = []
            cfg['all_results'] = {'iterations': [], 'top_models': []}
            for iteration in range(1, cfg['iterations'] + 1):
                log.info(f"\n{'-'*30} ITERATION {iteration} OF {cfg['iterations']} {'-'*30}\n")
                try:
                    if iteration == 1:
                        iteration_models = GenerateModels.generate_initial_models(clean_df, task, target_var, clean_targets, cfg)
                    else:
                        iteration_models = GenerateModels.generate_iteration_models_with_all_history(clean_df, task, target_var, clean_targets, sorted_all_models, cfg)
                    log.debug("Generated models for iteration %s", iteration)

                    for model_num in sorted(iteration_models.keys()):
                        # Extract model name from the parsed data
                        model_data = iteration_models[model_num]
                        model_name = "Unnamed Model"
                        
                        # If model_data is a dictionary with a model_name key, use that value
                        if isinstance(model_data, dict) and 'model_name' in model_data:
                            model_name = model_data['model_name']
                        # If model_data is a string, use the first line or a placeholder
                        elif isinstance(model_data, str):
                            # Try to extract first line as name
                            first_line = model_data.split('\n')[0].strip() if '\n' in model_data else model_data[:50].strip()
                            if first_line and len(first_line.split()) <= 10:
                                model_name = first_line
                        
                        metrics = GenerateModels.test_model(clean_df, target_var, clean_targets, model_num, model_name, cfg)
                        if metrics is not None:
                            metrics['model_num'] = model_num
                            metrics['iteration'] = iteration
                            all_models.append(metrics)
                            iteration_models[model_num] = metrics
                    # Filter out underperforming models (less than 90% of best accuracy)
                    if all_models:
                        sorted_all_models = ModelUtils.sort_models(all_models)
                        best_metric = sorted_all_models[0]['metrics'].get('accuracy_score', 0) if sorted_all_models else 0
                        threshold = best_metric * 0.9
                        filtered_models = [m for m in all_models if (m['metrics']['accuracy_score'] if m and 'metrics' in m and 'accuracy_score' in m['metrics'] else 0) >= threshold]
                        if filtered_models:
                            cfg['all_models'] = filtered_models
                            log.info(f"\nFiltered out {len(all_models) - len(filtered_models)} underperforming models (below {threshold:.4f})")
                            # Sort by accuracy (desc) then false negatives (asc)
                            sorted_all_models = ModelUtils.sort_models(filtered_models)

                    # Display models ranked by accuracy
                    log.info("\n" + "="*80)
                    log.info(f"MODELS FROM ITERATION {iteration} RANKED BY ACCURACY_SCORE:")
                    log.info("="*80)
                    for i, model in enumerate(sorted_all_models):
                        model_name = model.get('model_name', 'Unnamed Model')
                        accuracy = model['metrics']['accuracy_score'] if model and 'metrics' in model and 'accuracy_score' in model['metrics'] else 0
                        log.info(f"RANK {i+1}: MODEL {model.get('model_num', 'Unknown')} - {model_name} - Accuracy: {accuracy:.4f}")
                        
                        # Display hypothesis if available
                        hypothesis = model.get('hypothesis', '')
                        if not hypothesis and 'description' in model:
                            # Try to extract from description if no explicit hypothesis
                            description = model['description']
                            if isinstance(description, str):
                                # Look for sentences containing hypothesis keywords
                                pattern = r'([^.!?]*(?:hypothesis|inference|test|predict|relationship|correlation|feature)[^.!?]*[.!?])'
                                matches = re.findall(pattern, description, re.IGNORECASE)
                                if matches:
                                    hypothesis = matches[0].strip()
                        
                        if hypothesis:
                            log.info(f"Hypothesis: {hypothesis}")
                        
                        log.info(f"Metrics: {model.get('metrics', 'No metrics')}")
                        log.info(f"Description: {model.get('description', 'No description')}")
                        log.info("-"*80)

                    log.info(f"sorted_all_models: {sorted_all_models}")

                    # Prepare iteration results for JSON
                    iteration_summary = {
                        'iteration_number': iteration,
                        'models': all_models,
                        'top_models': sorted_all_models[:3]
                    }
                    cfg['all_results']['iterations'].append(iteration_summary)
                    cfg['all_results']['top_models'] = sorted_all_models[:10]
                    # Set results_filename dynamically based on username and run_name
                    if 'results_filename' not in cfg:
                        username = cfg.get('username', 'user')
                        run_name = cfg.get('run_name', 'run')
                        results_dir = cfg.get('results_dir', 'explora_backend/results')
                        os.makedirs(results_dir, exist_ok=True)
                        cfg['results_filename'] = os.path.join(results_dir, f"{username}_{run_name}.json")
                    # Save after each iteration
                    try:
                        from omegaconf import OmegaConf
                        with open(cfg['results_filename'], 'w') as f:
                            json.dump(OmegaConf.to_container(cfg['all_results'], resolve=True), f, indent=2)
                        log.info(f"\nSaved results to {cfg['results_filename']}")
                    except Exception as json_error:
                        log.error(f"Error saving JSON file: {json_error}")
                except Exception as e:
                    log.error(f"An error occurred in iteration {iteration}: {e}")
                    log.info("Saving current results and continuing to next iteration if possible")
                    if 'results_filename' not in cfg:
                        username = cfg.get('username', 'user')
                        run_name = cfg.get('run_name', 'run')
                        results_dir = cfg.get('results_dir', 'explora_backend/results')
                        os.makedirs(results_dir, exist_ok=True)
                        cfg['results_filename'] = os.path.join(results_dir, f"{username}_{run_name}.json")
                    try:
                        partial_filename = os.path.join(cfg['results_dir'], f"partial_{os.path.basename(cfg['results_filename'])}")
                        with open(partial_filename, 'w') as f:
                            json.dump(OmegaConf.to_container(cfg['all_results'], resolve=True), f, indent=2)
                        log.info(f"Saved partial results to {partial_filename}")
                    except:
                        pass
        except Exception as e:
            log.error(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            return None
