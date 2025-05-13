import polars as pl
from omegaconf import DictConfig
from typing import Dict, List, Any, Optional
import os
import re
from sklearn.metrics import classification_report
import numpy as np
from explora_backend.integrations.gemini_client import GeminiClient
from explora_backend.utils.logger_config import log
from explora_backend.utils.gemini_parser import GeminiParser
from explora_backend.utils.polars_utils import PolarsUtils
from explora_backend.utils.calculations_utils import CalculationUtils

class GenerateModels:
    def generate_initial_models(df: pl.DataFrame, task: str, target_var: str, clean_targets: list, cfg: DictConfig) -> Optional[str]:
        prompt = f"""You are a world-class data scientist creating original conceptual models for {task}.
        I have a dataset with the following features, where '{target_var}' is the target variable and we are classifying into {clean_targets}
        {PolarsUtils.print_df(df)[:1000]}
        
        I've split the dataset into 80% for training and 20% for validation.
        
        Please create 5 different classification models to predict '{target_var}' given classes {clean_targets}.
        Each model should test a specific inference or hypothesis about the data.

        For each model:
        1. Label it as MODEL 1, MODEL 2, etc.
        2. Give it a SHORT, DESCRIPTIVE NAME (max 5-10 words) that instantly communicates its purpose and key design features
        3. State the specific inference or hypothesis about the data that this model will test
        4. Explain how the model design tests this inference
        5. Describe which features are most relevant to testing this hypothesis
        6. Explain how the model's performance would validate or invalidate the inference

        IMPORTANT: 
        - Each model should test a unique hypothesis about the data's structure or relationships
        - These should be original conceptual models that you create yourself, not copied from existing papers
        - Focus on clear, testable inferences about how different features relate to the classification
        - Keep explanations focused on the inference being tested and how the model tests it
        - Model names should be specific, concise, and descriptive (e.g., "Feature-Weighted Random Forest" or "Threshold-Based Decision Tree")
        - No need to include detailed implementation code"""
        response = GeminiClient.generate_gemini_content(prompt, cfg)
        return GeminiParser.parse_models(response)

    def generate_iteration_models_with_all_history(df: pl.DataFrame, task: str, target_var: str, clean_targets: list, models: Dict, cfg: DictConfig):
        # Assume models are already sorted when passed in
        top_models = models[:3] if len(models) >= 3 else models

        prompt = f"""You are a world-class data scientist working on a {task} problem.

    I have a dataset with the following features, where '{target_var}' is the target variable and we are classifying between {clean_targets}:
    {PolarsUtils.print_df(df)[:1000]}

    I've already tried several models in previous iterations. Here are the top performing models so far:

    """
        for i, model in enumerate(top_models):
            accuracy = model.get('metrics', {}).get('accuracy_score', 0)
            model_name = model.get('model_name', 'Unnamed Model')
            prompt += f"""TOP MODEL {i+1}: {model_name} (Accuracy: {accuracy:.4f})
            Iteration: {model.get('iteration', 'Unknown')}, Model #: {model.get('model_num', 'Unknown')}
            Key Hypothesis: {model.get('description', 'No description')[:500] if isinstance(model.get('description'), str) else 'No description'}
            
            """
        
        # Organize all past models by iteration
        iterations = {}
        for model in models:
            iter_num = model.get('iteration', 0)
            if iter_num not in iterations:
                iterations[iter_num] = []
            iterations[iter_num].append(model)
        
        # Add summarized history of all models by iteration
        prompt += """HISTORY OF ALL PREVIOUS MODELS BY ITERATION:
        
        """
        
        for iter_num in sorted(iterations.keys()):
            prompt += f"ITERATION {iter_num}:\n"
            iter_models = iterations[iter_num]
            
            # Sort models within each iteration by accuracy
            iter_models = sorted(iter_models, key=lambda x: x.get('metrics', {}).get('accuracy_score', 0), reverse=True)
            
            for model in iter_models:
                accuracy = model.get('metrics', {}).get('accuracy_score', 0)
                model_name = model.get('model_name', 'Unnamed Model')
                model_num = model.get('model_num', 'Unknown')
                
                # Extract a concise hypothesis from the description
                description = model.get('description', '')
                hypothesis = ""
                if isinstance(description, str):
                    # Try to find a sentence containing hypothesis, inference, or tests
                    pattern = r'([^.!?]*(?:hypothesis|inference|test|predict|relationship|correlation|feature)[^.!?]*[.!?])'
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    if matches:
                        hypothesis = matches[0].strip()
                    else:
                        # Just get the first sentence if no match
                        sentences = re.split(r'[.!?]', description)
                        if sentences:
                            hypothesis = sentences[0].strip()[:200] + "..."
                
                prompt += f"- Model {model_num}: {model_name} (Accuracy: {accuracy:.4f})\n"
                prompt += f"  Hypothesis: {hypothesis if hypothesis else 'Not specified'}\n\n"
        
        # Determine exploration strategy based on iteration
        remaining_iterations = 10 - max(model['iteration'] for model in models)
        
        prompt += f"""\nPlease create 5 NEW models for {task}. For each model:
    1. Label it as MODEL 1, MODEL 2, etc.
    2. Give it a SHORT, DESCRIPTIVE NAME (max 5-10 words) that instantly communicates its purpose and key design features
    3. State the specific inference or hypothesis about the data that this model will test
    4. Provide a detailed explanation of how the model tests this inference
    5. Justify why testing this inference is valuable 
    6. Explain how this inference relates to or differs from previous model insights

    REQUIREMENTS:
    - **Make the models more complex than the previous models**
    - Each model must test a meaningfully different inference about the data
    - Focus on hypotheses that could lead to exceeding our current best accuracy
    - If retesting a previously invalidated inference, clearly explain the new evidence or approach
    - Ensure each model has a clear path to implementation using standard ML libraries
    - Prioritize both accuracy and reliability, with special attention to false negative rates
    - EXPLICITLY combine features and techniques from the top-performing models
    - At least 3 of your 5 models should directly iterate on or combine elements from the top models
    - Models should reuse successful components while addressing identified weaknesses
    - Model names should be specific, concise, and descriptive (e.g., "Feature-Weighted Random Forest" or "Threshold-Based Decision Tree")"""
        response = GeminiClient.generate_gemini_content(prompt, cfg)
        return GeminiParser.parse_models(response)
        
    def test_model(df_clean: pl.DataFrame, target_var: str, clean_targets: list, model_num: int, model_name: str, cfg: DictConfig) -> tuple:
        max_test_samples = cfg['max_test_samples']
        test_features_df = df_clean.sample(n=min(max_test_samples, len(df_clean)))
        # Get ground-truth labels for test set (y_true)
        y_true = test_features_df[target_var].to_list()
        # Drop the target column from the features used for prediction
        test_features_df_no_target = test_features_df.drop(target_var)
        # Set index to start at 1 for Gemini prompt compatibility
        df_for_prompt = test_features_df_no_target.to_pandas().copy()
        df_for_prompt.index = np.arange(1, len(df_for_prompt) + 1)
        test_features_str = df_for_prompt.to_string(index=True, float_format='%.4f')
        log.debug(f"Prompt table length: {len(test_features_str)}. Preview:\n{test_features_str[:500]} ... \n[truncated]")
        log.debug(test_features_str)
        prompt = f"""Based on the MODEL {model_num} ({model_name}) you described earlier, predict whether each sample is Malignant (M) or Benign (B).

TASK:
Analyze the feature values for each sample and determine whether it represents a malignant or benign cancer.

RESPONSE FORMAT:
For each sample, return ONLY the sample number and your prediction, like this:
1: M
2: B
...etc.
No explanation, only the numbered predictions in the exact format shown above.

TEST SAMPLES:
{test_features_str}"""
        response = GeminiClient.generate_gemini_content(prompt, cfg)
        if response:
            log.info(f"RAW GEMINI RESPONSE:\n{repr(response)}")
            log.info(f"MODEL {model_num} ({model_name}) PREDICTIONS:")
            log.info(response)
            # Log the prompt sent to Gemini for debugging
            log.info(f"PROMPT SENT TO GEMINI (length={len(prompt)}):\n{prompt}")
            log.info(f"TEST FEATURES TABLE (length={len(test_features_str)}):\n{test_features_str}")
        # Extract predictions from Gemini's response (use old approach)
        predictions = []
        pattern_strict = r'^\s*(\d+)\s*:\s*(M|B)\s*$'
        
        lines = response.strip().split('\n')
        log.debug(f"Split Gemini response into {len(lines)} lines: {lines}")
        for line in lines:
            match = re.match(pattern_strict, line.strip())
            if match:
                pred = match.group(2).strip()
                predictions.append(pred)
        # If count does not match, try the lenient backup extraction
        if len(predictions) != max_test_samples:
            predictions = []
            pattern_lenient = r'(\d+)\s*:\s*(M|B)'
            matches = re.findall(pattern_lenient, response)
            if matches:
                sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                predictions = [m[1] for m in sorted_matches]
                if len(predictions) > max_test_samples:
                    predictions = predictions[:max_test_samples]

        # Remove empty predictions
        predictions = [p for p in predictions if p]

        # Calculate metrics
        if predictions:
            if len(predictions) == len(y_true):
                # Calculate metrics using CalculationUtils instead of direct sklearn functions
                metrics = CalculationUtils.calculate_classification_metrics(y_true, predictions)
                
                # Generate classification report
                report = classification_report(y_true, predictions)
                
                # Print results
                log.info(f"Accuracy: {metrics['accuracy']:.4f}")
                log.info(f"Macro F1: {metrics['macro_f1']:.4f}")
                log.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
                log.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
                log.info(f"FN: {metrics['false_negatives']}")
                log.info(f"FP: {metrics['false_positives']}")

                return {
                    'metrics': {
                        'accuracy_score': float(metrics['accuracy']),
                        'macro_f1': float(metrics['macro_f1']),
                        'macro_precision': float(metrics['macro_precision']),
                        'macro_recall': float(metrics['macro_recall']),
                        'weighted_f1': float(metrics['weighted_f1']),
                        'weighted_precision': float(metrics['weighted_precision']),
                        'weighted_recall': float(metrics['weighted_recall']),
                        'false_negatives': int(metrics['false_negatives']),
                        'false_positives': int(metrics['false_positives'])
                    },
                    'model_name': model_name,
                    'description': f"Model {model_num} ({model_name}) with accuracy {metrics['accuracy']:.4f}"
                }
            else:
                log.info(f"\nMismatch in number of predictions ({len(predictions)}) and true labels ({len(y_true)})")
        else:
            log.info("\nCould not extract predictions from Gemini's response")
        log.info("\n" + "-"*50 + "\n")
