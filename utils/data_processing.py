import polars as pl
from omegaconf import DictConfig
from typing import Dict, List, Any, Optional
import os

from explora_backend.integrations.gemini_client import GeminiClient
from explora_backend.utils.logger_config import log

class DataPreprocessing:
    def get_dataset_info(df: pl.DataFrame, cfg: DictConfig) -> Optional[Dict[str, str]]:
        prompt = f""" {df.head()}
        {df.sample(n=min(20,len(df)))}
        Instructions: Return the following with newlines in between each answer (make your best guess). Nothing else (ex. numbers, bullet points, etc.)

        1. task in which we should design ML models for (short, concise setence fragment, don't restate question)
        2. target variable column name"""

        log.debug("Target Discovery Prompt:\n%s", prompt)

        response_txt = GeminiClient.generate_gemini_content(prompt, cfg)
        if response_txt is None:
            log.error("Gemini call failed to determine task and/or target column after retries or due to error.")
            return None

        response = response_txt.strip().split('\n')
        if len(response) != 2:
            log.error("Unexpected response format from Gemini:\n%s", response_txt)
            return None
            
        task = response[0].strip()
        target_col = response[1].strip()
        
        if not target_col:
            log.error("Gemini returned an empty response for target column.")
            return None
        if target_col not in df.columns:
            log.error(f"Gemini suggested target column '{target_col}' which is not in the DataFrame columns.")
            log.debug(f"Available columns: {df.columns}")
            return None

        log.info("Gemini identified task: %s", task)
        log.info("Gemini identified target columns: %s", target_col)

        return {"task": task, "target_col": target_col}

    def get_classes_and_bad_columns(df: pl.DataFrame, target_col: str, cfg: DictConfig):
        classes = df[target_col].unique().to_list()
        columns = df.columns
        prompt = (
            f"Instructions: Given a small sample of the dataframe {df.sample(n=min(50,len(df)))}\n"
            f"and the target classes {classes} and all columns {columns}\n"
            "Return two lines:\n"
            "1) comma-separated list of classes that are unclean (dirty targets)\n"
            "2) comma-separated list of columns that are unhelpful or contain invalid data"
        )
        response_txt = GeminiClient.generate_gemini_content(prompt, cfg)
        if response_txt is None:
            log.error("Gemini call failed to determine classes and bad columns.")
            return None
        parts = [line.strip() for line in response_txt.strip().split('\n')]
        dirty_resp = parts[0].split(',') if len(parts) > 0 else []
        bad_cols_resp = parts[1].split(',') if len(parts) > 1 else []
        dirty_targets = [c.strip() for c in dirty_resp if any(c.strip() == cls or c.strip() == str(cls) for cls in classes)]
        clean_targets = [c for c in classes if c not in dirty_targets]
        bad_columns = [c.strip() for c in bad_cols_resp if c.strip() in columns]
        return {"clean_targets": clean_targets,
                "dirty_targets": dirty_targets,
                "bad_columns": bad_columns}

    def clean_data(df: pl.DataFrame, target_col: str, dirty_targets: list = None, bad_columns: list = None) -> pl.DataFrame:
        initial_rows = df.height
        if bad_columns:
            df = df.drop(bad_columns)
            log.info(f"Dropped bad columns: {bad_columns}")
        if dirty_targets:
            df = df.filter(~df[target_col].is_in(dirty_targets))
            log.info(f"Removed rows with dirty targets in '{target_col}': {dirty_targets}")
        before_dupes = df.height
        df = df.unique()
        log.info(f"Removed {before_dupes - df.height} duplicate rows.")
        return df