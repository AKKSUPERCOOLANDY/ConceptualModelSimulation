import polars as pl
from sklearn.model_selection import train_test_split
import json
import os
from explora_backend.scripts.run import Run
from omegaconf import OmegaConf
from explora_backend.utils.run_manager import RunManager
from explora_backend.utils.logger_config import log

def main(csv_name, cfg):
    df = pl.read_csv(f"data/{csv_name}.csv")
    # Strip whitespace from all column names
    df.columns = [col.strip() for col in df.columns]
    # Drop columns with empty names (caused by trailing comma in CSV header)
    if "" in df.columns:
        df = df.drop([""])
    print(f"Loaded columns: {df.columns}")
    log.info(f"Loaded columns: {df.columns}")
    log.info(Run.run_explora(cfg, df))
    return

cfg = OmegaConf.load("explora_backend/config/main.yaml")

# Prompt for a unique run name before running anything else
cache_cfg = cfg['cache']
run_name = RunManager.resolve_run_name(cfg)
cfg['run_name'] = run_name
RunManager.commit_run_for_user(run_name, cache_cfg)

main('breast', cfg)