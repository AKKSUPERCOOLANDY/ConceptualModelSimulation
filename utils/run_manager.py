from explora_backend.integrations.redis_manager import RedisManager
from omegaconf import DictConfig
from typing import List, Optional
from explora_backend.utils.logger_config import log

class RunManager:
    @staticmethod
    def commit_run_for_user(run_name: str, cache_cfg: DictConfig) -> bool:
        """
        Commit a run name for a user into Redis. Maintains a list of all runs for the user.
        Username is automatically retrieved from cache_cfg.
        """
        redis_manager = RedisManager(cache_cfg)
        client = redis_manager._get_client()
        username = cache_cfg.get('username', 'default_user')
        registry_key = f"{username}:all_run_names"
        try:
            existing_value = client.get(registry_key)
            if existing_value:
                runs = set(existing_value.split(','))
            else:
                runs = set()
            runs.add(run_name)
            client.set(registry_key, ','.join(sorted(runs)))
            log.info(f"Committed run '{run_name}' for user '{username}' in Redis.")
            return True
        except Exception as e:
            log.error(f"Failed to commit run '{run_name}' for user '{username}': {e}")
            return False

    @staticmethod
    def get_runs_for_user(cache_cfg: DictConfig) -> List[str]:
        """
        Retrieve all committed run names for a user from Redis.
        Username is automatically retrieved from cache_cfg.
        """
        redis_manager = RedisManager(cache_cfg)
        client = redis_manager._get_client()
        username = cache_cfg.get('username', 'default_user')
        registry_key = f"{username}:all_run_names"
        try:
            existing_value = client.get(registry_key)
            if existing_value:
                return [run for run in existing_value.split(',') if run.strip()]
            else:
                return []
        except Exception as e:
            log.error(f"Failed to retrieve runs for user '{username}': {e}")
            return []

    @staticmethod
    def delete_run_for_user(run_name: str, cache_cfg: DictConfig) -> bool:
        """
        Delete all data for a user's run using RedisManager.delete_run.
        """
        redis_manager = RedisManager(cache_cfg)
        username = cache_cfg.get('username', 'default_user')
        return redis_manager.delete_run(username, run_name)

    @staticmethod
    def resolve_run_name(cfg: DictConfig) -> str:
        """
        Determine the run name to use, respecting force_run_name_in_config if set.
        If force_run_name_in_config is True and run_name is present in config, use that value.
        Otherwise, always prompt the user for a unique run name.
        """
        cache_cfg = cfg['cache']
        force_from_config = cfg.get('force_run_name_in_config', False)
        autodelete = cfg.get('autodelete_duplicate_run', False)
        config_run_name = cfg.get('run_name', None)
        if force_from_config and config_run_name:
            # Check uniqueness
            existing_runs = RunManager.get_runs_for_user(cache_cfg)
            if config_run_name in existing_runs:
                if autodelete:
                    log.info(f"Run name '{config_run_name}' already exists. Autodeleting...")
                    RunManager.delete_run_for_user(config_run_name, cache_cfg)
                else:
                    while True:
                        resp = input(f"Run name '{config_run_name}' already exists. Delete previous run? (y/n): ").strip().lower()
                        if resp == 'y':
                            RunManager.delete_run_for_user(config_run_name, cache_cfg)
                            break
                        elif resp == 'n':
                            log.error("Please update your config with a new run_name or set force_run_name_in_config to false.")
                            exit(1)
                log.info(f"Using run name from config: {config_run_name}")
            return config_run_name
        # Otherwise always prompt
        return RunManager.prompt_for_unique_run_name(cache_cfg, autodelete=autodelete)

    @staticmethod
    def prompt_for_unique_run_name(cache_cfg: DictConfig, autodelete: bool = False) -> str:
        existing_runs = RunManager.get_runs_for_user(cache_cfg)
        while True:
            run_name = input(f"Enter a unique run name (existing: {existing_runs}): ").strip()
            if not run_name:
                log.warning("Run name cannot be empty.")
                continue
            if run_name in existing_runs:
                if autodelete:
                    log.info(f"Run name '{run_name}' already exists. Autodeleting...")
                    RunManager.delete_run_for_user(run_name, cache_cfg)
                    return run_name
                resp = input(f"Run name '{run_name}' already exists. Delete previous run? (y/n): ").strip().lower()
                if resp == 'y':
                    RunManager.delete_run_for_user(run_name, cache_cfg)
                    return run_name
                elif resp == 'n':
                    continue
                else:
                    log.warning("Please answer 'y' or 'n'.")
                    continue
            return run_name