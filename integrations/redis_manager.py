import hashlib
import redis
from omegaconf import DictConfig
from typing import Dict, List, Any, Optional
import functools
import sys

# Import logger config to ensure logging is set up for all modules
from explora_backend.utils.logger_config import log

def redis_error_handler(operation_description: str):
    """Decorator to log Redis operations and handle errors with fail-fast behavior.
    
    Args:
        operation_description: String describing the Redis operation being performed.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            operation_name = func.__name__
            
            # Always try to get a client - will raise an exception if Redis is down
            client = self._get_client()
            if not client:
                log.error(f"{self._log_prefix}: Cannot perform {operation_name} - {operation_description}, no valid client.")
                # Fail fast with system exit
                sys.exit(1)
                
            try:
                return func(self, client, *args, **kwargs)
            except redis.exceptions.RedisError as e:
                log.error(f"{self._log_prefix}: Redis error during {operation_name} - {operation_description}: {e}")
                # Propagate the exception
                raise
            except Exception as e:
                log.exception(f"{self._log_prefix}: Unexpected error during {operation_name} - {operation_description}: {e}")
                # Propagate the exception
                raise
        return wrapper
    return decorator

class RedisManager:
    """Manager for short-term Redis caching with fail-fast behavior."""
    
    def __init__(self, cache_cfg: DictConfig):
        """Initialize Redis Manager with required configuration.
        
        Args:
            cache_cfg: Redis configuration containing host, port, db details
        """
        self.cfg = cache_cfg
        self.client: Optional[redis.Redis] = None
        self._log_prefix = f"SCache ({self.cfg.redis_host}:{self.cfg.redis_port}/{self.cfg.redis_db})"
        log.info(f"{self._log_prefix}: Initializing short-term Redis cache manager")

    @staticmethod
    def get_file_hash(filepath):
        """Calculates the SHA256 hash of a file."""
        hasher = hashlib.sha256()
        try:
            with open(filepath,'rb') as file:
                while chunk := file.read(8192):
                    hasher.update(chunk)
                return hasher.hexdigest()
        except FileNotFoundError:
            log.error(f"Cannot calculate hash: File not found at {filepath}")
            sys.exit(1)  # Fail fast
        except Exception as e:
            log.error(f"Error calculating hash for {filepath}: {e}")
            sys.exit(1)  # Fail fast

    def _connect(self) -> bool:
        """Connect to Redis server. Fails fast if connection cannot be established."""
        # If already connected and ping works, reuse connection
        if self.client:
            try:
                if self.client.ping():
                    log.debug(f"{self._log_prefix}: Connection already active.")
                    return True
            except redis.exceptions.ConnectionError:
                log.error(f"{self._log_prefix}: Existing connection lost. Forcing reconnect.")
                self.client = None
            except Exception as e:
                log.error(f"{self._log_prefix}: Unexpected error checking existing connection {e}. Forcing reconnect.")
                self.client = None
        
        log.debug(f"{self._log_prefix}: Attempting to connect to Redis at {self.cfg.redis_host}:{self.cfg.redis_port}...")
        try:
            self.client = redis.Redis(
                host = self.cfg.redis_host,
                port = self.cfg.redis_port,
                db = self.cfg.redis_db,
                decode_responses = True,
                socket_timeout = 5,
                socket_connect_timeout = 5
            )

            if self.client.ping():
                log.debug(f"{self._log_prefix}: Connection successful.")
                return True
            else:
                log.error(f"{self._log_prefix}: Connection check (ping) failed after establishing connection.")
                self.client = None
                # Fail fast - Redis is required
                sys.exit(1)
        except redis.exceptions.ConnectionError as e:
            log.error(f"{self._log_prefix}: Connection error: {e}")
            self.client = None
            # Fail fast - Redis is required
            sys.exit(1)
        except Exception as e:
            log.exception(f"{self._log_prefix}: Unexpected error during connection: {e}")
            self.client = None
            # Fail fast - Redis is required
            sys.exit(1)
            
    def _get_client(self) -> Optional[redis.Redis]:
        """Get a valid Redis client or exit if not possible."""
        if self.client is None:
            if not self._connect():
                log.error(f"{self._log_prefix}: Failed to establish connection to Redis")
                sys.exit(1)  # Fail fast - Redis is required
            elif self.client is None:
                log.error(f"{self._log_prefix}: _connect reported success but client is still None")
                sys.exit(1)  # Fail fast - Redis is required
            
        # Check if existing connection is still working
        try:
            if self.client and self.client.ping():
                return self.client
            else:
                log.warning(f"{self._log_prefix}: Connection lost before returning client. Attempting reconnect.")
                if self._connect():
                    return self.client
                else:
                    log.error(f"{self._log_prefix}: Reconnect failed")
                    sys.exit(1)  # Fail fast - Redis is required
        except redis.exceptions.ConnectionError:
            log.warning(f"{self._log_prefix}: Connection lost before returning client (ping failed). Attempting reconnect.")
            if self._connect():
                return self.client
            else:
                log.error(f"{self._log_prefix}: Reconnect failed")
                sys.exit(1)  # Fail fast - Redis is required
        except Exception as e:
            log.exception(f"{self._log_prefix}: Unexpected error checking connection: {e}")
            sys.exit(1)  # Fail fast - Redis is required

    def check_connection(self) -> bool:
        """Check if Redis connection is working. Exits if connection fails."""
        try:
            if self.client and self.client.ping():
                log.debug(f"{self._log_prefix}: Connection is active")
                return True
            else:
                log.warning(f"{self._log_prefix}: No connection. Attempting connect.")
                return self._connect()  # _connect handles fail-fast behavior
        except redis.exceptions.ConnectionError:
            log.warning(f"{self._log_prefix}: Connection lost (ping failed). Attempting reconnect.")
            if self._connect():
                return True
            else:
                log.error(f"{self._log_prefix}: Reconnect failed")
                sys.exit(1)  # Fail fast - Redis is required
        except Exception as e:
            log.exception(f"{self._log_prefix}: Unexpected error checking connection: {e}")
            sys.exit(1)  # Fail fast - Redis is required
    
    @redis_error_handler("storing key-value pairs")
    def store_values(self, client: redis.Redis, data: Dict[str, str]) -> bool:        
        """Store multiple key-value pairs in Redis."""
        if not data:
            log.info(f"{self._log_prefix}: No data to store. Skipping.")
            return True

        pipe = client.pipeline()
        
        for key, value in data.items():
            pipe.set(str(key), str(value))

        ttl = self.cfg.get('ttl_seconds')
        if ttl is not None and ttl > 0:
            for key in data.keys():
                pipe.expire(str(key), ttl)

        pipe.execute()
        log.info(f"{self._log_prefix}: Successfully stored {len(data)} key-value pairs")
        return True

    @redis_error_handler("retrieving values")
    def get_values(self, client: redis.Redis, keys: List[str]) -> Dict[str, str]:
        """Retrieve multiple values from Redis."""
        if not keys:
            log.warning(f"{self._log_prefix}: No keys requested. Returning empty dict.")
            return {}
        
        result = {}
        pipe = client.pipeline()
        for key in keys:
            pipe.get(str(key))
        values = pipe.execute()
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = value
        log.info(f"{self._log_prefix}: Retrieved {len(result)}/{len(keys)} keys")
        return result

    @redis_error_handler("deleting keys")
    def delete_keys(self, client: redis.Redis, keys: List[str]) -> int:
        """Delete multiple keys from Redis."""
        if not keys:
            log.warning(f"{self._log_prefix}: No keys to delete. Skipping.")
            return 0
        
        result = client.delete(*[str(key) for key in keys])
        log.info(f"{self._log_prefix}: Deleted {result}/{len(keys)} keys.")
        return result
    
    @redis_error_handler("appending value to key")
    def append_value(self, client: redis.Redis, key: str, value: str) -> bool:
        """Appends a string value to a key in Redis. Creates key if it doesn't exist."""
        if not key or value is None: # Basic validation
            log.warning(f"{self._log_prefix}: Invalid key or value for append. Key: '{key}', Value: '{value}'")
            return False
        # The append command returns the length of the string after the append operation.
        # We consider it successful if the length is positive.
        new_length = client.append(str(key), str(value))
        log.debug(f"{self._log_prefix}: Appended to key '{key}'. New length: {new_length}")
        return new_length > 0 # Success if length increased
    
    @redis_error_handler("tracking key")
    def track_key(self, client: redis.Redis, username: str, key: str) -> bool:
        """Track a key in the user's key registry.
        
        Args:
            client: Redis client
            username: The username to track keys for
            key: The key to add to the registry
            
        Returns:
            bool: True if successful
        """
        registry_key = f"{username}:all_keys"
        
        # First check if the key is already in the registry
        registry_value = client.get(registry_key)
        if registry_value:
            keys = registry_value.split(',')
            if key in keys:
                # Already tracked
                return True
            
            # Add the key to the registry
            updated_registry = f"{registry_value},{key}"
        else:
            # Create new registry
            updated_registry = key
            
        # Store the updated registry
        client.set(registry_key, updated_registry)
        log.debug(f"{self._log_prefix}: Added key '{key}' to user '{username}' registry")
        return True
    
    def track_keys(self, username: str, keys: List[str]) -> bool:
        """Track multiple keys in the user's key registry.
        
        Args:
            username: The username to track keys for
            keys: List of keys to add to the registry
            
        Returns:
            bool: True if successful
        """
        client = self._get_client()
        success = True
        
        for key in keys:
            if not self.track_key(username, key):  # Removed client parameter as it's added by the decorator
                success = False
                
        return success
    
    @redis_error_handler("retrieving user keys")
    def get_user_keys(self, client: redis.Redis, username: str) -> List[str]:
        """Get all keys tracked for a specific user.
        
        Args:
            client: Redis client
            username: The username to get keys for
            
        Returns:
            List[str]: List of keys for the user
        """
        registry_key = f"{username}:all_keys"
        registry_value = client.get(registry_key)
        
        if registry_value:
            return [key for key in registry_value.split(',') if key.strip()]
        
        return []
    
    @redis_error_handler("deleting run data")
    def delete_run_data(self, client: redis.Redis, username: str, run_name: str, filepath: str = None) -> bool:
        """Delete all Redis data associated with a specific run name.
        
        Args:
            client: Redis client
            username: The username that owns the data
            run_name: The run name to delete data for
            filepath: Optional filepath to target specific file data
            
        Returns:
            bool: True if deletion was successful
        """
        log.warning(f"{self._log_prefix}: Attempting to delete all data for run '{run_name}'...")
        
        # Get all keys for this user
        all_keys = self.get_user_keys(username)  # Removed client parameter
        
        # Filter keys for this run name
        if filepath:
            pattern = f"{username}:{filepath}:{run_name}:"
            keys_to_delete = [key for key in all_keys if key.startswith(pattern)]
            log.info(f"{self._log_prefix}: Found {len(keys_to_delete)} keys for run '{run_name}' with filepath '{filepath}'")
        else:
            pattern = f"{username}:"
            keys_to_delete = [key for key in all_keys if f":{run_name}:" in key]
            log.info(f"{self._log_prefix}: Found {len(keys_to_delete)} keys for run '{run_name}' across all files")
        
        # Delete the keys
        if keys_to_delete:
            deleted_count = client.delete(*keys_to_delete)
            log.info(f"{self._log_prefix}: Deleted {deleted_count}/{len(keys_to_delete)} keys for run '{run_name}'")
            
            # Update the registry to remove deleted keys
            remaining_keys = [key for key in all_keys if key not in keys_to_delete]
            updated_registry = ','.join(remaining_keys)
            client.set(f"{username}:all_keys", updated_registry)
        else:
            log.warning(f"{self._log_prefix}: No keys found for run '{run_name}'")
        
        # Update the run names registry to remove this name
        run_names_key = f"{username}:all_run_names"
        run_names_value = client.get(run_names_key)
        
        if run_names_value:
            existing_names = run_names_value.split(',')
            existing_names = [name for name in existing_names if name.strip() and name != run_name]
            updated_names = ','.join(existing_names)
            client.set(run_names_key, updated_names)
            log.info(f"{self._log_prefix}: Removed '{run_name}' from run names registry")
        
        log.warning(f"{self._log_prefix}: Successfully deleted data for run '{run_name}'")
        return True
    
    def delete_run(self, username: str, run_name: str, filepath: str = None) -> bool:
        """Public method to delete all data for a run.
        
        Args:
            username: The username that owns the data
            run_name: The run name to delete data for
            filepath: Optional filepath to target specific file data
            
        Returns:
            bool: True if deletion was successful
        """
        client = self._get_client()  # Still need this to check connection
        return self.delete_run_data(username, run_name, filepath)  # Removed client parameter
    
    def function_call_with_redis(self, function, args, cfg: DictConfig, run_name: Optional[str] = None) -> List[Any]:
        """Call a function and cache the result in redis."""
        rn = run_name or cfg.run_name
        if rn is not None:
            cache_key = f"{cfg.username}:{rn}:{function.__name__}"
        else:
            cache_key = f"{cfg.username}:{function.__name__}"
        result = function(*args) if args is not None else function()
        if not isinstance(result, dict):
            log.error(f"{self._log_prefix}: Expected dict from {function.__name__}, got {type(result)}")
            raise TypeError(f"Expected dict result, got {type(result)}")
        self.store_values(result)
        self.track_keys(cfg.username, list(result.keys()))
        return list(result.values())

    def retrieve_or_call(self, function, args, cfg: DictConfig, run_name: Optional[str] = None) -> List[Any]:
        """Retrieve a result from cache or call a function and store it in cache."""
        rn = run_name or cfg.run_name
        cache_key = f"{cfg.username}:{rn}:{function.__name__}"
        cached = self.get_values([cache_key])
        if cache_key in cached:
            log.info(f"{self._log_prefix}: Retrieved {cache_key} from cache")
            return cached[cache_key]
        log.info(f"{self._log_prefix}: Cache miss for key {cache_key}, calling function {function.__name__}")
        result = function(*args) if args is not None else function()
        if not isinstance(result, dict):
            log.error(f"{self._log_prefix}: Expected dict from {function.__name__}, got {type(result)}")
            raise TypeError(f"Expected dict result, got {type(result)}")
        self.store_values(result)
        self.track_keys(cfg.username, list(result.keys()))
        return list(result.values())