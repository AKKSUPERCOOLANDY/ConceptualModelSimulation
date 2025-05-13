import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import time
from omegaconf import DictConfig
from typing import Optional

from explora_backend.integrations.redis_manager import RedisManager
from explora_backend.utils.logger_config import log

class GeminiClient:
    # --- Configuration for Key Rotation and Retries ---
    # Assumes API keys are stored in GOOGLE_API_KEY_1 and GOOGLE_API_KEY_2 env vars
    _NUM_API_KEYS = 4
    _current_key_index = 0
    _MAX_RETRIES = 60
    _RETRY_DELAY_SECONDS = 1
    # ---

    def generate_gemini_content(prompt: str, cfg: DictConfig, model_override: Optional[str] = None) -> Optional[str]:
        """
        Generates content using a Gemini model, handling API key rotation and rate limit retries.

        Args:
            prompt: The text prompt to send to the model.
            cfg: The Hydra configuration object (used for default model name).
            model_override: Optional. If provided, overrides the model name from cfg.

        Returns:
            The generated text content as a string, or None if an error occurs
            after retries or if the prompt is blocked.
        """
        # 1. Determine API Key to Use (Rotation + Fallback)
        key_index_to_use = GeminiClient._current_key_index
        api_key_var = f"GOOGLE_API_KEY_{key_index_to_use + 1}"
        api_key = os.getenv(api_key_var)

        if not api_key:
            log.warning(f"API Key variable {api_key_var} not found. Falling back to GOOGLE_API_KEY.")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                log.error(f"No Google API Key found in environment variables ({api_key_var} or GOOGLE_API_KEY). Cannot call Gemini.")
                return None
        else:
            # Rotate index only if the numbered key was found and used
            GeminiClient._current_key_index = (GeminiClient._current_key_index + 1) % GeminiClient._NUM_API_KEYS
            log.debug(f"Using API Key from {api_key_var}")


        # 2. Determine Model Name
        # Use override if provided, otherwise use config, default to a known flash model if needed
        model_name = model_override if model_override else cfg.gemini.model

        # Ensure the model name doesn't have the "models/" prefix for the library
        if model_name.startswith("models/"):
            model_name = model_name.split("/")[-1]

        log.debug(f"Attempting Gemini call with model '{model_name}' (Using key ending with ...{api_key[-4:]})")

        # 3. Configure API Key for this call
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            log.exception(f"Failed to configure Gemini API Key: {e}")
            return None # Cannot proceed without configuration

        # 4. Get Model Instance
        try:
            model = genai.GenerativeModel(model_name)
        except Exception as e:
            log.exception(f"Failed to instantiate Gemini model '{model_name}': {e}")
            return None

        # 5. Call API with Retry Logic
        for attempt in range(GeminiClient._MAX_RETRIES):
            try:
                # Use temperature from config if available
                generation_config = None
                if hasattr(cfg.gemini, 'temperature'):
                    generation_config = genai.GenerationConfig(temperature=cfg.gemini.temperature)
                    log.debug(f"Using temperature {cfg.gemini.temperature} for generation")
                
                # Generate content with the specified temperature if available
                response = model.generate_content(
                    prompt, 
                    generation_config=generation_config
                )
                log.debug(f"Gemini call successful on attempt {attempt + 1}")

                # Check for empty/blocked response - accessing text raises error sometimes
                if not response.candidates:
                    log.warning(f"Gemini response has no candidates (potentially blocked).")
                    try:
                        # Log block reason if available
                        block_reason = response.prompt_feedback.block_reason
                        log.warning(f"Prompt blocked due to: {block_reason}")
                    except Exception:
                        log.warning("Could not determine block reason from prompt_feedback.")
                    return None # Treat blocked prompt as failure

                # Safely access text
                try:
                    response_text = response.text
                    return response_text
                except ValueError as ve:
                    # Handle cases where accessing .text fails (e.g., blocked content)
                    log.warning(f"Could not extract text from Gemini response (attempt {attempt+1}): {ve}")
                    log.warning("Response details: %s", response) # Log full response for debugging
                    return None # Treat as failure


            except google_exceptions.ResourceExhausted as e:
                # Only log at debug level for individual rate limits to keep output clean
                if cfg.verbose:
                    log.debug(f"Rate limit hit on attempt {attempt + 1}/{GeminiClient._MAX_RETRIES} for model {model_name}. Retrying in {GeminiClient._RETRY_DELAY_SECONDS}s...")
                if attempt < GeminiClient._MAX_RETRIES - 1:
                    time.sleep(GeminiClient._RETRY_DELAY_SECONDS)
                else:
                    log.error(f"Max retries ({GeminiClient._MAX_RETRIES}) exceeded for rate limit error. Giving up.")
                    return None # Max retries exceeded

            except Exception as e:
                # Includes potential API errors like InvalidArgument, PermissionDenied etc.
                log.exception(f"An unexpected error occurred during Gemini API call on attempt {attempt + 1}: {e}")
                # For unexpected errors, typically don't retry immediately, just fail.
                return None

        # Fallback return if loop completes unexpectedly
        log.error("Exited Gemini retry loop without success or expected error handling.")
        return None