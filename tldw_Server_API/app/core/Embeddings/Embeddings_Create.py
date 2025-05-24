# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
import os
import time
from functools import wraps
from threading import Lock, Timer
from typing import List, Optional, Dict, Any
#
# 3rd-Party Imports:
import numpy as np
import onnxruntime as ort
import requests
from transformers import AutoTokenizer, AutoModel
import torch
# Make sure optimum is installed if you want on-the-fly ONNX conversion
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    OPTIMUM_AVAILABLE = True
except ImportError:
    ORTModelForFeatureExtraction = None
    OPTIMUM_AVAILABLE = False
#
# Local Imports:
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import get_openai_embeddings_batch
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:
# Global cache for embedding models and a lock for thread-safe access
embedding_models: Dict[str, Any] = {}
embedding_models_lock = Lock()

# Commit hashes for specific model revisions (can be part of model config too)
COMMIT_HASHES = {
    "jinaai/jina-embeddings-v3": "4be32c2f5d65b95e4bcce473545b7883ec8d2edd",
    "Alibaba-NLP/gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
    "dunzhang/setll_en_400M_v5": "2aa5579fcae1c579de199a3866b6e514bbbf5d10"
    # Add other models as needed
}


class HuggingFaceEmbedder:
    def __init__(self, model_identifier: str, model_config: Dict[str, Any], hf_cache_dir: str):
        self.model_identifier = model_identifier  # e.g., "minilm_hf_local"
        self.model_name_or_path = model_config['model_name_or_path']  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
        self.config = model_config
        self.hf_cache_dir = hf_cache_dir

        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.timeout_seconds = self.config.get('unload_timeout_seconds', 300)
        self.max_length = self.config.get('max_length', 512)
        self.revision = self.config.get('revision') or COMMIT_HASHES.get(self.model_name_or_path)

        self.last_used_time = 0
        self.unload_timer = None
        log_counter("huggingface_embedder_init", labels={"model_id": self.model_identifier})

    def load_model(self):
        log_counter("huggingface_model_load_attempt", labels={"model_id": self.model_identifier})
        start_time = time.time()
        if self.model is None:
            logging.info(f"Loading HuggingFace model: {self.model_name_or_path} (ID: {self.model_identifier})")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.config.get('trust_remote_code', True),
                cache_dir=self.hf_cache_dir,
                revision=self.revision
            )
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.config.get('trust_remote_code', True),
                cache_dir=self.hf_cache_dir,
                revision=self.revision
            )
            self.model.to(self.device)
            logging.info(
                f"HuggingFace model {self.model_name_or_path} loaded on {self.device}. Max length: {self.max_length}, Timeout: {self.timeout_seconds}s.")

        self.last_used_time = time.time()
        self.reset_timer()
        load_time = time.time() - start_time
        log_histogram("huggingface_model_load_duration", load_time, labels={"model_id": self.model_identifier})
        log_counter("huggingface_model_load_success", labels={"model_id": self.model_identifier})

    def unload_model(self):
        log_counter("huggingface_model_unload", labels={"model_id": self.model_identifier})
        if self.model is not None:
            logging.info(f"Unloading HuggingFace model {self.model_name_or_path} (ID: {self.model_identifier})")
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
        if self.unload_timer:
            self.unload_timer.cancel()

    def reset_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
        self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
        self.unload_timer.start()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        log_counter("huggingface_create_embeddings_attempt", labels={"model_id": self.model_identifier})
        start_time = time.time()
        self.load_model()

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Common pooling strategy: mean of last hidden states
            embeddings_tensor = outputs.last_hidden_state.mean(dim=1)
        except RuntimeError as e:
            if "Got unsupported ScalarType BFloat16" in str(e) and self.model.dtype == torch.bfloat16:
                logging.warning(
                    f"BFloat16 not supported for {self.model_name_or_path} on {self.device}. Falling back to float32.")
                self.model = self.model.float()  # Convert model
                with torch.no_grad():  # Retry
                    outputs = self.model(**inputs)
                embeddings_tensor = outputs.last_hidden_state.mean(dim=1)
                log_counter("huggingface_bfloat16_fallback", labels={"model_id": self.model_identifier})
            else:
                log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
                logging.error(f"RuntimeError during HuggingFace embedding for {self.model_name_or_path}: {e}",
                              exc_info=True)
                raise
        except Exception as e:
            log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
            logging.error(f"Unexpected error during HuggingFace embedding for {self.model_name_or_path}: {e}",
                          exc_info=True)
            raise

        embedding_time = time.time() - start_time
        log_histogram("huggingface_create_embeddings_duration", embedding_time,
                      labels={"model_id": self.model_identifier})
        log_counter("huggingface_create_embeddings_success", labels={"model_id": self.model_identifier})
        return embeddings_tensor.cpu().float().numpy()


class ONNXEmbedder:
    def __init__(self, model_identifier: str, model_config: Dict[str, Any], onnx_model_storage_dir: str):
        self.model_identifier = model_identifier
        self.original_hf_model_name = model_config['model_name_or_path']  # Original HF name for tokenizer/conversion
        self.config = model_config

        # Construct path for the ONNX model file itself
        # e.g., <onnx_model_storage_dir>/<model_name_from_hf>/<configured_onnx_filename>.onnx
        model_specific_onnx_dir = os.path.join(onnx_model_storage_dir, self.original_hf_model_name.split('/')[-1])
        self.onnx_model_filename = self.config.get('onnx_filename',
                                                   f"{self.original_hf_model_name.split('/')[-1]}.onnx")
        self.onnx_model_path = os.path.join(model_specific_onnx_dir, self.onnx_model_filename)

        self.tokenizer_cache_dir = model_specific_onnx_dir  # Store tokenizer with its ONNX model
        os.makedirs(self.tokenizer_cache_dir, exist_ok=True)

        self.revision = self.config.get('revision') or COMMIT_HASHES.get(self.original_hf_model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.original_hf_model_name,
            trust_remote_code=self.config.get('trust_remote_code', True),
            cache_dir=self.tokenizer_cache_dir,
            revision=self.revision
        )
        self.session = None
        self.timeout_seconds = self.config.get('unload_timeout_seconds', 300)
        self.max_length = self.config.get('max_length', 512)

        self.last_used_time = 0
        self.unload_timer = None
        self.device_providers = self.config.get('onnx_providers',
                                                ['CPUExecutionProvider'])  # e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']

        log_counter("onnx_embedder_init", labels={"model_id": self.model_identifier})

    def _ensure_model_converted(self):
        if not os.path.exists(self.onnx_model_path):
            if not OPTIMUM_AVAILABLE or ORTModelForFeatureExtraction is None:
                logging.error("`optimum` library is not available. Cannot convert model to ONNX on-the-fly.")
                raise FileNotFoundError(
                    f"ONNX model not found at {self.onnx_model_path} and auto-conversion tool (optimum) is missing.")

            logging.warning(
                f"ONNX model not found at {self.onnx_model_path}. Attempting to convert {self.original_hf_model_name}...")
            try:
                os.makedirs(os.path.dirname(self.onnx_model_path), exist_ok=True)
                # Exporting directly to the final directory might be simpler if optimum allows it,
                # otherwise, export to a temp dir within tokenizer_cache_dir then move.
                # The task might need to be specified for some models, e.g., 'feature-extraction'
                # FIXME - Function doesn't exist
                ort_model = ORTModelForFeatureExtraction.from_pretrained(
                    self.original_hf_model_name,
                    export=True,
                    trust_remote_code=self.config.get('trust_remote_code', True),
                    revision=self.revision,
                    cache_dir=self.tokenizer_cache_dir  # For downloading original HF model
                )
                # save_pretrained will save 'model.onnx' and other files in the specified directory.
                # We want to save it in a way that self.onnx_model_path is correct.
                # Let's save to a temporary subdirectory and then move the model.onnx
                temp_export_location = os.path.join(self.tokenizer_cache_dir, "_temp_onnx_export")
                ort_model.save_pretrained(temp_export_location)

                exported_onnx_file = os.path.join(temp_export_location, "model.onnx")  # Default name by optimum
                if os.path.exists(exported_onnx_file):
                    os.rename(exported_onnx_file, self.onnx_model_path)
                    logging.info(
                        f"ONNX model for {self.original_hf_model_name} successfully exported to {self.onnx_model_path}")
                    # Clean up: remove other files from temp_export_location and the dir itself
                    for item in os.listdir(temp_export_location):
                        item_path = os.path.join(temp_export_location, item)
                        if os.path.isfile(item_path): os.remove(item_path)
                    os.rmdir(temp_export_location)
                else:
                    raise FileNotFoundError(
                        f"ONNX 'model.onnx' not found in {temp_export_location} after export attempt.")

            except Exception as e:
                logging.error(f"Failed to export/download ONNX model for {self.original_hf_model_name}: {e}",
                              exc_info=True)
                raise FileNotFoundError(f"ONNX model not found at {self.onnx_model_path} and auto-conversion failed.")
        logging.info(
            f"ONNX Model {self.original_hf_model_name} (ID: {self.model_identifier}) ready at {self.onnx_model_path}")

    def load_model(self):
        log_counter("onnx_model_load_attempt", labels={"model_id": self.model_identifier})
        start_time = time.time()
        if self.session is None:
            self._ensure_model_converted()  # Check and convert if needed
            logging.info(f"Loading ONNX model from {self.onnx_model_path} with providers: {self.device_providers}")
            self.session = ort.InferenceSession(self.onnx_model_path, providers=self.device_providers)
            logging.info(
                f"ONNX model {self.original_hf_model_name} loaded. Max length: {self.max_length}, Timeout: {self.timeout_seconds}s.")

        self.last_used_time = time.time()
        self.reset_timer()
        load_time = time.time() - start_time
        log_histogram("onnx_model_load_duration", load_time, labels={"model_id": self.model_identifier})
        log_counter("onnx_model_load_success", labels={"model_id": self.model_identifier})

    def unload_model(self):
        log_counter("onnx_model_unload", labels={"model_id": self.model_identifier})
        if self.session is not None:
            logging.info(f"Unloading ONNX model {self.original_hf_model_name} (ID: {self.model_identifier})")
            del self.session  # Release C++ resources
            self.session = None
        if self.unload_timer:
            self.unload_timer.cancel()

    def reset_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
        self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
        self.unload_timer.start()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        log_counter("onnx_create_embeddings_attempt", labels={"model_id": self.model_identifier})
        start_time = time.time()
        self.load_model()

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="np",  # ONNX runtime prefers numpy
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            if "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            ort_outputs = self.session.run(None, ort_inputs)

            # Common pooling: Mean pooling of the last hidden state, considering attention mask
            last_hidden_state = ort_outputs[0]
            input_mask_expanded = np.expand_dims(ort_inputs["attention_mask"], -1).astype(float)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = np.maximum(np.sum(input_mask_expanded, 1), 1e-9)  # Avoid division by zero
            embeddings_np = sum_embeddings / sum_mask

        except Exception as e:
            log_counter("onnx_create_embeddings_failure", labels={"model_id": self.model_identifier})
            logging.error(f"Error creating embeddings with ONNX model {self.original_hf_model_name}: {str(e)}",
                          exc_info=True)
            raise

        embedding_time = time.time() - start_time
        log_histogram("onnx_create_embeddings_duration", embedding_time, labels={"model_id": self.model_identifier})
        log_counter("onnx_create_embeddings_success", labels={"model_id": self.model_identifier})
        return embeddings_np


class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assumes user_embedding_config is the second argument (index 1) for create_embeddings_batch/create_embedding
            # Or find it in kwargs
            user_embedding_config = None
            if len(args) > 1 and isinstance(args[1], dict) and 'user_embedding_config' in args[1]:
                user_embedding_config = args[1]
            elif 'user_embedding_config' in kwargs and isinstance(kwargs['user_embedding_config'], dict) and 'user_embedding_config' in kwargs[
                'user_embedding_config']:
                user_embedding_config = kwargs['user_embedding_config']

            if user_embedding_config:
                rl_cfg = user_embedding_config.get('user_embedding_config', {}).get('rate_limiter', {})
                # Allow dynamic override of rate limiter settings from config if desired
                # For simplicity, we assume RateLimiter is initialized with static values for now.
                # If dynamic, RateLimiter would need to access config or be re-created.
                # Here, we'll just use the initially configured values.
                pass  # Using self.max_calls, self.period set at __init__

            with self.lock:
                now = time.time()
                self.calls = [call_time for call_time in self.calls if call_time > now - self.period]
                if len(self.calls) >= self.max_calls:
                    time_to_wait = (self.calls[0] + self.period) - now
                    if time_to_wait > 0:
                        logging.info(
                            f"Rate limit for {func.__name__} hit. Sleeping for {time_to_wait:.2f}s. (Calls: {len(self.calls)}/{self.max_calls} per {self.period}s)")
                        time.sleep(time_to_wait)
                    now = time.time()  # Re-evaluate time
                    self.calls = [call_time for call_time in self.calls if call_time > now - self.period]
                self.calls.append(now)
            return func(*args, **kwargs)

        return wrapper


def exponential_backoff(default_max_retries=3, default_base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_embedding_config = None
            if len(args) > 1 and isinstance(args[1], dict) and 'user_embedding_config' in args[1]:
                user_embedding_config = args[1]
            elif 'user_embedding_config' in kwargs and isinstance(kwargs['user_embedding_config'], dict) and 'user_embedding_config' in kwargs[
                'user_embedding_config']:
                user_embedding_config = kwargs['user_embedding_config']

            retry_cfg = user_embedding_config.get('user_embedding_config', {}).get('retry_config', {}) if user_embedding_config else {}
            max_retries = retry_cfg.get('max_retries', default_max_retries)
            base_delay = retry_cfg.get('base_delay', default_base_delay)

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:  # More general for network issues
                    is_retryable_http = False
                    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                        # Retry on 5xx server errors or 429 Too Many Requests
                        if 500 <= e.response.status_code < 600 or e.response.status_code == 429:
                            is_retryable_http = True

                    if not is_retryable_http and not isinstance(e, (requests.exceptions.Timeout,
                                                                    requests.exceptions.ConnectionError)):
                        logging.error(f"Non-retryable RequestException for {func.__name__}: {str(e)}")
                        raise  # Don't retry for non-transient request errors unless explicitly http retryable

                    if attempt == max_retries - 1:
                        logging.error(
                            f"Final attempt ({max_retries}) failed for {func.__name__} due to RequestException: {str(e)}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} for {func.__name__} failed with RequestException. Retrying in {delay}s. Error: {str(e)}")
                    time.sleep(delay)
                except Exception as e:  # Catch other potentially transient errors
                    # Be cautious about retrying all exceptions. For now, keeping broad.
                    if attempt == max_retries - 1:
                        logging.error(f"Final attempt ({max_retries}) failed for {func.__name__}. Error: {str(e)}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} for {func.__name__} failed. Retrying in {delay}s. Error: {str(e)}")
                    time.sleep(delay)

        return wrapper

    return decorator


# Example of how user_embedding_config['user_embedding_config'] should be structured:
# user_embedding_config = {
#   "embedding_config": {
#     "default_model_id": "minilm_hf", # Identifier for the default model config to use
#     "model_storage_base_dir": "/path/to/your/models/embedding_models/", # Point 9
#     "rate_limiter": {"max_calls": 20, "period": 60},
#     "retry_config": {"max_retries": 3, "base_delay": 1},
#     "models": {
#       "minilm_hf": { # This is a model_id
#         "provider": "huggingface",
#         "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
#         "hf_cache_dir_subpath": "huggingface_cache", # Subpath under model_storage_base_dir
#         "max_length": 384, # Point 10: Model-specific max length
#         "unload_timeout_seconds": 180,
#         "trust_remote_code": True # Optional, default True
#       },
#       "gte_onnx": {
#         "provider": "onnx",
#         "model_name_or_path": "Alibaba-NLP/gte-large-en-v1.5", # Original HF name
#         "onnx_storage_dir_subpath": "onnx_models", # Subpath under model_storage_base_dir
#         "onnx_filename": "gte-large-custom.onnx", # Optional: custom name for the .onnx file
#         "max_length": 512,
#         "unload_timeout_seconds": 300,
#         "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"] # Example
#       },
#       "openai_ada_v2": {
#         "provider": "openai",
#         "model_name_or_path": "text-embedding-ada-002", # OpenAI's model name
#         # max_length for OpenAI is more about their token limits, client doesn't usually enforce
#       },
#       "local_api_model": {
#         "provider": "local_api",
#         "model_name_or_path": "some_model_served_locally", # Name your API expects
#         "api_url": "http://localhost:8001/embed",
#         "api_key": "your_local_api_key_if_any" # Handle securely
#       }
#     }
#   }
#   # ... other application settings
# }


# Apply decorators with default settings. These will be further configured by user_embedding_config inside the wrapper.
@exponential_backoff()
@RateLimiter(max_calls=20, period=60)  # Default values, can be effectively overridden by config inside
def create_embeddings_batch(texts: List[str],
                            user_embedding_config: Dict[str, Any],
                            model_id_override: Optional[str] = None
                            ) -> List[List[float]]:
    """
    Creates embeddings for a batch of texts.
    Requires user_embedding_config, which contains 'user_embedding_config' with model definitions.
    Allows overriding the default model_id from config.
    """
    global embedding_models
    global embedding_models_lock

    if not texts:
        return []

    emb_cfg = user_embedding_config.get('user_embedding_config')
    if not emb_cfg:
        logging.error("`user_embedding_config` not found in `user_embedding_config`.")
        raise ValueError("`user_embedding_config` missing from application configuration.")

    model_id_to_use = model_id_override if model_id_override else emb_cfg.get('default_model_id')
    if not model_id_to_use:
        logging.error("No `model_id` specified and no `default_model_id` found in `embedding_config`.")
        raise ValueError("Embedding model ID not specified.")

    model_spec = emb_cfg.get('models', {}).get(model_id_to_use)
    if not model_spec:
        logging.error(f"Configuration for `model_id` '{model_id_to_use}' not found in `embedding_config.models`.")
        raise ValueError(f"Invalid `model_id`: {model_id_to_use}")

    provider = model_spec.get('provider')
    model_name = model_spec.get('model_name_or_path')  # HF name, OpenAI name, local model name

    # Base directory for all models (HuggingFace cache, ONNX models)
    # Point 9: Get base model storage path from config
    model_storage_base_dir = emb_cfg.get('model_storage_base_dir', './embedding_models_data/')
    os.makedirs(model_storage_base_dir, exist_ok=True)

    log_counter("create_embeddings_batch_attempt", labels={"provider": provider, "model_id": model_id_to_use})
    start_time = time.time()
    embeddings_list: List[List[float]] = []

    try:
        if provider.lower() == 'huggingface':
            embedder_instance = None
            with embedding_models_lock:  # Point 7: Thread lock for cache access
                if model_id_to_use not in embedding_models:
                    logging.info(f"HuggingFace model ID {model_id_to_use} not in cache. Initializing.")
                    hf_cache_dir = os.path.join(model_storage_base_dir,
                                                model_spec.get('hf_cache_dir_subpath', 'huggingface_cache'))
                    os.makedirs(hf_cache_dir, exist_ok=True)
                    embedding_models[model_id_to_use] = HuggingFaceEmbedder(model_id_to_use, model_spec, str(hf_cache_dir))
                embedder_instance = embedding_models[model_id_to_use]

            if embedder_instance:
                embeddings_np = embedder_instance.create_embeddings(texts)
                embeddings_list = embeddings_np.tolist()

        elif provider.lower() == 'onnx':
            embedder_instance = None
            with embedding_models_lock:  # Point 7
                if model_id_to_use not in embedding_models:
                    logging.info(f"ONNX model ID {model_id_to_use} not in cache. Initializing.")
                    onnx_storage_dir = os.path.join(model_storage_base_dir,
                                                    model_spec.get('onnx_storage_dir_subpath', 'onnx_models'))
                    os.makedirs(onnx_storage_dir, exist_ok=True)
                    embedding_models[model_id_to_use] = ONNXEmbedder(model_id_to_use, model_spec, str(onnx_storage_dir))
                embedder_instance = embedding_models[model_id_to_use]

            if embedder_instance:
                embeddings_np = embedder_instance.create_embeddings(
                    texts)  # ONNXEmbedder handles its own config for conversion
                embeddings_list = embeddings_np.tolist()

        elif provider.lower() == 'openai':
            logging.debug(f"Creating embeddings for {len(texts)} texts via OpenAI API with model {model_name}")
            # Point 3: Ensure get_openai_embeddings_batch is used and properly implemented
            # It should handle API key from its own configuration or environment variables
            if not callable(get_openai_embeddings_batch):
                logging.error(
                    "`get_openai_embeddings_batch` is not available or not callable. OpenAI batch processing failed.")
                raise NotImplementedError("OpenAI batch embedding function is not properly set up.")
            embeddings_list = get_openai_embeddings_batch(texts, model=model_name, app_config=user_embedding_config)


        elif provider.lower() == 'local_api':
            api_url = model_spec.get('api_url')
            api_key = model_spec.get('api_key')  # Securely managed
            if not api_url:
                raise ValueError(f"Local API URL not configured for model_id '{model_id_to_use}'.")

            logging.debug(f"Creating {len(texts)} embeddings via local API ({api_url}) with model {model_name}")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {"texts": texts, "model": model_name}  # Adjust payload as per your local API spec

            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            if 'embeddings' not in response_data or not isinstance(response_data['embeddings'], list):
                logging.error(f"Local API at {api_url} returned unexpected data format: {response_data}")
                raise ValueError("Local API embedding response format error.")
            embeddings_list = response_data['embeddings']

        else:
            raise ValueError(f"Unsupported embedding provider: {provider} for model_id '{model_id_to_use}'")

        embedding_time = time.time() - start_time
        log_histogram("create_embeddings_batch_duration", embedding_time,
                      labels={"provider": provider, "model_id": model_id_to_use})
        log_counter("create_embeddings_batch_success", labels={"provider": provider, "model_id": model_id_to_use})
        return embeddings_list

    except Exception as e:
        log_counter("create_embeddings_batch_error",
                    labels={"provider": provider, "model_id": model_id_to_use, "error_type": type(e).__name__})
        logging.error(
            f"Error in create_embeddings_batch for model_id '{model_id_to_use}' (Provider: {provider}): {str(e)}",
            exc_info=True)
        raise


def create_embedding(text: str,
                     user_embedding_config: Dict[str, Any],
                     model_id_override: Optional[str] = None
                     ) -> List[float]:
    """
    Creates an embedding for a single text using the batch function.
    Requires user_embedding_config.
    """
    if not text:  # Handle empty string input
        # Decide on behavior: empty list, error, or embedding of empty string if model supports
        logging.warning("`create_embedding` called with empty text.")
        # Returning an empty list or a pre-defined zero vector might be options
        # For now, let it pass to batch and see how model handles it, or raise.
        # Let's assume batch will handle it or model will produce something.
        pass

    # For logging purposes, determine provider and model_id to be used
    emb_cfg = user_embedding_config.get('user_embedding_config', {})
    model_id_to_use = model_id_override if model_id_override else emb_cfg.get('default_model_id', 'unknown_model_id')
    model_spec = emb_cfg.get('models', {}).get(model_id_to_use, {})
    provider_to_use = model_spec.get('provider', 'unknown_provider')

    log_counter("create_embedding_attempt", labels={"provider": provider_to_use, "model_id": model_id_to_use})
    start_time = time.time()

    embeddings_list = create_embeddings_batch(
        texts=[text],
        user_embedding_config=user_embedding_config,
        model_id_override=model_id_to_use  # Pass the determined model_id
    )

    if not embeddings_list or not embeddings_list[0]:
        logging.error(
            f"Failed to generate embedding for single text with model_id '{model_id_to_use}'. Batch returned: {embeddings_list}")
        raise ValueError(f"Embedding generation failed for model_id '{model_id_to_use}'")

    embedding_data = embeddings_list[0]

    embedding_time = time.time() - start_time
    log_histogram("create_embedding_duration", embedding_time,
                  labels={"provider": provider_to_use, "model_id": model_id_to_use})
    log_counter("create_embedding_success", labels={"provider": provider_to_use, "model_id": model_id_to_use})
    return embedding_data

#
# End of File.
#######################################################################################################################
