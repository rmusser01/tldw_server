# Embeddings_Create.py
#
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports
from __future__ import annotations
#
import os
import time
import threading
from functools import wraps
from typing import Any, Dict, List, Optional
#
# Third-party Libraries
import numpy as np
import onnxruntime as ort
import requests
import torch
from huggingface_hub import model_info  # Assuming this is used in _ensure_hf_revision
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge  # Assuming these are defined elsewhere or used directly
from transformers import AutoModel, AutoTokenizer
#
# Local Imports
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import get_openai_embeddings_batch
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram  # Keep your existing metrics
from tldw_Server_API.app.core.Utils.Utils import logging

#
########################################################################################################################
#
# Stuff:
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    OPTIMUM_AVAILABLE = True
except ImportError:
    ORTModelForFeatureExtraction = None
    OPTIMUM_AVAILABLE = False

COMMIT_HASHES: Dict[str, str] = {
    "jinaai/jina-embeddings-v3": "4be32c2f5d65b95e4bcce473545b7883ec8d2edd",
    "Alibaba-NLP/gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
    "dunzhang/setll_en_400M_v5": "2aa5579fcae1c579de199a3866b6e514bbbf5d10",
}

embedding_models: Dict[str, Any] = {}
embedding_models_lock = threading.Lock()  # Global lock for the embedding_models dictionary

# Prometheus Metrics (Ensure these are correctly defined and registered in your application)
ACTIVE_EMBEDDERS = Gauge("active_embedder_instances", "Number of active embedder instances",
                         labelnames=("provider", "model_id"))
EMBEDDINGS_REQUESTS = Counter("embedding_requests_total", "Total number of embedding requests",
                              labelnames=("provider", "model_id"))
MODEL_CACHE_HITS = Counter("embedding_model_cache_hits_total", "Total number of model cache hits",
                           labelnames=("model_id",))


# Add other metrics from your previous version or as needed, e.g., for load times, creation times

class RetryCfg(BaseModel):
    max_retries: int = Field(3, ge=0)
    base_delay: int = Field(1, ge=0)


class RateLimiterCfg(BaseModel):
    max_calls: int = Field(20, ge=1)
    period: int = Field(60, ge=1)


class BaseModelCfg(BaseModel):
    provider: str
    model_name_or_path: str
    trust_remote_code: bool = False
    revision: Optional[str] = None
    max_length: int = 512
    unload_timeout_seconds: int = 300


class HFModelCfg(BaseModelCfg):
    provider: str = "huggingface"
    hf_cache_dir_subpath: str = "huggingface_cache"


class ONNXModelCfg(BaseModelCfg):
    provider: str = "onnx"
    onnx_storage_dir_subpath: str = "onnx_models"
    onnx_providers: List[str] = ["CPUExecutionProvider"]


class OpenAIModelCfg(BaseModelCfg):
    provider: str = "openai"


class LocalAPICfg(BaseModelCfg):
    provider: str = "local_api"
    api_url: str
    api_key: Optional[str] = None
    # Consider adding chunk_size for local_api batching
    # chunk_size: int = 100


ModelCfg = HFModelCfg | ONNXModelCfg | OpenAIModelCfg | LocalAPICfg


class EmbeddingConfigSchema(BaseModel):
    default_model_id: str
    model_storage_base_dir: Optional[str] = Field(default="./embedding_models_data/")
    # These are currently NOT used by the global decorators.
    # If dynamic configuration is needed, decorators must be applied differently.
    rate_limiter: RateLimiterCfg = RateLimiterCfg()
    retry_config: RetryCfg = RetryCfg()
    models: Dict[str, ModelCfg]


def _ensure_hf_revision(model_name_or_path: str, expected_sha: Optional[str]) -> None:
    if expected_sha is None:
        logging.debug(f"No revision SHA provided for {model_name_or_path}, skipping check.")
        return
    try:
        info = model_info(model_name_or_path, revision=expected_sha)  # Check against the specific revision
        actual_sha = info.sha
        if actual_sha != expected_sha:
            logging.error(
                f"SHA mismatch for model {model_name_or_path}. Expected: {expected_sha}, Got: {actual_sha}. "
                f"The model on Hugging Face Hub may have changed for this commit hash."
            )
            raise RuntimeError(
                f"SHA mismatch for model {model_name_or_path}. Expected: {expected_sha}, Got: {actual_sha}")
        logging.info(f"Successfully verified revision SHA {expected_sha} for model {model_name_or_path}.")
    except Exception as e:  # Catch network errors or if model/revision not found
        logging.error(f"Failed to verify revision for {model_name_or_path} (SHA: {expected_sha}): {e}", exc_info=True)
        # Decide if this should be a fatal error. For now, we'll raise to prevent using a potentially wrong model.
        raise RuntimeError(f"Failed to verify model revision for {model_name_or_path}: {e}")


class TokenBucketLimiter:
    def __init__(self, capacity: int, period: int):
        self.capacity = capacity
        self.period = period  # seconds
        self.tokens = float(capacity)
        self.last_refill_time = time.monotonic()
        self.lock = threading.Lock()
        logging.info(f"TokenBucketLimiter initialized with capacity {capacity} tokens per {period} seconds.")

    def _acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed_time = now - self.last_refill_time
                # Calculate tokens to add based on rate (tokens per second)
                # Ensure period is not zero to avoid division by zero
                rate = self.capacity / self.period if self.period > 0 else float('inf')

                tokens_to_add = elapsed_time * rate
                self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                self.last_refill_time = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return  # Token acquired

                # Calculate wait time if no token is available
                # This is the time until one full token is generated
                wait_time = (1 - self.tokens) / rate if rate > 0 else float('inf')

            if wait_time == float('inf'):  # Should not happen if capacity/period are sane
                logging.error("TokenBucketLimiter: Cannot acquire token, rate is zero or invalid.")
                time.sleep(self.period)  # Fallback wait
            elif wait_time > 0:
                logging.debug(f"TokenBucketLimiter: Waiting {wait_time:.2f}s for token.")
                time.sleep(wait_time)
            # Loop again to re-evaluate after waiting

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            self._acquire()
            return fn(*args, **kwargs)

        return wrapper


def exponential_backoff(max_retries: int = 3, base_delay: int = 1):
    """
    Decorator for exponential backoff.
    Note: This uses fixed max_retries and base_delay defined at decoration time.
    It does not use RetryCfg from EmbeddingConfigSchema for dynamic configuration per call.
    """
    logging.info(f"ExponentialBackoff decorator configured with max_retries={max_retries}, base_delay={base_delay}s.")

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):  # +1 to include the initial attempt
                try:
                    return fn(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    status = getattr(e.response, "status_code", None)
                    is_retryable_http = (
                            status == 429  # Too Many Requests
                            or (isinstance(status, int) and 500 <= status < 600)  # Server errors
                    )
                    is_network_error = isinstance(e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))

                    if not (is_retryable_http or is_network_error):
                        logging.error(f"Non-retryable RequestException for {fn.__name__}: {e}", exc_info=True)
                        raise

                    if attempt == max_retries:  # Last attempt failed
                        logging.error(
                            f"Final attempt ({attempt + 1}/{max_retries + 1}) failed for {fn.__name__} "
                            f"due to RequestException: {e}"
                        )
                        raise

                    delay = base_delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {fn.__name__} failed with RequestException. "
                        f"Retrying in {delay}s. Error: {e}"
                    )
                    time.sleep(delay)
                except Exception as e:  # Catch other potentially transient errors (be cautious)
                    # For now, this will retry on any exception other than RequestException.
                    # Consider if this is too broad.
                    if attempt == max_retries:
                        logging.error(
                            f"Final attempt ({attempt + 1}/{max_retries + 1}) failed for {fn.__name__}. Error: {e}")
                        raise
                    delay = base_delay * (2 ** attempt)
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} for {fn.__name__} failed. Retrying in {delay}s. Error: {e}")
                    time.sleep(delay)

        return wrapper

    return decorator


class HuggingFaceEmbedder:
    def __init__(self, model_identifier: str, config: HFModelCfg, hf_cache_dir: str):
        self._lock = threading.RLock()
        self.model_identifier = model_identifier
        self.config = config
        self.hf_cache_dir = hf_cache_dir

        self.revision = config.revision or COMMIT_HASHES.get(config.model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize as Optional, to be populated by load_model
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None  # AutoModel is a class that returns a model instance

        self.unload_timer: Optional[threading.Timer] = None
        self.last_used_time: float = 0.0
        log_counter("huggingface_embedder_init", labels={"model_id": self.model_identifier})
        logging.info(f"HuggingFaceEmbedder initialized for {model_identifier} (model: {config.model_name_or_path})")

    def _reset_timer(self) -> None:
        if self.config.unload_timeout_seconds <= 0:
            return
        with self._lock:
            if self.unload_timer:
                self.unload_timer.cancel()
            self.unload_timer = threading.Timer(self.config.unload_timeout_seconds, self.unload_model)
            self.unload_timer.daemon = True
            self.unload_timer.start()
            logging.debug(
                f"Unload timer reset for {self.model_identifier}, timeout {self.config.unload_timeout_seconds}s")

    def load_model(self) -> None:
        model_load_attempted = False
        start_time = time.time()

        with self._lock:
            if self.model is None or self.tokenizer is None:  # Ensure both are loaded
                model_load_attempted = True
                log_counter("huggingface_model_load_attempt", labels={"model_id": self.model_identifier})
                logging.info(
                    f"Loading HuggingFace model/tokenizer: {self.config.model_name_or_path} (ID: {self.model_identifier}) on device {self.device}")

                _ensure_hf_revision(self.config.model_name_or_path, self.revision)

                # Ensure AutoTokenizer and AutoModel are the classes from transformers
                # These lines assign INSTANCES to self.tokenizer and self.model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name_or_path,
                    cache_dir=self.hf_cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.config.trust_remote_code,
                )
                # AutoModel.from_pretrained returns an instance of a model class (e.g., BertModel, RobertaModel)
                # which is a subclass of PreTrainedModel, which is a torch.nn.Module.
                loaded_model = AutoModel.from_pretrained(
                    self.config.model_name_or_path,
                    cache_dir=self.hf_cache_dir,
                    revision=self.revision,
                    trust_remote_code=self.config.trust_remote_code,
                )
                self.model = loaded_model.to(self.device)
                self.model.eval()

                ACTIVE_EMBEDDERS.labels(provider="huggingface", model_id=self.model_identifier).inc()
                log_counter("huggingface_model_load_success", labels={"model_id": self.model_identifier})
                logging.info(
                    f"HuggingFace model {self.config.model_name_or_path} loaded. Max length: {self.config.max_length}, Timeout: {self.config.unload_timeout_seconds}s."
                )

            self.last_used_time = time.time()

        self._reset_timer()

        if model_load_attempted:
            load_time = time.time() - start_time
            log_histogram("huggingface_model_load_duration", load_time, labels={"model_id": self.model_identifier})

    def unload_model(self) -> None:
        with self._lock:
            log_counter("huggingface_model_unload", labels={"model_id": self.model_identifier})
            if self.model is not None or self.tokenizer is not None:
                logging.info(
                    f"Unloading HuggingFace model/tokenizer {self.config.model_name_or_path} (ID: {self.model_identifier})")
                del self.model
                del self.tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
                self.tokenizer = None
                ACTIVE_EMBEDDERS.labels(provider="huggingface", model_id=self.model_identifier).dec()
                logging.info(f"HuggingFace model {self.model_identifier} unloaded.")

            if self.unload_timer:
                self.unload_timer.cancel()
                self.unload_timer = None

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        self.load_model()

        # --- Start of critical section for using model and tokenizer ---
        # We need to ensure model and tokenizer are not None when used.
        # The lock here protects against the model being unloaded by the timer
        # thread *during* the tokenization and inference process.
        with self._lock:
            # Explicit checks to satisfy type checkers and for runtime safety
            if self.tokenizer is None or self.model is None:
                logging.error(
                    f"Model or tokenizer not loaded for {self.model_identifier} despite load_model call. This indicates a critical issue.")
                # Attempt a final reload under lock, though this state should ideally not be reached.
                self.load_model()
                if self.tokenizer is None or self.model is None:
                    raise RuntimeError(
                        f"Model {self.model_identifier} failed to load even after explicit reload attempt.")

            # At this point, self.tokenizer and self.model are confirmed to be loaded and not None.
            # The type checker should now understand they are instances, not Optional.

            # Re-assign to local variables for type checker to potentially infer non-Optional type better
            # although the checks above should be enough for modern type checkers.
            current_tokenizer = self.tokenizer
            current_model = self.model

            log_counter("huggingface_create_embeddings_attempt", labels={"model_id": self.model_identifier})
            start_time_embed = time.time()
            embeddings_tensor: Optional[torch.Tensor] = None

            try:
                # current_tokenizer is an instance of AutoTokenizer, which is callable
                inputs = current_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # current_model is an instance of a PreTrainedModel, which is callable (its forward method)
                    outputs = current_model(**inputs)
                embeddings_tensor = outputs.last_hidden_state.mean(dim=1)

            except RuntimeError as e:
                # Handle BFloat16 issue
                # The hasattr check is good, add an explicit None check for self.model.dtype
                if "Got unsupported ScalarType BFloat16" in str(e) and \
                        current_model is not None and \
                        hasattr(current_model, 'dtype') and \
                        current_model.dtype == torch.bfloat16:  # current_model is not None here

                    logging.warning(
                        f"BFloat16 not supported for {self.config.model_name_or_path} on {self.device}. "
                        f"Falling back to float32 for model {self.model_identifier}."
                    )

                    # current_model is a torch.nn.Module, so .float() is a valid method.
                    # Re-assign to self.model as well if the change should persist.
                    self.model = current_model.float()  # self.model is now the float version
                    current_model = self.model  # Update local var for current execution
                    log_counter("huggingface_bfloat16_fallback", labels={"model_id": self.model_identifier})

                    # Retry embedding creation with the converted model
                    logging.info(f"Retrying embedding creation for {self.model_identifier} with float32 model.")
                    inputs = current_tokenizer(  # Use current_tokenizer, it hasn't changed
                        texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = current_model(**inputs)  # Use the now-float current_model
                    embeddings_tensor = outputs.last_hidden_state.mean(dim=1)
                else:
                    log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
                    logging.error(f"RuntimeError during HuggingFace embedding for {self.model_identifier}: {e}",
                                  exc_info=True)
                    raise
            except Exception as e:
                log_counter("huggingface_create_embeddings_failure", labels={"model_id": self.model_identifier})
                logging.error(f"Unexpected error during HuggingFace embedding for {self.model_identifier}: {e}",
                              exc_info=True)
                raise

            if embeddings_tensor is None:
                # This should not happen if the try-except block is complete
                logging.error(f"Embeddings tensor is None after processing for {self.model_identifier}")
                raise RuntimeError(f"Failed to produce embeddings tensor for {self.model_identifier}")

            embedding_time = time.time() - start_time_embed
            log_histogram("huggingface_create_embeddings_duration", embedding_time,
                          labels={"model_id": self.model_identifier})
            log_counter("huggingface_create_embeddings_success", labels={"model_id": self.model_identifier})
            return embeddings_tensor.cpu().float().numpy()
        # --- End of critical section ---

    def __del__(self):
        logging.debug(f"HuggingFaceEmbedder {self.model_identifier} is being deleted.")
        if self.unload_timer:
            self.unload_timer.cancel()
            self.unload_timer = None


class ONNXEmbedder:
    def __init__(self, model_identifier: str, config: ONNXModelCfg, onnx_model_base_storage_dir: str):
        self._lock = threading.RLock()  # Reentrant lock for this instance
        self.model_identifier = model_identifier
        self.config = config

        self.revision = config.revision or COMMIT_HASHES.get(config.model_name_or_path)

        # Directory for this specific ONNX model's files (model.onnx, tokenizer, config)
        self.model_specific_onnx_dir = os.path.join(
            onnx_model_base_storage_dir,  # e.g., .../onnx_models/
            config.model_name_or_path.split("/")[-1]  # e.g., gte-large-en-v1.5
        )
        os.makedirs(self.model_specific_onnx_dir, exist_ok=True)
        self.onnx_model_file_path = os.path.join(self.model_specific_onnx_dir, "model.onnx")  # Standard name by optimum

        # Tokenizer is usually stored with the ONNX model by optimum
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,  # Original HF name for tokenizer
            cache_dir=self.model_specific_onnx_dir,  # Store/load tokenizer from the model's ONNX directory
            revision=self.revision,
            trust_remote_code=config.trust_remote_code,
        )

        self.session: Optional[ort.InferenceSession] = None
        self.unload_timer: Optional[threading.Timer] = None
        self.last_used_time: float = 0.0
        self.device_providers = config.onnx_providers

        log_counter("onnx_embedder_init", labels={"model_id": self.model_identifier})
        logging.info(f"ONNXEmbedder initialized for {model_identifier} (model: {config.model_name_or_path})")

    def _ensure_model_converted_and_ready(self) -> None:
        # This method is called from load_model, which holds the instance lock.
        if os.path.exists(self.onnx_model_file_path):
            logging.debug(f"ONNX model file already exists at {self.onnx_model_file_path} for {self.model_identifier}")
            return

        if not OPTIMUM_AVAILABLE or ORTModelForFeatureExtraction is None:
            msg = "`optimum` library is not available. Cannot convert model to ONNX on-the-fly."
            logging.error(msg)
            raise RuntimeError(msg)

        logging.warning(
            f"ONNX model file not found at {self.onnx_model_file_path} for {self.model_identifier}. "
            f"Attempting to convert '{self.config.model_name_or_path}' and save to '{self.model_specific_onnx_dir}'."
        )

        _ensure_hf_revision(self.config.model_name_or_path, self.revision)

        try:
            # ORTModelForFeatureExtraction.from_pretrained with export=True downloads the PyTorch model,
            # converts it, and then save_pretrained saves it to disk.
            # The `cache_dir` for from_pretrained here is where the *original* HF PyTorch model parts are downloaded.
            # It can be the same as self.model_specific_onnx_dir or a temporary HF cache.
            # For simplicity, let's use the model_specific_onnx_dir to keep related files together.
            logging.info(f"Downloading and converting {self.config.model_name_or_path} to ONNX...")
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                self.config.model_name_or_path,
                export=True,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.revision,
                cache_dir=self.model_specific_onnx_dir  # For downloading original HF model before conversion
            )

            logging.info(f"Saving converted ONNX model to {self.model_specific_onnx_dir}...")
            ort_model.save_pretrained(self.model_specific_onnx_dir)  # Saves model.onnx, config.json, etc.

            if not os.path.exists(self.onnx_model_file_path):
                raise FileNotFoundError(
                    f"ONNX 'model.onnx' (expected at {self.onnx_model_file_path}) was not found in "
                    f"{self.model_specific_onnx_dir} after export and save attempt."
                )
            logging.info(
                f"ONNX model for {self.config.model_name_or_path} (ID: {self.model_identifier}) "
                f"successfully exported and saved to {self.model_specific_onnx_dir}"
            )
        except Exception as e:
            logging.error(f"Failed to export/download ONNX model for {self.model_identifier}: {e}", exc_info=True)
            # Basic cleanup: if model.onnx was partially created, remove it.
            if os.path.exists(self.onnx_model_file_path):
                try:
                    os.remove(self.onnx_model_file_path)
                except OSError:
                    pass
            raise RuntimeError(f"ONNX model conversion failed for {self.model_identifier}.")

    def _reset_timer(self) -> None:
        # This method must be thread-safe
        if self.config.unload_timeout_seconds <= 0:
            return
        with self._lock:  # Protect timer manipulation
            if self.unload_timer:
                self.unload_timer.cancel()
            self.unload_timer = threading.Timer(self.config.unload_timeout_seconds, self.unload_model)
            self.unload_timer.daemon = True
            self.unload_timer.start()
            logging.debug(
                f"Unload timer reset for ONNX model {self.model_identifier}, timeout {self.config.unload_timeout_seconds}s")

    def load_model(self) -> None:
        # This entire method needs to be atomic per instance.
        session_load_attempted = False
        start_time = time.time()

        with self._lock:
            if self.session is None:
                session_load_attempted = True
                log_counter("onnx_model_load_attempt", labels={"model_id": self.model_identifier})

                self._ensure_model_converted_and_ready()  # This runs under the same lock

                logging.info(
                    f"Loading ONNX model for {self.model_identifier} from {self.onnx_model_file_path} "
                    f"with providers: {self.device_providers}"
                )
                self.session = ort.InferenceSession(self.onnx_model_file_path, providers=self.device_providers)

                ACTIVE_EMBEDDERS.labels(provider="onnx", model_id=self.model_identifier).inc()
                log_counter("onnx_model_load_success", labels={"model_id": self.model_identifier})
                logging.info(
                    f"ONNX model {self.model_identifier} loaded. Max length: {self.config.max_length}, Timeout: {self.config.unload_timeout_seconds}s."
                )

            self.last_used_time = time.time()

        self._reset_timer()  # Call after releasing main lock

        if session_load_attempted:
            load_time = time.time() - start_time
            log_histogram("onnx_model_load_duration", load_time, labels={"model_id": self.model_identifier})

    def unload_model(self) -> None:
        with self._lock:  # Ensure thread-safety
            log_counter("onnx_model_unload", labels={"model_id": self.model_identifier})
            if self.session is not None:
                logging.info(f"Unloading ONNX model {self.config.model_name_or_path} (ID: {self.model_identifier})")
                del self.session  # Allow OrtInferenceSession to clean up
                self.session = None
                ACTIVE_EMBEDDERS.labels(provider="onnx", model_id=self.model_identifier).dec()
                logging.info(f"ONNX model {self.model_identifier} unloaded.")

            if self.unload_timer:
                self.unload_timer.cancel()
                self.unload_timer = None

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        self.load_model()  # Handles locking, model loading/conversion, and timer reset

        if self.session is None or self.tokenizer is None:
            logging.error(
                f"ONNX session or tokenizer not loaded for {self.model_identifier} before create_embeddings call.")
            raise RuntimeError(f"ONNX model {self.model_identifier} not loaded properly.")

        log_counter("onnx_create_embeddings_attempt", labels={"model_id": self.model_identifier})
        start_time_embed = time.time()

        try:
            # Inference needs to be under lock to prevent unload during operation
            with self._lock:
                # Re-check session status in case it was unloaded by a concurrent timer thread
                if self.session is None:  # Should be rare
                    logging.warning(f"ONNX session for {self.model_identifier} became None unexpectedly. Reloading.")
                    self.load_model()
                    if self.session is None:  # Still none, critical error
                        raise RuntimeError(f"ONNX session for {self.model_identifier} could not be reloaded.")

                inputs = self.tokenizer(
                    texts,
                    return_tensors="np",  # ONNX runtime uses NumPy arrays
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                )
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }
                # Some models need token_type_ids, some don't. Check if tokenizer provides them.
                if "token_type_ids" in inputs and inputs["token_type_ids"] is not None:
                    model_input_names = [inp.name for inp in self.session.get_inputs()]
                    if "token_type_ids" in model_input_names:
                        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
                    elif "token_type_ids" in ort_inputs:  # remove if tokenizer provided but model doesn't want
                        del ort_inputs["token_type_ids"]

                ort_outputs = self.session.run(None, ort_inputs)

                # Pooling: Mean pooling of the last hidden state, considering attention mask
                last_hidden_state = ort_outputs[0]  # Typically the first output
                if not isinstance(last_hidden_state, np.ndarray):
                    raise TypeError(f"Expected numpy array from ONNX output, got {type(last_hidden_state)}")

                input_mask_expanded = np.expand_dims(ort_inputs["attention_mask"], -1).astype(float)
                sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
                sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)  # Avoid division by zero
                embeddings_np = sum_embeddings / sum_mask

        except Exception as e:
            log_counter("onnx_create_embeddings_failure", labels={"model_id": self.model_identifier})
            logging.error(f"Error creating embeddings with ONNX model {self.model_identifier}: {e}", exc_info=True)
            raise

        embedding_time = time.time() - start_time_embed
        log_histogram("onnx_create_embeddings_duration", embedding_time, labels={"model_id": self.model_identifier})
        log_counter("onnx_create_embeddings_success", labels={"model_id": self.model_identifier})
        return embeddings_np.astype(np.float32)  # Ensure float32 output

    def __del__(self):
        logging.debug(f"ONNXEmbedder {self.model_identifier} is being deleted.")
        if self.unload_timer:
            self.unload_timer.cancel()
            self.unload_timer = None


# Global limiter instance. Parameters (20 calls per 60s) are fixed.
# To make this dynamic per model_config, the limiter would need to be
# managed differently, perhaps per-embedder or applied inside create_embeddings_batch.
limiter = TokenBucketLimiter(capacity=20, period=60)


# Exponential backoff decorator with fixed parameters.
# To make this dynamic per model_config, apply similarly to limiter.
@exponential_backoff(max_retries=3, base_delay=1)
@limiter  # Applied to all calls to create_embeddings_batch
def create_embeddings_batch(
        texts: List[str],
        user_app_config: Dict[str, Any],  # Renamed for clarity: this is the top-level app config
        model_id_override: Optional[str] = None,
) -> List[List[float]]:
    """
    Creates embeddings for a batch of texts.
    `user_app_config` should contain an 'embedding_config' key with EmbeddingConfigSchema structure.
    """
    if not texts:
        logging.warning("create_embeddings_batch called with empty list of texts.")
        return []

    try:
        # Extract and validate the specific embedding configuration part
        if "embedding_config" not in user_app_config:
            logging.error("'embedding_config' key not found in user_app_config.")
            raise ValueError("'embedding_config' key missing from application configuration.")

        # Pydantic will parse and validate. If it fails, it raises a ValidationError.
        embedding_service_config = EmbeddingConfigSchema(**user_app_config["embedding_config"])
    except Exception as e:  # Catch Pydantic ValidationError or other parsing issues
        logging.error(f"Failed to parse embedding_config: {e}", exc_info=True)
        raise ValueError(f"Invalid embedding_config structure: {e}")

    model_id_to_use = model_id_override if model_id_override else embedding_service_config.default_model_id
    if not model_id_to_use:
        logging.error("No `model_id` specified and no `default_model_id` found in embedding_config.")
        raise ValueError("Embedding model ID not specified or configured as default.")

    model_spec = embedding_service_config.models.get(model_id_to_use)
    if not model_spec:
        logging.error(f"Configuration for `model_id` '{model_id_to_use}' not found in `embedding_config.models`.")
        raise ValueError(f"Invalid `model_id` or configuration missing: {model_id_to_use}")

    provider = model_spec.provider
    # Ensure model_storage_base_dir exists
    base_dir = embedding_service_config.model_storage_base_dir
    os.makedirs(base_dir, exist_ok=True)

    EMBEDDINGS_REQUESTS.labels(provider=provider, model_id=model_id_to_use).inc()
    start_time_batch = time.time()
    embeddings_list: List[List[float]] = []

    try:
        embedder_instance: Any = None  # To hold HFEmbedder or ONNXEmbedder

        if provider.lower() == "huggingface":
            if not isinstance(model_spec, HFModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not HFModelCfg.")

            with embedding_models_lock:  # Protect access to the global embedding_models cache
                if model_id_to_use not in embedding_models:
                    logging.info(f"HuggingFace model ID {model_id_to_use} not in cache. Initializing.")
                    hf_cache_dir = os.path.join(base_dir, model_spec.hf_cache_dir_subpath)
                    os.makedirs(hf_cache_dir, exist_ok=True)
                    embedding_models[model_id_to_use] = HuggingFaceEmbedder(model_id_to_use, model_spec, hf_cache_dir)
                else:
                    MODEL_CACHE_HITS.labels(model_id=model_id_to_use).inc()
                embedder_instance = embedding_models[model_id_to_use]

            if embedder_instance:
                embeddings_np = embedder_instance.create_embeddings(texts)
                embeddings_list = embeddings_np.tolist()

        elif provider.lower() == "onnx":
            if not isinstance(model_spec, ONNXModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not ONNXModelCfg.")

            with embedding_models_lock:
                if model_id_to_use not in embedding_models:
                    logging.info(f"ONNX model ID {model_id_to_use} not in cache. Initializing.")
                    # The ONNXEmbedder itself will construct the full path to its model directory
                    # using base_dir and onnx_storage_dir_subpath.
                    # We just need to pass the base for all ONNX models.
                    embedding_models[model_id_to_use] = ONNXEmbedder(model_id_to_use, model_spec, base_dir)
                else:
                    MODEL_CACHE_HITS.labels(model_id=model_id_to_use).inc()
                embedder_instance = embedding_models[model_id_to_use]

            if embedder_instance:
                embeddings_np = embedder_instance.create_embeddings(texts)
                embeddings_list = embeddings_np.tolist()

        elif provider.lower() == "openai":
            if not isinstance(model_spec, OpenAIModelCfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not OpenAIModelCfg.")

            logging.debug(
                f"Creating embeddings for {len(texts)} texts via OpenAI API with model {model_spec.model_name_or_path}")
            if not callable(get_openai_embeddings_batch):  # Basic check
                logging.error("`get_openai_embeddings_batch` is not available or not callable.")
                raise NotImplementedError("OpenAI batch embedding function is not properly set up.")

            # Pass the full user_app_config as it might contain API keys or other necessary settings
            # for get_openai_embeddings_batch
            embeddings_list = get_openai_embeddings_batch(
                texts,
                model=model_spec.model_name_or_path,
                app_config=user_app_config  # Or pass only relevant parts if get_openai_embeddings_batch is refactored
            )

        elif provider.lower() == "local_api":
            if not isinstance(model_spec, LocalAPICfg):
                raise ValueError(f"Model spec for {model_id_to_use} is not LocalAPICfg.")

            # TODO: Implement chunking for texts if len(texts) is large, based on model_spec.chunk_size
            logging.debug(
                f"Creating {len(texts)} embeddings via local API ({model_spec.api_url}) with model {model_spec.model_name_or_path}")
            headers = {"Content-Type": "application/json"}
            if model_spec.api_key:
                headers["Authorization"] = f"Bearer {model_spec.api_key}"

            payload = {"texts": texts, "model": model_spec.model_name_or_path}

            # The requests.post call is already wrapped by @exponential_backoff and @limiter
            response = requests.post(model_spec.api_url, json=payload, headers=headers, timeout=60)  # Default timeout
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            if 'embeddings' not in response_data or not isinstance(response_data['embeddings'], list):
                logging.error(f"Local API at {model_spec.api_url} returned unexpected data format: {response_data}")
                raise ValueError("Local API embedding response format error.")
            embeddings_list = response_data['embeddings']

        else:
            logging.error(f"Unsupported embedding provider: {provider} for model_id '{model_id_to_use}'")
            raise ValueError(f"Unsupported embedding provider: {provider}")

        batch_time = time.time() - start_time_batch
        log_histogram("create_embeddings_batch_duration", batch_time,
                      labels={"provider": provider, "model_id": model_id_to_use})
        log_counter("create_embeddings_batch_success", labels={"provider": provider, "model_id": model_id_to_use})
        return embeddings_list

    except ValueError as ve:  # Configuration or validation errors
        log_counter("create_embeddings_batch_error",
                    labels={"provider": provider if 'provider' in locals() else 'unknown',
                            "model_id": model_id_to_use if 'model_id_to_use' in locals() else 'unknown',
                            "error_type": type(ve).__name__})
        logging.error(f"Configuration or Value error in create_embeddings_batch: {ve}", exc_info=True)
        raise
    except RuntimeError as rte:  # Model loading, conversion, or runtime issues
        log_counter("create_embeddings_batch_error",
                    labels={"provider": provider if 'provider' in locals() else 'unknown',
                            "model_id": model_id_to_use if 'model_id_to_use' in locals() else 'unknown',
                            "error_type": type(rte).__name__})
        logging.error(f"Runtime error in create_embeddings_batch: {rte}", exc_info=True)
        raise
    except requests.exceptions.RequestException as req_e:  # Handled by backoff, but re-raised if all retries fail
        log_counter("create_embeddings_batch_error",
                    labels={"provider": provider if 'provider' in locals() else 'unknown',
                            "model_id": model_id_to_use if 'model_id_to_use' in locals() else 'unknown',
                            "error_type": type(req_e).__name__})
        logging.error(f"RequestException after retries in create_embeddings_batch: {req_e}", exc_info=True)
        raise
    except Exception as e:  # Catch-all for unexpected errors
        log_counter("create_embeddings_batch_error",
                    labels={"provider": provider if 'provider' in locals() else 'unknown',
                            "model_id": model_id_to_use if 'model_id_to_use' in locals() else 'unknown',
                            "error_type": type(e).__name__})
        logging.error(
            f"Unexpected error in create_embeddings_batch for model_id '{model_id_to_use if 'model_id_to_use' in locals() else 'unknown'}' "
            f"(Provider: {provider if 'provider' in locals() else 'unknown'}): {e}",
            exc_info=True)
        raise


def create_embedding(
        text: str,
        user_app_config: Dict[str, Any],
        model_id_override: Optional[str] = None,
) -> List[float]:
    """
    Creates an embedding for a single text using the batch function.
    `user_app_config` should contain an 'embedding_config' key.
    """
    if not text:
        logging.warning("`create_embedding` called with empty text. Behavior depends on model.")
        # Models might return a specific embedding for empty string, or error.
        # For now, proceed and let the batch/model handle it.

    # Determine provider and model_id for logging purposes before calling batch,
    # as batch might raise an error before these are determined internally.
    provider_to_log = 'unknown_provider'
    model_id_to_log = 'unknown_model_id'
    try:
        if "embedding_config" in user_app_config:
            temp_config = EmbeddingConfigSchema(**user_app_config["embedding_config"])
            model_id_to_log = model_id_override or temp_config.default_model_id
            if model_id_to_log in temp_config.models:
                provider_to_log = temp_config.models[model_id_to_log].provider
    except Exception:
        pass  # Ignore parsing errors here, batch function will handle and log properly

    log_counter("create_embedding_attempt", labels={"provider": provider_to_log, "model_id": model_id_to_log})
    start_time_single = time.time()

    # The create_embeddings_batch function is already decorated with rate limiter and backoff
    embeddings_list = create_embeddings_batch(
        texts=[text],
        user_app_config=user_app_config,
        model_id_override=model_id_override  # Pass override if provided
    )

    if not embeddings_list or not embeddings_list[0]:
        # This path should ideally be caught by errors within create_embeddings_batch
        # or the specific embedder if it's a model-specific issue.
        log_counter("create_embedding_failure", labels={"provider": provider_to_log, "model_id": model_id_to_log})
        logging.error(
            f"Failed to generate embedding for single text with model_id '{model_id_to_log}'. Batch returned empty or invalid."
        )
        raise ValueError(f"Embedding generation failed for single text using model_id '{model_id_to_log}'.")

    embedding_data = embeddings_list[0]

    single_time = time.time() - start_time_single
    log_histogram("create_embedding_duration", single_time,
                  labels={"provider": provider_to_log, "model_id": model_id_to_log})
    log_counter("create_embedding_success", labels={"provider": provider_to_log, "model_id": model_id_to_log})
    return embedding_data

#
# Legacy exports for backward compatibility
# Load embedding configuration from settings
from tldw_Server_API.app.core.config import settings

embedding_config = settings.get("EMBEDDING_CONFIG", {})
embedding_provider = embedding_config.get('embedding_provider', 'openai')
embedding_model = embedding_config.get('embedding_model', 'text-embedding-3-small')
embedding_api_url = embedding_config.get('embedding_api_url', 'http://localhost:8080/v1/embeddings')
embedding_api_key = embedding_config.get('embedding_api_key', '')

#
# End of File.
#######################################################################################################################