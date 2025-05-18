# Embeddings_Create.py
# Description: Functions for Creating and managing Embeddings in ChromaDB with LLama.cpp/OpenAI/Transformers
#
# Imports:
import os
import time
from functools import wraps
from threading import Lock, Timer
from typing import List
#
# 3rd-Party Imports:
import numpy as np
import onnxruntime as ort
import requests
from transformers import AutoTokenizer, AutoModel
import torch
#
# Local Imports:
from PoC_Version.App_Function_Libraries.LLM_API_Calls import get_openai_embeddings
from PoC_Version.App_Function_Libraries.Utils.Utils import load_and_log_configs, logging
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

# Load configuration
loaded_config = load_and_log_configs()
embedding_provider = loaded_config['embedding_config']['embedding_provider']
embedding_model = loaded_config['embedding_config']['embedding_model']
embedding_api_url = loaded_config['embedding_config']['embedding_api_url']
embedding_api_key = loaded_config['embedding_config']['embedding_api_key']
model_dir = loaded_config['embedding_config']['model_dir'] or './App_Function_Libraries/models/embedding_models/'

# Embedding Chunking Settings
chunk_size = loaded_config['embedding_config']['chunk_size']
overlap = loaded_config['embedding_config']['chunk_overlap']

# Global cache for embedding models
embedding_models = {}

# Commit hashes
commit_hashes = {
    "jinaai/jina-embeddings-v3": "4be32c2f5d65b95e4bcce473545b7883ec8d2edd",
    "Alibaba-NLP/gte-large-en-v1.5": "104333d6af6f97649377c2afbde10a7704870c7b",
    "dunzhang/setll_en_400M_v5": "2aa5579fcae1c579de199a3866b6e514bbbf5d10"
}

class HuggingFaceEmbedder:
    def __init__(self, model_name, cache_dir, timeout_seconds=30):
        self.model_name = model_name
        self.cache_dir = cache_dir  # Store cache_dir
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout_seconds = timeout_seconds
        self.last_used_time = 0
        self.unload_timer = None
        log_counter("huggingface_embedder_init", labels={"model_name": model_name})

    def load_model(self):
        log_counter("huggingface_model_load_attempt", labels={"model_name": self.model_name})
        start_time = time.time()
        # https://huggingface.co/docs/transformers/custom_models
        if self.model is None:
            # Pass cache_dir to from_pretrained to specify download directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,  # Specify cache directory
                revision=commit_hashes.get(self.model_name, None)  # Pass commit hash
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir,  # Specify cache directory
                revision=commit_hashes.get(self.model_name, None)  # Pass commit hash
            )
            self.model.to(self.device)
        self.last_used_time = time.time()
        self.reset_timer()
        load_time = time.time() - start_time
        log_histogram("huggingface_model_load_duration", load_time, labels={"model_name": self.model_name})
        log_counter("huggingface_model_load_success", labels={"model_name": self.model_name})

    def unload_model(self):
        log_counter("huggingface_model_unload", labels={"model_name": self.model_name})
        if self.model is not None:
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

    def create_embeddings(self, texts):
        log_counter("huggingface_create_embeddings_attempt", labels={"model_name": self.model_name})
        start_time = time.time()
        self.load_model()
        # https://huggingface.co/docs/transformers/custom_models
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().float().numpy()  # Convert to float32 before returning
        except RuntimeError as e:
            if "Got unsupported ScalarType BFloat16" in str(e):
                logging.warning("BFloat16 not supported. Falling back to float32.")
                # Convert model to float32
                self.model = self.model.float()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embedding_time = time.time() - start_time
                log_histogram("huggingface_create_embeddings_duration", embedding_time,
                              labels={"model_name": self.model_name})
                log_counter("huggingface_create_embeddings_success", labels={"model_name": self.model_name})
                return embeddings.cpu().float().numpy()
            else:
                log_counter("huggingface_create_embeddings_failure", labels={"model_name": self.model_name})
                raise

class ONNXEmbedder:
    def __init__(self, model_name, onnx_model_dir, timeout_seconds=30):
        self.model_name = model_name
        self.model_path = os.path.join(onnx_model_dir, f"{model_name}.onnx")
        # https://huggingface.co/docs/transformers/custom_models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=onnx_model_dir,  # Ensure tokenizer uses the same directory
            revision=commit_hashes.get(model_name, None)  # Pass commit hash
        )
        self.session = None
        self.timeout_seconds = timeout_seconds
        self.last_used_time = 0
        self.unload_timer = None
        self.device = "cpu"  # ONNX Runtime will default to CPU unless GPU is configured
        log_counter("onnx_embedder_init", labels={"model_name": model_name})

    def load_model(self):
        log_counter("onnx_model_load_attempt", labels={"model_name": self.model_name})
        start_time = time.time()
        if self.session is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX model not found at {self.model_path}")
            logging.info(f"Loading ONNX model from {self.model_path}")
            self.session = ort.InferenceSession(self.model_path)
        self.last_used_time = time.time()
        self.reset_timer()
        load_time = time.time() - start_time
        log_histogram("onnx_model_load_duration", load_time, labels={"model_name": self.model_name})
        log_counter("onnx_model_load_success", labels={"model_name": self.model_name})

    def unload_model(self):
        log_counter("onnx_model_unload", labels={"model_name": self.model_name})
        if self.session is not None:
            logging.info("Unloading ONNX model to free resources.")
            self.session = None
        if self.unload_timer:
            self.unload_timer.cancel()

    def reset_timer(self):
        if self.unload_timer:
            self.unload_timer.cancel()
        self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
        self.unload_timer.start()

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        log_counter("onnx_create_embeddings_attempt", labels={"model_name": self.model_name})
        start_time = time.time()
        self.load_model()
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            ort_outputs = self.session.run(None, ort_inputs)

            last_hidden_state = ort_outputs[0]
            embeddings = np.mean(last_hidden_state, axis=1)

            embedding_time = time.time() - start_time
            log_histogram("onnx_create_embeddings_duration", embedding_time, labels={"model_name": self.model_name})
            log_counter("onnx_create_embeddings_success", labels={"model_name": self.model_name})
            return embeddings.tolist()
        except Exception as e:
            log_counter("onnx_create_embeddings_failure", labels={"model_name": self.model_name})
            logging.error(f"Error creating embeddings with ONNX model: {str(e)}")
            raise

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [call for call in self.calls if call > now - self.period]
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.calls[0] - (now - self.period)
                    time.sleep(sleep_time)
                self.calls.append(time.time())
            return func(*args, **kwargs)
        return wrapper

def exponential_backoff(max_retries=5, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds. Error: {str(e)}")
                    time.sleep(delay)
        return wrapper
    return decorator

@exponential_backoff()
@RateLimiter(max_calls=50, period=60)
def create_embeddings_batch(texts: List[str],
                            provider: str,
                            model: str,
                            api_url: str,
                            timeout_seconds: int = 300
                            ) -> List[List[float]]:
    global embedding_models
    log_counter("create_embeddings_batch_attempt", labels={"provider": provider, "model": model})
    start_time = time.time()

    try:
        if provider == 'huggingface' or "Huggingface" in provider:
            if model not in embedding_models:
                if model == "dunzhang/stella_en_400M_v5":
                    embedding_models[model] = ONNXEmbedder(model, model_dir, timeout_seconds)
                else:
                    # Pass model_dir to HuggingFaceEmbedder
                    embedding_models[model] = HuggingFaceEmbedder(model, model_dir, timeout_seconds)
            embedder = embedding_models[model]
            embedding_time = time.time() - start_time
            log_histogram("create_embeddings_batch_duration", embedding_time,
                          labels={"provider": provider, "model": model})
            log_counter("create_embeddings_batch_success", labels={"provider": provider, "model": model})
            return embedder.create_embeddings(texts)

        elif provider.lower() == 'openai':
            logging.debug(f"Creating embeddings for {len(texts)} texts using OpenAI API")
            embedding_time = time.time() - start_time
            log_histogram("create_embeddings_batch_duration", embedding_time,
                          labels={"provider": provider, "model": model})
            log_counter("create_embeddings_batch_success", labels={"provider": provider, "model": model})
            return [create_openai_embedding(text, model) for text in texts]

        elif provider.lower() == 'local':
            response = requests.post(
                api_url,
                json={"texts": texts, "model": model},
                headers={"Authorization": f"Bearer {embedding_api_key}"}
            )
            if response.status_code == 200:
                embedding_time = time.time() - start_time
                log_histogram("create_embeddings_batch_duration", embedding_time,
                              labels={"provider": provider, "model": model})
                log_counter("create_embeddings_batch_success", labels={"provider": provider, "model": model})
                return response.json()['embeddings']
            else:
                raise Exception(f"Error from local API: {response.text}")
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    except Exception as e:
        log_counter("create_embeddings_batch_error", labels={"provider": provider, "model": model, "error": str(e)})
        logging.error(f"Error in create_embeddings_batch: {str(e)}")
        raise

def create_embedding(text: str, provider: str, model: str, api_url: str) -> List[float]:
    log_counter("create_embedding_attempt", labels={"provider": provider, "model": model})
    start_time = time.time()
    embedding = create_embeddings_batch([text], provider, model, api_url)[0]
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    embedding_time = time.time() - start_time
    log_histogram("create_embedding_duration", embedding_time, labels={"provider": provider, "model": model})
    log_counter("create_embedding_success", labels={"provider": provider, "model": model})
    return embedding

def create_openai_embedding(text: str, model: str) -> List[float]:
    log_counter("create_openai_embedding_attempt", labels={"model": model})
    start_time = time.time()
    embedding = get_openai_embeddings(text, model)
    embedding_time = time.time() - start_time
    log_histogram("create_openai_embedding_duration", embedding_time, labels={"model": model})
    log_counter("create_openai_embedding_success", labels={"model": model})
    return embedding


#
# ##############################################################
# #
# # ONNX Embeddings Functions
#
# # FIXME - UPDATE
# # Define the model path
# model_dir = "/tldw/App_Function_Libraries/models/embedding_models/"
# model_name = "your-huggingface-model-name"
# onnx_model_path = os.path.join(model_dir, model_name, "model.onnx")
#
# # Tokenizer download (if applicable)
# #tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Ensure the model directory exists
# #if not os.path.exists(onnx_model_path):
#     # You can add logic to download the ONNX model from a remote source
#     # if it's not already available in the folder.
#     # Example: huggingface_hub.download (if model is hosted on Hugging Face Hub)
# #    raise Exception(f"ONNX model not found at {onnx_model_path}")
#
# class ONNXEmbedder:
#     def __init__(self, model_name, model_dir, timeout_seconds=120):
#         self.model_name = model_name
#         self.model_path = os.path.join(model_dir, f"{model_name}.onnx")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.session = None
#         self.timeout_seconds = timeout_seconds
#         self.last_used_time = 0
#         self.unload_timer = None
#         self.device = "cpu"  # ONNX Runtime will default to CPU unless GPU is configured
#
#     def load_model(self):
#         if self.session is None:
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"ONNX model not found at {self.model_path}")
#             logging.info(f"Loading ONNX model from {self.model_path}")
#             self.session = ort.InferenceSession(self.model_path)
#         self.last_used_time = time.time()
#         self.reset_timer()
#
#     def unload_model(self):
#         if self.session is not None:
#             logging.info("Unloading ONNX model to free resources.")
#             self.session = None
#         if self.unload_timer:
#             self.unload_timer.cancel()
#
#     def reset_timer(self):
#         if self.unload_timer:
#             self.unload_timer.cancel()
#         self.unload_timer = Timer(self.timeout_seconds, self.unload_model)
#         self.unload_timer.start()
#
#     def create_embeddings(self, texts: List[str]) -> List[List[float]]:
#         self.load_model()
#
#         try:
#             inputs = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)
#             input_ids = inputs["input_ids"].astype(np.int64)
#             attention_mask = inputs["attention_mask"].astype(np.int64)
#
#             ort_inputs = {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask
#             }
#
#             ort_outputs = self.session.run(None, ort_inputs)
#
#             last_hidden_state = ort_outputs[0]
#             embeddings = np.mean(last_hidden_state, axis=1)
#
#             return embeddings.tolist()
#         except Exception as e:
#             logging.error(f"Error creating embeddings with ONNX model: {str(e)}")
#             raise
#
# # Global cache for the ONNX embedder instance
# onnx_embedder = None
#
# # Global cache for embedding models
# embedding_models = {}
#
# def create_onnx_embeddings(texts: List[str]) -> List[List[float]]:
#     global onnx_embedder
#     model_dir = "/tldw/App_Function_Libraries/models/embedding_models/"
#     model_name = "your-huggingface-model-name"  # This can be pulled from config
#
#     if onnx_embedder is None:
#         onnx_embedder = ONNXEmbedder(model_name=model_name, model_dir=model_dir)
#
#     # Generate embeddings
#     embeddings = onnx_embedder.create_embeddings(texts)
#     return embeddings
#
# #
# # End of ONNX Embeddings Functions
# ##############################################################

#
# End of File.
#######################################################################################################################
