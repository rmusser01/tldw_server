# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Sequence, Literal
import threading
# 3rd-Party Imports:
import chromadb
from chromadb import Settings
from chromadb.errors import ChromaError
from itertools import islice
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
#
# Local Imports:
from tldw_Server_API.app.core.Chunking.Chunk_Lib import chunk_for_embedding  # Assuming this is correct
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embedding, create_embeddings_batch
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze  # Assuming this is correct
from tldw_Server_API.app.core.Utils.Utils import logger  # Assuming this is 'logging' aliased or a custom logger
#
#######################################################################################################################
#
# Functions:
ChromaIncludeLiteral = Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]

class ChromaDBManager:
    """
    Manages ChromaDB instances and operations for specific users.
    Each instance of this class corresponds to a user's isolated ChromaDB storage.
    """
    DEFAULT_COLLECTION_NAME_PREFIX = "user_embeddings_for_"  # Can be made configurable

    def __init__(self, user_id: str, user_embedding_config: Dict[str, Any]):
        """
        Initializes the ChromaDBManager for a specific user.

        Args:
            user_id (str): The ID of the user for whom this ChromaDB instance is.
            user_embedding_config (Dict[str, Any]): The global application configuration dictionary.
        """
        if not user_id:
            logger.error("Initialization failed: user_id cannot be empty for ChromaDBManager.")
            raise ValueError("user_id cannot be empty for ChromaDBManager.")
        if not user_embedding_config:
            logger.error("Initialization failed: user_embedding_config cannot be empty for ChromaDBManager.")
            raise ValueError("user_embedding_config cannot be empty for ChromaDBManager.")

        self.user_id = str(user_id)
        self.user_embedding_config = user_embedding_config
        self._lock = threading.RLock()  # Instance-specific lock

        # --- Configuration Usage (Point 1) ---
        user_db_base_dir_str = self.user_embedding_config.get("USER_DB_BASE_DIR")
        if not user_db_base_dir_str:
            logger.critical("USER_DB_BASE_DIR not found in user_embedding_config. ChromaDBManager cannot be initialized.")
            raise ValueError("USER_DB_BASE_DIR not configured in application settings.")

        self.user_chroma_path: Path = (Path(user_db_base_dir_str) / self.user_id / "chroma_storage").resolve()
        try:
            self.user_chroma_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.critical(
                f"Failed to create ChromaDB storage path {self.user_chroma_path} for user '{self.user_id}': {e}",
                exc_info=True)
            raise RuntimeError(f"Could not create ChromaDB storage directory: {e}") from e

        logger.info(f"ChromaDBManager for user '{self.user_id}' initialized. Path: {self.user_chroma_path}")

        chroma_client_settings_config = self.user_embedding_config.get("chroma_client_settings", {})
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.user_chroma_path),
                settings=Settings(
                    anonymized_telemetry=chroma_client_settings_config.get("anonymized_telemetry", False),
                    allow_reset=chroma_client_settings_config.get("allow_reset", True)
                    # consider is_persistent=True explicitly if needed, though PersistentClient implies it.
                )
            )
        except Exception as e:  # Catch broader exceptions during client initialization
            logger.critical(
                f"Failed to initialize ChromaDB PersistentClient for user '{self.user_id}' at {self.user_chroma_path}: {e}",
                exc_info=True)
            raise RuntimeError(f"ChromaDB client initialization failed: {e}") from e

        # Default embedding model_id for this manager instance.
        # This can be set by the user/application when creating the ChromaDBManager instance
        # or fall back to a system default from user_embedding_config.
        self.embedding_config = self.user_embedding_config.get("embedding_config", {})
        # Point 3: Allow user to choose their default model for this manager instance.
        # This can be passed in via user_embedding_config specifically for this user, or a more direct param.
        # For now, let's assume it's derived from a general default if not specified for the user.
        self.default_embedding_model_id = self.embedding_config.get('default_model_id')

        if not self.default_embedding_model_id:
            logger.warning(  # Changed to warning, operations might still succeed if model_id is always overridden
                f"User '{self.user_id}': No 'default_model_id' found in 'embedding_config'. "
                "Operations will require explicit 'embedding_model_id_override'."
            )
            # Not raising an error to allow flexibility if all calls provide an override.

        model_details = self.embedding_config.get("models", {}).get(self.default_embedding_model_id, {})
        logger.info(
            f"User '{self.user_id}' ChromaDBManager configured. "
            f"Default Embedding Model ID: {self.default_embedding_model_id or 'Not Set (Override Required)'} "
            f"(Provider: {model_details.get('provider', 'N/A')}, Name: {model_details.get('model_name_or_path', 'N/A')})"
        )

    def _batched(self, iterable, n):
        """Helper to yield batches from an iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                return
            yield batch

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans metadata to ensure compatibility with ChromaDB."""
        cleaned = {}
        if not isinstance(metadata, dict):
            logger.warning(
                f"User '{self.user_id}': Received non-dict metadata: {type(metadata)}. Returning empty dict.")
            return cleaned

        for key, value in metadata.items():
            if value is None:  # ChromaDB can handle None, but explicit skip or convert might be safer.
                # cleaned[key] = None # Or skip
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (np.int32, np.int64, np.int16, np.int8)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.float32, np.float64, np.float16)):
                cleaned[key] = float(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            elif isinstance(value, (list, tuple)):  # Chroma allows lists of primitives
                cleaned[key] = [self._clean_metadata_value(v) for v in value]
            else:  # Fallback to string, log a warning for unexpected types
                logger.debug(
                    f"User '{self.user_id}': Converting metadata value of type {type(value)} for key '{key}' to string.")
                cleaned[key] = str(value)
        return cleaned

    def _clean_metadata_value(self, value: Any) -> Any:
        """Helper for cleaning individual values within a list in metadata."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (np.int32, np.int64, np.int16, np.int8)):
            return int(value)
        if isinstance(value, (np.float32, np.float64, np.float16)):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        logger.debug(f"User '{self.user_id}': Converting list element of type {type(value)} to string in metadata.")
        return str(value)

    def get_user_default_collection_name(self) -> str:
        """Gets the default collection name for the user, incorporating user ID."""
        # Sanitize user_id for collection name if it can contain special characters
        # For now, assuming user_id is safe or ChromaDB handles it.
        return f"{self.DEFAULT_COLLECTION_NAME_PREFIX}{self.user_id}"

    # Point 2: Collection Management# FIXME - Implement this
    # When creating a new collection for a specific model
    model_id_for_new_collection = "user_chosen_model_for_this_collection"
    # You'd need a way to get the dimension for this model_id, perhaps from Embeddings_Create or config
    # model_dimension = get_dimension_for_model(app_config, model_id_for_new_collection)
    # chroma_manager.get_or_create_collection(
    #     collection_name="my_new_collection_name",
    #     collection_metadata={
    #         "source_embedding_model_id": model_id_for_new_collection,
    #         "embedding_dimension": model_dimension,
    #         "hnsw:space": "cosine" # or other relevant Chroma params
    #     }
    # )
    # Then, when process_and_store_content or vector_search operate on a named collection, they should ideally:
    #
    #     Retrieve the collection.
    #
    #     Check its metadata for the source_embedding_model_id.
    #
    #     Use that model ID for embedding generation/querying, overriding the ChromaDBManager's default_embedding_model_id.
    #     This makes each collection self-contained regarding its embedding model.
    #
    # The current code in store_in_chroma has improved dimension checking and will recreate the collection if a dimension mismatch occurs, logging the embedding_model_id_for_dim_check. It also attempts to store the dimension in the collection metadata upon recreation.

    def get_or_create_collection(self, collection_name: Optional[str] = None,
                                 collection_metadata: Optional[Dict[str, Any]] = None) -> Collection:
        """
        Gets or creates a ChromaDB collection.

        Args:
            collection_name (Optional[str]): Name of the collection. Defaults to user's default.
            collection_metadata (Optional[Dict[str, Any]]): Metadata for the collection,
                                                           e.g., {'hnsw:space': 'cosine'}.

        Returns:
            Collection: The ChromaDB collection object.

        Raises:
            RuntimeError: If collection creation or retrieval fails.
        """
        name_to_use = collection_name or self.get_user_default_collection_name()
        with self._lock:
            try:
                # The embedding_function parameter is for Chroma to generate embeddings.
                # Since we provide embeddings directly, it's not strictly needed here.
                # However, setting metadata like hnsw:space can be useful.
                collection = self.client.get_or_create_collection(
                    name=name_to_use,
                    metadata=self._clean_metadata(collection_metadata) if collection_metadata else None
                )
                logger.info(f"User '{self.user_id}': Accessed/Created collection '{name_to_use}'.")
                return collection
            except Exception as e:
                logger.error(f"User '{self.user_id}': Failed to get or create collection '{name_to_use}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Failed to access or create collection '{name_to_use}': {e}") from e

    # Point 4: Situate Context - Async/Batching exploration
    # FIXME - Explore async/batching for situate_context LLM calls
    # This would likely involve using an async HTTP client for `analyze` or a batching LLM API.
    def situate_context(self, api_name_for_context: str, doc_content: str, chunk_content: str) -> str:
        """Generates a succinct context for a chunk within a larger document using an LLM."""
        # Prompts could be made configurable
        doc_content_prompt = f"<document>\n{doc_content}\n</document>"
        chunk_context_prompt = (
            f"\n\n\n\n\nHere is the chunk we want to situate within the whole document\n<chunk>\n{chunk_content}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of the chunk.\n"
            "Answer only with the succinct context and nothing else."
        )
        try:
            # Assuming `analyze` handles its own LLM config (API key, model selection via api_name)
            # FIXME - Update to proper analyze function call args
            response = analyze(
                api_name=api_name_for_context,
                input_data=chunk_content,
                prompt=chunk_context_prompt,
                context=doc_content_prompt,

                # temp=0, # Pass relevant params for `analyze`
                # system_message=None
                user_embedding_config=self.user_embedding_config  # Pass user_embedding_config if analyze needs it for LLM keys/endpoints
            )
            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"User '{self.user_id}': Error in situate_context with LLM '{api_name_for_context}': {e}",
                         exc_info=True)
            # Depending on desired behavior, either return empty string or raise
            return ""  # Fail gracefully for contextualization

    def process_and_store_content(self,
                                  content: str,
                                  media_id: Union[int, str],  # TODO: Update type based on new MediaDatabase
                                  file_name: str,  # TODO: Get from new MediaDatabase if it stores this
                                  collection_name: Optional[str] = None,
                                  embedding_model_id_override: Optional[str] = None,
                                  create_embeddings: bool = True,
                                  create_contextualized: bool = False,  # Default to False due to cost/speed
                                  llm_model_for_context: Optional[str] = None,  # e.g., "gpt-3.5-turbo"
                                  chunk_options: Optional[Dict] = None):
        """
        Processes content by chunking, optionally contextualizing, generating embeddings,
        and storing them in ChromaDB and references in SQL DB.
        """
        target_collection = self.get_or_create_collection(collection_name)

        current_op_embedding_model_id = embedding_model_id_override or self.default_embedding_model_id
        if not current_op_embedding_model_id and create_embeddings:
            logger.error(
                f"User '{self.user_id}': No embedding model ID (default or override) for media_id {media_id}. Cannot create embeddings.")
            raise ValueError("Embedding model ID not specified for content processing with embeddings.")

        effective_llm_model_for_context = llm_model_for_context or self.embedding_config.get(
            "default_llm_for_contextualization", "gpt-3.5-turbo")

        logger.info(
            f"User '{self.user_id}': Processing content for media_id {media_id} "
            f"in collection '{target_collection.name}' using embedding_model_id '{current_op_embedding_model_id or 'N/A'}'. "
            f"Contextualization: {create_contextualized} with LLM '{effective_llm_model_for_context if create_contextualized else 'N/A'}'."
        )
        try:
            # Chunking
            chunks = chunk_for_embedding(content, file_name, chunk_options)
            if not chunks:
                logger.warning(
                    f"User '{self.user_id}': No chunks generated for media_id {media_id}, file {file_name}. Skipping storage.")
                return

            # TODO: Point 6 - MediaDatabase interaction
            # Placeholder for new MediaDatabase interactions:
            # sql_db_chunks_to_add = []
            # for i, chunk_info in enumerate(chunks):
            #     sql_db_chunks_to_add.append({
            #         "text": chunk_info['text'],
            #         "start_index": chunk_info['metadata'].get('start_index'),
            #         "end_index": chunk_info['metadata'].get('end_index'),
            #         # ... other fields for the new MediaDatabase ...
            #     })
            # if sql_db_chunks_to_add:
            #     # media_db_instance.add_media_chunks_in_batches(media_id=media_id, chunks_to_add=sql_db_chunks_to_add)
            #     logger.info(f"User '{self.user_id}': TODO - Stored {len(sql_db_chunks_to_add)} chunk references in SQL DB for media_id {media_id}.")
            # End TODO MediaDatabase

            if create_embeddings:
                docs_for_chroma = []  # This will hold the text that's actually stored in Chroma
                texts_for_embedding_generation = []  # This will hold the text used to generate embeddings

                for chunk in chunks:
                    chunk_text = chunk['text']
                    docs_for_chroma.append(chunk_text)  # Store original chunk text in Chroma document

                    if create_contextualized:
                        context_summary = self.situate_context(effective_llm_model_for_context, content, chunk_text)
                        # Embed the chunk + context, but store only the chunk text (or chunk+context if preferred)
                        text_to_embed = f"{chunk_text}\n\nContextual Summary: {context_summary}"
                        texts_for_embedding_generation.append(text_to_embed)
                        # If you want to store the contextualized text in Chroma's document field:
                        # docs_for_chroma[-1] = text_to_embed
                    else:
                        texts_for_embedding_generation.append(chunk_text)

                if not texts_for_embedding_generation:
                    logger.warning(
                        f"User '{self.user_id}': No texts prepared for embedding for media_id {media_id}. Skipping embedding creation.")
                else:
                    # TODO: Point 4 - Async/Batching for create_embeddings_batch if it supports async
                    embeddings = create_embeddings_batch(
                        texts=texts_for_embedding_generation,
                        user_embedding_config=self.user_embedding_config,
                        model_id_override=current_op_embedding_model_id
                    )

                    ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]  # 0-indexed chunks

                    metadatas = []
                    for i, chunk_info in enumerate(chunks):
                        meta = {
                            "media_id": str(media_id),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_name": str(file_name),  # Or chunk_info['metadata']['file_name']
                            "contextualized": create_contextualized,
                            # Store original text for reference even if docs_for_chroma has contextualized text
                            "original_chunk_text_ref": chunk_info['text'][:200] + "..." if len(
                                chunk_info['text']) > 200 else chunk_info['text']
                        }
                        # Add metadata from chunk_for_embedding
                        meta.update(chunk_info.get('metadata', {}))

                        if create_contextualized:
                            # If docs_for_chroma contains original text, but texts_for_embedding_generation has context,
                            # you might want to store the generated context summary in metadata.
                            context_part = texts_for_embedding_generation[i].split("\n\nContextual Summary: ", 1)
                            if len(context_part) > 1:
                                meta["contextual_summary_ref"] = context_part[1]

                        metadatas.append(meta)

                    self.store_in_chroma(
                        collection_name=target_collection.name,
                        texts=docs_for_chroma,  # Text to store in ChromaDB document field
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas,
                        embedding_model_id_for_dim_check=current_op_embedding_model_id
                    )

            # TODO: Point 6 - MediaDatabase interaction
            # mark_media_as_processed(media_db_instance, media_id)
            # media_db_instance.execute_query(
            # "INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?",
            # (media_id,),
            # commit=True
            # )
            logger.info(f"User '{self.user_id}': TODO - Mark media {media_id} as processed and update FTS.")
            # End TODO MediaDatabase

            logger.info(f"User '{self.user_id}': Finished processing and storing content for media_id {media_id}")

        except ValueError as ve:  # Catch specific configuration/input errors
            logger.error(f"User '{self.user_id}': Input or configuration error processing media_id {media_id}: {ve}",
                         exc_info=True)
            raise  # Re-raise to signal failure
        except RuntimeError as rte:  # Catch ChromaDB or system-level issues
            logger.error(f"User '{self.user_id}': Runtime error processing media_id {media_id}: {rte}", exc_info=True)
            raise
        except Exception as e:  # General catch-all
            logger.error(
                f"User '{self.user_id}': Unexpected error in process_and_store_content for media_id {media_id}: {e}",
                exc_info=True)
            raise  # Re-raise for unhandled issues

    def store_in_chroma(self, collection_name: Optional[str], texts: List[str],
                        embeddings: Union[np.ndarray, List[List[float]]],  # Type hint improved
                        ids: List[str], metadatas: List[Dict[str, Any]],
                        embedding_model_id_for_dim_check: Optional[str] = None):
        """Stores embeddings and associated data into a ChromaDB collection."""
        if not texts or not ids or not metadatas or embeddings is None or len(
                embeddings) == 0:  # Check embeddings has content
            logger.error(
                f"User '{self.user_id}': Invalid input to store_in_chroma. Texts, ids, metadatas, or embeddings are empty/None.")
            raise ValueError(
                "Texts, ids, metadatas lists must be non-empty, and embeddings must be provided and non-empty.")

        if not (len(texts) == len(embeddings) == len(ids) == len(metadatas)):
            error_msg = (f"Input list length mismatch: Texts({len(texts)}), Embeddings({len(embeddings)}), "
                         f"IDs({len(ids)}), Metadatas({len(metadatas)})")
            logger.error(f"User '{self.user_id}': {error_msg}")
            raise ValueError(error_msg)

        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        elif isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
            embeddings_list = embeddings
        else:
            logger.error(
                f"User '{self.user_id}': Embeddings type mismatch. Expected List[List[float]] or np.ndarray, got {type(embeddings)}.")
            raise TypeError("Embeddings must be a list of lists (vectors) or a 2D numpy array.")

        if not embeddings_list or not isinstance(embeddings_list[0], list) or not embeddings_list[0]:
            logger.error(f"User '{self.user_id}': Embeddings list is empty or malformed after conversion.")
            raise ValueError("No valid embeddings provided after potential conversion.")

        target_collection = self.get_or_create_collection(collection_name)
        new_embedding_dim = len(embeddings_list[0])

        with self._lock:
            logger.info(
                f"User '{self.user_id}': Attempting to store {len(embeddings_list)} embeddings (dim: {new_embedding_dim}) "
                f"in ChromaDB Collection: '{target_collection.name}'.")
            try:
                cleaned_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]

                # Dimension Check and Collection Recreation (if needed)
                # This check is more robust if the collection stores its expected dimension in metadata
                collection_meta = target_collection.metadata
                existing_dim_from_meta = None
                if collection_meta and "embedding_dimension" in collection_meta:
                    existing_dim_from_meta = int(collection_meta["embedding_dimension"])

                if existing_dim_from_meta and existing_dim_from_meta != new_embedding_dim:
                    logger.warning(
                        f"User '{self.user_id}': Embedding dimension mismatch for collection '{target_collection.name}'. "
                        f"Collection expected dim (from metadata): {existing_dim_from_meta}, New: {new_embedding_dim} "
                        f"(from model_id '{embedding_model_id_for_dim_check or 'Unknown'}'). Recreating collection."
                    )
                    self.client.delete_collection(name=target_collection.name)
                    new_coll_meta = {"embedding_dimension": new_embedding_dim}
                    if embedding_model_id_for_dim_check:
                        new_coll_meta["source_model_id"] = embedding_model_id_for_dim_check
                    target_collection = self.client.create_collection(name=target_collection.name,
                                                                      metadata=new_coll_meta)
                elif not existing_dim_from_meta and target_collection.count() > 0:  # Has items but no dim in meta
                    # Fallback: get an existing embedding to check dimension
                    existing_item = target_collection.get(limit=1, include=['embeddings'])
                    if existing_item['embeddings'] and existing_item['embeddings'][0]:
                        existing_dim_from_sample = len(existing_item['embeddings'][0])
                        if existing_dim_from_sample != new_embedding_dim:
                            logger.warning(
                                f"User '{self.user_id}': Dim mismatch (sampled). Existing: {existing_dim_from_sample}, New: {new_embedding_dim}. Recreating '{target_collection.name}'."
                            )
                            self.client.delete_collection(name=target_collection.name)
                            new_coll_meta = {"embedding_dimension": new_embedding_dim}
                            if embedding_model_id_for_dim_check: new_coll_meta[
                                "source_model_id"] = embedding_model_id_for_dim_check
                            target_collection = self.client.create_collection(name=target_collection.name,
                                                                              metadata=new_coll_meta)

                # Batch upsert for potentially large number of embeddings
                # ChromaDB's upsert handles batching internally, but if we had extremely large lists,
                # we might do it with self._batched here. For now, single upsert is fine.
                target_collection.upsert(
                    documents=texts,
                    embeddings=embeddings_list,
                    ids=ids,
                    metadatas=cleaned_metadatas
                )
                logger.info(
                    f"User '{self.user_id}': Successfully upserted {len(embeddings_list)} items to '{target_collection.name}'.")

            except chromadb.errors.ChromaError as ce:  # Catch specific ChromaDB errors
                logger.error(
                    f"User '{self.user_id}': ChromaDB error in store_in_chroma for collection '{target_collection.name}': {ce}",
                    exc_info=True)
                raise RuntimeError(f"ChromaDB operation failed: {ce}") from ce
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Unexpected error in store_in_chroma for collection '{target_collection.name}': {e}",
                    exc_info=True)
                raise RuntimeError(f"Unexpected error during ChromaDB storage: {e}") from e
        return target_collection

    def vector_search(self, query: str, collection_name: Optional[str] = None, k: int = 10,
                      embedding_model_id_override: Optional[str] = None,
                      where_filter: Optional[Dict[str, Any]] = None,
                      # Use the Literal type for include_fields
                      include_fields: Optional[List[ChromaIncludeLiteral]] = None
                      ) -> List[Dict[str, Any]]:
        """Performs a vector search in the specified collection."""
        target_collection = self.get_or_create_collection(collection_name)

        query_embedding_model_id = embedding_model_id_override or self.default_embedding_model_id
        if not query_embedding_model_id:
            logger.error(
                f"User '{self.user_id}': No embedding model ID (default or override) for vector search. Cannot generate query embedding.")
            raise ValueError("Embedding model ID not specified for vector search.")

        # The default value must also conform to List[ChromaIncludeLiteral]
        effective_include_fields: List[ChromaIncludeLiteral]
        if include_fields is None:
            effective_include_fields = ["documents", "metadatas", "distances"]
        else:
            effective_include_fields = include_fields  # Assume caller provides a correctly typed list

        with self._lock:
            try:
                logger.info(
                    f"User '{self.user_id}': Vector search in '{target_collection.name}' for query: '{query[:50]}...' "
                    f"using model_id '{query_embedding_model_id}'. k={k}, Filter: {where_filter is not None}."
                )

                # Corrected call to create_embedding (from a previous iteration)
                query_embedding_single: List[float] = create_embedding(
                    text=query,
                    user_embedding_config=self.user_embedding_config,  # Pass the main app_config
                    model_id_override=query_embedding_model_id
                )

                if not query_embedding_single or not isinstance(query_embedding_single, list) or \
                        not all(isinstance(x, (float, int)) for x in query_embedding_single):
                    logger.error(
                        f"User '{self.user_id}': create_embedding did not return a valid List[float] for query '{query[:50]}...'. Got: {type(query_embedding_single)}")
                    if not query_embedding_single:
                        raise ValueError(f"Failed to generate query embedding for query: {query[:50]}...")
                    raise TypeError(f"Query embedding is malformed: {query_embedding_single}")

                query_embedding_list_for_chroma: List[List[float]] = [query_embedding_single]

                if not query_embedding_list_for_chroma or not query_embedding_list_for_chroma[0]:
                    logger.error(
                        f"User '{self.user_id}': Failed to prepare a valid query embedding list for '{query[:50]}...'.")
                    return []

                results: QueryResult = target_collection.query(
                    query_embeddings=query_embedding_list_for_chroma,
                    n_results=k,
                    where=self._clean_metadata(where_filter) if where_filter else None,
                    include=effective_include_fields  # Pass the correctly typed list
                )

                # Process results
                output = []
                if not results or not results.get('ids') or not results['ids'][0]:
                    logger.info(
                        f"User '{self.user_id}': No results found for the query in collection '{target_collection.name}'.")
                    return []

                num_results_for_first_query = len(results['ids'][0])
                for i in range(num_results_for_first_query):
                    item = {}
                    # Check if the key exists in results before trying to access results[key][0]
                    # And also check if the specific field was requested in effective_include_fields
                    if "documents" in effective_include_fields and results.get('documents') and results['documents'] and \
                            results['documents'][0]:
                        item["content"] = results['documents'][0][i]
                    if "metadatas" in effective_include_fields and results.get('metadatas') and results['metadatas'] and \
                            results['metadatas'][0]:
                        item["metadata"] = results['metadatas'][0][i]
                    if "distances" in effective_include_fields and results.get('distances') and results['distances'] and \
                            results['distances'][0]:
                        item["distance"] = results['distances'][0][i]
                    if "embeddings" in effective_include_fields and results.get('embeddings') and results[
                        'embeddings'] and results['embeddings'][0]:
                        item["embedding"] = results['embeddings'][0][i]
                    if "uris" in effective_include_fields and results.get('uris') and results['uris'] and \
                            results['uris'][0]:  # Added uris
                        item["uri"] = results['uris'][0][i]
                    if "data" in effective_include_fields and results.get('data') and results['data'] and \
                            results['data'][0]:  # Added data
                        item["data"] = results['data'][0][i]

                    # IDs are generally always included by ChromaDB if results are found
                    if results.get('ids') and results['ids'][0]:
                        item["id"] = results['ids'][0][i]
                    else:  # Should not happen if num_results_for_first_query > 0
                        logger.warning(
                            f"User '{self.user_id}': Missing 'ids' in results despite having num_results > 0. Index: {i}")
                        continue  # Skip this potentially incomplete result item

                    output.append(item)

                logger.info(
                    f"User '{self.user_id}': Found {len(output)} results for query in '{target_collection.name}'.")
                return output

            except ValueError as ve:
                logger.error(
                    f"User '{self.user_id}': Value error during vector search in '{target_collection.name}': {ve}",
                    exc_info=True)
                raise
            # Use specific ChromaError imports if they work
            except Exception as e:  # More general catch
                error_str = str(e).lower()
                # Try to identify if it's a Chroma-related "collection not found"
                # This is more heuristic and less reliable than specific exception types
                is_chroma_not_found = (
                        type(e).__module__.startswith('chromadb.') and  # Check if error originates from chromadb
                        "collection" in error_str and
                        ("not found" in error_str or "does not exist" in error_str)
                )

                if is_chroma_not_found:
                    # For vector_search:
                    logger.warning(
                        f"User '{self.user_id}': Collection '{target_collection.name}' not found during search: {e}")
                    return []
                    # For delete_collection:
                    # logger.warning(f"User '{self.user_id}': Collection '{collection_name}' not found for deletion: {e}")
                    # # Potentially do not raise
                elif type(e).__module__.startswith('chromadb.'):  # Other Chroma-originated error
                    logger.error(f"User '{self.user_id}': ChromaDB-related error: {e}", exc_info=True)
                    raise RuntimeError(f"ChromaDB operation failed: {e}") from e
                else:  # Other unexpected error
                    logger.error(f"User '{self.user_id}': Unexpected error: {e}", exc_info=True)
                    raise RuntimeError(f"Unexpected error during operation: {e}") from e

    def reset_chroma_collection(self, collection_name: Optional[str] = None):
        """Resets (deletes and recreates) a ChromaDB collection."""
        name_to_reset = collection_name or self.get_user_default_collection_name()
        with self._lock:
            try:
                logger.info(f"User '{self.user_id}': Attempting to reset ChromaDB collection: '{name_to_reset}'.")
                self.client.delete_collection(name=name_to_reset)
                # No specific metadata needed on basic recreate, store_in_chroma will handle dim metadata
                self.client.create_collection(name=name_to_reset)
                logger.info(f"User '{self.user_id}': Successfully reset ChromaDB collection: '{name_to_reset}'.")
            except chromadb.errors.ChromaError as ce:
                # If deleting a non-existent collection, Chroma might error.
                # We still want to ensure it's created.
                if "does not exist" in str(ce).lower():  # Check if it's an error about non-existence
                    logger.warning(
                        f"User '{self.user_id}': Collection '{name_to_reset}' did not exist during delete for reset. Will attempt creation.")
                else:  # Other Chroma error during delete
                    logger.error(
                        f"User '{self.user_id}': ChromaDB error deleting collection '{name_to_reset}' during reset: {ce}",
                        exc_info=True)
                    # Decide if we should still try to create or raise
                try:
                    self.client.create_collection(name=name_to_reset)  # Attempt creation
                    logger.info(
                        f"User '{self.user_id}': Created ChromaDB collection '{name_to_reset}' after (failed) delete attempt.")
                except Exception as ice:  # Inner create exception
                    logger.error(
                        f"User '{self.user_id}': Failed to create collection '{name_to_reset}' after reset attempt: {ice}",
                        exc_info=True)
                    raise RuntimeError(f"Failed to finalize reset for collection '{name_to_reset}': {ice}") from ice
            except Exception as e:  # Catch other errors during delete
                logger.error(f"User '{self.user_id}': Unexpected error resetting collection '{name_to_reset}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error during collection reset: {e}") from e

    def delete_from_collection(self, ids: List[str], collection_name: Optional[str] = None):
        """Deletes items from a collection by their IDs."""
        if not ids:
            logger.warning(f"User '{self.user_id}': No IDs provided for deletion. Skipping.")
            return

        target_collection = self.get_or_create_collection(collection_name)  # Ensures collection exists
        with self._lock:
            try:
                target_collection.delete(ids=ids)
                logger.info(f"User '{self.user_id}': Deleted IDs {ids} from collection '{target_collection.name}'.")
            except chromadb.errors.ChromaError as ce:
                logger.error(f"User '{self.user_id}': ChromaDB error deleting from '{target_collection.name}': {ce}",
                             exc_info=True)
                raise RuntimeError(f"ChromaDB deletion failed: {ce}") from ce
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error deleting from '{target_collection.name}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error during deletion: {e}") from e

    def query_collection_with_precomputed_embeddings(
            self, query_embeddings: List[List[float]],
            n_results: int = 5,
            where_clause: Optional[Dict[str, Any]] = None,
            collection_name: Optional[str] = None,
            # Use the Literal type for include_fields
            include_fields: Optional[List[ChromaIncludeLiteral]] = None
    ) -> QueryResult:
        """
        Queries a collection using pre-computed embeddings.
        Args:
            query_embeddings (List[List[float]]): A list of query embedding vectors.
            include_fields (Optional[List[ChromaIncludeLiteral]]): A list of fields to include in the results.
        """
        target_collection = self.get_or_create_collection(collection_name)

        # The default value must also conform to List[ChromaIncludeLiteral]
        effective_include_fields: List[ChromaIncludeLiteral]
        if include_fields is None:
            effective_include_fields = ["documents", "metadatas", "distances"]
        else:
            effective_include_fields = include_fields

        with self._lock:
            try:
                if not query_embeddings or not query_embeddings[0]:  # Check the first embedding vector exists
                    logger.error(
                        f"User '{self.user_id}': Empty or malformed query_embeddings provided to query_collection_with_precomputed_embeddings.")
                    raise ValueError("Query embeddings cannot be empty and must contain valid vectors.")

                # Ensure all sub-lists (vectors) in query_embeddings are not empty
                if any(not vec for vec in query_embeddings):
                    logger.error(
                        f"User '{self.user_id}': One or more embedding vectors in query_embeddings is empty.")
                    raise ValueError("All embedding vectors in query_embeddings must be non-empty.")

                return target_collection.query(
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=self._clean_metadata(where_clause) if where_clause else None,
                    include=effective_include_fields  # Pass the correctly typed list
                )
            # Use the more specific ChromaError imports if they work for your version
            except ChromaError as ce:
                logger.error(
                    f"User '{self.user_id}': ChromaDB error querying collection '{target_collection.name}': {ce}",
                    exc_info=True)
                raise RuntimeError(f"ChromaDB query with precomputed embeddings failed: {ce}") from ce
            except ValueError as ve:  # Catch ValueErrors from our checks
                logger.error(
                    f"User '{self.user_id}': Input validation error in query_collection_with_precomputed_embeddings: {ve}",
                    exc_info=True)
                raise  # Re-raise the ValueError
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Unexpected error querying '{target_collection.name}' with precomputed embeddings: {e}",
                    exc_info=True)
                raise RuntimeError(f"Unexpected error during query with precomputed embeddings: {e}") from e

    def count_items_in_collection(self, collection_name: Optional[str] = None) -> int:
        """Counts the number of items in a collection."""
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                return target_collection.count()
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Error counting items in collection '{target_collection.name}': {e}",
                    exc_info=True)
                # Depending on severity, either return 0 or raise
                raise RuntimeError(f"Failed to count items in collection: {e}") from e

    def list_collections(self) -> Sequence[Collection]:
        """Lists all collections for the current user's client."""
        with self._lock:
            try:
                return self.client.list_collections()
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error listing collections: {e}", exc_info=True)
                raise RuntimeError(f"Failed to list collections: {e}") from e

    def delete_collection(self, collection_name: str):
        """Deletes a specific collection by name."""
        if not collection_name:
            raise ValueError("collection_name must be provided for deletion.")
        with self._lock:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"User '{self.user_id}': Successfully deleted collection '{collection_name}'.")
            except chromadb.errors.ChromaError as ce:
                # Handle if collection doesn't exist gracefully or re-raise
                logger.warning(
                    f"User '{self.user_id}': ChromaDB error deleting collection '{collection_name}' (it might not exist): {ce}")
                # Depending on desired strictness, you might not re-raise if it's "does not exist"
                if "does not exist" not in str(ce).lower():
                    raise RuntimeError(f"ChromaDB failed to delete collection '{collection_name}': {ce}") from ce
            except Exception as e:
                logger.error(f"User '{self.user_id}': Unexpected error deleting collection '{collection_name}': {e}",
                             exc_info=True)
                raise RuntimeError(f"Unexpected error deleting collection '{collection_name}': {e}") from e

# Example of how you might instantiate and use it (outside this file):
# from tldw_Server_API.app.core.config import settings
# from tldw_Server_API.app.core.Vector_DB_Management.ChromaDB_Library import ChromaDBManager

# current_user_id = "user123" # Get this from auth context
# chroma_manager = ChromaDBManager(user_id=current_user_id, settings=settings)
# results = chroma_manager.vector_search(query="some query text", k=5)
# chroma_manager.store_in_chroma(texts=["..."], embeddings=[[...]], ids=["..."], metadatas=[{...}])
#
# End of Functions for ChromaDB
#######################################################################################################################

# Compatibility layer for legacy code expecting module-level functions
# This creates a default instance for single-user mode or tests
_default_chroma_manager = None
_manager_lock = threading.Lock()

def get_default_chroma_manager():
    """Get or create the default ChromaDB manager for backward compatibility."""
    global _default_chroma_manager
    with _manager_lock:
        if _default_chroma_manager is None:
            # Use default user ID 0 for single-user mode
            from tldw_Server_API.app.core.config import settings
            user_id = str(settings.get("SINGLE_USER_FIXED_ID", "0"))
            embedding_config = settings.get("EMBEDDING_CONFIG", {})
            _default_chroma_manager = ChromaDBManager(user_id=user_id, user_embedding_config=embedding_config)
        return _default_chroma_manager

# Legacy function exports for backward compatibility
def store_in_chroma(texts, embeddings, ids, metadatas, collection_name="default_collection"):
    """Legacy function for storing embeddings in ChromaDB."""
    manager = get_default_chroma_manager()
    return manager.store_in_chroma(texts=texts, embeddings=embeddings, ids=ids, 
                                  metadatas=metadatas, collection_name=collection_name)

# Create a chroma_client property for backward compatibility
class ChromaClientProxy:
    """Proxy object that delegates to the default manager's chroma_client."""
    def __getattr__(self, name):
        manager = get_default_chroma_manager()
        return getattr(manager.chroma_client, name)

chroma_client = ChromaClientProxy()
