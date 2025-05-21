# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
# 3rd-Party Imports:
import chromadb
from chromadb import Settings
from itertools import islice
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryResult
#
# Local Imports:
from tldw_Server_API.app.core.Chunking.Chunk_Lib import chunk_for_embedding
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.Embeddings.Embeddings_Create import create_embedding, create_embeddings_batch
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Utils import logger

#
#######################################################################################################################
#
# Functions:

#_chroma_lock = threading.Lock()
_chroma_lock = threading.RLock()

class ChromaDBManager:
    """
    Manages ChromaDB instances and operations for specific users.
    Each instance of this class corresponds to a user's isolated ChromaDB storage.
    """
    DEFAULT_COLLECTION_NAME_PREFIX = "user_embeddings_for_"

    def __init__(self, user_id: str, settings: Dict[str, Any]):
        """
        Initializes the ChromaDBManager for a specific user.

        Args:
            user_id (str): The ID of the user for whom this ChromaDB instance is.
            settings (Dict[str, Any]): The global application settings dictionary.
        """
        if not user_id:
            raise ValueError("user_id cannot be empty for ChromaDBManager.")
        if not settings:
            raise ValueError("settings cannot be empty for ChromaDBManager.")

        self.user_id = str(user_id)
        self.settings = settings
        self._lock = threading.RLock()

        user_base_path: Optional[Path] = self.settings.get("USER_DB_BASE_DIR")
        if not user_base_path:
            # This case should ideally be prevented by robust settings load,
            # but good to have a fallback or clearer error.
            logger.critical("USER_DB_BASE_DIR not found in settings. ChromaDBManager cannot be initialized.")
            raise ValueError("USER_DB_BASE_DIR not configured in application settings.")

        self.user_chroma_path: Path = (user_base_path / self.user_id / "chroma_storage").resolve()
        self.user_chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChromaDBManager for user '{self.user_id}' initialized. Path: {self.user_chroma_path}")

        self.client = chromadb.PersistentClient(
            path=str(self.user_chroma_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)  # allow_reset=True can be useful
        )

        embedding_config = self.settings.get("EMBEDDING_CONFIG", {})
        self.embedding_provider = embedding_config.get('embedding_provider', 'openai')
        self.embedding_model = embedding_config.get('embedding_model', 'text-embedding-3-small')
        self.embedding_api_key = embedding_config.get('embedding_api_key', '')
        self.embedding_api_url = embedding_config.get('embedding_api_url', '')

        logger.info(
            f"User '{self.user_id}' ChromaDBManager using Embedding Provider: {self.embedding_provider}, Model: {self.embedding_model}")

    def _batched(self, iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                return
            yield batch

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, (np.int32, np.int64)):
                    cleaned[key] = int(value)
                elif isinstance(value, (np.float32, np.float64)):
                    cleaned[key] = float(value)
                else:
                    cleaned[key] = str(value)
        return cleaned

    def get_user_default_collection_name(self) -> str:
        return f"{self.DEFAULT_COLLECTION_NAME_PREFIX}{self.user_id}"

    def get_or_create_collection(self, collection_name: Optional[str] = None) -> Collection:
        name_to_use = collection_name or self.get_user_default_collection_name()
        with self._lock:
            # Note: Default embedding function can be set at collection creation
            # if all embeddings in it will use the same model.
            # However, since we pass embeddings directly to upsert, this is less critical here.
            return self.client.get_or_create_collection(name=name_to_use)

    def situate_context(self, api_name: str, doc_content: str, chunk_content: str) -> str:
        doc_content_prompt = f"<document>\n{doc_content}\n</document>"
        chunk_context_prompt = (
            f"\n\n\n\n\nHere is the chunk we want to situate within the whole document\n<chunk>\n{chunk_content}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of the chunk.\n"
            "Answer only with the succinct context and nothing else."
        )
        response = analyze(chunk_context_prompt, doc_content_prompt, api_name, api_key=None, temp=0,
                           system_message=None)
        return response

    def process_and_store_content(self, media_db_instance: MediaDatabase, content: str,
                                  media_id: int, file_name: str,
                                  collection_name: Optional[str] = None,
                                  create_embeddings: bool = True, create_contextualized: bool = True,
                                  api_name: str = "gpt-3.5-turbo",
                                  chunk_options: Optional[Dict] = None):
        collection_to_use = self.get_or_create_collection(collection_name)
        target_collection_name = collection_to_use.name

        try:
            logger.info(
                f"User '{self.user_id}': Processing content for media_id {media_id} in collection {target_collection_name}")
            chunks = chunk_for_embedding(content, file_name, chunk_options)

            sql_db_chunks_to_add = []
            for i, chunk_info in enumerate(chunks):
                sql_db_chunks_to_add.append({
                    "text": chunk_info['text'],
                    "start_index": chunk_info['metadata'].get('start_index'),
                    "end_index": chunk_info['metadata'].get('end_index'),
                })

            if sql_db_chunks_to_add:
                media_db_instance.add_media_chunks_in_batches(media_id=media_id, chunks_to_add=sql_db_chunks_to_add)
                logger.info(
                    f"User '{self.user_id}': Stored {len(sql_db_chunks_to_add)} chunk references in SQL DB for media_id {media_id}.")

            if create_embeddings:
                texts_for_embedding = []
                contextualized_docs_for_chroma = []

                for chunk in chunks:
                    chunk_text = chunk['text']
                    texts_for_embedding.append(chunk_text)

                    if create_contextualized:
                        context = self.situate_context(api_name, content, chunk_text)
                        contextualized_text_for_embedding = f"{chunk_text}\n\nContextual Summary: {context}"
                        contextualized_docs_for_chroma.append(contextualized_text_for_embedding)
                    else:
                        contextualized_docs_for_chroma.append(chunk_text)

                # Corrected: Removed self.embedding_api_key from the call
                embeddings = create_embeddings_batch(
                    texts=contextualized_docs_for_chroma,
                    provider_override=self.embedding_provider,  # Use self for provider/model/url overrides
                    model_override=self.embedding_model,
                    api_url_override=self.embedding_api_url
                )

                ids = [f"{media_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]

                metadatas = []
                for i, chunk in enumerate(chunks, 1):
                    meta = {
                        "media_id": str(media_id),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "start_index": int(chunk['metadata']['start_index']),
                        "end_index": int(chunk['metadata']['end_index']),
                        "file_name": str(chunk['metadata']['file_name']),
                        "relative_position": float(chunk['metadata']['relative_position']),
                        "contextualized": create_contextualized,
                        "original_text": chunk['text']
                    }
                    if create_contextualized and len(contextualized_docs_for_chroma) >= i:
                        summary_part = contextualized_docs_for_chroma[i - 1].split("\n\nContextual Summary: ", 1)
                        if len(summary_part) > 1:
                            meta["contextual_summary"] = summary_part[1]
                        else:
                            meta["contextual_summary"] = ""
                    else:
                        meta["contextual_summary"] = ""
                    metadatas.append(meta)

                self.store_in_chroma(target_collection_name, contextualized_docs_for_chroma, embeddings, ids, metadatas)

                # Correctly call the standalone function imported from DB_Manager
                mark_media_as_processed(media_db_instance, media_id)

            media_db_instance.execute_query(
                "INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?",
                (media_id,),
                commit=True
            )
            logger.info(f"User '{self.user_id}': Finished processing and storing content for media_id {media_id}")

        except Exception as e:
            logger.error(f"User '{self.user_id}': Error in process_and_store_content for media_id {media_id}: {str(e)}",
                         exc_info=True)
            raise

    def store_in_chroma(self, collection_name: Optional[str], texts: List[str], embeddings: Any,
                        ids: List[str], metadatas: List[Dict[str, Any]]):
        if not all([texts, ids, metadatas]) or embeddings is None:  # Check embeddings is not None
            raise ValueError("Texts, ids, metadatas lists must be non-empty, and embeddings must be provided.")
        if not (len(texts) == len(embeddings) == len(ids) == len(metadatas)):
            raise ValueError("All input lists (texts, embeddings, ids, metadatas) must have the same length.")

        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        elif isinstance(embeddings, list):
            embeddings_list = embeddings
        else:
            raise TypeError("Embeddings must be either a list or a numpy array")

        if not embeddings_list:
            raise ValueError("No embeddings provided after potential conversion.")

        target_collection = self.get_or_create_collection(collection_name)
        embedding_dim = len(embeddings_list[0])

        with self._lock:
            logger.info(f"User '{self.user_id}': Storing embeddings in ChromaDB - Collection: {target_collection.name}")
            logger.info(f"Number of embeddings: {len(embeddings_list)}, Dimension: {embedding_dim}")

            try:
                cleaned_metadatas = [self._clean_metadata(metadata) for metadata in metadatas]

                existing_embeddings_sample_result = target_collection.get(limit=1, include=['embeddings'])
                existing_embeddings_sample = existing_embeddings_sample_result['embeddings']

                if existing_embeddings_sample and existing_embeddings_sample[0] is not None:
                    existing_dim = len(existing_embeddings_sample[0])
                    if existing_dim != embedding_dim:
                        logger.warning(
                            f"User '{self.user_id}': Embedding dimension mismatch for collection '{target_collection.name}'. Existing: {existing_dim}, New: {embedding_dim}. Deleting and recreating collection.")
                        self.client.delete_collection(name=target_collection.name)
                        target_collection = self.client.create_collection(name=target_collection.name)

                target_collection.upsert(
                    documents=texts,
                    embeddings=embeddings_list,
                    ids=ids,
                    metadatas=cleaned_metadatas
                )
                logger.info(
                    f"User '{self.user_id}': Successfully upserted {len(embeddings_list)} embeddings to '{target_collection.name}'")
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error in store_in_chroma: {str(e)}", exc_info=True)
                raise
        return target_collection

    def vector_search(self, query: str, collection_name: Optional[str] = None, k: int = 10) -> List[Dict[str, Any]]:
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                logger.info(
                    f"User '{self.user_id}': Performing vector search in collection '{target_collection.name}' for query: '{query[:50]}...'")

                # Corrected: Removed self.embedding_api_key from the call
                query_embedding_single = create_embedding(
                    text=query,
                    provider_override=self.embedding_provider,
                    model_override=self.embedding_model,
                    api_url_override=self.embedding_api_url
                )

                query_embedding_list_for_chroma: List[List[float]]
                if isinstance(query_embedding_single, np.ndarray):
                    query_embedding_list_for_chroma = [query_embedding_single.tolist()]
                elif isinstance(query_embedding_single, list) and query_embedding_single and isinstance(
                        query_embedding_single[0], (float, int)):
                    query_embedding_list_for_chroma = [query_embedding_single]
                elif isinstance(query_embedding_single, list) and query_embedding_single and isinstance(
                        query_embedding_single[0], list):  # Already a list of lists
                    query_embedding_list_for_chroma = query_embedding_single
                else:
                    logger.error(
                        f"User '{self.user_id}': create_embedding returned an unexpected type or structure: {type(query_embedding_single)}")
                    return []

                results: QueryResult = target_collection.query(
                    query_embeddings=query_embedding_list_for_chroma,
                    n_results=k,
                    include=["documents", "metadatas", "distances"]
                )

                documents = results.get('documents')
                metadatas = results.get('metadatas')
                distances = results.get('distances')

                if not documents or not documents[0]:
                    logger.warning(
                        f"User '{self.user_id}': No results found for the query in collection '{target_collection.name}'.")
                    return []

                output = []
                # documents[0], metadatas[0], distances[0] correspond to the results for the first query embedding
                res_docs = documents[0]
                res_metas = metadatas[0] if metadatas and metadatas[0] is not None else [None] * len(res_docs)
                res_dists = distances[0] if distances and distances[0] is not None else [None] * len(res_docs)

                for doc, meta, dist in zip(res_docs, res_metas, res_dists):
                    output.append({"content": doc, "metadata": meta, "distance": dist})

                logger.info(
                    f"User '{self.user_id}': Found {len(output)} results for query in '{target_collection.name}'.")
                return output
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Error in vector_search for collection '{target_collection.name}': {str(e)}",
                    exc_info=True)
                return []

    def reset_chroma_collection(self, collection_name: Optional[str] = None):
        name_to_reset = collection_name or self.get_user_default_collection_name()
        with self._lock:
            try:
                self.client.delete_collection(name=name_to_reset)
                self.client.create_collection(name=name_to_reset)
                logger.info(f"User '{self.user_id}': Reset ChromaDB collection: {name_to_reset}")
            except Exception as e:
                logger.warning(
                    f"User '{self.user_id}': Error resetting ChromaDB collection '{name_to_reset}': {str(e)}. It might not have existed.")
                try:
                    self.client.create_collection(name=name_to_reset)
                    logger.info(
                        f"User '{self.user_id}': Created ChromaDB collection after failed reset: {name_to_reset}")
                except Exception as ce:
                    logger.error(
                        f"User '{self.user_id}': Failed to create collection '{name_to_reset}' after reset attempt: {str(ce)}")

    def delete_from_collection(self, ids: List[str], collection_name: Optional[str] = None):
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                target_collection.delete(ids=ids)
                logger.info(f"User '{self.user_id}': Deleted IDs {ids} from collection '{target_collection.name}'.")
            except Exception as e:
                logger.error(
                    f"User '{self.user_id}': Error deleting from collection '{target_collection.name}': {str(e)}",
                    exc_info=True)
                raise

    def query_collection(self, query_embedding: List[float], n_results: int = 5,
                         where_clause: Optional[Dict[str, Any]] = None,
                         collection_name: Optional[str] = None) -> QueryResult: # Corrected return type
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            try:
                # Ensure query_embedding is wrapped in a list if it's a single embedding vector
                # ChromaDB's query_embeddings expects a list of embeddings (List[List[float]])
                query_embeddings_list: List[List[float]]
                if query_embedding and isinstance(query_embedding[0], (int, float)): # It's a single embedding vector
                    query_embeddings_list = [query_embedding]
                elif query_embedding and isinstance(query_embedding[0], list): # It's already a list of embeddings
                    query_embeddings_list = query_embedding
                else: # Fallback or error for unexpected structure
                    logger.warning(f"User '{self.user_id}': query_embedding has an unexpected structure in query_collection. Assuming it's a single embedding.")
                    query_embeddings_list = [query_embedding]


                return target_collection.query(
                    query_embeddings=query_embeddings_list,
                    n_results=n_results,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"User '{self.user_id}': Error querying collection '{target_collection.name}': {str(e)}",
                             exc_info=True)
                raise

    def count_items_in_collection(self, collection_name: Optional[str] = None) -> int:
        target_collection = self.get_or_create_collection(collection_name)
        with self._lock:
            return target_collection.count()

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
