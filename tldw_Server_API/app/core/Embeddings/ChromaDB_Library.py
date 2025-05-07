# ChromaDB_Library.py
# Description: Functions for managing embeddings in ChromaDB
#
# Imports:
from typing import List, Dict, Any
import threading
# 3rd-Party Imports:
import chromadb
from chromadb import Settings
from itertools import islice
import numpy as np
#
# Local Imports:
from tldw_Server_API.app.core.Utils.Chunk_Lib import chunk_for_embedding, chunk_options
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.Media_DB import process_chunks
from tldw_Server_API.app.core.Embeddings.Embeddings_Create import create_embedding, create_embeddings_batch
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Utils import get_database_path, ensure_directory_exists, load_and_log_configs, logger, \
    logging
#
#######################################################################################################################
#
# Config Settings for ChromaDB Functions
#
# FIXME - Refactor so that all globals are set in summarize.py
# Load config
config = load_and_log_configs()
#
# ChromaDB settings
chroma_db_path = config['db_config']['chroma_db_path'] or get_database_path('chroma_db')
ensure_directory_exists(chroma_db_path)
chroma_client = chromadb.PersistentClient(path=chroma_db_path, settings=Settings(anonymized_telemetry=False))
#
# Embedding settings
embedding_provider = config['embedding_config']['embedding_provider'] or 'openai'
embedding_model = config['embedding_config']['embedding_model'] or 'text-embedding-3-small'
embedding_api_key = config['embedding_config']['embedding_api_key'] or ''
embedding_api_url = config['embedding_config']['embedding_api_url'] or ''
#
# End of Config Settings
#######################################################################################################################
#
# Functions:

#_chroma_lock = threading.Lock()
_chroma_lock = threading.RLock()

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def situate_context(api_name, doc_content: str, chunk_content: str) -> str:
    doc_content_prompt = f"""
    <document>
    {doc_content}
    </document>
    """

    chunk_context_prompt = f"""
    \n\n\n\n\n
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    response = analyze(chunk_context_prompt, doc_content_prompt, api_name, api_key=None, temp=0, system_message=None)
    return response


# FIXME - update all uses to reflect 'api_name' parameter
def process_and_store_content(database, content: str, collection_name: str, media_id: int, file_name: str,
                              create_embeddings: bool = True, create_contextualized: bool = True, api_name: str = "gpt-3.5-turbo",
                              chunk_options = None, embedding_provider: str = None,
                              embedding_model: str = None, embedding_api_url: str = None):
    try:
        logger.info(f"Processing content for media_id {media_id} in collection {collection_name}")

        chunks = chunk_for_embedding(content, file_name, chunk_options)

        # Process chunks synchronously
        process_chunks(database, chunks, media_id)

        if create_embeddings:
            texts = []
            contextualized_chunks = []
            for chunk in chunks:
                chunk_text = chunk['text']
                if create_contextualized:
                    context = situate_context(api_name, content, chunk_text)
                    contextualized_text = f"{chunk_text}\n\nContextual Summary: {context}"
                    contextualized_chunks.append(contextualized_text)
                else:
                    contextualized_chunks.append(chunk_text)
                texts.append(chunk_text)  # Store original text for database

            embeddings = create_embeddings_batch(contextualized_chunks, embedding_provider, embedding_model, embedding_api_url)
            ids = [f"{media_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]
            metadatas = [{
                "media_id": str(media_id),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "start_index": int(chunk['metadata']['start_index']),
                "end_index": int(chunk['metadata']['end_index']),
                "file_name": str(chunk['metadata']['file_name']),
                "relative_position": float(chunk['metadata']['relative_position']),
                "contextualized": create_contextualized,
                "original_text": chunk['text'],
                "contextual_summary": contextualized_chunks[i-1].split("\n\nContextual Summary: ")[-1] if create_contextualized else ""
            } for i, chunk in enumerate(chunks, 1)]

            store_in_chroma(collection_name, contextualized_chunks, embeddings, ids, metadatas)

            # Mark the media as processed
            mark_media_as_processed(database, media_id)

        # Update full-text search index
        database.execute_query(
            "INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?",
            (media_id,)
        )

        logger.info(f"Finished processing and storing content for media_id {media_id}")

    except Exception as e:
        logger.error(f"Error in process_and_store_content for media_id {media_id}: {str(e)}")
        raise

# Usage example:
# process_and_store_content(db, content, "my_collection", 1, "example.txt", create_embeddings=True, create_summary=True, api_name="gpt-3.5-turbo")


def check_embedding_status(selected_item, item_mapping):
    if not selected_item:
        return "Please select an item", ""
    logging.info("DEBUG: item_mapping type:", type(item_mapping), item_mapping)
    try:
        item_id = item_mapping.get(selected_item)
        if item_id is None:
            return f"Invalid item selected: {selected_item}", ""

        item_title = selected_item.rsplit(' (', 1)[0]
        with _chroma_lock:
            collection = chroma_client.get_or_create_collection(name="all_content_embeddings")

            result = collection.get(ids=[f"doc_{item_id}"], include=["embeddings", "metadatas"])
            logging.info(f"ChromaDB result for item '{item_title}' (ID: {item_id}): {result}")

            if not result['ids']:
                return f"No embedding found for item '{item_title}' (ID: {item_id})", ""

            if not result['embeddings'] or not result['embeddings'][0]:
                return f"Embedding data missing for item '{item_title}' (ID: {item_id})", ""

            embedding = result['embeddings'][0]
            metadata = result['metadatas'][0] if result['metadatas'] else {}
            embedding_preview = str(embedding[:50])
            status = f"Embedding exists for item '{item_title}' (ID: {item_id})"
            return status, f"First 50 elements of embedding:\n{embedding_preview}\n\nMetadata: {metadata}"

    except Exception as e:
        logging.error(f"Error in check_embedding_status: {str(e)}")
        return f"Error processing item: {selected_item}. Details: {str(e)}", ""

def reset_chroma_collection(collection_name: str):
    with _chroma_lock:
        try:
            chroma_client.delete_collection(collection_name)
            chroma_client.create_collection(collection_name)
            logging.info(f"Reset ChromaDB collection: {collection_name}")
        except Exception as e:
            logging.error(f"Error resetting ChromaDB collection: {str(e)}")


#v2
def store_in_chroma(collection_name: str, texts: List[str], embeddings: Any, ids: List[str],
                    metadatas: List[Dict[str, Any]]):
    """
    Stores text, embeddings, and metadata in ChromaDB using upsert.
    """
    # Input validation
    if not all([texts, embeddings, ids, metadatas]):
        raise ValueError("All input lists (texts, embeddings, ids, metadatas) must be non-empty.")

    if not (len(texts) == len(embeddings) == len(ids) == len(metadatas)):
        raise ValueError("All input lists must have the same length.")

    # Convert embeddings to list if it's a numpy array
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    elif not isinstance(embeddings, list):
        raise TypeError("Embeddings must be either a list or a numpy array")

    if not embeddings:  # Check for empty embeddings list after conversion
      raise ValueError("No embeddings provided")
    embedding_dim = len(embeddings[0])

    with _chroma_lock:
        logging.info(f"Storing embeddings in ChromaDB - Collection: {collection_name}")
        logging.info(f"Number of embeddings: {len(embeddings)}, Dimension: {embedding_dim}")

        try:
            # Clean metadata
            cleaned_metadatas = [clean_metadata(metadata) for metadata in metadatas]

            # Try to get or create the collection
            try:
                collection = chroma_client.get_collection(name=collection_name)
                logging.info(f"Existing collection '{collection_name}' found")

                # Check dimension of existing embeddings
                existing_embeddings = collection.get(limit=1, include=['embeddings'])['embeddings']
                if existing_embeddings:
                    existing_dim = len(existing_embeddings[0])
                    if existing_dim != embedding_dim:
                        logging.warning(f"Embedding dimension mismatch. Existing: {existing_dim}, New: {embedding_dim}")
                        logging.warning("Deleting existing collection and creating a new one")
                        chroma_client.delete_collection(name=collection_name)
                        collection = chroma_client.create_collection(name=collection_name)
                else:
                    logging.info("No existing embeddings in the collection")
            except Exception as e:
                logging.info(f"Collection '{collection_name}' not found. Creating new collection")
                collection = chroma_client.create_collection(name=collection_name)

            # Perform the upsert operation
            collection.upsert(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=cleaned_metadatas
            )
            logging.info(f"Successfully upserted {len(embeddings)} embeddings")

            # Verify all stored embeddings
            results = collection.get(ids=ids, include=["documents", "embeddings", "metadatas"])

            for i, doc_id in enumerate(ids):
                if results['embeddings'][i] is None:
                    raise ValueError(f"Failed to store embedding for {doc_id}")
                else:
                    logging.debug(f"Embedding stored successfully for {doc_id}")
                    logging.debug(f"Stored document preview: {results['documents'][i][:100]}...")
                    logging.debug(f"Stored metadata: {results['metadatas'][i]}")

            logging.info("Successfully stored and verified all embeddings in ChromaDB")

        except Exception as e:
            logging.error(f"Error in store_in_chroma: {str(e)}")
            raise

        return collection


# Function to perform vector search using ChromaDB + Keywords from the media_db
#v2
def vector_search(collection_name: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    with _chroma_lock:
        try:
            collection = chroma_client.get_collection(name=collection_name)

            # Fetch a sample of embeddings to check metadata
            sample_results = collection.get(limit=10, include=["metadatas"])
            if not sample_results.get('metadatas') or not any(sample_results['metadatas']):
                logging.warning(f"No metadata found in the collection '{collection_name}'. Skipping this collection.")
                return []

            # Check if all embeddings use the same model and provider
            embedding_models = [
                metadata.get('embedding_model') for metadata in sample_results['metadatas']
                if metadata and metadata.get('embedding_model')
            ]
            embedding_providers = [
                metadata.get('embedding_provider') for metadata in sample_results['metadatas']
                if metadata and metadata.get('embedding_provider')
            ]

            if not embedding_models or not embedding_providers:
                raise ValueError("Embedding model or provider information not found in metadata")

            embedding_model = max(set(embedding_models), key=embedding_models.count)
            embedding_provider = max(set(embedding_providers), key=embedding_providers.count)

            logging.info(f"Using embedding model: {embedding_model} from provider: {embedding_provider}")

            # Generate query embedding using the existing create_embedding function
            query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)

            # Ensure query_embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas"]
            )

            if not results['documents'][0]:
                logging.warning(f"No results found for the query in collection '{collection_name}'.")
                return []

            return [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        except Exception as e:
            logging.error(f"Error in vector_search for collection '{collection_name}': {str(e)}", exc_info=True)
            return []


def schedule_embedding(media_id: int, content: str, media_name: str):
    try:
        chunks = chunk_for_embedding(content, media_name, chunk_options)
        texts = [chunk['text'] for chunk in chunks]
        embeddings = create_embeddings_batch(texts, embedding_provider, embedding_model, embedding_api_url)
        ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "media_id": str(media_id),
            "chunk_index": i,
            "total_chunks": len(chunks),
            "start_index": chunk['metadata']['start_index'],
            "end_index": chunk['metadata']['end_index'],
            "file_name": media_name,
            "relative_position": chunk['metadata']['relative_position']
        } for i, chunk in enumerate(chunks)]
        with _chroma_lock:
            store_in_chroma("all_content_embeddings", texts, embeddings, ids, metadatas)

    except Exception as e:
        logging.error(f"Error scheduling embedding for media_id {media_id}: {str(e)}")


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata by removing None values and converting to appropriate types"""
    cleaned = {}
    for key, value in metadata.items():
        if value is not None:  # Skip None values
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, (np.int32, np.int64)):
                cleaned[key] = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                cleaned[key] = float(value)
            else:
                cleaned[key] = str(value)  # Convert other types to string
    return cleaned


def count_items_in_collection(collection_name):
    """
    Counts the number of items in a specified ChromaDB collection.
    Args:
        collection_name (str): The name of the collection.
    Returns:
        int: The number of items in the collection.
    """
    with _chroma_lock:
        collection = chroma_client.get_collection(name=collection_name)
        return collection.count()


def get_chroma_collection(collection_name):
    """Retrieves a specified ChromaDB collection.
    Args:
        collection_name (str): The name of the collection to retrieve.
    Returns:
        chromadb.Collection: The requested ChromaDB collection.
    """
    # Directly return the result of get_collection
    with _chroma_lock:
        return chroma_client.get_collection(collection_name)


def delete_from_chroma(collection_name, ids):
    """
    Deletes entries from a ChromaDB collection by their IDs.
    :param collection_name: Name of the collection.
    :param ids: List of IDs to delete.
    """
    with _chroma_lock:
        collection = chroma_client.get_collection(name=collection_name)
        collection.delete(ids=ids)


def query_chroma(collection_name, query_embedding, n_results=5, where_clause=None):
    """
    Queries ChromaDB for the most similar embeddings.
    :param collection_name: Name of the collection.
    :param query_embedding: The embedding to query for.
    :param n_results: Number of results to return.
    :param where_clause: Optional where clause for filtering results.
    :return: Query results.
    """
    with _chroma_lock:
        collection = chroma_client.get_collection(name=collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )

#
# End of Functions for ChromaDB
#######################################################################################################################
