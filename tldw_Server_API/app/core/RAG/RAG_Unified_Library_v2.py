# unified_rag_service.py
# Description: A singular RAG service consolidating RAG_QA_Chat, RAG_Library_2, and RAG_Persona_Chat.
# Updated to use CharactersRAGDB and Media_DB_v2.Database.

# Imports
# External Imports
import json
import tempfile
import threading
import time
import os
import configparser
from typing import List, Tuple, IO, Dict, Any, Optional
from enum import Enum

import chromadb
# 3rd-Party Imports
from flashrank import Ranker, RerankRequest

# Local Imports
# DB Management - New Libraries
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database as MediaDatabase, DatabaseError as MediaDBError, \
    DatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

# Embeddings & Vector DB
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import chroma_client, store_in_chroma
from tldw_Server_API.app.core.Embeddings.Embeddings_Create import create_embedding, embedding_provider, embedding_model, \
    embedding_api_url, create_embeddings_batch  # From RAG_Persona_Chat

# LLM Calls
# from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import \
#     analyze  # analyze was used by old generate_answer, now chat is used. Keep if needed elsewhere.

# Chat Functions (used by generate_answer)
from tldw_Server_API.app.core.Chat.Chat_Functions import process_user_input, ChatDictionary, \
    parse_user_dict_markdown_file, chat

# Web Scraping
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article  # From RAG_Library_2

# Utilities & Metrics
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.config import RAG_SEARCH_CONFIG, load_and_log_configs

#
########################################################################################################################
# Configuration Loading & Global Instances
########################################################################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
config = configparser.ConfigParser()

if os.path.exists(config_path):
    config.read(config_path)
elif os.path.exists(os.path.join(os.getcwd(), 'config.txt')):
    config.read(os.path.join(os.getcwd(), 'config.txt'))
else:
    logging.warning(
        f"Config file 'config.txt' not found in {config_path} or current working directory. Using default fallbacks.")

# Instantiate Ranker globally for performance
try:
    GLOBAL_RANKER = Ranker()
    logging.info("FlashRank Ranker initialized globally.")
except Exception as e:
    logging.error(f"Failed to initialize global FlashRank Ranker: {e}", exc_info=True)
    GLOBAL_RANKER = None

# Define an Enum for database types
class DatabaseType(Enum):
    MEDIA_DB = "Media DB"
    RAG_CHAT = "RAG Chat" # Covers Character Chat as well for FTS context
    CHARACTER_CHAT = "Character Chat" # Used for specific Chat RAG pipeline context
    RAG_NOTES = "RAG Notes"
    CHARACTER_CARDS = "Character Cards"


########################################################################################################################
# Chat History Management
########################################################################################################################
def save_chat_history(history: List[Tuple[str, str]]) -> str:
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    # NOTE: Using delete=False means these files are not automatically cleaned up.
    # Consider a proper storage strategy if these are not truly temporary.
    # FIXME
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_file:
            json.dump(history, temp_file)
            save_duration = time.time() - start_time
            log_histogram("save_chat_history_duration", save_duration)
            log_counter("save_chat_history_success")
            return temp_file.name
    except Exception as e:
        log_counter("save_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error saving chat history: {str(e)}")
        raise


def load_chat_history(file: IO[str]) -> List[Tuple[str, str]]:
    log_counter("load_chat_history_attempt")
    start_time = time.time()
    try:
        # Ensure file is read with utf-8 if it was written with it.
        # The IO[str] type hint usually implies text mode, which respects encoding.
        history = json.load(file)
        load_duration = time.time() - start_time
        log_histogram("load_chat_history_duration", load_duration)
        log_counter("load_chat_history_success")
        return history
    except Exception as e:
        log_counter("load_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error loading chat history: {str(e)}")
        raise

########################################################################################################################
# Persona Chat Embedding & Vector Search (from RAG_Persona_Chat.py)
# These functions interact with ChromaDB for chat message embeddings.
# The chat data itself (IDs, history) would come from CharactersRAGDB.
########################################################################################################################
# ... (embed_and_store_chat_messages and perform_vector_search_chat_messages unchanged for this pass) ...
# NOTE: embed_and_store_chat_messages still loops for create_embedding. Batching is preferred if available.
# NOTE: perform_vector_search_chat_messages assumes chroma_client.get_collection raises ValueError if not found.
#       This might depend on the chromadb version. `chromadb.errors.CollectionNotDefinedError` is more specific.
def embed_and_store_chat_messages(
        chat_id: str,  # Assuming conversation_id from CharactersRAGDB (which is UUID string)
        character_id: int,  # For metadata
        chat_history: List[Tuple[str, str]],  # List of (user_message, bot_response)
        conversation_title: Optional[str] = "Untitled Conversation"
):
    log_counter("embed_and_store_chat_messages_attempt", labels={"chat_id": chat_id, "character_id": character_id})
    start_time = time.time()
    try:
        collection_name = config.get('Embeddings', 'chat_embeddings_collection', fallback="chat_message_embeddings")
        texts_to_embed = []
        embeddings_list = []
        doc_ids = []
        metadatas_list = []

        for idx, (user_msg, bot_msg) in enumerate(chat_history, 1):
            combined_content = f"User: {user_msg}\nBot: {bot_msg}"
            texts_to_embed.append(combined_content)
            document_id = f"{chat_id}_turn_{idx}"
            doc_ids.append(document_id)
            metadata = {
                "conversation_id": chat_id,
                "character_id": character_id,
                "turn_index": idx,
                "conversation_title": conversation_title,
                "original_user_message": user_msg[:256],
                "original_bot_response": bot_msg[:256]
            }
            metadatas_list.append(metadata)

        if texts_to_embed:
            # NOTE: Batch create embeddings if your create_embedding function supports it.
            logging.info(f"Creating batch embeddings for {len(texts_to_embed)} chat turns...")
            embeddings_list = create_embeddings_batch(texts_to_embed,
                                                      provider_override=embedding_provider,
                                                      model_override=embedding_model,
                                                      api_url_override=embedding_api_url)
            logging.info(f"Finished batch embeddings for chat turns.")

            store_in_chroma(
                collection_name=collection_name,
                texts=texts_to_embed,
                embeddings=embeddings_list,
                ids=doc_ids,
                metadatas=metadatas_list
            )
            logging.info(
                f"Stored {len(texts_to_embed)} message turns for chat ID {chat_id} in ChromaDB collection '{collection_name}'.")

        duration = time.time() - start_time
        log_histogram("embed_and_store_chat_messages_duration", duration, labels={"chat_id": chat_id})
        log_counter("embed_and_store_chat_messages_success",
                    labels={"chat_id": chat_id, "message_count": len(texts_to_embed)})
    # CHANGED: Added more specific error handling for Chroma if possible, otherwise general
    except chromadb.errors.ChromaError as ce: # Example if chromadb has a base error
        log_counter("embed_and_store_chat_messages_error", labels={"chat_id": chat_id, "error_type": "ChromaError", "error": str(ce)})
        logging.error(f"ChromaDB error embedding/storing chat (ID: {chat_id}): {ce}", exc_info=True)
        # Decide if to raise. For now, logging.
    except Exception as e:
        log_counter("embed_and_store_chat_messages_error", labels={"chat_id": chat_id, "error_type": "General", "error": str(e)})
        logging.error(f"Error embedding and storing chat (ID: {chat_id}): {e}", exc_info=True)

def perform_vector_search_chat_messages(query: str, relevant_conversation_ids: List[str], k: int = 10) -> List[
    Dict[str, Any]]:
    log_counter("perform_vector_search_chat_messages_attempt")
    start_time = time.time()

    if not relevant_conversation_ids:
        logging.debug("No relevant_conversation_ids provided for chat message vector search.")
        return []

    try:
        collection_name = config.get('Embeddings', 'chat_embeddings_collection', fallback="chat_message_embeddings")
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except ValueError: # Or more specific like chromadb.errors.CollectionNotDefinedError
            logging.warning(f"ChromaDB collection '{collection_name}' not found for chat message vector search.")
            return []
        except chromadb.errors.CollectionNotDefinedError as ce: # Catch other chroma errors
            logging.error(f"ChromaDB error getting collection '{collection_name}': {ce}", exc_info=True)
            return []


        query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)
        if not query_embedding:
            logging.error("Failed to generate query embedding for chat message vector search.")
            return []

        where_filter = {"conversation_id": {"$in": relevant_conversation_ids}}
        results = collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                search_results.append({"content": doc, "metadata": meta, "distance": dist})
        search_results.sort(key=lambda x: x.get('distance', float('inf')))

        duration = time.time() - start_time
        log_histogram("perform_vector_search_chat_messages_duration", duration)
        log_counter("perform_vector_search_chat_messages_success", labels={"result_count": len(search_results)})
        return search_results
    # CHANGED: Added more specific error handling for Chroma if possible, otherwise general
    except chromadb.errors.ChromaError as ce:
        log_counter("perform_vector_search_chat_messages_error", labels={"error_type": "ChromaError", "error": str(ce)})
        logging.error(f"ChromaDB error in perform_vector_search_chat_messages: {ce}", exc_info=True)
        return []
    except Exception as e:
        log_counter("perform_vector_search_chat_messages_error", labels={"error_type": "General", "error": str(e)})
        logging.error(f"Error in perform_vector_search_chat_messages: {e}", exc_info=True)
        return []


########################################################################################################################
# Core RAG Functions (Primarily from RAG_Library_2.py, updated for new DBs)
########################################################################################################################

# NOTE: load_and_log_configs() is still called on every generate_answer.
# For a service, this should ideally be loaded once at startup or cached.
LOADED_APP_CONFIGS = None
CONFIG_LOAD_LOCK = threading.Lock()

def get_app_configs(): # CHANGED: Helper to load configs once
    global LOADED_APP_CONFIGS
    if LOADED_APP_CONFIGS is None:
        with CONFIG_LOAD_LOCK:
            if LOADED_APP_CONFIGS is None: # Double-check locking
                LOADED_APP_CONFIGS = load_and_log_configs()
    return LOADED_APP_CONFIGS


def generate_answer(api_choice: Optional[str], context: str, query: str) -> str:
    log_counter("generate_answer_attempt", labels={"api_choice": api_choice or "default"})
    start_time = time.time()
    logging.debug("Entering generate_answer function")

    # CHANGED: Use helper to get configs
    loaded_config_data = get_app_configs()
    if not loaded_config_data:
        logging.error("Failed to load configurations for generate_answer.")
        return "Error: System configuration missing."

    chat_dict_config = loaded_config_data.get('chat_dictionaries', {})
    rag_prompts_file_path = chat_dict_config.get('chat_dict_RAG_prompts')
    default_rag_prompt_template = chat_dict_config.get('default_rag_prompt', "Query: {query}") # default_RAG_prompt in original
    initial_query_for_chatdict = default_rag_prompt_template.replace("{query}", query)

    rag_prompt_entries: List[ChatDictionary] = []
    if rag_prompts_file_path and os.path.exists(rag_prompts_file_path):
        try:
            rag_prompt_dict_data = parse_user_dict_markdown_file(rag_prompts_file_path)
            for k, v_content in rag_prompt_dict_data.items(): # Iterate key-value pairs
                rag_prompt_entries.append(ChatDictionary(key=k, content=str(v_content))) # Ensure content is string
        except Exception as e:
            logging.error(f"Failed to parse RAG prompts dictionary from {rag_prompts_file_path}: {e}")

    processed_query_part = process_user_input(initial_query_for_chatdict, rag_prompt_entries)
    logging.debug(f"Processed query part for RAG: {processed_query_part}")

    max_context_len_chars = int(config.get('LLM_Limits', 'max_context_chars_rag', fallback=15000))
    if len(context) > max_context_len_chars:
        logging.warning(f"Context length ({len(context)} chars) exceeds limit ({max_context_len_chars}). Truncating.")
        context = context[:max_context_len_chars - len("... (context truncated)")] + "... (context truncated)"

    if api_choice:
        api_choice_lower = api_choice.lower()
        # Construct the key carefully based on your load_and_log_configs structure
        # It seems like it might be 'openai_api', 'anthropic_api' etc.
        api_config_key = f'{api_choice_lower}_api' # Assuming your config keys are like 'openai_api', 'gemini_api'
        specific_api_config = loaded_config_data.get(api_config_key, {})

        if not specific_api_config or 'api_key' not in specific_api_config : # Check if dict itself is empty or key missing
            logging.error(f"Configuration for API '{api_choice}' (key: {api_config_key}) not found or missing API key.")
            log_counter("generate_answer_error", labels={"api_choice": api_choice, "error": "API_config_missing"})
            return f"Error: Configuration for API '{api_choice}' is missing or incomplete."

        try:
            chat_message = processed_query_part
            # Ensure media_content is structured as expected by the `chat` function
            chat_media_content = {"rag_context": context} # Assuming 'rag_context' is the key it looks for
            chat_selected_parts = ["rag_context"] # This part must match a key in chat_media_content

            chat_api_key = specific_api_config['api_key']
            rag_system_message = (
                "You are a helpful AI assistant. Your task is to answer the user's question based "
                "on the provided 'rag_context'. Analyze the context thoroughly. "
                "If the context contains relevant information, use it to construct your answer. "
                "If the context does not seem relevant or is insufficient to answer the question, "
                "clearly state that the provided context is not helpful and then answer the question "
                "based on your general knowledge. Be concise and directly answer the question."
            )
            chat_temperature = float(specific_api_config.get('temperature', 0.7))
            chat_model = specific_api_config.get('model') # This should be the actual model name string
            chat_max_tokens = int(specific_api_config.get('max_tokens', 1000))
            chat_topp = specific_api_config.get('topp') # May need float conversion if not None
            chat_topk = specific_api_config.get('topk') # May need int conversion if not None
            chat_minp = specific_api_config.get('minp') # May need float
            chat_maxp = specific_api_config.get('maxp') # May need float

            # Convert to appropriate types if not None
            chat_topp = float(chat_topp) if chat_topp is not None else None
            chat_topk = int(chat_topk) if chat_topk is not None else None
            chat_minp = float(chat_minp) if chat_minp is not None else None
            chat_maxp = float(chat_maxp) if chat_maxp is not None else None


            logging.debug(f"Calling `chat` function with: api_endpoint='{api_choice_lower}', "
                          f"temperature={chat_temperature}, model='{chat_model}', max_tokens={chat_max_tokens}")

            result = chat(
                message=chat_message, history=[], media_content=chat_media_content,
                selected_parts=chat_selected_parts, api_endpoint=api_choice_lower, # This is the 'provider' like 'openai', 'anthropic'
                api_key=chat_api_key, custom_prompt=None, temperature=chat_temperature,
                system_message=rag_system_message, streaming=False, minp=chat_minp,
                maxp=chat_maxp, model=chat_model, topp=chat_topp, topk=chat_topk, # Pass the actual model name
                chatdict_entries=None, max_tokens=chat_max_tokens # Pass the pre-parsed rag_prompt_entries if needed by chat()
            )

            answer_generation_duration = time.time() - start_time
            log_histogram("generate_answer_duration", answer_generation_duration, labels={"api_choice": api_choice})
            log_counter("generate_answer_success", labels={"api_choice": api_choice})
            return result

        except Exception as e:
            log_counter("generate_answer_error", labels={"api_choice": api_choice, "error": str(e)})
            logging.error(f"Error in generate_answer calling `chat` function for API '{api_choice}': {str(e)}",
                          exc_info=True)
            return "An error occurred while generating the answer using the chat function."
    else:
        log_counter("generate_answer_error", labels={"api_choice": "None", "error": "API_choice_not_provided"})
        logging.error("API choice not provided to generate_answer.")
        return "Error: API choice not specified for generating answer."


def perform_general_vector_search(query: str, relevant_media_ids: Optional[List[str]] = None, top_k: int = 10) -> List[
    Dict[str, Any]]:
    log_counter("perform_general_vector_search_attempt")
    start_time = time.time()

    excluded_collections = {
        config.get('Embeddings', 'chat_embeddings_collection', fallback="chat_message_embeddings"),
        config.get('Embeddings', 'rag_notes_embeddings_collection', fallback="rag_notes_embeddings"),
        config.get('Embeddings', 'article_embeddings_collection', fallback="article_content_embeddings")
        # Add others if they are searched specifically elsewhere (e.g., future character_card_embeddings)
    }
    all_collections_objects = chroma_client.list_collections()
    general_purpose_collections_names = [
        col.name for col in all_collections_objects if col.name not in excluded_collections
    ]

    if not general_purpose_collections_names:
        logging.warning("No general-purpose ChromaDB collections found for vector search after exclusions.")
        return []

    vector_results: List[Dict[str, Any]] = []

    chat_collection_name = config.get('Embeddings', 'chat_embeddings_collection', fallback="chat_message_embeddings")
    # Consider if other patterns for CharactersRAGDB item collections should be excluded too
    # e.g., "rag_notes_embeddings", "character_card_embeddings"
    general_purpose_collections_names = [col.name for col in all_collections_objects if col.name != chat_collection_name]

    if not general_purpose_collections_names:
        logging.warning("No general-purpose ChromaDB collections found for vector search.")
        return []

    try:
        query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)
        if not query_embedding:
            logging.error("Failed to generate query embedding for general vector search.")
            return []

        for collection_name_str in general_purpose_collections_names:
            try:
                collection_instance = chroma_client.get_collection(name=collection_name_str)
            except ValueError: # Or more specific like chromadb.errors.CollectionNotDefinedError
                logging.warning(f"Collection '{collection_name_str}' could not be retrieved for general search, skipping.")
                continue
            except chromadb.errors.ChromaError as ce:
                 logging.error(f"ChromaDB error getting collection '{collection_name_str}': {ce}", exc_info=True)
                 continue


            # CHANGED: Use ChromaDB's native filtering if possible
            # The imported `chroma_vector_search` might not support a `where` clause.
            # We will use collection.query directly.
            # `chroma_vector_search` from ChromaDB_Library.py needs to be inspected.
            # Assuming it internally calls collection.query()
            # For this fix, I'll assume `chroma_vector_search` can somehow filter or we post-filter.
            # If `chroma_vector_search` doesn't support `where`, this filtering remains post-hoc.

            # Option 1: If chroma_vector_search supports a where filter (ideal)
            # where_filter_chroma = None
            # if relevant_media_ids:
            #     where_filter_chroma = {"media_id": {"$in": relevant_media_ids}} # Assuming 'media_id' is string in metadata
            # collection_results = chroma_vector_search(collection_name_str, query, k=top_k, where_filter=where_filter_chroma)

            # Option 2: Using collection.query directly (more control)
            current_where_filter = None
            if relevant_media_ids:
                # Ensure IDs are strings if metadata stores them as strings
                str_relevant_media_ids = [str(mid) for mid in relevant_media_ids]
                current_where_filter = {"media_id": {"$in": str_relevant_media_ids}}

            query_results_obj = collection_instance.query(
                query_embeddings=[query_embedding], # Must be a list of embeddings
                n_results=top_k,
                where=current_where_filter, # Pass the filter here
                include=["documents", "metadatas", "distances"]
            )

            collection_results_processed = []
            if query_results_obj['documents'] and query_results_obj['documents'][0]:
                for doc, meta, dist in zip(query_results_obj['documents'][0], query_results_obj['metadatas'][0], query_results_obj['distances'][0]):
                    collection_results_processed.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "score": 1 - dist # Example score if needed, assuming lower distance is better
                    })
            vector_results.extend(collection_results_processed)


        # Sort and trim results (as in original RAG_Library_2)
        if vector_results:
            sort_key_info = {}
            # Prefer distance if available as it's more standard for vector similarity
            if 'distance' in vector_results[0] and vector_results[0]['distance'] is not None:
                sort_key_info = {'key': 'distance', 'reverse': False, 'default': float('inf')}
            elif 'score' in vector_results[0] and vector_results[0]['score'] is not None:
                sort_key_info = {'key': 'score', 'reverse': True, 'default': -float('inf')}

            if sort_key_info:
                vector_results.sort(key=lambda x: x.get(sort_key_info['key'], sort_key_info['default']),
                                    reverse=sort_key_info['reverse'])
            else: # Fallback if no distance or score, or they are None
                logging.warning("No 'distance' or 'score' key found in vector search results for sorting.")


        final_vector_results = vector_results[:top_k]

        search_duration = time.time() - start_time
        log_histogram("perform_general_vector_search_duration", search_duration)
        log_counter("perform_general_vector_search_success", labels={"result_count": len(final_vector_results)})
        return final_vector_results
    # CHANGED: Added more specific error handling
    except chromadb.errors.ChromaError as ce:
        log_counter("perform_general_vector_search_error", labels={"error_type": "ChromaError", "error": str(ce)})
        logging.error(f"ChromaDB error in perform_general_vector_search: {ce}", exc_info=True)
        # raise # Decide whether to raise or return empty
        return []
    except Exception as e:
        log_counter("perform_general_vector_search_error", labels={"error_type": "General", "error": str(e)})
        logging.error(f"Error in perform_general_vector_search: {str(e)}", exc_info=True)
        raise # Re-raise general errors as it might be critical



def perform_full_text_search(
        media_db: Optional[MediaDatabase],
        char_rag_db: Optional[CharactersRAGDB],
        query: str,
        database_type: DatabaseType, # CHANGED: Use Enum
        relevant_ids: Optional[List[str]] = None,
        fts_top_k: int = RAG_SEARCH_CONFIG.get('fts_top_k', 10),
        search_fields_media: Optional[List[str]] = None,
        character_id_context: Optional[int] = None
) -> List[Dict[str, Any]]:
    log_counter("perform_full_text_search_attempt", labels={"database_type": database_type.value})
    start_time = time.time()
    effective_fts_top_k = fts_top_k if fts_top_k and fts_top_k > 0 else 10
    results: List[Dict[str, Any]] = []

    try:
        if database_type == DatabaseType.MEDIA_DB:
            if not media_db:
                raise ValueError("media_db instance is required for 'Media DB' search.")
            effective_search_fields = search_fields_media if search_fields_media else ["title", "content"]
            # Ensure relevant_ids are integers for MediaDB
            media_ids_int_filter = None
            if relevant_ids:
                media_ids_int_filter = [int(x) for x in relevant_ids if x.isdigit()]

            media_db_results, _ = media_db.search_media_db(
                search_query=query,
                search_fields=effective_search_fields,
                media_ids_filter=media_ids_int_filter,
                page=1, results_per_page=effective_fts_top_k
            )
            for item in media_db_results:
                item_id = item.get('id')
                item_uuid = item.get('uuid') # MediaDB items have UUIDs
                item_content = item.get('content', item.get('title', '')) # Fallback if content is empty
                results.append({
                    "content": item_content,
                    "metadata": {
                        'item_id': str(item_id) if item_id is not None else None,
                        'uuid': item_uuid,
                        'source_db': DatabaseType.MEDIA_DB.value,
                        'title': item.get('title'),
                        'media_id': str(item_id) if item_id is not None else None # For consistency
                    }
                })

        # CHANGED: Simplified this block with Enum and direct FTS for Character Cards
        elif database_type in [DatabaseType.RAG_CHAT, DatabaseType.CHARACTER_CHAT, DatabaseType.RAG_NOTES, DatabaseType.CHARACTER_CARDS]:
            if not char_rag_db:
                raise ValueError(f"char_rag_db instance is required for '{database_type.value}' search.")

            temp_results_from_db: List[Dict[str, Any]] = []

            if database_type == DatabaseType.RAG_CHAT or database_type == DatabaseType.CHARACTER_CHAT:
                # relevant_ids are conversation_ids (UUID strings)
                # character_id_context is for filtering conversations by character
                conv_ids_to_search_within = relevant_ids if relevant_ids else []

                if not conv_ids_to_search_within and character_id_context is not None:
                    # Fetch all convos for this character if no specific conv_ids given
                    char_conversations = char_rag_db.get_conversations_for_character(character_id_context, limit=1000)
                    conv_ids_to_search_within = [c['id'] for c in char_conversations]

                if conv_ids_to_search_within:
                    for conv_id_str in conv_ids_to_search_within:
                        # ChaChaNotes_DB.search_messages_by_content searches within ONE conversation
                        messages = char_rag_db.search_messages_by_content(
                            content_query=query,
                            conversation_id=conv_id_str,
                            limit=effective_fts_top_k # Get top_k for each relevant conversation then combine
                        )
                        temp_results_from_db.extend(messages)
                elif not relevant_ids and not character_id_context: # Broad search across all messages
                    messages = char_rag_db.search_messages_by_content(content_query=query, limit=effective_fts_top_k * 5)
                    temp_results_from_db.extend(messages)
                # else: no conv_ids and no character_id_context means no messages to search from specific convos

                # Deduplicate based on content (or ID if available and reliable)
                seen_message_content_fts = set()
                deduped_messages_fts = []
                for item in temp_results_from_db:
                    msg_content = item.get('content')
                    if msg_content and msg_content not in seen_message_content_fts:
                        deduped_messages_fts.append(item)
                        seen_message_content_fts.add(msg_content)
                temp_results_from_db = deduped_messages_fts[:effective_fts_top_k]


            elif database_type == DatabaseType.RAG_NOTES:
                temp_results_from_db = char_rag_db.search_notes(search_term=query, limit=effective_fts_top_k)
                if relevant_ids: # Post-filter if search_notes doesn't take ID list
                    temp_results_from_db = [r for r in temp_results_from_db if r.get('id') in relevant_ids]

            elif database_type == DatabaseType.CHARACTER_CARDS:
                # CHANGED: Use FTS from ChaChaNotes_DB directly
                temp_results_from_db = char_rag_db.search_character_cards(search_term=query, limit=effective_fts_top_k)
                # `search_character_cards` already searches name, description, personality, scenario, system_prompt
                # Post-filter by relevant_ids if provided (these are integer IDs for char cards)
                if relevant_ids:
                    str_relevant_ids = [str(rid) for rid in relevant_ids] # Ensure comparison with string IDs
                    temp_results_from_db = [r for r in temp_results_from_db if str(r.get('id')) in str_relevant_ids]


            # Standardize metadata for char_rag_db results
            for item_db in temp_results_from_db:
                item_id = item_db.get('id') # int or str
                item_content = None
                item_title = None

                if database_type == DatabaseType.CHARACTER_CARDS:
                    item_title = item_db.get('name', query[:30])
                    # Construct content string for character cards
                    item_content_parts = [
                        f"Name: {item_db.get('name', '')}",
                        f"Description: {item_db.get('description', '')}",
                        f"Personality: {item_db.get('personality', '')}",
                        f"Scenario: {item_db.get('scenario', '')}",
                        f"System Prompt: {item_db.get('system_prompt', '')}"
                    ]
                    item_content = "\n".join(filter(None, item_content_parts))
                elif database_type == DatabaseType.RAG_NOTES:
                    item_title = item_db.get('title', query[:30])
                    item_content = item_db.get('content') # Notes have 'content'
                elif database_type == DatabaseType.RAG_CHAT or database_type == DatabaseType.CHARACTER_CHAT:
                    item_title = f"Chat snippet from conv: {item_db.get('conversation_id', 'unknown')}"
                    item_content = item_db.get('content') # Messages have 'content'

                if item_content: # Only add if there's content
                    result_item = {
                        "content": item_content,
                        "metadata": {
                            'item_id': str(item_id),
                            'source_db': database_type.value,
                            'title': item_title
                        }
                    }
                    if database_type == DatabaseType.RAG_CHAT or database_type == DatabaseType.CHARACTER_CHAT:
                        result_item['metadata']['conversation_id'] = item_db.get('conversation_id')
                        result_item['metadata']['character_id'] = character_id_context # Add if available
                    results.append(result_item)
        else:
            raise ValueError(f"Unsupported database type for FTS: {database_type.value}")

        search_duration = time.time() - start_time
        log_histogram("perform_full_text_search_duration", search_duration, labels={"database_type": database_type.value})
        log_counter("perform_full_text_search_success",
                    labels={"database_type": database_type.value, "result_count": len(results)})
        return results

    except (MediaDBError, CharactersRAGDBError, ValueError) as e:
        log_counter("perform_full_text_search_error",
                    labels={"database_type": database_type.value, "error_type": type(e).__name__})
        logging.error(f"DB error in FTS ({database_type.value}): {str(e)}", exc_info=True)
        raise
    except Exception as e:
        log_counter("perform_full_text_search_error",
                    labels={"database_type": database_type.value, "error_type": type(e).__name__})
        logging.error(f"Error in FTS ({database_type.value}): {str(e)}", exc_info=True)
        raise


def fetch_relevant_ids_by_keywords(
        media_db: Optional[MediaDatabase],
        char_rag_db: Optional[CharactersRAGDB],
        db_type: DatabaseType,
        keyword_texts: List[str]
) -> List[str]:
    """Fetches relevant item IDs (strings) for a given db_type based on keyword texts."""
    if not keyword_texts:
        return []

    ids_set = set()
    try:
        if db_type == DatabaseType.MEDIA_DB:
            if not media_db: return []
            # fetch_media_keywords_standalone needs a MediaDatabase instance.
            # The function fetch_relevant_media_ids_for_media_db from original RAG_Library_2.py did this correctly.
            media_items_by_keyword: Dict[str, List[Dict[str, Any]]] = media_db.fetch_media_for_keywords(
                keywords=keyword_texts, include_trash=False
            )
            for single_keyword_media_list in media_items_by_keyword.values():
                for media_item in single_keyword_media_list:
                    if 'id' in media_item and media_item['id'] is not None:  # 'id' from Media table
                        ids_set.add(str(media_item['id']))
                    elif 'media_id' in media_item and media_item[
                        'media_id'] is not None:  # alias from fetch_media_for_keywords
                        ids_set.add(str(media_item['media_id']))
            return list(ids_set)

        elif char_rag_db:  # Operations for CharactersRAGDB
            # First, get keyword_ids for all keyword_texts
            keyword_id_map: Dict[str, int] = {}
            for kw_text in keyword_texts:
                kw_obj = char_rag_db.get_keyword_by_text(kw_text)
                if kw_obj:
                    keyword_id_map[kw_text] = kw_obj['id']

            relevant_keyword_ids = [kid for kid in keyword_id_map.values()]
            if not relevant_keyword_ids: return []

            if db_type == DatabaseType.RAG_CHAT or db_type == DatabaseType.CHARACTER_CHAT:  # Conversations
                for kw_id in relevant_keyword_ids:
                    convs = char_rag_db.get_conversations_for_keyword(kw_id,
                                                                      limit=500)  # High limit for comprehensive ID gathering
                    for conv in convs:
                        ids_set.add(str(conv['id']))  # Conversation ID is UUID string
            elif db_type == DatabaseType.RAG_NOTES:  # Notes
                for kw_id in relevant_keyword_ids:
                    notes = char_rag_db.get_notes_for_keyword(kw_id, limit=500)
                    for note in notes:
                        ids_set.add(str(note['id']))  # Note ID is UUID string
            elif db_type == DatabaseType.CHARACTER_CARDS:
                # CHANGED: Strict tag-based filtering for Character Cards.
                # This requires loading cards and checking their 'tags' field in Python
                # because ChaChaNotes_DB FTS does not index the 'tags' JSON field.
                # NOTE: This is INEFFICIENT for large numbers of character cards.
                # Ideal solution: Enhance ChaChaNotes_DB to support querying tags directly
                # (e.g., JSON functions if SQLite version supports, or a normalized tag mapping table).

                if not keyword_texts:  # No keywords to filter by
                    # Optionally, return all card IDs if no keywords are specified,
                    # or an empty list if keywords are mandatory for this path.
                    # For now, returning empty if no keywords, consistent with other paths.
                    return []

                    # Fetch all active character cards (this is the inefficient part)
                all_cards = char_rag_db.list_character_cards(limit=100000)  # Potentially very many

                normalized_keyword_texts = {kw.lower().strip() for kw in keyword_texts}

                for card in all_cards:  # card is a dict
                    if card.get('deleted'):
                        continue

                    tags_str = card.get('tags')  # This is a JSON string or None
                    card_tags_set = set()
                    if tags_str:
                        try:
                            tags_list = json.loads(tags_str)
                            if isinstance(tags_list, list):
                                card_tags_set = {str(tag).lower().strip() for tag in tags_list}
                        except json.JSONDecodeError:
                            logging.warning(
                                f"Could not parse tags JSON for character card ID {card.get('id')}: {tags_str}")

                    # Check if ALL provided keywords are present in the card's tags
                    # To match original intent of "filter by keyword", this should be ANY keyword.
                    # If it should be ALL keywords, change `any` to `all`.
                    # User said "search only cards that have been tagged with a specific keyword" - implies ANY.
                    if normalized_keyword_texts and not card_tags_set.isdisjoint(
                            normalized_keyword_texts):  # isdisjoint is false if there's an intersection
                        ids_set.add(str(card['id']))  # Character Card ID is int
            else:
                logging.warning(f"Unsupported db_type '{db_type}' for keyword ID fetching.")
                return None
        else:
            logging.warning(f"Required DB instance not provided for '{db_type}' keyword ID fetching.")
            return list(ids_set)
    except Exception as e:
        logging.error(f"Error fetching relevant IDs for {db_type} with keywords '{keyword_texts}': {e}", exc_info=True)
        return []


def enhanced_rag_pipeline(
        media_db: MediaDatabase,
        char_rag_db: CharactersRAGDB,
        query: str,
        api_choice: str,
        keywords: Optional[str] = None,
        fts_top_k: int = RAG_SEARCH_CONFIG.get('fts_top_k', 10),
        vector_top_k: int = RAG_SEARCH_CONFIG.get('vector_top_k', 10),
        search_fields_media: Optional[List[str]] = None,
        apply_re_ranking: bool = True,
        database_types: Optional[List[DatabaseType]] = None, # CHANGED: Use Enum
        ranker_instance: Optional[Ranker] = GLOBAL_RANKER # CHANGED: Pass ranker
) -> Dict[str, Any]:
    if database_types is None:
        database_types = [DatabaseType.MEDIA_DB] # Default to Media DB if not specified

    # Log with Enum values
    log_counter("enhanced_rag_pipeline_attempt",
                labels={"api_choice": api_choice, "db_types": ",".join([db_type.value for db_type in database_types])})
    start_time = time.time()

    search_fields_for_media_db = search_fields_media if search_fields_media is not None else ["title", "content"]

    try:
        keyword_list_texts = [k.strip().lower() for k in keywords.split(',')] if keywords and keywords.strip() else []

        relevant_ids_by_type: Dict[DatabaseType, Optional[List[str]]] = {} # CHANGED: Enum as key
        if keyword_list_texts:
            for db_type_enum in database_types:
                ids = fetch_relevant_ids_by_keywords(media_db, char_rag_db, db_type_enum, keyword_list_texts)
                relevant_ids_by_type[db_type_enum] = ids
                logging.debug(f"Relevant IDs for {db_type_enum.value} from keywords: {ids}")

        all_vector_results: List[Dict[str, Any]] = []
        # NOTE: Vector search is primarily for MediaDB here.
        # Needs extension for CharactersRAGDB items (Notes, Cards) if they are embedded.
        if DatabaseType.MEDIA_DB in database_types:
            media_db_ids_for_vector_filter = relevant_ids_by_type.get(DatabaseType.MEDIA_DB)
            all_vector_results.extend(
                perform_general_vector_search(query, media_db_ids_for_vector_filter, top_k=vector_top_k))
        logging.debug(f"Total general vector results: {len(all_vector_results)}")

        # Vector search for RAG Notes
        if DatabaseType.RAG_NOTES in database_types:
            rag_notes_ids_for_vector_filter = relevant_ids_by_type.get(DatabaseType.RAG_NOTES)
            notes_vector_results = perform_rag_notes_vector_search(query, rag_notes_ids_for_vector_filter, k=vector_top_k)
            all_vector_results.extend(notes_vector_results)
            logging.debug(f"RAG Notes vector results: {len(notes_vector_results)}")
        logging.debug(f"Total combined vector results before deduplication: {len(all_vector_results)}")

        all_fts_results: List[Dict[str, Any]] = []

        for db_type_enum in database_types:
            ids_for_fts_filter = relevant_ids_by_type.get(db_type_enum)
            # Only Media DB uses specific search_fields_media for FTS
            current_search_fields = search_fields_for_media_db if db_type_enum == DatabaseType.MEDIA_DB else None

            fts_res = perform_full_text_search(
                media_db=media_db, char_rag_db=char_rag_db, query=query,
                database_type=db_type_enum, # Pass Enum
                relevant_ids=ids_for_fts_filter,
                fts_top_k=fts_top_k, search_fields_media=current_search_fields
            )
            all_fts_results.extend(fts_res) # perform_full_text_search now standardizes output
        logging.debug(f"Total FTS results: {len(all_fts_results)}")

        combined_docs_for_rerank: List[Dict[str, Any]] = []
        doc_counter = 0
        for res_item in all_vector_results:
            text_content = res_item.get('content') # Assumes 'content' key from vector search
            if text_content:
                combined_docs_for_rerank.append({
                    "text": text_content, "metadata": res_item.get('metadata', {}),
                    "rerank_id": f"vec_{doc_counter}", "source": "vector"
                })
                doc_counter += 1
        for res_item in all_fts_results: # FTS results are already Dict[str, Any] with 'content' and 'metadata'
            text_content = res_item.get('content')
            if text_content:
                combined_docs_for_rerank.append({
                    "text": text_content, "metadata": res_item.get('metadata', {}), # Metadata should be set by FTS
                    "rerank_id": f"fts_{doc_counter}", "source": "fts"
                })
                doc_counter += 1

        seen_texts = set()
        unique_combined_docs = []
        for doc in combined_docs_for_rerank:
            if doc['text'] not in seen_texts:
                unique_combined_docs.append(doc)
                seen_texts.add(doc['text'])
        combined_docs_for_rerank = unique_combined_docs
        logging.debug(f"Combined unique documents for reranking: {len(combined_docs_for_rerank)}")

        final_context_docs = combined_docs_for_rerank
        if apply_re_ranking and ranker_instance and combined_docs_for_rerank: # CHANGED: use ranker_instance
            passages_for_rerank = [{"id": item["rerank_id"], "text": item["text"]} for item in combined_docs_for_rerank if item.get("text")]
            if passages_for_rerank:
                rerank_request = RerankRequest(query=query, passages=passages_for_rerank)
                try:
                    reranked_scores = ranker_instance.rerank(rerank_request) # CHANGED
                    score_map = {score_item['id']: score_item['score'] for score_item in reranked_scores}
                    for item in combined_docs_for_rerank:
                        item['rerank_score'] = score_map.get(item['rerank_id'], -float('inf'))
                    final_context_docs = sorted(combined_docs_for_rerank, key=lambda x: x['rerank_score'], reverse=True)
                    logging.debug(
                        f"Reranked {len(final_context_docs)} documents. Top 3 scores: {[r['rerank_score'] for r in final_context_docs[:3] if 'rerank_score' in r]}")
                except Exception as e_rank:
                    logging.error(f"Error during re-ranking: {e_rank}", exc_info=True)
        elif apply_re_ranking and not ranker_instance:
            logging.warning("Re-ranking is enabled but ranker_instance is not available.")


        LLM_CONTEXT_DOC_LIMIT = RAG_SEARCH_CONFIG.getint('llm_context_document_limit', 10)
        context_pieces = [doc['text'] for doc in final_context_docs[:LLM_CONTEXT_DOC_LIMIT] if doc.get('text')]
        context = "\n\n---\n\n".join(context_pieces)
        logging.debug(
            f"Final context length: {len(context)}. Using top {min(len(context_pieces), LLM_CONTEXT_DOC_LIMIT)} documents.")

        if not final_context_docs and not context:
            logging.info(f"No results found after search. Query: {query}, Keywords: {keywords}")
            empty_context_answer = generate_answer(api_choice, "", query) # Pass empty context
            return {
                "answer": "No relevant information based on your query and keywords was found in the database. "
                          "The LLM provided this direct answer: \n\n" + empty_context_answer,
                "context": "No relevant information found. Query was: " + query,
                "source_documents": [] # CHANGED: Add source_documents
            }

        answer = generate_answer(api_choice, context, query)
        pipeline_duration = time.time() - start_time
        log_histogram("enhanced_rag_pipeline_duration", pipeline_duration, labels={"api_choice": api_choice})
        log_counter("enhanced_rag_pipeline_success", labels={"api_choice": api_choice})
        # CHANGED: Return source documents for context
        return {"answer": answer, "context": context, "source_documents": final_context_docs[:LLM_CONTEXT_DOC_LIMIT]}

    except Exception as e:
        log_counter("enhanced_rag_pipeline_error", labels={"api_choice": api_choice, "error_type": type(e).__name__})
        logging.error(f"Critical error in enhanced_rag_pipeline: {str(e)}", exc_info=True)
        try:
            direct_llm_answer = generate_answer(api_choice, "", query)
            return {
                "answer": f"An error occurred processing request with context. LLM direct answer: \n\n{direct_llm_answer}",
                "context": f"Error during RAG pipeline: {str(e)}",
                "source_documents": [] # CHANGED
            }
        except Exception as llm_e:
            logging.error(f"Fallback LLM call also failed: {str(llm_e)}", exc_info=True)
            return {
                "answer": "A critical error occurred, and the LLM could not be reached.",
                "context": f"RAG pipeline error: {str(e)}; Fallback LLM error: {str(llm_e)}",
                "source_documents": [] # CHANGED
            }


def rag_web_scraping_pipeline(
        media_db: MediaDatabase,
        url: str,
        query: str,
        api_choice: Optional[str] = None
) -> Dict[str, Any]:
    log_counter("rag_web_scraping_pipeline_attempt", labels={"url": url})
    start_time = time.time()
    try:
        article_data: Dict[str, str]
        try:
            article_data = scrape_article(url)
            content = article_data.get('content', '')
            title = article_data.get('title', f'Untitled Article from {url}')
            if not content:
                logging.error(f"No content extracted from {url}")
                return {"error": "Failed to extract content from article", "details": "Content was empty.",
                        "answer": "", "context": "", "media_id": None, "media_uuid": None}
        except Exception as e:
            logging.error(f"Error scraping article from {url}: {str(e)}", exc_info=True)
            return {"error": "Failed to scrape article", "details": str(e), "answer": "", "context": "",
                    "media_id": None, "media_uuid": None}

        media_id: Optional[int] = None
        media_uuid: Optional[str] = None
        try:
            # add_media_with_keywords should handle embedding and FTS update internally or trigger it.
            media_id, media_uuid, msg = media_db.add_media_with_keywords(
                url=url, title=title, media_type='article', content=content, overwrite=True
            )
            logging.info(f"Media DB action for {url}: {msg}. ID: {media_id}, UUID: {media_uuid}")
            if not media_id:  # If add_media_with_keywords returns None for ID on failure
                raise MediaDBError(f"Failed to add/update media for {url}: {msg}")
        except MediaDBError as e:
            logging.error(f"Database error adding article {url} to Media_DB: {str(e)}", exc_info=True)
            return {"error": "Failed to store article in Media_DB", "details": str(e), "answer": "", "context": "",
                    "media_id": None, "media_uuid": None}

        # NOTE: process_and_store_content from ChromaDB_Library.py seems to be a more general
        # function that could be used after media is added, if add_media_with_keywords doesn't
        # handle embedding itself. This depends on the workflow of Media_DB_v2.
        # For now, assuming Media_DB_v2's add_media_with_keywords + subsequent ChromaDB processing handles it.

        vector_search_results_content: List[str] = []
        if media_id is not None:
            article_content_embedding = create_embedding(content, embedding_provider, embedding_model,
                                                         embedding_api_url)
            # Define a collection name for articles, e.g., from config
            article_collection_name = config.get('Embeddings', 'article_embeddings_collection',
                                                 fallback="article_content_embeddings")
            store_in_chroma(
                collection_name=article_collection_name,
                texts=[content],
                embeddings=[article_content_embedding],
                ids=[f"media_{media_id}"],  # Or use media_uuid if preferred for Chroma ID
                metadatas=[{
                    "media_id": str(media_id),
                    "media_uuid": media_uuid,
                    "title": title,
                    "url": url,
                    "source_db": DatabaseType.MEDIA_DB.value
                }]
            )
            logging.info(f"Embedded and stored article content for media_id {media_id} in ChromaDB.")
            # Assuming vector search is desired on the newly added article.
            # perform_general_vector_search will look in all general collections.
            # Filtering by this media_id is crucial.
            raw_vector_results = perform_general_vector_search(query, relevant_media_ids=[str(media_id)], top_k=RAG_SEARCH_CONFIG.get('web_vector_top_k', 10))
            if raw_vector_results:
                for res in raw_vector_results:
                    text = res.get('content', res.get('text'))
                    if text: vector_search_results_content.append(text)

        fts_search_results_content: List[str] = []
        if media_id is not None:
            media_results_list, _ = media_db.search_media_db(
                search_query=query, search_fields=['title', 'content'],
                media_ids_filter=[media_id],
                results_per_page=5
            )
            if media_results_list:
                for item in media_results_list:
                    if item.get('content'): fts_search_results_content.append(item['content'])

        all_search_content = vector_search_results_content + fts_search_results_content
        unique_content_pieces = list(dict.fromkeys(all_search_content))  # Simple deduplication
        context_str = "\n\n---\n\n".join(unique_content_pieces[:10])  # Limit context size

        if not context_str and query:
            context_str = content  # Fallback to full article if no search results for query
            logging.info(f"No specific search results for '{query}' in article '{title}'. Using full article content.")
        elif not context_str:  # No query results and no query (or query was empty)
            context_str = f"No specific content found for query '{query}' in article '{title}'."

        answer = generate_answer(api_choice, context_str, query)

        pipeline_duration = time.time() - start_time
        log_histogram("rag_web_scraping_pipeline_duration", pipeline_duration,
                      labels={"api_choice": api_choice or "default"})
        log_counter("rag_web_scraping_pipeline_success", labels={"api_choice": api_choice or "default"})
        return {"answer": answer, "context": context_str, "media_id": media_id, "media_uuid": media_uuid}

    except Exception as e:
        # Ensure duration is logged even on error
        pipeline_duration = time.time() - start_time
        log_histogram("rag_web_scraping_pipeline_duration", pipeline_duration,
                      labels={"api_choice": api_choice or "default"})
        log_counter("rag_web_scraping_pipeline_error",
                    labels={"api_choice": api_choice or "default", "error_type": type(e).__name__})
        logging.error(f"Unexpected error in rag_web_scraping_pipeline for {url}: {str(e)}", exc_info=True)
        return {"error": "An unexpected error occurred in web scraping RAG pipeline", "details": str(e), "answer": "",
                "context": "", "media_id": None, "media_uuid": None}


# --- Section 7: fetch_all_chat_ids_for_character and enhanced_rag_pipeline_chat ---
def fetch_all_chat_ids_for_character(char_rag_db: CharactersRAGDB, character_id: int) -> List[str]:
    log_counter("fetch_all_chat_ids_for_character_attempt", labels={"character_id": character_id})
    start_time = time.time()
    try:
        # get_conversations_for_character returns List[Dict]
        conversations = char_rag_db.get_conversations_for_character(character_id=character_id, limit=10000) # High limit
        # Filter out deleted conversations if the DB method doesn't do it already
        chat_ids = [conv['id'] for conv in conversations if isinstance(conv, dict) and 'id' in conv and not conv.get('deleted')]

        duration = time.time() - start_time
        log_histogram("fetch_all_chat_ids_for_character_duration", duration)
        log_counter("fetch_all_chat_ids_for_character_success",
                    labels={"character_id": character_id, "chat_count": len(chat_ids)})
        return chat_ids
    except CharactersRAGDBError as dbe:
        log_counter("fetch_all_chat_ids_for_character_error",
                    labels={"character_id": character_id, "error_type": "CharactersRAGDBError", "error": str(dbe)})
        logging.error(f"DB error fetching all chat IDs for character {character_id}: {str(dbe)}", exc_info=True)
        return []
    except Exception as e:
        log_counter("fetch_all_chat_ids_for_character_error",
                    labels={"character_id": character_id, "error_type": "General", "error": str(e)})
        logging.error(f"Error fetching all chat IDs for character {character_id}: {str(e)}", exc_info=True)
        return []


def enhanced_rag_pipeline_chat(
        char_rag_db: CharactersRAGDB,
        query: str,
        api_choice: str,
        character_id: int,
        keywords: Optional[str] = None,
        vector_top_k: int = RAG_SEARCH_CONFIG.get('vector_top_k', 10),
        fts_top_k: int = RAG_SEARCH_CONFIG.get('fts_top_k', 10),
        ranker_instance: Optional[Ranker] = GLOBAL_RANKER # CHANGED: Pass ranker
) -> Dict[str, Any]:
    log_counter("enhanced_rag_pipeline_chat_attempt", labels={"api_choice": api_choice, "character_id": character_id})
    start_time = time.time()
    try:
        keyword_texts = [k.strip().lower() for k in keywords.split(',')] if keywords and keywords.strip() else []

        relevant_conversation_ids: List[str] = []
        if keyword_texts:
            keyword_id_map: Dict[str, int] = {}
            for kw_text in keyword_texts:
                kw_obj = char_rag_db.get_keyword_by_text(kw_text)
                if kw_obj and not kw_obj.get('deleted'): keyword_id_map[kw_text] = kw_obj['id']

            relevant_kw_ids = list(keyword_id_map.values())
            if relevant_kw_ids:
                conv_ids_set = set()
                for kw_id in relevant_kw_ids:
                    convs_for_kw = char_rag_db.get_conversations_for_keyword(kw_id, limit=100)
                    for conv in convs_for_kw:
                        # Filter by character_id AND ensure conversation is not deleted
                        if conv.get('character_id') == character_id and not conv.get('deleted'):
                            conv_ids_set.add(conv['id'])
                relevant_conversation_ids = list(conv_ids_set)
        else:
            relevant_conversation_ids = fetch_all_chat_ids_for_character(char_rag_db, character_id)

        logging.debug(
            f"Chat RAG: Relevant conversation IDs for char {character_id} (keywords: {keywords}): {relevant_conversation_ids}")

        if not relevant_conversation_ids:
            logging.info(
                f"No relevant chats found for character_id {character_id} with keywords: {keyword_texts}. Generating answer without specific chat context.")
            answer = generate_answer(api_choice, "", query)
            duration_no_ctx = time.time() - start_time
            log_histogram("enhanced_rag_pipeline_chat_duration", duration_no_ctx, labels={"api_choice": api_choice, "context_found": "false"})
            log_counter("enhanced_rag_pipeline_chat_success_no_context", labels={"api_choice": api_choice, "character_id": character_id})
            return {"answer": answer, "context": "No specific chat history found for context.", "source_documents": []}

        # Vector search within relevant chat messages
        vector_results_chat = perform_vector_search_chat_messages(query, relevant_conversation_ids, k=vector_top_k)
        logging.debug(f"Chat RAG: Vector search (chat messages) results: {len(vector_results_chat)} items")

        fts_results_chat_messages: List[Dict[str, Any]] = []
        for conv_id_str in relevant_conversation_ids:
            messages = char_rag_db.search_messages_by_content(
                content_query=query, conversation_id=conv_id_str, limit=fts_top_k
            )
            for msg in messages: # Messages are dicts
                if not msg.get('deleted'): # Ensure message itself is not deleted
                    # Standardize output format for FTS results if needed, similar to perform_full_text_search
                    # Here, assuming search_messages_by_content returns good structure.
                    # If metadata is missing, populate it.
                    if 'metadata' not in msg: msg['metadata'] = {}
                    msg['metadata']['source_db'] = DatabaseType.CHARACTER_CHAT.value
                    msg['metadata']['conversation_id'] = conv_id_str
                    msg['metadata']['item_id'] = msg.get('id') # Message UUID
                    msg['metadata']['title'] = f"Message from conv {conv_id_str}"
                    # Ensure 'content' key exists for reranker
                    if 'content' not in msg and 'text' in msg : msg['content'] = msg['text']
                    elif 'content' not in msg: msg['content'] = "" # Fallback

                    fts_results_chat_messages.append(msg)
        logging.debug(f"Chat RAG: FTS (message content) results: {len(fts_results_chat_messages)} items")


        all_chat_results_for_rerank: List[Dict[str, Any]] = []
        chat_doc_counter = 0
        for res_item in vector_results_chat:
            text_content = res_item.get('content')
            if text_content:
                all_chat_results_for_rerank.append({
                    "text": text_content, "metadata": res_item.get('metadata', {}),
                    "rerank_id": f"vec_chat_{chat_doc_counter}", "source": "vector_chat_message"
                })
                chat_doc_counter += 1
        for res_item in fts_results_chat_messages: # res_item is already a dict
            text_content = res_item.get('content')
            if text_content:
                all_chat_results_for_rerank.append({
                    "text": text_content, "metadata": res_item.get('metadata', {}),
                    "rerank_id": f"fts_chat_{chat_doc_counter}", "source": "fts_chat_message"
                })
                chat_doc_counter += 1

        seen_chat_texts = set()
        unique_chat_docs = []
        for doc in all_chat_results_for_rerank:
            if doc.get('text') and doc['text'] not in seen_chat_texts: # Check if text exists
                unique_chat_docs.append(doc)
                seen_chat_texts.add(doc['text'])
        all_chat_results_for_rerank = unique_chat_docs

        final_chat_context_docs = all_chat_results_for_rerank
        apply_re_ranking_chat = config.getboolean('RAG', 'apply_reranking_chat', fallback=True)

        if apply_re_ranking_chat and ranker_instance and all_chat_results_for_rerank: # CHANGED: use ranker_instance
            logging.debug("Chat RAG: Applying Re-Ranking to chat results.")
            passages_chat = [{"id": item["rerank_id"], "text": item["text"]} for item in all_chat_results_for_rerank if item.get("text")]
            if passages_chat:
                rerank_request_chat = RerankRequest(query=query, passages=passages_chat)
                try:
                    reranked_chat_scores = ranker_instance.rerank(rerank_request_chat) # CHANGED
                    score_map_chat = {score_item['id']: score_item['score'] for score_item in reranked_chat_scores}
                    for item in all_chat_results_for_rerank:
                        item['rerank_score'] = score_map_chat.get(item['rerank_id'], -float('inf'))
                    final_chat_context_docs = sorted(all_chat_results_for_rerank, key=lambda x: x['rerank_score'],
                                                     reverse=True)
                except Exception as e_rank_chat:
                    logging.error(f"Error during chat results re-ranking: {e_rank_chat}", exc_info=True)
        elif apply_re_ranking_chat and not ranker_instance:
            logging.warning("Chat RAG Re-ranking is enabled but ranker_instance is not available.")


        chat_context_limit = RAG_SEARCH_CONFIG.get('chat_context_limit', 10)
        chat_context_pieces = [doc['text'] for doc in final_chat_context_docs[:chat_context_limit] if doc.get('text')]
        context_chat = "\n\n---\n\n".join(chat_context_pieces)
        logging.debug(
            f"Chat RAG Context length: {len(context_chat)}. Using top {min(len(chat_context_pieces), chat_context_limit)} documents.")

        if not final_chat_context_docs and not context_chat :
            # This case should ideally be caught by 'if not relevant_conversation_ids:' earlier.
            # If reached, means relevant_conversation_ids existed, but yielded no usable context.
            logging.info(f"Chat RAG: No context built despite having relevant conversation IDs. Query: {query}")
            answer = generate_answer(api_choice, "", query) # Fallback to direct LLM
            # ... (log duration and specific success/failure metric)
            return {
                "answer": "No specific chat history snippets were found for your query. " + answer,
                "context": "No relevant chat snippets found. Query: " + query,
                "source_documents": []
            }

        answer = generate_answer(api_choice, context_chat, query)
        pipeline_duration = time.time() - start_time
        log_histogram("enhanced_rag_pipeline_chat_duration", pipeline_duration, labels={"api_choice": api_choice, "context_found": "true"})
        log_counter("enhanced_rag_pipeline_chat_success",
                    labels={"api_choice": api_choice, "character_id": character_id})
        return {"answer": answer, "context": context_chat, "source_documents": final_chat_context_docs[:chat_context_limit]}

    except Exception as e:
        log_counter("enhanced_rag_pipeline_chat_error",
                    labels={"api_choice": api_choice, "character_id": character_id, "error_type": type(e).__name__})
        logging.error(f"Error in enhanced_rag_pipeline_chat: {str(e)}", exc_info=True)
        try:
            direct_llm_answer_chat = generate_answer(api_choice, "", query)
            return {
                "answer": f"An error occurred retrieving chat context. LLM direct answer:\n\n{direct_llm_answer_chat}",
                "context": f"Error during Chat RAG pipeline: {str(e)}",
                "source_documents": []
            }
        except Exception as llm_e_chat:
            logging.error(f"Fallback LLM call also failed for chat RAG: {str(llm_e_chat)}", exc_info=True)
            return {
                "answer": "A critical error occurred processing your chat request, and the LLM could not be reached.",
                "context": f"Chat RAG pipeline error: {str(e)}; Fallback LLM error: {str(llm_e_chat)}",
                "source_documents": []
            }


# --- Section 8: rag_qa_chat, DB Utilities, and Placeholder ---
########################################################################################################################
# QA Chat Main Function (from RAG_QA_Chat.py)
########################################################################################################################
def rag_qa_chat(
        query: str,
        history: List[Tuple[str, str]],
        context_source_identifier: Optional[str], # This seems less used now with explicit target_db_types
        api_choice: str,
        media_db: MediaDatabase,
        char_rag_db: CharactersRAGDB,
        keywords: Optional[str] = None,
        apply_re_ranking: bool = False, # Default in RAG_QA_Chat was False
        target_db_types: Optional[List[DatabaseType]] = None, # CHANGED: Use Enum
        ranker_instance: Optional[Ranker] = GLOBAL_RANKER # CHANGED: Pass ranker
):
    if target_db_types is None:
        target_db_types = [DatabaseType.MEDIA_DB]

    log_counter("rag_qa_chat_attempt",
                labels={"api_choice": api_choice, "context_source": str(context_source_identifier)})
    start_time = time.time()

    try:
        answer = ""
        context_generated = ""
        source_docs_for_answer = [] # Store source documents used

        if target_db_types: # If specific DBs are targeted for RAG
            log_counter("rag_qa_chat_use_enhanced_pipeline", labels={"db_types": ",".join([db.value for db in target_db_types])})

            result = enhanced_rag_pipeline(
                media_db=media_db,
                char_rag_db=char_rag_db,
                query=query,
                api_choice=api_choice,
                keywords=keywords,
                apply_re_ranking=apply_re_ranking,
                database_types=target_db_types, # Pass Enum list
                ranker_instance=ranker_instance # Pass ranker
            )
            answer = result['answer']
            context_generated = result['context'] # Context used by LLM
            source_docs_for_answer = result.get('source_documents', []) # Get source documents
        else: # No DBs specified, implies direct LLM call or pre-supplied context (not handled here)
            log_counter("rag_qa_chat_direct_llm_call")
            # Assuming context_source_identifier MIGHT be a pre-fetched string of context.
            # If it's an ID, the logic above should handle fetching.
            # If it's a raw context string and target_db_types is empty, use it:
            pre_supplied_context = str(context_source_identifier) if isinstance(context_source_identifier, str) and not target_db_types else ""
            answer = generate_answer(api_choice, pre_supplied_context, query)
            context_generated = pre_supplied_context

        new_history = history + [(query, answer)]
        duration = time.time() - start_time
        log_histogram("rag_qa_chat_duration", duration, labels={"api_choice": api_choice})
        log_counter("rag_qa_chat_success", labels={"api_choice": api_choice})
        # Return history, answer, and the context that was actually used (and optionally sources)
        return new_history, answer, context_generated, source_docs_for_answer

    except Exception as e:
        log_counter("rag_qa_chat_error", labels={"api_choice": api_choice, "error": str(e)})
        logging.error(f"Error in rag_qa_chat: {str(e)}", exc_info=True)
        error_message = "An error occurred while processing your request."
        # Return empty context and sources on error
        return history + [(query, error_message)], error_message, "", []


########################################################################################################################
# Database Utility Functions (from RAG_QA_Chat.py, updated for new DBs)
########################################################################################################################
# These seem to use the DB instances correctly.

def search_media_database_utility(
        media_db: MediaDatabase, # Instance of Media_DB_v2.Database
        query: str,
        search_fields: Optional[List[str]] = None,
        page: int = 1,
        results_per_page: int = 10
) -> List[Dict[str, Any]]:
    if search_fields is None:
        search_fields = ["title", "content"] # Default fields
    try:
        log_counter("search_media_database_utility_attempt")
        start_time = time.time()
        # search_media_db is a method of the MediaDatabase instance
        results, total_matches = media_db.search_media_db(
            search_query=query,
            search_fields=search_fields,
            page=page,
            results_per_page=results_per_page
            # media_ids_filter, keywords, include_trash, include_deleted could be added if needed
        )
        search_duration = time.time() - start_time
        log_histogram("search_media_database_utility_duration", search_duration)
        log_counter("search_media_database_utility_success",
                    labels={"result_count": len(results), "total_matches": total_matches})
        return results # list of media item dicts
    except DatabaseError as de: # Catch specific DB errors from media_db
        log_counter("search_media_database_utility_error", labels={"error_type": "DatabaseError", "error": str(de)})
        logging.error(f"Database error searching media database: {str(de)}", exc_info=True)
        raise
    except Exception as e:
        log_counter("search_media_database_utility_error", labels={"error_type": "General", "error": str(e)})
        logging.error(f"Error searching media database: {str(e)}", exc_info=True)
        raise


def get_existing_media_files_paginated(
        media_db: MediaDatabase, # Instance of Media_DB_v2.Database
        page: int = 1,
        results_per_page: int = 50
) -> Tuple[List[Dict[str, Any]], int, int]:
    log_counter("get_existing_media_files_paginated_attempt")
    start_time = time.time()
    try:
        # get_paginated_media_list is a method of the MediaDatabase instance
        media_items, total_pages, _, total_items = media_db.get_paginated_media_list(
            page=page,
            results_per_page=results_per_page
        )
        fetch_duration = time.time() - start_time
        log_histogram("get_existing_media_files_paginated_duration", fetch_duration)
        log_counter("get_existing_media_files_paginated_success",
                    labels={"file_count_page": len(media_items), "total_files": total_items})
        return media_items, total_pages, total_items
    except DatabaseError as de:
        log_counter("get_existing_media_files_paginated_error", labels={"error_type": "DatabaseError", "error": str(de)})
        logging.error(f"Database error fetching existing media files: {str(de)}", exc_info=True)
        raise
    except Exception as e:
        log_counter("get_existing_media_files_paginated_error", labels={"error_type": "General", "error": str(e)})
        logging.error(f"Error fetching existing media files: {str(e)}", exc_info=True)
        raise


# --- RAG Notes Embedding and Vector Search ---

def embed_and_store_rag_notes(
        char_rag_db: CharactersRAGDB,
        note_ids: List[str],  # List of Note UUIDs to process
        collection_name: Optional[str] = None
):
    """
    Fetches RAG notes by their IDs, embeds their content, and stores them in ChromaDB.
    """
    if not note_ids:
        logging.info("No note IDs provided for embedding.")
        return

    _collection_name = collection_name or config.get('Embeddings', 'rag_notes_embeddings_collection',
                                                     fallback="rag_notes_embeddings")

    log_counter("embed_and_store_rag_notes_attempt",
                labels={"note_count_input": len(note_ids), "collection": _collection_name})
    start_time = time.time()

    notes_to_embed_texts = []
    embeddings_list = []
    doc_ids_for_chroma = []
    metadatas_list_for_chroma = []
    processed_note_count = 0

    for note_uuid in note_ids:
        try:
            note_data = char_rag_db.get_note_by_id(note_uuid)  # Fetch full note data
            if not note_data or note_data.get('deleted'):
                logging.warning(f"Note UUID {note_uuid} not found or is deleted. Skipping embedding.")
                continue

            content = note_data.get('content')
            title = note_data.get('title', "Untitled Note")

            if not content:
                logging.warning(f"Note UUID {note_uuid} has no content. Skipping embedding.")
                continue

            # We could combine title and content for embedding if desired:
            # text_to_embed = f"Title: {title}\n\nContent:\n{content}"
            text_to_embed = content  # Or just content

            notes_to_embed_texts.append(text_to_embed)
            doc_ids_for_chroma.append(f"note_{note_uuid}")  # Unique ID for ChromaDB

            # Prepare metadata for ChromaDB
            chroma_metadata = {
                "note_id": note_uuid,  # Store the note's UUID
                "title": title,
                "source_db": DatabaseType.RAG_NOTES.value,
                "created_at": note_data.get('created_at'),  # Assuming ISO string
                "last_modified": note_data.get('last_modified')  # Assuming ISO string
                # Add any other relevant metadata from note_data
            }
            metadatas_list_for_chroma.append(chroma_metadata)
            processed_note_count += 1

        except Exception as e_fetch:
            logging.error(f"Error fetching or preparing note UUID {note_uuid} for embedding: {e_fetch}", exc_info=True)
            log_counter("embed_and_store_rag_notes_item_error", labels={"note_id": note_uuid, "error": str(e_fetch)})

    if notes_to_embed_texts:
        try:
            # Batch create embeddings if available, loop otherwise
            # Assuming create_embedding processes one by one for now (as in chat messages)
            logging.info(f"Creating batch embeddings for {len(notes_to_embed_texts)} RAG notes...")
            embeddings_list = create_embeddings_batch(notes_to_embed_texts,
                                                      provider_override=embedding_provider,
                                                      model_override=embedding_model,
                                                      api_url_override=embedding_api_url)
            logging.info(f"Finished batch embeddings for RAG notes.")

            logging.info(
                f"Storing {len(notes_to_embed_texts)} note embeddings in ChromaDB collection '{_collection_name}'...")
            store_in_chroma(
                collection_name=_collection_name,
                texts=notes_to_embed_texts,
                embeddings=embeddings_list,
                ids=doc_ids_for_chroma,
                metadatas=metadatas_list_for_chroma
            )
            logging.info(f"Successfully stored {len(notes_to_embed_texts)} note embeddings.")

        except Exception as e_embed_store:
            log_counter("embed_and_store_rag_notes_batch_error", labels={"error": str(e_embed_store)})
            logging.error(f"Error during batch embedding or storing of RAG notes: {e_embed_store}", exc_info=True)
            # Decide if to raise or just log. For now, log.

    duration = time.time() - start_time
    log_histogram("embed_and_store_rag_notes_duration", duration, labels={"collection": _collection_name})
    log_counter("embed_and_store_rag_notes_success",
                labels={"processed_count": processed_note_count, "stored_count": len(embeddings_list),
                        "collection": _collection_name})


def perform_rag_notes_vector_search(
    query: str,
    relevant_note_ids: Optional[List[str]] = None, # UUIDs of notes
    k: int = 10,
    collection_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs vector search for RAG Notes in their dedicated ChromaDB collection.
    """
    _collection_name = collection_name or config.get('Embeddings', 'rag_notes_embeddings_collection', fallback="rag_notes_embeddings")
    log_counter("perform_rag_notes_vector_search_attempt", labels={"collection": _collection_name})
    start_time = time.time()

    search_results = []
    try:
        try:
            collection = chroma_client.get_collection(name=_collection_name)
        except ValueError: # Or more specific chromadb error
            logging.warning(f"ChromaDB collection '{_collection_name}' for RAG notes not found.")
            return []
        except chromadb.errors.ChromaError as ce:
            logging.error(f"ChromaDB error getting collection '{_collection_name}': {ce}", exc_info=True)
            return []


        query_embedding = create_embedding(query, embedding_provider, embedding_model, embedding_api_url)
        if not query_embedding:
            logging.error("Failed to generate query embedding for RAG notes vector search.")
            return []

        where_filter = None
        if relevant_note_ids:
            # Assuming 'note_id' (UUID string) is stored in metadata for each embedded note.
            where_filter = {"note_id": {"$in": relevant_note_ids}}

        results_obj = collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        if results_obj['documents'] and results_obj['documents'][0]:
            for doc, meta, dist in zip(results_obj['documents'][0], results_obj['metadatas'][0], results_obj['distances'][0]):
                search_results.append({
                    "content": doc, # This is the text that was embedded
                    "metadata": meta, # Includes note_id, title, etc.
                    "distance": dist
                })
        search_results.sort(key=lambda x: x.get('distance', float('inf')))

    except chromadb.errors.ChromaError as ce:
        log_counter("perform_rag_notes_vector_search_error", labels={"error_type": "ChromaError", "error": str(ce)})
        logging.error(f"ChromaDB error in perform_rag_notes_vector_search: {ce}", exc_info=True)
        return []
    except Exception as e:
        log_counter("perform_rag_notes_vector_search_error", labels={"error_type": "General", "error": str(e)})
        logging.error(f"Error in perform_rag_notes_vector_search: {e}", exc_info=True)
        return [] # Or raise

    duration = time.time() - start_time
    log_histogram("perform_rag_notes_vector_search_duration", duration)
    log_counter("perform_rag_notes_vector_search_success", labels={"result_count": len(search_results)})
    return search_results



########################################################################################################################
# Placeholder for Preprocessing (from RAG_Library_2.py)
# This needs significant thought regarding how different content types are embedded
# and stored in ChromaDB, especially for items from CharactersRAGDB.
########################################################################################################################
# def preprocess_all_content_unified(
#         media_db: Optional[MediaDatabase] = None,
#         char_rag_db: Optional[CharactersRAGDB] = None,
#         # ... other parameters for specific preprocessing tasks ...
# ):
#     logging.info("Starting unified preprocessing...")
#
#     if media_db:
#         unprocessed_media = media_db.get_unprocessed_media() # Assuming this method exists
#         logging.info(f"Found {len(unprocessed_media)} unprocessed media items in Media_DB.")
#         for item in unprocessed_media:
#             media_id = item['id']
#             content = item['content']
#             media_type = item['type']
#             file_name = item.get('title', f"{media_type}_{media_id}")
#             # NOTE: ChromaDB collection strategy needs careful design.
#             # One collection per item is generally not scalable.
#             # Consider collections per content type (e.g., "articles", "videos_transcripts")
#             # or a single large collection with good metadata filtering.
#             collection_name_media = f"media_content_type_{media_type}" # Example
#             try:
#                 # This is where you'd call a function like ChromaDB_Library.process_and_store_content
#                 # but adapted for this unified context.
#                 # process_and_store_generic_content(
#                 #    content_to_embed=content,
#                 #    collection_name=collection_name_media,
#                 #    metadata={"source_db": "MediaDB", "media_id": str(media_id), "type": media_type, ...},
#                 #    item_id_for_chroma=f"media_{media_id}" # Unique ID for Chroma doc
#                 # )
#                 logging.info(f"Placeholder: Would process and embed media_id {media_id}")
#                 media_db.mark_media_as_processed(media_id)
#             except Exception as e_media_proc:
#                 logging.error(f"Error processing MediaDB item {media_id}: {e_media_proc}", exc_info=True)
#
#     if char_rag_db:
#         # --- RAG Notes ---
#         # You'd need a way to get "unprocessed" notes if they are to be embedded.
#         # For example, all notes without a corresponding ChromaDB entry.
#         all_notes = char_rag_db.list_notes(limit=100000) # Potentially very large
#         logging.info(f"Found {len(all_notes)} notes in CharactersRAGDB for potential embedding.")
#         notes_collection_name = "character_rag_notes" # Example collection
#         for note in all_notes:
#             if not note.get('deleted'):
#                 note_id_uuid = note['id']
#                 note_content = note['content']
#                 # Check if already embedded, e.g., by querying ChromaDB for "note_" + note_id_uuid
#                 # if not already_embedded(notes_collection_name, f"note_{note_id_uuid}"):
#                 # process_and_store_generic_content(
#                 #    content_to_embed=note_content,
#                 #    collection_name=notes_collection_name,
#                 #    metadata={"source_db": "CharactersRAGDB_Notes", "note_uuid": note_id_uuid, "title": note.get('title'), ...},
#                 #    item_id_for_chroma=f"note_{note_id_uuid}"
#                 # )
#                 logging.info(f"Placeholder: Would process and embed RAG Note UUID {note_id_uuid}")
#
#         # --- Character Cards ---
#         # Similar logic for character cards if their fields (description, personality, etc.) are to be embedded.
#         all_cards = char_rag_db.list_character_cards(limit=10000)
#         logging.info(f"Found {len(all_cards)} character cards for potential embedding.")
#         cards_collection_name = "character_rag_cards"
#         for card in all_cards:
#            if not card.get('deleted'):
#                 card_id_int = card['id']
#                 # Construct a combined text from card fields for embedding
#                 card_text_to_embed = f"Name: {card.get('name')}\nDescription: {card.get('description')}\nPersonality: {card.get('personality')}"
#                 # if not already_embedded(cards_collection_name, f"charcard_{card_id_int}"):
#                 # process_and_store_generic_content(
#                 #    content_to_embed=card_text_to_embed,
#                 #    collection_name=cards_collection_name,
#                 #    metadata={"source_db": "CharactersRAGDB_Cards", "card_id": str(card_id_int), "name": card.get('name'), ...},
#                 #    item_id_for_chroma=f"charcard_{card_id_int}"
#                 # )
#                 logging.info(f"Placeholder: Would process and embed Character Card ID {card_id_int}")
#
#     logging.info("Unified preprocessing finished (placeholders).")

#
# End of unified_rag_service.py
########################################################################################################################
