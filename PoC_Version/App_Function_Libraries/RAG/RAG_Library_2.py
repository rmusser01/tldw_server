# RAG_Library_2.py
# Description: This script contains the main RAG pipeline function and related functions for the RAG pipeline.
#
# Import necessary modules and functions
import configparser
import os
import time
from typing import Dict, Any, List, Optional
#
# Local Imports
from PoC_Version.App_Function_Libraries.RAG.ChromaDB_Library import vector_search, chroma_client
from PoC_Version.App_Function_Libraries.RAG.RAG_Persona_Chat import perform_vector_search_chat
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import summarize
from PoC_Version.App_Function_Libraries.DB.DB_Manager import fetch_keywords_for_media, search_media_db, get_notes_by_keywords, \
    search_conversations_by_keywords
from PoC_Version.App_Function_Libraries.Utils.Utils import load_and_log_configs, logging
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.Chat.Chat_Functions import process_user_input, ChatDictionary, parse_user_dict_markdown_file
from PoC_Version.App_Function_Libraries.DB.Character_Chat_DB import get_character_chats, perform_full_text_search_chat, \
    fetch_keywords_for_chats, search_character_chat, search_character_cards, fetch_character_ids_by_keywords
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import search_rag_chat, search_rag_notes
#
# 3rd-Party Imports
import openai
from flashrank import Ranker, RerankRequest
#
########################################################################################################################
#
# Functions:
loaded_config_data = load_and_log_configs()

# Initialize OpenAI client (adjust this based on your API key management)
openai.api_key = "your-openai-api-key"

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the config file
config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
# Read the config file
config = configparser.ConfigParser()
# Read the configuration file
config.read('config.txt')


search_functions = {
    "Media DB": search_media_db,
    "RAG Chat": search_rag_chat,
    "RAG Notes": search_rag_notes,
    "Character Chat": search_character_chat,
    "Character Cards": search_character_cards
}

# --- Adapter functions for search_functions ---
def _adapt_search_media_db(query_str: str, top_k_val: int, relevant_ids_list_str: Optional[List[str]]):
    ids_int = [int(id_val) for id_val in relevant_ids_list_str] if relevant_ids_list_str else None
    # The call below depends on the actual signature of search_media_db_manager
    # or a more direct SQLite_DB.search_media_db.
    # Assuming search_media_db_manager or its underlying call expects these:
    # This is a placeholder, you'll need to adjust based on how search_media_db_manager
    # actually passes relevant_ids to the SQLite layer and what it expects for 'keywords'
    # and 'search_fields'.
    from PoC_Version.App_Function_Libraries.DB.SQLite_DB import search_media_db as sqlite_search_media_db # Direct import for clarity
    return sqlite_search_media_db(
        search_query=query_str,
        search_fields=['title', 'content', 'summary', 'transcription'], # Example default fields
        keywords=None, # This is for FTS keywords, not filtering by ID
        page=1,
        results_per_page=top_k_val
    )

def _adapt_search_rag_chat(query_str: str, top_k_val: int, relevant_ids_list_str: Optional[List[str]]):
    # Assuming search_rag_chat expects relevant_ids as List[str] or List[int]
    # Convert if necessary, or ensure search_rag_chat handles List[str]
    ids_for_db = [int(id_val) for id_val in relevant_ids_list_str] if relevant_ids_list_str else None
    return search_rag_chat(query=query_str) # Pass converted IDs

def _adapt_search_rag_notes(query_str: str, top_k_val: int, relevant_ids_list_str: Optional[List[str]]):
    ids_for_db = [int(id_val) for id_val in relevant_ids_list_str] if relevant_ids_list_str else None
    return search_rag_notes(query=query_str)

def _adapt_search_character_chat(query_str: str, top_k_val: int, relevant_ids_list_str: Optional[List[str]]):
    # search_character_chat needs character_id. The generic perform_full_text_search doesn't have it.
    # This adapter might need to search across all characters or handle character_id=None.
    logging.warning("Character Chat search from generic FTS might not be specific to a character.")
    ids_for_db = [int(id_val) for id_val in relevant_ids_list_str] if relevant_ids_list_str else None
    return search_character_chat(query=query_str) # Assuming character_id=None means search all

def _adapt_search_character_cards(query_str: str, top_k_val: int, relevant_ids_list_str: Optional[List[str]]):
    ids_for_db = [int(id_val) for id_val in relevant_ids_list_str] if relevant_ids_list_str else None
    return search_character_cards(query=query_str)

# Updated search_functions dictionary
search_functions = {
    "Media DB": _adapt_search_media_db,
    "RAG Chat": _adapt_search_rag_chat,
    "RAG Notes": _adapt_search_rag_notes,
    "Character Chat": _adapt_search_character_chat,
    "Character Cards": _adapt_search_character_cards
}

# RAG pipeline function for web scraping
# def rag_web_scraping_pipeline(url: str, query: str, api_choice=None) -> Dict[str, Any]:
#     try:
#         # Extract content
#         try:
#             article_data = scrape_article(url)
#             content = article_data['content']
#             title = article_data['title']
#         except Exception as e:
#             logging.error(f"Error scraping article: {str(e)}")
#             return {"error": "Failed to scrape article", "details": str(e)}
#
#         # Store the article in the database and get the media_id
#         try:
#             media_id = add_media_to_database(url, title, 'article', content)
#         except Exception as e:
#             logging.error(f"Error adding article to database: {str(e)}")
#             return {"error": "Failed to store article in database", "details": str(e)}
#
#         # Process and store content
#         collection_name = f"article_{media_id}"
#         try:
#             # Assuming you have a database object available, let's call it 'db'
#             db = get_database_connection()
#
#             process_and_store_content(
#                 database=db,
#                 content=content,
#                 collection_name=collection_name,
#                 media_id=media_id,
#                 file_name=title,
#                 create_embeddings=True,
#                 create_contextualized=True,
#                 api_name=api_choice
#             )
#         except Exception as e:
#             logging.error(f"Error processing and storing content: {str(e)}")
#             return {"error": "Failed to process and store content", "details": str(e)}
#
#         # Perform searches
#         try:
#             vector_results = vector_search(collection_name, query, k=5)
#             fts_results = search_db(query, ["content"], "", page=1, results_per_page=5)
#         except Exception as e:
#             logging.error(f"Error performing searches: {str(e)}")
#             return {"error": "Failed to perform searches", "details": str(e)}
#
#         # Combine results with error handling for missing 'content' key
#         all_results = []
#         for result in vector_results + fts_results:
#             if isinstance(result, dict) and 'content' in result:
#                 all_results.append(result['content'])
#             else:
#                 logging.warning(f"Unexpected result format: {result}")
#                 all_results.append(str(result))
#
#         context = "\n".join(all_results)
#
#         # Generate answer using the selected API
#         try:
#             answer = generate_answer(api_choice, context, query)
#         except Exception as e:
#             logging.error(f"Error generating answer: {str(e)}")
#             return {"error": "Failed to generate answer", "details": str(e)}
#
#         return {
#             "answer": answer,
#             "context": context
#         }
#
#     except Exception as e:
#         logging.error(f"Unexpected error in rag_pipeline: {str(e)}")
#         return {"error": "An unexpected error occurred", "details": str(e)}


# RAG Search with keyword filtering
# FIXME - Update each called function to support modifiable top-k results
def enhanced_rag_pipeline(
    query: str,
    api_choice: str,
    keywords: Optional[str] = None,
    fts_top_k: int = 10,
    apply_re_ranking: bool = True,
    database_types: List[str] = ["Media DB"]
) -> Dict[str, Any]:
    """
    Perform full text search across specified database type.

    Args:
        query: Search query string
        api_choice: API to use for generating the response
        keywords: Optional list of media IDs to filter results
        fts_top_k: Maximum number of results to return
        apply_re_ranking: Whether to apply re-ranking to results
        database_types: Type of database to search

    Returns:
        Dictionary containing search results with content
    """
    log_counter("enhanced_rag_pipeline_attempt", labels={"api_choice": api_choice})
    start_time = time.time()

    try:
        # Load embedding provider from config, or fallback to 'openai'
        embedding_provider = config.get('Embeddings', 'provider', fallback='openai')
        logging.debug(f"Using embedding provider: {embedding_provider}")

        # Initialize relevant IDs dictionary
        relevant_ids: Dict[str, Optional[List[str]]] = {}

        # Process keywords if provided
        if keywords:
            keyword_list = [k.strip().lower() for k in keywords.split(',')]
            logging.debug(f"enhanced_rag_pipeline - Keywords: {keyword_list}")

            try:
                for db_type in database_types:
                    if db_type == "Media DB":
                        media_ids = fetch_relevant_media_ids(keyword_list)
                        relevant_ids[db_type] = [str(id_) for id_ in media_ids]
                    elif db_type == "RAG Chat":
                        conversations, _, _ = search_conversations_by_keywords(keywords=keyword_list)
                        relevant_ids[db_type] = [str(conv['conversation_id']) for conv in conversations]
                    elif db_type == "RAG Notes":
                        notes, _, _ = get_notes_by_keywords(keyword_list)
                        relevant_ids[db_type] = [str(note_id) for note_id, _, _, _ in notes]
                    elif db_type == "Character Chat":
                        relevant_ids[db_type] = [str(id_) for id_ in fetch_keywords_for_chats(keyword_list)]
                    elif db_type == "Character Cards":
                        relevant_ids[db_type] = [str(id_) for id_ in fetch_character_ids_by_keywords(keyword_list)]
                    else:
                        logging.error(f"Unsupported database type: {db_type}")

                    logging.debug(f"enhanced_rag_pipeline - {db_type} relevant IDs: {relevant_ids[db_type]}")
            except Exception as e:
                logging.error(f"Error fetching relevant IDs: {str(e)}")
                relevant_ids = {db_type: None for db_type in database_types}
        else:
            relevant_ids = {db_type: None for db_type in database_types}

        # Perform vector search
        vector_results = []
        for db_type in database_types:
            try:
                db_relevant_ids = relevant_ids.get(db_type)
                results = perform_vector_search(query, db_relevant_ids, top_k=fts_top_k)
                vector_results.extend(results)
                logging.debug(f"\nenhanced_rag_pipeline - Vector search results for {db_type}: {results}")
            except Exception as e:
                logging.error(f"Error performing vector search on {db_type}: {str(e)}")

        # Perform vector search
        # FIXME
        #vector_results = perform_vector_search(query, relevant_media_ids)
        #ogging.debug(f"\n\nenhanced_rag_pipeline - Vector search results: {vector_results}")

        # Perform full-text search
        #v1
        #fts_results = perform_full_text_search(query, database_type, relevant_media_ids, fts_top_k)

        # v2
        # Perform full-text search across specified databases
        fts_results = []
        for db_type in database_types:
            try:
                db_relevant_ids = relevant_ids.get(db_type)
                db_results = perform_full_text_search(query, db_type, db_relevant_ids, fts_top_k)
                fts_results.extend(db_results)
                logging.debug(f"enhanced_rag_pipeline - FTS results for {db_type}: {db_results}")
            except Exception as e:
                logging.error(f"Error performing full-text search on {db_type}: {str(e)}")

        #logging.debug("\n\nenhanced_rag_pipeline - Full-text search results:")
        logging.debug(
            "\n\nenhanced_rag_pipeline - Full-text search results:\n" + "\n".join(
                [str(item) for item in fts_results]) + "\n"
        )

        # Combine results
        all_results = vector_results + fts_results

        # FIXME - specify model + add param to modify at call time
        # You can specify a model if necessary, e.g., model_name="ms-marco-MiniLM-L-12-v2"
        # Apply re-ranking if enabled and results exist
        if apply_re_ranking and all_results:
            logging.debug(f"\nenhanced_rag_pipeline - Applying Re-Ranking")

            if all_results:
                ranker = Ranker()

                # Prepare passages for re-ranking
                passages = [{"id": i, "text": result['content']} for i, result in enumerate(all_results)]
                rerank_request = RerankRequest(query=query, passages=passages)

                # Rerank the results
                reranked_results = ranker.rerank(rerank_request)

                # Sort results based on the re-ranking score
                reranked_results = sorted(reranked_results, key=lambda x: x['score'], reverse=True)

                # Log reranked results
                logging.debug(f"\n\nenhanced_rag_pipeline - Reranked results: {reranked_results}")

                # Update all_results based on reranking
                all_results = [all_results[result['id']] for result in reranked_results]

        # Extract content from results (top fts_top_k by default)
        context = "\n".join([result['content'] for result in all_results[:fts_top_k]])
        #logging.debug(f"Context length: {len(context)}")
        logging.debug(f"Context: {context[:200]}")

        # Generate answer using the selected API
        answer = generate_answer(api_choice, context, query)

        if not all_results:
            logging.info(f"No results found. Query: {query}, Keywords: {keywords}")
            return {
                "answer": "No relevant information based on your query and keywords were found in the database. Your query has been directly passed to the LLM, and here is its answer: \n\n" + answer,
                "context": "No relevant information based on your query and keywords were found in the database. The only context used was your query: \n\n" + query
            }

        # Log metrics
        pipeline_duration = time.time() - start_time
        log_histogram("enhanced_rag_pipeline_duration", pipeline_duration, labels={"api_choice": api_choice})
        log_counter("enhanced_rag_pipeline_success", labels={"api_choice": api_choice})

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        log_counter("enhanced_rag_pipeline_error", labels={"api_choice": api_choice, "error": str(e)})
        logging.error(f"Error in enhanced_rag_pipeline: {str(e)}")
        logging.error(f"Error in enhanced_rag_pipeline: {str(e)}")
        return {
            "answer": "An error occurred while processing your request.",
            "context": ""
        }


# Need to write a test for this function FIXME
def generate_answer(api_choice: str, context: str, query: str) -> str:
    # Metrics
    log_counter("generate_answer_attempt", labels={"api_choice": api_choice})
    start_time = time.time()
    logging.debug("Entering generate_answer function")
    loaded_config_data = load_and_log_configs()

    # Prep the RAG Prompt Dictionary
    file_path = loaded_config_data['chat_dictionaries']['chat_dict_RAG_prompts']
    rag_prompt_placeholder = loaded_config_data['chat_dictionaries']['default_rag_prompt']
    query = rag_prompt_placeholder + query
    rag_prompt_entries = []
    rag_prompt_dict_data = parse_user_dict_markdown_file(file_path)
    print("DEBUG: rag_prompt_dict_data =", rag_prompt_dict_data)
    for k, v in rag_prompt_dict_data.items():
        # k is your "key", v is your "content"
        rag_prompt_entries.append(ChatDictionary(key=k, content=v))

    rag_prompt = process_user_input(query, rag_prompt_entries)
    rag_prompt = f"RAG Prompt: {rag_prompt}\n\n{context}\n\nQuestion: {query}"
    # Non-Prompt Dictionary Version
    #prompt = f"Context: {context}\n\nQuestion: {query}"
    if api_choice:
        try:
            answer_generation_duration = time.time() - start_time
            log_histogram("generate_answer_duration", answer_generation_duration, labels={"api_choice": api_choice})
            api_choice = api_choice.casefold()
            result = summarize(rag_prompt, "", api_choice, loaded_config_data[f'{api_choice}_api']['api_key'], None,
                               None, None)
            log_counter("generate_answer_success", labels={"api_choice": api_choice})
            return result

        except Exception as e:
            log_counter("generate_answer_error", labels={"api_choice": api_choice, "error": str(e)})
            logging.error(f"Error in generate_answer: {str(e)}")
            return "An error occurred while generating the answer."
    else:
        log_counter("generate_answer_error", labels={"api_choice": api_choice, "error": str()})
        raise ValueError(f"Unsupported API choice: {api_choice}")


def perform_vector_search(query: str, relevant_media_ids: List[str] = None, top_k=10) -> List[Dict[str, Any]]:
    log_counter("perform_vector_search_attempt")
    start_time = time.time()
    all_collections = chroma_client.list_collections()
    vector_results = []
    try:
        for collection in all_collections:
            collection_results = vector_search(collection.name, query, k=top_k)
            if not collection_results:
                continue  # Skip empty results
            filtered_results = [
                result for result in collection_results
                if relevant_media_ids is None or result['metadata'].get('media_id') in relevant_media_ids
            ]
            vector_results.extend(filtered_results)
        search_duration = time.time() - start_time
        log_histogram("perform_vector_search_duration", search_duration)
        log_counter("perform_vector_search_success", labels={"result_count": len(vector_results)})
        return vector_results
    except Exception as e:
        log_counter("perform_vector_search_error", labels={"error": str(e)})
        logging.error(f"Error in perform_vector_search: {str(e)}")
        raise


# V2
def perform_full_text_search(query: str, database_type: str, relevant_ids: List[str] = None, fts_top_k=None) -> List[Dict[str, Any]]:
    """
    Perform full-text search on a specified database type.

    Args:
        query: Search query string
        database_type: Type of database to search ("Media DB", "RAG Chat", "RAG Notes", "Character Chat", "Character Cards")
        relevant_ids: Optional list of media IDs to filter results
        fts_top_k: Maximum number of results to return

    Returns:
        List of search results with content and metadata
    """
    log_counter("perform_full_text_search_attempt", labels={"database_type": database_type})
    start_time = time.time()

    try:
        # Set default for fts_top_k
        if fts_top_k is None:
            fts_top_k = 10

        # Call appropriate search function based on database type
        if database_type not in search_functions:
            raise ValueError(f"Unsupported database type: {database_type}")

        # Call the appropriate search function
        results = search_functions[database_type](query, fts_top_k, relevant_ids)

        search_duration = time.time() - start_time
        log_histogram("perform_full_text_search_duration", search_duration,
                      labels={"database_type": database_type})
        log_counter("perform_full_text_search_success",
                    labels={"database_type": database_type, "result_count": len(results)})

        return results

    except Exception as e:
        log_counter("perform_full_text_search_error",
                    labels={"database_type": database_type, "error": str(e)})
        logging.error(f"Error in perform_full_text_search ({database_type}): {str(e)}")
        raise


# v1
# def perform_full_text_search(query: str, relevant_media_ids: List[str] = None, fts_top_k=None) -> List[Dict[str, Any]]:
#     log_counter("perform_full_text_search_attempt")
#     start_time = time.time()
#     try:
#         fts_results = search_db(query, ["content"], "", page=1, results_per_page=fts_top_k or 10)
#         filtered_fts_results = [
#             {
#                 "content": result['content'],
#                 "metadata": {"media_id": result['id']}
#             }
#             for result in fts_results
#             if relevant_media_ids is None or result['id'] in relevant_media_ids
#         ]
#         search_duration = time.time() - start_time
#         log_histogram("perform_full_text_search_duration", search_duration)
#         log_counter("perform_full_text_search_success", labels={"result_count": len(filtered_fts_results)})
#         return filtered_fts_results
#     except Exception as e:
#         log_counter("perform_full_text_search_error", labels={"error": str(e)})
#         logging.error(f"Error in perform_full_text_search: {str(e)}")
#         raise


def fetch_relevant_media_ids(keywords: List[str], top_k=10) -> List[int]:
    log_counter("fetch_relevant_media_ids_attempt", labels={"keyword_count": len(keywords)})
    start_time = time.time()
    relevant_ids = set()
    for keyword in keywords:
        try:
            media_ids = fetch_keywords_for_media(keyword)
            relevant_ids.update(media_ids)
        except Exception as e:
            log_counter("fetch_relevant_media_ids_error", labels={"error": str(e)})
            logging.error(f"Error fetching relevant media IDs for keyword '{keyword}': {str(e)}")
            # Continue processing other keywords

    fetch_duration = time.time() - start_time
    log_histogram("fetch_relevant_media_ids_duration", fetch_duration)
    log_counter("fetch_relevant_media_ids_success", labels={"result_count": len(relevant_ids)})
    return list(relevant_ids)


def filter_results_by_keywords(results: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    log_counter("filter_results_by_keywords_attempt", labels={"result_count": len(results), "keyword_count": len(keywords)})
    start_time = time.time()
    if not keywords:
        return results

    filtered_results = []
    for result in results:
        try:
            metadata = result.get('metadata', {})
            if metadata is None:
                logging.warning(f"No metadata found for result: {result}")
                continue
            if not isinstance(metadata, dict):
                logging.warning(f"Unexpected metadata type: {type(metadata)}. Expected dict.")
                continue

            media_id = metadata.get('media_id')
            if media_id is None:
                logging.warning(f"No media_id found in metadata: {metadata}")
                continue

            media_keywords = fetch_keywords_for_media(media_id)
            if any(keyword.lower() in [mk.lower() for mk in media_keywords] for keyword in keywords):
                filtered_results.append(result)
        except Exception as e:
            logging.error(f"Error processing result: {result}. Error: {str(e)}")

    filter_duration = time.time() - start_time
    log_histogram("filter_results_by_keywords_duration", filter_duration)
    log_counter("filter_results_by_keywords_success", labels={"filtered_count": len(filtered_results)})
    return filtered_results

# FIXME: to be implememted
def extract_media_id_from_result(result: str) -> Optional[int]:
    # Implement this function based on how you store the media_id in your results
    # For example, if it's stored at the beginning of each result:
    try:
        return int(result.split('_')[0])
    except (IndexError, ValueError):
        logging.error(f"Failed to extract media_id from result: {result}")
        return None

#
#
########################################################################################################################


############################################################################################################
#
# Chat RAG

def enhanced_rag_pipeline_chat(query: str, api_choice: str, character_id: int, keywords: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced RAG pipeline tailored for the Character Chat tab.

    Args:
        query (str): The user's input query.
        api_choice (str): The API to use for generating the response.
        character_id (int): The ID of the character being interacted with.
        keywords (Optional[str]): Comma-separated keywords to filter search results.

    Returns:
        Dict[str, Any]: Contains the generated answer and the context used.
    """
    log_counter("enhanced_rag_pipeline_chat_attempt", labels={"api_choice": api_choice, "character_id": character_id})
    start_time = time.time()
    try:
        # Load embedding provider from config, or fallback to 'openai'
        embedding_provider = config.get('Embeddings', 'provider', fallback='openai')
        logging.debug(f"Using embedding provider: {embedding_provider}")

        # Process keywords if provided
        keyword_list = [k.strip().lower() for k in keywords.split(',')] if keywords else []
        logging.debug(f"enhanced_rag_pipeline_chat - Keywords: {keyword_list}")

        # Fetch relevant chat IDs based on character_id and keywords
        if keyword_list:
            relevant_chat_ids = fetch_keywords_for_chats(keyword_list)
        else:
            relevant_chat_ids = fetch_all_chat_ids(character_id)
        logging.debug(f"enhanced_rag_pipeline_chat - Relevant chat IDs: {relevant_chat_ids}")

        if not relevant_chat_ids:
            logging.info(f"No chats found for the given keywords and character ID: {character_id}")
            # Fallback to generating answer without context
            answer = generate_answer(api_choice, "", query)
            # Metrics
            pipeline_duration = time.time() - start_time
            log_histogram("enhanced_rag_pipeline_chat_duration", pipeline_duration, labels={"api_choice": api_choice})
            log_counter("enhanced_rag_pipeline_chat_success",
                        labels={"api_choice": api_choice, "character_id": character_id})
            return {
                "answer": answer,
                "context": ""
            }

        # Perform vector search within the relevant chats
        vector_results = perform_vector_search_chat(query, relevant_chat_ids)
        logging.debug(f"enhanced_rag_pipeline_chat - Vector search results: {vector_results}")

        # Perform full-text search within the relevant chats
        # FIXME - Update for DB Selection
        fts_results = perform_full_text_search_chat(query, relevant_chat_ids)
        logging.debug("enhanced_rag_pipeline_chat - Full-text search results:")
        logging.debug("\n".join([str(item) for item in fts_results]))

        # Combine results
        all_results = vector_results + fts_results

        apply_re_ranking = True
        if apply_re_ranking:
            logging.debug("enhanced_rag_pipeline_chat - Applying Re-Ranking")
            ranker = Ranker()

            # Prepare passages for re-ranking
            passages = [{"id": i, "text": result['content']} for i, result in enumerate(all_results)]
            rerank_request = RerankRequest(query=query, passages=passages)

            # Rerank the results
            reranked_results = ranker.rerank(rerank_request)

            # Sort results based on the re-ranking score
            reranked_results = sorted(reranked_results, key=lambda x: x['score'], reverse=True)

            # Log reranked results
            logging.debug(f"enhanced_rag_pipeline_chat - Reranked results: {reranked_results}")

            # Update all_results based on reranking
            all_results = [all_results[result['id']] for result in reranked_results]

        # Extract context from top results (limit to top 10)
        context = "\n".join([result['content'] for result in all_results[:10]])
        logging.debug(f"Context length: {len(context)}")
        logging.debug(f"Context: {context[:200]}")  # Log only the first 200 characters for brevity

        # Generate answer using the selected API
        answer = generate_answer(api_choice, context, query)

        if not all_results:
            logging.info(f"No results found. Query: {query}, Keywords: {keywords}")
            return {
                "answer": "No relevant information based on your query and keywords were found in the database. Your query has been directly passed to the LLM, and here is its answer: \n\n" + answer,
                "context": "No relevant information based on your query and keywords were found in the database. The only context used was your query: \n\n" + query
            }

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        log_counter("enhanced_rag_pipeline_chat_error", labels={"api_choice": api_choice, "character_id": character_id, "error": str(e)})
        logging.error(f"Error in enhanced_rag_pipeline_chat: {str(e)}")
        return {
            "answer": "An error occurred while processing your request.",
            "context": ""
        }


def fetch_relevant_chat_ids(character_id: int, keywords: List[str]) -> List[int]:
    """
    Fetch chat IDs associated with a character and filtered by keywords.

    Args:
        character_id (int): The ID of the character.
        keywords (List[str]): List of keywords to filter chats.

    Returns:
        List[int]: List of relevant chat IDs.
    """
    log_counter("fetch_relevant_chat_ids_attempt", labels={"character_id": character_id, "keyword_count": len(keywords)})
    start_time = time.time()
    relevant_ids = set()
    try:
        media_ids = fetch_keywords_for_chats(keywords)
        fetch_duration = time.time() - start_time
        log_histogram("fetch_relevant_chat_ids_duration", fetch_duration)
        log_counter("fetch_relevant_chat_ids_success",
                    labels={"character_id": character_id, "result_count": len(relevant_ids)})
        relevant_ids.update(media_ids)
        return list(relevant_ids)
    except Exception as e:
        log_counter("fetch_relevant_chat_ids_error", labels={"character_id": character_id, "error": str(e)})
        logging.error(f"Error fetching relevant chat IDs: {str(e)}")
        return []


def fetch_all_chat_ids(character_id: int) -> List[int]:
    """
    Fetch all chat IDs associated with a specific character.

    Args:
        character_id (int): The ID of the character.

    Returns:
        List[int]: List of all chat IDs for the character.
    """
    log_counter("fetch_all_chat_ids_attempt", labels={"character_id": character_id})
    start_time = time.time()
    try:
        chats = get_character_chats(character_id=character_id)
        chat_ids = [chat['id'] for chat in chats]
        fetch_duration = time.time() - start_time
        log_histogram("fetch_all_chat_ids_duration", fetch_duration)
        log_counter("fetch_all_chat_ids_success", labels={"character_id": character_id, "chat_count": len(chat_ids)})
        return chat_ids
    except Exception as e:
        log_counter("fetch_all_chat_ids_error", labels={"character_id": character_id, "error": str(e)})
        logging.error(f"Error fetching all chat IDs for character {character_id}: {str(e)}")
        return []

#
# End of Chat RAG
############################################################################################################

# Function to preprocess and store all existing content in the database
# def preprocess_all_content(database, create_contextualized=True, api_name="gpt-3.5-turbo"):
#     unprocessed_media = get_unprocessed_media()
#     total_media = len(unprocessed_media)
#
#     for index, row in enumerate(unprocessed_media, 1):
#         media_id, content, media_type, file_name = row
#         collection_name = f"{media_type}_{media_id}"
#
#         logger.info(f"Processing media {index} of {total_media}: ID {media_id}, Type {media_type}")
#
#         try:
#             process_and_store_content(
#                 database=database,
#                 content=content,
#                 collection_name=collection_name,
#                 media_id=media_id,
#                 file_name=file_name or f"{media_type}_{media_id}",
#                 create_embeddings=True,
#                 create_contextualized=create_contextualized,
#                 api_name=api_name
#             )
#
#             # Mark the media as processed in the database
#             mark_media_as_processed(database, media_id)
#
#             logger.info(f"Successfully processed media ID {media_id}")
#         except Exception as e:
#             logger.error(f"Error processing media ID {media_id}: {str(e)}")
#
#     logger.info("Finished preprocessing all unprocessed content")

############################################################################################################
#
# ElasticSearch Retriever

# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-elasticsearch
#
# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-self-query

#
# End of RAG_Library_2.py
############################################################################################################
