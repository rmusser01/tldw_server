# RAG_Library_2.py
# Description: This script contains the main RAG pipeline function and related functions for the RAG pipeline.
#
# Import necessary modules and functions
import configparser
import os
import time
from typing import Dict, Any, List, Optional
# 3rd-Party Imports
#
from flashrank import Ranker, RerankRequest
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database, DatabaseError, fetch_keywords_for_media
from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import vector_search, chroma_client
from tldw_Server_API.app.core.RAG.RAG_Persona_Chat import perform_vector_search_chat
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_notes_by_keywords, search_conversations_by_keywords
from tldw_Server_API.app.core.Utils.Utils import load_and_log_configs, logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Chat.Chat_Functions import process_user_input, ChatDictionary, \
    parse_user_dict_markdown_file, chat
#from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import
from tldw_Server_API.app.core.DB_Management.Character_Chat_DB import get_character_chats, perform_full_text_search_chat, \
    fetch_keywords_for_chats, search_character_chat, search_character_cards, fetch_character_ids_by_keywords
from tldw_Server_API.app.core.DB_Management.RAG_QA_Chat_DB import search_rag_chat, search_rag_notes
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article
#
########################################################################################################################
#
# Functions:

# Initialize OpenAI client (adjust this based on your API key management)
# openai.api_key = "your-openai-api-key" # Ensure this is securely managed and loaded, e.g., from env variables

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the config file
config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
# Read the config file
config = configparser.ConfigParser()
# Read the configuration file
if os.path.exists(config_path):
    config.read(config_path)
elif os.path.exists('config.txt'):  # Fallback to current dir if relative path is used
    config.read('config.txt')
else:
    logging.warning(
        f"Config file 'config.txt' not found in {config_path} or current directory. Using default fallbacks.")

# Search functions for database types OTHER THAN "Media DB".
# "Media DB" will be handled directly by db_instance.search_media_db via perform_full_text_search.
# These functions will need to be updated when their respective DBs are redesigned.
search_functions_non_media_db = {
    "RAG Chat": search_rag_chat,
    "RAG Notes": search_rag_notes,
    "Character Chat": search_character_chat,
    "Character Cards": search_character_cards
}


# RAG pipeline function for web scraping
def rag_web_scraping_pipeline(
        db_instance: Database,  # Added: instance of Media_DB_v2.Database
        url: str,
        query: str,
        api_choice: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs RAG on content scraped from a URL.
    Uses db_instance for Media DB operations.
    """
    log_counter("rag_web_scraping_pipeline_attempt", labels={"url": url})
    start_time = time.time()
    try:
        # Extract content
        article_data: Dict[str, str]
        try:
            # Ensure scrape_article is available, e.g., by importing it:
            # from tldw_Server_API.app.core.Ingestion.Web_Scraping.Scraping_Functions import scrape_article
            if 'scrape_article' not in globals():
                logging.warning("`scrape_article` function not found. Using placeholders for web scraping pipeline.")
                article_data = {'content': f'Placeholder content for {url}', 'title': f'Placeholder Title for {url}'}
            else:
                article_data = scrape_article(url)  # Call the actual function

            content = article_data.get('content', '')
            title = article_data.get('title', f'Untitled Article from {url}')
            if not content:
                logging.error(f"No content extracted from {url}")
                return {"error": "Failed to extract content from article", "details": "Content was empty."}
        except Exception as e:
            logging.error(f"Error scraping article from {url}: {str(e)}", exc_info=True)
            return {"error": "Failed to scrape article", "details": str(e)}

        # Store the article in the Media_DB_v2 database
        media_id: Optional[int] = None
        media_uuid: Optional[str] = None
        try:
            media_id, media_uuid, msg = db_instance.add_media_with_keywords(
                url=url,
                title=title,
                media_type='article',
                content=content,
                overwrite=True  # Or False, depending on desired behavior, True makes sense for repeated scrapes
            )
            logging.info(f"Media DB action for {url}: {msg}. ID: {media_id}, UUID: {media_uuid}")
            if not media_id:  # Check if media_id was successfully returned
                raise DatabaseError(f"Failed to add/update media for {url}: {msg}")
        except DatabaseError as e:
            logging.error(f"Database error adding article {url} to Media_DB_v2: {str(e)}", exc_info=True)
            return {"error": "Failed to store article in Media_DB_v2", "details": str(e)}
        except Exception as e:  # Catch any other unexpected error
            logging.error(f"Unexpected error adding article {url} to Media_DB_v2: {str(e)}", exc_info=True)
            return {"error": "Unexpected error storing article in Media_DB_v2", "details": str(e)}

        # Process and store content (e.g., embeddings in ChromaDB)
        # This part needs your specific 'process_and_store_content' implementation.
        # It might involve creating chunks in Media_DB_v2 (e.g., db_instance.process_unvectorized_chunks)
        # and then generating embeddings for ChromaDB.
        collection_name = f"article_{media_id}"  # Chroma collection name
        try:
            if 'process_and_store_content' in globals() and media_id is not None:
                # Example call, adjust based on your function's signature
                # process_and_store_content(
                #     database_instance=db_instance, # If it needs Media_DB_v2 instance
                #     content=content,
                #     collection_name=collection_name, # For ChromaDB
                #     media_id=str(media_id),      # ChromaDB might expect string IDs
                #     file_name=title,
                #     create_embeddings=True,      # For ChromaDB
                #     # create_contextualized=True,  # If this is part of its logic
                #     api_name=api_choice
                # )
                pass  # Replace with actual call
            else:
                logging.warning(
                    f"'process_and_store_content' function not found or media_id is None. Skipping embedding/ChromaDB storage for {url}.")
        except Exception as e:
            logging.error(f"Error in 'process_and_store_content' for {url}: {str(e)}", exc_info=True)
            # Depending on criticality, you might return an error or just log and continue

        # Perform searches
        vector_search_results_content: List[str] = []
        fts_search_results_content: List[str] = []
        try:
            # Vector search (ChromaDB)
            if media_id is not None:  # Only search if we have a media_id for the collection
                raw_vector_results = vector_search(collection_name, query, k=5)
                if raw_vector_results:
                    for res in raw_vector_results:
                        text = res.get('content', res.get('text'))
                        if text: vector_search_results_content.append(text)

            # FTS search (Media_DB_v2, restricted to the current article)
            if media_id is not None:
                media_results_list, _ = db_instance.search_media_db(
                    search_query=query,
                    search_fields=['title', 'content'],  # Search these fields within the article
                    media_ids_filter=[str(media_id)],  # Crucially, filter to *only this article*
                    results_per_page=5
                )
                if media_results_list:
                    for item in media_results_list:
                        if item.get('content'): fts_search_results_content.append(item['content'])
        except DatabaseError as e:
            logging.error(f"Database error performing searches for {url}: {str(e)}", exc_info=True)
            return {"error": "Database error during search", "details": str(e)}
        except Exception as e:
            logging.error(f"Error performing searches for {url}: {str(e)}", exc_info=True)
            return {"error": "Failed to perform searches", "details": str(e)}

        # Combine results (simple concatenation for now, consider deduplication or better merging)
        all_search_content = vector_search_results_content + fts_search_results_content
        # Basic deduplication while preserving order for the first N unique items
        unique_content_pieces = list(dict.fromkeys(all_search_content))
        context = "\n\n---\n\n".join(unique_content_pieces[:10])  # Limit context length if needed

        if not context and query:  # If no specific content found, use the article's main content
            context = content  # Fallback to the full article content
            logging.info(
                f"No specific search results for query '{query}' in article '{title}'. Using full article content as context.")
        elif not context:
            context = f"No specific content found for query '{query}' in {title}."

        # Generate answer
        try:
            answer = generate_answer(api_choice, context, query)
        except Exception as e:
            logging.error(f"Error generating answer for {url} query '{query}': {str(e)}", exc_info=True)
            return {"error": "Failed to generate answer", "details": str(e)}

        pipeline_duration = time.time() - start_time
        log_histogram("rag_web_scraping_pipeline_duration", pipeline_duration,
                      labels={"api_choice": api_choice or "default"})
        log_counter("rag_web_scraping_pipeline_success", labels={"api_choice": api_choice or "default"})
        return {
            "answer": answer,
            "context": context,
            "media_id": media_id,
            "media_uuid": media_uuid
        }
    except Exception as e:
        pipeline_duration = time.time() - start_time
        log_histogram("rag_web_scraping_pipeline_duration", pipeline_duration,
                      labels={"api_choice": api_choice or "default"})
        log_counter("rag_web_scraping_pipeline_error",
                    labels={"api_choice": api_choice or "default", "error_type": type(e).__name__})
        logging.error(f"Unexpected error in rag_web_scraping_pipeline for {url}: {str(e)}", exc_info=True)
        return {"error": "An unexpected error occurred in web scraping RAG pipeline", "details": str(e)}


def fetch_relevant_media_ids_for_media_db(db_instance: Database, keywords_list: List[str]) -> List[str]:
    """
    Fetches unique media IDs (as strings) from Media_DB_v2 associated with the given keywords.
    Uses db_instance.fetch_media_for_keywords.
    """
    if not keywords_list:
        return []
    try:
        media_items_by_keyword: Dict[str, List[Dict[str, Any]]] = db_instance.fetch_media_for_keywords(
            keywords=keywords_list,
            include_trash=False  # Sensible default to exclude trashed items for relevance
        )
        relevant_ids_set = set()
        for single_keyword_media_list in media_items_by_keyword.values():
            for media_item in single_keyword_media_list:
                # 'id' is the primary key of the Media table in Media_DB_v2
                if 'id' in media_item and media_item['id'] is not None:
                    relevant_ids_set.add(str(media_item['id']))
                # The returned dict from fetch_media_for_keywords uses aliased names like 'media_id'
                elif 'media_id' in media_item and media_item['media_id'] is not None:
                    relevant_ids_set.add(str(media_item['media_id']))

        logging.debug(
            f"Fetched {len(relevant_ids_set)} unique media IDs for Media DB based on keywords: {keywords_list}")
        return list(relevant_ids_set)
    except DatabaseError as e:
        logging.error(f"Database error fetching relevant media IDs for Media DB: {str(e)}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching relevant media IDs for Media DB: {str(e)}", exc_info=True)
        return []


def enhanced_rag_pipeline(
        db_instance: Database,  # Added: instance of Media_DB_v2.Database
        query: str,
        api_choice: str,
        keywords: Optional[str] = None,
        fts_top_k: int = 10,
        vector_top_k: int = 10,
        search_fields: Optional[List[str]] = None,
        apply_re_ranking: bool = True,
        database_types: List[str] = ["Media DB"]
) -> Dict[str, Any]:
    """
    Perform RAG search across specified database types, using Media_DB_v2 for "Media DB".
    """
    log_counter("enhanced_rag_pipeline_attempt",
                labels={"api_choice": api_choice, "db_types": ",".join(database_types)})
    start_time = time.time()

    # Default search fields for Media DB if not provided
    search_fields_for_media_db = search_fields if search_fields is not None else ["title", "content"]

    try:
        embedding_provider = config.get('Embeddings', 'provider', fallback='openai')
        logging.debug(f"Using embedding provider: {embedding_provider}")

        relevant_ids_by_type: Dict[str, Optional[List[str]]] = {db_type: None for db_type in database_types}
        keyword_list = [k.strip().lower() for k in keywords.split(',')] if keywords and keywords.strip() else []

        if keyword_list:
            logging.debug(f"enhanced_rag_pipeline - Processing keywords: {keyword_list}")
            for db_type in database_types:
                try:
                    if db_type == "Media DB":
                        media_ids_for_filter = fetch_relevant_media_ids_for_media_db(db_instance, keyword_list)
                        relevant_ids_by_type[db_type] = media_ids_for_filter
                    # Placeholder for other DB keyword logic - adapt when those DBs are redesigned
                    elif db_type == "RAG Chat":
                        conversations, _, _ = search_conversations_by_keywords(
                            keywords=keyword_list)  # Old DB_Manager import
                        relevant_ids_by_type[db_type] = [str(conv['conversation_id']) for conv in conversations]
                    elif db_type == "RAG Notes":
                        notes, _, _ = get_notes_by_keywords(keyword_list)  # Old DB_Manager import
                        relevant_ids_by_type[db_type] = [str(note[0]) for note in
                                                         notes]  # Assuming note_id is first element
                    elif db_type == "Character Chat":
                        relevant_ids_by_type[db_type] = [str(id_) for id_ in fetch_keywords_for_chats(
                            keyword_list)]  # From Character_Chat_DB
                    elif db_type == "Character Cards":
                        relevant_ids_by_type[db_type] = [str(id_) for id_ in fetch_character_ids_by_keywords(
                            keyword_list)]  # From Character_Chat_DB
                    else:
                        logging.warning(f"Unsupported database type for keyword filtering: {db_type}")
                    logging.debug(
                        f"enhanced_rag_pipeline - Relevant IDs for {db_type} from keywords: {relevant_ids_by_type[db_type]}")
                except Exception as e:
                    logging.error(f"Error fetching relevant IDs for {db_type} with keywords: {str(e)}", exc_info=True)

        # Perform vector search (ChromaDB)
        all_vector_results: List[Dict[str, Any]] = []
        for db_type in database_types:
            ids_for_vector_filter = relevant_ids_by_type.get(db_type)
            try:
                # perform_vector_search is external to Media_DB_v2, uses Chroma
                results = perform_vector_search(query, ids_for_vector_filter, top_k=vector_top_k)
                all_vector_results.extend(results)  # results are List[Dict]
                logging.debug(
                    f"enhanced_rag_pipeline - Vector search results for {db_type} (filtered by {len(ids_for_vector_filter) if ids_for_vector_filter else 0} IDs): {len(results)} results.")
            except Exception as e:
                logging.error(f"Error performing vector search for {db_type}: {str(e)}", exc_info=True)

        # Perform full-text search
        all_fts_results: List[Dict[str, Any]] = []
        for db_type in database_types:
            ids_for_fts_filter = relevant_ids_by_type.get(db_type)
            try:
                # Use specific search fields for Media DB, None for others (letting them use defaults)
                current_search_fields = search_fields_for_media_db if db_type == "Media DB" else None
                results = perform_full_text_search(
                    db_instance=db_instance,
                    query=query,
                    database_type=db_type,
                    relevant_ids=ids_for_fts_filter,
                    fts_top_k=fts_top_k,
                    search_fields=current_search_fields
                )
                all_fts_results.extend(results)  # results are List[Dict]
                logging.debug(
                    f"enhanced_rag_pipeline - FTS results for {db_type} (filtered by {len(ids_for_fts_filter) if ids_for_fts_filter else 0} IDs): {len(results)} results.")
            except Exception as e:
                logging.error(f"Error performing full-text search for {db_type}: {str(e)}", exc_info=True)

        logging.debug(
            f"Total vector results pre-rerank: {len(all_vector_results)}, Total FTS results pre-rerank: {len(all_fts_results)}")

        # Combine results for re-ranking
        combined_docs_for_rerank: List[Dict[str, Any]] = []
        doc_counter = 0
        # Add vector results
        for res_item in all_vector_results:
            text_content = res_item.get('content', res_item.get('text'))
            if text_content:
                combined_docs_for_rerank.append({
                    "text": text_content,
                    "metadata": res_item.get('metadata', {}),
                    "rerank_id": f"vec_{doc_counter}",  # Unique ID for reranking
                    "source": "vector"
                })
                doc_counter += 1
        # Add FTS results
        for res_item in all_fts_results:
            text_content = res_item.get('content', res_item.get('text'))
            if text_content:
                # Ensure FTS results also have 'metadata' if vector results do, for consistency
                fts_metadata = res_item.get('metadata', {})
                if not fts_metadata:  # If perform_full_text_search didn't add it for this db_type
                    fts_metadata = {'item_id': str(res_item.get('id')),
                                    'source_db': res_item.get('source_db_type', 'unknown_fts')}

                combined_docs_for_rerank.append({
                    "text": text_content,
                    "metadata": fts_metadata,
                    "rerank_id": f"fts_{doc_counter}",  # Unique ID for reranking
                    "source": "fts"
                })
                doc_counter += 1

        # Deduplicate based on text content after combining, before reranking
        # This simple deduplication prefers the first occurrence (vector results then FTS)
        seen_texts = set()
        unique_combined_docs = []
        for doc in combined_docs_for_rerank:
            if doc['text'] not in seen_texts:
                unique_combined_docs.append(doc)
                seen_texts.add(doc['text'])
        combined_docs_for_rerank = unique_combined_docs
        logging.debug(f"Combined unique documents for reranking: {len(combined_docs_for_rerank)}")

        final_context_docs = combined_docs_for_rerank  # Default to combined if no reranking

        if apply_re_ranking and combined_docs_for_rerank:
            logging.debug(
                f"enhanced_rag_pipeline - Applying Re-Ranking to {len(combined_docs_for_rerank)} combined documents.")
            ranker = Ranker()  # Consider model selection for Ranker if needed, e.g., Ranker(model_name="ms-marco-MiniLM-L-12-v2")

            passages_for_rerank = [{"id": item["rerank_id"], "text": item["text"]} for item in combined_docs_for_rerank]

            if passages_for_rerank:
                rerank_request = RerankRequest(query=query, passages=passages_for_rerank)
                try:
                    reranked_scores = ranker.rerank(rerank_request)  # Returns List[{'id': any, 'score': float}]

                    score_map = {score_item['id']: score_item['score'] for score_item in reranked_scores}

                    for item in combined_docs_for_rerank:
                        item['rerank_score'] = score_map.get(item['rerank_id'], -float('inf'))  # Default low score

                    reranked_docs_sorted = sorted(combined_docs_for_rerank, key=lambda x: x['rerank_score'],
                                                  reverse=True)
                    final_context_docs = reranked_docs_sorted
                    logging.debug(
                        f"enhanced_rag_pipeline - Reranked {len(final_context_docs)} documents. Top 3 scores: {[r['rerank_score'] for r in final_context_docs[:3]]}")
                except Exception as e:
                    logging.error(f"Error during re-ranking: {str(e)}", exc_info=True)
                    # Fallback to using non-reranked (but combined and deduplicated) results
            else:
                logging.debug("No valid passages to rerank.")

        # Extract content for the final context string
        # Use fts_top_k as the limit for the number of documents in the context.
        # This could be a separate parameter like `context_max_docs`.
        context_limit = fts_top_k
        context_pieces = [doc['text'] for doc in final_context_docs[:context_limit] if doc.get('text')]
        context = "\n\n---\n\n".join(context_pieces)  # Use a clear separator

        logging.debug(
            f"Final context length: {len(context)}. Using top {min(len(context_pieces), context_limit)} documents.")
        if len(context) > 1000:
            logging.debug(f"Context snippet: {context[:500]}...{context[-500:]}")
        else:
            logging.debug(f"Context: {context}")

        # Generate answer
        answer = generate_answer(api_choice, context, query)

        if not final_context_docs and not context:  # If absolutely no context could be formed
            logging.info(f"No results found after search and filtering. Query: {query}, Keywords: {keywords}")
            # Pass original query to LLM if no context is found
            empty_context_answer = generate_answer(api_choice, "", query)  # "" for context
            return {
                "answer": "No relevant information based on your query and keywords was found in the database. "
                          "Your query has been directly passed to the LLM, and here is its answer: \n\n" + empty_context_answer,
                "context": "No relevant information based on your query and keywords was found in the database. "
                           "The only context used was your query: \n\n" + query
            }

        pipeline_duration = time.time() - start_time
        log_histogram("enhanced_rag_pipeline_duration", pipeline_duration, labels={"api_choice": api_choice})
        log_counter("enhanced_rag_pipeline_success", labels={"api_choice": api_choice})

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        log_counter("enhanced_rag_pipeline_error", labels={"api_choice": api_choice, "error_type": type(e).__name__})
        logging.error(f"Critical error in enhanced_rag_pipeline: {str(e)}", exc_info=True)
        try:
            direct_llm_answer = generate_answer(api_choice, "", query)  # Fallback LLM call
            return {
                "answer": f"An error occurred while processing your request with contextual search. "
                          f"However, the LLM provided a direct answer to your query: \n\n{direct_llm_answer}",
                "context": f"Error during RAG pipeline: {str(e)}"
            }
        except Exception as llm_e:
            logging.error(f"Fallback LLM call also failed: {str(llm_e)}", exc_info=True)
            return {
                "answer": "A critical error occurred while processing your request, and the LLM could not be reached.",
                "context": f"RAG pipeline error: {str(e)}; Fallback LLM error: {str(llm_e)}"
            }


def generate_answer(api_choice: Optional[str], context: str, query: str) -> str:
    # Metrics
    log_counter("generate_answer_attempt", labels={"api_choice": api_choice or "default"})
    start_time = time.time()
    logging.debug("Entering generate_answer function")

    loaded_config_data = load_and_log_configs()
    if not loaded_config_data:
        logging.error("Failed to load configurations for generate_answer.")
        return "Error: System configuration missing."

    # Prepare the RAG Prompt using ChatDictionary for the query part
    chat_dict_config = loaded_config_data.get('chat_dictionaries', {})
    rag_prompts_file_path = chat_dict_config.get('chat_dict_RAG_prompts')
    default_rag_prompt_template = chat_dict_config.get('default_rag_prompt', "Query: {query}")

    initial_query_for_chatdict = default_rag_prompt_template.replace("{query}", query)  # Safer replacement

    rag_prompt_entries: List[ChatDictionary] = []
    if rag_prompts_file_path and os.path.exists(rag_prompts_file_path):
        try:
            rag_prompt_dict_data = parse_user_dict_markdown_file(rag_prompts_file_path)
            for k, v in rag_prompt_dict_data.items():
                rag_prompt_entries.append(ChatDictionary(key=k, content=v))  # type: ignore
        except Exception as e:
            logging.error(f"Failed to parse RAG prompts dictionary from {rag_prompts_file_path}: {e}")

    processed_query_part = process_user_input(initial_query_for_chatdict, rag_prompt_entries)
    logging.debug(f"Processed query part for RAG: {processed_query_part}")

    # Context truncation (remains the same)
    max_context_len_chars = 15000
    if len(context) > max_context_len_chars:
        logging.warning(f"Context length ({len(context)} chars) exceeds limit ({max_context_len_chars}). Truncating.")
        context = context[:max_context_len_chars] + "\n... (context truncated)"

    # The `final_llm_prompt` is no longer directly used. Instead, its components
    # (`context` and `processed_query_part`) are passed to the `chat` function.

    if api_choice:
        api_choice_lower = api_choice.lower()
        api_config_key = f'{api_choice_lower}_api'
        specific_api_config = loaded_config_data.get(api_config_key, {})  # Default to empty dict

        if 'api_key' not in specific_api_config:
            logging.error(f"Configuration for API '{api_choice}' (key: {api_config_key}) not found or missing API key.")
            log_counter("generate_answer_error", labels={"api_choice": api_choice, "error": "API_config_missing"})
            return f"Error: Configuration for API '{api_choice}' is missing or incomplete."

        # --- Call the `chat` function instead of `analyze` ---
        try:
            # Prepare parameters for the `chat` function
            chat_message = processed_query_part
            chat_media_content = {"rag_context": context}  # Package context for `chat`
            chat_selected_parts = ["rag_context"]
            chat_api_key = specific_api_config['api_key']

            # Define a RAG-specific system message
            rag_system_message = (
                "You are a helpful AI assistant. Your task is to answer the user's question based "
                "on the provided 'rag_context'. Analyze the context thoroughly. "
                "If the context contains relevant information, use it to construct your answer. "
                "If the context does not seem relevant or is insufficient to answer the question, "
                "clearly state that the provided context is not helpful and then answer the question "
                "based on your general knowledge. Be concise and directly answer the question."
            )

            # Get other LLM parameters from config, with defaults matching `chat` or being None
            chat_temperature = float(specific_api_config.get('temperature', 0.7))
            chat_model = specific_api_config.get('model')  # Can be None
            chat_max_tokens = int(specific_api_config.get('max_tokens', 500))  # Default from chat
            chat_topp = specific_api_config.get('topp')  # Can be None
            chat_topk = specific_api_config.get('topk')  # Can be None
            # minp, maxp seem less common, default to None or get from config if available
            chat_minp = specific_api_config.get('minp')
            chat_maxp = specific_api_config.get('maxp')

            logging.debug(f"Calling `chat` function with: api_endpoint='{api_choice_lower}', "
                          f"temperature={chat_temperature}, model='{chat_model}', max_tokens={chat_max_tokens}")

            result = chat(
                message=chat_message,
                history=[],  # RAG is typically single-turn for this step
                media_content=chat_media_content,
                selected_parts=chat_selected_parts,
                api_endpoint=api_choice_lower,  # `chat` uses api_endpoint
                api_key=chat_api_key,
                custom_prompt=None,  # Use system_message for primary instruction
                temperature=chat_temperature,
                system_message=rag_system_message,
                streaming=False,  # `generate_answer` is non-streaming
                minp=chat_minp,
                maxp=chat_maxp,
                model=chat_model,
                topp=chat_topp,
                topk=chat_topk,
                chatdict_entries=None,  # Query part already processed by generate_answer
                max_tokens=chat_max_tokens,
                # strategy= "sorted_evenly" # Use chat's default strategy
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


def perform_vector_search(query: str, relevant_media_ids: Optional[List[str]] = None, top_k: int = 10) -> List[
    Dict[str, Any]]:
    """
    Performs vector search using ChromaDB across all collections, optionally filtered by relevant_media_ids.
    """
    log_counter("perform_vector_search_attempt")
    start_time = time.time()
    all_collections = chroma_client.list_collections()
    vector_results: List[Dict[str, Any]] = []
    if not all_collections:
        logging.warning("No ChromaDB collections found for vector search.")
        return []
    try:
        for collection in all_collections:
            # vector_search returns List[Dict] with 'content', 'metadata', 'distance'/'score'
            collection_results = vector_search(collection.name, query, k=top_k)
            if not collection_results:
                continue

            # Filter results if relevant_media_ids are provided
            # This assumes 'media_id' is stored in result['metadata']['media_id'] as a string.
            if relevant_media_ids:
                filtered_for_collection = []
                for result in collection_results:
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, dict) and metadata.get('media_id') in relevant_media_ids:
                        filtered_for_collection.append(result)
                vector_results.extend(filtered_for_collection)
            else:  # No filtering by media_id, take all results from this collection
                vector_results.extend(collection_results)

        # If results from multiple collections, sort them globally by distance/score if available
        # and then take top_k. For simplicity, this is not done here yet.
        # The current approach concatenates top_k from each collection.
        # A more refined approach would fetch more from each, then globally rerank/select.
        # For now, just ensure we don't exceed top_k * too_many_collections.
        # A simple sort and trim if distance/score is present:
        if vector_results and 'distance' in vector_results[0]:  # Chroma often uses 'distance'
            vector_results.sort(key=lambda x: x.get('distance', float('inf')))  # Lower distance is better
        elif vector_results and 'score' in vector_results[0]:  # Other systems might use 'score'
            vector_results.sort(key=lambda x: x.get('score', -float('inf')), reverse=True)  # Higher score is better

        final_vector_results = vector_results[:top_k]  # Trim to overall top_k

        search_duration = time.time() - start_time
        log_histogram("perform_vector_search_duration", search_duration)
        log_counter("perform_vector_search_success", labels={"result_count": len(final_vector_results)})
        return final_vector_results
    except Exception as e:
        log_counter("perform_vector_search_error", labels={"error": str(e)})
        logging.error(f"Error in perform_vector_search: {str(e)}", exc_info=True)
        raise  # Re-raise to be caught by the calling pipeline


def perform_full_text_search(
        db_instance: Database,
        query: str,
        database_type: str,
        relevant_ids: Optional[List[str]] = None,
        fts_top_k: Optional[int] = 10,
        search_fields: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Perform full-text search on a specified database type.
    Uses db_instance for "Media DB" searches.
    """
    log_counter("perform_full_text_search_attempt", labels={"database_type": database_type})
    start_time = time.time()

    effective_fts_top_k = fts_top_k if fts_top_k and fts_top_k > 0 else 10
    results: List[Dict[str, Any]] = []

    try:
        if database_type == "Media DB":
            if not db_instance:
                logging.error("db_instance (Media_DB_v2) is required for 'Media DB' search but was not provided.")
                raise ValueError("db_instance is required for 'Media DB' search.")

            effective_search_fields = search_fields if search_fields else ["title", "content"]

            media_db_results, total_matches = db_instance.search_media_db(
                search_query=query,
                search_fields=effective_search_fields,
                keywords=None,  # Keyword filtering is assumed to be done via relevant_ids
                media_ids_filter=relevant_ids,
                page=1,
                results_per_page=effective_fts_top_k
            )
            logging.info(f"Media DB FTS found {total_matches} total, returning up to {len(media_db_results)}")
            # Structure Media DB results for consistency
            for item in media_db_results:
                item['metadata'] = {'media_id': str(item.get('id')), 'uuid': item.get('uuid'), 'source_db': 'Media DB',
                                    'title': item.get('title')}
                # Ensure 'content' field. If Media table stores main text in 'content', it's already there.
                if 'content' not in item or not item['content']:  # If content is empty or missing
                    item['content'] = item.get('title', '')  # Fallback to title
            results = media_db_results

        elif database_type in search_functions_non_media_db:
            search_func = search_functions_non_media_db[database_type]
            # These functions need to be adapted if their signature for relevant_ids or top_k changes
            # Assuming they return a list of dicts or a tuple (list_of_dicts, count, ...)
            retrieved_data = search_func(query, effective_fts_top_k, relevant_ids)

            actual_results_list: List[Dict[str, Any]] = []
            if isinstance(retrieved_data, tuple) and len(retrieved_data) > 0 and isinstance(retrieved_data[0], list):
                actual_results_list = retrieved_data[0]
            elif isinstance(retrieved_data, list):
                actual_results_list = retrieved_data
            else:
                logging.warning(
                    f"Unexpected return type from {database_type} search: {type(retrieved_data)}. Expected list or tuple.")

            # Standardize structure for other DB types
            for item in actual_results_list:
                if isinstance(item, dict):
                    item_id_key = 'id' if 'id' in item else 'conversation_id' if 'conversation_id' in item else 'note_id'
                    item_id_val = item.get(item_id_key, f"unknown_{database_type}_id")
                    item_title = item.get('title', item.get('name', query[:30]))  # Some sensible title
                    item['metadata'] = item.get('metadata', {'item_id': str(item_id_val), 'source_db': database_type,
                                                             'title': item_title})
                    if 'content' not in item or not item['content']:
                        item['content'] = item.get('text', item.get('summary', item_title))  # Common fallbacks
                    results.append(item)
                else:  # If item is not a dict, try to coerce
                    results.append(
                        {'content': str(item), 'metadata': {'source_db': database_type, 'title': str(item)[:30]}})
        else:
            logging.error(f"Unsupported database type for FTS: {database_type}")
            raise ValueError(f"Unsupported database type for FTS: {database_type}")

        search_duration = time.time() - start_time
        log_histogram("perform_full_text_search_duration", search_duration, labels={"database_type": database_type})
        log_counter("perform_full_text_search_success",
                    labels={"database_type": database_type, "result_count": len(results)})
        return results

    except DatabaseError as e:  # Specifically for Media_DB_v2 errors
        log_counter("perform_full_text_search_error",
                    labels={"database_type": database_type, "error_type": type(e).__name__})
        logging.error(f"Media_DB_v2 Database error in perform_full_text_search ({database_type}): {str(e)}",
                      exc_info=True)
        raise
    except Exception as e:
        log_counter("perform_full_text_search_error",
                    labels={"database_type": database_type, "error_type": type(e).__name__})
        logging.error(f"Error in perform_full_text_search ({database_type}): {str(e)}", exc_info=True)
        raise


def filter_results_by_keywords(
        db_instance: Database,
        results: List[Dict[str, Any]],
        keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    Filters a list of search results based on whether their associated media items
    (for "Media DB" type) have any of the specified keywords.
    For other DB types, this function currently doesn't apply keyword filtering
    unless their metadata explicitly contains keywords that can be matched.
    """
    log_counter("filter_results_by_keywords_attempt",
                labels={"result_count": len(results), "keyword_count": len(keywords)})
    start_time = time.time()
    if not keywords:
        return results  # No keywords to filter by

    # Normalize input keywords
    normalized_filter_keywords = [k.lower() for k in keywords]
    filtered_results = []

    for result in results:
        try:
            metadata = result.get('metadata', {})
            if not isinstance(metadata, dict):
                logging.warning(
                    f"Unexpected metadata type {type(metadata)} for result. Skipping keyword filter for this item.")
                filtered_results.append(result)  # Keep if cannot determine keywords
                continue

            source_db = metadata.get('source_db')
            item_matches_keyword = False

            if source_db == "Media DB":
                media_id_str = metadata.get('media_id')
                if media_id_str:
                    try:
                        media_id_int = int(media_id_str)
                        media_item_keywords = fetch_keywords_for_media(media_id_int, db_instance)
                        normalized_media_item_keywords = [mk.lower() for mk in media_item_keywords]
                        if any(filter_kw in normalized_media_item_keywords for filter_kw in normalized_filter_keywords):
                            item_matches_keyword = True
                    except ValueError:
                        logging.warning(
                            f"Could not convert media_id '{media_id_str}' to int for Media DB keyword fetching.")
                    except NameError:
                        logging.error(
                            "`standalone_fetch_keywords_for_media` not imported. Cannot filter Media DB items by keyword.")
                    except Exception as kw_e:
                        logging.error(f"Error fetching keywords for media_id {media_id_str} (Media DB): {kw_e}")
                else:  # No media_id for Media DB item, cannot filter by keywords
                    logging.warning(f"Media DB item missing 'media_id' in metadata: {metadata}")
            else:
                # For other DB types, we assume they might have keywords directly in metadata (e.g., metadata['keywords'])
                # Or this part needs to be extended for each DB type when they are redesigned.
                # For now, if not Media DB, we keep the result if no specific keyword logic exists for it.
                # This means filter_results_by_keywords primarily works for Media DB items.
                item_matches_keyword = True  # Default to keeping if not Media DB and no other logic
                # Example for other DB types if they store keywords:
                # item_db_keywords = metadata.get('keywords', []) # Assuming a list of strings
                # normalized_item_db_keywords = [k.lower() for k in item_db_keywords]
                # if any(filter_kw in normalized_item_db_keywords for filter_kw in normalized_filter_keywords):
                #    item_matches_keyword = True

            if item_matches_keyword:
                filtered_results.append(result)

        except Exception as e:
            logging.error(f"Error processing result during keyword filtering: {result}. Error: {str(e)}", exc_info=True)
            filtered_results.append(result)  # Keep result if error occurs during its keyword check

    filter_duration = time.time() - start_time
    log_histogram("filter_results_by_keywords_duration", filter_duration)
    log_counter("filter_results_by_keywords_success", labels={"filtered_count": len(filtered_results)})
    return filtered_results


# Deprecated / Placeholder functions from original
def fetch_relevant_media_ids(keywords: List[str], top_k=10) -> List[int]:
    """
    DEPRECATED - This function was for the old DB_Manager.
    Use fetch_relevant_media_ids_for_media_db for Media_DB_v2.
    For other DBs, specific fetchers should be used.
    """
    logging.warning("`fetch_relevant_media_ids` is deprecated. Use specific fetchers for each DB type.")
    # This was likely calling the old `fetch_keywords_for_media` from DB_Manager which is removed for Media DB.
    # For other DB types, their respective keyword-to-ID functions are called directly in enhanced_rag_pipeline.
    return []


def extract_media_id_from_result(result: str) -> Optional[int]:
    # This function's logic depends heavily on the format of `result` string.
    # Given results are now dictionaries, this string parsing is likely not needed.
    # If it were, result['metadata']['media_id'] would be the way.
    logging.warning("`extract_media_id_from_result` is likely deprecated as results are dicts.")
    try:
        # Example: if result was "123_some_text"
        return int(result.split('_')[0])
    except (IndexError, ValueError, AttributeError):
        # logging.error(f"Failed to extract media_id from result string: {result}") # Too noisy
        return None


############################################################################################################
#
# Chat RAG (enhanced_rag_pipeline_chat and its helpers)
# These functions primarily interact with Character_Chat_DB and RAG_Persona_Chat.
# They do NOT use the Media_DB_v2.Database instance directly, so they remain largely unchanged
# with respect to Media_DB_v2 integration.
# If Character Chat ever needs to reference media from Media_DB_v2, then db_instance
# would need to be passed into enhanced_rag_pipeline_chat.

def enhanced_rag_pipeline_chat(query: str, api_choice: str, character_id: int, keywords: Optional[str] = None) -> Dict[
    str, Any]:
    """
    Enhanced RAG pipeline tailored for the Character Chat tab.
    Uses Character_Chat_DB and RAG_Persona_Chat, not Media_DB_v2.
    """
    log_counter("enhanced_rag_pipeline_chat_attempt", labels={"api_choice": api_choice, "character_id": character_id})
    start_time = time.time()
    try:
        embedding_provider = config.get('Embeddings', 'provider', fallback='openai')
        logging.debug(f"Using embedding provider for chat RAG: {embedding_provider}")

        keyword_list = [k.strip().lower() for k in keywords.split(',')] if keywords and keywords.strip() else []
        logging.debug(f"enhanced_rag_pipeline_chat - Keywords: {keyword_list}")

        # Fetch relevant chat IDs (from Character_Chat_DB)
        relevant_chat_ids: List[int]
        if keyword_list:
            relevant_chat_ids = fetch_keywords_for_chats(keyword_list)  # From Character_Chat_DB
        else:
            relevant_chat_ids = fetch_all_chat_ids(character_id)  # From Character_Chat_DB
        logging.debug(f"enhanced_rag_pipeline_chat - Relevant chat IDs: {relevant_chat_ids}")

        if not relevant_chat_ids:
            logging.info(
                f"No chats found for character_id {character_id} with keywords: {keyword_list}. Generating answer without specific chat context.")
            answer = generate_answer(api_choice, "", query)  # Empty context
            pipeline_duration = time.time() - start_time
            log_histogram("enhanced_rag_pipeline_chat_duration", pipeline_duration, labels={"api_choice": api_choice})
            log_counter("enhanced_rag_pipeline_chat_success_no_context",
                        labels={"api_choice": api_choice, "character_id": character_id})
            return {"answer": answer, "context": "No specific chat history found for context."}

        # Perform vector search within the relevant chats (from RAG_Persona_Chat)
        # perform_vector_search_chat likely uses ChromaDB with collections specific to chats
        vector_results_chat = perform_vector_search_chat(query, relevant_chat_ids, top_k=10)  # Assuming top_k=10
        logging.debug(f"enhanced_rag_pipeline_chat - Vector search results: {len(vector_results_chat)} items")

        # Perform full-text search within the relevant chats (from Character_Chat_DB)
        fts_results_chat = perform_full_text_search_chat(query, relevant_chat_ids, top_k=10)  # Assuming top_k=10
        logging.debug(f"enhanced_rag_pipeline_chat - FTS results: {len(fts_results_chat)} items")
        # logging.debug("\n".join([str(item.get('content', item)[:100]) for item in fts_results_chat[:3]]))

        # Combine results (these are specific to chat content)
        all_chat_results_for_rerank: List[Dict[str, Any]] = []
        chat_doc_counter = 0
        for res_list, source_type in [(vector_results_chat, "vector_chat"), (fts_results_chat, "fts_chat")]:
            for res_item in res_list:
                text_content = res_item.get('content', res_item.get('text'))
                if text_content:
                    all_chat_results_for_rerank.append({
                        "text": text_content,
                        "metadata": res_item.get('metadata', {}),  # Preserve metadata
                        "rerank_id": f"{source_type}_{chat_doc_counter}",
                        "source": source_type
                    })
                    chat_doc_counter += 1

        # Deduplicate
        seen_chat_texts = set()
        unique_chat_docs = []
        for doc in all_chat_results_for_rerank:
            if doc['text'] not in seen_chat_texts:
                unique_chat_docs.append(doc)
                seen_chat_texts.add(doc['text'])
        all_chat_results_for_rerank = unique_chat_docs

        final_chat_context_docs = all_chat_results_for_rerank

        apply_re_ranking_chat = True  # Configurable?
        if apply_re_ranking_chat and all_chat_results_for_rerank:
            logging.debug("enhanced_rag_pipeline_chat - Applying Re-Ranking to chat results.")
            ranker = Ranker()
            passages_chat = [{"id": item["rerank_id"], "text": item["text"]} for item in all_chat_results_for_rerank]
            if passages_chat:
                rerank_request_chat = RerankRequest(query=query, passages=passages_chat)
                try:
                    reranked_chat_scores = ranker.rerank(rerank_request_chat)
                    score_map_chat = {score_item['id']: score_item['score'] for score_item in reranked_chat_scores}
                    for item in all_chat_results_for_rerank:
                        item['rerank_score'] = score_map_chat.get(item['rerank_id'], -float('inf'))

                    final_chat_context_docs = sorted(all_chat_results_for_rerank, key=lambda x: x['rerank_score'],
                                                     reverse=True)
                    logging.debug(
                        f"Reranked {len(final_chat_context_docs)} chat documents. Top 3 scores: {[r['rerank_score'] for r in final_chat_context_docs[:3]]}")
                except Exception as e_rank_chat:
                    logging.error(f"Error during chat results re-ranking: {e_rank_chat}", exc_info=True)

        # Extract context from top results (limit to top 10 for chat)
        chat_context_limit = 10
        chat_context_pieces = [doc['text'] for doc in final_chat_context_docs[:chat_context_limit] if doc.get('text')]
        context_chat = "\n\n---\n\n".join(chat_context_pieces)

        logging.debug(
            f"Chat RAG Context length: {len(context_chat)}. Using top {min(len(chat_context_pieces), chat_context_limit)} documents.")
        if len(context_chat) > 500:
            logging.debug(f"Chat Context snippet: {context_chat[:250]}...{context_chat[-250:]}")
        else:
            logging.debug(f"Chat Context: {context_chat}")

        # Generate answer
        answer = generate_answer(api_choice, context_chat, query)

        if not final_chat_context_docs and not context_chat:
            logging.info(f"No chat results found for RAG. Query: {query}, Keywords: {keywords}")
            # Fallback: generate_answer already called with empty context if no relevant_chat_ids
            # If relevant_chat_ids existed but search yielded nothing, this provides a targeted message.
            return {
                "answer": "No specific chat history snippets were found for your query. The LLM will answer based on its general knowledge and the character's persona (if defined).\n\n" + answer,
                # answer here is from empty context
                "context": "No relevant chat snippets found. The query was: " + query
            }

        pipeline_duration = time.time() - start_time
        log_histogram("enhanced_rag_pipeline_chat_duration", pipeline_duration, labels={"api_choice": api_choice})
        log_counter("enhanced_rag_pipeline_chat_success",
                    labels={"api_choice": api_choice, "character_id": character_id})
        return {
            "answer": answer,
            "context": context_chat
        }

    except Exception as e:
        log_counter("enhanced_rag_pipeline_chat_error",
                    labels={"api_choice": api_choice, "character_id": character_id, "error_type": type(e).__name__})
        logging.error(f"Error in enhanced_rag_pipeline_chat: {str(e)}", exc_info=True)
        try:
            direct_llm_answer_chat = generate_answer(api_choice, "", query)
            return {
                "answer": f"An error occurred retrieving chat context. The LLM provided this direct answer:\n\n{direct_llm_answer_chat}",
                "context": f"Error during Chat RAG pipeline: {str(e)}"
            }
        except Exception as llm_e_chat:
            logging.error(f"Fallback LLM call also failed for chat RAG: {str(llm_e_chat)}", exc_info=True)
            return {
                "answer": "A critical error occurred processing your chat request, and the LLM could not be reached.",
                "context": f"Chat RAG pipeline error: {str(e)}; Fallback LLM error: {str(llm_e_chat)}"
            }


def fetch_relevant_chat_ids(character_id: int, keywords: List[str]) -> List[int]:
    """
    DEPRECATED - Original was likely a general placeholder.
    Use fetch_keywords_for_chats from Character_Chat_DB.py.
    This function definition is kept to avoid breaking old calls if any, but logs a warning.
    """
    logging.warning("`fetch_relevant_chat_ids` is deprecated. Use `fetch_keywords_for_chats` from Character_Chat_DB.")
    if keywords:
        return fetch_keywords_for_chats(keywords)
    return fetch_all_chat_ids(character_id)  # Fallback if no keywords


def fetch_all_chat_ids(character_id: int) -> List[int]:
    """
    Fetch all chat IDs associated with a specific character using Character_Chat_DB.
    """
    log_counter("fetch_all_chat_ids_attempt", labels={"character_id": character_id})
    start_time = time.time()
    try:
        chats = get_character_chats(character_id=character_id)  # From Character_Chat_DB
        chat_ids = [chat['id'] for chat in chats if isinstance(chat, dict) and 'id' in chat]
        fetch_duration = time.time() - start_time
        log_histogram("fetch_all_chat_ids_duration", fetch_duration)
        log_counter("fetch_all_chat_ids_success", labels={"character_id": character_id, "chat_count": len(chat_ids)})
        return chat_ids
    except Exception as e:
        log_counter("fetch_all_chat_ids_error", labels={"character_id": character_id, "error_type": type(e).__name__})
        logging.error(f"Error fetching all chat IDs for character {character_id}: {str(e)}", exc_info=True)
        return []

#
# End of Chat RAG
############################################################################################################

# preprocess_all_content function was commented out in the original RAG library.
# If re-enabled, it would need significant updates to use Media_DB_v2 instance methods
# for fetching unprocessed media, marking as processed, and potentially for chunking/storing
# content if that's part of its role before embedding.
#
# Example structure if it were to be updated:
# def preprocess_all_content(db_instance: Database, create_contextualized=True, api_name="gpt-3.5-turbo"):
#     # 1. Fetch unprocessed media using db_instance.get_unprocessed_media() (this method exists in Media_DB_v2)
#     unprocessed_media_items = db_instance.get_unprocessed_media() # Returns List[Dict]
#     total_media = len(unprocessed_media_items)
#     logger.info(f"Found {total_media} unprocessed media items in Media_DB_v2.")
#
#     for index, media_item in enumerate(unprocessed_media_items, 1):
#         media_id = media_item['id']
#         content = media_item['content']
#         media_type = media_item['type']
#         file_name = media_item.get('title', f"{media_type}_{media_id}") # Use title as filename
#         collection_name = f"{media_type}_{media_id}" # For ChromaDB
#
#         logger.info(f"Processing media {index} of {total_media}: ID {media_id}, Type {media_type}")
#
#         try:
#             # This is where your 'process_and_store_content' function would be called.
#             # It needs to handle:
#             # - Potentially chunking content (maybe using db_instance.process_unvectorized_chunks)
#             # - Storing chunks/metadata in Media_DB_v2 if not already done.
#             # - Generating embeddings for chunks.
#             # - Storing embeddings in ChromaDB (using collection_name, media_id as metadata).
#             # - Optionally, creating "contextualized" versions (details depend on function).
#
#             # Placeholder:
#             # process_and_store_content(
#             #     database_instance=db_instance, # Pass if it needs to write to Media_DB_v2
#             #     content=content,
#             #     collection_name=collection_name,
#             #     media_id=str(media_id),
#             #     file_name=file_name,
#             #     create_embeddings=True,
#             #     create_contextualized=create_contextualized,
#             #     api_name=api_name
#             # )
#
#             # After successful processing (including embedding and storage in Chroma):
#             db_instance.mark_media_as_processed(media_id) # Mark vector_processing=1
#             logger.info(f"Successfully processed and marked media ID {media_id} as processed.")
#         except Exception as e:
#             logger.error(f"Error processing media ID {media_id}: {str(e)}", exc_info=True)
#
#     logger.info("Finished preprocessing all unprocessed content from Media_DB_v2.")

############################################################################################################
#
# ElasticSearch Retriever (Placeholder Comments)
# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-elasticsearch
# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-self-query
#
# End of RAG_Library_2.py
############################################################################################################