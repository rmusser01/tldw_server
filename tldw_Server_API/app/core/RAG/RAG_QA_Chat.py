# RAG_QA_Chat.py
# Description: Functions supporting the RAG QA Chat functionality
#
# Imports
#
# External Imports
import json
import tempfile
import time
from typing import List, Tuple, IO
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import db, search_media_db, DatabaseError
from PoC_Version.App_Function_Libraries.RAG.RAG_Library_2 import generate_answer, enhanced_rag_pipeline
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.Utils import logging
#
########################################################################################################################
#
# Functions:

def rag_qa_chat(query, history, context, api_choice, keywords=None, apply_re_ranking=False):
    log_counter("rag_qa_chat_attempt", labels={"api_choice": api_choice})
    start_time = time.time()

    try:
        if isinstance(context, str):
            log_counter("rag_qa_chat_string_context")
            # Use the answer and context directly from enhanced_rag_pipeline
            result = enhanced_rag_pipeline(query, api_choice, keywords, apply_re_ranking)
            answer = result['answer']
        else:
            log_counter("rag_qa_chat_no_context")
            # If no context is provided, call generate_answer directly
            answer = generate_answer(api_choice, "", query)

        # Update history
        new_history = history + [(query, answer)]

        # Metrics
        duration = time.time() - start_time
        log_histogram("rag_qa_chat_duration", duration, labels={"api_choice": api_choice})
        log_counter("rag_qa_chat_success", labels={"api_choice": api_choice})

        return new_history, answer
    except Exception as e:
        log_counter("rag_qa_chat_error", labels={"api_choice": api_choice, "error": str(e)})
        logging.error(f"Error in rag_qa_chat: {str(e)}")
        return history + [(query, "An error occurred while processing your request.")], "An error occurred while processing your request."


def save_chat_history(history: List[Tuple[str, str]]) -> str:
    # Save chat history to a file
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
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
        # Load chat history from a file
        history = json.load(file)
        load_duration = time.time() - start_time
        log_histogram("load_chat_history_duration", load_duration)
        log_counter("load_chat_history_success")
        return history
    except Exception as e:
        log_counter("load_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error loading chat history: {str(e)}")
        raise

def search_database(query: str) -> List[Tuple[int, str]]:
    try:
        log_counter("search_database_attempt")
        start_time = time.time()
        # Implement database search functionality
        results = search_media_db(query, ["title", "content"], "", page=1, results_per_page=10)
        search_duration = time.time() - start_time
        log_histogram("search_database_duration", search_duration)
        log_counter("search_database_success", labels={"result_count": len(results)})
        return [(result['id'], result['title']) for result in results]
    except Exception as e:
        log_counter("search_database_error", labels={"error": str(e)})
        logging.error(f"Error searching database: {str(e)}")
        raise


def get_existing_files() -> List[Tuple[int, str]]:
    log_counter("get_existing_files_attempt")
    start_time = time.time()
    try:
        # Fetch list of existing files from the database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM Media ORDER BY title")
            results = cursor.fetchall()
        fetch_duration = time.time() - start_time
        log_histogram("get_existing_files_duration", fetch_duration)
        log_counter("get_existing_files_success", labels={"file_count": len(results)})
        return results
    except Exception as e:
        log_counter("get_existing_files_error", labels={"error": str(e)})
        logging.error(f"Error fetching existing files: {str(e)}")
        raise

######################################################
#
# Notes



#
# End of RAG_QA_Chat.py
########################################################################################################################
