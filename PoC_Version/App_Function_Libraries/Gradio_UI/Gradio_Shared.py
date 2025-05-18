# Gradio_Shared.py
# Gradio UI functions that are shared across multiple tabs
#
# Imports
import sqlite3
import traceback
from functools import wraps
from typing import List, Tuple
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import list_prompts, db, search_and_display, fetch_prompt_details
from App_Function_Libraries.DB.SQLite_DB import DatabaseError
from App_Function_Libraries.Utils.Utils import format_transcription, logging

#
##############################################################################################################
#
# Functions:

whisper_models = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en", "distil-small.en",
        "distil-large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2", "nyrahealth/faster_CrisperWhisper"]

# Sample data
prompts_category_1 = [
    "What are the key points discussed in the video?",
    "Summarize the main arguments made by the speaker.",
    "Describe the conclusions of the study presented."
]

prompts_category_2 = [
    "How does the proposed solution address the problem?",
    "What are the implications of the findings?",
    "Can you explain the theory behind the observed phenomenon?"
]

all_prompts = prompts_category_1 + prompts_category_2



#FIXME - SQL Functions that need to be addressed/added to DB manager
def search_media(query, fields, keyword, page):
    try:
        results = search_and_display(query, fields, keyword, page)
        return results
    except Exception as e:
        logging.error(f"Error searching media: {e}")
        return str(e)

def fetch_items_by_title_or_url(search_query: str, search_type: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by {search_type}: {e}")

def fetch_items_by_keyword(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.title, m.url
                FROM Media m
                JOIN MediaKeywords mk ON m.id = mk.media_id
                JOIN Keywords k ON mk.keyword_id = k.id
                WHERE k.keyword LIKE ?
            """, (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by keyword: {e}")

# FIXME - Raw SQL not using DB_Manager...
def fetch_items_by_content(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by content: {e}")



# FIXME - RAW SQL not using DB_Manager...
def fetch_item_details_single(media_id: int):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt, summary 
                FROM MediaModifications 
                WHERE media_id = ? 
                ORDER BY modification_date DESC 
                LIMIT 1
            """, (media_id,))
            prompt_summary_result = cursor.fetchone()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()

            prompt = prompt_summary_result[0] if prompt_summary_result else ""
            summary = prompt_summary_result[1] if prompt_summary_result else ""
            content = content_result[0] if content_result else ""

            return prompt, summary, content
    except sqlite3.Error as e:
        raise Exception(f"Error fetching item details: {e}")


# FIXME - RAW SQL not using DB_Manager...
def fetch_item_details(media_id: int):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt, summary 
                FROM MediaModifications 
                WHERE media_id = ? 
                ORDER BY modification_date DESC 
                LIMIT 1
            """, (media_id,))
            prompt_summary_result = cursor.fetchone()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()

            prompt = prompt_summary_result[0] if prompt_summary_result else ""
            summary = prompt_summary_result[1] if prompt_summary_result else ""
            content = content_result[0] if content_result else ""

            return content, prompt, summary
    except sqlite3.Error as e:
        logging.error(f"Error fetching item details: {e}")
        return "", "", ""  # Return empty strings if there's an error

# Handle prompt selection
def handle_prompt_selection(prompt):
    return f"You selected: {prompt}"


def update_user_prompt(preset_name):
    details = fetch_prompt_details(preset_name)
    if details:
        # Return a dictionary with all details
        return {
            "title": details[0],
            "author": details[1],
            "details": details[2],
            "system_prompt": details[3],
            "user_prompt": details[4] if len(details) > 3 else "",
        }
    return {"title": "", "details": "", "system_prompt": "", "user_prompt": "", "author": ""}

def browse_items(search_query, search_type):
    if search_type == 'Keyword':
        results = fetch_items_by_keyword(search_query)
    elif search_type == 'Content':
        results = fetch_items_by_content(search_query)
    else:
        results = fetch_items_by_title_or_url(search_query, search_type)
    return results


def update_dropdown(search_query, search_type):
    results = browse_items(search_query, search_type)
    item_options = [f"{item[1]} ({item[2]})" for item in results]
    new_item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}
    print(f"Debug - Update Dropdown - New Item Mapping: {new_item_mapping}")
    return gr.update(choices=item_options), new_item_mapping



def get_media_id(selected_item, item_mapping):
    return item_mapping.get(selected_item)


def update_detailed_view(item, item_mapping):
    # Function to update the detailed view based on selected item
    if item:
        item_id = item_mapping.get(item)
        if item_id:
            content, prompt, summary = fetch_item_details(item_id)
            if content or prompt or summary:
                details_html = "<h4>Details:</h4>"
                if prompt:
                    formatted_prompt = format_transcription(prompt)
                    details_html += f"<h4>Prompt:</h4>{formatted_prompt}</p>"
                if summary:
                    formatted_summary = format_transcription(summary)
                    details_html += f"<h4>Summary:</h4>{formatted_summary}</p>"
                # Format the transcription content for better readability
                formatted_content = format_transcription(content)
                #content_html = f"<h4>Transcription:</h4><div style='white-space: pre-wrap;'>{content}</div>"
                content_html = f"<h4>Transcription:</h4><div style='white-space: pre-wrap;'>{formatted_content}</div>"
                return details_html, content_html
            else:
                return "No details available.", "No details available."
        else:
            return "No item selected", "No item selected"
    else:
        return "No item selected", "No item selected"


def format_content(content):
    # Format content using markdown
    formatted_content = f"```\n{content}\n```"
    return formatted_content


def display_prompt_details(selected_prompt):
    if selected_prompt:
        prompts = update_user_prompt(selected_prompt)
        if prompts["title"]:  # Check if we have any details
            details_str = f"<h4>Details:</h4><p>{prompts['details']}</p>"
            system_str = f"<h4>System:</h4><p>{prompts['system_prompt']}</p>"
            user_str = f"<h4>User:</h4><p>{prompts['user_prompt']}</p>" if prompts['user_prompt'] else ""
            return details_str + system_str + user_str
    return "No details available."

def search_media_database(query: str) -> List[Tuple[int, str, str]]:
    return browse_items(query, 'Title')


def load_media_content(media_id: int) -> dict:
    try:
        print(f"Debug - Load Media Content - Media ID: {media_id}")
        item_details = fetch_item_details(media_id)
        print(f"Debug - Load Media Content - Item Details: \n\n{item_details}\n\n\n\n")

        if isinstance(item_details, tuple) and len(item_details) == 3:
            content, prompt, summary = item_details
        else:
            print(f"Debug - Load Media Content - Unexpected item_details format: \n\n{item_details}\n\n\n\n")
            content, prompt, summary = "", "", ""

        return {
            "content": content or "No content available",
            "prompt": prompt or "No prompt available",
            "summary": summary or "No summary available"
        }
    except Exception as e:
        print(f"Debug - Load Media Content - Error: {str(e)}")
        return {"content": "", "prompt": "", "summary": ""}


def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"Error in {func.__name__}: {str(e)}"
            logging.error(f"{error_message}\n{traceback.format_exc()}")
            return {"error": error_message, "details": traceback.format_exc()}
    return wrapper


def create_chunking_inputs():
    chunk_text_by_words_checkbox = gr.Checkbox(label="Chunk Text by Words", value=False, visible=True)
    max_words_input = gr.Number(label="Max Words", value=300, precision=0, visible=True)
    chunk_text_by_sentences_checkbox = gr.Checkbox(label="Chunk Text by Sentences", value=False, visible=True)
    max_sentences_input = gr.Number(label="Max Sentences", value=10, precision=0, visible=True)
    chunk_text_by_paragraphs_checkbox = gr.Checkbox(label="Chunk Text by Paragraphs", value=False, visible=True)
    max_paragraphs_input = gr.Number(label="Max Paragraphs", value=5, precision=0, visible=True)
    chunk_text_by_tokens_checkbox = gr.Checkbox(label="Chunk Text by Tokens", value=False, visible=True)
    max_tokens_input = gr.Number(label="Max Tokens", value=1000, precision=0, visible=True)
    gr_semantic_chunk_long_file = gr.Checkbox(label="Semantic Chunking by Sentence similarity", value=False, visible=True)
    gr_semantic_chunk_long_file_size = gr.Number(label="Max Chunk Size", value=2000, visible=True)
    gr_semantic_chunk_long_file_overlap = gr.Number(label="Max Chunk Overlap Size", value=100, visible=True)
    return [chunk_text_by_words_checkbox, max_words_input, chunk_text_by_sentences_checkbox, max_sentences_input,
            chunk_text_by_paragraphs_checkbox, max_paragraphs_input, chunk_text_by_tokens_checkbox, max_tokens_input]


def ask_clear_chat():
    """First step: show confirmation UI (Markdown + 2 new buttons)."""
    return (
        gr.update(visible=True),  # Show the confirmation row
        "Are you sure you want to clear the chat?"
    )

def confirm_clear_chat(confirm):
    """Second step: either clear the chat or do nothing."""
    if confirm == "yes":
        # Actual clear-chat logic
        return [], None, gr.update(visible=False), ""
    else:
        # Cancel, just hide the confirmation row
        return None, None, gr.update(visible=False), ""

#
# End of Gradio_Shared.py
#######################################################################################################################
