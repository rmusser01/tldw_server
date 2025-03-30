# RAG_QA_Chat_tab.py
# Description: Gradio UI for RAG QA Chat
#
# Imports
import csv
import json
import os
import re
import time
from datetime import datetime
#
# External Imports
import docx2txt
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Books.Book_Ingestion_Lib import read_epub
from PoC_Version.App_Function_Libraries.DB.Character_Chat_DB import search_character_chat, search_character_cards
from PoC_Version.App_Function_Libraries.DB.DB_Manager import DatabaseError, get_paginated_files, add_media_with_keywords, \
    get_all_conversations, get_note_by_id, get_notes_by_keywords, start_new_conversation, update_note, save_notes, \
    clear_keywords_from_note, add_keywords_to_note, load_chat_history, save_message, add_keywords_to_conversation, \
    get_keywords_for_note, delete_note, search_conversations_by_keywords, get_conversation_title, delete_conversation, \
    update_conversation_title, fetch_all_conversations, fetch_all_notes, fetch_conversations_by_ids, fetch_notes_by_ids, \
    search_media_db, search_notes_titles, list_prompts
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import get_notes, delete_messages_in_conversation, search_rag_notes, \
    search_rag_chat, get_conversation_rating, set_conversation_rating
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import update_user_prompt
from PoC_Version.App_Function_Libraries.PDF.PDF_Ingestion_Lib import extract_text_and_format_from_pdf
from PoC_Version.App_Function_Libraries.RAG.RAG_Library_2 import generate_answer, enhanced_rag_pipeline
from PoC_Version.App_Function_Libraries.RAG.RAG_QA_Chat import rag_qa_chat
from PoC_Version.App_Function_Libraries.TTS.TTS_Providers import generate_audio
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, \
    load_and_log_configs, logging
#
########################################################################################################################
#
# Functions:

def create_rag_qa_chat_tab():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    with gr.TabItem("RAG QA Chat", visible=True):
        gr.Markdown("# RAG QA Chat")

        state = gr.State({
            "page": 1,
            "context_source": "Entire Media Database",
            "conversation_messages": [],
            "conversation_id": None
        })

        note_state = gr.State({"note_id": None})

        def auto_save_conversation(message, response, state_value, auto_save_enabled):
            """Automatically save the conversation if auto-save is enabled"""
            try:
                if not auto_save_enabled:
                    return state_value

                conversation_id = state_value.get("conversation_id")
                if not conversation_id:
                    # Create new conversation with default title
                    title = "Auto-saved Conversation " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    conversation_id = start_new_conversation(title=title)
                    state_value = state_value.copy()
                    state_value["conversation_id"] = conversation_id

                # Save the messages
                save_message(conversation_id, "user", message)
                save_message(conversation_id, "assistant", response)

                return state_value
            except Exception as e:
                logging.error(f"Error in auto-save: {str(e)}")
                return state_value

        # Update the conversation list function
        def update_conversation_list():
            conversations, total_pages, total_count = get_all_conversations()
            choices = [
                f"{conversation['title']} (ID: {conversation['conversation_id']}) - Rating: {conversation['rating'] or 'Not Rated'}"
                for conversation in conversations
            ]
            return choices

        with gr.Row():
            with gr.Column(scale=1):
                # FIXME - Offer the user to search 2+ databases at once
                database_types = ["Media DB", "RAG Chat", "RAG Notes", "Character Chat", "Character Cards"]
                db_choice = gr.CheckboxGroup(
                    label="Select Database(s)",
                    choices=database_types,
                    value=["Media DB"],
                    interactive=True
                )
                context_source = gr.Radio(
                    ["All Files in the Database", "Search Database", "Upload File"],
                    label="Context Source",
                    value="All Files in the Database"
                )
                existing_file = gr.Dropdown(label="Select Existing File", choices=[], interactive=True)
                file_page = gr.State(value=1)
                with gr.Row():
                    prev_page_btn = gr.Button("Previous Page")
                    next_page_btn = gr.Button("Next Page")
                    page_info = gr.HTML("Page 1")
                top_k_input = gr.Number(value=10, label="Maximum amount of results to use (Default: 10)", minimum=1, maximum=50, step=1, precision=0, interactive=True)
                keywords_input = gr.Textbox(label="Keywords (comma-separated) to filter results by)", value="rag_qa_default_keyword" ,visible=True)
                use_query_rewriting = gr.Checkbox(label="Use Query Rewriting", value=True)
                use_re_ranking = gr.Checkbox(label="Use Re-ranking", value=True)
                loaded_config = load_and_log_configs()
                auto_save_value = loaded_config['auto-save']['save_character_chats']
                if auto_save_value is None:
                    auto_save_value = False
                else:
                    auto_save_value = auto_save_value.lower() == True
                auto_save_checkbox = gr.Checkbox(
                    label="Save chats automatically",
                    value=auto_save_value,
                    info="When enabled, conversations will be saved automatically after each message"
                )

                initial_prompts, total_pages, current_page = list_prompts(page=1, per_page=10)

                preset_prompt_checkbox = gr.Checkbox(
                    label="View Custom Prompts(have to copy/paste them)",
                    value=False,
                    visible=True
                )

                with gr.Row(visible=False) as preset_prompt_controls:
                    prev_prompt_page = gr.Button("Previous")
                    current_prompt_page_text = gr.Text(f"Page {current_page} of {total_pages}")
                    next_prompt_page = gr.Button("Next")
                    current_prompt_page_state = gr.State(value=1)

                preset_prompt = gr.Dropdown(
                    label="Select Preset Prompt",
                    choices=initial_prompts,
                    visible=False
                )
                user_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Enter custom prompt here",
                    lines=3,
                    visible=False
                )

                system_prompt_input = gr.Textbox(
                    label="System Prompt",
                    lines=3,
                    visible=False
                )

                search_query = gr.Textbox(label="Search Query", visible=False)
                search_button = gr.Button("Search", visible=False)
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                file_upload = gr.File(
                    label="Upload File",
                    visible=False,
                    file_types=["txt", "pdf", "epub", "md", "rtf", "json", "csv", "docx"]
                )
                convert_to_text = gr.Checkbox(label="Convert to plain text", visible=False)

            with gr.Column(scale=1):
                load_conversation = gr.Dropdown(
                    label="Load Conversation",
                    choices=update_conversation_list()
                )
                new_conversation = gr.Button("New Conversation")
                save_conversation_button = gr.Button("Save Conversation")
                conversation_title = gr.Textbox(
                    label="Conversation Title",
                    placeholder="Enter a title for the new conversation"
                )
                keywords = gr.Textbox(label="Keywords (comma-separated)", visible=True)

                # Add the rating display and input
                rating_display = gr.Markdown(value="", visible=False)
                rating_input = gr.Radio(
                    choices=["1", "2", "3"],
                    label="Rate this Conversation (1-3 stars)",
                    visible=False
                )

                # Refactored API selection dropdown
                api_choice = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Response (Optional)"
                )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=700)
                msg = gr.Textbox(label="Enter your message")
                submit = gr.Button("Submit")
                with gr.Row():
                    speak_button = gr.Button("Speak Response")
                    tts_status = gr.Textbox(label="TTS Status", interactive=False)
                with gr.Row():
                    audio_output = gr.Audio(label="Generated Audio", visible=False)
                clear_chat = gr.Button("Clear Chat History")

            with gr.Column(scale=1):
                # Adding UI elements for notes
                note_title = gr.Textbox(label="Note Title", placeholder="Enter a title for the note")
                notes = gr.TextArea(label="Notes", placeholder="Enter your notes here...", lines=25)
                keywords_for_notes = gr.Textbox(
                    label="Keywords for Notes (comma-separated)",
                    placeholder="Enter keywords for the note",
                    visible=True,
                )
                save_notes_btn = gr.Button("Save Note")
                clear_notes_btn = gr.Button("Clear Current Note text")

                new_note_btn = gr.Button("New Note")
                # FIXME - Change from only keywords to generalized search
                search_notes_title = gr.Textbox(label="Search Notes by Title")
                search_notes_by_keyword = gr.Textbox(label="Search Notes by Keyword")
                search_notes_button = gr.Button("Search Notes")
                note_results = gr.Dropdown(label="Notes", choices=[])
                load_note = gr.Dropdown(label="Load Note", choices=[])

        loading_indicator = gr.HTML("Loading...", visible=False)
        status_message = gr.HTML()
        auto_save_status = gr.HTML()



        # Function Definitions
        def update_prompt_page(direction, current_page_val):
            new_page = max(1, min(total_pages, current_page_val + direction))
            prompts, _, _ = list_prompts(page=new_page, per_page=10)
            return (
                gr.update(choices=prompts),
                gr.update(value=f"Page {new_page} of {total_pages}"),
                new_page
            )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def toggle_preset_prompt(checkbox_value):
            return (
                gr.update(visible=checkbox_value),
                gr.update(visible=checkbox_value),
                gr.update(visible=False),
                gr.update(visible=False)
            )

        prev_prompt_page.click(
            lambda x: update_prompt_page(-1, x),
            inputs=[current_prompt_page_state],
            outputs=[preset_prompt, current_prompt_page_text, current_prompt_page_state]
        )

        next_prompt_page.click(
            lambda x: update_prompt_page(1, x),
            inputs=[current_prompt_page_state],
            outputs=[preset_prompt, current_prompt_page_text, current_prompt_page_state]
        )

        preset_prompt.change(
            update_prompts,
            inputs=preset_prompt,
            outputs=[user_prompt, system_prompt_input]
        )

        preset_prompt_checkbox.change(
            toggle_preset_prompt,
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt, preset_prompt_controls, user_prompt, system_prompt_input]
        )

        def update_state(state, **kwargs):
            new_state = state.copy()
            new_state.update(kwargs)
            return new_state

        def create_new_note():
            return gr.update(value='un-named note'), gr.update(value=''), {"note_id": None}

        new_note_btn.click(
            create_new_note,
            outputs=[note_title, notes, note_state]
        )

        def search_notes(search_notes_title, keywords):
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',')]
                notes_data, total_pages, total_count = get_notes_by_keywords(keywords_list)
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"Found {total_count} notes")
            elif search_notes_title:
                notes_data, total_pages, total_count = search_notes_titles(search_notes_title)
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"Found {total_count} notes")
            else:
                # This will now return all notes, ordered by timestamp
                notes_data, total_pages, total_count = search_notes_titles("")
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"All notes ({total_count} total)")

        search_notes_button.click(
            search_notes,
            inputs=[search_notes_title, search_notes_by_keyword],
            outputs=[note_results]
        )

        def load_selected_note(note_selection):
            if note_selection:
                note_id = int(note_selection.split(' ')[1])
                note_data = get_note_by_id(note_id)
                if note_data:
                    note_id, title, content = note_data[0]
                    updated_note_state = {"note_id": note_id}
                    return gr.update(value=title), gr.update(value=content), updated_note_state
            return gr.update(value=''), gr.update(value=''), {"note_id": None}

        note_results.change(
            load_selected_note,
            inputs=[note_results],
            outputs=[note_title, notes, note_state]
        )

        def save_notes_function(note_title_text, notes_content, keywords_content, note_state_value, state_value):
            """Save the notes and associated keywords to the database."""
            logging.info(f"Starting save_notes_function with state: {state_value}")
            logging.info(f"Note title: {note_title_text}")
            logging.info(f"Notes content length: {len(notes_content) if notes_content else 0}")

            try:
                # Check current state
                conversation_id = state_value.get("conversation_id")
                logging.info(f"Current conversation_id: {conversation_id}")

                # Create new conversation if none exists
                if not conversation_id:
                    logging.info("No conversation ID found, creating new conversation")
                    conversation_title = note_title_text if note_title_text else "Untitled Conversation"
                    conversation_id = start_new_conversation(title=conversation_title)
                    state_value = state_value.copy()  # Create a new copy of the state
                    state_value["conversation_id"] = conversation_id
                    logging.info(f"Created new conversation with ID: {conversation_id}")

                if not notes_content:
                    logging.warning("No notes content provided")
                    return notes_content, note_state_value, state_value, gr.update(
                        value="<p style='color:red;'>Cannot save empty notes.</p>")

                # Save or update note
                note_id = note_state_value.get("note_id")
                if note_id:
                    logging.info(f"Updating existing note with ID: {note_id}")
                    update_note(note_id, note_title_text, notes_content)
                else:
                    logging.info(f"Creating new note for conversation: {conversation_id}")
                    note_id = save_notes(conversation_id, note_title_text or "Untitled Note", notes_content)
                    note_state_value = {"note_id": note_id}
                    logging.info(f"Created new note with ID: {note_id}")

                # Handle keywords
                if keywords_content:
                    logging.info("Processing keywords")
                    clear_keywords_from_note(note_id)
                    keywords = [kw.strip() for kw in keywords_content.split(',')]
                    add_keywords_to_note(note_id, keywords)
                    logging.info(f"Added keywords: {keywords}")

                logging.info("Notes saved successfully")
                return (
                    notes_content,
                    note_state_value,
                    state_value,
                    gr.update(value="<p style='color:green;'>Notes saved successfully!</p>")
                )

            except Exception as e:
                logging.error(f"Error in save_notes_function: {str(e)}", exc_info=True)
                return (
                    notes_content,
                    note_state_value,
                    state_value,
                    gr.update(value=f"<p style='color:red;'>Error saving notes: {str(e)}</p>")
                )

        save_notes_btn.click(
            save_notes_function,
            inputs=[note_title, notes, keywords_for_notes, note_state, state],
            outputs=[notes, note_state, state, status_message]
        )

        def clear_notes_function():
            """Clear notes for the current note."""
            return gr.update(value=''), {"note_id": None}

        clear_notes_btn.click(
            clear_notes_function,
            outputs=[notes, note_state]
        )

        # Initialize the conversation list
        load_conversation.choices = update_conversation_list()

        def load_conversation_history(selected_conversation, state_value):
            try:
                if not selected_conversation:
                    return [], state_value, "", gr.update(value="", visible=False), gr.update(visible=False)
                # Extract conversation ID
                match = re.search(r'\(ID: ([0-9a-fA-F\-]+)\)', selected_conversation)
                if not match:
                    logging.error(f"Invalid conversation format: {selected_conversation}")
                    return [], state_value, "", gr.update(value="", visible=False), gr.update(visible=False)
                conversation_id = match.group(1)
                chat_data, total_pages_val, _ = load_chat_history(conversation_id, 1, 50)
                # Update state with valid conversation id
                updated_state = state_value.copy()
                updated_state["conversation_id"] = conversation_id
                updated_state["conversation_messages"] = chat_data
                # Format chat history
                history = []
                for role, content in chat_data:
                    if role == 'user':
                        history.append((content, ''))
                    elif history:
                        history[-1] = (history[-1][0], content)
                # Fetch and display the conversation rating
                rating = get_conversation_rating(conversation_id)
                if rating is not None:
                    rating_text = f"**Current Rating:** {rating} star(s)"
                    rating_display_update = gr.update(value=rating_text, visible=True)
                    rating_input_update = gr.update(value=str(rating), visible=True)
                else:
                    rating_display_update = gr.update(value="**Current Rating:** Not Rated", visible=True)
                    rating_input_update = gr.update(value=None, visible=True)
                notes_content = get_notes(conversation_id)
                return history, updated_state, "\n".join(
                    notes_content) if notes_content else "", rating_display_update, rating_input_update
            except Exception as e:
                logging.error(f"Error loading conversation: {str(e)}")
                return [], state_value, "", gr.update(value="", visible=False), gr.update(visible=False)

        load_conversation.change(
            load_conversation_history,
            inputs=[load_conversation, state],
            outputs=[chatbot, state, notes, rating_display, rating_input]
        )

        # Modify save_conversation_function to use gr.update()
        def save_conversation_function(conversation_title_text, keywords_text, rating_value, state_value):
            conversation_messages = state_value.get("conversation_messages", [])
            conversation_id = state_value.get("conversation_id")
            if not conversation_messages:
                return gr.update(
                    value="<p style='color:red;'>No conversation to save.</p>"
                ), state_value, gr.update(), gr.update(value="", visible=False), gr.update(visible=False)
            # Start a new conversation in the database if not existing
            if not conversation_id:
                conversation_id = start_new_conversation(
                    conversation_title_text if conversation_title_text else "Untitled Conversation"
                )
            else:
                # Update the conversation title if it has changed
                update_conversation_title(conversation_id, conversation_title_text)
            # Save the messages
            for role, content in conversation_messages:
                save_message(conversation_id, role, content)
            # Save keywords if provided
            if keywords_text:
                add_keywords_to_conversation(conversation_id, [kw.strip() for kw in keywords_text.split(',')])
            # Save the rating if provided
            try:
                if rating_value:
                    set_conversation_rating(conversation_id, int(rating_value))
            except ValueError as ve:
                logging.error(f"Invalid rating value: {ve}")
                return gr.update(
                    value=f"<p style='color:red;'>Invalid rating: {ve}</p>"
                ), state_value, gr.update(), gr.update(value="", visible=False), gr.update(visible=False)

            # Update state
            updated_state = update_state(state_value, conversation_id=conversation_id)
            # Update the conversation list
            conversation_choices = update_conversation_list()
            # Reset rating display and input
            rating_display_update = gr.update(value=f"**Current Rating:** {rating_value} star(s)", visible=True)
            rating_input_update = gr.update(value=rating_value, visible=True)
            return gr.update(
                value="<p style='color:green;'>Conversation saved successfully.</p>"
            ), updated_state, gr.update(choices=conversation_choices), rating_display_update, rating_input_update

        save_conversation_button.click(
            save_conversation_function,
            inputs=[conversation_title, keywords, rating_input, state],
            outputs=[status_message, state, load_conversation, rating_display, rating_input]
        )

        def start_new_conversation_wrapper(title, state_value):
            # Reset the state with no conversation_id and empty conversation messages
            updated_state = update_state(state_value, conversation_id=None, page=1, conversation_messages=[])
            # Clear the chat history and reset rating components
            return [], updated_state, gr.update(value="", visible=False), gr.update(value=None, visible=False)

        new_conversation.click(
            start_new_conversation_wrapper,
            inputs=[conversation_title, state],
            outputs=[chatbot, state, rating_display, rating_input]
        )

        def update_file_list(page):
            files, total_pages, current_page = get_paginated_files(page)
            choices = [f"{title} (ID: {id})" for id, title in files]
            return gr.update(choices=choices), gr.update(value=f"Page {current_page} of {total_pages}"), current_page

        def next_page_fn(current_page):
            return update_file_list(current_page + 1)

        def prev_page_fn(current_page):
            return update_file_list(max(1, current_page - 1))

        def update_context_source(choice):
            # Update visibility based on context source choice
            return {
                existing_file: gr.update(visible=choice == "Existing File"),
                prev_page_btn: gr.update(visible=choice == "Search Database"),
                next_page_btn: gr.update(visible=choice == "Search Database"),
                page_info: gr.update(visible=choice == "Search Database"),
                search_query: gr.update(visible=choice == "Search Database"),
                search_button: gr.update(visible=choice == "Search Database"),
                search_results: gr.update(visible=choice == "Search Database"),
                file_upload: gr.update(visible=choice == "Upload File"),
                convert_to_text: gr.update(visible=choice == "Upload File"),
                keywords: gr.update(visible=choice == "Upload File")
            }

        context_source.change(update_context_source, context_source,
                              [existing_file, prev_page_btn, next_page_btn, page_info, search_query, search_button,
                               search_results, file_upload, convert_to_text, keywords])

        next_page_btn.click(next_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])
        prev_page_btn.click(prev_page_fn, inputs=[file_page], outputs=[existing_file, page_info, file_page])

        # Initialize the file list when context source is changed to "Existing File"
        context_source.change(lambda choice: update_file_list(1) if choice == "Existing File" else (gr.update(), gr.update(), 1),
                              inputs=[context_source], outputs=[existing_file, page_info, file_page])

        def perform_search(query, selected_databases, keywords):
            try:
                results = []

                # Iterate over selected database types and perform searches accordingly
                for database_type in selected_databases:
                    if database_type == "Media DB":
                        # FIXME - check for existence of keywords before setting as search field
                        search_fields = ["title", "content", "keywords"]
                        results += search_media_db(query, search_fields, keywords, page=1, results_per_page=25)
                    elif database_type == "RAG Chat":
                        results += search_rag_chat(query)
                    elif database_type == "RAG Notes":
                        results += search_rag_notes(query)
                    elif database_type == "Character Chat":
                        results += search_character_chat(query)
                    elif database_type == "Character Cards":
                        results += search_character_cards(query)

                # Remove duplicate results if necessary
                results = list(set(results))
                return gr.update(choices=results)
            except Exception as e:
                gr.Error(f"Error performing search: {str(e)}")
                return gr.update(choices=[])

        # Click Event for the DB Search Button
        search_button.click(
            perform_search,
            inputs=[search_query, db_choice, keywords_input],
            outputs=[search_results]
        )

        def rephrase_question(history, latest_question, api_choice):
            logging.info("RAG QnA: Rephrasing question")
            conversation_history = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history[:-1]])
            prompt = f"""You are a helpful assistant. Given the conversation history and the latest question, resolve any ambiguous references in the latest question.

Conversation History:
{conversation_history}

Latest Question:
{latest_question}

Rewritten Question:"""

            # Use the selected API to generate the rephrased question
            rephrased_question = generate_answer(api_choice, prompt, "")
            logging.info(f"Rephrased question: {rephrased_question}")
            return rephrased_question.strip()

        # FIXME - RAG DB selection
        def rag_qa_chat_wrapper(
                message, history, context_source, existing_file, search_results, file_upload,
                convert_to_text, keywords, api_choice, use_query_rewriting, state_value,
                keywords_input, top_k_input, use_re_ranking, db_choices, auto_save_enabled
        ):
            try:
                logging.info(f"Starting rag_qa_chat_wrapper with message: {message}")
                logging.info(f"Context source: {context_source}")
                logging.info(f"API choice: {api_choice}")
                logging.info(f"Query rewriting: {'enabled' if use_query_rewriting else 'disabled'}")
                logging.info(f"Selected DB Choices: {db_choices}")

                # Show loading indicator
                yield history, "", gr.update(visible=True), state_value, gr.update(visible=False), gr.update(
                    visible=False)

                conversation_id = state_value.get("conversation_id")
                conversation_messages = state_value.get("conversation_messages", [])

                # Save the user's message
                if conversation_id:
                    save_message(conversation_id, "user", message)
                else:
                    # Append to in-memory messages
                    conversation_messages.append(("user", message))
                    state_value["conversation_messages"] = conversation_messages

                # Ensure api_choice is a string
                api_choice_str = api_choice.value if isinstance(api_choice, gr.components.Dropdown) else api_choice
                logging.info(f"Resolved API choice: {api_choice_str}")

                # Only rephrase the question if it's not the first query and query rewriting is enabled
                if len(history) > 0 and use_query_rewriting:
                    rephrased_question = rephrase_question(history, message, api_choice_str)
                    logging.info(f"Original question: {message}")
                    logging.info(f"Rephrased question: {rephrased_question}")
                else:
                    rephrased_question = message
                    logging.info(f"Using original question: {message}")

                if context_source == "All Files in the Database":
                    # Use the enhanced_rag_pipeline to search the selected databases
                    context = enhanced_rag_pipeline(
                        rephrased_question, api_choice_str, keywords_input, top_k_input, use_re_ranking,
                        database_types=db_choices  # Pass the list of selected databases
                    )
                    logging.info(f"Using enhanced_rag_pipeline for database search")
                elif context_source == "Search Database":
                    context = f"media_id:{search_results.split('(ID: ')[1][:-1]}"
                    logging.info(f"Using search result with context: {context}")
                else:
                    # Upload File
                    logging.info("Processing uploaded file")
                    if file_upload is None:
                        raise ValueError("No file uploaded")
                    # Process the uploaded file
                    file_path = file_upload.name
                    file_name = os.path.basename(file_path)
                    logging.info(f"Uploaded file: {file_name}")

                    if convert_to_text:
                        logging.info("Converting file to plain text")
                        content = convert_file_to_text(file_path)
                    else:
                        logging.info("Reading file content")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    logging.info(f"File content length: {len(content)} characters")

                    # Process keywords
                    if not keywords:
                        keywords = "default,rag-file-upload"
                    logging.info(f"Keywords: {keywords}")

                    # Add the content to the database and get the media_id
                    logging.info("Adding content to database")
                    result = add_media_with_keywords(
                        url=file_name,
                        title=file_name,
                        media_type='document',
                        content=content,
                        keywords=keywords,
                        prompt='No prompt for uploaded files',
                        summary='No summary for uploaded files',
                        transcription_model='None',
                        author='Unknown',
                        ingestion_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    logging.info(f"Result from add_media_with_keywords: {result}")
                    if isinstance(result, tuple):
                        media_id, _ = result
                    else:
                        media_id = result
                    context = f"media_id:{media_id}"
                    logging.info(f"Context for uploaded file: {context}")

                logging.info("Calling rag_qa_chat function")
                new_history, response = rag_qa_chat(rephrased_question, history, context, api_choice_str)

                # Log first 100 chars of response
                logging.info(f"Response received from rag_qa_chat: {response[:100]}...")

                # Save assistant's response
                if conversation_id:
                    save_message(conversation_id, "assistant", response)
                else:
                    conversation_messages.append(("assistant", response))
                    state_value["conversation_messages"] = conversation_messages

                # Update the state
                updated_state = auto_save_conversation(message, response, state_value, auto_save_enabled)
                updated_state["conversation_messages"] = conversation_messages

                # Safely update history
                if new_history:
                    new_history[-1] = (message, response)
                else:
                    new_history = [(message, response)]

                # Get the current rating and update display
                conversation_id = updated_state.get("conversation_id")
                if conversation_id:
                    rating = get_conversation_rating(conversation_id)
                    if rating is not None:
                        rating_display_update = gr.update(value=f"**Current Rating:** {rating} star(s)", visible=True)
                        rating_input_update = gr.update(value=str(rating), visible=True)
                    else:
                        rating_display_update = gr.update(value="**Current Rating:** Not Rated", visible=True)
                        rating_input_update = gr.update(value=None, visible=True)
                else:
                    rating_display_update = gr.update(value="", visible=False)
                    rating_input_update = gr.update(value=None, visible=False)

                gr.Info("Response generated successfully")
                logging.info("rag_qa_chat_wrapper completed successfully")
                yield new_history, "", gr.update(
                    visible=False), updated_state, rating_display_update, rating_input_update

            except ValueError as e:
                logging.error(f"Input error in rag_qa_chat_wrapper: {str(e)}")
                gr.Error(f"Input error: {str(e)}")
                yield history, "", gr.update(visible=False), state_value, gr.update(visible=False), gr.update(
                    visible=False)
            except DatabaseError as e:
                logging.error(f"Database error in rag_qa_chat_wrapper: {str(e)}")
                gr.Error(f"Database error: {str(e)}")
                yield history, "", gr.update(visible=False), state_value, gr.update(visible=False), gr.update(
                    visible=False)
            except Exception as e:
                logging.error(f"Unexpected error in rag_qa_chat_wrapper: {e}", exc_info=True)
                gr.Error("An unexpected error occurred. Please try again later.")
                yield history, "", gr.update(visible=False), state_value, gr.update(visible=False), gr.update(
                    visible=False)

        def clear_chat_history():
            return [], "", gr.update(value="", visible=False), gr.update(value=None, visible=False)

        submit.click(
            rag_qa_chat_wrapper,
            inputs=[
                msg,
                chatbot,
                context_source,
                existing_file,
                search_results,
                file_upload,
                convert_to_text,
                keywords,
                api_choice,
                use_query_rewriting,
                state,
                keywords_input,
                top_k_input,
                use_re_ranking,
                db_choice,
                auto_save_checkbox
            ],
            outputs=[chatbot, msg, loading_indicator, state, rating_display, rating_input],
        )

        clear_chat.click(
            clear_chat_history,
            outputs=[chatbot, msg, rating_display, rating_input]
        )

        # TTS Generation and Playback
        def speak_last_response(chatbot):
            """Generate audio for the last response and return the audio file"""
            logging.debug("Starting speak_last_response")
            try:
                if not chatbot or len(chatbot) == 0:
                    return "No messages to speak", None

                last_message = chatbot[-1][1]
                logging.debug(f"Last message to speak: {last_message}")

                # Generate unique filename
                timestamp = int(time.time())
                output_file = f"response_{timestamp}.mp3"

                # Generate audio file
                audio_file = generate_audio(
                    api_key=None,
                    text=last_message,
                    provider="openai",
                    output_file=output_file
                )

                if audio_file and os.path.exists(audio_file):
                    return "Audio ready", audio_file
                return "Audio generation failed", None

            except Exception as e:
                logging.error(f"Error in speak_last_response: {str(e)}")
                return f"Error: {str(e)}", None

        speak_button.click(
            fn=speak_last_response,
            inputs=[chatbot],
            outputs=[tts_status, audio_output],
            api_name="speak_response"
        ).then(
            lambda: gr.update(visible=True),
            outputs=audio_output
        )

    return (
        context_source,
        existing_file,
        search_query,
        search_button,
        search_results,
        file_upload,
        convert_to_text,
        keywords,
        api_choice,
        use_query_rewriting,
        chatbot,
        msg,
        submit,
        clear_chat,
    )


def create_rag_qa_notes_management_tab():
    # New Management Tab
    with gr.TabItem("Notes Management", visible=True):
        gr.Markdown("# RAG QA Notes Management")
        management_state = gr.State({
            "selected_conversation_id": None,
            "selected_note_id": None,
        })

        with gr.Row():
            with gr.Column(scale=1):
                # Search Notes
                search_notes_title = gr.Textbox(label="Search Notes by Title")
                search_notes_by_keyword = gr.Textbox(label="Search Notes by Keywords")
                search_notes_button = gr.Button("Search Notes")
                notes_list = gr.Dropdown(label="Notes", choices=[])

                # Manage Notes
                load_note_button = gr.Button("Load Note")
                delete_note_button = gr.Button("Delete Note")
                note_title_input = gr.Textbox(label="Note Title")
                note_content_input = gr.TextArea(label="Note Content", lines=20)
                note_keywords_input = gr.Textbox(label="Note Keywords (comma-separated)", value="default_note_keyword")
                save_note_button = gr.Button("Save Note")
                create_new_note_button = gr.Button("Create New Note")
                status_message = gr.HTML()

        # Function Definitions
        def search_notes(search_notes_title, keywords):
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',')]
                notes_data, total_pages, total_count = get_notes_by_keywords(keywords_list)
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"Found {total_count} notes")
            elif search_notes_title:
                notes_data, total_pages, total_count = search_notes_titles(search_notes_title)
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"Found {total_count} notes")
            else:
                # This will now return all notes, ordered by timestamp
                notes_data, total_pages, total_count = search_notes_titles("")
                choices = [f"Note {note_id} - {title} ({timestamp})" for
                           note_id, title, content, timestamp, conversation_id in notes_data]
                return gr.update(choices=choices, label=f"All notes ({total_count} total)")

        search_notes_button.click(
            search_notes,
            inputs=[search_notes_title, search_notes_by_keyword],
            outputs=[notes_list]
        )

        def load_selected_note(selected_note, state_value):
            if selected_note:
                note_id = int(selected_note.split('(ID: ')[1][:-1])
                note_data = get_note_by_id(note_id)
                if note_data:
                    note_id, title, content = note_data[0]
                    state_value["selected_note_id"] = note_id
                    # Get keywords for the note
                    keywords = get_keywords_for_note(note_id)
                    keywords_str = ', '.join(keywords)
                    return (
                        gr.update(value=title),
                        gr.update(value=content),
                        gr.update(value=keywords_str),
                        state_value
                    )
            return gr.update(value=''), gr.update(value=''), gr.update(value=''), state_value

        load_note_button.click(
            load_selected_note,
            inputs=[notes_list, management_state],
            outputs=[note_title_input, note_content_input, note_keywords_input, management_state]
        )

        def save_note_function(title, content, keywords_str, state_value):
            note_id = state_value["selected_note_id"]
            if note_id:
                update_note(note_id, title, content)
                if keywords_str:
                    # Clear existing keywords and add new ones
                    clear_keywords_from_note(note_id)
                    keywords_list = [kw.strip() for kw in keywords_str.split(',')]
                    add_keywords_to_note(note_id, keywords_list)
                return gr.Info("Note updated successfully.")
            else:
                # Create new note
                conversation_id = state_value.get("selected_conversation_id")
                if conversation_id:
                    note_id = save_notes(conversation_id, title, content)
                    state_value["selected_note_id"] = note_id
                    if keywords_str:
                        keywords_list = [kw.strip() for kw in keywords_str.split(',')]
                        add_keywords_to_note(note_id, keywords_list)
                    return gr.Info("New note created successfully.")
                else:
                    return gr.Error("No conversation selected. Cannot create a new note.")

        save_note_button.click(
            save_note_function,
            inputs=[note_title_input, note_content_input, note_keywords_input, management_state],
            outputs=[]
        )

        def delete_selected_note(state_value):
            note_id = state_value["selected_note_id"]
            if note_id:
                delete_note(note_id)
                # Reset state
                state_value["selected_note_id"] = None
                # Update notes list
                updated_notes = search_notes("", "")
                return updated_notes, gr.update(value="Note deleted successfully."), state_value
            else:
                return gr.update(), gr.update(value="No note selected."), state_value

        delete_note_button.click(
            delete_selected_note,
            inputs=[management_state],
            outputs=[notes_list, status_message, management_state]
        )

        def create_new_note_function(state_value):
            state_value["selected_note_id"] = None
            return gr.update(value=''), gr.update(value=''), gr.update(value=''), state_value

        create_new_note_button.click(
            create_new_note_function,
            inputs=[management_state],
            outputs=[note_title_input, note_content_input, note_keywords_input, management_state]
        )


def create_rag_qa_chat_management_tab():
    # New Management Tab
    with gr.TabItem("Chat Management", visible=True):
        gr.Markdown("# RAG QA Chat Conversation Management")

        management_state = gr.State({
            "selected_conversation_id": None,
            "selected_note_id": None,
        })

        # State to store the mapping between titles and IDs
        conversation_mapping = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                # Search Conversations
                with gr.Group():
                    gr.Markdown("## Search Conversations")
                    title_search = gr.Textbox(
                        label="Search by Title",
                        placeholder="Enter title to search..."
                    )
                    content_search = gr.Textbox(
                        label="Search in Chat Content",
                        placeholder="Enter text to search in messages..."
                    )
                    keyword_search = gr.Textbox(
                        label="Filter by Keywords (comma-separated)",
                        placeholder="keyword1, keyword2, ..."
                    )
                search_conversations_button = gr.Button("Search Conversations")
                conversations_list = gr.Dropdown(label="Conversations", choices=[])
                new_conversation_button = gr.Button("New Conversation")

                # Manage Conversations
                load_conversation_button = gr.Button("Load Conversation")
                delete_conversation_button = gr.Button("Delete Conversation")
                conversation_title_input = gr.Textbox(label="Conversation Title")
                conversation_content_input = gr.TextArea(label="Conversation Content", lines=20)
                save_conversation_button = gr.Button("Save Conversation")
                status_message = gr.HTML()

        # Function Definitions
        def search_conversations(title_query, content_query, keywords):
            try:
                # Parse keywords if provided
                keywords_list = None
                if keywords and keywords.strip():
                    keywords_list = [kw.strip() for kw in keywords.split(',')]

                # Search using existing search_conversations_by_keywords function with all criteria
                results, total_pages, total_count = search_conversations_by_keywords(
                    keywords=keywords_list,
                    title_query=title_query if title_query.strip() else None,
                    content_query=content_query if content_query.strip() else None
                )

                # Build choices as list of titles (ensure uniqueness)
                choices = []
                mapping = {}
                for conv in results:
                    conversation_id = conv['conversation_id']
                    title = conv['title']
                    display_title = f"{title} (ID: {conversation_id[:8]})"
                    choices.append(display_title)
                    mapping[display_title] = conversation_id

                return gr.update(choices=choices), mapping

            except Exception as e:
                logging.error(f"Error in search_conversations: {str(e)}")
                return gr.update(choices=[]), {}

        # Update the search button click event
        search_conversations_button.click(
            search_conversations,
            inputs=[title_search, content_search, keyword_search],
            outputs=[conversations_list, conversation_mapping]
        )

        def load_selected_conversation(selected_title, state_value, mapping):
            conversation_id = mapping.get(selected_title)
            if conversation_id:
                # Load conversation title
                conversation_title = get_conversation_title(conversation_id)
                # Load conversation messages
                messages, total_pages, total_count = load_chat_history(conversation_id)
                # Concatenate messages into a single string
                conversation_content = ""
                for role, content in messages:
                    conversation_content += f"{role}: {content}\n\n"
                # Update state
                new_state = state_value.copy()
                new_state["selected_conversation_id"] = conversation_id
                return (
                    gr.update(value=conversation_title),
                    gr.update(value=conversation_content.strip()),
                    new_state
                )
            return gr.update(value=''), gr.update(value=''), state_value

        load_conversation_button.click(
            load_selected_conversation,
            inputs=[conversations_list, management_state, conversation_mapping],
            outputs=[conversation_title_input, conversation_content_input, management_state]
        )

        def save_conversation(title, content, state_value):
            conversation_id = state_value["selected_conversation_id"]
            if conversation_id:
                # Update conversation title
                update_conversation_title(conversation_id, title)

                # Clear existing messages
                delete_messages_in_conversation(conversation_id)

                # Parse the content back into messages
                messages = []
                for line in content.strip().split('\n\n'):
                    if ': ' in line:
                        role, message_content = line.split(': ', 1)
                        messages.append((role.strip(), message_content.strip()))
                    else:
                        # If the format is incorrect, skip or handle accordingly
                        continue

                # Save new messages
                for role, message_content in messages:
                    save_message(conversation_id, role, message_content)

                return (
                    gr.HTML("<p style='color: green;'>Conversation updated successfully.</p>"),
                    gr.update(value=title),
                    gr.update(value=content),
                    state_value
                )
            else:
                return (
                    gr.HTML("<p style='color: red;'>No conversation selected to save.</p>"),
                    gr.update(value=title),
                    gr.update(value=content),
                    state_value
                )

        save_conversation_button.click(
            save_conversation,
            inputs=[conversation_title_input, conversation_content_input, management_state],
            outputs=[status_message, conversation_title_input, conversation_content_input, management_state]
        )

        def delete_selected_conversation(state_value, mapping):
            conversation_id = state_value["selected_conversation_id"]
            if conversation_id:
                delete_conversation(conversation_id)
                # Reset state
                new_state = state_value.copy()
                new_state["selected_conversation_id"] = None
                # Update conversations list and mapping
                conversations, _, _ = get_all_conversations()
                choices = []
                new_mapping = {}
                for conv_id, title in conversations:
                    display_title = f"{title} (ID: {conv_id[:8]})"
                    choices.append(display_title)
                    new_mapping[display_title] = conv_id
                return (
                    gr.update(choices=choices, value=None),
                    gr.HTML("<p style='color: green;'>Conversation deleted successfully.</p>"),
                    new_state,
                    gr.update(value=''),
                    gr.update(value=''),
                    new_mapping
                )
            else:
                return (
                    gr.update(),
                    gr.HTML("<p style='color: red;'>No conversation selected.</p>"),
                    state_value,
                    gr.update(),
                    gr.update(),
                    mapping
                )

        delete_conversation_button.click(
            delete_selected_conversation,
            inputs=[management_state, conversation_mapping],
            outputs=[
                conversations_list,
                status_message,
                management_state,
                conversation_title_input,
                conversation_content_input,
                conversation_mapping
            ]
        )

        def create_new_conversation(state_value, mapping):
            conversation_id = start_new_conversation()
            # Update state
            new_state = state_value.copy()
            new_state["selected_conversation_id"] = conversation_id
            # Update conversations list and mapping
            conversations, _, _ = get_all_conversations()
            choices = []
            new_mapping = {}
            for conv_id, title in conversations:
                display_title = f"{title} (ID: {conv_id[:8]})"
                choices.append(display_title)
                new_mapping[display_title] = conv_id
            # Set the new conversation as selected
            selected_title = f"Untitled Conversation (ID: {conversation_id[:8]})"
            return (
                gr.update(choices=choices, value=selected_title),
                gr.update(value='Untitled Conversation'),
                gr.update(value=''),
                gr.HTML("<p style='color: green;'>New conversation created.</p>"),
                new_state,
                new_mapping
            )

        new_conversation_button.click(
            create_new_conversation,
            inputs=[management_state, conversation_mapping],
            outputs=[
                conversations_list,
                conversation_title_input,
                conversation_content_input,
                status_message,
                management_state,
                conversation_mapping
            ]
        )

        def delete_messages_in_conversation_wrapper(conversation_id):
            """Wrapper function to delete all messages in a conversation."""
            try:
                delete_messages_in_conversation(conversation_id)
                logging.info(f"Messages in conversation '{conversation_id}' deleted successfully.")
            except Exception as e:
                logging.error(f"Error deleting messages in conversation '{conversation_id}': {e}")
                raise

        def get_conversation_title_wrapper(conversation_id):
            """Helper function to get the conversation title."""
            result = get_conversation_title(conversation_id)
            if result:
                return result[0][0]
            else:
                return "Untitled Conversation"



def create_export_data_tab():
    with gr.TabItem("Export Data"):
        gr.Markdown("# Export Data")

        export_option = gr.Radio(
            ["Export All", "Export Selected"],
            label="Export Option",
            value="Export All"
        )

        conversations_checklist = gr.CheckboxGroup(
            choices=[],
            label="Select Conversations",
            visible=False
        )

        notes_checklist = gr.CheckboxGroup(
            choices=[],
            label="Select Notes",
            visible=False
        )

        export_button = gr.Button("Export")
        download_link = gr.File(label="Download Exported Data", visible=False)
        status_message = gr.HTML()

        # Function to update visibility and populate checklists
        def update_visibility(export_option_value):
            if export_option_value == "Export Selected":
                # Fetch conversations and notes to populate the checklists
                conversations = fetch_all_conversations()
                notes = fetch_all_notes()

                conversation_choices = [f"{title} (ID: {conversation_id})" for conversation_id, title, _ in conversations]
                note_choices = [f"{title} (ID: {note_id})" for note_id, title, _ in notes]

                return (
                    gr.update(visible=True, choices=conversation_choices),
                    gr.update(visible=True, choices=note_choices)
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        export_option.change(
            update_visibility,
            inputs=[export_option],
            outputs=[conversations_checklist, notes_checklist]
        )

        import zipfile
        import io
        def update_visibility(export_option_value):
            if export_option_value == "Export Selected":
                # Fetch conversations and notes to populate the checklists
                conversations = fetch_all_conversations()
                notes = fetch_all_notes()

                conversation_choices = [f"{title} (ID: {conversation_id})" for conversation_id, title, _ in
                                        conversations]
                note_choices = [f"{title} (ID: {note_id})" for note_id, title, _ in notes]

                return (
                    gr.update(visible=True, choices=conversation_choices),
                    gr.update(visible=True, choices=note_choices)
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        export_option.change(
            update_visibility,
            inputs=[export_option],
            outputs=[conversations_checklist, notes_checklist]
        )

        def export_data_function(export_option, selected_conversations, selected_notes):
            try:
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    if export_option == "Export All":
                        # Fetch all conversations and notes
                        conversations = fetch_all_conversations()
                        notes = fetch_all_notes()
                    else:
                        # Fetch selected conversations and notes
                        conversation_ids = [int(item.split(' (ID: ')[1][:-1]) for item in selected_conversations]
                        note_ids = [int(item.split(' (ID: ')[1][:-1]) for item in selected_notes]
                        conversations = fetch_conversations_by_ids(conversation_ids)
                        notes = fetch_notes_by_ids(note_ids)

                    # Export conversations
                    for conversation in conversations:
                        conversation_id, title, _ = conversation
                        filename = f"conversation_{conversation_id}_{title.replace(' ', '_')}.md"
                        zip_file.writestr(filename, conversation)

                    # Export notes
                    for note in notes:
                        note_id, title, _ = note
                        filename = f"note_{note_id}_{title.replace(' ', '_')}.md"
                        zip_file.writestr(filename, note)

                zip_buffer.seek(0)
                return zip_buffer, gr.update(visible=True), gr.update(
                    value="<p style='color:green;'>Export successful!</p>")
            except Exception as e:
                logging.error(f"Error exporting data: {str(e)}")
                return None, gr.update(visible=False), gr.update(value=f"<p style='color:red;'>Error: {str(e)}</p>")

        export_button.click(
            export_data_function,
            inputs=[export_option, conversations_checklist, notes_checklist],
            outputs=[download_link, download_link, status_message]
        )


def convert_file_to_text(file_path):
    """Convert various file types to plain text."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_and_format_from_pdf(file_path)
    elif file_extension == '.epub':
        return read_epub(file_path)
    elif file_extension in ['.json', '.csv']:
        return read_structured_file(file_path)
    elif file_extension == '.docx':
        return docx2txt.process(file_path)
    elif file_extension in ['.txt', '.md', '.rtf']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def read_structured_file(file_path):
    """Read and convert JSON or CSV files to text."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.json':
        with open(file_path, 'r') as file:
            data = json.load(file)
        return json.dumps(data, indent=2)

    elif file_extension == '.csv':
        with open(file_path, 'r', newline='') as file:
            csv_reader = csv.reader(file)
            return '\n'.join([','.join(row) for row in csv_reader])

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

#
# End of RAG_QA_Chat_tab.py
########################################################################################################################
