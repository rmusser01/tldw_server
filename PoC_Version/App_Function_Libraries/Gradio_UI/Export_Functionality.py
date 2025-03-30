# Export_Functionality.py
# Functionality for exporting items as markdown files
#
# Imports
import os
import json
import math
import shutil
import tempfile
from typing import List, Dict, Optional, Tuple, Any
#
# 3rd-Party Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import DatabaseError, fetch_all_notes, fetch_all_conversations, \
    get_keywords_for_note, fetch_notes_by_ids, fetch_conversations_by_ids
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import get_keywords_for_conversation
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import fetch_item_details, fetch_items_by_keyword, browse_items
from PoC_Version.App_Function_Libraries.Utils.Utils import logger, logging


#
#######################################################################################################################
#
# Functions:

def export_item_as_markdown(media_id: int) -> Tuple[Optional[str], str]:
    try:
        content, prompt, summary = fetch_item_details(media_id)
        title = f"Item {media_id}"  # You might want to fetch the actual title
        markdown_content = f"# {title}\n\n## Prompt\n{prompt}\n\n## Summary\n{summary}\n\n## Content\n{content}"

        filename = f"export_item_{media_id}.md"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Successfully exported item {media_id} to {filename}")
        return filename, f"Successfully exported item {media_id} to {filename}"
    except Exception as e:
        error_message = f"Error exporting item {media_id}: {str(e)}"
        logger.error(error_message)
        return None, error_message


def export_items_by_keyword(keyword: str) -> str:
    try:
        items = fetch_items_by_keyword(keyword)
        if not items:
            logger.warning(f"No items found for keyword: {keyword}")
            return f"No items found for keyword: {keyword}"

        # Create a temporary directory to store individual markdown files
        with tempfile.TemporaryDirectory() as temp_dir:
            folder_name = f"export_keyword_{keyword}"
            export_folder = os.path.join(temp_dir, folder_name)
            os.makedirs(export_folder)

            for item in items:
                content, prompt, summary = fetch_item_details(item['id'])
                markdown_content = f"# {item['title']}\n\n## Prompt\n{prompt}\n\n## Summary\n{summary}\n\n## Content\n{content}"

                # Create individual markdown file for each item
                file_name = f"{item['id']}_{item['title'][:50]}.md"  # Limit filename length
                file_path = os.path.join(export_folder, file_name)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(markdown_content)

            # Create a zip file containing all markdown files
            zip_filename = f"{folder_name}.zip"
            shutil.make_archive(os.path.join(temp_dir, folder_name), 'zip', export_folder)

            # Move the zip file to a location accessible by Gradio
            final_zip_path = os.path.join(os.getcwd(), zip_filename)
            shutil.move(os.path.join(temp_dir, zip_filename), final_zip_path)

        logger.info(f"Successfully exported {len(items)} items for keyword '{keyword}' to {zip_filename}")
        return final_zip_path
    except Exception as e:
        logger.error(f"Error exporting items for keyword '{keyword}': {str(e)}")
        return f"Error exporting items for keyword '{keyword}': {str(e)}"


def export_selected_items(selected_items: List[Dict]) -> Tuple[Optional[str], str]:
    try:
        logger.debug(f"Received selected_items: {selected_items}")
        if not selected_items:
            logger.warning("No items selected for export")
            return None, "No items selected for export"

        markdown_content = "# Selected Items\n\n"
        for item in selected_items:
            logger.debug(f"Processing item: {item}")
            try:
                # Check if 'value' is a string (JSON) or already a dictionary
                if isinstance(item, str):
                    item_data = json.loads(item)
                elif isinstance(item, dict) and 'value' in item:
                    item_data = item['value'] if isinstance(item['value'], dict) else json.loads(item['value'])
                else:
                    item_data = item

                logger.debug(f"Item data after processing: {item_data}")

                if 'id' not in item_data:
                    logger.error(f"'id' not found in item data: {item_data}")
                    continue

                content, prompt, summary = fetch_item_details(item_data['id'])
                markdown_content += f"## {item_data.get('title', 'Item {}'.format(item_data['id']))}\n\n### Prompt\n{prompt}\n\n### Summary\n{summary}\n\n### Content\n{content}\n\n---\n\n"
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
                markdown_content += f"## Error\n\nUnable to process this item.\n\n---\n\n"

        filename = "export_selected_items.md"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Successfully exported {len(selected_items)} selected items to {filename}")
        return filename, f"Successfully exported {len(selected_items)} items to {filename}"
    except Exception as e:
        error_message = f"Error exporting selected items: {str(e)}"
        logger.error(error_message)
        return None, error_message


def display_search_results_export_tab(search_query: str, search_type: str, page: int = 1, items_per_page: int = 10):
    logger.info(f"Searching with query: '{search_query}', type: '{search_type}', page: {page}")
    try:
        results = browse_items(search_query, search_type)
        logger.info(f"browse_items returned {len(results)} results")

        if not results:
            return [], f"No results found for query: '{search_query}'", 1, 1

        total_pages = math.ceil(len(results) / items_per_page)
        start_index = (page - 1) * items_per_page
        end_index = start_index + items_per_page
        paginated_results = results[start_index:end_index]

        checkbox_data = [
            {
                "name": f"Name: {item[1]}\nURL: {item[2]}",
                "value": {"id": item[0], "title": item[1], "url": item[2]}
            }
            for item in paginated_results
        ]

        logger.info(f"Returning {len(checkbox_data)} items for checkbox (page {page} of {total_pages})")
        return checkbox_data, f"Found {len(results)} results (showing page {page} of {total_pages})", page, total_pages

    except DatabaseError as e:
        error_message = f"Error in display_search_results_export_tab: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1
    except Exception as e:
        error_message = f"Unexpected error in display_search_results_export_tab: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1

#
# End of Media DB Export functionality
################################################################


################################################################
#
# Functions for RAG Chat DB Export functionality


def export_rag_conversations_as_json(
    selected_conversations: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Optional[str], str]:
    """
    Export conversations to a JSON file.

    Args:
        selected_conversations: Optional list of conversation dictionaries

    Returns:
        Tuple of (filename or None, status message)
    """
    try:
        if selected_conversations:
            # Extract conversation IDs from selected items
            conversation_ids = []
            for item in selected_conversations:
                if isinstance(item, str):
                    item_data = json.loads(item)
                elif isinstance(item, dict) and 'value' in item:
                    item_data = item['value'] if isinstance(item['value'], dict) else json.loads(item['value'])
                else:
                    item_data = item
                conversation_ids.append(item_data['conversation_id'])

            conversations = fetch_conversations_by_ids(conversation_ids)
        else:
            conversations = fetch_all_conversations()

        export_data = []
        for conversation_id, title, messages in conversations:
            # Get keywords for the conversation
            keywords = get_keywords_for_conversation(conversation_id)

            conversation_data = {
                "conversation_id": conversation_id,
                "title": title,
                "keywords": keywords,
                "messages": [
                    {"role": role, "content": content}
                    for role, content in messages
                ]
            }
            export_data.append(conversation_data)

        filename = "rag_conversations_export.json"
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully exported {len(export_data)} conversations to {filename}")
        return filename, f"Successfully exported {len(export_data)} conversations to {filename}"
    except Exception as e:
        error_message = f"Error exporting conversations: {str(e)}"
        logger.error(error_message)
        return None, error_message


def export_rag_notes_as_json(
    selected_notes: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Optional[str], str]:
    """
    Export notes to a JSON file.

    Args:
        selected_notes: Optional list of note dictionaries

    Returns:
        Tuple of (filename or None, status message)
    """
    try:
        if selected_notes:
            # Extract note IDs from selected items
            note_ids = []
            for item in selected_notes:
                if isinstance(item, str):
                    item_data = json.loads(item)
                elif isinstance(item, dict) and 'value' in item:
                    item_data = item['value'] if isinstance(item['value'], dict) else json.loads(item['value'])
                else:
                    item_data = item
                note_ids.append(item_data['id'])

            notes = fetch_notes_by_ids(note_ids)
        else:
            notes = fetch_all_notes()

        export_data = []
        for note_id, title, content in notes:
            # Get keywords for the note
            keywords = get_keywords_for_note(note_id)

            note_data = {
                "note_id": note_id,
                "title": title,
                "content": content,
                "keywords": keywords
            }
            export_data.append(note_data)

        filename = "rag_notes_export.json"
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully exported {len(export_data)} notes to {filename}")
        return filename, f"Successfully exported {len(export_data)} notes to {filename}"
    except Exception as e:
        error_message = f"Error exporting notes: {str(e)}"
        logger.error(error_message)
        return None, error_message


def display_rag_conversations(search_query: str = "", page: int = 1, items_per_page: int = 10):
    """Display conversations for selection in the export tab."""
    try:
        conversations = fetch_all_conversations()

        if search_query:
            # Simple search implementation - can be enhanced based on needs
            conversations = [
                conv for conv in conversations
                if search_query.lower() in conv[1].lower()  # Search in title
            ]

        # Implement pagination
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paginated_conversations = conversations[start_idx:end_idx]
        total_pages = (len(conversations) + items_per_page - 1) // items_per_page

        # Format for checkbox group
        checkbox_data = [
            {
                "name": f"Title: {title}\nMessages: {len(messages)}",
                "value": {"conversation_id": conv_id, "title": title}
            }
            for conv_id, title, messages in paginated_conversations
        ]

        return (
            checkbox_data,
            f"Found {len(conversations)} conversations (showing page {page} of {total_pages})",
            page,
            total_pages
        )
    except Exception as e:
        error_message = f"Error displaying conversations: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1


def display_rag_notes(search_query: str = "", page: int = 1, items_per_page: int = 10):
    """Display notes for selection in the export tab."""
    try:
        notes = fetch_all_notes()

        if search_query:
            # Simple search implementation - can be enhanced based on needs
            notes = [
                note for note in notes
                if search_query.lower() in note[1].lower()  # Search in title
                   or search_query.lower() in note[2].lower()  # Search in content
            ]

        # Implement pagination
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paginated_notes = notes[start_idx:end_idx]
        total_pages = (len(notes) + items_per_page - 1) // items_per_page

        # Format for checkbox group
        checkbox_data = [
            {
                "name": f"Title: {title}\nContent preview: {content[:100]}...",
                "value": {"id": note_id, "title": title}
            }
            for note_id, title, content in paginated_notes
        ]

        return (
            checkbox_data,
            f"Found {len(notes)} notes (showing page {page} of {total_pages})",
            page,
            total_pages
        )
    except Exception as e:
        error_message = f"Error displaying notes: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1


def create_rag_export_tab():
    """Create the RAG QA Chat export tab interface."""
    with gr.Tab("RAG QA Chat Export"):
        with gr.Tabs():
            # Conversations Export Tab
            with gr.Tab("Export Conversations"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Export RAG QA Chat Conversations")
                        conversation_search = gr.Textbox(label="Search Conversations")
                        conversation_search_button = gr.Button("Search")

                    with gr.Column():
                        conversation_prev_button = gr.Button("Previous Page")
                        conversation_next_button = gr.Button("Next Page")

                conversation_current_page = gr.State(1)
                conversation_total_pages = gr.State(1)

                conversation_results = gr.CheckboxGroup(label="Select Conversations to Export")
                export_selected_conversations_button = gr.Button("Export Selected Conversations")
                export_all_conversations_button = gr.Button("Export All Conversations")

                conversation_export_output = gr.File(label="Download Exported Conversations")
                conversation_status = gr.Textbox(label="Status", interactive=False)

            # Notes Export Tab
            with gr.Tab("Export Notes"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Export RAG QA Chat Notes")
                        notes_search = gr.Textbox(label="Search Notes")
                        notes_search_button = gr.Button("Search")

                    with gr.Column():
                        notes_prev_button = gr.Button("Previous Page")
                        notes_next_button = gr.Button("Next Page")

                notes_current_page = gr.State(1)
                notes_total_pages = gr.State(1)

                notes_results = gr.CheckboxGroup(label="Select Notes to Export")
                export_selected_notes_button = gr.Button("Export Selected Notes")
                export_all_notes_button = gr.Button("Export All Notes")

                notes_export_output = gr.File(label="Download Exported Notes")
                notes_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers for conversations
        def search_conversations(query, page):
            return display_rag_conversations(query, page)

        conversation_search_button.click(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        def update_conversation_page(current, total, direction):
            new_page = max(1, min(total, current + direction))
            return new_page

        conversation_prev_button.click(
            fn=update_conversation_page,
            inputs=[conversation_current_page, conversation_total_pages, gr.State(-1)],
            outputs=[conversation_current_page]
        ).then(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        conversation_next_button.click(
            fn=update_conversation_page,
            inputs=[conversation_current_page, conversation_total_pages, gr.State(1)],
            outputs=[conversation_current_page]
        ).then(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        export_selected_conversations_button.click(
            fn=export_rag_conversations_as_json,
            inputs=[conversation_results],
            outputs=[conversation_export_output, conversation_status]
        )

        export_all_conversations_button.click(
            fn=lambda: export_rag_conversations_as_json(),
            outputs=[conversation_export_output, conversation_status]
        )

        # Event handlers for notes
        def search_notes(query, page):
            return display_rag_notes(query, page)

        notes_search_button.click(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        def update_notes_page(current, total, direction):
            new_page = max(1, min(total, current + direction))
            return new_page

        notes_prev_button.click(
            fn=update_notes_page,
            inputs=[notes_current_page, notes_total_pages, gr.State(-1)],
            outputs=[notes_current_page]
        ).then(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        notes_next_button.click(
            fn=update_notes_page,
            inputs=[notes_current_page, notes_total_pages, gr.State(1)],
            outputs=[notes_current_page]
        ).then(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        export_selected_notes_button.click(
            fn=export_rag_notes_as_json,
            inputs=[notes_results],
            outputs=[notes_export_output, notes_status]
        )

        export_all_notes_button.click(
            fn=lambda: export_rag_notes_as_json(),
            outputs=[notes_export_output, notes_status]
        )

#
# End of RAG Chat DB Export functionality
#####################################################

def create_export_tabs():
    """Create the unified export interface with all export tabs."""
    with gr.Tabs():
        # Media DB Export Tab
        with gr.Tab("Media DB Export"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("# Search and Export Items")
                    gr.Markdown("Search for items and export them as markdown files")
                    gr.Markdown("You can also export items by keyword")
                    search_query = gr.Textbox(label="Search Query")
                    search_type = gr.Radio(["Title", "URL", "Keyword", "Content"], label="Search By")
                    search_button = gr.Button("Search")

                with gr.Column():
                    prev_button = gr.Button("Previous Page")
                    next_button = gr.Button("Next Page")

            current_page = gr.State(1)
            total_pages = gr.State(1)

            search_results = gr.CheckboxGroup(label="Search Results", choices=[])
            export_selected_button = gr.Button("Export Selected Items")

            keyword_input = gr.Textbox(label="Enter keyword for export")
            export_by_keyword_button = gr.Button("Export items by keyword")

            export_output = gr.File(label="Download Exported File")
            error_output = gr.Textbox(label="Status/Error Messages", interactive=False)

        # Conversations Export Tab
        with gr.Tab("RAG Conversations Export"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Export RAG QA Chat Conversations")
                    conversation_search = gr.Textbox(label="Search Conversations")
                    conversation_search_button = gr.Button("Search")

                with gr.Column():
                    conversation_prev_button = gr.Button("Previous Page")
                    conversation_next_button = gr.Button("Next Page")

            conversation_current_page = gr.State(1)
            conversation_total_pages = gr.State(1)

            conversation_results = gr.CheckboxGroup(label="Select Conversations to Export")
            export_selected_conversations_button = gr.Button("Export Selected Conversations")
            export_all_conversations_button = gr.Button("Export All Conversations")

            conversation_export_output = gr.File(label="Download Exported Conversations")
            conversation_status = gr.Textbox(label="Status", interactive=False)

        # Notes Export Tab
        with gr.Tab("RAG Notes Export"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Export RAG QA Chat Notes")
                    notes_search = gr.Textbox(label="Search Notes")
                    notes_search_button = gr.Button("Search")

                with gr.Column():
                    notes_prev_button = gr.Button("Previous Page")
                    notes_next_button = gr.Button("Next Page")

            notes_current_page = gr.State(1)
            notes_total_pages = gr.State(1)

            notes_results = gr.CheckboxGroup(label="Select Notes to Export")
            export_selected_notes_button = gr.Button("Export Selected Notes")
            export_all_notes_button = gr.Button("Export All Notes")

            notes_export_output = gr.File(label="Download Exported Notes")
            notes_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers for media DB
        def search_and_update(query, search_type, page):
            results, message, current, total = display_search_results_export_tab(query, search_type, page)
            logger.debug(f"search_and_update results: {results}")
            return results, message, current, total, gr.update(choices=results)

        def update_page(current, total, direction):
            new_page = max(1, min(total, current + direction))
            return new_page

        def handle_export_selected(selected_items):
            logger.debug(f"Exporting selected items: {selected_items}")
            return export_selected_items(selected_items)

        def handle_item_selection(selected_items):
            logger.debug(f"Selected items: {selected_items}")
            if not selected_items:
                return None, "No item selected"

            try:
                selected_item = selected_items[0]
                logger.debug(f"First selected item: {selected_item}")

                if isinstance(selected_item['value'], str):
                    item_data = json.loads(selected_item['value'])
                else:
                    item_data = selected_item['value']

                logger.debug(f"Item data: {item_data}")
                item_id = item_data['id']
                return export_item_as_markdown(item_id)
            except Exception as e:
                error_message = f"Error processing selected item: {str(e)}"
                logger.error(error_message)
                return None, error_message

        search_button.click(
            fn=search_and_update,
            inputs=[search_query, search_type, current_page],
            outputs=[search_results, error_output, current_page, total_pages, search_results],
            show_progress="full"
        )

        prev_button.click(
            fn=update_page,
            inputs=[current_page, total_pages, gr.State(-1)],
            outputs=[current_page]
        ).then(
            fn=search_and_update,
            inputs=[search_query, search_type, current_page],
            outputs=[search_results, error_output, current_page, total_pages],
            show_progress=True
        )

        next_button.click(
            fn=update_page,
            inputs=[current_page, total_pages, gr.State(1)],
            outputs=[current_page]
        ).then(
            fn=search_and_update,
            inputs=[search_query, search_type, current_page],
            outputs=[search_results, error_output, current_page, total_pages],
            show_progress=True
        )

        export_selected_button.click(
            fn=handle_export_selected,
            inputs=[search_results],
            outputs=[export_output, error_output],
            show_progress="full"
        )

        export_by_keyword_button.click(
            fn=export_items_by_keyword,
            inputs=[keyword_input],
            outputs=[export_output, error_output],
            show_progress="full"
        )

        search_results.select(
            fn=handle_item_selection,
            inputs=[search_results],
            outputs=[export_output, error_output],
            show_progress="full"
        )

        # Event handlers for conversations
        def search_conversations(query, page):
            return display_rag_conversations(query, page)

        def update_conversation_page(current, total, direction):
            new_page = max(1, min(total, current + direction))
            return new_page

        conversation_search_button.click(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        conversation_prev_button.click(
            fn=update_conversation_page,
            inputs=[conversation_current_page, conversation_total_pages, gr.State(-1)],
            outputs=[conversation_current_page]
        ).then(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        conversation_next_button.click(
            fn=update_conversation_page,
            inputs=[conversation_current_page, conversation_total_pages, gr.State(1)],
            outputs=[conversation_current_page]
        ).then(
            fn=search_conversations,
            inputs=[conversation_search, conversation_current_page],
            outputs=[conversation_results, conversation_status, conversation_current_page, conversation_total_pages]
        )

        export_selected_conversations_button.click(
            fn=export_rag_conversations_as_json,
            inputs=[conversation_results],
            outputs=[conversation_export_output, conversation_status]
        )

        export_all_conversations_button.click(
            fn=lambda: export_rag_conversations_as_json(),
            outputs=[conversation_export_output, conversation_status]
        )

        # Event handlers for notes
        def search_notes(query, page):
            return display_rag_notes(query, page)

        def update_notes_page(current, total, direction):
            new_page = max(1, min(total, current + direction))
            return new_page

        notes_search_button.click(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        notes_prev_button.click(
            fn=update_notes_page,
            inputs=[notes_current_page, notes_total_pages, gr.State(-1)],
            outputs=[notes_current_page]
        ).then(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        notes_next_button.click(
            fn=update_notes_page,
            inputs=[notes_current_page, notes_total_pages, gr.State(1)],
            outputs=[notes_current_page]
        ).then(
            fn=search_notes,
            inputs=[notes_search, notes_current_page],
            outputs=[notes_results, notes_status, notes_current_page, notes_total_pages]
        )

        export_selected_notes_button.click(
            fn=export_rag_notes_as_json,
            inputs=[notes_results],
            outputs=[notes_export_output, notes_status]
        )

        export_all_notes_button.click(
            fn=lambda: export_rag_notes_as_json(),
            outputs=[notes_export_output, notes_status]
        )

        with gr.TabItem("Export Prompts", visible=True):
            gr.Markdown("# Export Prompts Database Content")

            with gr.Row():
                with gr.Column():
                    export_type = gr.Radio(
                        choices=["All Prompts", "Prompts by Keyword"],
                        label="Export Type",
                        value="All Prompts"
                    )

                    # Keyword selection for filtered export
                    with gr.Column(visible=False) as keyword_col:
                        keyword_input = gr.Textbox(
                            label="Enter Keywords (comma-separated)",
                            placeholder="Enter keywords to filter prompts..."
                        )

                    # Export format selection
                    export_format = gr.Radio(
                        choices=["CSV", "Markdown (ZIP)"],
                        label="Export Format",
                        value="CSV"
                    )

                    # Export options
                    include_options = gr.CheckboxGroup(
                        choices=[
                            "Include System Prompts",
                            "Include User Prompts",
                            "Include Details",
                            "Include Author",
                            "Include Keywords"
                        ],
                        label="Export Options",
                        value=["Include Keywords", "Include Author"]
                    )

                    # Markdown-specific options (only visible when Markdown is selected)
                    with gr.Column(visible=False) as markdown_options_col:
                        markdown_template = gr.Radio(
                            choices=[
                                "Basic Template",
                                "Detailed Template",
                                "Custom Template"
                            ],
                            label="Markdown Template",
                            value="Basic Template"
                        )
                        custom_template = gr.Textbox(
                            label="Custom Template",
                            placeholder="Use {title}, {author}, {details}, {system}, {user}, {keywords} as placeholders",
                            visible=False
                        )

                    export_button = gr.Button("Export Prompts")

                with gr.Column():
                    export_status = gr.Textbox(label="Export Status", interactive=False)
                    export_file = gr.File(label="Download Export")

            def update_ui_visibility(export_type, format_choice, template_choice):
                """Update UI elements visibility based on selections"""
                show_keywords = export_type == "Prompts by Keyword"
                show_markdown_options = format_choice == "Markdown (ZIP)"
                show_custom_template = template_choice == "Custom Template" and show_markdown_options

                return [
                    gr.update(visible=show_keywords),  # keyword_col
                    gr.update(visible=show_markdown_options),  # markdown_options_col
                    gr.update(visible=show_custom_template)  # custom_template
                ]

            def handle_export(export_type, keywords, export_format, options, markdown_template, custom_template):
                """Handle the export process based on selected options"""
                try:
                    # Parse options
                    include_system = "Include System Prompts" in options
                    include_user = "Include User Prompts" in options
                    include_details = "Include Details" in options
                    include_author = "Include Author" in options
                    include_keywords = "Include Keywords" in options

                    # Handle keyword filtering
                    keyword_list = None
                    if export_type == "Prompts by Keyword" and keywords:
                        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

                    # Get the appropriate template
                    template = None
                    if export_format == "Markdown (ZIP)":
                        if markdown_template == "Custom Template":
                            template = custom_template
                        else:
                            template = markdown_template

                    # Perform export
                    from PoC_Version.App_Function_Libraries.DB.Prompts_DB import export_prompts
                    status, file_path = export_prompts(
                        export_format=export_format.split()[0].lower(),  # 'csv' or 'markdown'
                        filter_keywords=keyword_list,
                        include_system=include_system,
                        include_user=include_user,
                        include_details=include_details,
                        include_author=include_author,
                        include_keywords=include_keywords,
                        markdown_template=template
                    )

                    return status, file_path

                except Exception as e:
                    error_msg = f"Export failed: {str(e)}"
                    logging.error(error_msg)
                    return error_msg, None

            # Event handlers
            export_type.change(
                fn=lambda t, f, m: update_ui_visibility(t, f, m),
                inputs=[export_type, export_format, markdown_template],
                outputs=[keyword_col, markdown_options_col, custom_template]
            )

            export_format.change(
                fn=lambda t, f, m: update_ui_visibility(t, f, m),
                inputs=[export_type, export_format, markdown_template],
                outputs=[keyword_col, markdown_options_col, custom_template]
            )

            markdown_template.change(
                fn=lambda t, f, m: update_ui_visibility(t, f, m),
                inputs=[export_type, export_format, markdown_template],
                outputs=[keyword_col, markdown_options_col, custom_template]
            )

            export_button.click(
                fn=handle_export,
                inputs=[
                    export_type,
                    keyword_input,
                    export_format,
                    include_options,
                    markdown_template,
                    custom_template
                ],
                outputs=[export_status, export_file]
            )

#
# End of Export_Functionality.py
######################################################################################################################
