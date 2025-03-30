# View_DB_Items_tab.py
# Description: This file contains the code for the search tab in the Gradio UI
#
# Imports
import html
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import view_database, get_all_document_versions, \
    fetch_paginated_data, fetch_item_details, get_latest_transcription
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import get_keywords_for_note, search_conversations_by_keywords, \
    get_notes_by_keywords, get_keywords_for_conversation, get_db_connection, get_all_conversations, load_chat_history, \
    get_notes
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import get_document_version, fetch_items_by_keyword, fetch_all_keywords
from PoC_Version.App_Function_Libraries.Utils.Utils import logging
#
####################################################################################################
#
# Functions

def format_as_html(content, title):
    escaped_content = html.escape(content)
    formatted_content = escaped_content.replace('\n', '<br>')
    return f"""
    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
        <h3>{title}</h3>
        <div style="max-height: 700px; overflow-y: auto;">
            {formatted_content}
        </div>
    </div>
    """

def extract_prompt_and_summary(content: str):
    # Implement this function based on how prompt and summary are stored in your DocumentVersions content
    # This is a placeholder implementation
    parts = content.split('\n\n', 2)
    prompt = parts[0] if len(parts) > 0 else "No prompt available."
    summary = parts[1] if len(parts) > 1 else "No summary available."
    return prompt, summary


def create_view_all_mediadb_with_versions_tab():
    with gr.TabItem("View All MediaDB Items", visible=True):
        gr.Markdown("# View All Media Database Entries with Version Selection")
        with gr.Row():
            with gr.Column(scale=1):
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
            with gr.Column(scale=2):
                items_output = gr.Dropdown(label="Select Item to View Details", choices=[])
                version_dropdown = gr.Dropdown(label="Select Version", choices=[], visible=False)
        with gr.Row():
            with gr.Column(scale=1):
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column(scale=2):
                prompt_output = gr.Textbox(label="Prompt Used", visible=True)
                summary_output = gr.HTML(label="Summary", visible=True)
                transcription_output = gr.HTML(label="Transcription", visible=True)

        item_mapping = gr.State({})

        def update_page(page, entries_per_page):
            results, total_entries = fetch_paginated_data(page, entries_per_page)
            total_pages = (total_entries + entries_per_page - 1) // entries_per_page
            pagination = f"Page {page} of {total_pages} (Total items: {total_entries})"

            choices = [f"{item[1]} (ID: {item[0]})" for item in results]
            new_item_mapping = {f"{item[1]} (ID: {item[0]})": item[0] for item in results}

            next_disabled = page >= total_pages
            prev_disabled = page <= 1

            return (gr.update(choices=choices, value=None),
                    pagination,
                    page,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled),
                    gr.update(visible=False, choices=[]),
                    "", "", "",
                    new_item_mapping)

        def format_as_html(content, title):
            if content is None:
                content = "No content available."
            escaped_content = html.escape(str(content))
            formatted_content = escaped_content.replace('\n', '<br>')
            return f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                <h3>{title}</h3>
                <div style="max-height: 700px; overflow-y: auto;">
                    {formatted_content}
                </div>
            </div>
            """

        def display_item_details(selected_item, item_mapping):
            if selected_item and item_mapping and selected_item in item_mapping:
                media_id = item_mapping[selected_item]
                prompt, summary, transcription = fetch_item_details(media_id)
                versions = get_all_document_versions(media_id)

                # Filter out duplicate versions and sort them
                unique_versions = list(set((v['version_number'], v['created_at']) for v in versions))
                unique_versions.sort(key=lambda x: x[0], reverse=True)
                version_choices = [f"Version {v[0]} ({v[1]})" for v in unique_versions]

                summary_html = format_as_html(summary, "Summary")
                transcription_html = format_as_html(transcription, "Transcription")

                return (
                    gr.update(visible=True, choices=version_choices,
                              value=version_choices[0] if version_choices else None),
                    prompt if prompt is not None else "",
                    summary_html,
                    transcription_html
                )
            return gr.update(visible=False, choices=[]), "", "", ""

        def update_version_content(selected_item, item_mapping, selected_version):
            if selected_item and item_mapping and selected_item in item_mapping and selected_version:
                media_id = item_mapping[selected_item]
                version_number = int(selected_version.split()[1].split('(')[0])
                version_data = get_document_version(media_id, version_number)

                if 'error' not in version_data:
                    content = version_data['content']
                    prompt, summary = extract_prompt_and_summary(content)
                    transcription = get_latest_transcription(media_id)

                    summary_html = format_as_html(summary, "Summary")
                    transcription_html = format_as_html(transcription, "Transcription")

                    return prompt if prompt is not None else "", summary_html, transcription_html
            return gr.update(value=selected_item), gr.update(), gr.update()

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, prompt_output, summary_output, transcription_output, item_mapping]
        )

        next_page_button.click(
            fn=lambda page, entries: update_page(page + 1, entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, prompt_output, summary_output, transcription_output, item_mapping]
        )

        previous_page_button.click(
            fn=lambda page, entries: update_page(max(1, page - 1), entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, prompt_output, summary_output, transcription_output, item_mapping]
        )

        items_output.change(
            fn=display_item_details,
            inputs=[items_output, item_mapping],
            outputs=[version_dropdown, prompt_output, summary_output, transcription_output]
        )

        version_dropdown.change(
            fn=update_version_content,
            inputs=[items_output, item_mapping, version_dropdown],
            outputs=[prompt_output, summary_output, transcription_output]
        )


def create_mediadb_keyword_search_tab():
    with gr.TabItem("Search MediaDB by Keyword", visible=True):
        gr.Markdown("# List Media Database Items by Keyword")

        with gr.Row():
            with gr.Column(scale=1):
                # Keyword selection dropdown - initialize with empty list, will be populated on load
                keyword_dropdown = gr.Dropdown(
                    label="Select Keyword",
                    choices=fetch_all_keywords(),  # Initialize with keywords on creation
                    value=None
                )
                entries_per_page = gr.Dropdown(
                    choices=[10, 20, 50, 100],
                    label="Entries per Page",
                    value=10
                )
                page_number = gr.Number(
                    value=1,
                    label="Page Number",
                    precision=0
                )

                # Navigation buttons
                refresh_keywords_button = gr.Button("Refresh Keywords")
                view_button = gr.Button("View Results")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")

                # Pagination information
                pagination_info = gr.Textbox(
                    label="Pagination Info",
                    interactive=False
                )

            with gr.Column(scale=2):
                # Results area
                results_table = gr.HTML(
                    label="Search Results"
                )
                item_details = gr.HTML(
                    label="Item Details",
                    visible=True
                )

        def update_keyword_choices():
            try:
                keywords = fetch_all_keywords()
                return gr.update(choices=keywords)
            except Exception as e:
                return gr.update(choices=[], value=None)

        def search_items(keyword, page, entries_per_page):
            try:
                # Calculate offset for pagination
                offset = (page - 1) * entries_per_page

                # Fetch items for the selected keyword
                items = fetch_items_by_keyword(keyword)
                total_items = len(items)
                total_pages = (total_items + entries_per_page - 1) // entries_per_page

                # Paginate results
                paginated_items = items[offset:offset + entries_per_page]

                # Generate HTML table for results
                table_html = "<table style='width:100%; border-collapse: collapse;'>"
                table_html += "<tr><th style='border: 1px solid black; padding: 8px;'>Title</th>"
                table_html += "<th style='border: 1px solid black; padding: 8px;'>URL</th></tr>"

                for item_id, title, url in paginated_items:
                    table_html += f"""
                    <tr>
                        <td style='border: 1px solid black; padding: 8px;'>{html.escape(title)}</td>
                        <td style='border: 1px solid black; padding: 8px;'>{html.escape(url)}</td>
                    </tr>
                    """
                table_html += "</table>"

                # Update pagination info
                pagination = f"Page {page} of {total_pages} (Total items: {total_items})"

                # Determine button states
                next_disabled = page >= total_pages
                prev_disabled = page <= 1

                return (
                    table_html,
                    pagination,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled)
                )
            except Exception as e:
                return (
                    f"<p>Error: {str(e)}</p>",
                    "Error in pagination",
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )

        def go_to_next_page(keyword, current_page, entries_per_page):
            next_page = current_page + 1
            return search_items(keyword, next_page, entries_per_page) + (next_page,)

        def go_to_previous_page(keyword, current_page, entries_per_page):
            previous_page = max(1, current_page - 1)
            return search_items(keyword, previous_page, entries_per_page) + (previous_page,)

        # Event handlers
        refresh_keywords_button.click(
            fn=update_keyword_choices,
            inputs=[],
            outputs=[keyword_dropdown]
        )

        view_button.click(
            fn=search_items,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[results_table, pagination_info, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[results_table, pagination_info, next_page_button, previous_page_button, page_number]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[results_table, pagination_info, next_page_button, previous_page_button, page_number]
        )


def create_viewing_mediadb_tab():
    with gr.TabItem("View Media Database Entries", visible=True):
        gr.Markdown("# View Media Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column():
                results_display = gr.HTML()


        def update_page(page, entries_per_page):
            results, pagination, total_pages = view_database(page, entries_per_page)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return results, pagination, page, gr.update(interactive=not next_disabled), gr.update(interactive=not prev_disabled)

        def go_to_next_page(current_page, entries_per_page):
            next_page = current_page + 1
            return update_page(next_page, entries_per_page)

        def go_to_previous_page(current_page, entries_per_page):
            previous_page = max(1, current_page - 1)
            return update_page(previous_page, entries_per_page)

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

#####################################################################
#
# RAG DB Viewing Functions:

def create_viewing_ragdb_tab():
    with gr.TabItem("View RAG Database Entries", visible=True):
        gr.Markdown("# View RAG Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column():
                results_display = gr.HTML()

        def format_conversations_table(conversations):
            table_html = "<table style='width:100%; border-collapse: collapse;'>"
            table_html += """
                <tr>
                    <th style='border: 1px solid black; padding: 8px;'>Title</th>
                    <th style='border: 1px solid black; padding: 8px;'>Keywords</th>
                    <th style='border: 1px solid black; padding: 8px;'>Notes</th>
                    <th style='border: 1px solid black; padding: 8px;'>Rating</th>
                </tr>
            """

            for conversation in conversations:
                conv_id = conversation['conversation_id']
                title = conversation['title']
                rating = conversation.get('rating', '')  # Use get() to handle cases where rating might not exist

                keywords = get_keywords_for_conversation(conv_id)
                notes = get_notes(conv_id)

                table_html += f"""
                    <tr>
                        <td style='border: 1px solid black; padding: 8px;'>{html.escape(str(title))}</td>
                        <td style='border: 1px solid black; padding: 8px;'>{html.escape(', '.join(keywords))}</td>
                        <td style='border: 1px solid black; padding: 8px;'>{len(notes)} note(s)</td>
                        <td style='border: 1px solid black; padding: 8px;'>{html.escape(str(rating))}</td>
                    </tr>
                """
            table_html += "</table>"
            return table_html

        def update_page(page, entries_per_page):
            try:
                conversations, total_pages, total_count = get_all_conversations(page, entries_per_page)
                results_html = format_conversations_table(conversations)
                pagination = f"Page {page} of {total_pages} (Total conversations: {total_count})"

                next_disabled = page >= total_pages
                prev_disabled = page <= 1

                return (
                    results_html,
                    pagination,
                    page,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled)
                )
            except Exception as e:
                return (
                    f"<p>Error: {str(e)}</p>",
                    "Error in pagination",
                    page,
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )

        def go_to_next_page(current_page, entries_per_page):
            return update_page(current_page + 1, entries_per_page)

        def go_to_previous_page(current_page, entries_per_page):
            return update_page(max(1, current_page - 1), entries_per_page)

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )


def create_view_all_rag_notes_tab():
    with gr.TabItem("View All RAG notes/Conversation Items", visible=True):
        gr.Markdown("# View All RAG Notes/Conversation Entries")
        with gr.Row():
            with gr.Column(scale=1):
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
            with gr.Column(scale=2):
                items_output = gr.Dropdown(label="Select Conversation to View Details", choices=[])
                conversation_title = gr.Textbox(label="Conversation Title", visible=True)
        with gr.Row():
            with gr.Column(scale=1):
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column(scale=2):
                keywords_output = gr.Textbox(label="Keywords", visible=True)
                chat_history_output = gr.HTML(label="Chat History", visible=True)
                notes_output = gr.HTML(label="Associated Notes", visible=True)

        item_mapping = gr.State({})

        def update_page(page, entries_per_page):
            try:
                conversations, total_pages, total_count = get_all_conversations(page, entries_per_page)
                pagination = f"Page {page} of {total_pages} (Total conversations: {total_count})"

                # Handle the dictionary structure correctly
                choices = [f"{conv['title']} (ID: {conv['conversation_id']})" for conv in conversations]
                new_item_mapping = {
                    f"{conv['title']} (ID: {conv['conversation_id']})": conv['conversation_id']
                    for conv in conversations
                }

                next_disabled = page >= total_pages
                prev_disabled = page <= 1

                return (
                    gr.update(choices=choices, value=None),
                    pagination,
                    page,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled),
                    "",  # conversation_title
                    "",  # keywords_output
                    "",  # chat_history_output
                    "",  # notes_output
                    new_item_mapping
                )
            except Exception as e:
                logging.error(f"Error in update_page: {str(e)}", exc_info=True)
                return (
                    gr.update(choices=[], value=None),
                    f"Error: {str(e)}",
                    page,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "", "", "", "",
                    {}
                )

        def format_as_html(content, title):
            if content is None:
                content = "No content available."
            escaped_content = html.escape(str(content))
            formatted_content = escaped_content.replace('\n', '<br>')
            return f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;">
                <h3>{title}</h3>
                <div style="max-height: 700px; overflow-y: auto;">
                    {formatted_content}
                </div>
            </div>
            """

        def format_chat_history(messages):
            html_content = "<div style='max-height: 500px; overflow-y: auto;'>"
            for role, content in messages:
                role_class = "assistant" if role.lower() == "assistant" else "user"
                html_content += f"""
                <div class='{role_class}-message' style='margin: 10px 0; padding: 10px; border-radius: 5px; 
                    background-color: {"#f0f0f0" if role_class == "user" else "#e3f2fd"}'>
                    <strong>{html.escape(role)}:</strong><br>
                    {html.escape(content)}
                </div>
                """
            html_content += "</div>"
            return html_content

        def display_conversation_details(selected_item, item_mapping):
            if selected_item and item_mapping and selected_item in item_mapping:
                conv_id = item_mapping[selected_item]

                # Get keywords
                keywords = get_keywords_for_conversation(conv_id)
                keywords_text = ", ".join(keywords) if keywords else "No keywords"

                # Get chat history
                chat_messages, _, _ = load_chat_history(conv_id)
                chat_html = format_chat_history(chat_messages)

                # Get associated notes
                notes = get_notes(conv_id)
                notes_html = ""
                for note in notes:
                    notes_html += format_as_html(note, "Note")
                if not notes:
                    notes_html = "<p>No notes associated with this conversation.</p>"

                return (
                    selected_item.split(" (ID:")[0],  # Conversation title
                    keywords_text,
                    chat_html,
                    notes_html
                )
            return "", "", "", ""

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[
                items_output,
                pagination_info,
                page_number,
                next_page_button,
                previous_page_button,
                conversation_title,
                keywords_output,
                chat_history_output,
                notes_output,
                item_mapping
            ]
        )

        next_page_button.click(
            fn=lambda page, entries: update_page(page + 1, entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     conversation_title, keywords_output, chat_history_output, notes_output, item_mapping]
        )

        previous_page_button.click(
            fn=lambda page, entries: update_page(max(1, page - 1), entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     conversation_title, keywords_output, chat_history_output, notes_output, item_mapping]
        )

        items_output.change(
            fn=display_conversation_details,
            inputs=[items_output, item_mapping],
            outputs=[conversation_title, keywords_output, chat_history_output, notes_output]
        )


def create_ragdb_keyword_items_tab():
    with gr.TabItem("View RAG Notes/Conversations by Keyword", visible=True):
        gr.Markdown("# View RAG Notes and Conversations by Keyword")

        with gr.Row():
            with gr.Column(scale=1):
                # Keyword selection
                keyword_dropdown = gr.Dropdown(
                    label="Select Keyword",
                    choices=[],
                    value=None,
                    multiselect=True
                )
                entries_per_page = gr.Dropdown(
                    choices=[10, 20, 50, 100],
                    label="Entries per Page",
                    value=10
                )
                page_number = gr.Number(
                    value=1,
                    label="Page Number",
                    precision=0
                )

                # Navigation buttons
                refresh_keywords_button = gr.Button("Refresh Keywords")
                view_button = gr.Button("View Items")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(
                    label="Pagination Info",
                    interactive=False
                )

            with gr.Column(scale=2):
                # Results tabs for conversations and notes
                with gr.Tabs():
                    with gr.Tab("Notes"):
                        notes_results = gr.HTML()
                    with gr.Tab("Conversations"):
                        conversation_results = gr.HTML()

        def update_keyword_choices():
            """Fetch all available keywords for the dropdown."""
            try:
                query = "SELECT keyword FROM rag_qa_keywords ORDER BY keyword"
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    keywords = [row[0] for row in cursor.fetchall()]
                return gr.update(choices=keywords)
            except Exception as e:
                return gr.update(choices=[], value=None)

        def format_conversations_html(conversations_data):
            """Format conversations data as HTML."""
            if not conversations_data:
                return "<p>No conversations found for selected keywords.</p>"

            html_content = "<div class='results-container'>"
            for conv_id, title in conversations_data:
                html_content += f"""
                <div style='border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;'>
                    <h3>{html.escape(title)}</h3>
                    <p>Conversation ID: {html.escape(conv_id)}</p>
                    <p><strong>Keywords:</strong> {', '.join(html.escape(k) for k in get_keywords_for_conversation(conv_id))}</p>
                </div>
                """
            html_content += "</div>"
            return html_content

        def format_notes_html(notes_data):
            """Format notes data as HTML."""
            if not notes_data:
                return "<p>No notes found for selected keywords.</p>"

            html_content = "<div class='results-container'>"
            for note_id, title, content, timestamp in notes_data:
                keywords = get_keywords_for_note(note_id)
                html_content += f"""
                <div style='border: 1px solid #ddd; padding: 10px; margin-bottom: 10px;'>
                    <h3>{html.escape(title)}</h3>
                    <p><strong>Created:</strong> {timestamp}</p>
                    <p><strong>Keywords:</strong> {', '.join(html.escape(k) for k in keywords)}</p>
                    <div style='background: #f5f5f5; padding: 10px; margin-top: 10px;'>
                        {html.escape(content)}
                    </div>
                </div>
                """
            html_content += "</div>"
            return html_content

        def view_items(keywords, page, entries_per_page):
            if not keywords or (isinstance(keywords, list) and len(keywords) == 0):
                return (
                    "<p>Please select at least one keyword.</p>",
                    "<p>Please select at least one keyword.</p>",
                    "No results",
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )

            try:
                # Ensure keywords is a list
                keywords_list = keywords if isinstance(keywords, list) else [keywords]

                # Get conversations for selected keywords
                conversations, conv_total_pages, conv_count = search_conversations_by_keywords(
                    keywords_list, page, entries_per_page
                )

                # Get notes for selected keywords
                notes, notes_total_pages, notes_count = get_notes_by_keywords(
                    keywords_list, page, entries_per_page
                )

                # Format results as HTML
                conv_html = format_conversations_html(conversations)
                notes_html = format_notes_html(notes)

                # Create pagination info
                pagination = f"Page {page} of {max(conv_total_pages, notes_total_pages)} "
                pagination += f"(Conversations: {conv_count}, Notes: {notes_count})"

                # Determine button states
                max_pages = max(conv_total_pages, notes_total_pages)
                next_disabled = page >= max_pages
                prev_disabled = page <= 1

                return (
                    conv_html,
                    notes_html,
                    pagination,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled)
                )
            except Exception as e:
                logging.error(f"Error in view_items: {str(e)}")
                return (
                    f"<p>Error: {str(e)}</p>",
                    f"<p>Error: {str(e)}</p>",
                    "Error in retrieval",
                    gr.update(interactive=False),
                    gr.update(interactive=False)
                )

        def go_to_next_page(keywords, current_page, entries_per_page):
            return view_items(keywords, current_page + 1, entries_per_page)

        def go_to_previous_page(keywords, current_page, entries_per_page):
            return view_items(keywords, max(1, current_page - 1), entries_per_page)

        # Event handlers
        refresh_keywords_button.click(
            fn=update_keyword_choices,
            inputs=[],
            outputs=[keyword_dropdown]
        )

        view_button.click(
            fn=view_items,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[
                conversation_results,
                notes_results,
                pagination_info,
                next_page_button,
                previous_page_button
            ]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[
                conversation_results,
                notes_results,
                pagination_info,
                next_page_button,
                previous_page_button
            ]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[keyword_dropdown, page_number, entries_per_page],
            outputs=[
                conversation_results,
                notes_results,
                pagination_info,
                next_page_button,
                previous_page_button
            ]
        )

        # Initialize keyword dropdown on page load
        keyword_dropdown.value = update_keyword_choices()

#
# End of RAG DB Viewing tabs
################################################################

#
#######################################################################################################################
