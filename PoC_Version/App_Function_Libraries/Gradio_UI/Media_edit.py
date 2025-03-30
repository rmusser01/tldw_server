# Media_edit.py
# Functions for Gradio Media_Edit UI
#
# Imports
import uuid
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_prompt, update_media_content, db, add_or_update_prompt, \
    fetch_keywords_for_media, update_keywords_for_media, fetch_prompt_details, list_prompts
from PoC_Version.App_Function_Libraries.DB.Prompts_DB import fetch_item_details_with_keywords
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import update_dropdown
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import fetch_item_details
from PoC_Version.App_Function_Libraries.Utils.Utils import logging


#
#######################################################################################################################
#
# Functions:

def create_media_edit_tab():
    with gr.TabItem("Edit Existing Items in the Media DB", visible=True):
        gr.Markdown("# Search and Edit Media Items")

        with gr.Row():
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title", label="Search By")
            search_button = gr.Button("Search")

        with gr.Row():
            items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
            item_mapping = gr.State({})

        content_input = gr.Textbox(label="Edit Content", lines=10)
        prompt_input = gr.Textbox(label="Edit Prompt", lines=3)
        summary_input = gr.Textbox(label="Edit Summary", lines=5)

        # Adding keyword input box for editing
        keywords_input = gr.Textbox(label="Edit Keywords (comma-separated)", placeholder="Enter keywords here...")

        update_button = gr.Button("Update Media Content")
        status_message = gr.Textbox(label="Status", interactive=False)

        # Function to update the dropdown with search results
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        # Function to load selected media content including keywords
        def load_selected_media_content(selected_item, item_mapping):
            if selected_item and item_mapping and selected_item in item_mapping:
                media_id = item_mapping[selected_item]
                content, prompt, summary = fetch_item_details(media_id)

                # Fetch keywords for the selected item
                keywords = fetch_keywords_for_media(media_id)
                keywords_str = ", ".join(keywords) if keywords else ""

                return content, prompt, summary, keywords_str
            return "No item selected or invalid selection", "", "", ""

        # Load the selected media content and associated keywords
        items_output.change(
            fn=load_selected_media_content,
            inputs=[items_output, item_mapping],
            outputs=[content_input, prompt_input, summary_input, keywords_input]
        )

        # Function to update media content, prompt, summary, and keywords
        def update_media_with_keywords(selected_item, item_mapping, content, prompt, summary, keywords):
            if selected_item and item_mapping and selected_item in item_mapping:
                media_id = item_mapping[selected_item]

                # Split keywords into a list
                keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]

                # Update content, prompt, summary, and keywords in the database
                status = update_media_content(media_id, content, prompt, summary)
                keyword_status = update_keywords_for_media(media_id, keyword_list)

                return f"{status}\nKeywords: {keyword_status}"
            return "No item selected or invalid selection"

        # Update button click event
        update_button.click(
            fn=update_media_with_keywords,
            inputs=[items_output, item_mapping, content_input, prompt_input, summary_input, keywords_input],
            outputs=status_message
        )


def create_media_edit_and_clone_tab():
    with gr.TabItem("Clone and Edit Existing Items in the Media DB", visible=True):
        gr.Markdown("# Search, Edit, and Clone Existing Items")

        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
            with gr.Column():
                search_button = gr.Button("Search")
                clone_button = gr.Button("Clone Item")
            save_clone_button = gr.Button("Save Cloned Item", visible=False)
        with gr.Row():
            items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
            item_mapping = gr.State({})

        content_input = gr.Textbox(label="Edit Content", lines=10)
        prompt_input = gr.Textbox(label="Edit Prompt", lines=3)
        summary_input = gr.Textbox(label="Edit Summary", lines=5)
        keywords_input = gr.Textbox(label="Edit Keywords (comma-separated)", lines=2)
        new_title_input = gr.Textbox(label="New Title (for cloning)", visible=False)
        status_message = gr.Textbox(label="Status", interactive=False)

        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def load_selected_media_content(selected_item, item_mapping):
            if selected_item and item_mapping and selected_item in item_mapping:
                media_id = item_mapping[selected_item]
                content, prompt, summary, keywords = fetch_item_details_with_keywords(media_id)
                return content, prompt, summary, keywords, gr.update(visible=True), gr.update(visible=False)
            return "No item selected or invalid selection", "", "", "", gr.update(visible=False), gr.update(visible=False)

        items_output.change(
            fn=load_selected_media_content,
            inputs=[items_output, item_mapping],
            outputs=[content_input, prompt_input, summary_input, keywords_input, clone_button, save_clone_button]
        )

        def prepare_for_cloning(selected_item):
            return gr.update(value=f"Copy of {selected_item}", visible=True), gr.update(visible=True)

        clone_button.click(
            fn=prepare_for_cloning,
            inputs=[items_output],
            outputs=[new_title_input, save_clone_button]
        )

        def save_cloned_item(selected_item, item_mapping, content, prompt, summary, keywords, new_title):
            if selected_item and item_mapping and selected_item in item_mapping:
                original_media_id = item_mapping[selected_item]
                try:
                    with db.get_connection() as conn:
                        cursor = conn.cursor()

                        # Fetch the original item's details
                        cursor.execute("SELECT type, url FROM Media WHERE id = ?", (original_media_id,))
                        original_type, original_url = cursor.fetchone()

                        # Generate a new unique URL
                        new_url = f"{original_url}_clone_{uuid.uuid4().hex[:8]}"

                        # Insert new item into Media table
                        cursor.execute("""
                            INSERT INTO Media (title, content, url, type)
                            VALUES (?, ?, ?, ?)
                        """, (new_title, content, new_url, original_type))

                        new_media_id = cursor.lastrowid

                        # Insert new item into MediaModifications table
                        cursor.execute("""
                            INSERT INTO MediaModifications (media_id, prompt, summary, modification_date)
                            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                        """, (new_media_id, prompt, summary))

                        # Handle keywords
                        keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                        for keyword in keyword_list:
                            # Insert keyword if it doesn't exist
                            cursor.execute("INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)", (keyword,))
                            cursor.execute("SELECT id FROM Keywords WHERE keyword = ?", (keyword,))
                            keyword_id = cursor.fetchone()[0]

                            # Associate keyword with the new media item
                            cursor.execute("INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", (new_media_id, keyword_id))

                        # Update full-text search index
                        cursor.execute("""
                            INSERT INTO media_fts (rowid, title, content)
                            VALUES (?, ?, ?)
                        """, (new_media_id, new_title, content))

                        conn.commit()

                    return f"Cloned item saved successfully with ID: {new_media_id}", gr.update(
                        visible=False), gr.update(visible=False)
                except Exception as e:
                    logging.error(f"Error saving cloned item: {e}")
                    return f"Error saving cloned item: {str(e)}", gr.update(visible=True), gr.update(visible=True)
            else:
                return "No item selected or invalid selection", gr.update(visible=True), gr.update(visible=True)

        save_clone_button.click(
            fn=save_cloned_item,
            inputs=[items_output, item_mapping, content_input, prompt_input, summary_input, keywords_input, new_title_input],
            outputs=[status_message, new_title_input, save_clone_button]
        )


def create_prompt_edit_tab():
    # Initialize state variables for pagination
    current_page_state = gr.State(value=1)
    total_pages_state = gr.State(value=1)
    per_page = 10  # Number of prompts per page

    with gr.TabItem("Add & Edit Prompts", visible=True):
        with gr.Row():
            with gr.Column():
                prompt_dropdown = gr.Dropdown(
                    label="Select Prompt",
                    choices=[],
                    interactive=True
                )
                next_page_button = gr.Button("Next Page", visible=False)
                page_display = gr.Markdown("Page 1 of X", visible=False)
                prev_page_button = gr.Button("Previous Page", visible=False)
                prompt_list_button = gr.Button("List Prompts")

            with gr.Column():
                title_input = gr.Textbox(label="Title", placeholder="Enter the prompt title")
                author_input = gr.Textbox(label="Author", placeholder="Enter the prompt's author", lines=1)
                description_input = gr.Textbox(label="Description", placeholder="Enter the prompt description", lines=3)
                system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Enter the system prompt", lines=3)
                user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Enter the user prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas", lines=2)
                add_prompt_button = gr.Button("Add/Update Prompt")
                add_prompt_output = gr.HTML()

        # Function to update the prompt dropdown with pagination
        def update_prompt_dropdown(page=1):
            prompts, total_pages, current_page = list_prompts(page=page, per_page=per_page)
            page_display_text = f"Page {current_page} of {total_pages}"
            prev_button_visible = current_page > 1
            next_button_visible = current_page < total_pages
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text, visible=True),
                gr.update(visible=prev_button_visible),
                gr.update(visible=next_button_visible),
                current_page,
                total_pages
            )

        # Event handler for listing prompts
        prompt_list_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        # Functions to handle pagination
        def on_prev_page_click(current_page):
            new_page = max(current_page - 1, 1)
            return update_prompt_dropdown(page=new_page)

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            return update_prompt_dropdown(page=new_page)

        # Event handlers for pagination buttons
        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        # Modified function to add or update a prompt with keywords
        def add_or_update_prompt_with_keywords(title, author, description, system_prompt, user_prompt, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            result = add_or_update_prompt(title, author, description, system_prompt, user_prompt, keyword_list)
            return gr.HTML(result)

        # Event handler for adding or updating a prompt
        add_prompt_button.click(
            fn=add_or_update_prompt_with_keywords,
            inputs=[title_input, author_input, description_input, system_prompt_input, user_prompt_input, keywords_input],
            outputs=[add_prompt_output]
        ).then(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        # Function to load prompt details when a prompt is selected
        def load_prompt_details(selected_prompt):
            details = fetch_prompt_details(selected_prompt)
            if details:
                title, author, description, system_prompt, user_prompt, keywords = details
                return (
                    gr.update(value=title),
                    gr.update(value=author or ""),
                    gr.update(value=description or ""),
                    gr.update(value=system_prompt or ""),
                    gr.update(value=user_prompt or ""),
                    gr.update(value=keywords or "")
                )
            else:
                return (
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value="")
                )

        # Event handler for prompt selection change
        prompt_dropdown.change(
            fn=load_prompt_details,
            inputs=[prompt_dropdown],
            outputs=[
                title_input,
                author_input,
                description_input,
                system_prompt_input,
                user_prompt_input,
                keywords_input
            ]
        )


def create_prompt_clone_tab():
    # Initialize state variables for pagination
    current_page_state = gr.State(value=1)
    total_pages_state = gr.State(value=1)
    per_page = 10  # Number of prompts per page

    with gr.TabItem("Clone and Edit Prompts", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Clone and Edit Prompts")
                prompt_dropdown = gr.Dropdown(
                    label="Select Prompt",
                    choices=[],
                    interactive=True
                )
                next_page_button = gr.Button("Next Page", visible=False)
                page_display = gr.Markdown("Page 1 of X", visible=False)
                prev_page_button = gr.Button("Previous Page", visible=False)
                prompt_list_button = gr.Button("List Prompts")

            with gr.Column():
                title_input = gr.Textbox(label="Title", placeholder="Enter the prompt title")
                author_input = gr.Textbox(label="Author", placeholder="Enter the prompt's author", lines=1)
                description_input = gr.Textbox(label="Description", placeholder="Enter the prompt description", lines=3)
                system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Enter the system prompt", lines=3)
                user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Enter the user prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas", lines=2)
                clone_prompt_button = gr.Button("Clone Selected Prompt")
                save_cloned_prompt_button = gr.Button("Save Cloned Prompt", visible=False)
                add_prompt_output = gr.HTML()

        # Function to update the prompt dropdown with pagination
        def update_prompt_dropdown(page=1):
            prompts, total_pages, current_page = list_prompts(page=page, per_page=per_page)
            page_display_text = f"Page {current_page} of {total_pages}"
            prev_button_visible = current_page > 1
            next_button_visible = current_page < total_pages
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text, visible=True),
                gr.update(visible=prev_button_visible),
                gr.update(visible=next_button_visible),
                current_page,
                total_pages
            )

        # Event handler for listing prompts
        prompt_list_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        # Functions to handle pagination
        def on_prev_page_click(current_page):
            new_page = max(current_page - 1, 1)
            return update_prompt_dropdown(page=new_page)

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            return update_prompt_dropdown(page=new_page)

        # Event handlers for pagination buttons
        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[
                prompt_dropdown,
                page_display,
                prev_page_button,
                next_page_button,
                current_page_state,
                total_pages_state
            ]
        )

        # Load prompt details when selected
        def load_prompt_details(selected_prompt):
            if selected_prompt:
                details = fetch_prompt_details(selected_prompt)
                if details:
                    title, author, description, system_prompt, user_prompt, keywords = details
                    return (
                        gr.update(value=title),
                        gr.update(value=author or ""),
                        gr.update(value=description or ""),
                        gr.update(value=system_prompt or ""),
                        gr.update(value=user_prompt or ""),
                        gr.update(value=keywords or "")
                    )
            return (
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="")
            )

        prompt_dropdown.change(
            fn=load_prompt_details,
            inputs=[prompt_dropdown],
            outputs=[title_input, author_input, description_input, system_prompt_input, user_prompt_input, keywords_input]
        )

        # Prepare for cloning
        def prepare_for_cloning(selected_prompt):
            if selected_prompt:
                return gr.update(value=f"Copy of {selected_prompt}"), gr.update(visible=True)
            return gr.update(), gr.update(visible=False)

        clone_prompt_button.click(
            fn=prepare_for_cloning,
            inputs=[prompt_dropdown],
            outputs=[title_input, save_cloned_prompt_button]
        )

        # Function to save cloned prompt
        def save_cloned_prompt(title, author, description, system_prompt, user_prompt, keywords, current_page):
            try:
                keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                result = add_prompt(title, author, description, system_prompt, user_prompt, keyword_list)
                if result == "Prompt added successfully.":
                    # After adding, refresh the prompt dropdown
                    prompt_dropdown_update = update_prompt_dropdown(page=current_page)
                    return result, *prompt_dropdown_update
                else:
                    return result, gr.update(), gr.update(), gr.update(), gr.update(), current_page, \
                        total_pages_state.value
            except Exception as e:
                return f"Error saving cloned prompt: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), \
                    current_page, total_pages_state.value

        save_cloned_prompt_button.click(
            fn=save_cloned_prompt,
            inputs=[title_input, author_input, description_input, system_prompt_input, user_prompt_input,
                    keywords_input, current_page_state],
            outputs=[add_prompt_output, prompt_dropdown, page_display, prev_page_button, next_page_button,
                     current_page_state, total_pages_state]
        )

#
# End of Media_edit.py
#######################################################################################################################
