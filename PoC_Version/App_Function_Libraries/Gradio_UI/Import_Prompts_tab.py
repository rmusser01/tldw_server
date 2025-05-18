# Import_Functionality.py
# Functionality to import Prompts into the Prompts DB
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import list_prompts, insert_prompt_to_db
from PoC_Version.App_Function_Libraries.Prompt_Handling import import_prompts_from_zip, import_prompt_from_file
#
#######################################################################################################################
#
# Prompt Import Functionality

def create_import_single_prompt_tab():
    with gr.TabItem("Import a Prompt", visible=True):
        gr.Markdown("# Import a prompt into the database")

        with gr.Row():
            with gr.Column():
                import_file = gr.File(label="Upload file for import", file_types=["txt", "md"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                system_input = gr.Textbox(label="System", placeholder="Enter the system message for the prompt", lines=3)
                user_input = gr.Textbox(label="User", placeholder="Enter the user message for the prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas")
                import_button = gr.Button("Import Prompt")

            with gr.Column():
                import_output = gr.Textbox(label="Import Status")
                save_button = gr.Button("Save to Database")
                save_output = gr.Textbox(label="Save Status")

        def handle_import(file):
            result = import_prompt_from_file(file)
            if isinstance(result, tuple) and len(result) == 5:
                title, author, system, user, keywords = result
                return gr.update(value="File successfully imported. You can now edit the content before saving."), \
                       gr.update(value=title), gr.update(value=author), gr.update(value=system), \
                       gr.update(value=user), gr.update(value=", ".join(keywords))
            else:
                return gr.update(value=result), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        import_button.click(
            fn=handle_import,
            inputs=[import_file],
            outputs=[import_output, title_input, author_input, system_input, user_input, keywords_input]
        )

        def save_prompt_to_db(title, author, system, user, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            return insert_prompt_to_db(title, author, system, user, keyword_list)

        save_button.click(
            fn=save_prompt_to_db,
            inputs=[title_input, author_input, system_input, user_input, keywords_input],
            outputs=save_output
        )


def create_import_multiple_prompts_tab():
    with gr.TabItem("Import Multiple Prompts", visible=True):
        gr.Markdown("# Import multiple prompts into the database")
        gr.Markdown("Upload a zip file containing multiple prompt files (txt or md)")

        # Initialize state variables for pagination
        current_page_state = gr.State(value=1)
        total_pages_state = gr.State(value=1)

        with gr.Row():
            with gr.Column():
                zip_file = gr.File(label="Upload zip file for import", file_types=["zip"])
                import_button = gr.Button("Import Prompts")
                prompts_dropdown = gr.Dropdown(label="Select Prompt to Edit", choices=[])
                prev_page_button = gr.Button("Previous Page", visible=False)
                page_display = gr.Markdown("Page 1 of X", visible=False)
                next_page_button = gr.Button("Next Page", visible=False)
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                system_input = gr.Textbox(label="System", placeholder="Enter the system message for the prompt",
                                          lines=3)
                user_input = gr.Textbox(label="User", placeholder="Enter the user message for the prompt", lines=3)
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords separated by commas")

            with gr.Column():
                import_output = gr.Textbox(label="Import Status")
                save_button = gr.Button("Save to Database")
                save_output = gr.Textbox(label="Save Status")
                prompts_display = gr.Textbox(label="Identified Prompts")

        # State to store imported prompts
        zip_import_state = gr.State([])

        # Function to handle zip import
        def handle_zip_import(zip_file):
            result = import_prompts_from_zip(zip_file)
            if isinstance(result, list):
                prompt_titles = [prompt['title'] for prompt in result]
                return gr.update(
                    value="Zip file successfully imported. Select a prompt to edit from the dropdown."), prompt_titles, gr.update(
                    value="\n".join(prompt_titles)), result
            else:
                return gr.update(value=result), [], gr.update(value=""), []

        import_button.click(
            fn=handle_zip_import,
            inputs=[zip_file],
            outputs=[import_output, prompts_dropdown, prompts_display, zip_import_state]
        )

        # Function to handle prompt selection from imported prompts
        def handle_prompt_selection(selected_title, prompts):
            selected_prompt = next((prompt for prompt in prompts if prompt['title'] == selected_title), None)
            if selected_prompt:
                return (
                    selected_prompt['title'],
                    selected_prompt.get('author', ''),
                    selected_prompt['system'],
                    selected_prompt.get('user', ''),
                    ", ".join(selected_prompt.get('keywords', []))
                )
            else:
                return "", "", "", "", ""

        zip_import_state = gr.State([])

        import_button.click(
            fn=handle_zip_import,
            inputs=[zip_file],
            outputs=[import_output, prompts_dropdown, prompts_display, zip_import_state]
        )

        prompts_dropdown.change(
            fn=handle_prompt_selection,
            inputs=[prompts_dropdown, zip_import_state],
            outputs=[title_input, author_input, system_input, user_input, keywords_input]
        )

        # Function to save prompt to the database
        def save_prompt_to_db(title, author, system, user, keywords):
            keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
            result = insert_prompt_to_db(title, author, system, user, keyword_list)
            return result

        save_button.click(
            fn=save_prompt_to_db,
            inputs=[title_input, author_input, system_input, user_input, keywords_input],
            outputs=[save_output]
        )

        # Adding pagination controls to navigate prompts in the database
        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[prompts_dropdown, page_display, current_page_state]
        )

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[prompts_dropdown, page_display, current_page_state]
        )

        # Function to update prompts dropdown after saving to the database
        def update_prompt_dropdown():
            prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(visible=True),
                gr.update(value=page_display_text, visible=True),
                current_page,
                total_pages
            )

        # Update the dropdown after saving
        save_button.click(
            fn=update_prompt_dropdown,
            inputs=[],
            outputs=[prompts_dropdown, prev_page_button, page_display, current_page_state, total_pages_state]
        )

#
# End of Prompt Import Functionality
#######################################################################################################################
