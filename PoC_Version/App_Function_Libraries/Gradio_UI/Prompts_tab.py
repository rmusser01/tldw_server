# Prompts_tab.py
# Description: This file contains the code for the prompts tab in the Gradio UI
#
# Imports
import html
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import fetch_prompt_details, list_prompts
from PoC_Version.App_Function_Libraries.Utils.Utils import logging


#
####################################################################################################
#
# Functions:

def create_prompt_view_tab():
    with gr.TabItem("View Prompt Database", visible=True):
        gr.Markdown("# View Prompt Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                previous_page_button = gr.Button("Previous Page", visible=True)
                next_page_button = gr.Button("Next Page", visible=True)
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
                prompt_selector = gr.Dropdown(label="Select Prompt to View", choices=[])
            with gr.Column():
                results_table = gr.HTML()
                selected_prompt_display = gr.HTML()

        # Function to view database entries
        def view_database(page, entries_per_page):
            try:
                # Use list_prompts to get prompts and total pages
                prompts, total_pages, current_page = list_prompts(page=int(page), per_page=int(entries_per_page))

                table_html = "<table style='width:100%; border-collapse: collapse;'>"
                table_html += "<tr><th style='border: 1px solid black; padding: 8px;'>Title</th><th style='border: 1px solid black; padding: 8px;'>Author</th></tr>"
                prompt_choices = []
                for prompt_name in prompts:
                    details = fetch_prompt_details(prompt_name)
                    if details:
                        title, author, _, _, _, _ = details
                        author = author or "Unknown"  # Handle None author
                        table_html += f"<tr><td style='border: 1px solid black; padding: 8px;'>{html.escape(title)}</td><td style='border: 1px solid black; padding: 8px;'>{html.escape(author)}</td></tr>"
                        prompt_choices.append(prompt_name)  # Using prompt_name as value
                table_html += "</table>"

                # Get total prompts if possible
                total_prompts = total_pages * int(entries_per_page)  # This might overestimate if the last page is not full

                pagination = f"Page {current_page} of {total_pages} (Total prompts: {total_prompts})"

                return table_html, pagination, total_pages, prompt_choices
            except Exception as e:
                return f"<p>Error fetching prompts: {e}</p>", "Error", 0, []

        # Function to update page content
        def update_page(page, entries_per_page):
            results, pagination, total_pages, prompt_choices = view_database(page, entries_per_page)
            page = int(page)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return (
                results,
                pagination,
                page,
                gr.update(visible=True, interactive=not prev_disabled),  # previous_page_button
                gr.update(visible=True, interactive=not next_disabled),   # next_page_button
                gr.update(choices=prompt_choices)
            )

        # Function to go to the next page
        def go_to_next_page(current_page, entries_per_page):
            next_page = int(current_page) + 1
            return update_page(next_page, entries_per_page)

        # Function to go to the previous page
        def go_to_previous_page(current_page, entries_per_page):
            previous_page = max(1, int(current_page) - 1)
            return update_page(previous_page, entries_per_page)

        # Function to display selected prompt details
        def display_selected_prompt(prompt_name):
            details = fetch_prompt_details(prompt_name)
            if details:
                title, author, description, system_prompt, user_prompt, keywords = details
                # Handle None values by converting them to empty strings
                description = description or ""
                system_prompt = system_prompt or ""
                user_prompt = user_prompt or ""
                author = author or "Unknown"
                keywords = keywords or ""

                html_content = f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 20px;">
                    <h3>{html.escape(title)}</h3> <h4>by {html.escape(author)}</h4>
                    <p><strong>Description:</strong> {html.escape(description)}</p>
                    <div style="margin-top: 10px;">
                        <strong>System Prompt:</strong>
                        <pre style="white-space: pre-wrap; word-wrap: break-word;">{html.escape(system_prompt)}</pre>
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>User Prompt:</strong>
                        <pre style="white-space: pre-wrap; word-wrap: break-word;">{html.escape(user_prompt)}</pre>
                    </div>
                    <p><strong>Keywords:</strong> {html.escape(keywords)}</p>
                </div>
                """
                return html_content
            else:
                return "<p>Prompt not found.</p>"

        # Event handlers
        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_table, pagination_info, page_number, previous_page_button, next_page_button, prompt_selector]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_table, pagination_info, page_number, previous_page_button, next_page_button, prompt_selector]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_table, pagination_info, page_number, previous_page_button, next_page_button, prompt_selector]
        )

        prompt_selector.change(
            fn=display_selected_prompt,
            inputs=[prompt_selector],
            outputs=[selected_prompt_display]
        )



def create_prompts_export_tab():
    """Creates a tab for exporting prompts database content with multiple format options"""
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
# End of Prompts_tab.py
####################################################################################################
