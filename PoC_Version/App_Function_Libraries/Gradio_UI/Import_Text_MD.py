# Import_Text_MD.py
# Description: This file contains functions for importing Markdown/plaintext files into the DB
#
# Import necessary libraries
#
# External Imports
import gradio as gr

from App_Function_Libraries.Plaintext.Plaintext_Files import import_data


#
# Local Imports

#
#######################################################################################################################
#
# Function Definitions

def create_import_item_tab():
    with gr.TabItem("Import Markdown/Text Files", visible=True):
        gr.Markdown("# Import a markdown file or text file into the database")
        gr.Markdown("...and have it tagged + summarized")
        with gr.Row():
            with gr.Column():
                import_file = gr.File(label="Upload file for import", file_types=["txt", "md"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                author_input = gr.Textbox(label="Author", placeholder="Enter the author's name")
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords, comma-separated")
                custom_prompt_input = gr.Textbox(label="Custom Prompt",
                                             placeholder="Enter a custom prompt for summarization (optional)")
                summary_input = gr.Textbox(label="Summary",
                                       placeholder="Enter a summary or leave blank for auto-summarization", lines=3)
                auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize", value=False)
                api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral", "OpenRouter",
                         "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM","ollama", "HuggingFace", "Custom-OpenAI-API"],
                label="API for Auto-summarization"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")
            with gr.Column():
                import_button = gr.Button("Import Data")
                import_output = gr.Textbox(label="Import Status")

        import_button.click(
            fn=import_data,
            inputs=[import_file, title_input, author_input, keywords_input, custom_prompt_input,
                    summary_input, auto_summarize_checkbox, api_name_input, api_key_input],
            outputs=import_output
        )

#
# End of Import_Text_MD.py
#######################################################################################################################
