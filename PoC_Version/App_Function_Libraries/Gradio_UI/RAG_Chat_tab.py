# Rag_Chat_tab.py
# Description: This file contains the code for the RAG Chat tab in the Gradio UI
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.RAG.RAG_Library_2 import enhanced_rag_pipeline
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging
#
########################################################################################################################
#
# Functions:

def create_rag_tab():
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

    with gr.TabItem("RAG Search", visible=True):
        gr.Markdown("# Retrieval-Augmented Generation (RAG) Search")

        with gr.Row():
            with gr.Column():
                search_query = gr.Textbox(label="Enter your question", placeholder="What would you like to know?")
                # Refactored API selection dropdown
                api_choice = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Response (Optional)"
                )
                search_button = gr.Button("Search")
                keyword_filtering_checkbox = gr.Checkbox(label="Enable Keyword Filtering", value=False)
                keywords_input = gr.Textbox(
                    label="Enter keywords (comma-separated)",
                    value="keyword1, keyword2, ...",
                    visible=False
                )
                keyword_instructions = gr.Markdown(
                    "Enter comma-separated keywords to filter your search results.",
                    visible=False
                )

            with gr.Column():
                result_output = gr.Textbox(label="Answer", lines=10)
                context_output = gr.Textbox(label="Context", lines=10, visible=True)

        def toggle_keyword_filtering(checkbox_value):
            return {
                keywords_input: gr.update(visible=checkbox_value),
                keyword_instructions: gr.update(visible=checkbox_value)
            }

        keyword_filtering_checkbox.change(
            toggle_keyword_filtering,
            inputs=[keyword_filtering_checkbox],
            outputs=[keywords_input, keyword_instructions]
        )

        def perform_rag_search(query, keywords, api_choice):
            if keywords == "keyword1, keyword2, ...":
                keywords = None
            result = enhanced_rag_pipeline(query, api_choice, keywords)
            return result['answer'], result['context']

        search_button.click(perform_rag_search, inputs=[search_query, keywords_input, api_choice], outputs=[result_output, context_output])



#
# End of file
########################################################################################################################

