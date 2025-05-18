###################################################################################################
# Evaluations_Benchmarks_tab.py - Gradio code for G-Eval testing
# We will use the G-Eval API to evaluate the quality of the generated summaries.
import logging

import gradio as gr
from PoC_Version.App_Function_Libraries.Benchmarks_Evaluations.ms_g_eval import run_geval
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name


def create_geval_tab():
    with gr.Tab("G-Eval", visible=True):
        gr.Markdown("# G-Eval Summarization Evaluation")
        with gr.Row():
            with gr.Column():
                document_input = gr.Textbox(label="Source Document", lines=10)
                summary_input = gr.Textbox(label="Summary", lines=5)
                api_name_input = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Cohere", "Groq", "OpenRouter", "DeepSeek", "HuggingFace", "Mistral", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "Local-LLM", "Ollama"],
                    label="Select API"
                )
                api_key_input = gr.Textbox(label="API Key (if required)", type="password")
                evaluate_button = gr.Button("Evaluate Summary")
            with gr.Column():
                output = gr.Textbox(label="Evaluation Results", lines=10)

        evaluate_button.click(
            fn=run_geval,
            inputs=[document_input, summary_input, api_name_input, api_key_input],
            outputs=output
        )

    return document_input, summary_input, api_name_input, api_key_input, evaluate_button, output


def create_infinite_bench_tab():
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
    with gr.Tab("Infinite Bench", visible=True):
        gr.Markdown("# Infinite Bench Evaluation (Coming Soon)")
        with gr.Row():
            with gr.Column():
                # Refactored API selection dropdown
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Summarization (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key (if required)", type="password")
                evaluate_button = gr.Button("Evaluate Summary")
            with gr.Column():
                output = gr.Textbox(label="Evaluation Results", lines=10)

        # evaluate_button.click(
        #     fn=run_geval,
        #     inputs=[api_name_input, api_key_input],
        #     outputs=output
        # )

    return api_name_input, api_key_input, evaluate_button, output


# If you want to run this as a standalone Gradio app
if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_geval_tab()
    demo.launch()
