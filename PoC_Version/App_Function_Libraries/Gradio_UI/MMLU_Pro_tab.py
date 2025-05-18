# MMLU_Pro_tab.py
# is a library that contains the Gradio UI code for the MMLU-Pro benchmarking tool.
#
##############################################################################################################
# Imports
import os

import gradio as gr
import logging
#
# External Imports
from tqdm import tqdm
# Local Imports
from PoC_Version.App_Function_Libraries.Benchmarks_Evaluations.MMLU_Pro.MMLU_Pro_rewritten import (
    load_mmlu_pro, run_mmlu_pro_benchmark, mmlu_pro_main, load_mmlu_pro_config
)
#
##############################################################################################################
#
# Functions:

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_categories():
    """Fetch categories using the dataset loader from MMLU_Pro_rewritten.py"""
    try:
        test_data, _ = load_mmlu_pro()  # Use the function from MMLU_Pro_rewritten.py
        return list(test_data.keys())  # Return the categories from the test dataset
    except Exception as e:
        logger.error(f"Failed to load categories: {e}")
        return ["Error loading categories"]


def load_categories():
    """Helper function to return the categories for the Gradio dropdown."""
    categories = get_categories()  # Fetch categories from the dataset
    if categories:
        return gr.update(choices=categories, value=categories[0])  # Update dropdown with categories
    else:
        return gr.update(choices=["Error loading categories"], value="Error loading categories")


def run_benchmark_from_ui(url, api_key, model, timeout, category, parallel, verbosity, log_prompt):
    """Function to run the benchmark with parameters from the UI."""

    # Override config with UI parameters
    config = load_mmlu_pro_config(
        url=url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        categories=[category] if category else None,
        parallel=parallel,
        verbosity=verbosity,
        log_prompt=log_prompt
    )

    # Run the benchmarking process
    try:
        # Call the main benchmarking function
        mmlu_pro_main()

        # Assume the final report is generated in "eval_results" folder
        report_path = os.path.join("eval_results", config["server"]["model"].replace("/", "-"), "final_report.txt")

        # Read the final report
        with open(report_path, "r") as f:
            report = f.read()

        return report
    except Exception as e:
        logger.error(f"An error occurred during benchmark execution: {e}")
        return f"An error occurred during benchmark execution. Please check the logs for more information. Error: {str(e)}"


def create_mmlu_pro_tab():
    """Create the Gradio UI tab for MMLU-Pro Benchmark."""
    with gr.TabItem("MMLU-Pro Benchmark", visible=True):
        gr.Markdown("## Run MMLU-Pro Benchmark")

        with gr.Row():
            with gr.Column():
                # Inputs for the benchmark
                url = gr.Textbox(label="Server URL")
                api_key = gr.Textbox(label="API Key", type="password")
                model = gr.Textbox(label="Model Name")
                timeout = gr.Number(label="Timeout (seconds)", value=30)
                category = gr.Dropdown(label="Category", choices=["Load categories..."])
                load_categories_btn = gr.Button("Load Categories")
                parallel = gr.Slider(label="Parallel Requests", minimum=1, maximum=10, step=1, value=1)
                verbosity = gr.Slider(label="Verbosity Level", minimum=0, maximum=2, step=1, value=1)
                log_prompt = gr.Checkbox(label="Log Prompt")

            with gr.Column():
                # Run button and output display
                run_button = gr.Button("Run Benchmark")
                output = gr.Textbox(label="Benchmark Results", lines=20)

        # When "Load Categories" is clicked, load the categories into the dropdown
        load_categories_btn.click(
            load_categories,
            outputs=category
        )

        # When "Run Benchmark" is clicked, trigger the run_benchmark_from_ui function
        run_button.click(
            run_benchmark_from_ui,  # Use the function defined to run the benchmark
            inputs=[url, api_key, model, timeout, category, parallel, verbosity, log_prompt],
            outputs=output
        )

    return [url, api_key, model, timeout, category, parallel, verbosity, log_prompt, run_button, output]