# Llamafile_tab.py
# Description: Gradio interface for configuring and launching Llamafile with Local LLMs

# Imports
import os
from typing import Tuple, Optional
#
# 3rd-party imports
import gradio as gr
#
# Local imports
from PoC_Version.App_Function_Libraries.Local_LLM.Local_LLM_Inference_Engine_Lib import (
    download_llm_model,
    llm_models,
    start_llamafile,
    get_gguf_llamafile_files
)
from PoC_Version.App_Function_Libraries.Utils.Utils import logging
#
#######################################################################################################################
#
# Functions:

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

def create_chat_with_llamafile_tab():
    # Function to update model path based on selection
    def on_local_model_change(selected_model: str, search_directory: str) -> str:
        if selected_model and isinstance(search_directory, str):
            model_path = os.path.abspath(os.path.join(search_directory, selected_model))
            logging.debug(f"Selected model path: {model_path}")  # Debug print for selected model path
            return model_path
        return "Invalid selection or directory."

    # Function to update the dropdown with available models
    def update_dropdowns(search_directory: str) -> Tuple[dict, str]:
        logging.debug(f"User-entered directory: {search_directory}")  # Debug print for directory
        if not os.path.isdir(search_directory):
            logging.debug(f"Directory does not exist: {search_directory}")  # Debug print for non-existing directory
            return gr.update(choices=[], value=None), "Directory does not exist."

        try:
            logging.debug(f"Directory exists: {search_directory}, scanning for files...")  # Confirm directory exists
            model_files = get_gguf_llamafile_files(search_directory)
            logging.debug("Completed scanning for model files.")
        except Exception as e:
            logging.error(f"Error scanning directory: {e}")
            return gr.update(choices=[], value=None), f"Error scanning directory: {e}"

        if not model_files:
            logging.debug(f"No model files found in {search_directory}")  # Debug print for no files found
            return gr.update(choices=[], value=None), "No model files found in the specified directory."

        # Update the dropdown choices with the model files found
        logging.debug(f"Models loaded from {search_directory}: {model_files}")  # Debug: Print model files loaded
        return gr.update(choices=model_files, value=None), f"Models loaded from {search_directory}."



    def download_preset_model(selected_model: str) -> Tuple[str, str]:
        """
        Downloads the selected preset model.

        Args:
            selected_model (str): The key of the selected preset model.

        Returns:
            Tuple[str, str]: Status message and the path to the downloaded model.
        """
        model_info = llm_models.get(selected_model)
        if not model_info:
            return "Invalid model selection.", ""

        try:
            model_path = download_llm_model(
                model_name=model_info["name"],
                model_url=model_info["url"],
                model_filename=model_info["filename"],
                model_hash=model_info["hash"]
            )
            return f"Model '{model_info['name']}' downloaded successfully.", model_path
        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            return f"Failed to download model: {e}", ""

    with gr.TabItem("Local LLM with Llamafile", visible=True):
        gr.Markdown("# Settings for Llamafile")

        with gr.Row():
            with gr.Column():
                am_noob = gr.Checkbox(label="Enable Sane Defaults", value=False, visible=True)
                advanced_mode_toggle = gr.Checkbox(label="Advanced Mode - Show All Settings", value=False)
                # Advanced Inputs
                verbose_checked = gr.Checkbox(label="Enable Verbose Output", value=False, visible=False)
                threads_checked = gr.Checkbox(label="Set CPU Threads", value=False, visible=False)
                threads_value = gr.Number(label="Number of CPU Threads", value=None, precision=0, visible=False)
                threads_batched_checked = gr.Checkbox(label="Enable Batched Inference", value=False, visible=False)
                threads_batched_value = gr.Number(label="Batch Size for Inference", value=None, precision=0, visible=False)
                model_alias_checked = gr.Checkbox(label="Set Model Alias", value=False, visible=False)
                model_alias_value = gr.Textbox(label="Model Alias", value="", visible=False)
                ctx_size_checked = gr.Checkbox(label="Set Prompt Context Size", value=False, visible=False)
                ctx_size_value = gr.Number(label="Prompt Context Size", value=8124, precision=0, visible=False)
                ngl_checked = gr.Checkbox(label="Enable GPU Layers", value=False, visible=True)
                ngl_value = gr.Number(label="Number of GPU Layers", value=None, precision=0, visible=True)
                batch_size_checked = gr.Checkbox(label="Set Batch Size", value=False, visible=False)
                batch_size_value = gr.Number(label="Batch Size", value=512, visible=False)
                memory_f32_checked = gr.Checkbox(label="Use 32-bit Floating Point", value=False, visible=False)
                numa_checked = gr.Checkbox(label="Enable NUMA", value=False, visible=False)
                server_timeout_value = gr.Number(label="Server Timeout", value=600, precision=0, visible=False)
                host_checked = gr.Checkbox(label="Set IP to Listen On", value=False, visible=False)
                host_value = gr.Textbox(label="Host IP Address", value="", visible=False)
                port_checked = gr.Checkbox(label="Set Server Port", value=False, visible=False)
                port_value = gr.Number(label="Port Number", value=8080, precision=0, visible=False)
                api_key_checked = gr.Checkbox(label="Set API Key", value=False, visible=False)
                api_key_value = gr.Textbox(label="API Key", value="", visible=False)
                http_threads_checked = gr.Checkbox(label="Set HTTP Server Threads", value=False, visible=False)
                http_threads_value = gr.Number(label="Number of HTTP Server Threads", value=None, precision=0, visible=False)
                hf_repo_checked = gr.Checkbox(label="Use Huggingface Repo Model", value=False, visible=False)
                hf_repo_value = gr.Textbox(label="Huggingface Repo Name", value="", visible=False)
                hf_file_checked = gr.Checkbox(label="Set Huggingface Model File", value=False, visible=False)
                hf_file_value = gr.Textbox(label="Huggingface Model File", value="", visible=False)

            with gr.Column():
                # Model Selection Section
                gr.Markdown("## Model Selection")

                # Option 1: Select from Local Filesystem
                with gr.Row():
                    search_directory = gr.Textbox(
                        label="Model Directory",
                        placeholder="Enter directory path (currently './Models')",
                        value=MODELS_DIR,
                        interactive=True
                    )

                # Initial population of local models
                initial_dropdown_update, _ = update_dropdowns(MODELS_DIR)
                logging.debug(f"Scanning directory: {MODELS_DIR}")
                refresh_button = gr.Button("Refresh Models")
                local_model_dropdown = gr.Dropdown(
                    label="Select Model from Directory",
                    choices=initial_dropdown_update["choices"],
                    value=None
                )
                # Display selected model path
                model_value = gr.Textbox(label="Selected Model File Path", value="", interactive=False)

                # Option 2: Download Preset Models
                gr.Markdown("## Download Preset Models")

                preset_model_dropdown = gr.Dropdown(
                    label="Select a Preset Model",
                    choices=list(llm_models.keys()),
                    value=None,
                    interactive=True,
                    info="Choose a preset model to download."
                )
                download_preset_button = gr.Button("Download Selected Preset")

        with gr.Row():
            with gr.Column():
                start_button = gr.Button("Start Llamafile")
                stop_button = gr.Button("Stop Llamafile (doesn't work)")
                output_display = gr.Markdown()


        # Show/hide advanced inputs based on toggle
        def update_visibility(show_advanced: bool):
            components = [
                verbose_checked, threads_checked, threads_value,
                http_threads_checked, http_threads_value,
                hf_repo_checked, hf_repo_value,
                hf_file_checked, hf_file_value,
                ctx_size_checked, ctx_size_value,
                ngl_checked, ngl_value,
                host_checked, host_value,
                port_checked, port_value
            ]
            return [gr.update(visible=show_advanced) for _ in components]

        def on_start_button_click(
                am_noob: bool,
                verbose_checked: bool,
                threads_checked: bool,
                threads_value: Optional[int],
                threads_batched_checked: bool,
                threads_batched_value: Optional[int],
                model_alias_checked: bool,
                model_alias_value: str,
                http_threads_checked: bool,
                http_threads_value: Optional[int],
                model_value: str,
                hf_repo_checked: bool,
                hf_repo_value: str,
                hf_file_checked: bool,
                hf_file_value: str,
                ctx_size_checked: bool,
                ctx_size_value: Optional[int],
                ngl_checked: bool,
                ngl_value: Optional[int],
                batch_size_checked: bool,
                batch_size_value: Optional[int],
                memory_f32_checked: bool,
                numa_checked: bool,
                server_timeout_value: Optional[int],
                host_checked: bool,
                host_value: str,
                port_checked: bool,
                port_value: Optional[int],
                api_key_checked: bool,
                api_key_value: str
        ) -> str:
            """
            Event handler for the Start Llamafile button.
            """
            try:
                result = start_llamafile(
                    am_noob,
                    verbose_checked,
                    threads_checked,
                    threads_value,
                    threads_batched_checked,
                    threads_batched_value,
                    model_alias_checked,
                    model_alias_value,
                    http_threads_checked,
                    http_threads_value,
                    model_value,
                    hf_repo_checked,
                    hf_repo_value,
                    hf_file_checked,
                    hf_file_value,
                    ctx_size_checked,
                    ctx_size_value,
                    ngl_checked,
                    ngl_value,
                    batch_size_checked,
                    batch_size_value,
                    memory_f32_checked,
                    numa_checked,
                    server_timeout_value,
                    host_checked,
                    host_value,
                    port_checked,
                    port_value,
                    api_key_checked,
                    api_key_value
                )
                return result
            except Exception as e:
                logging.error(f"Error starting Llamafile: {e}")
                return f"Failed to start Llamafile: {e}"

        advanced_mode_toggle.change(
            fn=update_visibility,
            inputs=[advanced_mode_toggle],
            outputs=[
                verbose_checked, threads_checked, threads_value,
                http_threads_checked, http_threads_value,
                hf_repo_checked, hf_repo_value,
                hf_file_checked, hf_file_value,
                ctx_size_checked, ctx_size_value,
                ngl_checked, ngl_value,
                host_checked, host_value,
                port_checked, port_value
            ]
        )

        start_button.click(
            fn=on_start_button_click,
            inputs=[
                am_noob,
                verbose_checked,
                threads_checked,
                threads_value,
                threads_batched_checked,
                threads_batched_value,
                model_alias_checked,
                model_alias_value,
                http_threads_checked,
                http_threads_value,
                model_value,
                hf_repo_checked,
                hf_repo_value,
                hf_file_checked,
                hf_file_value,
                ctx_size_checked,
                ctx_size_value,
                ngl_checked,
                ngl_value,
                batch_size_checked,
                batch_size_value,
                memory_f32_checked,
                numa_checked,
                server_timeout_value,
                host_checked,
                host_value,
                port_checked,
                port_value,
                api_key_checked,
                api_key_value
            ],
            outputs=output_display
        )

        download_preset_button.click(
            fn=download_preset_model,
            inputs=[preset_model_dropdown],
            outputs=[output_display, model_value]
        )

        # Click event for refreshing models
        refresh_button.click(
            fn=update_dropdowns,
            inputs=[search_directory],  # Ensure that the directory path (string) is passed
            outputs=[local_model_dropdown, output_display]  # Update dropdown and status
        )

        # Event to update model_value when a model is selected from the dropdown
        local_model_dropdown.change(
            fn=on_local_model_change,  # Function that calculates the model path
            inputs=[local_model_dropdown, search_directory],  # Inputs: selected model and directory
            outputs=[model_value]  # Output: Update the model_value textbox with the selected model path
        )

#
#
#######################################################################################################################