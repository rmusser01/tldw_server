# Local_LLM_ollama.py
# Description: This module provides functionality to interact with the Ollama API for managing and serving local LLM models.
#
# Imports
import platform
import subprocess
import psutil
import os
import signal
import threading
import shutil
# 3rd-Party Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Utils.Utils import logging
#
#######################################################################################################################
#
# Functions:

# Configure Logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("app.log"),
#         logging.StreamHandler()
#     ]
# )

def is_ollama_installed():
    """
    Checks if the 'ollama' executable is available in the system's PATH.
    Returns True if installed, False otherwise.
    """
    return shutil.which('ollama') is not None

def get_ollama_models():
    """
    Retrieves available Ollama models by executing 'ollama list'.
    Returns a list of model names or an empty list if an error occurs.
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, timeout=10)
        models = result.stdout.strip().split('\n')[1:]  # Skip header
        model_names = [model.split()[0] for model in models if model.strip()]
        logging.debug(f"Available Ollama models: {model_names}")
        return model_names
    except FileNotFoundError:
        logging.error("Ollama executable not found. Please ensure Ollama is installed and in your PATH.")
        return []
    except subprocess.TimeoutExpired:
        logging.error("Ollama 'list' command timed out.")
        return []
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing Ollama 'list': {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in get_ollama_models: {e}")
        return []

def pull_ollama_model(model_name):
    """
    Pulls the specified Ollama model if Ollama is installed.
    """
    if not is_ollama_installed():
        logging.error("Ollama is not installed.")
        return "Failed to pull model: Ollama is not installed or not in your PATH."

    try:
        subprocess.run(['ollama', 'pull', model_name], check=True, timeout=300)  # Adjust timeout as needed
        logging.info(f"Successfully pulled model: {model_name}")
        return f"Successfully pulled model: {model_name}"
    except subprocess.TimeoutExpired:
        logging.error(f"Pulling model '{model_name}' timed out.")
        return f"Failed to pull model '{model_name}': Operation timed out."
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to pull model '{model_name}': {e}")
        return f"Failed to pull model '{model_name}': {e}"
    except FileNotFoundError:
        logging.error("Ollama executable not found. Please ensure Ollama is installed and in your PATH.")
        return "Failed to pull model: Ollama executable not found."
    except Exception as e:
        logging.error(f"Unexpected error in pull_ollama_model: {e}")
        return f"Failed to pull model '{model_name}': {e}"

def serve_ollama_model(model_name, port):
    """
    Serves the specified Ollama model on the given port if Ollama is installed.
    """
    if not is_ollama_installed():
        logging.error("Ollama is not installed.")
        return "Error: Ollama is not installed or not in your PATH."

    try:
        # Check if a server is already running on the specified port
        for conn in psutil.net_connections():
            if conn.laddr.port == int(port):
                logging.warning(f"Port {port} is already in use.")
                return f"Error: Port {port} is already in use. Please choose a different port."

        # Start the Ollama server
        cmd = ['ollama', 'serve', model_name, '--port', str(port)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Started Ollama server for model '{model_name}' on port {port}. PID: {process.pid}")
        return f"Started Ollama server for model '{model_name}' on port {port}. Process ID: {process.pid}"
    except FileNotFoundError:
        logging.error("Ollama executable not found.")
        return "Error: Ollama executable not found. Please ensure Ollama is installed and in your PATH."
    except Exception as e:
        logging.error(f"Error starting Ollama server: {e}")
        return f"Error starting Ollama server: {e}"

def stop_ollama_server(pid):
    """
    Stops the Ollama server with the specified process ID if Ollama is installed.
    """
    if not is_ollama_installed():
        logging.error("Ollama is not installed.")
        return "Error: Ollama is not installed or not in your PATH."

    try:
        if platform.system() == "Windows":
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
        elif platform.system() in ["Linux", "Darwin"]:
            os.kill(pid, signal.SIGTERM)
        logging.info(f"Stopped Ollama server with PID {pid}")
        return f"Stopped Ollama server with PID {pid}"
    except ProcessLookupError:
        logging.warning(f"No process found with PID {pid}")
        return f"No process found with PID {pid}"
    except Exception as e:
        logging.error(f"Error stopping Ollama server: {e}")
        return f"Error stopping Ollama server: {e}"

def create_ollama_tab():
    """
    Creates the Ollama Model Serving tab in the Gradio interface with lazy loading.
    """
    ollama_installed = is_ollama_installed()

    with gr.Tab("Ollama Model Serving"):
        if not ollama_installed:
            gr.Markdown(
                "# Ollama Model Serving\n\n"
                "**Ollama is not installed or not found in your PATH. Please install Ollama to use this feature.**"
            )
            return  # Exit early, no need to add further components

        gr.Markdown("# Ollama Model Serving")

        with gr.Row():
            # Initialize Dropdowns with placeholders
            model_list = gr.Dropdown(
                label="Available Models",
                choices=["Click 'Refresh Model List' to load models"],
                value="Click 'Refresh Model List' to load models"
            )
            refresh_button = gr.Button("Refresh Model List")

        with gr.Row():
            new_model_name = gr.Textbox(label="Model to Pull", placeholder="Enter model name")
            pull_button = gr.Button("Pull Model")

        pull_output = gr.Textbox(label="Pull Status")

        with gr.Row():
            serve_model = gr.Dropdown(
                label="Model to Serve",
                choices=["Click 'Refresh Model List' to load models"],
                value="Click 'Refresh Model List' to load models"
            )
            port = gr.Number(label="Port", value=11434, precision=0)
            serve_button = gr.Button("Start Server")

        serve_output = gr.Textbox(label="Server Status")

        with gr.Row():
            pid = gr.Number(label="Server Process ID (Enter the PID to stop)", precision=0)
            stop_button = gr.Button("Stop Server")

        stop_output = gr.Textbox(label="Stop Status")

        def update_model_lists():
            """
            Retrieves the list of available Ollama models and updates the dropdowns.
            """
            models = get_ollama_models()
            if models:
                return gr.update(choices=models, value=models[0]), gr.update(choices=models, value=models[0])
            else:
                return gr.update(choices=["No models found"], value="No models found"), gr.update(choices=["No models found"], value="No models found")

        def async_update_model_lists():
            """
            Asynchronously updates the model lists to prevent blocking.
            """
            def task():
                choices1, choices2 = update_model_lists()
                model_list.update(choices=choices1['choices'], value=choices1.get('value'))
                serve_model.update(choices=choices2['choices'], value=choices2.get('value'))
            threading.Thread(target=task).start()

        # Bind the refresh button to the asynchronous update function
        refresh_button.click(fn=async_update_model_lists, inputs=[], outputs=[])

        # Bind the pull, serve, and stop buttons to their respective functions
        pull_button.click(fn=pull_ollama_model, inputs=[new_model_name], outputs=[pull_output])
        serve_button.click(fn=serve_ollama_model, inputs=[serve_model, port], outputs=[serve_output])
        stop_button.click(fn=stop_ollama_server, inputs=[pid], outputs=[stop_output])
#
# End of Local_LLM_ollama.py
#######################################################################################################################
