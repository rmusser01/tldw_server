# Media_wiki_tab.py
# Description: Gradio UI snippet that allows users to import a MediaWiki XML dump file into the application.
#
# Imports
import os
from threading import Thread
#
# 3rd-party Imports
import gradio as gr
import yaml
from ruamel.yaml import YAML
#
# Local Imports
from PoC_Version.App_Function_Libraries.MediaWiki.Media_Wiki import import_mediawiki_dump, media_wiki_import_config
#
#######################################################################################################################
#
# Create MediaWiki Import Tab

def create_mediawiki_import_tab():
    with gr.Tab("MediaWiki Import"):
        gr.Markdown("# Import MediaWiki Dump")
        with gr.Row():
            with gr.Column():
                file_path = gr.File(label="MediaWiki XML Dump File")
                wiki_name = gr.Textbox(label="Wiki Name", placeholder="Enter a unique name for this wiki")
                namespaces = gr.Textbox(label="Namespaces (comma-separated integers, leave empty for all)")
                skip_redirects = gr.Checkbox(label="Skip Redirects", value=True)
                single_item = gr.Checkbox(label="Import as Single Item", value=False)
                chunk_method = gr.Dropdown(
                    choices=["sentences", "words", "paragraphs", "tokens"],
                    value="sentences",
                    label="Chunking Method"
                )
                # FIXME - add API selection dropdown + Analysis/Summarization options
                # Refactored API selection dropdown
                # api_name_input = gr.Dropdown(
                #     choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                #     value=default_value,
                #     label="API for Summarization (Optional)"
                # )
                chunk_size = gr.Slider(minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size")
                chunk_overlap = gr.Slider(minimum=0, maximum=500, value=100, step=10, label="Chunk Overlap")
                # FIXME - Add checkbox for 'Enable Summarization upon ingestion' for API summarization of chunks
                # api_endpoint = gr.Dropdown(label="Select API Endpoint",
                #                            choices=["Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek",
                #                                     "Mistral", "OpenRouter",
                #                                     "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama",
                #                                     "HuggingFace"])
                # api_key = gr.Textbox(label="API Key (if required)", type="password")
                import_button = gr.Button("Import MediaWiki Dump")
                cancel_button = gr.Button("Cancel Import", visible=False)
            with gr.Column():
                output = gr.Markdown(label="Import Status")
                progress_bar = gr.Progress()

        def validate_inputs(file_path, wiki_name, namespaces):
            if not file_path:
                return "Please select a MediaWiki XML dump file."
            if not wiki_name:
                return "Please enter a name for the wiki."
            if namespaces:
                try:
                    [int(ns.strip()) for ns in namespaces.split(',')]
                except ValueError:
                    return "Invalid namespaces. Please enter comma-separated integers."
            return None

        def check_file_size(file_path):
            max_size_mb = 1000  # 1 GB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return f"Warning: The selected file is {file_size_mb:.2f} MB. Importing large files may take a long time."
            return None

        import_thread = None
        cancel_flag = False

        def run_import(file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                       chunk_overlap, progress=gr.Progress()):#, api_endpoint=None, api_key=None):
            validation_error = validate_inputs(file_path, wiki_name, namespaces)
            if validation_error:
                return gr.update(), gr.update(), validation_error

            file_size_warning = check_file_size(file_path.name)
            status_text = "# MediaWiki Import Process\n\n## Initializing\n- Starting import process...\n"
            if file_size_warning:
                status_text += f"- {file_size_warning}\n"

            chunk_options = {
                'method': chunk_method,
                'max_size': chunk_size,
                'overlap': chunk_overlap,
                'adaptive': True,
                'language': 'en'
            }
            namespaces_list = [int(ns.strip()) for ns in namespaces.split(',')] if namespaces else None

            pages_processed = 0

            try:
                for progress_info in import_mediawiki_dump(
                        file_path=file_path.name,
                        wiki_name=wiki_name,
                        namespaces=namespaces_list,
                        skip_redirects=skip_redirects,
                        chunk_options=chunk_options,
                        single_item=single_item,
                        progress_callback=progress,
#                        api_name=api_endpoint,
#                        api_key=api_key
                ):
                    if progress_info.startswith("Found"):
                        status_text += f"\n## Parsing\n- {progress_info}\n"
                    elif progress_info.startswith("Processed page"):
                        pages_processed += 1
                        if pages_processed % 10 == 0:  # Update every 10 pages to avoid too frequent updates
                            status_text += f"- {progress_info}\n"
                    elif progress_info.startswith("Successfully imported"):
                        status_text += f"\n## Completed\n- {progress_info}\n- Total pages processed: {pages_processed}"
                    else:
                        status_text += f"- {progress_info}\n"

                    yield gr.update(), gr.update(), status_text

                status_text += "\n## Import Process Completed Successfully"
            except Exception as e:
                status_text += f"\n## Error\n- An error occurred during the import process: {str(e)}"

            yield gr.update(visible=False), gr.update(visible=True), status_text

        def start_import(*args):
            nonlocal import_thread
            import_thread = Thread(target=run_import, args=args)
            import_thread.start()
            return gr.update(visible=True), gr.update(visible=False), gr.update(
                value="Import process started. Please wait...")

        def cancel_import():
            nonlocal cancel_flag
            cancel_flag = True
            return gr.update(visible=False), gr.update(visible=True)

        import_button.click(
            run_import,
            inputs=[file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size,
                    chunk_overlap],#, api_endpoint, api_key],
            outputs=[cancel_button, import_button, output]
        )

        cancel_button.click(
            cancel_import,
            outputs=[cancel_button, import_button]
        )

    return file_path, wiki_name, namespaces, skip_redirects, single_item, chunk_method, chunk_size, chunk_overlap, import_button, output


class PreservedTokenSafeDumper(yaml.SafeDumper):
    def represent_scalar(self, tag, value, style=None):
        if style is None and isinstance(value, str) and '\n' in value:
            style = '|'
        return super().represent_scalar(tag, value, style)


def update_yaml_file(file_path, updates):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    def format_value(value):
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return '[' + ', '.join(map(str, value)) + ']'
        else:
            return f"'{value}'"

    def update_line(line, updates, prefix=''):
        for key, value in updates.items():
            full_key = f"{prefix}{key}:" if prefix else f"{key}:"
            if line.strip().startswith(full_key):
                indentation = line[:line.index(full_key)]
                if isinstance(value, dict):
                    return line  # Keep the line as is for nested structures
                else:
                    return f"{indentation}{full_key} {format_value(value)}\n"
        return line

    updated_lines = []
    current_prefix = ''
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                current_prefix = ''
            elif ':' in stripped and not stripped.endswith(':'):
                current_prefix = '.'.join(current_prefix.split('.')[:-1]) + '.' if current_prefix else ''

            updated_line = update_line(line, updates, current_prefix)

            if updated_line == line and ':' in stripped and stripped.endswith(':'):
                key = stripped[:-1].strip()
                if current_prefix:
                    current_prefix += f"{key}."
                else:
                    current_prefix = f"{key}."

            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

#
#
#######################################################################################################################
#
# Config tab

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

def load_config():
    config_path = os.path.join('Config_Files', 'mediawiki_import_config.yaml')
    with open(config_path, 'r') as file:
        return yaml.load(file)

def save_config(updated_config):
    config_path = os.path.join('Config_Files', 'mediawiki_import_config.yaml')
    config = load_config()


def create_mediawiki_config_tab():
    with gr.TabItem("MediaWiki Import Configuration", visible=True):
        gr.Markdown("# MediaWiki Import Configuration (Broken currently/doesn't work)")
        with gr.Row():
            with gr.Column():
                namespaces = gr.Textbox(label="Default Namespaces (comma-separated integers)",
                                        value=','.join(map(str, media_wiki_import_config['import']['default_namespaces'])))
                skip_redirects = gr.Checkbox(label="Skip Redirects by Default",
                                             value=media_wiki_import_config['import']['default_skip_redirects'])
                single_item = gr.Checkbox(label="Import as Single Item by Default",
                                          value=media_wiki_import_config['import']['single_item_default'])
                batch_size = gr.Number(value=media_wiki_import_config['import']['batch_size'], label="Batch Size")

                chunk_method = gr.Dropdown(
                    choices=media_wiki_import_config['chunking']['methods'],
                    value=media_wiki_import_config['chunking']['default_method'],
                    label="Default Chunking Method"
                )
                chunk_size = gr.Slider(minimum=100, maximum=2000, value=media_wiki_import_config['chunking']['default_size'], step=100,
                                       label="Default Chunk Size")
                chunk_overlap = gr.Slider(minimum=0, maximum=500, value=media_wiki_import_config['chunking']['default_overlap'], step=10,
                                          label="Default Chunk Overlap")

            with gr.Column():
                max_workers = gr.Slider(minimum=1, maximum=16, value=media_wiki_import_config['processing']['max_workers'], step=1,
                                        label="Max Worker Threads")

                embedding_provider = gr.Dropdown(
                    choices=['openai', 'local', 'huggingface'],
                    value=media_wiki_import_config['embeddings']['provider'],
                    label="Embedding Provider"
                )
                embedding_model = gr.Textbox(label="Embedding Model", value=media_wiki_import_config['embeddings']['model'])
                api_key = gr.Textbox(label="API Key (if required)", type="password",
                                     value=media_wiki_import_config['embeddings'].get('api_key', ''))
                local_embedding_url = gr.Textbox(label="Local Embedding URL",
                                                 value=media_wiki_import_config['embeddings'].get('local_url', ''))

                checkpoints_enabled = gr.Checkbox(label="Enable Checkpoints", value=media_wiki_import_config['checkpoints']['enabled'])
                checkpoint_directory = gr.Textbox(label="Checkpoint Directory", value=media_wiki_import_config['checkpoints']['directory'])

                max_retries = gr.Number(value=media_wiki_import_config['error_handling']['max_retries'], label="Max Retries")
                retry_delay = gr.Number(value=media_wiki_import_config['error_handling']['retry_delay'], label="Retry Delay (seconds)")

        save_config_button = gr.Button("Save Configuration")
        config_output = gr.Markdown(label="Configuration Status")

        def update_config_from_ui(namespaces, skip_redirects, single_item, batch_size, chunk_method, chunk_size,
                                  chunk_overlap, max_workers, embedding_provider, embedding_model, api_key,
                                  local_embedding_url, checkpoints_enabled, checkpoint_directory, max_retries,
                                  retry_delay):
            current_config = load_config()
            updated_config = {}

            if namespaces != ','.join(map(str, current_config['import']['default_namespaces'])):
                updated_config.setdefault('import', {})['default_namespaces'] = [int(ns.strip()) for ns in
                                                                                 namespaces.split(',') if ns.strip()]
            if skip_redirects != current_config['import']['default_skip_redirects']:
                updated_config.setdefault('import', {})['default_skip_redirects'] = skip_redirects
            if single_item != current_config['import']['single_item_default']:
                updated_config.setdefault('import', {})['single_item_default'] = single_item
            if int(batch_size) != current_config['import']['batch_size']:
                updated_config.setdefault('import', {})['batch_size'] = int(batch_size)
            if chunk_method != current_config['chunking']['default_method']:
                updated_config.setdefault('chunking', {})['default_method'] = chunk_method
            if int(chunk_size) != current_config['chunking']['default_size']:
                updated_config.setdefault('chunking', {})['default_size'] = int(chunk_size)
            if int(chunk_overlap) != current_config['chunking']['default_overlap']:
                updated_config.setdefault('chunking', {})['default_overlap'] = int(chunk_overlap)
            if int(max_workers) != current_config['processing']['max_workers']:
                updated_config.setdefault('processing', {})['max_workers'] = int(max_workers)
            if embedding_provider != current_config['embeddings']['provider']:
                updated_config.setdefault('embeddings', {})['provider'] = embedding_provider
            if embedding_model != current_config['embeddings']['model']:
                updated_config.setdefault('embeddings', {})['model'] = embedding_model
            if api_key != current_config['embeddings'].get('api_key', ''):
                updated_config.setdefault('embeddings', {})['api_key'] = api_key
            if local_embedding_url != current_config['embeddings'].get('local_url', ''):
                updated_config.setdefault('embeddings', {})['local_url'] = local_embedding_url
            if checkpoints_enabled != current_config['checkpoints']['enabled']:
                updated_config.setdefault('checkpoints', {})['enabled'] = checkpoints_enabled
            if checkpoint_directory != current_config['checkpoints']['directory']:
                updated_config.setdefault('checkpoints', {})['directory'] = checkpoint_directory
            if int(max_retries) != current_config['error_handling']['max_retries']:
                updated_config.setdefault('error_handling', {})['max_retries'] = int(max_retries)
            if int(retry_delay) != current_config['error_handling']['retry_delay']:
                updated_config.setdefault('error_handling', {})['retry_delay'] = int(retry_delay)

            return updated_config

        def save_config_callback(*args):
            updated_config = update_config_from_ui(*args)
            save_config(updated_config)
            return "Configuration saved successfully."

        save_config_button.click(
            save_config_callback,
            inputs=[namespaces, skip_redirects, single_item, batch_size, chunk_method, chunk_size,
                    chunk_overlap, max_workers, embedding_provider, embedding_model, api_key,
                    local_embedding_url, checkpoints_enabled, checkpoint_directory, max_retries, retry_delay],
            outputs=config_output
        )

    return namespaces, skip_redirects, single_item, batch_size, chunk_method, chunk_size, chunk_overlap, max_workers, \
           embedding_provider, embedding_model, api_key, local_embedding_url, checkpoints_enabled, checkpoint_directory, \
           max_retries, retry_delay, save_config_button, config_output

#
# End of MediaWiki Import Tab
#######################################################################################################################
