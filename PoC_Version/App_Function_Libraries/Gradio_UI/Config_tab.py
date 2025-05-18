import gradio as gr
import configparser

# FIXME
CONFIG_PATH = './PoC_Version/Config_Files/config.txt'

def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config

def save_config(config):
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)

def get_config_as_text():
    with open(CONFIG_PATH, 'r') as file:
        content = file.read()
    return content, "Config refreshed successfully"

def save_config_from_text(text):
    with open(CONFIG_PATH, 'w') as file:
        file.write(text)
    return "Config saved successfully"


def create_config_editor_tab():
    with gr.TabItem("Edit Config", visible=True):
        gr.Markdown("# Edit Configuration File")

        with gr.Row():
            with gr.Column():
                refresh_button = gr.Button("Refresh Config")

            with gr.Column():
                config_text = gr.TextArea(label="Full Config", lines=30)
                save_text_button = gr.Button("Save Config")

        with gr.Row():
            output = gr.Textbox(label="Output")

        # Event handlers
        refresh_button.click(get_config_as_text, inputs=[], outputs=[config_text, output])

        config_text.change(lambda: None, None, None)  # Dummy handler to enable changes
        save_text_button.click(save_config_from_text, inputs=[config_text], outputs=[output])

        # Initialize the interface
        config_text.value = get_config_as_text()[0]  # Only set the config text, not the output message

    return refresh_button, config_text, save_text_button, output
