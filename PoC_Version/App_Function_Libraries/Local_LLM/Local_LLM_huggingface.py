# Local_LLM_huggingface.py
# Description: This file contains the functions that are used for performing inference with and managing Hugging Face Transformers models
#
# Imports
import os
# 3rd-Party Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
# Local Imports
#
#######################################################################################################################
#
# Functions:

# FIXME: This function is not complete
# Setup proper path/configurations for the models
HF_MODELS_DIR = "models"

# FIXME: This function is not complete
def get_local_models():
    if not os.path.exists(HF_MODELS_DIR):
        os.makedirs(HF_MODELS_DIR)
    return [d for d in os.listdir(HF_MODELS_DIR) if os.path.isdir(os.path.join(HF_MODELS_DIR, d))]


def chat_with_transformers(user_message, system_message, model_name=None, model_path=None, max_new_tokens=100):
    pass

# Prepare the input as before
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

# 1: Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2: Apply the chat template
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
# Move the tokenized inputs to the same device the model is on (GPU/CPU)
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)

# 4: Generate text from the model
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print("Generated tokens:\n", outputs)

# 5: Decode the output back to a string
decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
print("Decoded output:\n", decoded_output)


# # Assuming models are stored in a 'models' directory
# MODELS_DIR = "models"
#
#
# def get_local_models():
#     if not os.path.exists(MODELS_DIR):
#         os.makedirs(MODELS_DIR)
#     return [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
#
#
# def download_model(model_name):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#
#         # Save the model and tokenizer
#         save_path = os.path.join(MODELS_DIR, model_name.split('/')[-1])
#         tokenizer.save_pretrained(save_path)
#         model.save_pretrained(save_path)
#
#         return f"Successfully downloaded model: {model_name}"
#     except Exception as e:
#         return f"Failed to download model: {str(e)}"
#
#
# def run_inference(model_name, prompt):
#     try:
#         model_path = os.path.join(MODELS_DIR, model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForCausalLM.from_pretrained(model_path)
#
#         # Use GPU if available
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model.to(device)
#
#         # Create a text-generation pipeline
#         text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
#
#         # Generate text
#         result = text_generator(prompt, max_length=100, num_return_sequences=1)
#
#         return result[0]['generated_text']
#     except Exception as e:
#         return f"Error running inference: {str(e)}"
#
#
# def create_huggingface_tab():
#     with gr.Tab("Hugging Face Transformers"):
#         gr.Markdown("# Hugging Face Transformers Model Management")
#
#         with gr.Row():
#             model_list = gr.Dropdown(label="Available Models", choices=get_local_models())
#             refresh_button = gr.Button("Refresh Model List")
#
#         with gr.Row():
#             new_model_name = gr.Textbox(label="Model to Download (e.g., 'gpt2' or 'EleutherAI/gpt-neo-1.3B')")
#             download_button = gr.Button("Download Model")
#
#         download_output = gr.Textbox(label="Download Status")
#
#         with gr.Row():
#             run_model = gr.Dropdown(label="Model to Run", choices=get_local_models())
#             prompt = gr.Textbox(label="Prompt")
#             run_button = gr.Button("Run Inference")
#
#         run_output = gr.Textbox(label="Model Output")
#
#         def update_model_lists():
#             models = get_local_models()
#             return gr.update(choices=models), gr.update(choices=models)
#
#         refresh_button.click(update_model_lists, outputs=[model_list, run_model])
#         download_button.click(download_model, inputs=[new_model_name], outputs=[download_output])
#         run_button.click(run_inference, inputs=[run_model, prompt], outputs=[run_output])