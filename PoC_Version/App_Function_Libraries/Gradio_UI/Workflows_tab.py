# Chat_Workflows.py
# Description: Gradio UI for Chat Workflows
#
# Imports
import json
from pathlib import Path
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Gradio_UI.Chat_ui import chat_wrapper, search_conversations, \
    load_conversation
from App_Function_Libraries.Chat.Chat_Functions import save_chat_history_to_db_wrapper
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging
#
############################################################################################################
#
# Functions:

# Load workflows from a JSON file
json_path = Path('./App_Function_Libraries/Workflows/Workflows.json')
with json_path.open('r') as f:
    workflows = json.load(f)


def chat_workflows_tab():
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
    with gr.TabItem("Chat Workflows", visible=True):
        gr.Markdown("# Workflows using LLMs")
        chat_history = gr.State([])
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)
        workflow_state = gr.State({"current_step": 0, "max_steps": 0, "conversation_id": None})

        with gr.Row():
            with gr.Column():
                workflow_selector = gr.Dropdown(label="Select Workflow", choices=[wf['name'] for wf in workflows])
                api_selector = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Interaction (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key (optional)", type="password")
                temperature = gr.Slider(label="Temperature", minimum=0.00, maximum=1.0, step=0.05, value=0.7)
                save_conversation = gr.Checkbox(label="Save Conversation", value=False)
            with gr.Column():
                gr.Markdown("Placeholder")
        with gr.Row():
            with gr.Column():
                conversation_search = gr.Textbox(label="Search Conversations")
                search_conversations_btn = gr.Button("Search Conversations")
            with gr.Column():
                previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                load_conversations_btn = gr.Button("Load Selected Conversation")
        with gr.Row():
            with gr.Column():
                context_input = gr.Textbox(label="Initial Context", lines=5)
                chatbot = gr.Chatbot(label="Workflow Chat")
                msg = gr.Textbox(label="Your Input")
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Chat")
                chat_media_name = gr.Textbox(label="Custom Chat Name(optional)")
                save_btn = gr.Button("Save Chat to Database")
                save_status = gr.Textbox(label="Save Status", interactive=False)

        def update_workflow_ui(workflow_name):
            if not workflow_name:
                return {"current_step": 0, "max_steps": 0, "conversation_id": None}, "", []
            selected_workflow = next((wf for wf in workflows if wf['name'] == workflow_name), None)
            if selected_workflow:
                num_prompts = len(selected_workflow['prompts'])
                context = selected_workflow.get('context', '')
                first_prompt = selected_workflow['prompts'][0]
                initial_chat = [(None, f"{first_prompt}")]
                logging.info(f"Initializing workflow: {workflow_name} with {num_prompts} steps")
                return {"current_step": 0, "max_steps": num_prompts, "conversation_id": None}, context, initial_chat
            else:
                logging.error(f"Selected workflow not found: {workflow_name}")
                return {"current_step": 0, "max_steps": 0, "conversation_id": None}, "", []

        def process_workflow_step(message, history, context, workflow_name, api_endpoint, api_key, workflow_state, save_conv, temp):
            logging.info(f"Process workflow step called with message: {message}")
            logging.info(f"Current workflow state: {workflow_state}")
            try:
                selected_workflow = next((wf for wf in workflows if wf['name'] == workflow_name), None)
                if not selected_workflow:
                    logging.error(f"Selected workflow not found: {workflow_name}")
                    return history, workflow_state, gr.update(interactive=True)

                current_step = workflow_state["current_step"]
                max_steps = workflow_state["max_steps"]

                logging.info(f"Current step: {current_step}, Max steps: {max_steps}")

                if current_step >= max_steps:
                    logging.info("Workflow completed, disabling input")
                    return history, workflow_state, gr.update(interactive=False)

                prompt = selected_workflow['prompts'][current_step]
                full_message = f"{context}\n\nStep {current_step + 1}: {prompt}\nUser: {message}"

                logging.info(f"Calling chat_wrapper with full_message: {full_message[:100]}...")

                # Initialize bot message to accumulate stremed response
                bot_message = ""

                # call chat _wrapper
                for chunk, new_history, new_conversation_id in chat_wrapper(
                    full_message, history, media_content.value, selected_parts.value,
                    api_endpoint, api_key, "", workflow_state["conversation_id"],
                    save_conv, temp, "You are a helpful assistant guiding through a workflow."
                ):
                    bot_message = chunk  # Update bot message with the latest chunk
                    yield new_history, workflow_state, gr.update(interactive=True)
                    logging.info(f"Received bot_message: {bot_message[:50]}...")


                next_step = current_step + 1
                new_workflow_state = {
                    "current_step": next_step,
                    "max_steps": max_steps,
                    "conversation_id": new_conversation_id
                }

                if next_step >= max_steps:
                    logging.info("Workflow completed after this step")
                    yield history + [(message, bot_message)], new_workflow_state, gr.update(interactive=False)
                else:
                    next_prompt = selected_workflow['prompts'][next_step]
                    new_history = history + [(message, bot_message)]
                    new_history.append((None, f"Step {next_step + 1}: {next_prompt}"))
                    logging.info(f"Moving to next step: {next_step}")
                    yield new_history, new_workflow_state, gr.update(interactive=True)

            except Exception as e:
                logging.error(f"Error in process_workflow_step: {str(e)}")
                yield history, workflow_state, gr.update(interactive=True)

        workflow_selector.change(
            update_workflow_ui,
            inputs=[workflow_selector],
            outputs=[workflow_state, context_input, chatbot]
        )

        submit_btn.click(
            process_workflow_step,
            inputs=[msg, chatbot, context_input, workflow_selector, api_selector, api_key_input, workflow_state, save_conversation, temperature],
            outputs=[chatbot, workflow_state, msg]
        ).then(
            lambda: gr.update(value=""),
            outputs=[msg]
        )

        clear_btn.click(
            lambda: ([], {"current_step": 0, "max_steps": 0, "conversation_id": None}, ""),
            outputs=[chatbot, workflow_state, context_input]
        )

        save_btn.click(
            save_chat_history_to_db_wrapper,
            inputs=[chatbot, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, save_status]
        )

        search_conversations_btn.click(
            search_conversations,
            inputs=[conversation_search],
            outputs=[previous_conversations]
        )

        load_conversations_btn.click(
            lambda: ([], {"current_step": 0, "max_steps": 0, "conversation_id": None}, ""),
            outputs=[chatbot, workflow_state, context_input]
        ).then(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chatbot, conversation_id]
        )

    return workflow_selector, api_selector, api_key_input, context_input, chatbot, msg, submit_btn, clear_btn, save_btn

#
# End of script
############################################################################################################
