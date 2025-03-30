# Workflows.py
#
#########################################
# Workflow Library
# This library is used to facilitate chained prompt workflows
#
####
####################
# Function Categories
#
# Fixme
#
#
####################
# Function List
#
# 1. FIXME
#
####################
#
# Imports
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
#
# 3rd-Party Imports
#
# Local Imports
from PoC_Version.App_Function_Libraries.Chat.Chat_Functions import chat
from PoC_Version.App_Function_Libraries.Utils.Utils import logging

#
#######################################################################################################################
#
# Function Definitions

# Load workflows from a JSON file
json_path = Path('./App_Function_Libraries/Workflows/Workflows.json')

# Load workflows from a JSON file
def load_workflows(json_path: str = './App_Function_Libraries/Workflows/Workflows.json') -> List[Dict]:
    with Path(json_path).open('r') as f:
        return json.load(f)

# Initialize a workflow
def initialize_workflow(workflow_name: str, workflows: List[Dict]) -> Tuple[Dict, str, List[Tuple[Optional[str], str]]]:
    selected_workflow = next((wf for wf in workflows if wf['name'] == workflow_name), None)
    if selected_workflow:
        num_prompts = len(selected_workflow['prompts'])
        context = selected_workflow.get('context', '')
        first_prompt = selected_workflow['prompts'][0]
        initial_chat = [(None, f"{first_prompt}")]
        workflow_state = {"current_step": 0, "max_steps": num_prompts, "conversation_id": None}
        logging.info(f"Initializing workflow: {workflow_name} with {num_prompts} steps")
        return workflow_state, context, initial_chat
    else:
        logging.error(f"Selected workflow not found: {workflow_name}")
        return {"current_step": 0, "max_steps": 0, "conversation_id": None}, "", []


# Process a workflow step
def process_workflow_step(
        message: str,
        history: List[Tuple[Optional[str], str]],
        context: str,
        workflow_name: str,
        workflows: List[Dict],
        workflow_state: Dict,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        save_conv: bool = False,
        temp: float = 0.7,
        system_message: Optional[str] = None,
        media_content: Dict = {},
        selected_parts: List[str] = []
) -> Tuple[List[Tuple[Optional[str], str]], Dict, bool]:
    logging.info(f"Process workflow step called with message: {message}")
    logging.info(f"Current workflow state: {workflow_state}")

    try:
        selected_workflow = next((wf for wf in workflows if wf['name'] == workflow_name), None)
        if not selected_workflow:
            logging.error(f"Selected workflow not found: {workflow_name}")
            return history, workflow_state, True

        current_step = workflow_state["current_step"]
        max_steps = workflow_state["max_steps"]

        logging.info(f"Current step: {current_step}, Max steps: {max_steps}")

        if current_step >= max_steps:
            logging.info("Workflow completed")
            return history, workflow_state, False

        prompt = selected_workflow['prompts'][current_step]
        full_message = f"{context}\n\nStep {current_step + 1}: {prompt}\nUser: {message}"

        logging.info(f"Preparing to process message: {full_message[:100]}...")

        # Use the existing chat function
        bot_message = chat(
            full_message, history, media_content, selected_parts,
            api_endpoint, api_key, prompt, temp, system_message
        )

        logging.info(f"Received bot_message: {bot_message[:100]}...")

        new_history = history + [(message, bot_message)]
        next_step = current_step + 1
        new_workflow_state = {
            "current_step": next_step,
            "max_steps": max_steps,
            "conversation_id": workflow_state["conversation_id"]
        }

        if next_step >= max_steps:
            logging.info("Workflow completed after this step")
            return new_history, new_workflow_state, False
        else:
            next_prompt = selected_workflow['prompts'][next_step]
            new_history.append((None, f"Step {next_step + 1}: {next_prompt}"))
            logging.info(f"Moving to next step: {next_step}")
            return new_history, new_workflow_state, True

    except Exception as e:
        logging.error(f"Error in process_workflow_step: {str(e)}")
        return history, workflow_state, True


# Main function to run a workflow
def run_workflow(
        workflow_name: str,
        initial_context: str = "",
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        save_conv: bool = False,
        temp: float = 0.7,
        system_message: Optional[str] = None,
        media_content: Dict = {},
        selected_parts: List[str] = []
) -> List[Tuple[Optional[str], str]]:
    workflows = load_workflows()
    workflow_state, context, history = initialize_workflow(workflow_name, workflows)

    # Combine the initial_context with the workflow's context
    combined_context = f"{initial_context}\n\n{context}".strip()

    while True:
        user_input = input("Your input (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        history, workflow_state, continue_workflow = process_workflow_step(
            user_input, history, combined_context, workflow_name, workflows, workflow_state,
            api_endpoint, api_key, save_conv, temp, system_message, media_content, selected_parts
        )

        for _, message in history[-2:]:  # Print the last two messages (user input and bot response)
            print(message)

        if not continue_workflow:
            print("Workflow completed.")
            break

    return history

# Example usage
# if __name__ == "__main__":
#     workflow_name = "Example Workflow"
#     initial_context = "This is an example context."
#
#     final_history = run_workflow(
#         workflow_name,
#         initial_context,
#         api_endpoint="your_api_endpoint",
#         api_key="your_api_key",
#         save_conv=True,
#         temp=0.7,
#         system_message="You are a helpful assistant guiding through a workflow.",
#         media_content={},
#         selected_parts=[]
#     )
#
#     print("Final conversation history:")
#     for user_message, bot_message in final_history:
#         if user_message:
#             print(f"User: {user_message}")
#         print(f"Bot: {bot_message}")
#         print()

#
# End of Workflows.py
#######################################################################################################################
