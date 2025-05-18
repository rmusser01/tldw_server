# Re_summarize_tab.py
# Gradio UI for Re-summarizing items in the database
#
# Imports
import json
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Chunk_Lib import improved_chunking_process
from App_Function_Libraries.DB.DB_Manager import update_media_content, list_prompts
from App_Function_Libraries.Gradio_UI.Chat_ui import update_user_prompt
from App_Function_Libraries.Gradio_UI.Gradio_Shared import fetch_item_details, fetch_items_by_keyword, \
    fetch_items_by_content, fetch_items_by_title_or_url
from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_chunk
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging


#
######################################################################################################################
#
# Functions:

def create_resummary_tab():
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

    # Get initial prompts for first page
    initial_prompts, total_pages, current_page = list_prompts(page=1, per_page=20)

    with gr.TabItem("Re-Summarize", visible=True):
        gr.Markdown("# Re-Summarize Existing Content")
        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title", label="Search By")
                search_button = gr.Button("Search")

                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})

                with gr.Row():
                    api_name_input = gr.Dropdown(
                        choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                        value=default_value,
                        label="API for Analysis/Summarization (Optional)"
                    )
                    api_key_input = gr.Textbox(label="API Key", placeholder="Enter your API key here", type="password")

                chunking_options_checkbox = gr.Checkbox(label="Use Chunking", value=False)
                with gr.Row(visible=False) as chunking_options_box:
                    chunk_method = gr.Dropdown(choices=['words', 'sentences', 'paragraphs', 'tokens', 'chapters'],
                                               label="Chunking Method", value='words')
                    max_chunk_size = gr.Slider(minimum=100, maximum=1000, value=300, step=50, label="Max Chunk Size")
                    chunk_overlap = gr.Slider(minimum=0, maximum=100, value=0, step=10, label="Chunk Overlap")

                with gr.Row():
                    custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                     value=False,
                                                     visible=True)
                    preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
                                                     value=False,
                                                     visible=True)

                # Add pagination controls for preset prompts
                with gr.Row(visible=False) as preset_prompt_controls:
                    prev_page = gr.Button("Previous")
                    current_page_text = gr.Text(f"Page {current_page} of {total_pages}")
                    next_page = gr.Button("Next")
                    current_page_state = gr.State(value=1)

                with gr.Row():
                    preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                                choices=initial_prompts,
                                                visible=False)
                with gr.Row():
                    custom_prompt_input = gr.Textbox(label="Custom Prompt",
                                                     placeholder="Enter custom prompt here",
                                                     lines=3,
                                                     visible=False)
                with gr.Row():
                    system_prompt_input = gr.Textbox(label="System Prompt",
                                                     value="""<s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>{{ .Prompt }}
""",
                                                     lines=3,
                                                     visible=False)

                def update_prompt_page(direction, current_page_val):
                    new_page = max(1, min(total_pages, current_page_val + direction))
                    prompts, _, _ = list_prompts(page=new_page, per_page=10)
                    return (
                        gr.update(choices=prompts),
                        gr.update(value=f"Page {new_page} of {total_pages}"),
                        new_page
                    )

                def update_prompts(preset_name):
                    prompts = update_user_prompt(preset_name)
                    return (
                        gr.update(value=prompts["user_prompt"], visible=True),
                        gr.update(value=prompts["system_prompt"], visible=True)
                    )

                # Connect pagination buttons
                prev_page.click(
                    lambda x: update_prompt_page(-1, x),
                    inputs=[current_page_state],
                    outputs=[preset_prompt, current_page_text, current_page_state]
                )

                next_page.click(
                    lambda x: update_prompt_page(1, x),
                    inputs=[current_page_state],
                    outputs=[preset_prompt, current_page_text, current_page_state]
                )

                preset_prompt.change(
                    update_prompts,
                    inputs=preset_prompt,
                    outputs=[custom_prompt_input, system_prompt_input]
                )

                resummarize_button = gr.Button("Re-Summarize")
            with gr.Column():
                result_output = gr.Textbox(label="Result")

        custom_prompt_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[custom_prompt_checkbox],
            outputs=[custom_prompt_input, system_prompt_input]
        )
        preset_prompt_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt, preset_prompt_controls]
        )

    # Connect the UI elements
    search_button.click(
        fn=update_resummarize_dropdown,
        inputs=[search_query_input, search_type_input],
        outputs=[items_output, item_mapping]
    )

    chunking_options_checkbox.change(
        fn=lambda x: gr.update(visible=x),
        inputs=[chunking_options_checkbox],
        outputs=[chunking_options_box]
    )

    custom_prompt_checkbox.change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=[custom_prompt_checkbox],
        outputs=[custom_prompt_input, system_prompt_input]
    )

    resummarize_button.click(
        fn=resummarize_content_wrapper,
        inputs=[items_output, item_mapping, api_name_input, api_key_input, chunking_options_checkbox, chunk_method,
                max_chunk_size, chunk_overlap, custom_prompt_checkbox, custom_prompt_input],
        outputs=result_output
    )

    return (
        search_query_input, search_type_input, search_button, items_output,
        item_mapping, api_name_input, api_key_input, chunking_options_checkbox,
        chunking_options_box, chunk_method, max_chunk_size, chunk_overlap,
        custom_prompt_checkbox, custom_prompt_input, resummarize_button, result_output
    )


def update_resummarize_dropdown(search_query, search_type):
    if search_type in ['Title', 'URL']:
        results = fetch_items_by_title_or_url(search_query, search_type)
    elif search_type == 'Keyword':
        results = fetch_items_by_keyword(search_query)
    else:  # Content
        results = fetch_items_by_content(search_query)

    item_options = [f"{item[1]} ({item[2]})" for item in results]
    item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}
    logging.debug(f"item_options: {item_options}")
    logging.debug(f"item_mapping: {item_mapping}")
    return gr.update(choices=item_options), item_mapping


def resummarize_content_wrapper(selected_item, item_mapping, api_name, api_key=None, chunking_options_checkbox=None, chunk_method=None,
                                max_chunk_size=None, chunk_overlap=None, custom_prompt_checkbox=None, custom_prompt=None):
    logging.debug(f"resummarize_content_wrapper called with item_mapping type: {type(item_mapping)}")
    logging.debug(f"selected_item: {selected_item}")

    if not selected_item or not api_name:
        return "Please select an item and provide API details."

    # Handle potential string representation of item_mapping
    if isinstance(item_mapping, str):
        try:
            item_mapping = json.loads(item_mapping)
        except json.JSONDecodeError:
            return f"Error: item_mapping is a string but not valid JSON. Value: {item_mapping[:100]}..."

    if not isinstance(item_mapping, dict):
        return f"Error: item_mapping is not a dictionary or valid JSON string. Type: {type(item_mapping)}"

    media_id = item_mapping.get(selected_item)
    if not media_id:
        return f"Invalid selection. Selected item: {selected_item}, Available items: {list(item_mapping.keys())[:5]}..."

    content, old_prompt, old_summary = fetch_item_details(media_id)

    if not content:
        return "No content available for re-summarization."

    # Prepare chunking options
    chunk_options = {
        'method': chunk_method,
        'max_size': int(max_chunk_size) if max_chunk_size is not None else None,
        'overlap': int(chunk_overlap) if chunk_overlap is not None else None,
        'language': 'english',
        'adaptive': True,
        'multi_level': False,
    } if chunking_options_checkbox else None

    # Prepare summarization prompt
    summarization_prompt = custom_prompt if custom_prompt_checkbox and custom_prompt else None

    logging.debug(f"Calling resummarize_content with media_id: {media_id}")
    # Call the resummarize_content function
    result = resummarize_content(selected_item, item_mapping, content, api_name, api_key, chunk_options, summarization_prompt)

    return result


# FIXME - should be moved...
def resummarize_content(selected_item, item_mapping, content, api_name, api_key=None, chunk_options=None, summarization_prompt=None):
    logging.debug(f"resummarize_content called with selected_item: {selected_item}")

    # Chunking logic
    if chunk_options:
        chunks = improved_chunking_process(content, chunk_options)
    else:
        chunks = [{'text': content, 'metadata': {}}]

    # Use default prompt if not provided
    # FIXME - USE CHAT DICTIONARY HERE
    if not summarization_prompt:
        summarization_prompt = """<s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
                                    **Bulleted Note Creation Guidelines**
                                    
                                    **Headings**:
                                    - Based on referenced topics, not categories like quotes or terms
                                    - Surrounded by **bold** formatting 
                                    - Not listed as bullet points
                                    - No space between headings and list items underneath
                                    
                                    **Emphasis**:
                                    - **Important terms** set in bold font
                                    - **Text ending in a colon**: also bolded
                                    
                                    **Review**:
                                    - Ensure adherence to specified format
                                    - Do not reference these instructions in your response.</s> {{ .Prompt }}
                                """

    # Summarization logic
    summaries = []
    for chunk in chunks:
        chunk_text = chunk['text']
        try:
            chunk_summary = summarize_chunk(api_name, chunk_text, summarization_prompt, api_key)
            if chunk_summary:
                summaries.append(chunk_summary)
            else:
                logging.warning(f"Summarization failed for chunk: {chunk_text[:100]}...")
        except Exception as e:
            logging.error(f"Error during summarization: {str(e)}")
            return f"Error during summarization: {str(e)}"

    if not summaries:
        return "Summarization failed for all chunks."

    new_summary = " ".join(summaries)

    # Update the database with the new summary

    try:
        update_result = update_media_content(selected_item, item_mapping, content, summarization_prompt, new_summary)
        if "successfully" in update_result.lower():
            return f"Re-summarization complete. New summary: {new_summary}..."
        else:
            return f"Error during database update: {update_result}"
    except Exception as e:
        logging.error(f"Error updating database: {str(e)}")
        return f"Error updating database: {str(e)}"