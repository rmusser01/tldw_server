# Trash.py
# Gradio UI for managing trashed items in the database
#
# Imports
from typing import Tuple, List

import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import (
    get_trashed_items, user_delete_item, empty_trash,
    get_transcripts, fetch_item_details,
    search_media_database, mark_as_trash,
)


#
############################################################################################################
#
# Functions:


def list_trash():
    items = get_trashed_items()
    return "\n".join(
        [f"ID: {item['id']}, Title: {item['title']}, Trashed on: {item['trash_date']}" for item in items])


def delete_item(media_id, force):
    return user_delete_item(media_id, force)


def empty_trash_ui(days):
    deleted, remaining = empty_trash(days)
    return f"Deleted {deleted} items. {remaining} items remain in trash."


def get_media_transcripts(media_id):
    transcripts = get_transcripts(media_id)
    return "\n\n".join([f"Transcript ID: {t[0]}\nModel: {t[1]}\nCreated: {t[3]}\n{t[2][:200]}..." for t in transcripts])


def get_media_summaries(media_id):
    _, summary, _ = fetch_item_details(media_id)
    return summary if summary else "No summary available."


def get_media_prompts(media_id):
    prompt, _, _ = fetch_item_details(media_id)
    return prompt if prompt else "No prompt available."


def search_and_mark_trash(search_query: str) -> Tuple[List[Tuple[int, str, str]], str]:
    try:
        results = search_media_database(search_query)
        if not results:
            return [], "No items found matching the search query."
        return results, "Search completed successfully."
    except Exception as e:
        return [], f"Error during search: {str(e)}"


def mark_item_as_trash(media_id: int) -> str:
    try:
        mark_as_trash(media_id)
        return f"Item with ID {media_id} has been marked as trash."
    except Exception as e:
        return f"Error marking item as trash: {str(e)}"


def create_search_and_mark_trash_tab():
    with gr.TabItem("Search and Mark as Trash", visible=True):
        gr.Markdown("# Search for Items and Mark as Trash")

        search_input = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search")
        search_results = gr.Dropdown(label="Search Results", choices=[], interactive=True)
        search_status = gr.Textbox(label="Search Status")

        mark_trash_button = gr.Button("Mark Selected Item as Trash")
        mark_trash_status = gr.Textbox(label="Mark as Trash Status")

        def update_search_results(query):
            results, status = search_and_mark_trash(query)
            choices = [f"{id}: {title} ({url})" for id, title, url in results]
            return choices, status

        search_button.click(
            update_search_results,
            inputs=[search_input],
            outputs=[search_results, search_status]
        )

        def mark_selected_as_trash(selected_item):
            if selected_item:
                media_id = int(selected_item.split(":")[0])
                return mark_item_as_trash(media_id)
            return "No item selected."

        mark_trash_button.click(
            mark_selected_as_trash,
            inputs=[search_results],
            outputs=[mark_trash_status]
        )


def create_view_trash_tab():
    with gr.TabItem("View Trash", visible=True):
        view_button = gr.Button("View Trash")
        trash_list = gr.Textbox(label="Trashed Items")
        view_button.click(list_trash, inputs=[], outputs=trash_list)


def create_delete_trash_tab():
    with gr.TabItem("Delete DB Item", visible=True):
        gr.Markdown("# Delete Items from Databases")

        media_id_input = gr.Number(label="Media ID")
        media_force_checkbox = gr.Checkbox(label="Force Delete")
        media_delete_button = gr.Button("Delete Media")
        media_delete_output = gr.Textbox(label="Delete Result")

        media_delete_button.click(
            delete_item,
            inputs=[media_id_input, media_force_checkbox],
            outputs=media_delete_output
        )


def create_empty_trash_tab():
    with gr.TabItem("Empty Trash", visible=True):
        days_input = gr.Slider(minimum=15, maximum=90, step=5, label="Delete items older than (days)")
        empty_button = gr.Button("Empty Trash")
        empty_output = gr.Textbox(label="Result")
        empty_button.click(empty_trash_ui, inputs=[days_input], outputs=empty_output)

#
# End of File
############################################################################################################
