# Transcript_comparison.py
# Description: Gradio UI tab for comparing transcripts
#
# Imports
#
# 3rd-Party Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import get_transcripts
from App_Function_Libraries.Gradio_UI.Gradio_Shared import browse_items
from App_Function_Libraries.Utils.Utils import format_transcription, logging


def get_transcript_options(media_id):
    transcripts = get_transcripts(media_id)
    return [f"{t[0]}: {t[1]} ({t[3]})" for t in transcripts]


def update_transcript_options(media_id):
    options = get_transcript_options(media_id)
    return gr.update(choices=options), gr.update(choices=options)

def compare_transcripts(media_id, transcript1_id, transcript2_id):
    try:
        transcripts = get_transcripts(media_id)
        transcript1 = next((t for t in transcripts if t[0] == int(transcript1_id)), None)
        transcript2 = next((t for t in transcripts if t[0] == int(transcript2_id)), None)

        if not transcript1 or not transcript2:
            return "One or both selected transcripts not found."

        comparison = f"Transcript 1 (Model: {transcript1[1]}, Created: {transcript1[3]}):\n\n"
        comparison += format_transcription(transcript1[2])
        comparison += f"\n\nTranscript 2 (Model: {transcript2[1]}, Created: {transcript2[3]}):\n\n"
        comparison += format_transcription(transcript2[2])

        return comparison
    except Exception as e:
        logging.error(f"Error in compare_transcripts: {str(e)}")
        return f"Error comparing transcripts: {str(e)}"


def create_compare_transcripts_tab():
    with gr.TabItem("Compare Transcripts", visible=True):
        gr.Markdown("# Compare Transcripts")

        with gr.Row():
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title", label="Search By")
            search_button = gr.Button("Search")

        with gr.Row():
            media_id_output = gr.Dropdown(label="Select Media Item", choices=[], interactive=True)
            media_mapping = gr.State({})

        media_id_input = gr.Number(label="Media ID", visible=False)
        transcript1_dropdown = gr.Dropdown(label="Transcript 1")
        transcript2_dropdown = gr.Dropdown(label="Transcript 2")
        compare_button = gr.Button("Compare Transcripts")
        comparison_output = gr.Textbox(label="Comparison Result", lines=20)

        def update_media_dropdown(search_query, search_type):
            results = browse_items(search_query, search_type)
            item_options = [f"{item[1]} ({item[2]})" for item in results]
            new_item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}
            return gr.update(choices=item_options), new_item_mapping

        search_button.click(
            fn=update_media_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[media_id_output, media_mapping]
        )

        def load_selected_media_id(selected_media, media_mapping):
            if selected_media and media_mapping and selected_media in media_mapping:
                media_id = media_mapping[selected_media]
                return media_id
            return None

        media_id_output.change(
            fn=load_selected_media_id,
            inputs=[media_id_output, media_mapping],
            outputs=[media_id_input]
        )

        media_id_input.change(update_transcript_options, inputs=[media_id_input],
                              outputs=[transcript1_dropdown, transcript2_dropdown])
        compare_button.click(compare_transcripts, inputs=[media_id_input, transcript1_dropdown, transcript2_dropdown],
                             outputs=[comparison_output])