# View_tab.py
# Description: Gradio functions for the view tab
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import (
    search_media_database, get_specific_prompt, delete_specific_transcript,
    delete_specific_summary, delete_specific_prompt, get_specific_transcript, get_specific_summary,
    get_media_transcripts, get_media_summaries, get_media_prompts
)
#
############################################################################################################
#
# Functions:

# FIXME - add mark_as_trash ability to the UI


# FIXME - Doesn't work. also need ot merge this tab wtih Edit Existing Items tab....
def create_manage_items_tab():
    with gr.TabItem("Edit/Manage DB Items", visible=True):
        search_input = gr.Textbox(label="Search for Media (title or ID)")
        search_button = gr.Button("Search")
        media_selector = gr.Dropdown(label="Select Media", choices=[], interactive=True)

        with gr.Accordion("Transcripts"):
            get_transcripts_button = gr.Button("Get Transcripts")
            transcript_selector = gr.Dropdown(label="Select Transcript", choices=[], interactive=True)
            transcripts_output = gr.Textbox(label="Transcript Content", lines=10)
            delete_transcript_button = gr.Button("Delete Selected Transcript")

        with gr.Accordion("Summaries"):
            get_summaries_button = gr.Button("Get Summaries")
            summary_selector = gr.Dropdown(label="Select Summary", choices=[], interactive=True)
            summaries_output = gr.Textbox(label="Summary Content", lines=5)
            delete_summary_button = gr.Button("Delete Selected Summary")

        with gr.Accordion("Prompts"):
            get_prompts_button = gr.Button("Get Prompts")
            prompt_selector = gr.Dropdown(label="Select Prompt", choices=[], interactive=True)
            prompts_output = gr.Textbox(label="Prompt Content", lines=5)
            delete_prompt_button = gr.Button("Delete Selected Prompt")

        status_output = gr.Textbox(label="Status")

        def search_media(query):
            results = search_media_database(query)
            choices = [f"{result[0]}: {result[1]}" for result in results]
            return {"choices": choices, "value": None}

        search_button.click(search_media, inputs=[search_input], outputs=[media_selector])

        def get_transcripts(media_selection):
            if not media_selection:
                return {"choices": [], "value": None}
            media_id = int(media_selection.split(":")[0])
            transcripts = get_media_transcripts(media_id)
            choices = [f"{t[0]}: {t[3]}" for t in transcripts]
            return {"choices": choices, "value": None}

        def display_transcript(transcript_selection):
            if not transcript_selection:
                return "No transcript selected."
            transcript_id = int(transcript_selection.split(":")[0])
            transcript = get_specific_transcript(transcript_id)
            return transcript['content'] if 'content' in transcript else transcript.get('error', "Transcript not found.")

        get_transcripts_button.click(
            get_transcripts,
            inputs=[media_selector],
            outputs=[transcript_selector]
        )
        transcript_selector.change(
            display_transcript,
            inputs=[transcript_selector],
            outputs=[transcripts_output]
        )

        def get_summaries(media_selection):
            if not media_selection:
                return {"choices": [], "value": None}
            media_id = int(media_selection.split(":")[0])
            summaries = get_media_summaries(media_id)
            choices = [f"{s[0]}: {s[3]}" for s in summaries]
            return {"choices": choices, "value": None}

        def display_summary(summary_selection):
            if not summary_selection:
                return "No summary selected."
            summary_id = int(summary_selection.split(":")[0])
            summary = get_specific_summary(summary_id)
            return summary['content'] if 'content' in summary else summary.get('error', "Summary not found.")

        get_summaries_button.click(
            get_summaries,
            inputs=[media_selector],
            outputs=[summary_selector]
        )
        summary_selector.change(
            display_summary,
            inputs=[summary_selector],
            outputs=[summaries_output]
        )

        def get_prompts(media_selection):
            if not media_selection:
                return {"choices": [], "value": None}
            media_id = int(media_selection.split(":")[0])
            prompts = get_media_prompts(media_id)
            choices = [f"{p[0]}: {p[3]}" for p in prompts]
            return {"choices": choices, "value": None}

        def display_prompt(prompt_selection):
            if not prompt_selection:
                return "No prompt selected."
            prompt_id = int(prompt_selection.split(":")[0])
            prompt = get_specific_prompt(prompt_id)
            return prompt['content'] if 'content' in prompt else prompt.get('error', "Prompt not found.")

        get_prompts_button.click(
            get_prompts,
            inputs=[media_selector],
            outputs=[prompt_selector]
        )
        prompt_selector.change(
            display_prompt,
            inputs=[prompt_selector],
            outputs=[prompts_output]
        )

        def delete_transcript(transcript_selection):
            if not transcript_selection:
                return "No transcript selected."
            transcript_id = int(transcript_selection.split(":")[0])
            result = delete_specific_transcript(transcript_id)
            return result

        def delete_summary(summary_selection):
            if not summary_selection:
                return "No summary selected."
            summary_id = int(summary_selection.split(":")[0])
            result = delete_specific_summary(summary_id)
            return result

        def delete_prompt(prompt_selection):
            if not prompt_selection:
                return "No prompt selected."
            prompt_id = int(prompt_selection.split(":")[0])
            result = delete_specific_prompt(prompt_id)
            return result

        delete_transcript_button.click(
            delete_transcript,
            inputs=[transcript_selector],
            outputs=[status_output]
        )
        delete_summary_button.click(
            delete_summary,
            inputs=[summary_selector],
            outputs=[status_output]
        )
        delete_prompt_button.click(
            delete_prompt,
            inputs=[prompt_selector],
            outputs=[status_output]
        )