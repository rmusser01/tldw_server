# Arxiv_tab.py
# Description: This file contains the Gradio UI for searching, browsing, and ingesting arXiv papers.
#
# Imports
import tempfile
from datetime import datetime
import requests

from PoC_Version.App_Function_Libraries.PDF.PDF_Ingestion_Lib import extract_text_and_format_from_pdf
#
# Local Imports
from PoC_Version.App_Function_Libraries.Third_Party.Arxiv import convert_xml_to_markdown, fetch_arxiv_xml, parse_arxiv_feed, \
    build_query_url, ARXIV_PAGE_SIZE, fetch_arxiv_pdf_url
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
#
import gradio as gr
#
#####################################################################################################
#
# Functions:

def create_arxiv_tab():
    with gr.TabItem("Arxiv Search & Ingest", visible=True):
        gr.Markdown("# arXiv Search, Browse, Download, and Ingest")
        gr.Markdown("#### Thank you to arXiv for use of its open access interoperability.")
        with gr.Row():
            with gr.Column(scale=1):
                # Search Inputs
                with gr.Row():
                    with gr.Column():
                        search_query = gr.Textbox(label="Search Query", placeholder="e.g., machine learning")
                        author_filter = gr.Textbox(label="Author", placeholder="e.g., John Doe")
                        year_filter = gr.Number(label="Year", precision=0)
                        search_button = gr.Button("Search")

            with gr.Column(scale=2):
                # Pagination Controls
                    paper_selector = gr.Radio(label="Select a Paper", choices=[], interactive=True)
                    prev_button = gr.Button("Previous Page")
                    next_button = gr.Button("Next Page")
                    page_info = gr.Textbox(label="Page", value="1", interactive=False)

        # Ingestion Section
        with gr.Row():
            with gr.Column():
                # Paper Details View
                paper_view = gr.Markdown(label="Paper Details")
                arxiv_keywords = gr.Textbox(label="Additional Keywords (comma-separated)",
                                            placeholder="e.g., AI, Deep Learning")
                ingest_button = gr.Button("Ingest Selected Paper")
                ingest_result = gr.Textbox(label="Ingestion Result", interactive=False)

        # Define States for Pagination and Selection
        state = gr.State(value={"start": 0, "current_page": 1, "last_query": None, "entries": []})
        selected_paper_id = gr.State(value=None)

        def search_arxiv(query, author, year):
            start = 0
            url = build_query_url(query, author, year, start)
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return gr.update(value=[]), gr.update(value=f"**Error:** {str(e)}"), state.value

            entries = parse_arxiv_feed(response.text)
            state.value = {"start": start, "current_page": 1, "last_query": (query, author, year), "entries": entries}
            if not entries:
                return gr.update(value=[]), "No results found.", state.value

            # Update the dropdown with paper titles for selection
            titles = [entry['title'] for entry in entries]
            return gr.update(choices=titles), "1", state.value

        # Dead code? FIXME
        def handle_pagination(direction):
            current_state = state.value
            query, author, year = current_state["last_query"]
            new_page = current_state["current_page"] + direction
            if new_page < 1:
                new_page = 1
            start = (new_page - 1) * ARXIV_PAGE_SIZE
            url = build_query_url(query, author, year, start)
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return gr.update(), gr.update()

            entries = parse_arxiv_feed(response.text)
            if entries:
                current_state["start"] = start
                current_state["current_page"] = new_page
                current_state["entries"] = entries
                state.value = current_state

                # Update the dropdown with paper titles for the new page
                titles = [entry['title'] for entry in entries]
                return gr.update(choices=titles), str(new_page)
            else:
                # If no entries, do not change the page
                return gr.update(), gr.update()

        def load_selected_paper(selected_title):
            if not selected_title:
                return "Please select a paper to view."

            # Find the selected paper from state
            for entry in state.value["entries"]:
                if entry['title'] == selected_title:
                    paper_id = entry['id']
                    break
            else:
                return "Paper not found."

            try:
                # Fetch the PDF URL and download the full-text
                pdf_url = fetch_arxiv_pdf_url(paper_id)
                response = requests.get(pdf_url)
                response.raise_for_status()

                # Save the PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf_path = temp_pdf.name

                # Convert PDF to markdown using your PDF ingestion function
                full_text_markdown = extract_text_and_format_from_pdf(temp_pdf_path)

                selected_paper_id.value = paper_id
                return full_text_markdown
            except Exception as e:
                return f"Error loading full paper: {str(e)}"

        def process_and_ingest_arxiv_paper(paper_id, additional_keywords):
            try:
                if not paper_id:
                    return "Please select a paper to ingest."

                # Fetch the PDF URL
                pdf_url = fetch_arxiv_pdf_url(paper_id)

                # Download the PDF
                response = requests.get(pdf_url)
                response.raise_for_status()

                # Save the PDF temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf_path = temp_pdf.name

                # Convert PDF to markdown using your PDF ingestion function
                markdown_text = extract_text_and_format_from_pdf(temp_pdf_path)

                # Fetch metadata from arXiv to get title, authors, and categories
                xml_content = fetch_arxiv_xml(paper_id)
                _, title, authors, categories = convert_xml_to_markdown(xml_content)

                # Prepare the arXiv paper URL for access/download
                paper_url = f"https://arxiv.org/abs/{paper_id}"

                # Prepare the keywords for ingestion
                keywords = f"arxiv,{','.join(categories)}"
                if additional_keywords:
                    keywords += f",{additional_keywords}"

                # Ingest full paper markdown content
                add_media_with_keywords(
                    url=paper_url,
                    title=title,
                    media_type='document',
                    content=markdown_text,  # Full paper content in markdown
                    keywords=keywords,
                    prompt='No prompt for arXiv papers',
                    summary='Full arXiv paper ingested from PDF',
                    transcription_model='None',
                    author=', '.join(authors),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )

                # Return success message with paper title and authors
                return f"arXiv paper '{title}' by {', '.join(authors)} ingested successfully."
            except Exception as e:
                # Return error message if anything goes wrong
                return f"Error processing arXiv paper: {str(e)}"

        # Event Handlers
        # Connect Search Button
        search_button.click(
            fn=search_arxiv,
            inputs=[search_query, author_filter, year_filter],
            outputs=[paper_selector, page_info, state],
            queue=True
        )

        # Connect Next Button
        next_button.click(
            fn=lambda: handle_pagination(1),
            inputs=None,
            outputs=[paper_selector, page_info],
            queue=True
        )

        # Connect Previous Button
        prev_button.click(
            fn=lambda: handle_pagination(-1),
            inputs=None,
            outputs=[paper_selector, page_info],
            queue=True
        )

        # When the user selects a paper in the Dropdown
        paper_selector.change(
            fn=load_selected_paper,
            inputs=paper_selector,
            outputs=paper_view,
            queue=True
        )

        # Connect Ingest Button
        ingest_button.click(
            fn=process_and_ingest_arxiv_paper,
            inputs=[selected_paper_id, arxiv_keywords],
            outputs=ingest_result,
            queue=True
        )

#
# End of File
#####################################################################################################
