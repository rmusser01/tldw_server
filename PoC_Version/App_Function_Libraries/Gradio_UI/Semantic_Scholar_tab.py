# Sematnic_Scholar_tab.py
# Description: contains the code to create the Semantic Scholar tab in the Gradio UI.
#
# Imports
#
# External Libraries
import gradio as gr
#
# Internal Libraries
from App_Function_Libraries.Third_Party.Semantic_Scholar import search_and_display, FIELDS_OF_STUDY, PUBLICATION_TYPES


#
######################################################################################################################
# Functions
def create_semantic_scholar_tab():
    """Create the Semantic Scholar tab for the Gradio UI"""
    with gr.Tab("Semantic Scholar Search"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                    ## Semantic Scholar Paper Search

                    This interface allows you to search for academic papers using the Semantic Scholar API with advanced filtering options:

                    ### Search Options
                    - **Keywords**: Search across titles, abstracts, and other paper content
                    - **Year Range**: Filter papers by publication year (e.g., "2020-2023" or "2020")
                    - **Venue**: Filter by publication venue (journal or conference)
                    - **Minimum Citations**: Filter papers by minimum citation count
                    - **Fields of Study**: Filter papers by academic field
                    - **Publication Types**: Filter by type of publication
                    - **Open Access**: Option to show only papers with free PDF access

                    ### Results Include
                    - Paper title
                    - Author list
                    - Publication year and venue
                    - Citation count
                    - Publication types
                    - Abstract
                    - Links to PDF (when available) and Semantic Scholar page
                    """)
            with gr.Column(scale=2):
                gr.Markdown("""
                    ### Pagination
                    - 10 results per page
                    - Navigate through results using Previous/Next buttons
                    - Current page number and total results displayed

                    ### Usage Tips
                    - Combine multiple filters for more specific results
                    - Use specific terms for more focused results
                    - Try different combinations of filters if you don't find what you're looking for
                    """)
        with gr.Row():
            with gr.Column(scale=2):
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter keywords to search for papers...",
                    lines=1
                )

                # Advanced search options
                with gr.Row():
                    year_range = gr.Textbox(
                        label="Year Range",
                        placeholder="e.g., 2020-2023 or 2020",
                        lines=1
                    )
                    venue = gr.Textbox(
                        label="Venue",
                        placeholder="e.g., Nature, Science",
                        lines=1
                    )
                    min_citations = gr.Number(
                        label="Minimum Citations",
                        value=0,
                        minimum=0,
                        step=1
                    )

                with gr.Row():
                    fields_of_study = gr.Dropdown(
                        choices=FIELDS_OF_STUDY,
                        label="Fields of Study",
                        multiselect=True,
                        value=[]
                    )
                    publication_types = gr.Dropdown(
                        choices=PUBLICATION_TYPES,
                        label="Publication Types",
                        multiselect=True,
                        value=[]
                    )

                open_access_only = gr.Checkbox(
                    label="Open Access Only",
                    value=False
                )

            with gr.Column(scale=1):
                search_button = gr.Button("Search", variant="primary")

                # Pagination controls
                with gr.Row():
                    prev_button = gr.Button("← Previous")
                    current_page = gr.Number(value=0, label="Page", minimum=0, step=1)
                    max_page = gr.Number(value=0, label="Max Page", visible=False)
                    next_button = gr.Button("Next →")

                total_results = gr.Textbox(
                    label="Total Results",
                    value="0",
                    interactive=False
                )

        output_text = gr.Markdown(
            label="Results",
            value="Use the search options above to find papers."
        )

        def update_page(direction, current, maximum):
            new_page = current + direction
            if new_page < 0:
                return 0
            if new_page > maximum:
                return maximum
            return new_page

        # Handle search and pagination
        def search_from_button(query, fields_of_study, publication_types, year_range, venue, min_citations,
                               open_access_only):
            """Wrapper to always search from page 0 when search button is clicked"""
            return search_and_display(
                query=query,
                page=0,  # Force page 0 for new searches
                fields_of_study=fields_of_study,
                publication_types=publication_types,
                year_range=year_range,
                venue=venue,
                min_citations=min_citations,
                open_access_only=open_access_only
            )
        normal_search = search_and_display

        search_button.click(
            fn=search_from_button,
            inputs=[
                search_input, fields_of_study, publication_types,
                year_range, venue, min_citations, open_access_only
            ],
            outputs=[output_text, current_page, max_page, total_results]
        )

        prev_button.click(
            fn=lambda curr, max_p: update_page(-1, curr, max_p),
            inputs=[current_page, max_page],
            outputs=current_page
        ).then(
            fn=normal_search,
            inputs=[
                search_input, current_page, fields_of_study, publication_types,
                year_range, venue, min_citations, open_access_only
            ],
            outputs=[output_text, current_page, max_page, total_results]
        )

        next_button.click(
            fn=lambda curr, max_p: update_page(1, curr, max_p),
            inputs=[current_page, max_page],
            outputs=current_page
        ).then(
            fn=normal_search,
            inputs=[
                search_input, current_page, fields_of_study, publication_types,
                year_range, venue, min_citations, open_access_only
            ],
            outputs=[output_text, current_page, max_page, total_results]
        )

#
# End of Semantic_Scholar_tab.py
######################################################################################################################
