# WebSearch_tab.py
# Gradio UI for performing web searches with aggregated results
#
# Imports
import asyncio
from typing import Dict
#
# External Imports
import gradio as gr

#
# Local Imports
from PoC_Version.App_Function_Libraries.Web_Scraping.WebSearch_APIs import generate_and_search, analyze_and_aggregate
from PoC_Version.App_Function_Libraries.Utils.Utils import loaded_config_data, logging


#
########################################################################################################################
#
# Functions:
def create_websearch_tab():
    with gr.TabItem("Web Search & Review"):
        with gr.Blocks() as interface:
            search_state = gr.State(value=None)
            # Basic styling
            gr.HTML("""
                <style>
                    .analysis-text {
                        white-space: pre-wrap !important;
                        word-wrap: break-word !important;
                        max-width: 100% !important;
                        margin: 16px 0 !important;
                    }

                    /* Make bullet points and lists render properly */
                    .analysis-text ul {
                        padding-left: 20px !important;
                        margin: 10px 0 !important;
                    }

                    .analysis-text li {
                        margin: 5px 0 !important;
                    }

                    /* Ensure headers have proper spacing */
                    .analysis-text h1, 
                    .analysis-text h2, 
                    .analysis-text h3 {
                        margin-top: 20px !important;
                        margin-bottom: 10px !important;
                    }

                    /* Your existing styles... */
                    .result-card { /* existing styles */ }
                    /* ... rest of your existing styles ... */
                </style>
            """)

            # Input Section
            with gr.Row():
                with gr.Column():
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="What would you like to search for?",
                        lines=2
                    )

                    with gr.Row():
                        engine = gr.Dropdown(
                            choices=["google", "bing", "duckduckgo", "brave"],
                            value="google",
                            label="Search Engine"
                        )
                        num_results = gr.Slider(
                            minimum=1, maximum=20, value=10, step=1,
                            label="Number of Results"
                        )

                    with gr.Row():
                        country = gr.Dropdown(
                            choices=["US", "UK", "CA", "AU"],
                            value="US",
                            label="Content Region"
                        )
                        language = gr.Dropdown(
                            choices=["en", "es", "fr", "de"],
                            value="en",
                            label="Language"
                        )

            # Action Buttons and Status
            with gr.Row():
                search_btn = gr.Button("Search", variant="primary")
                status = gr.Markdown("Ready")

            # Results Section
            results_display = gr.HTML(visible=False)
            # Analysis button and status container
            with gr.Row() as analyze_container:
                analyze_btn = gr.Button("Analyze Selected Results", visible=False)
                analysis_status = gr.HTML(
                    """
                    <div class="analyze-button-container">
                        <div class="status-spinner"></div>
                        <span class="status-text"></span>
                    </div>
                    """,
                    visible=False
                )

            # Final Output Section
            with gr.Column(visible=False) as output_section:
                # Single markdown box for all analysis text
                answer = gr.Markdown(
                    label="Analysis Results",
                    elem_classes="analysis-text"
                )
                # Sources box
                sources = gr.JSON(label="Sources")

            def format_results(results: list) -> str:
                """Format search results as HTML."""
                html = ""
                for idx, result in enumerate(results):
                    html += f"""
                    <div class="result-card">
                        <input type="checkbox" id="result-{idx}" checked>
                        <div class="result-title">{result.get('title', 'No title')}</div>
                        <div class="result-url">{result.get('url', 'No URL')}</div>
                        <div class="result-preview">{result.get('content', 'No content')[:200]}...</div>
                    </div>
                    """
                return html

            relevance_analysis_llm = loaded_config_data['search_settings']["relevance_analysis_llm"]
            final_answer_llm = loaded_config_data['search_settings']["final_answer_llm"]

            def perform_search(query: str, engine: str, num_results: int,
                               country: str, language: str) -> Dict:
                """Execute the search operation."""
                search_params = {
                    "engine": engine,
                    "content_country": country,
                    "search_lang": language,
                    "output_lang": language,
                    "result_count": num_results,
                    # Add LLM settings
                    "relevance_analysis_llm": relevance_analysis_llm,
                    "final_answer_llm": final_answer_llm
                }

                return generate_and_search(query, search_params)

            def search_handler(query, engine, num_results, country, language):
                try:
                    # Call perform_search with individual arguments
                    results = perform_search(
                        query=query,
                        engine=engine,
                        num_results=num_results,
                        country=country,
                        language=language
                    )

                    logging.debug(f"Search results: {results}")

                    if not results.get("web_search_results_dict") or not results["web_search_results_dict"].get(
                            "results"):
                        raise ValueError("No search results returned")

                    results_html = format_results(results["web_search_results_dict"]["results"])

                    # Store complete results including search params
                    state_to_store = {
                        "web_search_results_dict": results["web_search_results_dict"],
                        "sub_query_dict": results.get("sub_query_dict", {}),
                        "search_params": {
                            "engine": engine,
                            "content_country": country,
                            "search_lang": language,
                            "output_lang": language,
                            "relevance_analysis_llm": relevance_analysis_llm,
                            "final_answer_llm": final_answer_llm
                        }
                    }

                    logging.info(
                        f"Storing state with {len(state_to_store['web_search_results_dict']['results'])} results")

                    return (
                        gr.Markdown("Search completed successfully"),
                        gr.HTML(results_html, visible=True),
                        gr.Button(visible=True),
                        gr.HTML(visible=True),
                        gr.Column(visible=False),
                        state_to_store
                    )
                except Exception as e:
                    logging.error(f"Search error: {str(e)}", exc_info=True)
                    return (
                        gr.Markdown(f"Error: {str(e)}"),
                        gr.HTML(visible=False),
                        gr.Button(visible=False),
                        gr.HTML(visible=False),
                        gr.Column(visible=False),
                        None
                    )

            async def analyze_handler(state):
                logging.debug(f"Received state for analysis: {state}")
                try:
                    yield (
                        gr.HTML(
                            """
                            <div class="analyze-button-container">
                                <div class="status-spinner visible"></div>
                                <span class="status-text">Processing results...</span>
                            </div>
                            """,
                            visible=True
                        ),
                        gr.Markdown("Analysis in progress..."),
                        gr.JSON(None),
                        gr.Column(visible=False)
                    )

                    if not state or not isinstance(state, dict):
                        raise ValueError(f"Invalid state received: {state}")

                    if not state.get("web_search_results_dict"):
                        raise ValueError("No web search results in state")

                    if not state["web_search_results_dict"].get("results"):
                        raise ValueError("No results array in web search results")

                    relevance_analysis_llm = loaded_config_data['search_settings']["relevance_analysis_llm"]
                    final_answer_llm = loaded_config_data['search_settings']["final_answer_llm"]

                    # Create search params with required LLM settings
                    search_params = {
                        "engine": state["web_search_results_dict"]["search_engine"],
                        "content_country": state["web_search_results_dict"]["content_country"],
                        "search_lang": state["web_search_results_dict"]["search_lang"],
                        "output_lang": state["web_search_results_dict"]["output_lang"],
                        # Add LLM settings
                        "relevance_analysis_llm": relevance_analysis_llm,
                        "final_answer_llm": final_answer_llm
                    }

                    # Analyze results
                    analysis = await analyze_and_aggregate(
                        state["web_search_results_dict"],
                        state.get("sub_query_dict", {}),
                        state.get("search_params", {})
                    )

                    logging.debug(f"Analysis results: {analysis}")

                    if not analysis.get("final_answer"):
                        raise ValueError("Analysis did not produce a final answer")

                    # Format the raw report with proper markdown
                    raw_report = analysis["final_answer"]["Report"]

                    # Ensure proper markdown formatting
                    formatted_answer = raw_report.replace('\n', '\n\n')  # Double line breaks
                    formatted_answer = formatted_answer.replace('•', '\n•')  # Bullet points on new lines
                    formatted_answer = formatted_answer.replace('- ', '\n- ')  # Dashed lists on new lines

                    # Handle numbered lists (assumes numbers followed by period or parenthesis)
                    import re
                    formatted_answer = re.sub(r'(\d+[\)\.]) ', r'\n\1 ', formatted_answer)

                    # Clean up any triple+ line breaks
                    formatted_answer = re.sub(r'\n{3,}', '\n\n', formatted_answer)

                    yield (
                        gr.HTML(
                            """
                            <div class="analyze-button-container">
                                <span class="status-text">✓ Analysis complete</span>
                            </div>
                            """,
                            visible=True
                        ),
                        gr.Markdown(formatted_answer),
                        analysis["final_answer"]["evidence"],
                        gr.Column(visible=True)
                    )
                except Exception as e:
                    logging.error(f"Analysis error: {str(e)}", exc_info=True)
                    yield (
                        gr.HTML(
                            f"""
                            <div class="analyze-button-container">
                                <span class="status-text">❌ Error: {str(e)}</span>
                            </div>
                            """,
                            visible=True
                        ),
                        gr.Markdown("Analysis failed"),
                        gr.JSON(None),
                        gr.Column(visible=False)
                    )

            # Connect event handlers
            search_btn.click(
                fn=search_handler,
                inputs=[
                    query,
                    engine,
                    num_results,
                    country,
                    language,
                ],
                outputs=[
                    status,
                    results_display,
                    analyze_btn,
                    analysis_status,
                    output_section,
                    search_state  # Update state
                ]
            )

            analyze_btn.click(
                fn=analyze_handler,
                inputs=[search_state],  # Use the state
                outputs=[
                    analysis_status,
                    answer,
                    sources,
                    output_section
                ]
            )

        return interface

#
# End of File
########################################################################################################################
