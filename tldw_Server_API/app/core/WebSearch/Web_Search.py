# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import asyncio
import json
import random
import re
import time
from html import unescape
from typing import Any, Optional, TypedDict
from urllib.parse import unquote, urlencode, urlparse

#
# 3rd-Party Imports
from lxml.etree import _Element
from lxml.html import document_fromstring

#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.http_client import RetryPolicy, fetch
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    get_adapter_or_raise,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article
from tldw_Server_API.app.core.Web_Scraping.ua_profiles import (
    build_browser_headers,
    pick_ua_profile,
)

_WEBSEARCH_PARSE_EXCEPTIONS = (
    AttributeError,
    IndexError,
    KeyError,
    LookupError,
    TypeError,
    ValueError,
)

_WEBSEARCH_RUNTIME_EXCEPTIONS = (
    ChatConfigurationError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    json.JSONDecodeError,
    AttributeError,
    IndexError,
    KeyError,
    LookupError,
    TypeError,
    ValueError,
)


def _set_processing_error(output_dict: dict[str, Any], message: str) -> None:
    output_dict["processing_error"] = message
    logging.error(message)


def _websearch_browser_headers(
        *,
        accept_lang: str = "en-US,en;q=0.5",
        referer: str = "https://www.google.com/",
) -> dict[str, str]:
    profile = pick_ua_profile("fixed")
    headers = build_browser_headers(
        profile=profile,
        accept_lang=accept_lang,
        accept_encoding="gzip, deflate",
    )
    headers.update({
        "Referer": referer,
        "Connection": "keep-alive",
    })
    return headers


def summarize(
        input_data: str,
        custom_prompt_arg: Optional[str] = None,
        api_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temp: float = 0.7,
        system_message: Optional[str] = None,
        streaming: bool = False,
        **extra_kwargs: Any,
) -> str:
    """
    Backwards-compatible summarization helper to keep legacy monkeypatch-based tests working.

    The newer implementation relies on :func:`analyze`, but exposing this wrapper allows unit
    tests (and any downstream code) to replace the summarizer without reaching into internal
    modules. All parameters map directly onto :func:`analyze`.
    """
    return analyze(
        input_data=input_data,
        custom_prompt_arg=custom_prompt_arg,
        api_name=api_name,
        api_key=api_key,
        temp=temp,
        system_message=system_message,
        streaming=streaming,
        **extra_kwargs,
    )


def _build_messages(
        *,
        system_prompt: Optional[str],
        user_prompt: Optional[str],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def _call_adapter_text(
        *,
        api_endpoint: str,
        messages_payload: list[dict[str, Any]],
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        app_config: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **extra_kwargs: Any,
) -> str:
    provider = normalize_provider(api_endpoint)
    if not provider:
        raise ChatConfigurationError(provider=api_endpoint, message="LLM provider is required.")
    cfg = ensure_app_config(app_config or loaded_config_data)
    resolved_model = model or resolve_provider_model(provider, cfg)
    if not resolved_model:
        raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
    system_message, cleaned_messages = split_system_message(messages_payload or [])
    request: dict[str, Any] = {
        "messages": cleaned_messages,
        "system_message": system_message,
        "model": resolved_model,
        "api_key": api_key or resolve_provider_api_key_from_config(provider, cfg),
        "temperature": temperature,
        "app_config": cfg,
    }
    request.update(extra_kwargs)
    response = get_adapter_or_raise(provider).chat(request, timeout=timeout)
    return extract_response_content(response) or str(response)


#
#######################################################################################################################
#
# Functions:
# 1. analyze_question
#
#######################################################################################################################
#
# Functions:

######################### Main Orchestration Workflow #########################

def initialize_web_search_results_dict(search_params: dict) -> dict:
    """
    Initializes and returns a dictionary for storing web search results and metadata.

    Args:
        search_params (Dict): A dictionary containing search parameters.

    Returns:
        Dict: A dictionary initialized with search metadata.
    """
    return {
        "search_engine": search_params.get('engine', 'google'),
        "search_query": "",
        "content_country": search_params.get('content_country', 'US'),
        "search_lang": search_params.get('search_lang', 'en'),
        "output_lang": search_params.get('output_lang', 'en'),
        "result_count": 0,
        "date_range": search_params.get('date_range'),
        "safesearch": search_params.get('safesearch', 'active'),
        "site_blacklist": search_params.get('site_blacklist', []),
        "exactTerms": search_params.get('exactTerms'),
        "excludeTerms": search_params.get('excludeTerms'),
        "filter": search_params.get('filter'),
        "geolocation": search_params.get('geolocation'),
        "search_result_language": search_params.get('search_result_language'),
        "sort_results_by": search_params.get('sort_results_by'),
        "results": [],
        "total_results_found": 0,
        "search_time": 0.0,
        "error": None,
        "processing_error": None
    }


def generate_and_search(question: str, search_params: dict) -> dict:
    """
    Generates sub-queries (if enabled) and performs web searches for each query.

    Args:
        question (str): The user's original question or query.
        search_params (Dict): A dictionary containing parameters for performing web searches
                              and specifying LLM endpoints.

    Returns:
        Dict: A dictionary containing all search results and related metadata.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    logging.info(f"Starting generate_and_search with query: {question}")

    # Validate input parameters
    if not question or not isinstance(question, str):
        raise ValueError("Invalid question parameter")
    if not search_params or not isinstance(search_params, dict):
        raise ValueError("Invalid search_params parameter")

    # Check for required keys in search_params
    required_keys = ["engine", "content_country", "search_lang", "output_lang", "result_count"]
    for key in required_keys:
        if key not in search_params:
            raise ValueError(f"Missing required key in search_params: {key}")

    # 1. Generate sub-queries if requested
    logging.info(f"Generating sub-queries for the query: {question}")
    sub_query_dict = {
        "main_goal": question,
        "sub_questions": [],
        "search_queries": [],
        "analysis_prompt": None
    }

    if search_params.get("subquery_generation", False):
        logging.info("Sub-query generation enabled")
        api_endpoint = search_params.get("subquery_generation_llm", "openai")
        sub_query_dict = analyze_question(question, api_endpoint)

    # Merge original question with sub-queries
    sub_queries = sub_query_dict.get("sub_questions", [])
    logging.info(f"Sub-queries generated: {sub_queries}")
    all_queries = [question] + sub_queries

    # 2. Initialize a single web_search_results_dict
    web_search_results_dict = initialize_web_search_results_dict(search_params)
    web_search_results_dict["search_query"] = question

    # 3. Perform searches and accumulate all raw results
    for q in all_queries:
        logging.info(f"Performing web search for query: {q}")
        raw_results = perform_websearch(
            search_engine=search_params.get('engine'),
            search_query=q,
            content_country=search_params.get('content_country', 'US'),
            search_lang=search_params.get('search_lang', 'en'),
            output_lang=search_params.get('output_lang', 'en'),
            result_count=search_params.get('result_count', 10),
            date_range=search_params.get('date_range'),
            safesearch=search_params.get('safesearch', 'active'),
            site_blacklist=search_params.get('site_blacklist', []),
            exactTerms=search_params.get('exactTerms'),
            excludeTerms=search_params.get('excludeTerms'),
            filter=search_params.get('filter'),
            geolocation=search_params.get('geolocation'),
            search_result_language=search_params.get('search_result_language'),
            sort_results_by=search_params.get('sort_results_by')
        )

        # Debug: Inspect raw results
        logging.debug(f"Raw results for query '{q}': {raw_results}")

        # Check for errors or invalid data
        if not isinstance(raw_results, dict) or raw_results.get("processing_error"):
            logging.error(f"Error or invalid data returned for query '{q}': {raw_results}")
            continue

        logging.info(f"Search results found for query '{q}': {len(raw_results.get('results', []))}")

        # Append results to the single web_search_results_dict
        web_search_results_dict["results"].extend(raw_results["results"])
        web_search_results_dict["total_results_found"] += raw_results.get("total_results_found", 0)
        web_search_results_dict["search_time"] += raw_results.get("search_time", 0.0)
        logging.info(f"Total results found so far: {len(web_search_results_dict['results'])}")

    return {
        "web_search_results_dict": web_search_results_dict,
        "sub_query_dict": sub_query_dict
    }


async def analyze_and_aggregate(
        web_search_results_dict: dict,
        sub_query_dict: dict,
        search_params: dict,
        cancel_event: Optional[asyncio.Event] = None
) -> dict:
    logging.info("Starting analyze_and_aggregate")

    # 4. Score/filter results
    logging.info("Scoring and filtering search results")
    sub_questions = sub_query_dict.get("sub_questions", [])
    relevant_results = await search_result_relevance(
        web_search_results_dict["results"],
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('relevance_analysis_llm'),
        cancel_event=cancel_event,
    )
    logging.debug("Relevant results returned by search_result_relevance:")
    logging.debug(json.dumps(relevant_results, indent=2))

    # 5. Allow user to review and select relevant results (if enabled)
    logging.info("Reviewing and selecting relevant results")
    if search_params.get("user_review", False):
        logging.info("User review enabled")
        relevant_results = review_and_select_results({"results": list(relevant_results.values())})

    # 6. Summarize/aggregate final answer
    final_answer = aggregate_results(
        relevant_results,
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('final_answer_llm')
    )

    if not isinstance(final_answer.get("text"), str):
        raise ValueError("Aggregation produced an invalid final_answer payload")

    # 7. Return the final data
    logging.info("Returning final websearch results")
    return {
        "final_answer": final_answer,
        "relevant_results": relevant_results,
        "web_search_results_dict": web_search_results_dict
    }




######################### Question Analysis #########################
#
#
def analyze_question(question: str, api_endpoint) -> dict:
    logging.debug(f"Analyzing question: {question} with API endpoint: {api_endpoint}")
    """
    Analyzes the input question and generates sub-questions

    Returns:
        Dict containing:
        - main_goal: str
        - sub_questions: List[str]
        - search_queries: List[str]
        - analysis_prompt: str
    """
    original_query = question
    sub_question_generation_prompt = f"""
            You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. Your goal is to generate queries that are diverse, specific, and highly relevant to the original query, ensuring comprehensive coverage of the topic.

            Important instructions:
            1. Generate between 2 and 6 queries unless a fixed count is specified. Generate more queries for complex or multifaceted topics and fewer for simple or straightforward ones.
            2. Ensure the queries are diverse, covering different aspects or perspectives of the original query, while remaining highly relevant to its core intent.
            3. Prefer specific queries over general ones, as they are more likely to yield targeted and useful results.
            4. If the query involves comparing two topics, generate separate queries for each topic.
            5. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
            6. If the original query is broad or ambiguous, generate queries that explore specific subtopics or clarify the intent.
            7. If the query is too specific or unclear, generate queries that explore related or broader topics to ensure useful results.
            8. Return the queries as a JSON array in the format ["query_1", "query_2", ...].

            Examples:
            1. For the query "What are the benefits of exercise?", generate queries like:
               ["health benefits of physical activity", "mental health benefits of exercise", "long-term effects of regular exercise", "how exercise improves cardiovascular health", "role of exercise in weight management"]

            2. For the query "Compare Python and JavaScript", generate queries like:
               ["key features of Python programming language", "advantages of JavaScript for web development", "use cases for Python vs JavaScript", "performance comparison of Python and JavaScript", "ease of learning Python vs JavaScript"]

            3. For the query "How does climate change affect biodiversity?", generate queries like:
               ["impact of climate change on species extinction", "effects of global warming on ecosystems", "role of climate change in habitat loss", "how rising temperatures affect marine biodiversity", "climate change and its impact on migratory patterns"]

            4. For the query "Best practices for remote work", generate queries like:
               ["tips for staying productive while working from home", "how to maintain work-life balance in remote work", "tools for effective remote team collaboration", "managing communication in remote teams", "ergonomic setup for home offices"]

            5. For the query "What is quantum computing?", generate queries like:
               ["basic principles of quantum computing", "applications of quantum computing in real-world problems", "difference between classical and quantum computing", "key challenges in developing quantum computers", "future prospects of quantum computing"]

            Original query: {original_query}
            """

    input_data = "Follow the above instructions."

    sub_questions: list[str] = []
    for attempt in range(3):
        try:
            logging.info(f"Generating sub-questions (attempt {attempt + 1})")

            messages_payload = _build_messages(
                system_prompt=sub_question_generation_prompt,
                user_prompt=input_data,
            )
            response = _call_adapter_text(
                api_endpoint=api_endpoint,
                messages_payload=messages_payload,
                temperature=0.7,
                app_config=loaded_config_data,
            )
            if response:
                try:
                    # Try to parse as JSON first
                    parsed_response = json.loads(response)
                    sub_questions = parsed_response.get("sub_questions", [])
                    if sub_questions:
                        logging.info("Successfully generated sub-questions from JSON")
                        break
                except json.JSONDecodeError:
                    # If JSON parsing fails, attempt a regex-based fallback
                    logging.warning("Failed to parse as JSON. Attempting regex extraction.")
                    matches = re.findall(r'"([^"]*)"', response)
                    sub_questions = matches if matches else []
                    if sub_questions:
                        logging.info("Successfully extracted sub-questions using regex")
                        break

        except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
            logging.error(f"Error generating sub-questions: {str(e)}")

    if not sub_questions:
        logging.error("Failed to extract sub-questions from API response after all attempts.")
        sub_questions = [original_query]  # Fallback to the original query

    # Construct and return the result dictionary
    logging.info("Sub-questions generated successfully")
    return {
        "main_goal": original_query,
        "sub_questions": sub_questions,
        "search_queries": sub_questions,
        "analysis_prompt": sub_question_generation_prompt
    }


######################### Relevance Analysis #########################
#
# TODO(websearch): Transition relevance parsing to structured outputs to reduce regex fragility.
async def search_result_relevance(
        search_results: list[dict],
        original_question: str,
        sub_questions: list[str],
        api_endpoint: str,
        cancel_event: Optional[asyncio.Event] = None
) -> dict[str, dict]:
    """
    Evaluate whether each search result is relevant to the original question and sub-questions.

    Args:
        search_results (List[Dict]): List of search results to evaluate.
        original_question (str): The original question posed by the user.
        sub_questions (List[str]): List of sub-questions generated from the original question.
        api_endpoint (str): The LLM or API endpoint to use for relevance analysis.

    Returns:
        Dict[str, Dict]: A dictionary of relevant results, keyed by a unique ID or index.
    """
    relevant_results = {}

    # Summarization prompt template
    summarization_prompt = """
    Summarize the following text in a concise way that captures the key information relevant to this question: "{question}"

    Text to summarize:
    {content}

    Instructions:
    1. Focus on information relevant to the question
    2. Keep the summary under 2000 characters
    3. Maintain factual accuracy
    4. Include key details and statistics if present
    """

    for idx, result in enumerate(search_results):
        if cancel_event and cancel_event.is_set():
            logging.info("Cancellation requested; stopping relevance evaluation")
            break
        content = result.get("content", "")
        if not content:
            logging.error("No Content found in search results array!")
            continue

        # First, evaluate relevance
        eval_prompt = f"""
                Given the following search results for the user's question: "{original_question}" and the generated sub-questions: {sub_questions}, evaluate the relevance of the search result to the user's question.
                Explain your reasoning for selection.

                Search Results:
                {content}

                Instructions:
                1. You MUST only answer TRUE or False while providing your reasoning for your answer.
                2. A result is relevant if the result most likely contains comprehensive and relevant information to answer the user's question.
                3. Provide a brief reason for selection.

                You MUST respond using EXACTLY this format and nothing else:

                Selected Answer: [True or False]
                Reasoning: [Your reasoning for the selections]
                """
        input_data = "Evaluate the relevance of the search result."
        messages_payload = _build_messages(
            system_prompt=eval_prompt,
            user_prompt=input_data,
        )

        try:
            # Add delay to avoid rate limiting
            sleep_time = random.uniform(0.2, 0.6)
            await asyncio.sleep(sleep_time)

            if cancel_event and cancel_event.is_set():
                logging.info("Cancellation detected after delay; aborting relevance evaluation")
                break

            # Evaluate relevance
            relevancy_result = _call_adapter_text(
                api_endpoint=api_endpoint,
                messages_payload=messages_payload,
                temperature=0.7,
                app_config=loaded_config_data,
            )

            logging.debug(f"[DEBUG] Relevancy LLM response for index {idx}:\n{relevancy_result}\n---")

            if relevancy_result:
                # Extract the selected answer and reasoning via regex
                logging.debug("LLM relevancy response for item %s: %s", idx, relevancy_result)
                selected_answer_match = re.search(
                    r"Selected Answer:\s*(True|False)",
                    relevancy_result,
                    re.IGNORECASE
                )
                reasoning_match = re.search(
                    r"Reasoning:\s*(.+)",
                    relevancy_result,
                    re.IGNORECASE
                )

                if selected_answer_match and reasoning_match:
                    is_relevant = selected_answer_match.group(1).strip().lower() == "true"
                    reasoning = reasoning_match.group(1).strip()

                    if is_relevant:
                        logging.debug("Relevant result found.")
                        # Use the 'id' from the result if available, otherwise use idx
                        result_id = result.get("id", str(idx))
                        # Scrape the content of the relevant result
                        scraped_content = await scrape_article(result['url'])

                        # Create Summarization prompt
                        logging.debug(f"Creating Summarization Prompt for result idx={idx}")
                        summary_prompt = summarization_prompt.format(
                            question=original_question,
                            content=scraped_content['content']
                        )

                        # Add delay before summarization
                        await asyncio.sleep(sleep_time)

                        # Generate summary using the summarize function
                        logging.info(f"Summarizing relevant result: ID={result_id}")
                        summary = summarize(
                            input_data=scraped_content['content'],
                            custom_prompt_arg=summary_prompt,
                            api_name=api_endpoint,
                            api_key=None,
                            temp=0.7,
                            system_message=None,
                            streaming=False
                        )

                        relevant_results[result_id] = {
                            "content": summary,  # Store the summary instead of full content
                            "original_content": scraped_content['content'],  # Keep original content if needed
                            "reasoning": reasoning
                        }
                        logging.info(f"Relevant result found and summarized: ID={result_id}; Reasoning={reasoning}")
                    else:
                        logging.info(f"Irrelevant result: {reasoning}")

                else:
                    logging.warning("Failed to parse the API response for relevance analysis.")
        except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
            logging.error(f"Error during relevance evaluation/summarization for result idx={idx}: {e}")

    return relevant_results


def review_and_select_results(web_search_results_dict: dict) -> dict:
    """
    Allows the user to review and select relevant results from the search results.

    Args:
        web_search_results_dict (Dict): The dictionary containing all search results.

    Returns:
        Dict: A dictionary containing only the user-selected relevant results.
    """
    relevant_results = {}
    print("Review the search results and select the relevant ones:")
    for idx, result in enumerate(web_search_results_dict["results"]):
        print(f"\nResult {idx + 1}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content'][:200]}...")  # Show a preview of the content
        user_input = input("Is this result relevant? (y/n): ").strip().lower()
        if user_input == 'y':
            relevant_results[str(idx)] = result

    return relevant_results


######################### Result Aggregation & Combination #########################
#
class FinalAnswerDict(TypedDict):
    """Structured payload returned by the aggregation phase."""
    text: str
    evidence: list[dict[str, Any]]
    confidence: float
    chunks: list[dict[str, Any]]


def aggregate_results(
        relevant_results: dict[str, dict],
        question: str,
        sub_questions: list[str],
        api_endpoint: Optional[str]
) -> FinalAnswerDict:
    """
    Combines and summarizes relevant results into a final answer.

    Args:
        relevant_results (Dict[str, Dict]): Dictionary of relevant articles/content.
        question (str): Original question.
        sub_questions (List[str]): List of sub-questions.
        api_endpoint (str): LLM or API endpoint for summarization.

    Returns:
        Dict containing:
        - summary (str): Final summarized answer.
        - evidence (List[Dict]): List of relevant content items included in the summary.
        - confidence (float): A rough confidence score (placeholder).
    """
    logging.info("Aggregating and summarizing relevant results")
    if not relevant_results:
        no_results: FinalAnswerDict = {
            "text": "No relevant results found. Unable to provide an answer.",
            "evidence": [],
            "confidence": 0.0,
            "chunks": [],
        }
        return no_results

    logging.info("Summarizing relevant results")

    def _build_chunk_infos(
            items: list[tuple[str, dict[str, Any]]],
            max_chars: int = 6000
    ) -> list[dict[str, Any]]:
        chunk_infos: list[dict[str, Any]] = []
        current_entries: list[tuple[str, str]] = []
        current_length = 0

        def flush_entries() -> None:
            nonlocal current_entries, current_length
            if not current_entries:
                return
            text = "\n\n".join(entry for _, entry in current_entries)
            chunk_infos.append({
                "index": len(chunk_infos) + 1,
                "result_ids": [rid for rid, _ in current_entries],
                "text": text,
                "truncated": False,
            })
            current_entries = []
            current_length = 0

        for rid, res in items:
            entry = f"ID: {rid}\nContent: {res.get('content', '')}\nReasoning: {res.get('reasoning', '')}"
            entry_length = len(entry)
            if entry_length >= max_chars:
                flush_entries()
                chunk_infos.append({
                    "index": len(chunk_infos) + 1,
                    "result_ids": [rid],
                    "text": entry[:max_chars],
                    "truncated": True,
                })
                continue

            if current_length + entry_length > max_chars and current_entries:
                flush_entries()

            current_entries.append((rid, entry))
            current_length += entry_length

        flush_entries()
        return chunk_infos

    def _estimate_confidence(
            relevant_count: int,
            chunk_count: int,
            failed_chunks: int,
            has_llm: bool
    ) -> float:
        if relevant_count <= 0:
            return 0.0
        coverage = min(relevant_count, 10) / 10.0
        chunk_success = 1.0 if chunk_count == 0 else (chunk_count - failed_chunks) / chunk_count
        base = 0.35 + 0.45 * coverage
        modifier = 0.6 + 0.4 * chunk_success
        llm_bonus = 0.1 if has_llm and failed_chunks == 0 else (0.05 if has_llm else 0.0)
        confidence = base * modifier + llm_bonus
        return max(0.1, min(0.99, round(confidence, 3)))

    result_items = list(relevant_results.items())
    chunk_infos = _build_chunk_infos(result_items)
    chunk_assignments: dict[str, int] = {}
    for info in chunk_infos:
        for rid in info["result_ids"]:
            chunk_assignments[rid] = info["index"]

    chunk_metadata: list[dict[str, Any]] = []
    evidence_payload: list[dict[str, Any]] = []

    for rid, res in relevant_results.items():
        evidence_payload.append({
            "id": rid,
            "content": res.get("content"),
            "original_content": res.get("original_content"),
            "reasoning": res.get("reasoning"),
            "chunk_index": chunk_assignments.get(rid),
        })

    if not api_endpoint:
        logging.warning("No final answer LLM configured; returning evidence summaries only.")
        for info in chunk_infos:
            preview = info["text"][:1500]
            chunk_metadata.append({
                "chunk_index": info["index"],
                "result_ids": info["result_ids"],
                "summary": preview,
                "generated": False,
                "source_characters": len(info["text"]),
                "truncated_source": info["truncated"],
            })
        combined_text = "\n\n".join(entry.get("content", "") or "" for entry in relevant_results.values())
        fallback_answer: FinalAnswerDict = {
            "text": combined_text or "Unable to generate a final answer without an LLM.",
            "evidence": evidence_payload,
            "confidence": _estimate_confidence(len(evidence_payload), len(chunk_infos), 0, has_llm=False),
            "chunks": chunk_metadata,
        }
        return fallback_answer

    summarized_chunks: list[str] = []
    failed_chunks = 0

    for info in chunk_infos:
        chunk_prompt = f"""
            Summarize the following set of relevant search snippets into a concise digest that preserves
            high-signal facts for answering the question: "{question}".

            Requirements:
            1. Keep the summary under 1500 characters.
            2. Focus on verifiable facts and key statistics.
            3. Mention the reasoning tags when helpful.

            <chunk id="{info['index']}">
            {info['text']}
            </chunk>
            """
        try:
            chunk_summary = summarize(
                input_data=info["text"],
                custom_prompt_arg=chunk_prompt,
                api_name=api_endpoint,
                api_key=None,
                temp=0.3,
                system_message=None,
                streaming=False
            )
            generated = True
        except _WEBSEARCH_RUNTIME_EXCEPTIONS as chunk_error:
            failed_chunks += 1
            logging.warning(f"Chunk summarization failed for chunk {info['index']}: {chunk_error}")
            chunk_summary = info["text"][:1500]
            generated = False

        chunk_metadata.append({
            "chunk_index": info["index"],
            "result_ids": info["result_ids"],
            "summary": chunk_summary,
            "generated": generated,
            "source_characters": len(info["text"]),
            "truncated_source": info["truncated"],
        })
        summarized_chunks.append(f"Chunk {info['index']} Summary:\n{chunk_summary}")

    context_payload = "\n\n".join(summarized_chunks)
    current_date = time.strftime("%Y-%m-%d")

    # Aggregation Prompt #1

    # Aggregation Prompt #2
    analyze_search_results_prompt_2 = (
        """INITIAL_QUERY: Here are some sources {context_payload}. Read these carefully, as you will be asked a Query about them.
        # General Instructions

        Write an accurate, detailed, and comprehensive response to the user's query located at INITIAL_QUERY. Additional context is provided as "USER_INPUT" after specific questions. Your answer should be informed by the provided "Search results". Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone. Your answer must be written in the same language as the query, even if language preference is different.

        You MUST cite the most relevant search results that answer the query. Do not mention any irrelevant results. You MUST ADHERE to the following instructions for citing search results:
        - to cite a search result, enclose its index located above the summary with brackets at the end of the corresponding sentence, for example "Ice is less dense than water[1][2]." or "Paris is the capital of France[1][4][5]."
        - NO SPACE between the last word and the citation, and ALWAYS use brackets. Only use this format to cite search results. NEVER include a References section at the end of your answer.
        - If you don't know the answer or the premise is incorrect, explain why.
        If the search results are empty or unhelpful, answer the query as well as you can with existing knowledge.

        You MUST NEVER use moralization or hedging language. AVOID using the following phrases:
        - "It is important to ..."
        - "It is inappropriate ..."
        - "It is subjective ..."

        You MUST ADHERE to the following formatting instructions:
        - Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
        - Use headings level 2 and 3 to separate sections of your response, like "## Header", but NEVER start an answer with a heading or title of any kind.
        - Use single new lines for lists and double new lines for paragraphs.
        - Use markdown to render images given in the search results.
        - NEVER write URLs or links.

        # Query type specifications

        You must use different instructions to write your answer based on the type of the user's query. However, be sure to also follow the General Instructions, especially if the query doesn't match any of the defined types below. Here are the supported types.

        ## Academic Research

        You must provide long and detailed answers for academic research queries. Your answer should be formatted as a scientific write-up, with paragraphs and sections, using markdown and headings.

        ## Recent News

        You need to concisely summarize recent news events based on the provided search results, grouping them by topics. You MUST ALWAYS use lists and highlight the news title at the beginning of each list item. You MUST choose news from diverse perspectives while also prioritizing trustworthy sources. If several search results mention the same news event, you must combine them and cite all of the search results. Prioritize more recent events, ensuring to compare timestamps. You MUST NEVER start your answer with a heading of any kind.

        ## Weather

        Your answer should be very short and only provide the weather forecast. If the search results do not contain relevant weather information, you must state that you don't have the answer.

        ## People

        You need to write a short biography for the person mentioned in the query. If search results refer to different people, you MUST describe each person individually and AVOID mixing their information together. NEVER start your answer with the person's name as a header.

        ## Coding

        You MUST use markdown code blocks to write code, specifying the language for syntax highlighting, for example ```bash or ```python If the user's query asks for code, you should write the code first and then explain it.

        ## Cooking Recipes

        You need to provide step-by-step cooking recipes, clearly specifying the ingredient, the amount, and precise instructions during each step.

        ## Translation

        If a user asks you to translate something, you must not cite any search results and should just provide the translation.

        ## Creative Writing

        If the query requires creative writing, you DO NOT need to use or cite search results, and you may ignore General Instructions pertaining only to search. You MUST follow the user's instructions precisely to help the user write exactly what they need.

        ## Science and Math

        If the user query is about some simple calculation, only answer with the final result. Follow these rules for writing formulas:
        - Always use \\( and\\) for inline formulas and\\[ and\\] for blocks, for example\\(x^4 = x - 3 \\)
        - To cite a formula add citations to the end, for example\\[ \\sin(x) \\] [1][2] or \\(x^2-2\\) [4].
        - Never use $ or $$ to render LaTeX, even if it is present in the user query.
        - Never use unicode to render math expressions, ALWAYS use LaTeX.
        - Never use the \\label instruction for LaTeX.

        ## URL Lookup

        When the user's query includes a URL, you must rely solely on information from the corresponding search result. DO NOT cite other search results, ALWAYS cite the first result, e.g. you need to end with [1]. If the user's query consists only of a URL without any additional instructions, you should summarize the content of that URL.

        ## Shopping

        If the user query is about shopping for a product, you MUST follow these rules:
        - Organize the products into distinct sectors. For example, you could group shoes by style (boots, sneakers, etc.)
        - Cite at most 9 search results using the format provided in General Instructions to avoid overwhelming the user with too many options.

        The current date is: {current_date}

        The user's query is: {question}
        """.format(
            context_payload=context_payload,
            current_date=current_date,
            question=question,
        )  # nosec B608
    )

    input_data = "Follow the above instructions."
    messages_payload = _build_messages(
        system_prompt=analyze_search_results_prompt_2,
        user_prompt=input_data,
    )

    try:
        logging.info("Generating the report")
        returned_response = _call_adapter_text(
            api_endpoint=api_endpoint,
            messages_payload=messages_payload,
            temperature=0.7,
            app_config=loaded_config_data,
        )
        logging.debug("Returned response from LLM for aggregation: %s", returned_response)
        if returned_response:
            success_answer: FinalAnswerDict = {
                "text": returned_response,
                "evidence": evidence_payload,
                "confidence": _estimate_confidence(
                    len(evidence_payload),
                    len(chunk_infos),
                    failed_chunks,
                    has_llm=True,
                ),
                "chunks": chunk_metadata,
            }
            return success_answer
    except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
        logging.error(f"Error aggregating results: {e}")

    logging.error("Could not create the report due to an error.")
    failure_answer: FinalAnswerDict = {
        "text": "Could not create the report due to an error.",
        "evidence": evidence_payload,
        "confidence": _estimate_confidence(
            len(evidence_payload),
            len(chunk_infos),
            failed_chunks=len(chunk_infos),
            has_llm=False,
        ),
        "chunks": chunk_metadata,
    }
    return failure_answer


#
# End of Orchestration functions
#######################################################################################################################


#######################################################################################################################
#
# Search Engine Functions
def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count,
                      date_range=None,
                      safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None,
                      geolocation=None, search_result_language=None, sort_results_by=None):
    try:
        search_engines_cfg = loaded_config_data.get('search_engines', {})

        if search_engine.lower() == "baidu":
            raise NotImplementedError("Baidu search is not implemented")

        elif search_engine.lower() == "bing":
            # Prepare the arguments for search_web_bing
            bing_args = {
                "search_query": search_query,
                "bing_lang": search_lang,
                "bing_country": content_country,
                "result_count": result_count,
                "bing_api_key": search_engines_cfg.get('bing_search_api_key'),
                # Fetch Bing API key from config
                "date_range": date_range,
            }

            # Call the search_web_bing function with the prepared arguments
            web_search_results = search_web_bing(**bing_args)

        elif search_engine.lower() == "brave":
            web_search_results = search_web_brave(
                search_term=search_query,
                country=content_country,
                search_lang=search_lang,
                ui_lang=output_lang,
                result_count=result_count,
                safesearch=safesearch or "moderate",
                date_range=date_range,
                site_blacklist=site_blacklist,
            )

        elif search_engine.lower() == "duckduckgo":
            # Prepare the arguments for search_web_duckduckgo
            ddg_args = {
                "keywords": search_query,
                "region": f"{content_country.lower()}-{search_lang.lower()}",  # Format: "us-en"
                "timelimit": date_range[0] if date_range else None,
                # Use first character of date_range (e.g., "y" -> "y")
                "max_results": result_count,
            }

            # Call the search_web_duckduckgo function with the prepared arguments
            ddg_results = search_web_duckduckgo(**ddg_args)

            # Wrap the results in a dictionary to match the expected format
            web_search_results = {"results": ddg_results}

        elif search_engine.lower() == "google":
            site_blacklist_list = site_blacklist if isinstance(site_blacklist, list) else None
            if site_blacklist_list:
                site_blacklist_value: Optional[str] = ",".join(site_blacklist_list)
            elif isinstance(site_blacklist, str):
                site_blacklist_value = site_blacklist
            else:
                site_blacklist_value = None

            # Prepare the arguments for search_web_google
            google_args = {
                "search_query": search_query,
                "google_search_api_key": loaded_config_data['search_engines']['google_search_api_key'],
                "google_search_engine_id": loaded_config_data['search_engines']['google_search_engine_id'],
                "result_count": result_count,
                "c2coff": "1",  # Default value
                "results_origin_country": content_country,
                "ui_language": output_lang,
                "search_result_language": search_result_language or "lang_en",  # Default value
                "geolocation": geolocation or "us",  # Default value
                "safesearch": safesearch or "off",  # Default value,
            }

            # If site_blacklist has multiple domains, do not use siteSearch
            if site_blacklist_list and len(site_blacklist_list) == 1:
                google_args["siteSearch"] = site_blacklist_list[0]
                google_args["siteSearchFilter"] = "e"
            elif isinstance(site_blacklist, str) and site_blacklist:
                google_args["siteSearch"] = site_blacklist
                google_args["siteSearchFilter"] = "e"

            # Add optional parameters only if they are provided
            if date_range:
                google_args["date_range"] = date_range
            if exactTerms:
                google_args["exactTerms"] = exactTerms
            if excludeTerms:
                google_args["excludeTerms"] = excludeTerms
            if filter:
                google_args["filter"] = filter
            if site_blacklist_value:
                google_args["site_blacklist"] = site_blacklist_value
            if sort_results_by:
                google_args["sort_results_by"] = sort_results_by

            # Call the search_web_google function with the prepared arguments
            web_search_results = search_web_google(**google_args)  # raw JSON
            web_search_results_dict = process_web_search_results(web_search_results, "google")
            return web_search_results_dict

        elif search_engine.lower() == "kagi":
            web_search_results = search_web_kagi(query=search_query, limit=result_count)

        elif search_engine.lower() == "serper":
            raise NotImplementedError("Serper search is not implemented")

        elif search_engine.lower() == "tavily":
            raise NotImplementedError("Tavily search is not implemented")

        elif search_engine.lower() == "searx":
            raise NotImplementedError("Searx search is not implemented")

        elif search_engine.lower() == "yandex":
            raise NotImplementedError("Yandex search is not implemented")

        else:
            return f"Error: Invalid Search Engine Name {search_engine}"

        web_search_results_dict = process_web_search_results(web_search_results, search_engine)
        return web_search_results_dict

    except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
        return {"processing_error": "Error performing web search"}

#
######################### Search Result Parsing ##################################################################
#

def process_web_search_results(search_results: dict, search_engine: str) -> dict:
    """
    Processes search results from a search engine and formats them into a standardized dictionary structure.

    Args:
        search_results (Dict): The raw search results from the search engine.
        search_engine (str): The name of the search engine (e.g., "Google", "Bing").

    Returns:
        Dict: A dictionary containing the processed search results in the specified structure.

    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [
            {
                "title": str,
                "url": str,
                "content": str,
                "metadata": {
                    "date_published": Optional[str],
                    "author": Optional[str],
                    "source": Optional[str],
                    "language": Optional[str],
                    "relevance_score": Optional[float],
                    "snippet": Optional[str]
                }
            },
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
    }
    """
    # Validate input parameters
    if not isinstance(search_results, dict):
        raise TypeError("search_results must be a dictionary")

    # Initialize the output dictionary with default values
    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range"),
        "safesearch": search_results.get("safesearch"),
        "site_blacklist": search_results.get("site_blacklist"),
        "exactTerms": search_results.get("exactTerms"),
        "excludeTerms": search_results.get("excludeTerms"),
        "filter": search_results.get("filter"),
        "geolocation": search_results.get("geolocation"),
        "search_result_language": search_results.get("search_result_language"),
        "sort_results_by": search_results.get("sort_results_by"),
        "results": [],
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error"),
        "processing_error": None
    }
    try:
        # Parse results based on the search engine
        if search_engine.lower() == "baidu":
            pass  # Placeholder for Baidu-specific parsing
        elif search_engine.lower() == "bing":
            parse_bing_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "brave":
            parse_brave_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "duckduckgo":
            parse_duckduckgo_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "google":
            parse_google_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "kagi":
            parse_kagi_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "serper":
            parse_serper_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "tavily":
            parse_tavily_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "searx":
            parse_searx_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "yandex":
            parse_yandex_results(search_results, web_search_results_dict)
        else:
            raise ValueError(f"Error: Invalid Search Engine Name {search_engine}")

    except _WEBSEARCH_PARSE_EXCEPTIONS as e:
        error_text = str(e)
        if error_text.startswith("Error: Invalid Search Engine Name "):
            _set_processing_error(web_search_results_dict, error_text)
        else:
            _set_processing_error(web_search_results_dict, "Error processing search results")

    return web_search_results_dict


def parse_html_search_results_generic(soup):
    results = []
    for result in soup.find_all('div', class_='result'):
        title = result.find('h3').text if result.find('h3') else ''
        url = result.find('a', class_='url')['href'] if result.find('a', class_='url') else ''
        content = result.find('p', class_='content').text if result.find('p', class_='content') else ''
        published_date = result.find('span', class_='published_date').text if result.find('span',
                                                                                          class_='published_date') else ''

        results.append({
            'title': title,
            'url': url,
            'content': content,
            'publishedDate': published_date
        })
    return results


######################### Baidu Search #########################
#
# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
# https://oxylabs.io/blog/how-to-scrape-baidu-search-results
def search_web_baidu(arg1, arg2, arg3):
    pass


def search_parse_baidu_results():
    pass


######################### Bing Search #########################
#
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview0
# https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/overview
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
# https://github.com/Azure-Samples/cognitive-services-REST-api-samples/tree/master/python/Search
def search_web_bing(search_query, bing_lang, bing_country, result_count=None, bing_api_key=None,
                    date_range=None):
    # Load Search API URL from config file
    search_url = loaded_config_data['search_engines']['bing_search_api_url']

    if not bing_api_key:
        # load key from config file
        bing_api_key = loaded_config_data['search_engines']['bing_search_api_key']
        if not bing_api_key:
            raise ValueError("Please Configure a valid Bing Search API key")

    if not result_count:
        # Perform check in config file for default search result count
        loaded_config_data['search_engines']['search_result_max']
    else:
        pass

    # date_range = "day", "week", "month", or `YYYY-MM-DD..YYYY-MM-DD`
    if not date_range:
        date_range = None

    # Language settings
    if not bing_lang:
        # do config check for default search language
        pass

    # Returns content for this Country market code
    if not bing_country:
        # do config check for default search country
        bing_country = loaded_config_data['search_engines']['bing_country_code']
    else:
        pass
    # Construct a request
    mkt = 'en-US'
    params = {'q': search_query, 'mkt': mkt}
    #    params = {"q": search_query, "mkt": bing_country, "textDecorations": True, "textFormat": "HTML", "count": answer_count,
    #             "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}

    # Call the API
    try:
        response = fetch(method="GET", url=search_url, headers=headers, params=params)
        logging.debug("Headers:  ")
        logging.debug(response.headers)

        logging.debug("JSON Response: ")
        logging.debug(response.json())
        bing_search_results = response.json()
        return bing_search_results
    except Exception:
        raise


def parse_bing_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Bing search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Bing API response
        output_dict (Dict): Dictionary to store processed results
    """
    logging.info(f"Raw Bing results received: {json.dumps(raw_results, indent=2)}")
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract web pages results
        if "webPages" in raw_results:
            web_pages = raw_results["webPages"]
            output_dict["total_results_found"] = web_pages.get("totalEstimatedMatches", 0)

            for result in web_pages.get("value", []):
                processed_result = {
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "content": result.get("snippet", ""),
                    "metadata": {
                        "date_published": None,  # Bing doesn't typically provide this
                        "author": None,  # Bing doesn't typically provide this
                        "source": result.get("displayUrl", None),
                        "language": None,  # Could be extracted from result.get("language") if available
                        "relevance_score": None,  # Could be calculated from result.get("rank") if available
                        "snippet": result.get("snippet", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Optionally process other result types
        if "news" in raw_results:
            for news_item in raw_results["news"].get("value", []):
                processed_result = {
                    "title": news_item.get("name", ""),
                    "url": news_item.get("url", ""),
                    "content": news_item.get("description", ""),
                    "metadata": {
                        "date_published": news_item.get("datePublished", None),
                        "author": news_item.get("provider", [{}])[0].get("name", None),
                        "source": news_item.get("provider", [{}])[0].get("name", None),
                        "language": None,
                        "relevance_score": None,
                        "snippet": news_item.get("description", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Add spell suggestions if available
        if "spellSuggestion" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spellSuggestion"]

        # Add related searches if available
        if "relatedSearches" in raw_results:
            output_dict["related_searches"] = [
                item.get("text", "")
                for item in raw_results["relatedSearches"].get("value", [])
            ]

    except _WEBSEARCH_PARSE_EXCEPTIONS:
        _set_processing_error(output_dict, "Error processing Bing results")


######################### Brave Search #########################
#
# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(
        search_term: str,
        country: Optional[str],
        search_lang: Optional[str],
        ui_lang: Optional[str],
        result_count: Optional[int],
        safesearch: Optional[str] = "moderate",
        brave_api_key: Optional[str] = None,
        result_filter: Optional[str] = None,
        search_type: str = "ai",
        date_range: Optional[str] = None,
        site_blacklist: Optional[list[str]] = None
) -> dict[str, Any]:
    search_url = "https://api.search.brave.com/res/v1/web/search"

    search_engines_cfg = loaded_config_data.get("search_engines", {})
    if search_type not in {"ai", "web"}:
        raise ValueError("Invalid search type. Please choose 'ai' or 'web'.")

    if not brave_api_key:
        key_name = "brave_search_ai_api_key" if search_type == "ai" else "brave_search_api_key"
        brave_api_key = search_engines_cfg.get(key_name)
    if not brave_api_key:
        raise ValueError("Please provide a valid Brave Search API subscription key")

    country = country or search_engines_cfg.get("search_engine_country_code_brave", "US")
    search_lang = search_lang or "en"
    ui_lang = ui_lang or "en"
    result_count = result_count or search_engines_cfg.get("search_result_max_per_query", 10)
    safesearch = (safesearch or "moderate").capitalize()
    result_filter = result_filter or "webpages"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": brave_api_key
    }

    params: dict[str, Any] = {
        "q": search_term,
        "count": result_count,
        "freshness": date_range,
        "promote": result_filter,
        "safeSearch": safesearch,
        "source": search_type,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
    }

    if site_blacklist:
        params["exclude_sites"] = ",".join(site_blacklist)

    # Drop None values to keep the request clean
    filtered_params = {key: value for key, value in params.items() if value is not None}

    # Use wrapper seam to allow clean monkeypatching in tests while using central client in production
    response = brave_http_get(search_url, headers=headers, params=filtered_params)
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results


def parse_brave_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Brave search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Brave API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract query information
        if "query" in raw_results:
            query_info = raw_results["query"]
            output_dict.update({
                "search_query": query_info.get("original", ""),
                "content_country": query_info.get("country", ""),
                "city": query_info.get("city", ""),
                "state": query_info.get("state", ""),
                "more_results_available": query_info.get("more_results_available", False)
            })

        # Process web results
        if "web" in raw_results and "results" in raw_results["web"]:
            for result in raw_results["web"]["results"]:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("description", ""),
                    "metadata": {
                        "date_published": result.get("page_age", None),
                        "author": None,
                        "source": result.get("profile", {}).get("name", None),
                        "language": result.get("language", None),
                        "relevance_score": None,
                        "snippet": result.get("description", None),
                        "family_friendly": result.get("family_friendly", None),
                        "type": result.get("type", None),
                        "subtype": result.get("subtype", None),
                        "thumbnail": result.get("thumbnail", {}).get("src", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Update total results count
        if "mixed" in raw_results:
            output_dict["total_results_found"] = len(raw_results["mixed"].get("main", []))

        # Set family friendly status
        if "mixed" in raw_results:
            output_dict["family_friendly"] = raw_results.get("family_friendly", True)

    except _WEBSEARCH_PARSE_EXCEPTIONS:
        _set_processing_error(output_dict, "Error processing Brave results")


######################### DuckDuckGo Search #########################
#
# https://github.com/deedy5/duckduckgo_search
# Copied request format/structure from https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
def search_web_duckduckgo(
        keywords: str,
        region: str = "wt-wt",
        timelimit: str | None = None,
        max_results: int | None = None,
) -> list[dict[str, str]]:
    assert keywords, "keywords is mandatory"

    payload = {
        "q": keywords,
        "s": "0",
        "o": "json",
        "api": "d.js",
        "vqd": "",
        "kl": region,
        "bing_market": region,
    }

    def _normalize_url(url: str) -> str:
        """Unquote URL and replace spaces with '+'."""
        return unquote(url).replace(" ", "+") if url else ""

    def _normalize(raw_html: str) -> str:
        """Strip HTML tags from the raw_html string."""
        REGEX_STRIP_TAGS = re.compile("<.*?>")
        return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""

    if timelimit:
        payload["df"] = timelimit

    cache = set()
    results: list[dict[str, str]] = []

    for _ in range(5):
        response = fetch(method="POST", url="https://html.duckduckgo.com/html", data=payload)
        resp_content = response.content
        if b"No  results." in resp_content:
            return results

        tree = document_fromstring(resp_content)
        elements = tree.xpath("//div[h2]")
        if not isinstance(elements, list):
            return results

        for e in elements:
            if isinstance(e, _Element):
                hrefxpath = e.xpath("./a/@href")
                href = str(hrefxpath[0]) if hrefxpath and isinstance(hrefxpath, list) else None
                if (
                        href
                        and href not in cache
                        and not href.startswith(
                    ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                )
                ):
                    cache.add(href)
                    titlexpath = e.xpath("./h2/a/text()")
                    title = str(titlexpath[0]) if titlexpath and isinstance(titlexpath, list) else ""
                    bodyxpath = e.xpath("./a//text()")
                    body = "".join(str(x) for x in bodyxpath) if bodyxpath and isinstance(bodyxpath, list) else ""
                    results.append(
                        {
                            "title": _normalize(title),
                            "href": _normalize_url(href),
                            "body": _normalize(body),
                        }
                    )
                    if max_results and len(results) >= max_results:
                        return results

        npx = tree.xpath('.//div[@class="nav-link"]')
        if not npx or not max_results:
            return results
        next_page = npx[-1] if isinstance(npx, list) else None
        if isinstance(next_page, _Element):
            names = next_page.xpath('.//input[@type="hidden"]/@name')
            values = next_page.xpath('.//input[@type="hidden"]/@value')
            if isinstance(names, list) and isinstance(values, list):
                payload = {str(n): str(v) for n, v in zip(names, values)}

    return results





def parse_duckduckgo_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse DuckDuckGo search results and update the output dictionary

    Args:
        raw_results (Dict): Raw DuckDuckGo response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # DuckDuckGo results are in a list of dictionaries
        results = raw_results.get("results", [])

        for result in results:
            # Extract information directly from the dictionary
            title = result.get("title", "")
            url = result.get("href", "")
            snippet = result.get("body", "")

            # Log warnings for missing data
            if not title:
                logging.warning("Missing title in result")
            if not url:
                logging.warning("Missing URL in result")
            if not snippet:
                logging.warning("Missing snippet in result")

            # Add the processed result to the output dictionary
            processed_result = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": None,  # DuckDuckGo doesn't typically provide this
                    "author": None,  # DuckDuckGo doesn't typically provide this
                    "source": extract_domain(url) if url else None,
                    "language": None,  # DuckDuckGo doesn't typically provide this
                    "relevance_score": None,  # DuckDuckGo doesn't typically provide this
                    "snippet": snippet
                }
            }

            output_dict["results"].append(processed_result)

        # Update total results count
        output_dict["total_results_found"] = len(output_dict["results"])

    except _WEBSEARCH_PARSE_EXCEPTIONS:
        _set_processing_error(output_dict, "Error processing DuckDuckGo results")


def extract_domain(url: str) -> str:
    """
    Extract domain name from URL

    Args:
        url (str): Full URL

    Returns:
        str: Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        return domain.replace('www.', '')
    except (AttributeError, TypeError, ValueError) as e:
        logging.warning(f"Failed to extract domain from URL {url}: {str(e)}")
        return url




######################### Google Search #########################
#
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def search_web_google(
        search_query: str,
        google_search_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        result_count: Optional[int] = None,
        c2coff: Optional[str] = None,
        results_origin_country: Optional[str] = None,
        date_range: Optional[str] = None,
        exactTerms: Optional[str] = None,
        excludeTerms: Optional[str] = None,
        filter: Optional[str] = None,
        geolocation: Optional[str] = None,
        ui_language: Optional[str] = None,
        search_result_language: Optional[str] = None,
        safesearch: Optional[str] = None,
        site_blacklist: Optional[str] = None,
        siteSearch: Optional[str] = None,
        siteSearchFilter: Optional[str] = None,
        sort_results_by: Optional[str] = None
) -> dict[str, Any]:
    """
    Perform a Google web search with the given parameters.

    :param search_query: The search query string
    :param google_search_api_key: Google Search API key
    :param google_search_engine_id: Google Search Engine ID
    :param result_count: Number of results to return
    :param c2coff: Enable/disable traditional Chinese search
    :param results_origin_country: Limit results to a specific country
    :param date_range: Limit results to a specific date range
    :param exactTerms: Exact terms that must appear in results
    :param excludeTerms: Terms that must not appear in results
    :param filter: Control duplicate content filter
    :param geolocation: Geolocation of the user
    :param ui_language: Language of the user interface
    :param search_result_language: Language of search results
    :param safesearch: Safe search setting
    :param site_blacklist: Single Site to exclude from search
    :param siteSearch: Google CSE siteSearch parameter
    :param siteSearchFilter: Google CSE siteSearchFilter parameter (e=exclude, i=include)
    :param sort_results_by: Sorting criteria for results
    :return: JSON response from Google Search API
    """
    try:
        # Load Search API URL from config file
        search_url = loaded_config_data['search_engines']['google_search_api_url']
        logging.info(f"Using search URL: {search_url}")

        # Initialize params dictionary
        params: dict[str, Any] = {"q": search_query}

        # Handle c2coff
        if c2coff is None:
            c2coff = loaded_config_data['search_engines']['google_simp_trad_chinese']
        if c2coff is not None:
            params["c2coff"] = c2coff

        # Handle results_origin_country
        if results_origin_country is None:
            limit_country_search = loaded_config_data['search_engines']['limit_google_search_to_country']
            if limit_country_search:
                results_origin_country = loaded_config_data['search_engines']['google_search_country']
        if results_origin_country:
            params["cr"] = results_origin_country

        # Handle google_search_engine_id
        if google_search_engine_id is None:
            google_search_engine_id = loaded_config_data['search_engines']['google_search_engine_id']
        if not google_search_engine_id:
            raise ValueError("Please set a valid Google Search Engine ID in the config file")
        params["cx"] = google_search_engine_id

        # Handle google_search_api_key
        if google_search_api_key is None:
            google_search_api_key = loaded_config_data['search_engines']['google_search_api_key']
        if not google_search_api_key:
            raise ValueError("Please provide a valid Google Search API subscription key")
        params["key"] = google_search_api_key

        # Handle other parameters
        if result_count:
            params["num"] = result_count
        if date_range:
            params["dateRestrict"] = date_range
        if exactTerms:
            params["exactTerms"] = exactTerms
        if excludeTerms:
            params["excludeTerms"] = excludeTerms
        if filter:
            params["filter"] = filter
        if geolocation:
            params["gl"] = geolocation
        if ui_language:
            params["hl"] = ui_language
        if search_result_language:
            params["lr"] = search_result_language
        if safesearch is None:
            safesearch = loaded_config_data['search_engines']['google_safe_search']
        if safesearch:
            params["safe"] = safesearch
        if siteSearch:
            params["siteSearch"] = siteSearch
        if siteSearchFilter:
            params["siteSearchFilter"] = siteSearchFilter
        if sort_results_by:
            params["sort"] = sort_results_by

        logging.info(f"Prepared parameters for Google Search: {params}")

        # Make the API call
        response = fetch(method="GET", url=search_url, params=params)
        google_search_results = response.json()

        logging.info(
            f"Successfully retrieved search results. Items found: {len(google_search_results.get('items', []))}")

        return google_search_results

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        raise

    except Exception as e:
        logging.error(f"Error during API request: {str(e)}")
        raise



def parse_google_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Google Custom Search API results and update the output dictionary.

    Args:
        raw_results (Dict): Raw Google API response.
        output_dict (Dict): Dictionary to store processed results.
    """
    # Lower verbosity: raw payload only at debug level
    logging.debug(f"Raw results received: {json.dumps(raw_results, indent=2)}")
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract search information
        if "searchInformation" in raw_results:
            search_info = raw_results["searchInformation"]
            output_dict["total_results_found"] = int(search_info.get("totalResults", "0"))
            output_dict["search_time"] = float(search_info.get("searchTime", 0.0))

        # Extract spelling suggestions
        if "spelling" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spelling"].get("correctedQuery")

        # Extract search parameters from queries
        if "queries" in raw_results and "request" in raw_results["queries"]:
            request = raw_results["queries"]["request"][0]
            output_dict.update({
                "search_query": request.get("searchTerms", ""),
                "search_lang": request.get("language", ""),
                "result_count": request.get("count", 0),
                "safesearch": request.get("safe", None),
                "exactTerms": request.get("exactTerms", None),
                "excludeTerms": request.get("excludeTerms", None),
                "filter": request.get("filter", None),
                "geolocation": request.get("gl", None),
                "search_result_language": request.get("hl", None),
                "sort_results_by": request.get("sort", None)
            })

        # Process search results
        if "items" in raw_results:
            for item in raw_results["items"]:
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    # IMPORTANT: 'snippet' is used as 'content'
                    "content": item.get("snippet", ""),
                    "metadata": {
                        "date_published": item.get("pagemap", {})
                        .get("metatags", [{}])[0]
                        .get("article:published_time"),
                        "author": item.get("pagemap", {})
                        .get("metatags", [{}])[0]
                        .get("article:author"),
                        "source": item.get("displayLink", None),
                        "language": item.get("language", None),
                        "relevance_score": None,  # Google doesn't provide this directly
                        "snippet": item.get("snippet", None),
                        "file_format": item.get("fileFormat", None),
                        "mime_type": item.get("mime", None),
                        "cache_url": item.get("cacheId", None)
                    }
                }

                # Extract additional metadata if available
                if "pagemap" in item:
                    pagemap = item["pagemap"]
                    if "metatags" in pagemap and pagemap["metatags"]:
                        metatags = pagemap["metatags"][0]
                        processed_result["metadata"].update({
                            "description": metatags.get("og:description",
                                                        metatags.get("description")),
                            "keywords": metatags.get("keywords"),
                            "site_name": metatags.get("og:site_name")
                        })

                output_dict["results"].append(processed_result)

        # Add pagination information
        output_dict["pagination"] = {
            "has_next": "nextPage" in raw_results.get("queries", {}),
            "has_previous": "previousPage" in raw_results.get("queries", {}),
            "current_page": raw_results.get("queries", {})
            .get("request", [{}])[0]
            .get("startIndex", 1)
        }

    except _WEBSEARCH_PARSE_EXCEPTIONS:
        _set_processing_error(output_dict, "Error processing Google results")





######################### Kagi Search #########################
#
# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(query: str, limit: int = 10) -> dict:
    search_url = "https://kagi.com/api/v0/search"

    # load key from config file
    kagi_api_key = loaded_config_data['search_engines']['kagi_search_api_key']
    if not kagi_api_key:
        raise ValueError("Please provide a valid Kagi Search API subscription key")

    """
    Queries the Kagi Search API with the given query and limit.
    """
    if kagi_api_key is None:
        raise ValueError("API key is required.")

    headers = {"Authorization": f"Bot {kagi_api_key}"}
    endpoint = f"{search_url}/search"
    params = {"q": query, "limit": limit}

    response = fetch(method="GET", url=endpoint, headers=headers, params=params)
    logging.debug(response.json())
    return response.json()





def parse_kagi_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Kagi search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Kagi API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract metadata
        if "meta" in raw_results:
            meta = raw_results["meta"]
            output_dict["search_time"] = meta.get("ms", 0) / 1000.0  # Convert to seconds
            output_dict["api_balance"] = meta.get("api_balance")
            output_dict["search_id"] = meta.get("id")
            output_dict["node"] = meta.get("node")

        # Process search results
        if "data" in raw_results:
            for item in raw_results["data"]:
                # Skip related searches (type 1)
                if item.get("t") == 1:
                    output_dict["related_searches"] = item.get("list", [])
                    continue

                # Process regular search results (type 0)
                if item.get("t") == 0:
                    processed_result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("snippet", ""),
                        "metadata": {
                            "date_published": item.get("published"),
                            "author": None,  # Kagi doesn't typically provide this
                            "source": None,  # Could be extracted from URL if needed
                            "language": None,  # Kagi doesn't typically provide this
                            "relevance_score": None,
                            "snippet": item.get("snippet"),
                            "thumbnail": item.get("thumbnail", {}).get("url") if "thumbnail" in item else None
                        }
                    }
                    output_dict["results"].append(processed_result)

            # Update total results count
            output_dict["total_results_found"] = len([
                item for item in raw_results["data"]
                if item.get("t") == 0
            ])

    except _WEBSEARCH_PARSE_EXCEPTIONS:
        _set_processing_error(output_dict, "Error processing Kagi results")




######################### SearX Search #########################
#
# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
def search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general',
                     searx_url=None):
    """
    Perform a search using a Searx instance.

    Args:
        search_query (str): The search query.
        language (str): Language for the search results.
        time_range (str): Time range for the search results.
        safesearch (int): Safe search level (0=off, 1=moderate, 2=strict).
        pageno (int): Page number of the results.
        categories (str): Categories to search in (e.g., 'general', 'news').
        searx_url (str): Custom Searx instance URL (optional).

    Returns:
        str: JSON string containing the search results or an error message.
    """
    # Use the provided Searx URL or fall back to the configured one
    if not searx_url:
        searx_url = loaded_config_data['search_engines']['searx_search_api_url']
    if not searx_url:
        return json.dumps({
                              "error": "SearX Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."})

    # Validate and construct URL
    try:
        parsed_url = urlparse(searx_url)
        params = {
            'q': search_query,
            'language': language,
            'time_range': time_range,
            'safesearch': safesearch,
            'pageno': pageno,
            'categories': categories
        }
        search_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(params)}"
        logging.info(f"Search URL: {search_url}")
    except _WEBSEARCH_PARSE_EXCEPTIONS as e:
        return json.dumps({"error": "Invalid URL configuration."})

    # Perform the search request
    try:
        headers = _websearch_browser_headers(accept_lang="en-US,en;q=0.5")

        # Add a random delay to mimic human behavior
        delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
        time.sleep(delay)

        response = fetch(method="GET", url=search_url, headers=headers)

        # Check if the response is JSON
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            search_data = response.json()
        else:
            # If not JSON, assume it's HTML and parse it
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            search_data = parse_html_search_results_generic(soup)

        # Process results
        data = []
        for result in search_data:
            data.append({
                'title': result.get('title'),
                'link': result.get('url'),
                'snippet': result.get('content'),
                'publishedDate': result.get('publishedDate')
            })

        if not data:
            return json.dumps({"error": "No information was found online for the search query."})

        return json.dumps(data)

    except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
        logging.error("Error searching for content.")
        return json.dumps({"error": "There was an error searching for content."})





def parse_searx_results(searx_search_results, web_search_results_dict):
    pass




######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_serper():
    pass




def parse_serper_results(serper_search_results, web_search_results_dict):
    pass


######################### Tavily Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_tavily(search_query, result_count=10, site_whitelist=None, site_blacklist=None):
    # Check if API URL is configured
    tavily_api_url = "https://api.tavily.com/search"

    tavily_api_key = loaded_config_data['search_engines']['tavily_search_api_key']

    # Prepare the request payload
    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "max_results": result_count
    }

    # Add optional parameters if provided
    if site_whitelist:
        payload["include_domains"] = site_whitelist
    if site_blacklist:
        payload["exclude_domains"] = site_blacklist

    # Perform the search request
    try:
        headers = {'Content-Type': 'application/json'}
        ua_headers = _websearch_browser_headers(accept_lang="en-US,en;q=0.5")
        if "User-Agent" in ua_headers:
            headers["User-Agent"] = ua_headers["User-Agent"]

        response = fetch(method="POST", url=tavily_api_url, headers=headers, data=json.dumps(payload))
        return response.json()
    except _WEBSEARCH_RUNTIME_EXCEPTIONS as e:
        return "There was an error searching for content."



def parse_tavily_results(tavily_search_results, web_search_results_dict):
    pass




######################### Yandex Search #########################
#
# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex():
    pass




def parse_yandex_results(yandex_search_results, web_search_results_dict):
    pass

#
# End of Web_Search.py
#######################################################################################################################
def brave_http_get(url: str, *, headers: dict[str, str], params: dict[str, Any]):
    """Wrapper seam for Brave HTTP GET used by tests to monkeypatch easily.

    Production path routes through centralized http client with retries and egress checks.
    Tests can patch this symbol to inject a fake response without relying on requests.get.
    """
    policy = RetryPolicy()
    return fetch(method="GET", url=url, headers=headers, params=params, retry=policy)
