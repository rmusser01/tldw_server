# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import asyncio
import json
from html import unescape
import pytest
import random
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urlencode, unquote
#
# 3rd-Party Imports
import requests
from lxml.etree import _Element
from lxml.html import document_fromstring
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3 import Retry
#
# Local Imports
from tldw_Server_API.app.core.Utils.Utils import loaded_config_data, logging
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article
from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
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
#
# FIXME - Add Logging

def initialize_web_search_results_dict(search_params: Dict) -> Dict:
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


def generate_and_search(question: str, search_params: Dict) -> Dict:
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
        sleep_time = random.uniform(1, 1.5)  # Add a random delay to avoid rate limiting
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
            print(f"Error or invalid data returned for query '{q}': {raw_results}")
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


async def analyze_and_aggregate(web_search_results_dict: Dict, sub_query_dict: Dict, search_params: Dict) -> Dict:
    logging.info("Starting analyze_and_aggregate")

    # 4. Score/filter results
    logging.info("Scoring and filtering search results")
    sub_questions = sub_query_dict.get("sub_questions", [])
    relevant_results = await search_result_relevance(
        web_search_results_dict["results"],
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('relevance_analysis_llm')
    )
    # FIXME
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

    # 7. Return the final data
    logging.info("Returning final websearch results")
    return {
        "final_answer": final_answer,
        "relevant_results": relevant_results,
        "web_search_results_dict": web_search_results_dict
    }


@pytest.mark.asyncio
async def test_perplexity_pipeline():
    # Phase 1: Generate sub-queries and perform web searches
    search_params = {
        "engine": "google",
        "content_country": "countryUS",
        "search_lang": "en",
        "output_lang": "en",
        "result_count": 10,
        "date_range": None,
        "safesearch": "active",
        "site_blacklist": ["spam-site.com"],
        "exactTerms": None,
        "excludeTerms": None,
        "filter": None,
        "geolocation": None,
        "search_result_language": None,
        "sort_results_by": None,
        "subquery_generation": True,
        "subquery_generation_llm": "openai",
        "relevance_analysis_llm": "openai",
        "final_answer_llm": "openai"
    }
    phase1_results = generate_and_search("What is the capital of France?", search_params)
    # Review the results here if needed
    # Phase 2: Analyze relevance and aggregate final answer
    phase2_results = await analyze_and_aggregate(phase1_results["web_search_results_dict"], phase1_results["sub_query_dict"], search_params)
    print(phase2_results["final_answer"])


######################### Question Analysis #########################
#
#
def analyze_question(question: str, api_endpoint) -> Dict:
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

    sub_questions: List[str] = []
    for attempt in range(3):
        try:
            logging.info(f"Generating sub-questions (attempt {attempt + 1})")

            response = chat_api_call(api_endpoint, None, input_data, sub_question_generation_prompt, temp=0.7, system_message=None, streaming=False, minp=None, maxp=None, model=None)
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

        except Exception as e:
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
# FIXME - Ensure edge cases are handled properly / Structured outputs?
async def search_result_relevance(
    search_results: List[Dict],
    original_question: str,
    sub_questions: List[str],
    api_endpoint: str
) -> Dict[str, Dict]:
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

        try:
            # Add delay to avoid rate limiting
            sleep_time = random.uniform(0.2, 0.6)
            await asyncio.sleep(sleep_time)

            # Evaluate relevance
            relevancy_result = chat_api_call(
                api_endpoint=api_endpoint,
                api_key=None,
                input_data=input_data,
                prompt=eval_prompt,
                temp=0.7,
                system_message=None,
                streaming=False,
                minp=None,
                maxp=None,
                model=None,
                topk=None,
                topp=None,
            )

            # FIXME
            logging.debug(f"[DEBUG] Relevancy LLM response for index {idx}:\n{relevancy_result}\n---")

            if relevancy_result:
                # Extract the selected answer and reasoning via regex
                logging.debug(f"LLM Relevancy Response for item:", relevancy_result)
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
                        summary = analyze(
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
        except Exception as e:
            logging.error(f"Error during relevance evaluation/summarization for result idx={idx}: {e}")

    return relevant_results


def review_and_select_results(web_search_results_dict: Dict) -> Dict:
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
def aggregate_results(
    relevant_results: Dict[str, Dict],
    question: str,
    sub_questions: List[str],
    api_endpoint: str
) -> Dict:
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
        return {
            "Report": "No relevant results found. Unable to provide an answer.",
            "evidence": [],
            "confidence": 0.0
        }

    # FIXME - Add summarization loop
    logging.info("Summarizing relevant results")
    # ADD Code here to summarize the relevant results


    # FIXME - Validate and test thoroughly, also structured generation
    # Concatenate relevant contents for final analysis
    concatenated_texts = "\n\n".join(
        f"ID: {rid}\nContent: {res['content']}\nReasoning: {res['reasoning']}"
        for rid, res in relevant_results.items()
    )

    current_date = time.strftime("%Y-%m-%d")

    # Aggregation Prompt #1
    analyze_search_results_prompt_1 = f"""
        Generate a comprehensive, well-structured, and informative answer for a given question, 
        using ONLY the information found in the provided web Search Results (URL, Page Title, Summary).
        Use an unbiased, journalistic tone, adapting the level of formality to match the user’s question.
        
        • Cite your statements using [number] notation, placing citations at the end of the relevant sentence.
        • Only cite the most relevant results. If multiple sources support the same point, cite all relevant sources [e.g., 1, 2, 3].
        • If sources conflict, present both perspectives clearly and cite the respective sources.
        • If different sources refer to different entities with the same name, provide separate answers.
        • Do not add any external or fabricated information.
        • Do not include URLs or a reference section; cite inline with [number] format only.
        • Do not repeat the question or include unnecessary redundancy.
        • Use markdown formatting (e.g., **bold**, bullet points, ## headings) to organize the information.
        • If the provided results are insufficient to answer the question, explicitly state what information is missing or unclear.
        
        Structure your answer like this:
        1. **Short introduction**: Briefly summarize the topic (1–2 sentences).
        2. **Bulleted points**: Present key details, each with appropriate citations.
        3. **Conclusion**: Summarize the findings or restate the core answer (with citations if needed).
        
        Example:
        1. **Short introduction**: This topic explores the impact of climate change on agriculture.
        2. **Bulleted points**:
           - Rising temperatures have reduced crop yields in some regions [1].
           - Changes in rainfall patterns are affecting irrigation practices [2, 3].
        3. **Conclusion**: Climate change poses significant challenges to global agriculture [1, 2, 3].
        
        <context>
        {concatenated_texts}
        </context>
        ---------------------
        
        Make sure to match the language of the user's question.
        
        Question: {question}
        Answer (in the language of the user's question):
        """

    # Aggregation Prompt #2
    analyze_search_results_prompt_2 = f"""INITIAL_QUERY: Here are some sources {concatenated_texts}. Read these carefully, as you will be asked a Query about them.
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
        
        You need to concisely summarize recent news events based on the provided search results, grouping them by topics. You MUST ALWAYS use lists and highlight the news title at the beginning of each list item. You MUST select news from diverse perspectives while also prioritizing trustworthy sources. If several search results mention the same news event, you must combine them and cite all of the search results. Prioritize more recent events, ensuring to compare timestamps. You MUST NEVER start your answer with a heading of any kind.
        
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
        - Always use \( and\) for inline formulas and\[ and\] for blocks, for example\(x^4 = x - 3 \)
        - To cite a formula add citations to the end, for example\[ \sin(x) \] [1][2] or \(x^2-2\) [4].
        - Never use $ or $$ to render LaTeX, even if it is present in the user query.
        - Never use unicode to render math expressions, ALWAYS use LaTeX.
        - Never use the \label instruction for LaTeX.
        
        ## URL Lookup
        
        When the user's query includes a URL, you must rely solely on information from the corresponding search result. DO NOT cite other search results, ALWAYS cite the first result, e.g. you need to end with [1]. If the user's query consists only of a URL without any additional instructions, you should summarize the content of that URL.
        
        ## Shopping
        
        If the user query is about shopping for a product, you MUST follow these rules:
        - Organize the products into distinct sectors. For example, you could group shoes by style (boots, sneakers, etc.)
        - Cite at most 9 search results using the format provided in General Instructions to avoid overwhelming the user with too many options.
        
        The current date is: {current_date}

        The user's query is: {question}
        """

    input_data = "Follow the above instructions."

    try:
        logging.info("Generating the report")
        returned_response = chat_api_call(
            api_endpoint=api_endpoint,
            api_key=None,
            input_data=input_data,
            prompt=analyze_search_results_prompt_2,
            temp=0.7,
            system_message=None,
            streaming=False,
            minp=None,
            maxp=None,
            model=None,
            topk=None,
            topp=None,
        )
        logging.debug(f"Returned response from LLM: {returned_response}")
        if returned_response:
            # You could do further parsing or confidence estimation here
            return {
                "Report": returned_response,
                "evidence": list(relevant_results.values()),
                "confidence": 0.9  # Hardcoded or computed as needed
            }
    except Exception as e:
        logging.error(f"Error aggregating results: {e}")

    logging.error("Could not create the report due to an error.")
    return {
        "summary": "Could not create the report due to an error.",
        "evidence": list(relevant_results.values()),
        "confidence": 0.0
    }

#
# End of Orchestration functions
#######################################################################################################################


#######################################################################################################################
#
# Search Engine Functions

# FIXME
def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None,
                      safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None):
    try:
        if search_engine.lower() == "baidu":
            web_search_results = search_web_baidu(search_query, None, None)

        elif search_engine.lower() == "bing":
            # Prepare the arguments for search_web_bing
            bing_args = {
                "search_query": search_query,
                "bing_lang": search_lang,
                "bing_country": content_country,
                "result_count": result_count,
                "bing_api_key": loaded_config_data['search_engines'].get('bing_api_key'),  # Fetch Bing API key from config
                "date_range": date_range,
            }

            # Call the search_web_bing function with the prepared arguments
            web_search_results = search_web_bing(**bing_args)

        elif search_engine.lower() == "brave":
            web_search_results = search_web_brave(search_query, content_country, search_lang, output_lang, result_count, safesearch,
                                    site_blacklist, date_range)

        elif search_engine.lower() == "duckduckgo":
            # Prepare the arguments for search_web_duckduckgo
            ddg_args = {
                "keywords": search_query,
                "region": f"{content_country.lower()}-{search_lang.lower()}",  # Format: "us-en"
                "timelimit": date_range[0] if date_range else None,  # Use first character of date_range (e.g., "y" -> "y")
                "max_results": result_count,
            }

            # Call the search_web_duckduckgo function with the prepared arguments
            ddg_results = search_web_duckduckgo(**ddg_args)

            # Wrap the results in a dictionary to match the expected format
            web_search_results = {"results": ddg_results}

        elif search_engine.lower() == "google":
            # Convert site_blacklist list to a comma-separated string
            if site_blacklist and isinstance(site_blacklist, list):
                site_blacklist = ",".join(site_blacklist)

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
            if site_blacklist and len(site_blacklist) == 1:
                google_args["siteSearch"] = site_blacklist[0]
                google_args["siteSearchFilter"] = "e"
            else:
                # Do not use siteSearch for multiple domains
                # Either skip it entirely or see Option 2 below
                google_args.pop("siteSearch", None)
                google_args.pop("siteSearchFilter", None)

            # Add optional parameters only if they are provided
            if date_range:
                google_args["date_range"] = date_range
            if exactTerms:
                google_args["exactTerms"] = exactTerms
            if excludeTerms:
                google_args["excludeTerms"] = excludeTerms
            if filter:
                google_args["filter"] = filter
            if site_blacklist:
                google_args["site_blacklist"] = site_blacklist
            if sort_results_by:
                google_args["sort_results_by"] = sort_results_by

            # Call the search_web_google function with the prepared arguments
            web_search_results = search_web_google(**google_args)  # raw JSON
            web_search_results_dict = process_web_search_results(web_search_results, "google")
            return web_search_results_dict

        elif search_engine.lower() == "kagi":
            web_search_results = search_web_kagi(search_query, content_country)

        elif search_engine.lower() == "serper":
            web_search_results = search_web_serper()

        elif search_engine.lower() == "tavily":
            web_search_results = search_web_tavily(search_query, result_count, site_blacklist)

        elif search_engine.lower() == "searx":
            web_search_results = search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general')

        elif search_engine.lower() == "yandex":
            web_search_results = search_web_yandex()

        else:
            return f"Error: Invalid Search Engine Name {search_engine}"

        # Process the raw search results
        web_search_results_dict = process_web_search_results(web_search_results, search_engine)
        # FIXME
        #logging.debug("After process_web_search_results:")
        #logging.debug(json.dumps(web_search_results_dict, indent=2))
        return web_search_results_dict

    except Exception as e:
        return {"processing_error": f"Error performing web search: {str(e)}"}


def test_perform_websearch_google():
    # Google Searches
    try:
        test_1 = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 1: {test_1}")
        # FIXME - Fails. Need to fix arg formatting
        test_2 = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10, date_range="y", safesearch="active", site_blacklist=["spam-site.com"])
        print(f"Test 2: {test_2}")
        test_3 = results = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 3: {test_3}")
    except Exception as e:
        print(f"Error performing google searches: {str(e)}")
    pass


def test_perform_websearch_bing():
    # Bing Searches
    try:
        test_4 = perform_websearch("bing", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 4: {test_4}")
        test_5 = perform_websearch("bing", "What is the capital of France?", "US", "en", "en", 10, date_range="y")
        print(f"Test 5: {test_5}")
    except Exception as e:
        print(f"Error performing bing searches: {str(e)}")


def test_perform_websearch_brave():
    # Brave Searches
    try:
        test_7 = perform_websearch("brave", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 7: {test_7}")
    except Exception as e:
        print(f"Error performing brave searches: {str(e)}")


def test_perform_websearch_ddg():
    # DuckDuckGo Searches
    try:
        test_6 = perform_websearch("duckduckgo", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 6: {test_6}")
        test_7 = perform_websearch("duckduckgo", "What is the capital of France?", "US", "en", "en", 10, date_range="y")
        print(f"Test 7: {test_7}")
    except Exception as e:
        print(f"Error performing duckduckgo searches: {str(e)}")


# FIXME
def test_perform_websearch_kagi():
    # Kagi Searches
    try:
        test_8 = perform_websearch("kagi", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 8: {test_8}")
    except Exception as e:
        print(f"Error performing kagi searches: {str(e)}")

# FIXME
def test_perform_websearch_serper():
    # Serper Searches
    try:
        test_9 = perform_websearch("serper", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 9: {test_9}")
    except Exception as e:
        print(f"Error performing serper searches: {str(e)}")

# FIXME
def test_perform_websearch_tavily():
    # Tavily Searches
    try:
        test_10 = perform_websearch("tavily", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 10: {test_10}")
    except Exception as e:
        print(f"Error performing tavily searches: {str(e)}")


# FIXME
def test_perform_websearch_searx():
    # Searx Searches
    try:
        test_11 = perform_websearch("searx", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 11: {test_11}")
    except Exception as e:
        print(f"Error performing searx searches: {str(e)}")


# FIXME
def test_perform_websearch_yandex():
    #Yandex Searches
    try:
        test_12 = perform_websearch("yandex", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 12: {test_12}")
    except Exception as e:
        print(f"Error performing yandex searches: {str(e)}")
    pass

#
######################### Search Result Parsing ##################################################################
#

def process_web_search_results(search_results: Dict, search_engine: str) -> Dict:
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
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [],
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
    }
    try:
        # Parse results based on the search engine
        if search_engine.lower() == "baidu":
            pass  # Placeholder for Baidu-specific parsing
        elif search_engine.lower() == "bing":
            parsed_results = parse_bing_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "brave":
            parsed_results = parse_brave_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "duckduckgo":
            parsed_results = parse_duckduckgo_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "google":
            parsed_results = parse_google_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "kagi":
            parsed_results = parse_kagi_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "serper":
            parsed_results = parse_serper_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "tavily":
            parsed_results = parse_tavily_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "searx":
            parsed_results = parse_searx_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "yandex":
            parsed_results = parse_yandex_results(search_results, web_search_results_dict)
        else:
            raise ValueError(f"Error: Invalid Search Engine Name {search_engine}")

    except Exception as e:
        web_search_results_dict["processing_error"] = f"Error processing search results: {str(e)}"
        logging.error(f"Error in process_web_search_results: {str(e)}")

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


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result

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
        answer_count = loaded_config_data['search_engines']['search_result_max']
    else:
        answer_count = result_count

    # date_range = "day", "week", "month", or `YYYY-MM-DD..YYYY-MM-DD`
    if not date_range:
         date_range = None

    # Language settings
    if not bing_lang:
        # do config check for default search language
        setlang = bing_lang

    # Returns content for this Country market code
    if not bing_country:
        # do config check for default search country
        bing_country = loaded_config_data['search_engines']['bing_country_code']
    else:
        setcountry = bing_country
    # Construct a request
    mkt = 'en-US'
    params = {'q': search_query, 'mkt': mkt}
#    params = {"q": search_query, "mkt": bing_country, "textDecorations": True, "textFormat": "HTML", "count": answer_count,
#             "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}

    # Call the API
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()

        logging.debug("Headers:  ")
        logging.debug(response.headers)

        logging.debug("JSON Response: ")
        logging.debug(response.json())
        bing_search_results = response.json()
        return bing_search_results
    except Exception as ex:
        raise ex


def test_search_web_bing():
    search_query = "How can I get started learning machine learning?"
    bing_lang = "en"
    bing_country = "US"
    result_count = 10
    bing_api_key = None
    date_range = None
    result = search_web_bing(search_query, bing_lang, bing_country, result_count, bing_api_key, date_range)
    # Unparsed results
    print("Bing Search Results:")
    print(result)
    # Parsed results
    output_dict = {"results": []}
    parse_bing_results(result, output_dict)
    print("Parsed Bing Results:")
    print(json.dumps(output_dict, indent=2))


def parse_bing_results(raw_results: Dict, output_dict: Dict) -> None:
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

    except Exception as e:
        logging.error(f"Error processing Bing results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Bing results: {str(e)}"


######################### Brave Search #########################
#
# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate",
                     brave_api_key=None, result_filter=None, search_type="ai", date_range=None):
    search_url = "https://api.search.brave.com/res/v1/web/search"
    if not brave_api_key and search_type == "web":
        # load key from config file
        brave_api_key = loaded_config_data['search_engines']['brave_search_api_key']
        if not brave_api_key:
            raise ValueError("Please provide a valid Brave Search API subscription key")
    if not country:
        brave_country = loaded_config_data['search_engines']['search_engine_country_code_brave']
    else:
        country = "US"
    if not search_lang:
        search_lang = "en"
    if not ui_lang:
        ui_lang = "en"
    if not result_count:
        result_count = 10
    # if not date_range:
    #     date_range = "month"
    if not result_filter:
        result_filter = "webpages"
    if search_type == "ai":
        brave_api_key = loaded_config_data['search_engines']['brave_search_ai_api_key']
    else:
        raise ValueError("Invalid search type. Please choose 'ai' or 'web'.")


    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": result_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results


def test_search_brave():
    search_term = "How can I bake a cherry cake"
    country = "US"
    search_lang = "en"
    ui_lang = "en"
    result_count = 10
    safesearch = "moderate"
    date_range = None
    result_filter = None
    result = search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch, date_range,
                             result_filter)
    print("Brave Search Results:")
    print(result)

    output_dict = {"results": []}
    parse_brave_results(result, output_dict)
    print("Parsed Brave Results:")
    print(json.dumps(output_dict, indent=2))


def parse_brave_results(raw_results: Dict, output_dict: Dict) -> None:
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

    except Exception as e:
        logging.error(f"Error processing Brave results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Brave results: {str(e)}"

def test_parse_brave_results():
    pass


######################### DuckDuckGo Search #########################
#
# https://github.com/deedy5/duckduckgo_search
# Copied request format/structure from https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

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
        response = requests.post("https://html.duckduckgo.com/html", data=payload)
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


def test_search_duckduckgo():
    try:
        results = search_web_duckduckgo(
            keywords="How can I bake a cherry cake?",
            region="us-en",
            timelimit="w",
            max_results=10
        )
        print(f"Number of results: {len(results)}")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Snippet: {result['body']}")
            print("---")

        # Parse the results
        output_dict = {"results": []}
        parse_duckduckgo_results({"results": results}, output_dict)
        print("Parsed DuckDuckGo Results:")
        print(json.dumps(output_dict, indent=2))

    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")


def parse_duckduckgo_results(raw_results: Dict, output_dict: Dict) -> None:
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

    except Exception as e:
        logging.error(f"Error processing DuckDuckGo results: {str(e)}")
        output_dict["processing_error"] = f"Error processing DuckDuckGo results: {str(e)}"


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
    except:
        return url


def test_parse_duckduckgo_results():
    pass



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
    sort_results_by: Optional[str] = None
) -> Dict[str, Any]:
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
    :param sort_results_by: Sorting criteria for results
    :return: JSON response from Google Search API
    """
    try:
        # Load Search API URL from config file
        search_url = loaded_config_data['search_engines']['google_search_api_url']
        logging.info(f"Using search URL: {search_url}")

        # Initialize params dictionary
        params: Dict[str, Any] = {"q": search_query}

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
        if sort_results_by:
            params["sort"] = sort_results_by

        logging.info(f"Prepared parameters for Google Search: {params}")

        # Make the API call
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        google_search_results = response.json()

        logging.info(f"Successfully retrieved search results. Items found: {len(google_search_results.get('items', []))}")

        return google_search_results

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        raise

    except RequestException as re:
        logging.error(f"Error during API request: {str(re)}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred: {str(e)}")
        raise


def test_search_google():
    search_query = "How can I bake a cherry cake?"
    google_search_api_key = loaded_config_data['search_engines']['google_search_api_key']
    google_search_engine_id = loaded_config_data['search_engines']['google_search_engine_id']
    result_count = 10
    c2coff = "1"
    results_origin_country = "countryUS"
    date_range = None
    exactTerms = None
    excludeTerms = None
    filter = None
    geolocation = "us"
    ui_language = "en"
    search_result_language = "lang_en"
    safesearch = "off"
    site_blacklist = None
    sort_results_by = None
    result = search_web_google(search_query,
                               google_search_api_key,
                               google_search_engine_id,
                               result_count,
                               c2coff,
                               results_origin_country,
                               date_range,
                               exactTerms,
                               excludeTerms,
                               filter,
                               geolocation,
                               ui_language,
                               search_result_language,
                               safesearch,
                               site_blacklist,
                               sort_results_by
                               )
    print(result)
    return result


def parse_google_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Google Custom Search API results and update the output dictionary.

    Args:
        raw_results (Dict): Raw Google API response.
        output_dict (Dict): Dictionary to store processed results.
    """
    logging.info(f"Raw results received: {json.dumps(raw_results, indent=2)}")
    # For debugging only FIXME
    logging.debug("Raw web_search_results from Google:")
    logging.debug(json.dumps(raw_results, indent=2))
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

    except Exception as e:
        logging.error(f"Error processing Google results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Google results: {str(e)}"


def test_parse_google_results():
    parsed_results = {}
    raw_results = {}
    raw_results = test_search_google()
    parse_google_results(raw_results, parsed_results)
    print(f"Parsed search results: {parsed_results}")
    pass



######################### Kagi Search #########################
#
# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(query: str, limit: int = 10) -> Dict:
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

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    logging.debug(response.json())
    return response.json()


def test_search_kagi():
    search_term = "How can I bake a cherry cake"
    result_count = 10
    result = search_web_kagi(search_term, result_count)
    print(result)


def parse_kagi_results(raw_results: Dict, output_dict: Dict) -> None:
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

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Kagi results: {str(e)}"


def test_parse_kagi_results():
    pass



######################### SearX Search #########################
#
# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
def searx_create_session() -> requests.Session:
    """
    Create a requests session with retry logic.
    """
    session = requests.Session()
    retries = Retry(
        total=3,  # Maximum number of retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["GET"]  # Only retry on GET requests
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general', searx_url=None):
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
        return json.dumps({"error": "SearX Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."})

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
    except Exception as e:
        return json.dumps({"error": f"Invalid URL configuration: {str(e)}"})

    # Perform the search request
    try:
        # Mimic browser headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Add a random delay to mimic human behavior
        delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
        time.sleep(delay)

        session = searx_create_session()
        response = session.get(search_url, headers=headers)
        response.raise_for_status()

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

    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching for content: {str(e)}")
        return json.dumps({"error": f"There was an error searching for content. {str(e)}"})

def test_search_searx():
    # Use a different Searx instance to avoid rate limiting
    searx_url = "https://searx.be"  # Example of a different Searx instance
    result = search_web_searx("What goes into making a cherry cake?", searx_url=searx_url)
    print(result)

def parse_searx_results(searx_search_results, web_search_results_dict):
    pass

def test_parse_searx_results():
    pass




######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_serper():
    pass


def test_search_serper():
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
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0'
        }

        response = requests.post(tavily_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"There was an error searching for content. {str(e)}"


def test_search_tavily():
    result = search_web_tavily("How can I bake a cherry cake?")
    print(result)


def parse_tavily_results(tavily_search_results, web_search_results_dict):
    pass


def test_parse_tavily_results():
    pass




######################### Yandex Search #########################
#
# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex():
    pass


def test_search_yandex():
    pass

def parse_yandex_results(yandex_search_results, web_search_results_dict):
    pass


#
# End of WebSearch_APIs.py
#######################################################################################################################
