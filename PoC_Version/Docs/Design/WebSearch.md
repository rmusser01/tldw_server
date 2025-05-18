# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.
- **High-Level Workflow**
    1. User inputs a search query.
    2. User selects a search engine (Option for default search engine + default query options in config file).
    3. User may select `Advanced Search` for additional search parameters (Language, Date Range, etc).
    4. The user presses 'Search'.
    5. Search is performed -> Results obtained, 
    6. Each individual item is first analyzed based on snippet, if relevant, entire page is fetched and analyzed, this is then stored in the results dictionary, and the process is repeated until all results are analyzed/limit is hit.
    7. Once all results are collected, they are then operated on, being used to create whatever final product is desired by the user.
    8. The final product is then passed back to the UI for display to the user.


### Current Status
- Bing, Brave, DDG, Google work for simple searches. Advanced search options are not fully working yet.
    - Brave: https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    - Bing: https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference
    - DuckDuckGo: https://duckduckgo.com/
    - Google: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
- Baidu, SearX, Serper, Tavily, Yandex are not implemented yet.
- Kagi & SearX are implemented but not working (Kagi because API access and SearX because I'm not sure why)
- Parsing works for Bing, Brave, DDG, Google.
- Currently no use of Structured outputs (to-do...)
- Full Pipeline works.
    1. User enters query + Search options
    2. Query is processed, sub-queries are generated(if specified)
    3. Each query is sent to the search engine API
    4. Search results are returned
    5. Each search result is analyzed, if relevant, the full page is fetched and stored in the results dict
    6. Results are then aggregated and presented to the user
- **To Do:**
    2. Implement the saving options (Web Search DB - allow for saving search results in DB, File, Clipboard, Notes DB - allow for saving search report + citations in a note)
    2. User can also select which results are relevant, and which are not, and remove irrelevant results from the `web_search_results_dict`
    3. Implement the remaining search engines (Baidu, SearX, Serper, Tavily, Yandex)
    4. Implement the advanced search options
    5. Implement the structured outputs
    6. Implement the various output options (Style/Format) / Allow the user to customize the aggregation prompt
    7. Provide the user with follow-up questions + potential other questions, if these are selected, the content is added to the 'ongoing' document


----------------
### Setting the Stage

- **Text Search Workflow**
    1. User inputs a search query.
    2. User selects a search engine + Query options (Option for default search engine + default query options in config file).
    3. The user presses 'Search'.
        - Gradio UI Function makes a call to `process_question` with the search query + search parameters as a dictionary
    4. `process_question()` checks the search params dict to see if sub-query creation is enabled, if so, it creates a list of sub-queries based on the search query with a call to `analyze_question()`.
    5. `analyze_question()` takes the search query and generates a list of sub-queries based on the search query, attempts this 3 times, making a call to the LLM API, and then returns the list of sub-queries if successful.
    6. once back in `process_question()`, all queries are combined into a single query list `#L113`.
    7. `process_question()` then iterates through the query list, making a call to `perform_websearch()` with each query in the list, and the matching search parameters.
    8. `perform_websearch()` makes a call to the selected search engine's API with the query and search parameters, and returns the results.
       - This function is a `sink' for all search engine API calls, and is responsible for handling calling the appropriate search engine API call
    9. `process_web_search_results()` then takes the results from the search engine, and processes them, converting them into a dictionary of results in the `web_search_results_dict` dictionary.
       - FIXME - this is where I lose track of what's happening, need to re-read the code
       - This function returns a filled `web_search_results_dict` dictionary
    10. `process_question()` then takes the `web_search_results_dict` and processes it, checking to make sure it is valid and contains results.
        - FIXME - Verify this is correct
    11. FIXME - Make it optional to display the results to the user, and allow them to select which results are relevant before continuing processing
    12. `process_question()` then iterates through each search result, checking if it is relevant, and if so, adds it to the `relevant_results_dict`
          - FIXME - The results should be added back to the `web_search_results_dict` if they are relevant.
    13. `process_question()` then calls into `aggregate_results()` function with the `web_search_results_dict`
    14. `aggregate_results()` then takes the `web_search_results_dict` and processes it, combining all the results into a single document
        - FIXME - This is not implemented yet and also want various options available for this.
    15. `process_question()` then returns the `web_search_results_dict` to the calling function.
    15. The calling function then takes the `web_search_results_dict` and processes it, extracting the final results/aggregated report and presenting it to the user
    16. The user then has the option to save the results to the DB, or ask follow-up questions, etc.
    17. The user can also select which results are relevant, and which are not, and remove irrelevant results from the `web_search_results_dict`


- **Function Execution Steps:**
    - `def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None, safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None)`
        - `search_engine` - The search engine to use for the search
        - `search_query` - The query to search for
        - `content_country` - The country of the content to search for
        - `search_lang` - The language to use for the search
        - `output_lang` - The language to use for the output
        - `result_count` - The number of results to return
        - `date_range` - The date range to search within
        - `safesearch` - Whether to enable safe search
        - `site_blacklist` - A list of sites to exclude from the search results
        - `exactTerms` - Terms that must be in the search results
        - `excludeTerms` - Terms that must not be in the search results
        - `filter` - A filter to apply to the search results
        - `geolocation` - The geolocation to use for the search
        - `search_result_language` - The language to use for the search results
        - `sort_results_by` - How to sort the search results
    - **Returns:** A list of search results as a dictionary. - FIXME: Define the structure of the dictionary
              - Each result should contain the title, URL, content, and metadata of the search result.



----------------
### Web Search API
- TBD
```
def perform_websearch(search_engine
                      search_query,
                      content_country,
                      search_lang,
                      output_lang,
                      result_count, - Number of results to return(not applicable to all engines, but necessary for consistency)
                      date_range=None,
                      safesearch=None,
                      site_blacklist=None, 
                      exactTerms=None, 
                      excludeTerms=None, 
                      filter=None, 
                      geolocation=None, 
                      search_result_language=None
                      sort_results_by=None):
```



### Implementation

config.txt options explained
```
# Search Defaults
search_provider_default = google
search_language_query = en - Language Queries will be performed in
search_language_results = en - Language Results will be returned in
search_language_analysis = en - Language Analysis will be performed in
search_default_max_queries = 10 - Maximum number of queries to perform
search_enable_subquery = True - Enable subqueries
search_enable_subquery_count_max = 5 - Maximum number of subqueries to generate
search_result_rerank = True - Enable result reranking
search_result_max = 15 - Maximum number of results to return
search_result_max_per_query = 10 - Maximum number of results to return per query
search_result_blacklist = []
search_result_display_type = list - Display type for search results, does nothing right now.
search_result_display_metadata = False - Display metadata for search results, does nothing right now.
search_result_save_to_db = True - Save search results to the database (not implemented yet)
# How you want the results to be written, think 'style' or voice 
search_result_analysis_tone = neutral - Tone of the analysis (not implemented yet)
relevance_analysis_llm = openai - LLM to use for relevance analysis
final_answer_llm = openai - LLM to use for final answer generation
#### Search Engines #####
# Bing
search_engine_country_code_bing = en - Country code for Bing, Where Search 'takes place from'
#
# Brave
search_engine_country_code_brave = US - Country code for Brave, Where Search 'takes place from'
#
# Google
# Restricts search results to documents originating in a particular country.
limit_google_search_to_country = False
google_search_country_code = US - Country code for Google, Where Search 'takes place from'
google_filter_setting = 1 - Filter setting for Google, 0 = No filtering, 1 = Moderate filtering, 2 = Strict filtering
google_user_geolocation = US - Geolocation for user performing the search
google_limit_search_results_to_language = False - Limit search results to a specific language
google_default_search_results = 10 - Default number of search results to return
google_safe_search = "active" - Safe search setting for Google, active, moderate, or off
google_enable_site_search = False - Enable site search
google_site_search_include = - Sites to include in the search
google_site_search_exclude = - Sites to exclude from the search
# https://developers.google.com/custom-search/docs/structured_search#sort-by-attribute
google_sort_results_by = - Sort results by attribute (I honestly couldn't find much about this one)
```

Results dictionary:
```
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
    ],
    "total_results_found": search_results.get("total_results_found", 0),
    "search_time": search_results.get("search_time", 0.0),
    "error": search_results.get("error", None),
    "processing_error": None
}
```

----------------
### Search Engines

#### Baidu Search
- [Baidu Search API](https://www.baidu.com/)
- Baidu doens't have an official english API, so we'll have to scrape the results or use Serper


#### Bing Search
- [Bing Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
- Getting Started with Bing Search API
    - Sign up for a Bing Search API key via Azure (1000 Free Searches a month) - https://www.microsoft.com/en-us/bing/apis/pricing
    - Use the Bing Search API to perform searches - Add the generated subscription key to your config.txt file.
          - If for some reason you're doing modifications to the code(Fuck MS), be aware: https://github.com/Azure-Samples/cognitive-services-REST-api-samples/issues/139
    - Perform searches using Bing!


#### Brave Search
Two APIs, 1 for 'AI' the other for 'regular' search
    - [Brave Search API](https://brave.com/search/api/)


#### DuckDuckGo Search
Uses query to direct DDG search, then scrape the results.
Structure/approach taken from https://github.com/deedy5/duckduckgo_search


#### Google Search
- [Google Search API](https://developers.google.com/custom-search/v1/overview)
- Have to create a custom search engine first, get the ID and then the API key
- Setup:
  - Setup a `Programmable Search Engine`
  - Get the `API Key`
  - 100 Search queries per day for free
- Documentation for making requests: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list


#### Kagi Search
- [Kagi Search API](https://help.kagi.com/kagi/api/search.html)
- Really straightforward, just a simple search API


#### SearX Search
- [Searx Search Documentation](https://searx.github.io/searx/)
- `SearXNG is a free internet metasearch engine which aggregates results from more than 70 search services. Users are neither tracked nor profiled.`
- Can host your own instance or use someone else's.
https://github.com/searxng/searxng-docker


#### Serper Search


#### Tavily Search


#### Yandex Search
- https://yandex.cloud/en/docs/search-api/quickstart/
- [Yandex Search API](https://yandex.com/dev/search/)

#### Arxiv Search


#### PubMedCentral Search
- https://www.ncbi.nlm.nih.gov/home/develop/api/



### Prompts used:

sub_question_generation_prompt =
```
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
#
search_result_relevance_eval_prompt = Given the following search results for the user's question: "{original_question}" and the generated sub-questions: {sub_questions}, evaluate the relevance of the search result to the user's question.
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

```

analyze_search_results_prompt =
```
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
```
