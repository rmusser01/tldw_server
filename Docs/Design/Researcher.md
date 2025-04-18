# Automated Researcher

## Introduction
- This page is to document efforts towards creating a 'research' agent workflow for use within the project. The goal is to create a system that can automatically generate research reports, summaries, and other research-related tasks.

### Researcher Goals
1. f
2. f
3. f
4. f
5. f
6. f
7. f
8. 


### Ideas
Gated, checkpoints with 'retry, skip, continue' options
s
Follow gptresearchers method at first, planner LLM -> query LLM -> analyzer LLM -> summarizer LLM


### Researcher Workflow


### Researcher Components
1. **Query/Search Engine**
    - f
    - f
2. **Planner**
    - f
    - f
3. **Analyzer**
    - f
    - f
4. **(Optional: Summarizer)**
    - f
    - f
5. **Report Generator**
    - f
    - f
6. **Knowledge Base Management**
    - f
    - f


### Researcher Config Definitions
- `default_search_engine`: The default search engine to use for queries

- Researcher config section
```
[researcher]
# Researcher settings
default_search_engine = google
# Options are: google, bing, yandex, baidu, searx, kagi, serper, tavily
default_search_type = web
# Options are: web, local, both
default_search_language = en
# Options are: FIXME
default_search_report_language = en
# Options are: FIXME
default_search_sort = relevance
# Options are: relevance, date
default_search_safe_search = moderate
# Options are: off, moderate, strict
default_search_planner = openai-o1-full
# Options are: FIXME
default_search_planner_max_tokens = 8192
default_search_analyzer = openai-o1-full
# Options are: FIXME
default_search_analyzer_max_tokens = 8192
default_search_summarization = openai-o1-full
# Options are: FIXME
default_search_summarization_max_tokens = 8192
search_max_results = 100
search_report_format = markdown
# Options are: markdown, html, pdf
search_max_iterations = 5
search_max_subtopics = 4
search_custom_user_agent = "CUSTOM_USER_AGENT_HERE"
search_blacklist_URLs = "URL1,URL2,URL3"
```


Perplexica
   https://github.com/ItzCrazyKns/Perplexica/blob/master/src/search/metaSearchAgent.ts
   https://github.com/ItzCrazyKns/Perplexica/blob/master/src/chains/suggestionGeneratorAgent.ts
   https://github.com/ItzCrazyKns/Perplexica/blob/master/src/chains/imageSearchAgent.ts
   https://github.com/ItzCrazyKns/Perplexica/blob/master/src/search/metaSearchAgent.ts

Falle
   https://github.com/rashadphz/farfalle/blob/main/src/backend/agent_search.py


### Link Dump:
Articles
   https://docs.gptr.dev/blog/gptr-hybrid
   https://docs.gptr.dev/docs/gpt-researcher/context/local-docs
   https://docs.gptr.dev/docs/gpt-researcher/context/tailored-research#
   https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package

Standford STORM
   https://arxiv.org/abs/2402.14207#
   https://storm.genie.stanford.edu/

Google Learn About
    https://learning.google.com/experiments/learn-about

Google Pinpoint
    https://journaliststudio.google.com/pinpoint/about/

Gemini Deepresearcher
    https://blog.google/products/gemini/google-gemini-deep-research/

STORM
    https://github.com/stanford-oval/storm/
    https://github.com/stanford-oval/storm/blob/main/examples/storm_examples/run_storm_wiki_claude.py
    https://storm-project.stanford.edu/research/storm/


https://github.com/assafelovic/gpt-researcher
https://arxiv.org/abs/2411.15114
https://github.com/masterFoad/NanoSage
https://github.com/pkargupta/tree-of-debate
https://github.com/binary-husky/gpt_academic/blob/master/docs/README.English.md
https://arxiv.org/abs/2409.13741
https://github.com/sentient-agi/OpenDeepSearch
https://github.com/qx-labs/agents-deep-research
https://researcher.iqidis.ai
https://github.com/zjunlp/OmniThink
https://github.com/neuml/annotateai
https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph
https://pub.towardsai.net/learn-anything-with-ai-and-the-feynman-technique-00a33f6a02bc
https://github.com/dzhng/deep-research
https://github.com/eRuaro/open-gemini-deep-research
https://github.com/atineiatte/deep-research-at-home
https://help.openalex.org/hc/en-us/articles/24396686889751-About-us
https://www.ginkgonotes.com/
https://github.com/jina-ai/node-DeepResearch[v1-f.py](../../../Sky/v1-f.py)
https://github.com/LearningCircuit/local-deep-research
https://www.reddit.com/r/Anki/comments/17u01ge/spaced_repetition_algorithm_a_threeday_journey/
https://github.com/open-spaced-repetition/fsrs4anki/wiki/Spaced-Repetition-Algorithm:-A-Three%E2%80%90Day-Journey-from-Novice-to-Expert#day-3-the-latest-progress
https://www.scrapingdog.com/blog/scrape-google-news/
https://arxiv.org/html/2501.03916
    https://unimodal4reasoning.github.io/Dolphin-project-page/
https://github.com/mistralai/cookbook/blob/main/third_party/LlamaIndex/llamaindex_arxiv_agentic_rag.ipynb
https://github.com/ai-christianson/RA.Aid
https://github.com/cbuccella/perplexity_research_prompt/blob/main/general_research_prompt.md
https://github.com/0xeb/TheBigPromptLibrary/blob/main/SystemPrompts/Perplexity.ai/20241024-Perplexity-Desktop-App.md
https://github.com/rashadphz/farfalle
https://huggingface.co/blog/open-deep-re[v3-f-single.py](../../../Sky/v3-f-single.py)search
https://danielkliewer.com/2025/02/05/open-deep-research
https://github.com/cbuccella/perplexity_research_prompt/blob/main/general_research_prompt.md
https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
https://github.com/neoneye/PlanExe
https://github.com/CJones-Optics/ChiCurate
https://www.emergentmind.com/
https://github.com/goodreasonai/nichey
https://milvus.io/blog/introduce-deepsearcher-a-local-open-source-deep-research.md
https://github.com/neuml/paperai
https://github.com/neuml/paperetl
https://github.com/ai-christianson/RA.Aid
https://github.com/Future-House/paper-qa
https://github.com/bytedance/pasa
https://huggingface.co/spaces/Felladrin/awesome-ai-web-search
https://allenai.org/blog/ai2-scholarqa
https://openreview.net/
https://github.com/HarshJ23/Deeper-Seeker
https://learning.google.com/experiments/learn-about/signup
https://departmentofproduct.substack.com/p/how-to-use-perplexity-to-automate
https://departmentofproduct.substack.com/p/deep-the-ux-of-search
https://www.researchrabbit.ai/
https://github.com/faraz18001/Sales-Llama
https://github.com/memgraph/memgraph
https://github.com/rashadphz/farfalle/tree/main/src/backend
https://arxiv.org/abs/2501.04306
https://github.com/SakanaAI/AI-Scientist
https://agentlaboratory.github.io/
https://gangiswag.github.io/infogent/
https://arxiv.org/abs/2501.03916
https://arxiv.org/abs/2502.19413
https://arxiv.org/abs/2502.18864
https://github.com/dendrite-systems/dendrite-python-sdk
https://github.com/rashadphz/farfalle/blob/main/src/backend/agent_search.py
https://github.com/rashadphz/farfalle/blob/main/src/backend/prompts.py

https://learning.google.com/experiments/learn-about
https://github.com/SamuelSchmidgall/AgentLaboratory
https://github.com/exa-labs/company-researcher
https://zjunlp.github.io/project/OmniThink/

Structured report Output
    https://www.youtube.com/watch?v=aqtX-sGbevw
    https://github.com/run-llama/llama_cloud_services/blob/main/examples/report/basic_report.ipynb

AI Web Researcher Ollama
    https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama/blob/main/Self_Improving_Search.py
    https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama





### Researcher Prompts
https://github.com/cbuccella/perplexity_research_prompt
https://github.com/rashadphz/farfalle/blob/main/src/backend/prompts.py

https://github.com/ItzCrazyKns/Perplexica/tree/master/src/prompts
https://github.com/SakanaAI/AI-Scientist

```
SEARCH_QUERY_PROMPT = """\
Generate a concise list of search queries to gather information for executing the given step.

You will be provided with:
1. A specific step to execute
2. The user's original query
3. Context from previous steps (if available)

Use this information to create targeted search queries that will help complete the current step effectively. Aim for the minimum number of queries necessary while ensuring they cover all aspects of the step.

IMPORTANT: Always incorporate relevant information from previous steps into your queries. This ensures continuity and builds upon already gathered information.

Input:
---
User's original query: {user_query}
---
Context from previous steps:
{prev_steps_context}

Your task:
1. Analyze the current step and its requirements
2. Consider the user's original query and any relevant previous context
3. Consider the user's original query
4. Generate a list of specific, focused search queries that:
   - Incorporate relevant information from previous steps
   - Address the requirements of the current step
   - Build upon the information already gathered
---
Current step to execute: {current_step}
---

Your search queries based:
"""
```



I use NotebookLM daily and find it incredibly helpful. However, I've noticed a potential improvement for the audio creation feature. Currently, when generating audio from a source, it primarily focuses on the provided text. I propose enhancing this by adding a "deep research" component that runs in the background during audio generation.

Imagine this: you provide NotebookLM with a news article about a new AI tool. When you click "create audio," instead of just reading the article, NotebookLM would:
    Analyze the Source: Understand the core topic, key terms, and context of the provided source.
    Conduct Background Research: Leverage Google's powerful search and knowledge graph to gather additional information related to the topic. This could include:
    Official documentation or websites for tools.
    Related news articles, blog posts, and research papers.
    Expert opinions and analyses.
    Relevant historical context.
    Integrate Findings: Seamlessly weave the researched information into the audio output, creating a more comprehensive and insightful experience. This could be done by:
    Adding explanatory segments or summaries.
    Providing context and background information.
    Highlighting different perspectives or opinions.
    Offering definitions of key terms.
Example:
If the source is an article about "LaMDA," NotebookLM could research:
    Google AI's official information on LaMDA.
    Recent advancements in large language models.
    Ethical considerations surrounding AI language models.
    Comparisons to other similar models.
This would result in an audio output that not only summarizes the original article but also provides valuable context and deeper understanding.
Benefits:
    More Comprehensive Content: Audio outputs become more informative and valuable for users.
    Saves User Time: Users don't have to conduct their own research to get the full picture.
    Enhanced Learning Experience: Provides a richer and more engaging way to consume information.
    Positions NotebookLM as an Expert Resource: By providing in-depth information, NotebookLM becomes a go-to tool for learning about various topics.
Suggested Implementation Details:
    Leverage Google's Existing Tools: Utilize Google Search, Knowledge Graph, and potentially the "deep research" module already present within Google's ecosystem. This would ensure seamless integration and efficient use of existing resources.
    Clear User Controls: Provide options for users to customize the depth of research (e.g., "basic," "moderate," "in-depth"). This gives users control over the process and prevents information overload.
    Citation and Source Linking: Include links to the researched sources within the NotebookLM document associated with the audio, providing transparency and allowing users to verify information.
    Integration with Google Lens: If an image is part of the source, use Google Lens to extract text and context, further enhancing the research capabilities.
Additional Features:
    Option to Exclude Research: Allow users to disable background research if they only want a direct reading of the source.
    Customizable Research Focus: Allow users to specify keywords or areas of focus for the background research, allowing for more targeted results.
    Multilingual Research: Expand research capabilities to multiple languages, making the feature more globally accessible.
By implementing this feature, NotebookLM can become an even more powerful tool for learning and understanding complex topics, providing users with comprehensive and insightful audio experiences.