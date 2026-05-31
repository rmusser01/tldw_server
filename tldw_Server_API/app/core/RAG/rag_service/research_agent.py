"""
Iterative Agentic Research Loop for RAG.

This module implements an LLM-driven iterative search loop where the model:
1. Reasons about what to search for
2. Executes a search action (web, academic, local DB, URL scrape, discussion)
3. Analyzes results
4. Decides whether to search again with refined queries or stop
5. Repeats until satisfied or iteration limit reached

Inspired by Perplexica's researcher pattern, adapted for tldw_server2's
existing retrieval and web search infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    StructuredOutputParseError,
    parse_structured_output,
)

from .query_classifier import QueryClassification

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ActionOutput:
    """Result from executing a single research action."""

    action_name: str
    success: bool = True
    results: list[dict[str, Any]] = field(default_factory=list)
    result_count: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchStep:
    """A single step in the research loop."""

    iteration: int
    reasoning: str  # LLM's reasoning for this step
    action_name: str
    action_params: dict[str, Any]
    output: ActionOutput | None = None
    duration_sec: float = 0.0


@dataclass
class ResearchOutput:
    """Final output of the research loop."""

    query: str
    standalone_query: str
    steps: list[ResearchStep] = field(default_factory=list)
    all_results: list[dict[str, Any]] = field(default_factory=list)
    total_iterations: int = 0
    total_results: int = 0
    total_duration_sec: float = 0.0
    final_reasoning: str = ""
    completed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchAction:
    """A pluggable research action that the agent can invoke."""

    name: str
    description: str
    schema: dict[str, Any]  # JSON schema for parameters
    enabled: Callable[[QueryClassification], bool]  # Whether available
    execute: Callable[..., Any]  # Async callable: (params) -> ActionOutput


# ---------------------------------------------------------------------------
# Action Registry
# ---------------------------------------------------------------------------

class ActionRegistry:
    """Pluggable registry of research actions."""

    def __init__(self) -> None:
        self._actions: dict[str, ResearchAction] = {}

    def register(self, action: ResearchAction) -> None:
        """Register a research action."""
        self._actions[action.name] = action

    def get(self, name: str) -> ResearchAction | None:
        """Get action by name."""
        return self._actions.get(name)

    def get_available(self, classification: QueryClassification) -> list[ResearchAction]:
        """Get actions available for the given classification."""
        available = []
        for action in self._actions.values():
            try:
                if action.enabled(classification):
                    available.append(action)
            except Exception as action_error:
                logger.debug(
                    "Research action capability check failed; action skipped",
                    error_type=type(action_error).__name__,
                )
        return available

    async def execute(self, name: str, params: dict[str, Any]) -> ActionOutput:
        """Execute a named action with given parameters."""
        action = self._actions.get(name)
        if action is None:
            return ActionOutput(
                action_name=name,
                success=False,
                error=f"Unknown action: {name}",
            )
        try:
            result = action.execute(params)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, ActionOutput):
                return result
            # Wrap raw results
            if isinstance(result, list):
                return ActionOutput(
                    action_name=name,
                    success=True,
                    results=result,
                    result_count=len(result),
                )
            if isinstance(result, dict):
                return ActionOutput(
                    action_name=name,
                    success=True,
                    results=[result],
                    result_count=1,
                    metadata=result,
                )
            return ActionOutput(action_name=name, success=True, results=[], result_count=0)
        except Exception as exc:
            logger.warning(f"Research action '{name}' failed: {exc!r}")
            return ActionOutput(
                action_name=name,
                success=False,
                error=str(exc),
            )

    def get_actions_description(self, classification: QueryClassification) -> str:
        """Get a formatted description of available actions for the LLM prompt."""
        available = self.get_available(classification)
        if not available:
            return "No actions available."

        parts = []
        for action in available:
            schema_str = json.dumps(action.schema, indent=2) if action.schema else "{}"
            parts.append(
                f"- **{action.name}**: {action.description}\n"
                f"  Parameters: {schema_str}"
            )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Built-in action factories
# ---------------------------------------------------------------------------

def _normalize_web_results(
    search_payload: Any,
    search_engine: str,
    process_results: Callable[[dict[str, Any], str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return normalized web results from raw or already-processed payloads."""
    if not isinstance(search_payload, dict):
        return []

    existing_results = search_payload.get("results")
    if isinstance(existing_results, list):
        normalized_results = [item for item in existing_results if isinstance(item, dict)]
        if normalized_results and any(
            ("url" in item) or ("content" in item) or ("snippet" in item)
            for item in normalized_results
        ):
            return normalized_results

    processed = process_results(search_payload, search_engine)
    if not isinstance(processed, dict):
        return []
    processed_results = processed.get("results")
    if not isinstance(processed_results, list):
        return []
    return [item for item in processed_results if isinstance(item, dict)]


def _create_local_db_search_action() -> ResearchAction:
    """Create action that wraps the existing MultiDatabaseRetriever."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        try:
            from .database_retrievers import MultiDatabaseRetriever, RetrievalConfig

            query = params.get("query", "")
            top_k = params.get("top_k", 10)
            sources = params.get("sources", ["media_db"])

            config = RetrievalConfig(
                search_mode="hybrid",
                top_k=top_k,
            )
            retriever = MultiDatabaseRetriever(config=config)

            # Build kwargs from available DB paths
            retrieve_kwargs: dict[str, Any] = {
                "query": query,
                "sources": sources,
            }
            for key in ("media_db_path", "notes_db_path", "character_db_path",
                         "kanban_db_path", "media_db", "chacha_db"):
                if key in params and params[key] is not None:
                    retrieve_kwargs[key] = params[key]

            results = await retriever.retrieve(**retrieve_kwargs)

            docs = []
            for doc in results:
                if hasattr(doc, "content"):
                    docs.append({
                        "id": getattr(doc, "id", ""),
                        "content": doc.content[:500],  # Truncate for agent context
                        "score": getattr(doc, "score", 0.0),
                        "source": str(getattr(doc, "source", "local_db")),
                        "metadata": getattr(doc, "metadata", {}),
                    })
                elif isinstance(doc, dict):
                    docs.append({
                        "id": doc.get("id", ""),
                        "content": str(doc.get("content", ""))[:500],
                        "score": doc.get("score", 0.0),
                        "source": "local_db",
                        "metadata": doc.get("metadata", {}),
                    })

            return ActionOutput(
                action_name="local_db_search",
                success=True,
                results=docs,
                result_count=len(docs),
            )
        except Exception as exc:
            return ActionOutput(
                action_name="local_db_search",
                success=False,
                error=str(exc),
            )

    return ResearchAction(
        name="local_db_search",
        description="Search the local database (media, notes, characters, chats) for relevant content",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (default: 10)"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Databases to search: media_db, notes, characters, chats, kanban",
                },
            },
            "required": ["query"],
        },
        enabled=lambda c: c.search_local_db,
        execute=_execute,
    )


def _create_web_search_action() -> ResearchAction:
    """Create action that wraps the existing web search infrastructure."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        try:
            from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
                perform_websearch,
                process_web_search_results,
            )

            query = params.get("query", "")
            engine = params.get("engine", "duckduckgo")
            result_count = params.get("result_count", 5)

            raw_results = await asyncio.to_thread(
                perform_websearch,
                search_engine=engine,
                search_query=query,
                content_country="US",
                search_lang="en",
                output_lang="en",
                result_count=result_count,
            )

            results_list = _normalize_web_results(
                search_payload=raw_results,
                search_engine=engine,
                process_results=process_web_search_results,
            )

            docs = []
            for r in results_list[:result_count]:
                docs.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": str(r.get("content", r.get("snippet", "")))[:500],
                    "source": "web",
                })

            return ActionOutput(
                action_name="web_search",
                success=True,
                results=docs,
                result_count=len(docs),
            )
        except Exception as exc:
            return ActionOutput(
                action_name="web_search",
                success=False,
                error=str(exc),
            )

    return ResearchAction(
        name="web_search",
        description="Search the web for current information, facts, and general knowledge",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "engine": {"type": "string", "description": "Search engine (default: duckduckgo)"},
                "result_count": {"type": "integer", "description": "Number of results (default: 5)"},
            },
            "required": ["query"],
        },
        enabled=lambda c: c.search_web,
        execute=_execute,
    )


def _create_academic_search_action() -> ResearchAction:
    """Create action that wraps existing academic search endpoints."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        try:
            from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
                perform_websearch,
                process_web_search_results,
            )

            query = params.get("query", "")
            # Use web search with academic site filters as a universal fallback
            academic_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org"

            raw_results = await asyncio.to_thread(
                perform_websearch,
                search_engine="duckduckgo",
                search_query=academic_query,
                content_country="US",
                search_lang="en",
                output_lang="en",
                result_count=params.get("result_count", 5),
            )

            results_list = _normalize_web_results(
                search_payload=raw_results,
                search_engine="duckduckgo",
                process_results=process_web_search_results,
            )

            docs = []
            for r in results_list:
                docs.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": str(r.get("content", r.get("snippet", "")))[:500],
                    "source": "academic",
                })

            return ActionOutput(
                action_name="academic_search",
                success=True,
                results=docs,
                result_count=len(docs),
            )
        except Exception as exc:
            return ActionOutput(
                action_name="academic_search",
                success=False,
                error=str(exc),
            )

    return ResearchAction(
        name="academic_search",
        description="Search academic sources (arXiv, Semantic Scholar, Google Scholar) for research papers and studies",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Academic search query"},
                "result_count": {"type": "integer", "description": "Number of results (default: 5)"},
            },
            "required": ["query"],
        },
        enabled=lambda c: c.search_academic,
        execute=_execute,
    )


def _create_discussion_search_action(
    default_platforms: list[str] | None = None,
) -> ResearchAction:
    """Create action for searching discussion forums."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        try:
            from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import search_discussions

            query = params.get("query", "")
            platforms = (
                params.get("platforms")
                or default_platforms
                or ["reddit", "stackoverflow", "hackernews"]
            )
            max_results = params.get("max_results", 10)

            results = await search_discussions(
                query=query,
                platforms=platforms,
                max_results=max_results,
            )

            return ActionOutput(
                action_name="discussion_search",
                success=True,
                results=results,
                result_count=len(results),
            )
        except Exception as exc:
            return ActionOutput(
                action_name="discussion_search",
                success=False,
                error=str(exc),
            )

    return ResearchAction(
        name="discussion_search",
        description="Search forums and discussion platforms (Reddit, StackOverflow, HackerNews) for community knowledge and opinions",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "platforms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Platforms to search: reddit, stackoverflow, hackernews",
                },
                "max_results": {"type": "integer", "description": "Max results (default: 10)"},
            },
            "required": ["query"],
        },
        enabled=lambda c: c.search_discussions,
        execute=_execute,
    )


def _create_scrape_url_action() -> ResearchAction:
    """Create action that wraps Article_Extractor_Lib.scrape_article()."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        try:
            from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article

            url = params.get("url", "")
            if not url:
                return ActionOutput(
                    action_name="scrape_url",
                    success=False,
                    error="No URL provided",
                )

            result = await scrape_article(url)

            if result.get("extraction_successful"):
                content = result.get("content", "")
                # Truncate for agent context
                if len(content) > 3000:
                    content = content[:3000] + "... [truncated]"

                return ActionOutput(
                    action_name="scrape_url",
                    success=True,
                    results=[{
                        "url": result.get("url", url),
                        "title": result.get("title", ""),
                        "content": content,
                        "author": result.get("author", ""),
                        "date": result.get("date", ""),
                        "source": "scraped_url",
                    }],
                    result_count=1,
                )
            else:
                return ActionOutput(
                    action_name="scrape_url",
                    success=False,
                    error=result.get("error", "Scraping failed"),
                )
        except Exception as exc:
            return ActionOutput(
                action_name="scrape_url",
                success=False,
                error=str(exc),
            )

    return ResearchAction(
        name="scrape_url",
        description="Fetch and extract full content from a URL when search snippets are insufficient",
        schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to scrape"},
            },
            "required": ["url"],
        },
        enabled=lambda _c: True,  # Always available
        execute=_execute,
    )


def _create_reasoning_preamble_action(
    on_progress: Callable[[ResearchProgressEvent], Any] | None = None,
) -> ResearchAction:
    """Create the '__reasoning_preamble' action for explicit reasoning before tool calls.

    In balanced/quality modes the LLM is required to call this action first
    to record its reasoning plan before executing any search actions.
    """

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        reasoning_text = params.get("reasoning", "")
        plan = params.get("plan", "")
        combined = f"{reasoning_text}\n{plan}".strip() if plan else reasoning_text

        # Emit reasoning event via progress callback
        if on_progress is not None:
            event = ResearchProgressEvent(
                event_type="research_reasoning_preamble",
                data={"text": combined},
            )
            try:
                import asyncio as _aio
                result = on_progress(event)
                if _aio.iscoroutine(result):
                    await result
            except Exception as progress_error:
                logger.debug(
                    "Research agent progress callback failed",
                    error_type=type(progress_error).__name__,
                )

        return ActionOutput(
            action_name="__reasoning_preamble",
            success=True,
            results=[],
            result_count=0,
            metadata={"reasoning": combined},
        )

    return ResearchAction(
        name="__reasoning_preamble",
        description=(
            "Record your reasoning plan BEFORE executing any search actions. "
            "Required in balanced and quality modes as the first action. "
            "Describe what you plan to search for and why."
        ),
        schema={
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your detailed reasoning about the research approach",
                },
                "plan": {
                    "type": "string",
                    "description": "Step-by-step plan for the research (optional)",
                },
            },
            "required": ["reasoning"],
        },
        enabled=lambda _c: True,
        execute=_execute,
    )


def _create_image_search_action() -> ResearchAction:
    """Create the 'image_search' action for finding relevant images."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        query = params.get("query", "")
        if not query:
            return ActionOutput(action_name="image_search", success=False, error="No query provided")

        max_results = int(params.get("max_results", 10))
        search_engine = params.get("search_engine", "duckduckgo")
        llm_provider = params.get("llm_provider", "openai")
        llm_model = params.get("llm_model")

        try:
            from .media_search import search_images
            images = await search_images(
                query=query,
                llm_provider=llm_provider,
                llm_model=llm_model,
                max_results=max_results,
                search_engine=search_engine,
            )
            return ActionOutput(
                action_name="image_search",
                success=True,
                results=images,
                result_count=len(images),
                metadata={"type": "images"},
            )
        except Exception as exc:
            return ActionOutput(action_name="image_search", success=False, error=str(exc))

    return ResearchAction(
        name="image_search",
        description="Search for relevant images related to the query",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Image search query"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
        enabled=lambda _c: True,
        execute=_execute,
    )


def _create_video_search_action() -> ResearchAction:
    """Create the 'video_search' action for finding relevant videos."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        query = params.get("query", "")
        if not query:
            return ActionOutput(action_name="video_search", success=False, error="No query provided")

        max_results = int(params.get("max_results", 10))
        search_engine = params.get("search_engine", "duckduckgo")
        llm_provider = params.get("llm_provider", "openai")
        llm_model = params.get("llm_model")

        try:
            from .media_search import search_videos
            videos = await search_videos(
                query=query,
                llm_provider=llm_provider,
                llm_model=llm_model,
                max_results=max_results,
                search_engine=search_engine,
            )
            return ActionOutput(
                action_name="video_search",
                success=True,
                results=videos,
                result_count=len(videos),
                metadata={"type": "videos"},
            )
        except Exception as exc:
            return ActionOutput(action_name="video_search", success=False, error=str(exc))

    return ResearchAction(
        name="video_search",
        description="Search for relevant videos (primarily YouTube) related to the query",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Video search query"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
        enabled=lambda _c: True,
        execute=_execute,
    )


def _create_done_action() -> ResearchAction:
    """Create the 'done' action that signals research completion."""

    async def _execute(params: dict[str, Any]) -> ActionOutput:
        return ActionOutput(
            action_name="done",
            success=True,
            results=[],
            result_count=0,
            metadata={"reason": params.get("reason", "Research complete")},
        )

    return ResearchAction(
        name="done",
        description="Signal that research is complete and you have enough information to answer the query",
        schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why research is complete"},
            },
            "required": ["reason"],
        },
        enabled=lambda _c: True,  # Always available
        execute=_execute,
    )


# ---------------------------------------------------------------------------
# Default registry with all built-in actions
# ---------------------------------------------------------------------------

def create_default_registry(
    *,
    discussion_platforms: list[str] | None = None,
    enable_url_scraping: bool = True,
    enable_image_search: bool = False,
    enable_video_search: bool = False,
    on_progress: Callable[[ResearchProgressEvent], Any] | None = None,
) -> ActionRegistry:
    """Create an ActionRegistry with built-in actions registered."""
    return create_configured_registry(
        discussion_platforms=discussion_platforms,
        enable_url_scraping=enable_url_scraping,
        enable_image_search=enable_image_search,
        enable_video_search=enable_video_search,
        on_progress=on_progress,
    )


def create_configured_registry(
    *,
    discussion_platforms: list[str] | None = None,
    enable_url_scraping: bool = True,
    enable_image_search: bool = False,
    enable_video_search: bool = False,
    on_progress: Callable[[ResearchProgressEvent], Any] | None = None,
) -> ActionRegistry:
    """Create an ActionRegistry with built-in actions and runtime toggles.

    Args:
        discussion_platforms: Optional default platforms used by discussion_search
            when action params don't provide an explicit list.
        enable_url_scraping: Whether to register scrape_url action.
        enable_image_search: Whether to register image_search action.
        enable_video_search: Whether to register video_search action.
        on_progress: Optional progress callback for reasoning preamble events.
    """
    registry = ActionRegistry()
    registry.register(_create_local_db_search_action())
    registry.register(_create_web_search_action())
    registry.register(_create_academic_search_action())
    registry.register(_create_discussion_search_action(default_platforms=discussion_platforms))
    if enable_url_scraping:
        registry.register(_create_scrape_url_action())
    registry.register(_create_reasoning_preamble_action(on_progress=on_progress))
    if enable_image_search:
        registry.register(_create_image_search_action())
    if enable_video_search:
        registry.register(_create_video_search_action())
    registry.register(_create_done_action())
    return registry


# ---------------------------------------------------------------------------
# Research loop LLM prompt
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT_BASE = """\
You are a research agent. Your goal is to find comprehensive, accurate information \
to answer the user's query. You have access to various search tools and must decide \
what to search for at each step.

At each step, you must respond with a JSON object:
{
  "reasoning": "<your reasoning about what to search for and why>",
  "action": "<action_name>",
  "params": { <action parameters> }
}

Core Principles:
- Start with the most likely source of relevant information
- Refine your searches based on what you find (or don't find)
- Try different angles or queries if initial results are poor
- Use scrape_url to get full content when a snippet looks promising
- Call "done" when you have sufficient information to answer the query
- Be efficient: don't repeat searches that already returned good results
- When scraping URLs, choose the most relevant 1-3 URLs from search results"""


def _build_speed_prompt() -> str:
    """Build the system prompt for speed mode.

    Speed mode: minimal iterations, no reasoning preamble, direct search.
    """
    return _RESEARCH_SYSTEM_PROMPT_BASE + """

## Speed Mode Rules
- You have a VERY LIMITED budget: complete in 1-2 search iterations maximum
- Use web_search as your primary tool for every query
- Do NOT use __reasoning_preamble — go straight to searching
- Pick the single best search query and execute it immediately
- Only do a second search if the first returned zero useful results
- Call "done" as soon as you have ANY relevant results
- Prefer breadth over depth: one good search beats multiple narrow ones
- Do NOT scrape URLs unless search results are completely empty"""


def _build_balanced_prompt() -> str:
    """Build the system prompt for balanced mode.

    Balanced mode: moderate depth, reasoning preamble required, 2-4 searches.
    """
    return _RESEARCH_SYSTEM_PROMPT_BASE + """

## Balanced Mode Rules
- You have a moderate budget: use 2-4 information-gathering calls (max 6 total tool calls)
- ALWAYS start with __reasoning_preamble as your FIRST action to plan your approach
- After reasoning, execute your search plan methodically
- Use a mix of search types when appropriate (web, academic, local DB, discussions)
- Refine queries based on what you find — don't repeat the same search
- Scrape 1-2 promising URLs if snippets look highly relevant
- Cross-reference key facts across at least 2 sources when possible
- Call "done" when you have sufficient coverage of the topic

## Common Mistakes to Avoid
- Skipping the reasoning preamble
- Doing too many searches on the same angle
- Not refining queries based on initial results
- Scraping too many URLs (stick to 1-2 max)"""


def _build_quality_prompt() -> str:
    """Build the system prompt for quality mode.

    Quality mode: deep research, comprehensive coverage, research strategy template.
    """
    return _RESEARCH_SYSTEM_PROMPT_BASE + """

## Quality Mode Rules
- You have a generous budget: use 4-7 information-gathering calls (more if needed)
- ALWAYS start with __reasoning_preamble to lay out a comprehensive research strategy
- Be thorough: try multiple angles, cross-reference, and scrape promising URLs

## Research Strategy Template
Follow this systematic approach:
1. **Definition & Overview**: Search for what the topic IS (definitions, core concepts)
2. **Key Features & Components**: Search for main aspects, features, or components
3. **Comparisons & Alternatives**: Search for how it compares to alternatives
4. **Recent Developments**: Search for recent news, updates, or changes
5. **Expert Opinions & Reviews**: Search discussions/forums for real-world perspectives
6. **Use Cases & Applications**: Search for practical applications and examples
7. **Limitations & Criticisms**: Search for drawbacks, controversies, or limitations

Not all steps apply to every query — adapt the strategy to the topic.

## Tool Usage Guidelines
- Use web_search for general information and recent content
- Use academic_search for scholarly or technical topics
- Use discussion_search for community opinions and real-world experiences
- Use local_db_search when the user's own content may be relevant
- Use scrape_url for 2-4 highly relevant URLs that need full content
- Cross-reference important claims across multiple sources

## Common Mistakes to Avoid
- Searching the same query multiple times with minor variations
- Not using discussion_search for topics that benefit from community input
- Stopping too early before covering different angles
- Not scraping URLs that clearly contain comprehensive information
- Failing to cross-reference controversial or disputed claims"""


def _build_research_prompt(
    query: str,
    standalone_query: str,
    available_actions: str,
    steps_so_far: list[ResearchStep],
    mode: str,
) -> list[dict[str, str]]:
    """Build the LLM messages for a research step."""
    prompt_builders = {
        "speed": _build_speed_prompt,
        "balanced": _build_balanced_prompt,
        "quality": _build_quality_prompt,
    }
    system_prompt = prompt_builders.get(mode, _build_balanced_prompt)()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Research query: {standalone_query}\n"
                f"Original query: {query}\n\n"
                f"Mode: {mode}\n\n"
                f"Available actions:\n{available_actions}\n"
            ),
        },
    ]

    # Add previous steps as context
    for step in steps_so_far:
        # Agent's decision
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "reasoning": step.reasoning,
                "action": step.action_name,
                "params": step.action_params,
            }),
        })
        # Result feedback
        if step.output:
            result_summary = f"Action '{step.action_name}': "
            if step.output.success:
                result_summary += f"Found {step.output.result_count} results."
                if step.output.results:
                    # Include brief summaries of top results
                    for i, r in enumerate(step.output.results[:3]):
                        title = r.get("title", r.get("id", ""))
                        content_preview = str(r.get("content", ""))[:150]
                        result_summary += f"\n  [{i+1}] {title}: {content_preview}"
            else:
                result_summary += f"Failed: {step.output.error}"

            messages.append({"role": "user", "content": result_summary})

    # Final instruction
    messages.append({
        "role": "user",
        "content": "What is your next action? Respond with JSON only.",
    })

    return messages


def _parse_research_action(raw: str) -> dict[str, Any]:
    """Parse LLM response into action dict."""
    text = raw.strip()
    try:
        payload = parse_structured_output(
            text,
            options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
        )
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    return dict(item)
    except StructuredOutputParseError:
        pass

    raise ValueError(f"Could not parse research action JSON: {text[:200]}")


# ---------------------------------------------------------------------------
# Progress callback types
# ---------------------------------------------------------------------------

@dataclass
class ResearchProgressEvent:
    """Event emitted during research for streaming/monitoring."""

    event_type: str  # research_reasoning, research_searching, research_results, research_complete
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mode → iteration limit mapping
# ---------------------------------------------------------------------------

_MODE_MAX_ITERATIONS: dict[str, int] = {
    "speed": 2,
    "balanced": 6,
    "quality": 25,
}


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

async def research_loop(
    query: str,
    classification: QueryClassification,
    mode: Literal["speed", "balanced", "quality"] = "balanced",
    llm_provider: str = "openai",
    llm_model: str | None = None,
    max_iterations: int | None = None,
    on_progress: Callable[[ResearchProgressEvent], Any] | None = None,
    registry: ActionRegistry | None = None,
    db_context: dict[str, Any] | None = None,
    discussion_platforms: list[str] | None = None,
    enable_url_scraping: bool = True,
    enable_action_dedup: bool = True,
    enable_image_search: bool = False,
    enable_video_search: bool = False,
) -> ResearchOutput:
    """Run the iterative agentic research loop.

    Args:
        query: The user's original query.
        classification: Result from query_classifier.classify_query().
        mode: Search depth mode (speed/balanced/quality).
        llm_provider: LLM provider for the research agent.
        llm_model: Optional model override.
        max_iterations: Override max iterations (auto from mode if None).
        on_progress: Optional callback for streaming progress events.
        registry: Optional custom ActionRegistry (uses default if None).
        db_context: Optional dict with DB paths/instances for local_db_search.
        discussion_platforms: Optional default platform list for discussion_search.
        enable_url_scraping: Whether scrape_url action is available.
        enable_action_dedup: Whether to skip repeated equivalent actions and reuse prior results.
        enable_image_search: Whether image_search action is available.
        enable_video_search: Whether video_search action is available.

    Returns:
        ResearchOutput with all steps, results, and metadata.
    """
    start_time = time.time()
    standalone_query = classification.standalone_query or query

    if registry is None:
        registry = create_configured_registry(
            discussion_platforms=discussion_platforms,
            enable_url_scraping=enable_url_scraping,
            enable_image_search=enable_image_search,
            enable_video_search=enable_video_search,
            on_progress=on_progress,
        )

    if max_iterations is None:
        max_iterations = _MODE_MAX_ITERATIONS.get(mode, 6)

    output = ResearchOutput(
        query=query,
        standalone_query=standalone_query,
    )

    # URL deduplication tracking
    seen_urls: dict[str, dict[str, Any]] = {}  # url -> merged result dict
    _dedup_merged = 0
    _duplicate_fetches_skipped = 0
    seen_action_signatures: dict[tuple[Any, ...], ActionOutput] = {}
    _action_dedup_skipped = 0
    _action_dedup_reused = 0

    # Preamble enforcement tracking
    _requires_reasoning_preamble = mode in {"balanced", "quality"}
    _reasoning_preamble_completed = False
    _preamble_auto_injected = 0
    _preamble_manual_calls = 0

    # Get available actions description for the LLM
    actions_desc = registry.get_actions_description(classification)
    steps: list[ResearchStep] = []

    def _extract_result_url(result_item: dict[str, Any]) -> str:
        """Extract and normalize URL from a result item."""
        raw_url = result_item.get("url", "") or result_item.get("link", "")
        try:
            return str(raw_url).strip()
        except Exception:
            return ""

    def _normalize_query(value: Any) -> str:
        try:
            return " ".join(str(value or "").strip().lower().split())
        except Exception:
            return ""

    def _normalized_tuple(values: Any) -> tuple[str, ...]:
        if not isinstance(values, (list, tuple, set)):
            return tuple()
        normalized = sorted(_normalize_query(v) for v in values if str(v or "").strip())
        return tuple(v for v in normalized if v)

    def _action_signature(action_name: str, params: dict[str, Any]) -> tuple[Any, ...]:
        q = _normalize_query(params.get("query", ""))
        if action_name == "web_search":
            return (
                action_name,
                q,
                _normalize_query(params.get("engine", "duckduckgo")),
                int(params.get("result_count", 5)),
            )
        if action_name == "academic_search":
            return (
                action_name,
                q,
                int(params.get("result_count", 5)),
            )
        if action_name == "discussion_search":
            return (
                action_name,
                q,
                _normalized_tuple(params.get("platforms") or []),
                int(params.get("max_results", 10)),
            )
        if action_name == "local_db_search":
            return (
                action_name,
                q,
                _normalized_tuple(params.get("sources") or []),
                int(params.get("top_k", 10)),
            )
        return (action_name,)

    async def _emit(event_type: str, data: dict[str, Any]) -> None:
        if on_progress is not None:
            event = ResearchProgressEvent(event_type=event_type, data=data)
            try:
                result = on_progress(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as progress_error:
                logger.debug(
                    "Research iteration progress callback failed",
                    error_type=type(progress_error).__name__,
                )

    for iteration in range(1, max_iterations + 1):
        logger.debug(f"Research loop iteration {iteration}/{max_iterations}")

        # Build prompt with context from previous steps
        messages = _build_research_prompt(
            query=query,
            standalone_query=standalone_query,
            available_actions=actions_desc,
            steps_so_far=steps,
            mode=mode,
        )

        # Call LLM for next action decision
        try:
            from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

            provider = (llm_provider or "openai").strip().lower()
            model = (llm_model or "").strip() or None

            call_kwargs: dict[str, Any] = {
                "api_provider": provider,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 500,
                "stream": False,
            }
            if model:
                call_kwargs["model"] = model

            raw_response = await asyncio.wait_for(
                perform_chat_api_call_async(**call_kwargs),
                timeout=30.0,
            )

            # Extract text
            response_text = ""
            if isinstance(raw_response, str):
                response_text = raw_response
            elif isinstance(raw_response, dict):
                choices = raw_response.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    response_text = msg.get("content", "")
                if not response_text:
                    response_text = raw_response.get("content", "") or raw_response.get("text", "")
            elif hasattr(raw_response, "content"):
                response_text = str(raw_response.content)
            else:
                response_text = str(raw_response)

            if not response_text.strip():
                logger.warning("Empty LLM response in research loop, stopping")
                break

            action_dict = _parse_research_action(response_text)

        except Exception as exc:
            logger.warning(f"Research loop LLM call failed at iteration {iteration}: {exc!r}")
            break

        action_name = action_dict.get("action", "done")
        action_params = action_dict.get("params", {})
        reasoning = action_dict.get("reasoning", "")

        # Emit reasoning event
        await _emit("research_reasoning", {
            "step": iteration,
            "text": reasoning,
            "action": action_name,
        })

        # Check for done action
        if action_name == "done":
            step = ResearchStep(
                iteration=iteration,
                reasoning=reasoning,
                action_name="done",
                action_params=action_params,
            )
            steps.append(step)
            output.final_reasoning = action_params.get("reason", reasoning)
            output.completed = True
            await _emit("research_complete", {
                "total_iterations": iteration,
                "reason": output.final_reasoning,
            })
            break

        # Balanced/quality enforcement: run reasoning preamble before first tool call.
        if (
            _requires_reasoning_preamble
            and not _reasoning_preamble_completed
            and action_name != "__reasoning_preamble"
        ):
            preamble_params = {
                "reasoning": reasoning or (
                    f"Planning required before using action '{action_name}'. "
                    "Gather broad coverage first, then refine."
                ),
                "plan": f"Initial intended action: {action_name}",
            }
            preamble_output = await registry.execute("__reasoning_preamble", preamble_params)
            if preamble_output.success:
                _reasoning_preamble_completed = True
            _preamble_auto_injected += 1
            logger.debug(
                "Auto-injected __reasoning_preamble before '{}' in {} mode",
                action_name,
                mode,
            )

        # Inject DB context into local_db_search params
        if action_name == "local_db_search" and db_context:
            for key in ("media_db_path", "notes_db_path", "character_db_path",
                         "kanban_db_path", "media_db", "chacha_db"):
                if key in db_context and key not in action_params:
                    action_params[key] = db_context[key]

        action_signature = _action_signature(action_name, action_params)
        if (
            enable_action_dedup
            and action_name not in {"scrape_url", "__reasoning_preamble"}
            and action_signature in seen_action_signatures
        ):
            cached_output = seen_action_signatures[action_signature]
            if cached_output.success and cached_output.result_count > 0:
                _action_dedup_skipped += 1
                _action_dedup_reused += int(cached_output.result_count)
                await _emit("research_searching", {
                    "step": iteration,
                    "action": action_name,
                    "queries": [action_params.get("query", query)],
                    "action_reused": True,
                })
                action_output = cached_output
                step_duration = 0.0
            else:
                action_output = None
                step_duration = 0.0
        else:
            action_output = None
            step_duration = 0.0

        # Skip duplicate URL fetch/scrape by reusing already seen result content.
        url_reused = False
        reused_url = ""
        if action_output is not None:
            pass
        elif action_name == "scrape_url":
            candidate_url = str(action_params.get("url", "")).strip()
            if candidate_url and candidate_url in seen_urls:
                url_reused = True
                reused_url = candidate_url
                _duplicate_fetches_skipped += 1
                await _emit("research_searching", {
                    "step": iteration,
                    "action": action_name,
                    "queries": [candidate_url],
                    "url_reused": True,
                })
                action_output = ActionOutput(
                    action_name="scrape_url",
                    success=True,
                    results=[seen_urls[candidate_url]],
                    result_count=1,
                    metadata={
                        "url_reused": True,
                        "reused_url": candidate_url,
                    },
                )
                step_duration = 0.0
            else:
                # Emit searching event
                await _emit("research_searching", {
                    "step": iteration,
                    "action": action_name,
                    "queries": [action_params.get("url") or action_params.get("query", query)],
                })
                # Execute the action
                step_start = time.time()
                action_output = await registry.execute(action_name, action_params)
                step_duration = time.time() - step_start
        else:
            # Emit searching event
            await _emit("research_searching", {
                "step": iteration,
                "action": action_name,
                "queries": [action_params.get("query", query)],
            })

            # Execute the action
            step_start = time.time()
            action_output = await registry.execute(action_name, action_params)
            step_duration = time.time() - step_start

        if (
            action_name not in {"scrape_url", "__reasoning_preamble"}
            and action_output is not None
        ):
            seen_action_signatures[action_signature] = action_output

        if action_name == "__reasoning_preamble" and action_output.success:
            _reasoning_preamble_completed = True
            _preamble_manual_calls += 1

        step = ResearchStep(
            iteration=iteration,
            reasoning=reasoning,
            action_name=action_name,
            action_params=action_params,
            output=action_output,
            duration_sec=step_duration,
        )
        steps.append(step)

        # Collect results with URL deduplication
        if action_output.success and action_output.results:
            for result_item in action_output.results:
                url = _extract_result_url(result_item)
                if url and url in seen_urls:
                    # Merge: append new content if different, update score if higher
                    existing = seen_urls[url]
                    new_content = str(result_item.get("content", ""))
                    existing_content = str(existing.get("content", ""))
                    if new_content and new_content not in existing_content:
                        existing["content"] = f"{existing_content}\n\n{new_content}"
                    new_score = result_item.get("score", 0)
                    if isinstance(new_score, (int, float)) and new_score > existing.get("score", 0):
                        existing["score"] = new_score
                    _dedup_merged += 1
                else:
                    output.all_results.append(result_item)
                    if url:
                        seen_urls[url] = result_item

        # Emit results event
        await _emit("research_results", {
            "step": iteration,
            "action": action_name,
            "count": action_output.result_count,
            "success": action_output.success,
            "url_reused": url_reused,
            "reused_url": reused_url,
        })

    # Finalize output
    output.steps = steps
    output.total_iterations = len(steps)
    output.total_results = len(output.all_results)
    output.total_duration_sec = time.time() - start_time

    if not output.completed:
        output.final_reasoning = "Iteration limit reached"
        output.completed = True
        await _emit("research_complete", {
            "total_iterations": output.total_iterations,
            "reason": "iteration_limit",
        })

    # Add preamble and dedup stats to output metadata
    output.metadata["reasoning_preamble"] = {
        "required": _requires_reasoning_preamble,
        "completed": _reasoning_preamble_completed,
        "manual_calls": _preamble_manual_calls,
        "auto_injected": _preamble_auto_injected,
    }
    output.metadata["url_dedup"] = {
        "urls_seen": len(seen_urls),
        "duplicates_merged": _dedup_merged,
        "duplicate_fetches_skipped": _duplicate_fetches_skipped,
    }
    output.metadata["action_dedup"] = {
        "enabled": bool(enable_action_dedup),
        "duplicates_skipped": _action_dedup_skipped,
        "reused_results_count": _action_dedup_reused,
    }
    if _dedup_merged > 0 or _duplicate_fetches_skipped > 0:
        logger.debug(
            "URL dedup: {} unique URLs, {} duplicates merged, {} duplicate fetches skipped",
            len(seen_urls),
            _dedup_merged,
            _duplicate_fetches_skipped,
        )

    return output
