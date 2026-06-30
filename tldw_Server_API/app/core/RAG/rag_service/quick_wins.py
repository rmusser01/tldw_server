# quick_wins.py
"""
Quick win features for the RAG service.

This module provides immediately useful features like spell checking,
result highlighting, cost tracking, and query templates.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import tiktoken
from loguru import logger
from spellchecker import SpellChecker

from tldw_Server_API.app.core.http_client import RetryPolicy, afetch


# 1. QUERY SPELL CHECKING
class QuerySpellChecker:
    """Spell checker for queries."""

    _MIN_CONFIDENT_FREQUENCY = 500
    _MIN_FREQUENCY_RATIO = 20.0

    def __init__(self, custom_dictionary: Optional[list[str]] = None):
        """
        Initialize spell checker.

        Args:
            custom_dictionary: Custom words to add to dictionary
        """
        self.spell_checker = SpellChecker()

        # Add custom dictionary
        if custom_dictionary:
            self.spell_checker.word_frequency.load_words(custom_dictionary)

        # Add common technical terms
        self.add_technical_terms()

    def add_technical_terms(self):
        """Add common technical terms to dictionary."""
        terms = [
            "api", "json", "xml", "html", "css", "javascript", "python",
            "database", "sql", "nosql", "mongodb", "redis", "elasticsearch",
            "docker", "kubernetes", "microservices", "serverless",
            "machine", "learning", "neural", "network", "embedding",
            "vector", "rag", "llm", "gpt", "bert", "transformer"
        ]
        self.spell_checker.word_frequency.load_words(terms)

    def _candidate_frequency(self, token: str) -> int:
        try:
            return int(self.spell_checker.word_frequency.dictionary.get(token.lower(), 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return 0

    def _should_apply_correction(
        self,
        word: str,
        correction: str,
        suggestions: set[str],
    ) -> bool:
        normalized_word = word.strip().lower()
        normalized_correction = correction.strip().lower()
        if not normalized_correction or normalized_correction == normalized_word:
            return False

        ranked_suggestions = sorted(
            {
                suggestion
                for suggestion in suggestions
                if isinstance(suggestion, str) and suggestion.strip()
            },
            key=lambda suggestion: (
                -self._candidate_frequency(suggestion),
                suggestion,
            ),
        )
        if not ranked_suggestions:
            return False

        top_frequency = self._candidate_frequency(ranked_suggestions[0])
        if top_frequency < self._MIN_CONFIDENT_FREQUENCY:
            return False

        second_frequency = (
            self._candidate_frequency(ranked_suggestions[1])
            if len(ranked_suggestions) > 1
            else 0
        )
        if second_frequency > 0:
            frequency_ratio = top_frequency / max(second_frequency, 1)
            if frequency_ratio < self._MIN_FREQUENCY_RATIO:
                return False

        return True

    def check_query(self, query: str) -> dict[str, Any]:
        """
        Check query for spelling errors.

        Args:
            query: Query to check

        Returns:
            Dictionary with corrections and suggestions
        """
        words = query.split()
        misspelled = self.spell_checker.unknown(words)

        corrections: dict[str, dict[str, Any]] = {}
        for word in misspelled:
            # Get correction suggestions
            suggestions = self.spell_checker.candidates(word)
            if suggestions:
                # Get most likely correction
                correction = self.spell_checker.correction(word) or word
                if self._should_apply_correction(word, correction, suggestions):
                    corrections[word] = {
                        "correction": correction,
                        "suggestions": list(suggestions)[:5]
                    }

        # Generate corrected query
        corrected_query = query
        for word, data in corrections.items():
            corrected_query = corrected_query.replace(word, data["correction"])

        return {
            "original": query,
            "corrected": corrected_query if corrections else query,
            "has_errors": bool(corrections),
            "corrections": corrections
        }

    def auto_correct(self, query: str, confidence_threshold: float = 0.8) -> str:
        """
        Automatically correct query if confidence is high.

        Args:
            query: Query to correct
            confidence_threshold: Minimum confidence for auto-correction

        Returns:
            Corrected query
        """
        result = self.check_query(query)

        if result["has_errors"]:
            # For now, return corrected query
            # In production, would calculate confidence
            corrected = result.get("corrected")
            return corrected if isinstance(corrected, str) else query

        return query


# 2. RESULT HIGHLIGHTING
class ResultHighlighter:
    """Highlights matched terms in search results."""

    def __init__(self, highlight_tag: str = "**", case_sensitive: bool = False):
        """
        Initialize highlighter.

        Args:
            highlight_tag: Tag to use for highlighting (e.g., "**" for markdown)
            case_sensitive: Whether to match case
        """
        self.highlight_tag = highlight_tag
        self.case_sensitive = case_sensitive

    def highlight_document(
        self,
        document: str,
        query_terms: list[str],
        context_window: int = 50
    ) -> dict[str, Any]:
        """
        Highlight query terms in document.

        Args:
            document: Document content
            query_terms: Terms to highlight
            context_window: Characters of context around matches

        Returns:
            Highlighted document with metadata
        """
        highlighted = document
        matches: list[dict[str, Any]] = []

        for term in query_terms:
            # Find all occurrences
            pattern = re.escape(term)
            flags = 0 if self.case_sensitive else re.IGNORECASE

            for re_match in re.finditer(pattern, document, flags):
                matches.append({
                    "term": term,
                    "start": re_match.start(),
                    "end": re_match.end(),
                    "context": self._extract_context(document, re_match.start(), re_match.end(), context_window)
                })

        # Sort matches by position (reverse for replacement)
        matches.sort(key=lambda x: int(x["start"]), reverse=True)

        # Apply highlights
        for match_info in matches:
            start = int(match_info["start"])
            end = int(match_info["end"])
            highlighted = (
                highlighted[:start] +
                f"{self.highlight_tag}{highlighted[start:end]}{self.highlight_tag}" +
                highlighted[end:]
            )

        return {
            "highlighted_text": highlighted,
            "matches": len(matches),
            "snippets": self._extract_snippets(document, matches, context_window)
        }

    def _extract_context(
        self,
        document: str,
        start: int,
        end: int,
        window: int
    ) -> str:
        """Extract context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(document), end + window)

        context = document[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(document):
            context = context + "..."

        return context

    def _extract_snippets(
        self,
        document: str,
        matches: list[dict],
        window: int
    ) -> list[str]:
        """Extract relevant snippets from document."""
        snippets: list[str] = []
        seen_ranges: list[tuple[int, int]] = []

        for match in matches[:5]:  # Limit to 5 snippets
            start = match["start"]
            end = match["end"]

            # Check if this range overlaps with seen ranges
            overlap = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlap = True
                    break

            if not overlap:
                snippet = self._extract_context(document, start, end, window)
                snippets.append(snippet)
                seen_ranges.append((start - window, end + window))

        return snippets


# 3. COST TRACKING
@dataclass
class LLMCost:
    """LLM API cost information."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks LLM API costs."""

    # Pricing per 1K tokens (update as needed)
    PRICING = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "anthropic": {
            # Current
            "claude-opus-4.1": {"input": 0.015, "output": 0.075},
            "claude-sonnet-4.5": {"input": 0.003, "output": 0.015},
            "claude-haiku-4.5": {"input": 0.001, "output": 0.005},
            # Back-compat
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
        "groq": {
            "mixtral-8x7b": {"input": 0.00027, "output": 0.00027},
            "llama-70b": {"input": 0.00059, "output": 0.00079},
        }
    }

    def __init__(self):
        """Initialize cost tracker."""
        self.costs: list[LLMCost] = []
        self.total_by_provider = defaultdict(float)
        self.total_by_model = defaultdict(float)

        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            logger.debug("Falling back to base tokenizer")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def track_cost(
        self,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> LLMCost:
        """
        Track cost for an LLM call.

        Args:
            provider: LLM provider
            model: Model name
            input_text: Input text
            output_text: Output text
            metadata: Additional metadata

        Returns:
            Cost information
        """
        # Count tokens
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)

        # Get pricing
        pricing = self.PRICING.get(provider, {}).get(model, {"input": 0.001, "output": 0.001})
        input_cost_per_1k = pricing["input"]
        output_cost_per_1k = pricing["output"]

        # Calculate cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost

        # Create cost record
        cost = LLMCost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
            total_cost=total_cost,
            metadata=metadata or {}
        )

        # Track
        self.costs.append(cost)
        self.total_by_provider[provider] += total_cost
        self.total_by_model[f"{provider}/{model}"] += total_cost

        logger.debug(
            f"LLM cost: {provider}/{model} - "
            f"Input: {input_tokens} tokens (${input_cost:.4f}), "
            f"Output: {output_tokens} tokens (${output_cost:.4f}), "
            f"Total: ${total_cost:.4f}"
        )

        return cost

    def get_summary(self, last_n: Optional[int] = None) -> dict[str, Any]:
        """Get cost summary."""
        costs_to_analyze = self.costs[-last_n:] if last_n else self.costs

        if not costs_to_analyze:
            return {"total_cost": 0, "call_count": 0}

        total_cost = sum(c.total_cost for c in costs_to_analyze)
        total_input_tokens = sum(c.input_tokens for c in costs_to_analyze)
        total_output_tokens = sum(c.output_tokens for c in costs_to_analyze)

        return {
            "total_cost": total_cost,
            "call_count": len(costs_to_analyze),
            "avg_cost_per_call": total_cost / len(costs_to_analyze),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_provider": dict(self.total_by_provider),
            "by_model": dict(self.total_by_model),
            "last_updated": self.costs[-1].timestamp.isoformat() if self.costs else None
        }


# 4. DEBUG MODE
class DebugMode:
    """Debug mode controller."""

    def __init__(self):
        """Initialize debug mode from environment."""
        self.enabled = os.getenv("RAG_DEBUG", "false").lower() == "true"
        self.verbose_components = os.getenv("RAG_DEBUG_COMPONENTS", "").split(",")

        if self.enabled:
            logger.info("🐛 Debug mode enabled")
            if self.verbose_components:
                logger.info(f"Verbose components: {self.verbose_components}")

    def is_enabled(self, component: Optional[str] = None) -> bool:
        """Check if debug is enabled for component."""
        if not self.enabled:
            return False

        if component:
            return not self.verbose_components or component in self.verbose_components

        return True

    def log(self, component: str, message: str, data: Optional[Any] = None):
        """Log debug message if enabled for component."""
        if self.is_enabled(component):
            if data:
                logger.debug(f"[{component}] {message}: {json.dumps(data, default=str)}")
            else:
                logger.debug(f"[{component}] {message}")


# 6. QUERY TEMPLATES
class QueryTemplate:
    """Predefined query template."""

    def __init__(self, name: str, pattern: str, variables: list[str], description: str = ""):
        """
        Initialize query template.

        Args:
            name: Template name
            pattern: Template pattern with {variables}
            variables: List of variable names
            description: Template description
        """
        self.name = name
        self.pattern = pattern
        self.variables = variables
        self.description = description

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return self.pattern.format(**kwargs)

    def validate(self, query: str) -> tuple[bool, dict[str, str]]:
        """Check if query matches template and extract variables."""
        # Simple pattern matching (would be more sophisticated in production)
        # Convert template pattern to regex
        regex_pattern = self.pattern
        for var in self.variables:
            regex_pattern = regex_pattern.replace(f"{{{var}}}", f"(?P<{var}>.*?)")

        match = re.match(regex_pattern, query)
        if match:
            return True, match.groupdict()

        return False, {}


class QueryTemplateLibrary:
    """Library of query templates."""

    def __init__(self):
        """Initialize with default templates."""
        self.templates = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default query templates."""
        templates = [
            QueryTemplate(
                "comparison",
                "What is the difference between {item1} and {item2}?",
                ["item1", "item2"],
                "Compare two items"
            ),
            QueryTemplate(
                "definition",
                "What is {term}?",
                ["term"],
                "Define a term"
            ),
            QueryTemplate(
                "how_to",
                "How do I {action}?",
                ["action"],
                "How-to question"
            ),
            QueryTemplate(
                "troubleshooting",
                "Why is {system} {problem}?",
                ["system", "problem"],
                "Troubleshooting question"
            ),
            QueryTemplate(
                "best_practice",
                "What are the best practices for {topic}?",
                ["topic"],
                "Best practices question"
            ),
            QueryTemplate(
                "explanation",
                "Explain {concept} in {context}",
                ["concept", "context"],
                "Explanation request"
            ),
            QueryTemplate(
                "timeline",
                "What happened to {subject} between {start_date} and {end_date}?",
                ["subject", "start_date", "end_date"],
                "Timeline query"
            ),
            QueryTemplate(
                "pros_cons",
                "What are the pros and cons of {option}?",
                ["option"],
                "Pros and cons analysis"
            )
        ]

        for template in templates:
            self.templates[template.name] = template

    def add_template(self, template: QueryTemplate):
        """Add a custom template."""
        self.templates[template.name] = template

    def match_query(self, query: str) -> Optional[tuple[QueryTemplate, dict[str, str]]]:
        """Find matching template for query."""
        for template in self.templates.values():
            matches, variables = template.validate(query)
            if matches:
                return template, variables

        return None

    def suggest_templates(self, partial_query: str) -> list[QueryTemplate]:
        """Suggest templates based on partial query."""
        suggestions = []

        for template in self.templates.values():
            # Simple substring matching
            if any(word in template.pattern.lower() for word in partial_query.lower().split()):
                suggestions.append(template)

        return suggestions[:5]


# 8. BATCH JOB WEBHOOKS
class WebhookNotifier:
    """Sends webhook notifications for batch jobs."""

    def __init__(self, default_timeout: int = 30):
        """
        Initialize webhook notifier.

        Args:
            default_timeout: Default timeout for webhook calls
        """
        self.default_timeout = default_timeout
        self.webhook_history: list[dict[str, Any]] = []

    async def _close_response(self, resp: Any) -> None:
        if resp is None:
            return
        close = getattr(resp, "aclose", None)
        if callable(close):
            await close()
            return
        close = getattr(resp, "close", None)
        if callable(close):
            close()

    async def send_notification(
        self,
        webhook_url: str,
        event_type: str,
        data: dict[str, Any],
        headers: Optional[dict[str, str]] = None
    ) -> bool:
        """
        Send webhook notification.

        Args:
            webhook_url: URL to send notification to
            event_type: Type of event
            data: Event data
            headers: Optional headers

        Returns:
            Success status
        """
        payload = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        resp = None
        retry_policy = RetryPolicy(attempts=1, retry_on_unsafe=False)
        try:
            resp = await afetch(
                method="POST",
                url=webhook_url,
                json=payload,
                headers=headers or {},
                timeout=self.default_timeout,
                retry=retry_policy,
            )
            success = int(resp.status_code) < 400

            # Log history
            self.webhook_history.append({
                "url": webhook_url,
                "event": event_type,
                "status": int(resp.status_code),
                "timestamp": datetime.now(),
                "success": success
            })

            if success:
                logger.info(f"Webhook sent successfully to {webhook_url}")
            else:
                logger.warning(f"Webhook failed with status {resp.status_code}")

            return success
        except Exception:
            logger.error("Webhook error")

            self.webhook_history.append({
                "url": webhook_url,
                "event": event_type,
                "error": "webhook_request_failed",
                "timestamp": datetime.now(),
                "success": False
            })

            return False
        finally:
            await self._close_response(resp)

    async def notify_batch_complete(
        self,
        webhook_url: str,
        job_id: str,
        success_rate: float,
        total_queries: int,
        processing_time: float,
        metadata: Optional[dict[str, Any]] = None
    ):
        """Send batch completion notification."""
        await self.send_notification(
            webhook_url,
            "batch_complete",
            {
                "job_id": job_id,
                "success_rate": success_rate,
                "total_queries": total_queries,
                "processing_time": processing_time,
                "metadata": metadata or {}
            }
        )

    async def notify_batch_failed(
        self,
        webhook_url: str,
        job_id: str,
        error: str,
        metadata: Optional[dict[str, Any]] = None
    ):
        """Send batch failure notification."""
        await self.send_notification(
            webhook_url,
            "batch_failed",
            {
                "job_id": job_id,
                "error": error,
                "metadata": metadata or {}
            }
        )


# Global instances
_spell_checker = None
_highlighter = None
_cost_tracker = None
_debug_mode = None
_template_library = None
_webhook_notifier = None


def get_spell_checker() -> QuerySpellChecker:
    """Get global spell checker instance."""
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = QuerySpellChecker()
    return _spell_checker


def get_highlighter() -> ResultHighlighter:
    """Get global highlighter instance."""
    global _highlighter
    if _highlighter is None:
        _highlighter = ResultHighlighter()
    return _highlighter


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def get_debug_mode() -> DebugMode:
    """Get global debug mode instance."""
    global _debug_mode
    if _debug_mode is None:
        _debug_mode = DebugMode()
    return _debug_mode


def get_template_library() -> QueryTemplateLibrary:
    """Get global template library instance."""
    global _template_library
    if _template_library is None:
        _template_library = QueryTemplateLibrary()
    return _template_library


def get_webhook_notifier() -> WebhookNotifier:
    """Get global webhook notifier instance."""
    global _webhook_notifier
    if _webhook_notifier is None:
        _webhook_notifier = WebhookNotifier()
    return _webhook_notifier


# Pipeline integration functions

async def spell_check_query(context: Any, **kwargs) -> Any:
    """
    Spell-check a query.

    Backward compatibility:
    - If ``context`` is a raw string query, return a corrected string.
    - If ``context`` is a pipeline context object, mutate and return it.
    """
    checker = get_spell_checker()

    # Unified pipeline currently calls this helper with a raw query string.
    if isinstance(context, str):
        result = checker.check_query(context)
        corrected = result.get("corrected")
        if result.get("has_errors") and isinstance(corrected, str):
            return corrected
        return context

    config_obj = getattr(context, "config", None)
    spell_cfg = config_obj.get("spell_check", {}) if isinstance(config_obj, dict) else {}
    if not spell_cfg.get("enabled", True):
        return context

    query_text = getattr(context, "query", None)
    if not isinstance(query_text, str):
        return context

    debug = get_debug_mode()
    result = checker.check_query(query_text)

    if result["has_errors"]:
        debug.log("spell_check", "Found spelling errors", result["corrections"])

        if spell_cfg.get("auto_correct", True):
            metadata = getattr(context, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                try:
                    setattr(context, "metadata", metadata)
                except Exception as metadata_attach_error:
                    logger.debug("Quick wins failed to attach metadata mapping to context")
            metadata["original_query_before_correction"] = query_text
            context.query = result["corrected"]
            metadata["spell_corrections"] = result["corrections"]
            try:
                import hashlib as _hl
                _qh = _hl.md5(
                    (getattr(context, 'query', '') or '').encode('utf-8'),
                    usedforsecurity=False,
                ).hexdigest()[:8]
                logger.info(f"Auto-corrected query hash={_qh}")
            except Exception:
                logger.info("Auto-corrected query")

    return context


async def highlight_results(context: Any, **kwargs) -> Any:
    """Highlight query terms in results."""
    if not context.config.get("highlighting", {}).get("enabled", True):
        return context

    highlighter = get_highlighter()

    # Extract query terms
    query_terms = context.query.lower().split()

    # Highlight each document
    for doc in context.documents:
        if hasattr(doc, "content"):
            result = highlighter.highlight_document(
                doc.content,
                query_terms,
                context_window=context.config.get("highlighting", {}).get("context_window", 50)
            )

            # Add highlighted version to metadata
            doc.metadata["highlighted"] = result["highlighted_text"]
            doc.metadata["match_count"] = result["matches"]
            doc.metadata["snippets"] = result["snippets"]

    return context


async def track_llm_cost(context: Any, **kwargs) -> Any:
    """Track LLM costs in pipeline."""
    if not context.config.get("cost_tracking", {}).get("enabled", True):
        return context

    tracker = get_cost_tracker()

    # Check if we have generation info
    if "generation" in context.metadata:
        gen_info = context.metadata["generation"]

        # Track cost
        cost = tracker.track_cost(
            provider=gen_info.get("provider", "unknown"),
            model=gen_info.get("model", "unknown"),
            input_text=context.query,
            output_text=context.response if hasattr(context, "response") else "",
            metadata={"query_id": context.metadata.get("query_id")}
        )

        context.metadata["cost"] = {
            "total": cost.total_cost,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens
        }

    return context
