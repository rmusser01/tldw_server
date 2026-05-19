"""Study suggestion context and topic-resolution helpers."""

from .flashcard_adapter import build_flashcard_suggestion_context, is_source_grounded_session
from .quiz_adapter import build_quiz_suggestion_context
from .topic_pipeline import (
    normalize_topic_labels,
    rank_suggestion_topics,
    resolve_topic_candidates,
)
from .types import NormalizedTopicLabel, RankedTopic, SuggestionContext, TopicCandidate

__all__ = [
    "NormalizedTopicLabel",
    "RankedTopic",
    "SuggestionContext",
    "TopicCandidate",
    "build_flashcard_suggestion_context",
    "build_quiz_suggestion_context",
    "is_source_grounded_session",
    "normalize_topic_labels",
    "rank_suggestion_topics",
    "resolve_topic_candidates",
]
