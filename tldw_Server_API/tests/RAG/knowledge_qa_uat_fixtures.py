from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


KNOWLEDGE_QA_UAT_MANIFEST_SCHEMA_VERSION = 1

KNOWN_CITED_QUERY = "What does the grounded QA checklist require?"
KNOWN_CITED_ANSWER_PHRASE = "Grounded answers cite visible evidence"
KNOWN_DISTRACTOR_PHRASE = "A distracting checklist also mentions citations"
SCOPED_EXCLUDED_PHRASE = "Excluded distractor should not appear"
SCOPED_INCLUDED_QUERY = "What exact note rule should scoped Knowledge QA find?"
SCOPED_INCLUDED_PHRASE = "Scoped note answers must stay inside the selected source"
NO_MATCH_QUERY = "What does the library say about nonexistent basalt telemetry?"
DEGRADED_UNCITED_QUERY = "Return an unsupported Knowledge QA draft fixture"
DEGRADED_UNCITED_ANSWER = "Unsupported draft answer without inspectable citations"

KNOWN_CITED_SOURCE_TITLE = "Knowledge QA UAT Grounded Checklist"
KNOWN_DISTRACTOR_SOURCE_TITLE = "Knowledge QA UAT Distractor Checklist"
KNOWN_NOTE_SOURCE_TITLE = "Knowledge QA UAT Scoped Note"

KNOWN_CITED_SOURCE_BODY = (
    "Knowledge QA UAT fixture. "
    f"{KNOWN_CITED_ANSWER_PHRASE}. "
    "Users must be able to inspect the cited source excerpt before trusting the answer."
)

KNOWN_DISTRACTOR_SOURCE_BODY = (
    "Knowledge QA UAT distractor fixture. "
    f"{KNOWN_DISTRACTOR_PHRASE}. "
    f"{SCOPED_EXCLUDED_PHRASE}. "
    "This source must be excluded when exact scoped source filters are active."
)

KNOWN_NOTE_SOURCE_BODY = (
    "Knowledge QA UAT scoped note fixture. "
    f"{SCOPED_INCLUDED_PHRASE}. "
    "This note validates exact-note search selection."
)


@dataclass(frozen=True)
class KnowledgeQaUatSource:
    key: str
    title: str
    body: str
    source_type: str


FIXTURE_SOURCES = (
    KnowledgeQaUatSource(
        key="cited_media",
        title=KNOWN_CITED_SOURCE_TITLE,
        body=KNOWN_CITED_SOURCE_BODY,
        source_type="media_db",
    ),
    KnowledgeQaUatSource(
        key="distractor_media",
        title=KNOWN_DISTRACTOR_SOURCE_TITLE,
        body=KNOWN_DISTRACTOR_SOURCE_BODY,
        source_type="media_db",
    ),
    KnowledgeQaUatSource(
        key="scoped_note",
        title=KNOWN_NOTE_SOURCE_TITLE,
        body=KNOWN_NOTE_SOURCE_BODY,
        source_type="notes",
    ),
)


def build_fixture_manifest(created_ids: dict[str, str | int] | None = None) -> dict[str, Any]:
    """Return the deterministic manifest contract consumed by live UAT tests."""

    ids = created_ids or {}
    return {
        "schemaVersion": KNOWLEDGE_QA_UAT_MANIFEST_SCHEMA_VERSION,
        "queries": {
            "cited": KNOWN_CITED_QUERY,
            "noMatch": NO_MATCH_QUERY,
            "scopedIncluded": SCOPED_INCLUDED_QUERY,
            "degradedUncited": DEGRADED_UNCITED_QUERY,
        },
        "expected": {
            "citedAnswerPhrase": KNOWN_CITED_ANSWER_PHRASE,
            "distractorPhrase": KNOWN_DISTRACTOR_PHRASE,
            "scopedExcludedPhrase": SCOPED_EXCLUDED_PHRASE,
            "scopedIncludedPhrase": SCOPED_INCLUDED_PHRASE,
            "degradedUncitedAnswer": DEGRADED_UNCITED_ANSWER,
        },
        "sources": {
            source.key: {
                **asdict(source),
                "id": ids.get(source.key),
            }
            for source in FIXTURE_SOURCES
        },
    }
