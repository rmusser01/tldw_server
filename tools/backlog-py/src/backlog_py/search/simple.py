from __future__ import annotations


def contains_query(text: str, query: str) -> bool:
    normalized_query = " ".join(query.casefold().split())
    if not normalized_query:
        return True
    normalized_text = " ".join(text.casefold().split())
    return normalized_query in normalized_text
