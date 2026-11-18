"""
Structured Response Writer.

Formats retrieved context as XML-tagged blocks and builds mode-specific
writer prompts that enforce inline citations, structured formatting,
and appropriate depth. Inspired by Perplexica's writer prompt pattern.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_page_location(metadata: dict[str, Any]) -> str:
    """Separate physical and printed pages without trusting arbitrary prompt markup."""
    spans = metadata.get("page_spans")
    if not isinstance(spans, list) or not spans:
        spans = [{"page_number": metadata.get("page_number")}]
    locations = []
    for span in spans[:20]:
        if not isinstance(span, dict):
            continue
        number = span.get("page_number")
        if type(number) is not int or number < 1:
            continue
        location = f"PDF page {number}"
        label = span.get("printed_page_label")
        if isinstance(label, str) and re.fullmatch(r"[\w.-]{1,24}", label):
            location += f" (printed page {label})"
        locations.append(location)
    return "; ".join(locations) or "Page location unavailable"


def format_page_content(content: str, metadata: dict[str, Any]) -> str:
    """Mark exact page boundaries only when bounded spans account for the whole passage."""
    spans = metadata.get("page_spans")
    start, end = metadata.get("start_char"), metadata.get("end_char")

    def offset(value: Any) -> bool:
        return type(value) is int and 0 <= value <= 9_007_199_254_740_991

    if (not isinstance(spans, list) or not 1 <= len(spans) <= 20
            or not offset(start) or not offset(end) or end - start != len(content)):
        return content
    parts = []
    previous_end, previous_page = start, 0
    for span in spans:
        if not isinstance(span, dict):
            return content
        page, lower, upper = (span.get(key) for key in ("page_number", "start_char", "end_char"))
        if (not all(offset(value) for value in (page, lower, upper)) or page <= previous_page
                or not previous_end <= lower < upper <= end or (previous_page == 0 and lower != start)):
            return content
        gap = content[previous_end - start:lower - start]
        if re.fullmatch(r"[ \t\r\n\f\v]*", gap) is None:
            return content
        location = format_page_location({"page_spans": [span]})
        parts.extend((gap, f"\n[{location} begins]\n", content[lower - start:upper - start]))
        previous_end, previous_page = upper, page
    return "".join(parts) if previous_end == end else content


def format_context_xml(chunks: list[dict[str, Any]]) -> str:
    """Format retrieval results as XML-tagged context blocks.

    Each chunk dict should have: content, title (optional), url/source (optional).
    Results are auto-indexed starting from 1.

    Returns:
        XML string with <result> elements wrapped in a <context> root.
    """
    if not chunks:
        return "<context>\n  (no results)\n</context>"

    parts = ["<context>"]
    emitted = 0
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        if not content.strip():
            continue
        emitted += 1

        title = chunk.get("title", "") or ""
        source = chunk.get("url", "") or chunk.get("source", "") or ""
        metadata = chunk.get("metadata", {}) or {}
        if not title and isinstance(metadata, dict):
            title = metadata.get("title", "")
        if not source and isinstance(metadata, dict):
            source = metadata.get("source", "") or metadata.get("url", "")

        attrs = f'index="{emitted}"'
        if title:
            safe_title = escape(str(title), quote=True)
            attrs += f' title="{safe_title}"'
        if source:
            safe_source = escape(str(source), quote=True)
            attrs += f' source="{safe_source}"'

        location = format_page_location(metadata) if isinstance(metadata, dict) else "Page location unavailable"
        if isinstance(metadata, dict):
            content = format_page_content(content, metadata)
        safe_content = escape(f"{location}\n{content}", quote=False)
        parts.append(f"  <result {attrs}>{safe_content}</result>")

    if emitted == 0:
        parts.append("  (no results)")

    parts.append("</context>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Writer system prompts per mode
# ---------------------------------------------------------------------------

_WRITER_BASE_RULES = """\
You are a knowledgeable assistant generating a response based on the provided context.

## Citation Rules
- EVERY factual claim must include at least one citation using [number] notation
- Citations reference the <result index="N"> elements in the context
- Place citations at the end of the relevant sentence, before the period: "fact [1]."
- Multiple citations can be grouped: "supported by multiple studies [1][3]."
- Do NOT fabricate information not present in the context
- If the context doesn't contain sufficient information, say so explicitly

## Formatting Rules
- Use clear markdown formatting: headings, subheadings, bullet points
- Start with a direct answer to the query
- Organize information logically with appropriate section headers
- Use paragraphs for detailed explanations, bullets for lists/comparisons"""

_SPEED_WRITER_PROMPT = _WRITER_BASE_RULES + """

## Mode: Speed (Concise)
- Keep the response concise and direct (150-400 words)
- Focus on answering the core question
- Use 1-2 short paragraphs, optionally with a brief bullet list
- Skip deep analysis — prioritize clarity and speed
- Include a brief 1-sentence conclusion if appropriate"""

_BALANCED_WRITER_PROMPT = _WRITER_BASE_RULES + """

## Mode: Balanced (Comprehensive)
- Provide a well-structured, moderately detailed response (400-1200 words)
- Use 2-4 sections with descriptive headings
- Include context, explanation, and practical implications
- Synthesize information from multiple sources when possible
- End with a concise conclusion or summary paragraph
- Balance depth with readability"""

_QUALITY_WRITER_PROMPT = _WRITER_BASE_RULES + """

## Mode: Quality (Research Report)
- Generate a DEEP, DETAILED, and COMPREHENSIVE response
- Target at minimum 2000 words for complex topics
- Structure as a professional research report:
  1. **Introduction**: Brief overview and scope
  2. **Background/Context**: Relevant history and definitions
  3. **Detailed Analysis**: Multiple sections covering different aspects
  4. **Comparisons**: Compare approaches, viewpoints, or alternatives where relevant
  5. **Practical Implications**: Real-world applications, use cases, or recommendations
  6. **Limitations & Considerations**: Caveats, edge cases, or areas of debate
  7. **Conclusion**: Synthesize key findings and provide a balanced summary
- Every section should cite sources extensively
- Use tables or structured comparisons where they add clarity
- Include nuanced analysis, not just surface-level summaries
- Address potential counterarguments or alternative perspectives"""


def get_writer_depth_policy(
    mode: str = "balanced",
    max_generation_tokens: int | None = None,
) -> dict[str, Any]:
    """Return depth policy metadata for the structured writer.

    This helps callers reason about whether quality mode's 2000+ word target
    is realistic for the configured output token budget.
    """
    normalized_mode = str(mode or "balanced").strip().lower()
    token_budget = None
    if isinstance(max_generation_tokens, int) and max_generation_tokens > 0:
        token_budget = max_generation_tokens

    # Approximation: ~0.75 words/token => ~2666 tokens for 2000 words.
    quality_word_target = 2000
    quality_required_tokens = 2666

    degraded = False
    target_word_min = 150
    target_word_max = 400

    if normalized_mode == "balanced":
        target_word_min = 400
        target_word_max = 1200
    elif normalized_mode == "quality":
        target_word_min = quality_word_target
        target_word_max = 3000
        if token_budget is not None and token_budget < quality_required_tokens:
            degraded = True
            # Keep lower bound practical while encouraging depth.
            est_words = max(600, int(token_budget * 0.75))
            target_word_min = est_words
            target_word_max = max(est_words + 200, int(est_words * 1.25))

    return {
        "mode": normalized_mode,
        "max_generation_tokens": token_budget,
        "quality_word_target": quality_word_target,
        "quality_required_tokens": quality_required_tokens,
        "degraded_due_to_token_budget": degraded,
        "target_word_range": [target_word_min, target_word_max],
    }


def build_writer_system_prompt(
    mode: str = "balanced",
    system_instructions: str = "",
    max_generation_tokens: int | None = None,
) -> str:
    """Build structured response generation system prompt.

    Args:
        mode: Search depth mode (speed/balanced/quality).
        system_instructions: Optional additional instructions to prepend.
        max_generation_tokens: Optional output token budget used to adapt
            quality-mode depth expectations.

    Returns:
        Complete system prompt string for the response writer.
    """
    prompt_map = {
        "speed": _SPEED_WRITER_PROMPT,
        "balanced": _BALANCED_WRITER_PROMPT,
        "quality": _QUALITY_WRITER_PROMPT,
    }
    normalized_mode = str(mode or "balanced").strip().lower()
    base = prompt_map.get(normalized_mode, _BALANCED_WRITER_PROMPT)

    budget_adaptation = ""
    if normalized_mode == "quality":
        policy = get_writer_depth_policy(
            mode=normalized_mode,
            max_generation_tokens=max_generation_tokens,
        )
        degraded = bool(policy.get("degraded_due_to_token_budget"))
        token_budget = policy.get("max_generation_tokens")
        target_range = policy.get("target_word_range", [2000, 3000])
        if degraded:
            budget_adaptation = f"""

## Budget Adaptation
- Available max generation tokens: {token_budget}
- A strict 2000+ word minimum is likely not feasible under this cap
- Produce the deepest possible report within budget, prioritizing high-signal analysis
- Target approximately {target_range[0]}-{target_range[1]} words while preserving citations"""
        else:
            if token_budget is not None:
                budget_adaptation = f"""

## Budget Adaptation
- Available max generation tokens: {token_budget}
- Current token budget supports the 2000+ word target
- Maintain full report depth with citations in every section"""

    base = f"{base}{budget_adaptation}"

    if system_instructions:
        return f"{system_instructions}\n\n{base}"
    return base


# ---------------------------------------------------------------------------
# Writer user prompt
# ---------------------------------------------------------------------------

def build_writer_user_prompt(
    query: str,
    context_xml: str,
) -> str:
    """Build the user prompt with XML-tagged context.

    Args:
        query: The user's query.
        context_xml: Pre-formatted XML context from format_context_xml().

    Returns:
        Complete user prompt string for the response writer.
    """
    return (
        f"Using the context below, answer the following query.\n\n"
        f"Query: {query}\n\n"
        f"{context_xml}\n\n"
        f"Provide your response with proper citations [number] referencing the results above."
    )
