"""Per-call hierarchical tree coordination through a live context."""

from __future__ import annotations

from typing import Any

from ..error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS
from ..exceptions import InvalidInputError
from .builder import build_hierarchy_tree
from .flatten import flatten_tree
from .models import HierarchyContext, HierarchyTextViews, ResolvedHierarchyOptions
from .spans import compute_paragraph_spans


class HierarchyService:
    """Coordinate hierarchy operations without snapshotting the context."""

    def __init__(self, context: HierarchyContext) -> None:
        self._context = context

    def flatten(self, tree: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate the public input before looking up the live normalizer."""
        if not isinstance(tree, dict):
            return []
        return flatten_tree(tree, self._context.normalize_chunk_type)

    def build_tree(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        template: dict[str, Any] | None = None,
        method_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve each call and wrap the constructed hierarchy root."""
        if not isinstance(text, str):
            raise InvalidInputError(f"Expected string input, got {type(text).__name__}")
        if not text:
            return {"type": "hierarchical", "schema_version": 1, "root": {"kind": "root", "children": []}}
        self._context._enforce_text_size(text, source="chunk_text_hierarchical_tree")
        method_opts = dict(method_options or {})
        # Hierarchical output text sanitization (default on; opt-out via method_options)
        sanitize_output = True
        if "sanitize_output" in method_opts:
            try:
                sanitize_output = bool(method_opts.get("sanitize_output"))
            except _CHUNKER_NONCRITICAL_EXCEPTIONS:
                sanitize_output = True
            method_opts.pop("sanitize_output", None)
        method = self._context._normalize_method_argument(method) or self._context.config.default_method.value
        max_size = max_size if max_size is not None else self._context.config.default_max_size
        overlap = overlap if overlap is not None else self._context.config.default_overlap
        language = language or self._context.config.language
        method = self._context._resolve_method(method, language, method_opts)

        # Sanitize once for stable offsets; output text can be sanitized or raw based on flag.
        clean_text = self._context._sanitize_input(text, suppress_security_log=True)
        output_text = clean_text if sanitize_output else text
        text_views = HierarchyTextViews(original=text, sanitized=clean_text, output=output_text)
        resolved_options = ResolvedHierarchyOptions(
            method=method,
            max_size=max_size,
            overlap=overlap,
            language=language,
            method_options=method_opts,
            sanitize_output=sanitize_output,
        )

        # Build blocks from spans
        spans = compute_paragraph_spans(text, template)
        root = build_hierarchy_tree(self._context, text_views, spans, resolved_options)
        return {
            "type": "hierarchical",
            "schema_version": 1,
            "method": method,
            "language": language,
            "max_size": max_size,
            "overlap": overlap,
            "root": root,
        }
