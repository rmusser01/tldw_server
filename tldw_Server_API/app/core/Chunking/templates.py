# templates.py
"""
Template-based chunking system for flexible processing pipelines.
Provides a way to define reusable chunking strategies with preprocessing
and postprocessing stages.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger
from tldw_Server_API.app.core.testing import is_truthy

# Pre-compiled regex patterns for common operations (performance optimization)
_RE_MULTI_SPACE = re.compile(r'[ \t]+')
_RE_CRLF = re.compile(r'\r\n')


@lru_cache(maxsize=16)
def _get_linebreak_pattern(max_breaks: int) -> re.Pattern:
    """Get compiled regex for limiting consecutive line breaks (cached)."""
    return re.compile(r'\n{' + str(max_breaks + 1) + ',}')


from .chunker import Chunker
from .exceptions import TemplateError
from .regex_safety import check_pattern as _rx_check
from .regex_safety import compile_flags as _rx_flags

_TEMPLATE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    re.error,
    json.JSONDecodeError,
)


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    """Normalize loose template options into stable booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return is_truthy(value.strip().lower())
    return bool(value)


@dataclass
class TemplateStage:
    """Represents a stage in the chunking pipeline."""
    name: str  # 'preprocess', 'chunk', 'postprocess'
    operations: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ChunkingTemplate:
    """Defines a complete chunking strategy template."""
    name: str
    description: str = ""
    base_method: str = "words"  # Default chunking method
    stages: list[TemplateStage] = field(default_factory=list)
    default_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TemplateProcessor:
    """Processes text through template-defined chunking pipelines."""

    def __init__(self, chunker: Optional[Chunker] = None):
        """Initialize the template processor.

        Args:
            chunker: Optional preconfigured Chunker instance (e.g., with LLM hooks)
        """
        self._operations: dict[str, Callable] = {}
        self._register_builtin_operations()
        self._chunker = chunker

        logger.debug("TemplateProcessor initialized")

    def _get_chunker(self) -> Chunker:
        """Get a chunker instance (fresh per call unless one was provided)."""
        if self._chunker is not None:
            return self._chunker
        return Chunker()

    def _register_builtin_operations(self):
        """Register built-in preprocessing and postprocessing operations."""
        # Preprocessing operations
        self.register_operation("normalize_whitespace", self._normalize_whitespace)
        self.register_operation("remove_headers", self._remove_headers)
        self.register_operation("extract_sections", self._extract_sections)
        self.register_operation("clean_markdown", self._clean_markdown)
        self.register_operation("detect_language", self._detect_language)

        # Postprocessing operations
        self.register_operation("add_overlap", self._add_overlap)
        self.register_operation("filter_empty", self._filter_empty)
        self.register_operation("merge_small", self._merge_small)
        self.register_operation("add_metadata", self._add_metadata)
        self.register_operation("format_chunks", self._format_chunks)

    def register_operation(self, name: str, func: Callable):
        """
        Register a custom operation.

        Args:
            name: Operation name
            func: Function to execute (signature: func(data, options) -> data)
        """
        self._operations[name] = func
        logger.debug(f"Registered operation: {name}")

    def process_template(self,
                        text: str,
                        template: ChunkingTemplate,
                        **options) -> list[dict[str, Any]]:
        """
        Process text through a template pipeline.

        Args:
            text: Text to process
            template: Template defining the pipeline
            **options: Additional options overriding template defaults

        Returns:
            List of processed chunks
        """
        # Merge options with template defaults
        final_options = {**template.default_options, **options}

        # Initialize data that flows through pipeline
        data = {"text": text, "chunks": [], "metadata": {}}

        # Process each stage
        for stage in template.stages:
            if not stage.enabled:
                continue

            logger.debug(f"Processing stage: {stage.name}")

            if stage.name == "preprocess":
                data = self._run_preprocess_stage(data, stage, final_options)
            elif stage.name == "chunk":
                data = self._run_chunk_stage(data, stage, final_options, template.base_method)
            elif stage.name == "postprocess":
                data = self._run_postprocess_stage(data, stage, final_options)
            else:
                logger.warning(f"Unknown stage: {stage.name}")

        # Ensure standardized shape: List[Dict{text, metadata}]
        out_chunks: list[dict[str, Any]] = []
        for ch in data.get("chunks", []) or []:
            if isinstance(ch, dict) and 'text' in ch:
                out_chunks.append(ch)
            elif isinstance(ch, str):
                out_chunks.append({'text': ch, 'metadata': {}})
        return out_chunks

    def _run_preprocess_stage(self,
                             data: dict[str, Any],
                             stage: TemplateStage,
                             options: dict[str, Any]) -> dict[str, Any]:
        """Run preprocessing operations on text.

        Supports both {"type", "params"} and {"operation", "config"} schemas.
        """
        text = data["text"]

        for operation in stage.operations:
            # Support both schemas
            op_name = operation.get("type") or operation.get("operation")
            raw_params = operation.get("params")
            if raw_params is None:
                raw_params = operation.get("config")
            op_options = {**options, **(raw_params or {})}

            if op_name in self._operations:
                logger.debug(f"Running preprocessing: {op_name}")
                result = self._operations[op_name](text, op_options)

                # Handle different return types
                if isinstance(result, str):
                    text = result
                elif isinstance(result, dict):
                    text = result.get("text", text)
                    data["metadata"].update(result.get("metadata", {}))
            else:
                logger.warning(f"Unknown operation: {op_name}")

        data["text"] = text
        return data

    def _run_chunk_stage(self,
                        data: dict[str, Any],
                        stage: TemplateStage,
                        options: dict[str, Any],
                        base_method: str) -> dict[str, Any]:
        """Run chunking operation.

        Supports both {"method", "params"} and {"method", "config"} schemas.
        """
        text = data["text"]

        # Get chunking parameters
        if not stage.operations:
            chunk_ops = {}
        else:
            if len(stage.operations) > 1:
                logger.warning("Chunk stage defines multiple operations; only the first will be applied")
            chunk_ops = stage.operations[0]
        method = chunk_ops.get("method", base_method)
        # Allow runtime override of method (apply-time)
        if isinstance(options.get('method'), str):
            method = options['method']  # override template/base method

        # Options can be top-level or nested under "config"
        nested_cfg = dict(chunk_ops.get("config", {}) or {})
        # max_size and overlap may be provided in either place; prefer explicit top-level first
        max_size = chunk_ops.get("max_size", nested_cfg.get("max_size", options.get("max_size", 400)))
        overlap = chunk_ops.get("overlap", nested_cfg.get("overlap", options.get("overlap", 50)))

        # Additional params may be under "params" or "config"
        extra_params_src = chunk_ops.get("params")
        extra_params = dict(extra_params_src) if isinstance(extra_params_src, dict) else dict(nested_cfg)
        # Determine language override precedence
        language_override = None
        for candidate in (
            options.get('language'),
            chunk_ops.get('language') if isinstance(chunk_ops, dict) else None,
            nested_cfg.get('language'),
            extra_params.get('language') if isinstance(extra_params, dict) else None,
        ):
            if isinstance(candidate, str) and candidate:
                language_override = candidate
                break
        # Avoid passing duplicates for standard args
        if isinstance(extra_params, dict):
            for k in ("max_size", "overlap", "method", "language"):
                if k in extra_params:
                    extra_params = {kk: vv for kk, vv in extra_params.items() if kk != k}
        method_options = dict(extra_params or {})
        for k in ("hierarchical", "hierarchical_template"):
            method_options.pop(k, None)

        # Perform chunking (supports hierarchical via config and overrides)
        chunker = self._get_chunker()
        hierarchical = False
        hier_template = None
        try:
            cfg = chunk_ops.get('config', {}) if isinstance(chunk_ops, dict) else {}
            hierarchical = (
                _coerce_bool_option(options.get('hierarchical'), False) or
                _coerce_bool_option(cfg.get('hierarchical'), False) or
                _coerce_bool_option((extra_params or {}).get('hierarchical'), False)
            )
            hier_template = (
                options.get('hierarchical_template') or
                cfg.get('hierarchical_template') or
                (extra_params or {}).get('hierarchical_template')
            )
        except _TEMPLATE_NONCRITICAL_EXCEPTIONS:
            hierarchical = False
            hier_template = None

        if hierarchical or hier_template:
            chunks = chunker.chunk_text_hierarchical_flat(
                text=text,
                method=method,
                max_size=max_size,
                overlap=overlap,
                language=language_override,
                template=hier_template if isinstance(hier_template, dict) else None,
                method_options=method_options,
            )
            data["chunks"] = chunks
        else:
            chunks = chunker.chunk_text(
                text=text,
                method=method,
                max_size=max_size,
                overlap=overlap,
                language=language_override,
                **(extra_params or {})
            )
            # Normalize to dict structure
            norm = [{'text': c, 'metadata': {}} if isinstance(c, str) else c for c in (chunks or [])]
            data["chunks"] = norm
        return data

    def _run_postprocess_stage(self,
                              data: dict[str, Any],
                              stage: TemplateStage,
                              options: dict[str, Any]) -> dict[str, Any]:
        """Run postprocessing operations on chunks.

        Supports both {"type", "params"} and {"operation", "config"} schemas.
        """
        # If there are no postprocessing operations, preserve existing
        # chunk structures (including any metadata) and return early.
        # This avoids unnecessarily stripping hierarchical metadata when
        # a template declares an empty postprocessing block.
        if not stage.operations:
            return data

        chunks = data.get("chunks", [])
        # Normalize to a list of strings for postprocessors which operate on text
        texts: list[str] = []
        for c in (chunks or []):
            if isinstance(c, dict) and 'text' in c:
                texts.append(str(c.get('text', '')))
            else:
                texts.append(str(c))

        for operation in stage.operations:
            # Support both schemas
            op_name = operation.get("type") or operation.get("operation")
            raw_params = operation.get("params")
            if raw_params is None:
                raw_params = operation.get("config")
            op_options = {**options, **(raw_params or {})}

            if op_name in self._operations:
                logger.debug(f"Running postprocessing: {op_name}")
                texts = self._operations[op_name](texts, op_options)
            else:
                logger.warning(f"Unknown operation: {op_name}")

        # Keep postprocess output as simple strings; the final normalization in process_template
        # will wrap them with minimal metadata for consistency.
        data["chunks"] = texts
        return data

    # Built-in preprocessing operations
    def _normalize_whitespace(self, text: str, options: dict[str, Any]) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space (using cached pattern)
        text = _RE_MULTI_SPACE.sub(' ', text)
        # Normalize line breaks (using cached pattern)
        text = _RE_CRLF.sub('\n', text)
        # Remove excessive line breaks (coerce provided value to int safely)
        try:
            max_breaks = int(options.get("max_line_breaks", 2))
        except (TypeError, ValueError):
            max_breaks = 2
        # Use cached compiled pattern for line break normalization
        linebreak_pattern = _get_linebreak_pattern(max_breaks)
        text = linebreak_pattern.sub('\n' * max_breaks, text)
        return text.strip()

    def _remove_headers(self, text: str, options: dict[str, Any]) -> str:
        """Remove headers/footers from text."""
        patterns = options.get("patterns", [])
        for pattern in patterns:
            try:
                # Validate and compile safely
                err = _rx_check(pattern, max_len=256)
                if err:
                    logger.warning(f"Skipping unsafe header pattern: {err}")
                    continue
                flags, ferr = _rx_flags("m")  # multiline by default
                if ferr is not None:
                    flags = re.MULTILINE
                text = re.sub(pattern, '', text, flags=flags)
            except _TEMPLATE_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to apply header removal pattern; skipping. Error: {e}")
        return text

    def _extract_sections(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Extract sections from text and store as metadata."""
        raw_pat = options.get("pattern", r'^#+\s+(.+)$')
        section_pattern = r'^#+\s+(.+)$'
        try:
            err = _rx_check(raw_pat, max_len=256)
            if not err:
                section_pattern = str(raw_pat)
            else:
                logger.warning(f"Unsafe section pattern provided; using default. Reason: {err}")
        except _TEMPLATE_NONCRITICAL_EXCEPTIONS:
            # Fall back to default
            pass
        sections = []

        try:
            for match in re.finditer(section_pattern, text, re.MULTILINE):
                try:
                    title = match.group(1)
                except (AttributeError, IndexError):
                    # If no capture group provided, use whole match
                    title = match.group(0)
                sections.append({
                    "title": title,
                    "position": match.start()
                })
        except _TEMPLATE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Section extraction regex failed; returning empty sections. Error: {e}")

        return {
            "text": text,
            "metadata": {"sections": sections}
        }

    def _clean_markdown(self, text: str, options: dict[str, Any]) -> str:
        """Clean markdown formatting from text."""
        if options.get("remove_links", False):
            # Remove markdown links but keep text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        if options.get("remove_images", False):
            # Remove markdown images
            text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

        if options.get("remove_formatting", False):
            # Remove bold/italic
            text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)
            text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)

        return text

    def _detect_language(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        """Detect text language and store as metadata."""
        # Simplified language detection based on character patterns
        language = "en"  # Default

        # Check for common non-English patterns
        if re.search(r'[\u4e00-\u9fff]', text):
            language = "zh"  # Chinese
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            language = "ja"  # Japanese
        elif re.search(r'[\uac00-\ud7af]', text):
            language = "ko"  # Korean
        elif re.search(r'[\u0600-\u06ff]', text):
            language = "ar"  # Arabic

        return {
            "text": text,
            "metadata": {"detected_language": language}
        }

    # Built-in postprocessing operations
    def _add_overlap(self, chunks: list[str], options: dict[str, Any]) -> list[str]:
        """Add overlap context between chunks."""
        overlap_size = options.get("size", 50)  # characters
        overlap_marker = options.get("marker", "")

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk

            # Add end of previous chunk
            if i > 0 and overlap_size > 0:
                prev_end = chunks[i-1][-overlap_size:]
                if overlap_marker:
                    enhanced_chunk = f"{overlap_marker}\n{prev_end}\n{overlap_marker}\n{enhanced_chunk}"
                else:
                    enhanced_chunk = prev_end + enhanced_chunk

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def _filter_empty(self, chunks: list[str], options: dict[str, Any]) -> list[str]:
        """Filter out empty or too-small chunks."""
        min_length = options.get("min_length", 10)
        return [chunk for chunk in chunks if len(chunk.strip()) >= min_length]

    def _merge_small(self, chunks: list[str], options: dict[str, Any]) -> list[str]:
        """Merge small chunks together."""
        min_size = options.get("min_size", 100)
        separator = options.get("separator", "\n\n")

        merged_chunks = []
        current_chunk = ""

        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
            elif len(current_chunk) < min_size:
                # Keep merging if current chunk is still too small
                current_chunk = current_chunk + separator + chunk
            else:
                # Current chunk is large enough, save it and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks

    def _add_metadata(self, chunks: list[str], options: dict[str, Any]) -> list[str]:
        """Add metadata to chunks."""
        prefix = options.get("prefix", "")
        suffix = options.get("suffix", "")

        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced = chunk

            if prefix:
                enhanced = prefix.format(index=i, total=len(chunks)) + enhanced
            if suffix:
                enhanced = enhanced + suffix.format(index=i, total=len(chunks))

            enhanced_chunks.append(enhanced)

        return enhanced_chunks

    def _format_chunks(self, chunks: list[str], options: dict[str, Any]) -> list[str]:
        """Format chunks according to template."""
        template_str = options.get("template", "{chunk}")

        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted = template_str.format(
                chunk=chunk,
                index=i,
                total=len(chunks)
            )
            formatted_chunks.append(formatted)

        return formatted_chunks


class TemplateManager:
    """Manages loading and saving of chunking templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize template manager.

        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "template_library"
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        self._cache: dict[str, ChunkingTemplate] = {}
        self._processor = TemplateProcessor()

        # Load built-in templates
        self._load_builtin_templates()

        logger.info(f"TemplateManager initialized with dir: {self.templates_dir}")

    def _load_builtin_templates(self):
        """Load built-in template definitions."""
        # Academic paper template
        self.register_template(ChunkingTemplate(
            name="academic_paper",
            description="Template for processing academic papers",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "normalize_whitespace", "params": {"max_line_breaks": 2}},
                    {"type": "extract_sections", "params": {"pattern": r"^#+\s+(.+)$"}},
                ]),
                TemplateStage("chunk", [
                    {"method": "sentences", "max_size": 5, "overlap": 1}
                ]),
                TemplateStage("postprocess", [
                    {"type": "filter_empty", "params": {"min_length": 20}},
                    {"type": "merge_small", "params": {"min_size": 200}},
                ])
            ],
            default_options={"max_size": 5, "overlap": 1}
        ))

        # Code documentation template
        self.register_template(ChunkingTemplate(
            name="code_documentation",
            description="Template for processing code documentation",
            base_method="structure_aware",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "clean_markdown", "params": {"remove_images": True}},
                ]),
                TemplateStage("chunk", [
                    {"method": "structure_aware", "max_size": 500, "overlap": 50,
                     "params": {"preserve_code_blocks": True, "preserve_headers": True}}
                ]),
                TemplateStage("postprocess", [
                    {"type": "filter_empty", "params": {"min_length": 50}},
                ])
            ]
        ))

        # Chat conversation template
        self.register_template(ChunkingTemplate(
            name="chat_conversation",
            description="Template for processing chat conversations",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "normalize_whitespace", "params": {"max_line_breaks": 1}},
                ]),
                TemplateStage("chunk", [
                    {"method": "sentences", "max_size": 10, "overlap": 2}
                ]),
                TemplateStage("postprocess", [
                    {"type": "add_overlap", "params": {"size": 100, "marker": "---"}},
                ])
            ]
        ))

    def register_template(self, template: ChunkingTemplate):
        """Register a template in the cache."""
        self._cache[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def get_template(self, name: str) -> Optional[ChunkingTemplate]:
        """Get a template by name."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Try to load from file
        template_file = self.templates_dir / f"{name}.json"
        if template_file.exists():
            try:
                return self.load_template(template_file)
            except _TEMPLATE_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Failed to load template {name}: {e}")

        return None

    def load_template(self, path: Path) -> ChunkingTemplate:
        """Load a template from file.

        Supports both the stage-based schema and the simpler
        {preprocessing, chunking, postprocessing} schema used in the DB.
        """
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        stages: list[TemplateStage] = []

        if "stages" in data:
            # Stage-based schema
            for stage_data in data.get("stages", []):
                stages.append(TemplateStage(
                    name=stage_data["name"],
                    operations=stage_data.get("operations", []),
                    enabled=stage_data.get("enabled", True)
                ))
            base_method = data.get("base_method", "words")
            default_options = data.get("default_options", {})
        else:
            # DB/file schema with preprocessing/chunking/postprocessing
            if 'preprocessing' in data:
                stages.append(TemplateStage(
                    name='preprocess',
                    operations=data.get('preprocessing', []),
                    enabled=True
                ))

            chunk_op = data.get('chunking', {})
            stages.append(TemplateStage(
                name='chunk',
                operations=[chunk_op],
                enabled=True
            ))

            if 'postprocessing' in data:
                stages.append(TemplateStage(
                    name='postprocess',
                    operations=data.get('postprocessing', []),
                    enabled=True
                ))

            base_method = chunk_op.get('method', 'words')
            default_options = chunk_op.get('config', {}) or {}

        template = ChunkingTemplate(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            base_method=base_method,
            stages=stages,
            default_options=default_options,
            metadata=data.get("metadata", {"tags": data.get("tags", [])})
        )

        # Cache it
        self._cache[template.name] = template
        return template

    def save_template(self, template: ChunkingTemplate, path: Optional[Path] = None):
        """Save a template to file."""
        if path is None:
            path = self.templates_dir / f"{template.name}.json"

        # Convert to JSON-serializable format
        data = {
            "name": template.name,
            "description": template.description,
            "base_method": template.base_method,
            "stages": [
                {
                    "name": stage.name,
                    "operations": stage.operations,
                    "enabled": stage.enabled
                }
                for stage in template.stages
            ],
            "default_options": template.default_options,
            "metadata": template.metadata
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved template {template.name} to {path}")

    def list_templates(self) -> list[str]:
        """List available template names."""
        templates = set(self._cache.keys())

        # Add templates from files
        if self.templates_dir.exists():
            for path in self.templates_dir.glob("*.json"):
                templates.add(path.stem)

        return sorted(templates)

    def process(self, text: str, template_name: str, **options) -> list[str]:
        """
        Process text using a named template.

        Args:
            text: Text to process
            template_name: Name of template to use
            **options: Additional options

        Returns:
            List of processed chunks

        Raises:
            TemplateError: If template not found
        """
        template = self.get_template(template_name)
        if not template:
            raise TemplateError(f"Template not found: {template_name}")

        return self._processor.process_template(text, template, **options)


# ---------------- Classifier & Learner (simple heuristics) ----------------

class TemplateClassifier:
    """Simple metadata-based classifier for choosing a template."""

    @staticmethod
    def score(template_cfg: dict[str, Any], *, media_type: Optional[str], title: Optional[str], url: Optional[str], filename: Optional[str]) -> float:
        cfg = template_cfg or {}
        # Allow classifier at top-level or under chunking.config
        classifier = (cfg.get('classifier') or ((cfg.get('chunking') or {}).get('config') or {}).get('classifier')) or {}
        if not isinstance(classifier, dict):
            return 0.0
        score = 0.0
        weight_media, weight_regex = 0.5, 0.5
        # Media type match
        mts = classifier.get('media_types') or []
        if isinstance(mts, list) and media_type:
            score += weight_media if (media_type in mts) else 0.0
        # Regex matches
        regex_hits = 0
        for key, text in (('filename_regex', filename), ('title_regex', title), ('url_regex', url)):
            pat = classifier.get(key)
            if isinstance(pat, str) and pat and text:
                try:
                    # Light guardrails: ignore overlong/unsafe patterns
                    if len(pat) > 128 or _rx_check(pat, max_len=128):
                        continue
                    if re.search(pat, text, re.IGNORECASE):
                        regex_hits += 1
                except _TEMPLATE_NONCRITICAL_EXCEPTIONS:
                    pass
        score += weight_regex * (regex_hits / 3.0)
        # Clamp with min_score if provided
        try:
            min_score = float(classifier.get('min_score', 0.0))
        except (TypeError, ValueError):
            min_score = 0.0
        return score if score >= min_score else 0.0


class TemplateLearner:
    """Learn a basic boundary rule-set from an example text."""

    @staticmethod
    def learn_boundaries(example_text: str) -> dict[str, Any]:
        if not isinstance(example_text, str) or not example_text.strip():
            return {"boundaries": []}
        patterns = []
        # Headings like: Chapter 1, Section 2.3, ABSTRACT, REFERENCES, etc.
        patterns.append({"kind": "chapter", "pattern": r"^\s*(Chapter\s+\d+\b)", "flags": "im"})
        patterns.append({"kind": "section", "pattern": r"^\s*(Section\s+\d+(?:\.\d+)*\b)", "flags": "im"})
        patterns.append({"kind": "abstract", "pattern": r"^\s*Abstract\b", "flags": "im"})
        patterns.append({"kind": "references", "pattern": r"^\s*References\b", "flags": "im"})
        patterns.append({"kind": "header_atx", "pattern": r"^\s*#{1,6}\s+.+$", "flags": "m"})
        # If the text already contains these constructs, keep them; otherwise return minimal ATX detection
        return {"boundaries": patterns}
