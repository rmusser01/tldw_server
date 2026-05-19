# Upload_Sink.py
# Description: Contains classes and functions to handle file uploads,
#              validate their safety and integrity, and perform sanitization.
#
import copy
import html
import importlib
import mimetypes
import os
import stat
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Union

#
# 3rd-party Libraries
try:
    import puremagic  # type: ignore
except ImportError:  # puremagic is optional; fall back to alternate MIME detection
    puremagic = None
try:
    import yara  # type: ignore
except ImportError:  # yara rules are optional; disable malware scanning if missing
    yara = None
import contextlib
import tarfile
import zipfile

#
# Local Imports (adjust path as per your project structure)
from tldw_Server_API.app.core.config import MAGIC_FILE_PATH, loaded_config_data
from tldw_Server_API.app.core.Utils.Utils import logging

# If the above import fails in a different context, fallback to standard logging:
# import logging
# logging.basicConfig(level=logging.INFO)

_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)
if puremagic is not None:
    _puremagic_error = getattr(puremagic, "PureError", None)
    if isinstance(_puremagic_error, type) and issubclass(_puremagic_error, Exception):
        _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS = (
            *_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS,
            _puremagic_error,
        )

_python_magic = None
_python_magic_loaded = False


def _get_python_magic_module():
    """Resolve python-magic lazily and skip it on Windows.

    The Windows CI runner has shown access violations while importing `magic`
    during module import. Keep the fallback optional and avoid that crash path.
    """
    global _python_magic, _python_magic_loaded

    if _python_magic is not None:
        return _python_magic
    if _python_magic_loaded:
        return None

    _python_magic_loaded = True
    if os.name == "nt":
        logging.info(
            "Skipping python-magic fallback on Windows; relying on puremagic or extension guesses."
        )
        return None

    try:
        _python_magic = importlib.import_module("magic")
    except ImportError:
        _python_magic = None
    except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as exc:
        logging.warning(f"Failed to import python-magic fallback: {exc}")
        _python_magic = None
    return _python_magic


class FileValidationError(Exception):
    """Custom exception for critical validation/setup errors."""

    def __init__(self, message, issues: Optional[list[str]] = None):
        super().__init__(message)
        self.issues = issues if issues is not None else [message]


class ValidationResult:
    """Holds the outcome of a file validation process."""

    def __init__(self, is_valid: bool, issues: Optional[list[str]] = None,
                 file_path: Optional[Path] = None,
                 detected_mime_type: Optional[str] = None,
                 detected_extension: Optional[str] = None):
        self.is_valid = is_valid
        self.issues = issues or []
        self.file_path = file_path
        self.detected_mime_type = detected_mime_type
        self.detected_extension = detected_extension  # Extension of the file on disk

    def __bool__(self):
        return self.is_valid

    def __str__(self):
        if self.is_valid:
            return (f"ValidationResult(is_valid=True, path='{self.file_path}', "
                    f"mime='{self.detected_mime_type}', ext='{self.detected_extension}')")
        return (f"ValidationResult(is_valid=False, issues={self.issues}, path='{self.file_path}', "
                f"mime='{self.detected_mime_type}', ext='{self.detected_extension}')")


# Get configuration values or use defaults
media_config = loaded_config_data.get('media_processing', {}) if loaded_config_data else {}

# Additional extension sets for specialized categories
CODE_FILE_EXTENSIONS: set[str] = {
    '.py', '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',
    '.cs', '.java', '.kt', '.kts', '.swift', '.rs', '.go',
    '.rb', '.php', '.pl', '.lua', '.sql', '.yaml',
    '.yml', '.toml', '.ini', '.cfg', '.conf', '.ts', '.tsx', '.jsx', '.js',
}
CODE_MIME_TYPES: set[str] = {
    'text/plain', 'text/markdown',
    # Python
    'text/x-python', 'application/x-python-code',
    # C/C++ common variants
    'text/x-c', 'text/x-csrc', 'text/x-chdr',
    'text/x-c++', 'text/x-cpp', 'text/x-c++src', 'text/x-c++hdr',
    # JavaScript/TypeScript
    'text/javascript', 'application/javascript', 'application/x-javascript', 'application/ecmascript', 'text/ecmascript',
    'text/x-typescript',
    # Other languages
    'text/x-java-source', 'text/x-kotlin', 'text/x-rust', 'text/x-go', 'text/x-ruby', 'text/x-php',
    'text/x-perl', 'text/x-sql', 'text/yaml', 'application/x-yaml', 'application/x-toml',
}

# Default configurations for common media types
# These can be overridden or extended via FileValidator's constructor
DEFAULT_MEDIA_TYPE_CONFIG = {
    "audio": {
        "allowed_extensions": {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'},
        "allowed_mimetypes": {'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/wave', 'audio/flac', 'audio/aac', 'audio/ogg',
                              'audio/mp4', 'audio/x-m4a', 'audio/mp4a-latm'},
        "max_size_mb": media_config.get('max_audio_file_size_mb', 500),
    },
    "video": {
        "allowed_extensions": {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'},
        "allowed_mimetypes": {'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm',
                              'video/x-flv'},
        "max_size_mb": media_config.get('max_video_file_size_mb', 1000),
    },
    "image": {
        "allowed_extensions": {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'},
        # SVG is treated as XML to enable sanitization; see 'xml' config
        "allowed_mimetypes": {'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'},
        "max_size_mb": 20,
    },
    "document": {  # Generic documents (JSON supported as document)
        "allowed_extensions": {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.txt', '.md', '.rtf', '.json', '.srt', '.vtt', '.ass',
        },
        "allowed_mimetypes": {
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain', 'text/markdown', 'text/rtf', 'application/rtf', 'application/x-rtf', 'application/json',
            'text/vtt', 'application/x-subrip', 'text/x-ssa', 'text/x-ass', 'application/x-ass',
        },
        "max_size_mb": media_config.get('max_document_file_size_mb', 50),
    },
    "ebook": {
        "allowed_extensions": {'.epub', '.mobi', '.azw'},
        "allowed_mimetypes": {'application/epub+zip', 'application/x-mobipocket-ebook'},
        "max_size_mb": media_config.get('max_epub_file_size_mb', 100),
    },
    "pdf": {
        "allowed_extensions": {'.pdf'},
        "allowed_mimetypes": {'application/pdf'},
        "max_size_mb": media_config.get('max_pdf_file_size_mb', 50),
    },
    "email": {
        "allowed_extensions": {'.eml', '.mbox', '.pst', '.ost'},
        # Note: some EML files may be detected as text/plain by magic; allow both
        "allowed_mimetypes": {
            'message/rfc822', 'text/plain', 'application/mbox',
            'application/vnd.ms-outlook', 'application/vnd.ms-office',
        },
        "max_size_mb": media_config.get('max_document_file_size_mb', 50),
    },
    "html": {
        "allowed_extensions": {'.html', '.htm'},
        "allowed_mimetypes": {'text/html'},
        "max_size_mb": 5,
        "sanitize": bool(media_config.get('sanitize_html_uploads', True)),
    },
    "xml": {
        # Include SVG as XML for safer handling
        "allowed_extensions": {'.xml', '.opml', '.svg'},
        "allowed_mimetypes": {'text/xml', 'application/xml', 'image/svg+xml'},
        "max_size_mb": 10,
        "sanitize": bool(media_config.get('sanitize_xml_uploads', True)),
    },
    "archive": {
        "allowed_extensions": {'.zip', '.tar', '.tgz', '.tar.gz', '.tbz2', '.tar.bz2', '.txz', '.tar.xz'},
        "allowed_mimetypes": {
            'application/zip', 'application/x-zip-compressed',
            'application/x-tar', 'application/gzip', 'application/x-gzip',
            'application/x-bzip2', 'application/x-xz'
        },
        # Compressed archive file size limit (applies to the .zip/.tar* file itself)
        "archive_file_size_mb": media_config.get('max_archive_file_size_mb', 200),
        "scan_contents": True,  # Scan archive members
        "max_internal_files": media_config.get('max_archive_internal_files', 100),
        # Uncompressed aggregate size limit of all members
        "max_internal_uncompressed_size_mb": media_config.get('max_archive_uncompressed_size_mb', 200),
        # Optional per-member uncompressed size cap for defense-in-depth
        "max_member_uncompressed_size_mb": media_config.get('max_archive_member_uncompressed_size_mb', 100),
        "max_depth": media_config.get('max_archive_nesting_depth', 2),
    },
    "code": {
        "allowed_extensions": CODE_FILE_EXTENSIONS,
        "allowed_mimetypes": CODE_MIME_TYPES,
        "max_size_mb": media_config.get('max_code_file_size_mb', media_config.get('max_document_file_size_mb', 50)),
    },
}

# Mapping from extension to a media_type_key (used by the dispatcher)
# Note: Endpoints may override classification contextually. For example, the
#       code processing endpoint accepts certain structured text types like
#       '.json' as code for chunking purposes, even though the default here
#       categorizes '.json' under 'document'.
# This can be auto-generated or manually maintained for clarity.
EXT_TO_MEDIA_TYPE_KEY = {
    '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio', '.aac': 'audio', '.ogg': 'audio', '.m4a': 'audio',
    '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.mkv': 'video', '.webm': 'video', '.flv': 'video',
    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image', '.webp': 'image', '.bmp': 'image',
    # Route SVG to XML pipeline so sanitizer applies
    '.svg': 'xml',
    '.pdf': 'pdf',
    '.doc': 'document', '.docx': 'document', '.ppt': 'document', '.pptx': 'document', '.xls': 'document',
    '.xlsx': 'document',
    '.txt': 'document', '.md': 'document', '.rtf': 'document',
    '.epub': 'ebook', '.mobi': 'ebook', '.azw': 'ebook',
    '.html': 'html', '.htm': 'html',
    '.xml': 'xml', '.opml': 'xml',
    '.zip': 'archive', '.tar': 'archive', '.tgz': 'archive', '.tbz2': 'archive', '.txz': 'archive',
    '.tar.gz': 'archive', '.tar.bz2': 'archive', '.tar.xz': 'archive',
    '.eml': 'email', '.mbox': 'email', '.pst': 'email', '.ost': 'email',
    '.py': 'code', '.c': 'code', '.h': 'code', '.cpp': 'code', '.hpp': 'code', '.cc': 'code',
    '.cxx': 'code', '.cs': 'code', '.java': 'code', '.kt': 'code', '.kts': 'code', '.swift': 'code',
    '.rs': 'code', '.go': 'code', '.rb': 'code', '.php': 'code', '.pl': 'code', '.lua': 'code',
    '.sql': 'code', '.json': 'document', '.yaml': 'code', '.yml': 'code', '.toml': 'code', '.ini': 'code',
    '.cfg': 'code', '.conf': 'code', '.ts': 'code', '.tsx': 'code', '.jsx': 'code',
}

def _extension_candidates(filename: Union[str, Path]) -> list[str]:
    """Return suffix candidates (longest to shortest) for a filename/path."""
    path_obj = Path(filename)
    suffixes = [suffix.lower() for suffix in path_obj.suffixes if suffix]
    candidates: list[str] = []
    for idx in range(len(suffixes)):
        candidate = ''.join(suffixes[idx:])
        if candidate:
            candidates.append(candidate)
    return candidates


def _resolve_media_type_key(filename: Union[str, Path]) -> Optional[str]:
    """Resolve media type key using the extension candidates helper."""
    for candidate in _extension_candidates(filename):
        media_key = EXT_TO_MEDIA_TYPE_KEY.get(candidate)
        if media_key:
            return media_key
    return None


HTML_DANGEROUS_TAGS: set[str] = {"script", "style", "noscript"}


class _HTMLBlockStripper(HTMLParser):
    """Strip specified tag blocks and optionally remove all tags without regex."""

    def __init__(self, drop_tags: set[str], keep_tags: bool, strip_comments: bool = True) -> None:
        super().__init__(convert_charrefs=False)
        self._drop_tags = {tag.lower() for tag in drop_tags}
        self._keep_tags = keep_tags
        self._strip_comments = strip_comments
        self._drop_stack: list[str] = []
        self._parts: list[str] = []

    def get_output(self) -> str:
        return "".join(self._parts)

    def _append(self, text: str) -> None:
        self._parts.append(text)

    def _format_start_tag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
        self_closing: bool = False,
    ) -> str:
        parts = [f"<{tag}"]
        for name, value in attrs:
            if value is None:
                parts.append(f" {name}")
            else:
                parts.append(f' {name}="{html.escape(value, quote=True)}"')
        parts.append(" />" if self_closing else ">")
        return "".join(parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._drop_tags:
            self._drop_stack.append(tag_lower)
            return
        if self._drop_stack:
            return
        if self._keep_tags:
            self._append(self._format_start_tag(tag, attrs))
        else:
            self._append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in self._drop_tags or self._drop_stack:
            return
        if self._keep_tags:
            self._append(self._format_start_tag(tag, attrs, self_closing=True))
        else:
            self._append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in self._drop_tags:
            if self._drop_stack and self._drop_stack[-1] == tag_lower:
                self._drop_stack.pop()
            return
        if self._drop_stack:
            return
        if self._keep_tags:
            self._append(f"</{tag}>")
        else:
            self._append(" ")

    def handle_data(self, data: str) -> None:
        if self._drop_stack:
            return
        self._append(data)

    def handle_entityref(self, name: str) -> None:
        if self._drop_stack:
            return
        self._append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._drop_stack:
            return
        self._append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        if self._strip_comments or self._drop_stack:
            return
        if self._keep_tags:
            self._append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        if self._drop_stack:
            return
        if self._keep_tags and not self._strip_comments:
            self._append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        if self._drop_stack:
            return
        if self._keep_tags and not self._strip_comments:
            self._append(f"<?{data}>")


def _drop_html_blocks(html_content: str, drop_tags: set[str]) -> str:
    """Remove dangerous tag blocks and comments while preserving other markup."""
    parser = _HTMLBlockStripper(drop_tags=drop_tags, keep_tags=True, strip_comments=True)
    parser.feed(html_content)
    parser.close()
    return parser.get_output()


def _strip_html_tags(html_content: str, drop_tags: set[str]) -> str:
    """Strip all HTML tags and comments to plain text."""
    parser = _HTMLBlockStripper(drop_tags=drop_tags, keep_tags=False, strip_comments=True)
    parser.feed(html_content)
    parser.close()
    return parser.get_output()


class FileValidator:
    def __init__(self, yara_rules_path: Optional[str] = None, custom_media_configs: Optional[dict] = None):
        self.magic_available = bool(puremagic)
        # Optional python-magic fallback when puremagic is unavailable
        self.python_magic_available = False
        self._python_magic_mime = None
        python_magic_module = _get_python_magic_module()
        if not self.magic_available and python_magic_module is not None:
            try:
                # Respect configured magic file if provided
                if MAGIC_FILE_PATH:
                    self._python_magic_mime = python_magic_module.Magic(mime=True, magic_file=MAGIC_FILE_PATH)
                else:
                    self._python_magic_mime = python_magic_module.Magic(mime=True)
                self.python_magic_available = True
                logging.info("python-magic is available and will be used for MIME detection fallback.")
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
                self._python_magic_mime = None
                self.python_magic_available = False
                logging.warning(f"Failed to initialize python-magic fallback: {e}")
        self.yara_available = bool(yara)
        # Prefer a neutral name; both zipfile and tarfile are stdlib
        self.archive_scanning_available = bool(zipfile or tarfile)
        # Back-compat alias to avoid breaking any external references
        self.zipfile_available = self.archive_scanning_available

        self.compiled_yara_rules = None
        # Allow environments to opt into fail-open on YARA scanner errors
        try:
            self._yara_fail_open = bool(media_config.get('yara_fail_open', False))
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
            self._yara_fail_open = False
        if self.yara_available and yara_rules_path:
            self._initialize_yara_scanner(yara_rules_path)
        elif yara_rules_path and not self.yara_available:
            logging.warning("Yara rules path provided, but yara-python is not installed. Yara scanning disabled.")

        if not self.magic_available and not self.python_magic_available:
            logging.warning(
                "puremagic/python-magic unavailable. MIME detection will rely on extension guesses. "
                "Install 'puremagic' or 'python-magic' for stronger detection."
            )

        self._custom_media_configs: dict[str, dict] = {}
        if custom_media_configs:
            for media_type, config_val in custom_media_configs.items():
                key = media_type.lower()
                self._custom_media_configs[key] = copy.deepcopy(config_val)

    def _compile_yara_rules(self, rules_path: str):
        if not self.yara_available:
            return None
        try:
            # Check if the rules file exists
            if not os.path.exists(rules_path):
                logging.warning(f"Yara rules file not found at {rules_path}. Yara scanning will be disabled.")
                return None
            rules = yara.compile(filepath=rules_path)
            logging.info(f"Yara rules compiled successfully from {rules_path}.")
            return rules
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"An unexpected error occurred during Yara rule compilation from {rules_path}: {e}")
        return None

    def _initialize_yara_scanner(self, rules_path: str):
        if not self.yara_available:
            return
        self.compiled_yara_rules = self._compile_yara_rules(rules_path)
        if self.compiled_yara_rules is None:
            logging.warning("Yara rules not loaded. Yara scanning will be disabled for this validator instance.")

    def _scan_file_with_yara(self, file_path: Path) -> tuple[bool, list[str]]:
        if not self.yara_available or not self.compiled_yara_rules:
            return True, []
        try:
            # Prefer scanning bytes to handle various file-like objects if path is abstract
            matches = self.compiled_yara_rules.match(filepath=str(file_path))
            if matches:
                match_details = [f"Rule:'{m.rule}',NS:'{m.namespace}',Tags:{m.tags},Meta:{m.meta}" for m in matches]
                logging.warning(f"Yara rule(s) matched for file: {file_path}. Matches: {match_details}")
                return False, match_details
            return True, []
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
            if getattr(self, '_yara_fail_open', False):
                logging.warning(f"Yara scanning error for {file_path}: {e}. Treating as pass due to configuration.")
                return True, []
            logging.error(f"Unexpected error scanning file {file_path} with Yara: {e}")
            return False, [f"Unexpected Yara scanning error: {e}"]

    def get_media_config(self, media_type_key: Optional[str]) -> Optional[dict]:
        if not media_type_key:
            return None
        key = media_type_key.lower()
        merged: dict = {}
        base_cfg = DEFAULT_MEDIA_TYPE_CONFIG.get(key)
        if base_cfg:
            merged.update(base_cfg)
        custom_cfg = self._custom_media_configs.get(key)
        if custom_cfg:
            merged.update(custom_cfg)
        return merged or None

    def validate_file(
            self,
            file_path: Union[str, Path],
            original_filename: Optional[str] = None,
            media_type_key: Optional[str] = None,
            allowed_extensions_override: Optional[set[str]] = None,
            allowed_mimetypes_override: Optional[set[str]] = None,
            max_size_mb_override: Optional[Union[int, float]] = None
    ) -> ValidationResult:
        issues: list[str] = []
        current_file_path = Path(file_path)

        # Determine original filename if not provided
        _original_filename = original_filename or current_file_path.name
        disk_candidates = _extension_candidates(current_file_path.name)
        disk_file_ext = disk_candidates[0] if disk_candidates else None

        # 1. Check existence and if it's a file
        if not current_file_path.exists():
            issues.append(f"File does not exist: {current_file_path}")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)
        if not current_file_path.is_file():
            issues.append(f"Path is not a file: {current_file_path}")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        # 1a. Centralized blocked extensions guard (defense-in-depth)
        BLOCKED_EXTENSIONS: set[str] = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.vbe',
            '.ws', '.wsf', '.wsc', '.wsh', '.ps1', '.ps1xml', '.ps2', '.ps2xml',
            '.psc1', '.psc2', '.msh', '.msh1', '.msh2', '.mshxml', '.msh1xml',
            '.msh2xml', '.scf', '.lnk', '.inf', '.reg', '.dll', '.app', '.sh',
            '.csh', '.ksh', '.bash', '.zsh', '.fish', '.jar',
            '.msi', '.dmg', '.pkg', '.deb', '.rpm', '.appimage', '.snap'
        }

        def _first_blocked(candidates: list[str]) -> Optional[str]:
            for c in candidates:
                # Allow .js when explicitly validating as code
                if media_type_key == 'code' and c == '.js':
                    continue
                if c in BLOCKED_EXTENSIONS:
                    return c
            return None

        blocked_claimed = _first_blocked(_extension_candidates(_original_filename))
        blocked_disk = _first_blocked(disk_candidates)
        first_blocked = blocked_claimed or blocked_disk
        if first_blocked:
            issues.append(f"File type '{first_blocked}' is not allowed for security reasons")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        # Determine configuration
        config = self.get_media_config(media_type_key)
        cfg_allowed_extensions = config.get("allowed_extensions") if config else None
        cfg_allowed_mimetypes = config.get("allowed_mimetypes") if config else None
        # Size limit: for archives use compressed file cap; others use generic max_size_mb
        cfg_max_size_mb = None
        if media_type_key == 'archive':
            cfg_max_size_mb = config.get("archive_file_size_mb") if config else None
        else:
            cfg_max_size_mb = config.get("max_size_mb") if config else None

        final_allowed_extensions = (
            allowed_extensions_override if allowed_extensions_override is not None else cfg_allowed_extensions
        )
        final_allowed_mimetypes = (
            allowed_mimetypes_override if allowed_mimetypes_override is not None else cfg_allowed_mimetypes
        )
        final_max_size_mb = max_size_mb_override if max_size_mb_override is not None else cfg_max_size_mb
        final_max_size_bytes = (final_max_size_mb * 1024 * 1024) if final_max_size_mb is not None else None

        if (
            final_allowed_extensions is None
            and final_allowed_mimetypes is None
            and final_max_size_bytes is None
        ):
            descriptor = f"media type '{media_type_key}'" if media_type_key else "this file type"
            issues.append(f"No validation rules configured for {descriptor}. Unsupported file type.")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        # 2. Check size
        try:
            file_size = current_file_path.stat().st_size
        except OSError as e:
            issues.append(f"Could not get file size for {current_file_path}: {e}")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        if file_size == 0:
            issues.append(f"File is empty: {current_file_path}")
            # Typically an error for uploads, but malware scan might still run if configured
            # For now, let's assume it's an error that prevents further validation for most cases
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        if final_max_size_bytes is not None and file_size > final_max_size_bytes:
            issues.append(
                f"File size {file_size / (1024 * 1024):.2f}MB exceeds limit of {final_max_size_bytes / (1024 * 1024):.2f}MB."
            )
            # Short-circuit early for efficiency once size limit exceeded
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        # 3. Validate extension (based on original_filename provided by user/client)
        claimed_candidates = _extension_candidates(_original_filename)
        claimed_ext_display = claimed_candidates[0] if claimed_candidates else None
        claimed_ext_display_str = claimed_ext_display or "<none>"
        ext_allowed_lower: set[str] = set()
        if final_allowed_extensions:
            ext_allowed_lower = {ext.lower() for ext in final_allowed_extensions}
            if not claimed_candidates:
                issues.append(f"Claimed filename '{_original_filename}' has no extension.")
            else:
                matched_claimed_ext = next((candidate for candidate in claimed_candidates if candidate in ext_allowed_lower), None)
                if matched_claimed_ext is None:
                    allowed_list_display = sorted(final_allowed_extensions)
                    issues.append(
                        f"Claimed extension '{claimed_ext_display_str}' from '{_original_filename}' is not allowed. "
                        f"Allowed: {allowed_list_display}"
                    )

        # 4. Validate MIME type
        detected_mime_type: Optional[str] = None
        mime_detection_source: str = "unavailable"
        normalized_allowed_mimetypes = (
            {mt.lower() for mt in final_allowed_mimetypes} if final_allowed_mimetypes else None
        )
        fallback_mime_type, _ = mimetypes.guess_type(_original_filename)
        if not fallback_mime_type:
            fallback_mime_type, _ = mimetypes.guess_type(str(current_file_path))
        if self.magic_available:
            try:
                detected_mime_type = puremagic.from_file(str(current_file_path), mime=True)
                if detected_mime_type:
                    detected_mime_type = detected_mime_type.strip()
                    mime_detection_source = "magic"
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:  # Catch other errors during MIME detection
                logging.warning(
                    "MIME magic detection failed for '%s': %s. Falling back to extension guesses.",
                    _original_filename,
                    e,
                )
                detected_mime_type = None
                mime_detection_source = "fallback"
        elif self.python_magic_available and self._python_magic_mime is not None:
            try:
                detected_mime_type = self._python_magic_mime.from_file(str(current_file_path))
                if detected_mime_type:
                    detected_mime_type = str(detected_mime_type).strip()
                    mime_detection_source = "python-magic"
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
                logging.warning(
                    "python-magic MIME detection failed for '%s': %s. Falling back to extension guesses.",
                    _original_filename,
                    e,
                )
                detected_mime_type = None
                mime_detection_source = "fallback"

        if not detected_mime_type and fallback_mime_type:
            detected_mime_type = fallback_mime_type
            if mime_detection_source == "unavailable":
                mime_detection_source = "fallback"

        if normalized_allowed_mimetypes:
            fallback_lower = (fallback_mime_type or '').lower() or None
            if detected_mime_type:
                detected_lower = detected_mime_type.lower()
                if detected_lower not in normalized_allowed_mimetypes:
                    # Hard-fail if we have a detected MIME and it is not allowed.
                    # Do NOT fall back to extension-derived MIME in this case.
                    issues.append(
                        f"Detected MIME type '{detected_mime_type}' for file '{_original_filename}' is not allowed. "
                        f"Allowed: {final_allowed_mimetypes}"
                    )
            else:
                # No detected MIME available; rely on extension guess if present
                if fallback_lower is None or fallback_lower not in normalized_allowed_mimetypes:
                    issues.append(
                        f"Unable to determine MIME for '{_original_filename}', and extension-based MIME "
                        f"'{fallback_mime_type or '<unknown>'}' is not allowed. Allowed: {final_allowed_mimetypes}"
                    )

        # For source-code files, be permissive on MIME mismatches in favor of
        # extension-based acceptance. Real-world code uploads often have ambiguous
        # or missing MIME types (e.g., text/plain, application/octet-stream).
        if media_type_key == 'code' and issues:
            mime_only = True
            for it in issues:
                s = str(it).lower()
                if ("detected mime type" in s) or ("unable to determine mime" in s):
                    continue
                mime_only = False
                break
            if mime_only:
                # Drop MIME-only issues and proceed; rely on extension and size checks.
                issues.clear()

        # For generic documents, permit extension-based acceptance when MIME is ambiguous or misclassified.
        # This handles cases where small text files are detected as vendor-specific types (e.g., application/x-mif).
        if media_type_key in {'document', 'json'} and issues and ext_allowed_lower:
            # Determine if the filename (claimed or on-disk) matches any allowed extension
            any_claimed_ok = any(candidate in ext_allowed_lower for candidate in claimed_candidates)
            any_disk_ok = any(candidate in ext_allowed_lower for candidate in (disk_candidates or []))
            if any_claimed_ok or any_disk_ok:
                mime_only = True
                for it in issues:
                    s = str(it).lower()
                    if ("detected mime type" in s) or ("unable to determine mime" in s):
                        continue
                    # Also allow the earlier claimed-extension error to remain a blocker
                    if "claimed extension" in s:
                        mime_only = False
                        break
                    mime_only = False
                    break
                if mime_only:
                    issues.clear()

        # 5. Malware Scan (Yara)
        is_safe_yara, yara_match_details = self._scan_file_with_yara(current_file_path)
        if not is_safe_yara:
            issues.append(f"Potential threat detected by Yara in '{_original_filename}'.")
            issues.extend([f"  Yara: {detail}" for detail in yara_match_details])

        if issues:
            logging.warning(f"Validation failed for '{_original_filename}' (path: {current_file_path}): {issues}")
            return ValidationResult(False, issues, current_file_path, detected_mime_type, disk_file_ext)

        claimed_ext_log = claimed_ext_display_str
        disk_ext_log = disk_file_ext or "<none>"
        logging.info(f"File '{_original_filename}' (path: {current_file_path}) validated successfully. "
                     f"MIME: {detected_mime_type}, Claimed Ext: {claimed_ext_log}, Disk Ext: {disk_ext_log}")
        return ValidationResult(True, file_path=current_file_path,
                                detected_mime_type=detected_mime_type, detected_extension=disk_file_ext)

    def validate_archive_contents(
        self,
        archive_path: Union[str, Path],
        *,
        _current_depth: int = 0,
    ) -> ValidationResult:
        """Validates an archive file and its contents."""
        archive_path_obj = Path(archive_path)
        issues: list[str] = []

        # Get config for "archive" type
        archive_config = self.get_media_config("archive")
        if not archive_config:
            issues.append("No configuration found for 'archive' type. Cannot validate archive contents.")
            return ValidationResult(False, issues, archive_path_obj)

        max_depth = max(0, int(archive_config.get("max_depth", 2)))
        if _current_depth > max_depth:
            issues.append(
                f"Archive nesting depth {_current_depth} exceeds maximum allowed depth {max_depth}."
            )
            return ValidationResult(False, issues, archive_path_obj)

        # Step 1: Validate the archive file itself
        logging.debug(f"Validating archive file: {archive_path_obj.name}")
        archive_file_validation_result = self.validate_file(archive_path_obj, media_type_key="archive")
        if not archive_file_validation_result:
            issues.append(f"Archive file '{archive_path_obj.name}' itself is invalid.")
            issues.extend([f"  - {issue}" for issue in archive_file_validation_result.issues])
            return ValidationResult(False, issues, archive_path_obj)

        if not archive_config.get("scan_contents", False):
            logging.info(f"Content scanning for archive '{archive_path_obj.name}' is disabled by config.")
            return archive_file_validation_result  # Return result of archive file validation

        if not self.archive_scanning_available:
            # Best-effort: treat the archive file itself as valid, but attach a warning
            issues.append("Archive content scanning not available. Skipping internal file validation.")
            logging.warning("Archive content scanning unavailable; returning archive file validation result with warning.")
            archive_file_validation_result.issues.extend(issues)
            return archive_file_validation_result

        # Step 2: Securely extract and validate contents
        max_internal_files = archive_config.get("max_internal_files", 100)
        max_uncompressed_size = archive_config.get("max_internal_uncompressed_size_mb", 200) * 1024 * 1024
        per_member_uncompressed_cap = archive_config.get("max_member_uncompressed_size_mb", 0)
        max_member_uncompressed_size = int(per_member_uncompressed_cap) * 1024 * 1024 if per_member_uncompressed_cap else None
        extracted_count = 0
        total_extracted_size = 0

        try:
            with tempfile.TemporaryDirectory(prefix="archive_extract_") as extract_dir_str:
                extract_dir = Path(extract_dir_str)
                logging.debug(f"Extracting archive '{archive_path_obj.name}' to '{extract_dir}'")

                # Determine archive type
                suffixes = [s.lower() for s in archive_path_obj.suffixes]
                is_zip = archive_path_obj.suffix.lower() == ".zip"
                is_tar = (
                    archive_path_obj.suffix.lower() == ".tar" or
                    suffixes[-2:] in [[".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]] or
                    archive_path_obj.suffix.lower() in {".tgz", ".tbz2", ".txz"}
                )

                if is_zip:
                    with zipfile.ZipFile(archive_path_obj, 'r') as zip_ref:
                        # Preliminary checks for zip bomb
                        if len(zip_ref.infolist()) > max_internal_files:
                            issues.append(
                                f"Archive contains too many files ({len(zip_ref.infolist())} > {max_internal_files}).")
                            return ValidationResult(False, issues, archive_path_obj)

                        # Check uncompressed size if possible (some zip files might not store this accurately)
                        total_uncompressed_size_members = sum(member.file_size for member in zip_ref.infolist())
                        if total_uncompressed_size_members > max_uncompressed_size:
                            issues.append(
                                f"Archive declared uncompressed size ({total_uncompressed_size_members / (1024 * 1024):.2f}MB) "
                                f"exceeds limit ({max_uncompressed_size / (1024 * 1024):.2f}MB).")
                            return ValidationResult(False, issues, archive_path_obj)

                        for member in zip_ref.infolist():
                            if member.is_dir():
                                continue  # Skip directories

                            # Path traversal prevention
                            # Normalize and validate the member filename
                            member_filename = member.filename
                            # Check for path traversal attempts
                            if '..' in member_filename or member_filename.startswith('/') or ':' in member_filename:
                                logging.warning(f"Skipping potentially malicious archive member: {member_filename}")
                                issues.append(f"Archive contains potentially malicious path: {member_filename}")
                                continue

                            # Additional check: ensure the extracted path stays within extract_dir
                            intended_path = (extract_dir / member_filename).resolve()
                            if not str(intended_path).startswith(str(extract_dir.resolve())):
                                logging.warning(f"Path traversal attempt detected: {member_filename}")
                                issues.append(f"Archive contains path traversal attempt: {member_filename}")
                                continue

                            # Per-member uncompressed size cap (ZIP reports uncompressed size as file_size)
                            try:
                                if max_member_uncompressed_size is not None and member.file_size > max_member_uncompressed_size:
                                    issues.append(
                                        f"Archive member exceeds per-file size cap: {member_filename} ({member.file_size} bytes)"
                                    )
                                    continue
                            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                                pass

                            extracted_count += 1
                            if extracted_count > max_internal_files:  # Double check during extraction
                                issues.append(
                                    f"Exceeded max internal file limit ({max_internal_files}) during extraction.")
                                break  # Stop extraction

                            # Reject encrypted ZIP members explicitly
                            try:
                                if getattr(member, 'flag_bits', 0) & 0x1:
                                    logging.warning(f"Encrypted ZIP member detected and rejected: {member_filename}")
                                    issues.append(f"Archive contains encrypted member: {member_filename}")
                                    continue
                            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                                # If flag_bits is absent or unexpected, continue with other checks
                                pass

                            external_type = (member.external_attr >> 16) & 0xFFFF
                            if external_type:
                                if stat.S_ISLNK(external_type):
                                    logging.warning(f"Skipping symbolic link inside archive: {member_filename}")
                                    issues.append(f"Archive contains unsupported symbolic link: {member_filename}")
                                    continue
                                # Treat entries lacking file-type bits (common on Windows zips) as regular files.
                                file_type_bits = external_type & 0xF000
                                if file_type_bits and not stat.S_ISREG(external_type):
                                    logging.warning(
                                        f"Skipping non-regular file inside archive: {member_filename} (mode={oct(external_type)})")
                                    issues.append(
                                        f"Archive contains unsupported entry type (mode {oct(external_type)}): {member_filename}")
                                    continue

                            total_extracted_size += member.file_size  # uncompressed size
                            if total_extracted_size > max_uncompressed_size:
                                issues.append(
                                    f"Exceeded max total uncompressed size ({max_uncompressed_size / (1024 * 1024)}MB) during extraction.")
                                break  # Stop extraction

                            # Secure extraction:
                            # Now that we've validated the path, proceed with extraction
                            try:
                                zip_ref.extract(member, path=extract_dir)
                                internal_file_path = extract_dir / member.filename

                                # Determine media_type_key for internal file based on its extension
                                internal_media_type_key = _resolve_media_type_key(internal_file_path)

                                if not internal_media_type_key:
                                    issues.append(
                                        f"Archive contains file with unsupported type: {member.filename}"
                                    )
                                    continue

                                logging.debug(
                                    f"Validating internal file: {member.filename} (as {internal_media_type_key or 'generic'})")
                                internal_validation_result = self.validate_file(
                                    internal_file_path,
                                    original_filename=internal_file_path.name,  # Use its own name
                                    media_type_key=internal_media_type_key
                                )
                                if not internal_validation_result:
                                    issues.append(
                                        f"Invalid file in archive '{archive_path_obj.name}': '{member.filename}'")
                                    issues.extend([f"    - {issue}" for issue in internal_validation_result.issues])
                                    # Option: break on first internal error or collect all
                                elif (
                                    internal_media_type_key == "archive"
                                    and archive_config.get("scan_contents")
                                ):
                                    nested_validation = self.validate_archive_contents(
                                        internal_file_path,
                                        _current_depth=_current_depth + 1,
                                    )
                                    if not nested_validation:
                                        issues.append(
                                            f"Invalid nested archive in '{archive_path_obj.name}': '{member.filename}'"
                                        )
                                        issues.extend([f"    - {issue}" for issue in nested_validation.issues])
                            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as extract_err:
                                issues.append(
                                    f"Error extracting/validating internal file '{member.filename}': {extract_err}")
                                logging.error(f"Error extracting internal file '{member.filename}': {extract_err}",
                                              exc_info=True)
                                break  # Stop on extraction error for an internal file

                elif is_tar:
                    try:
                        from pathlib import PurePosixPath
                        with tarfile.open(archive_path_obj, 'r:*') as tar:
                            members = list(tar.getmembers())
                            if len(members) > max_internal_files:
                                issues.append(
                                    f"Archive contains too many files ({len(members)} > {max_internal_files}).")
                                return ValidationResult(False, issues, archive_path_obj)

                            total_uncompressed_size_members = sum(m.size for m in members if m.isfile())
                            if total_uncompressed_size_members > max_uncompressed_size:
                                issues.append(
                                    f"Archive declared uncompressed size ({total_uncompressed_size_members / (1024 * 1024):.2f}MB) "
                                    f"exceeds limit ({max_uncompressed_size / (1024 * 1024):.2f}MB).")
                                return ValidationResult(False, issues, archive_path_obj)

                            for member in members:
                                if member.isdir():
                                    continue
                                if member.issym() or member.islnk():
                                    logging.warning(f"Skipping symbolic/hard link inside archive: {member.name}")
                                    issues.append(f"Archive contains unsupported link entry: {member.name}")
                                    continue
                                if not member.isfile():
                                    logging.warning(
                                        f"Skipping non-file archive member: {member.name} (type={member.type})")
                                    issues.append(
                                        f"Archive contains unsupported member type ({member.type}): {member.name}")
                                    continue
                                # Normalize member path to POSIX-like separators for consistent checks
                                member_filename = str(PurePosixPath(member.name))
                                if '..' in member_filename or member_filename.startswith('/') or ':' in member_filename:
                                    logging.warning(f"Skipping potentially malicious archive member: {member_filename}")
                                    issues.append(f"Archive contains potentially malicious path: {member_filename}")
                                    continue
                                # Per-member uncompressed size cap
                                try:
                                    if max_member_uncompressed_size is not None and member.size > max_member_uncompressed_size:
                                        issues.append(
                                            f"Archive member exceeds per-file size cap: {member.name} ({member.size} bytes)"
                                        )
                                        continue
                                except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                                    pass
                                intended_path = (extract_dir / member_filename).resolve()
                                if not str(intended_path).startswith(str(extract_dir.resolve())):
                                    logging.warning(f"Path traversal attempt detected: {member_filename}")
                                    issues.append(f"Archive contains path traversal attempt: {member_filename}")
                                    continue

                                extracted_count += 1
                                if extracted_count > max_internal_files:
                                    issues.append(
                                        f"Exceeded max internal file limit ({max_internal_files}) during extraction.")
                                    break

                                total_extracted_size += member.size
                                if total_extracted_size > max_uncompressed_size:
                                    issues.append(
                                        f"Exceeded max total uncompressed size ({max_uncompressed_size / (1024 * 1024)}MB) during extraction.")
                                    break

                                try:
                                    try:
                                        # Use safe extraction filter when available (Python >= 3.12)
                                        tar.extract(member, path=extract_dir, filter='data')
                                    except TypeError:
                                        # Older Python versions without 'filter' support: we validated member type
                                        # and path above; continue extracting with the safer prerequisites enforced.
                                        tar.extract(member, path=extract_dir)
                                    internal_file_path = extract_dir / member.name
                                    internal_media_type_key = _resolve_media_type_key(internal_file_path)

                                    if not internal_media_type_key:
                                        issues.append(
                                            f"Archive contains file with unsupported type: {member.name}"
                                        )
                                        continue

                                    logging.debug(
                                        f"Validating internal file: {member.name} (as {internal_media_type_key or 'generic'})")
                                    internal_validation_result = self.validate_file(
                                        internal_file_path,
                                        original_filename=internal_file_path.name,
                                        media_type_key=internal_media_type_key
                                    )
                                    if not internal_validation_result:
                                        issues.append(
                                            f"Invalid file in archive '{archive_path_obj.name}': '{member.name}'")
                                        issues.extend([f"    - {issue}" for issue in internal_validation_result.issues])
                                    elif (
                                        internal_media_type_key == "archive"
                                        and archive_config.get("scan_contents")
                                    ):
                                        nested_validation = self.validate_archive_contents(
                                            internal_file_path,
                                            _current_depth=_current_depth + 1,
                                        )
                                        if not nested_validation:
                                            issues.append(
                                                f"Invalid nested archive in '{archive_path_obj.name}': '{member.name}'"
                                            )
                                            issues.extend([f"    - {issue}" for issue in nested_validation.issues])
                                except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as extract_err:
                                    issues.append(
                                        f"Error extracting/validating internal file '{member.name}': {extract_err}")
                                    logging.error(f"Error extracting internal file '{member.name}': {extract_err}", exc_info=True)
                                    break
                    except tarfile.ReadError:
                        issues.append(f"Archive '{archive_path_obj.name}' is corrupted or not a valid TAR archive.")
                else:
                    issues.append(f"Unsupported archive type for content scanning: {archive_path_obj.suffix}")

                if issues:  # If any issues occurred during extraction or internal validation
                    return ValidationResult(False, issues, archive_path_obj)

        except zipfile.BadZipFile:
            issues.append(f"Archive '{archive_path_obj.name}' is corrupted or not a valid ZIP file.")
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
            issues.append(f"Error processing archive '{archive_path_obj.name}': {e}")
            logging.error(f"Error processing archive {archive_path_obj.name}: {e}", exc_info=True)

        if issues:
            return ValidationResult(False, issues, archive_path_obj)

        logging.info(f"Archive '{archive_path_obj.name}' and its contents validated successfully.")
        return ValidationResult(True, file_path=archive_path_obj)

    def sanitize_html_content(self, html_content: str, config: Optional[dict] = None) -> str:
        # Sanitizer using bleach with restricted protocols; otherwise strip tags.
        # Explicitly remove script/style/noscript contents and comments before cleaning to prevent code leakage.
        try:
            preprocessed = html_content
            try:
                from bs4 import BeautifulSoup, Comment  # type: ignore
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove script/style/noscript tags entirely (including their contents)
                for tag_name in ('script', 'style', 'noscript'):
                    for t in soup.find_all(tag_name):
                        try:
                            t.decompose()
                        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                            with contextlib.suppress(_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS):
                                t.extract()
                # Remove HTML comments which may contain scripts
                for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
                    with contextlib.suppress(_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS):
                        c.extract()
                preprocessed = str(soup)
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                try:
                    preprocessed = _drop_html_blocks(preprocessed, HTML_DANGEROUS_TAGS)
                except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                    preprocessed = html_content

            import bleach  # type: ignore
            allowed_tags = (config or {}).get("allowed_tags") or [
                'p', 'br', 'ul', 'ol', 'li', 'strong', 'em', 'b', 'i', 'u', 'a',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'code', 'pre'
            ]
            allowed_attrs = (config or {}).get("allowed_attributes") or {
                'a': ['href', 'title', 'rel', 'target'],
            }
            strip = bool((config or {}).get("strip", True))
            allowed_protocols = (config or {}).get("allowed_protocols") or ['http', 'https', 'mailto']
            cleaner = bleach.Cleaner(tags=allowed_tags, attributes=allowed_attrs, protocols=allowed_protocols, strip=strip)
            cleaned_html = cleaner.clean(preprocessed)
            # Enforce rel="noopener noreferrer" on anchors with target attribute to prevent tab-nabbing
            try:
                from bs4 import BeautifulSoup  # type: ignore
                soup2 = BeautifulSoup(cleaned_html, 'html.parser')
                for a in soup2.find_all('a'):
                    if a.has_attr('target'):
                        rel_vals = a.get('rel') or []
                        if isinstance(rel_vals, str):
                            rel_vals = rel_vals.split()
                        rel_set = set(rel_vals)
                        rel_set.update({'noopener', 'noreferrer'})
                        a['rel'] = ' '.join(sorted(rel_set))
                cleaned_html = str(soup2)
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                # If BeautifulSoup is unavailable, keep cleaned_html as-is.
                pass
            logging.info("HTML content sanitized with bleach (scripts/styles removed).")
            return cleaned_html
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
            logging.warning(f"Bleach not available or failed ({e}); falling back to tag stripping.")
            try:
                # Prefer the preprocessed (script/style/comments removed) content if available
                try:
                    _pre = preprocessed  # type: ignore[name-defined]
                except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                    _pre = html_content
                return _strip_html_tags(_pre, HTML_DANGEROUS_TAGS)
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
                # As a last resort, return the original content
                return html_content

    def sanitize_xml_content(self, xml_content: str, config: Optional[dict] = None) -> str:
        # Guarded XML parse using defusedxml; optionally strip comments/PIs.
        try:
            from defusedxml import ElementTree as DET  # type: ignore
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
            logging.warning("defusedxml not available; returning original XML content.")
            return xml_content

        try:
            root = DET.fromstring(xml_content.encode('utf-8', errors='ignore'))
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
            raise FileValidationError(f"Invalid XML content: {e}") from e

        # Optionally strip comments and processing instructions
        try:
            from xml.etree.ElementTree import Comment, ProcessingInstruction  # type: ignore
            strip_comments = bool((config or {}).get("strip_comments", True))
            strip_pi = bool((config or {}).get("strip_processing_instructions", True))
            if strip_comments or strip_pi:
                def _strip(parent):
                    for elem in list(parent):
                        tag_repr = getattr(elem, 'tag', None)
                        if strip_comments and tag_repr is Comment:
                            parent.remove(elem)
                            continue
                        if strip_pi and tag_repr is ProcessingInstruction:
                            parent.remove(elem)
                            continue
                        _strip(elem)
                _strip(root)
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
            pass

        try:
            cleaned = DET.tostring(root, encoding='unicode')
            logging.info("XML content sanitized with defusedxml.")
            return cleaned
        except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS:
            return xml_content


def process_and_validate_file(
        file_path: Union[str, Path],
        validator: FileValidator,  # Pass an initialized FileValidator instance
        original_filename: Optional[str] = None,
        # Optional: allow specifying the media_type_key directly, bypassing extension detection
        media_type_key_override: Optional[str] = None
) -> ValidationResult:
    """
    Determines media type by extension (or uses override) and validates the file.
    Handles archive content scanning if applicable.
    """
    p_file_path = Path(file_path)
    _original_filename = original_filename or p_file_path.name

    media_type_key = media_type_key_override
    if not media_type_key:
        media_type_key = _resolve_media_type_key(_original_filename)
    if not media_type_key:
        media_type_key = _resolve_media_type_key(p_file_path.name)

    if not media_type_key:
        issue = (f"Unsupported or unrecognized media type for '{_original_filename}'. "
                 "Rejecting file with unknown extension.")
        logging.warning(issue)
        return ValidationResult(
            False,
            [issue],
            p_file_path,
            detected_extension=p_file_path.suffix.lower() if p_file_path.suffix else None,
        )

    media_config = validator.get_media_config(media_type_key)

    # Perform main validation
    validation_result = validator.validate_file(p_file_path, _original_filename, media_type_key=media_type_key)

    if not validation_result:  # If basic validation failed, return immediately
        return validation_result

    # If basic validation passed, check for type-specific actions like archive scanning or sanitization
    if media_config:
        if media_type_key == "archive" and media_config.get("scan_contents"):
            logging.info(f"Performing content scan for archive: {_original_filename}")
            return validator.validate_archive_contents(p_file_path)

        # Optional content sanitization step for HTML/XML if enabled in config.
        if media_config.get("sanitize") and media_type_key in {"html", "xml"}:
            try:
                text = p_file_path.read_text(encoding='utf-8', errors='ignore')
                if media_type_key == "html":
                    sanitized = validator.sanitize_html_content(text, media_config)
                else:
                    sanitized = validator.sanitize_xml_content(text, media_config)
                if sanitized and sanitized != text:
                    p_file_path.write_text(sanitized, encoding='utf-8')
                    logging.info(f"Sanitized {media_type_key.upper()} file content: {_original_filename}")
            except _UPLOAD_SINK_NONCRITICAL_EXCEPTIONS as e:
                logging.warning(f"Sanitization failed for {_original_filename}: {e}")

    return validation_result

# Example usage (typically done at application startup):
# validator = FileValidator(
#     yara_rules_path="/path/to/your/yara_rules.yar",
#     custom_media_configs={
#         "my_custom_type": {
#             "allowed_extensions": {".custom"},
#             "allowed_mimetypes": {"application/x-custom"},
#             "max_size_mb": 5
#         }
#     }
# )
#
# # Later, when processing a file:
# # file_to_check = "/path/to/uploaded_file.mp3"
# # original_name_from_upload = "user_uploaded_music.mp3"
# # result = process_and_validate_file(file_to_check, validator, original_name_from_upload)
# #
# # if result: # True if result.is_valid is True
# #     print(f"File '{result.file_path.name}' is valid.")
# #     # Proceed with processing result.file_path
# # else:
# #     print(f"File '{original_name_from_upload}' is invalid. Issues:")
# #     for issue in result.issues:
# #         print(f"  - {issue}")
