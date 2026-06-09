# chatbook_validators.py
# Description: Input validation and sanitization for chatbook operations
#
"""
Chatbook Input Validators
-------------------------

Provides comprehensive validation and sanitization for chatbook operations
to prevent security vulnerabilities and ensure data integrity.
"""

import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Optional

from loguru import logger


def _get_env_int(name: str, default: int) -> int:
    """Get integer value from environment variable with fallback to default."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


class ChatbookValidator:
    """Validator for chatbook operations."""

    # File size limits (configurable via environment variables)
    MAX_UPLOAD_SIZE = _get_env_int("CHATBOOKS_MAX_UPLOAD_SIZE_MB", 100) * 1024 * 1024
    MAX_UNCOMPRESSED_SIZE = _get_env_int("CHATBOOKS_MAX_UNCOMPRESSED_SIZE_MB", 500) * 1024 * 1024
    MAX_FILE_IN_ARCHIVE = _get_env_int("CHATBOOKS_MAX_FILE_IN_ARCHIVE_MB", 50) * 1024 * 1024

    # Filename patterns
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._\-]+$')
    VALID_EXTENSIONS = {'.zip', '.chatbook'}

    # Security patterns
    PATH_TRAVERSAL_PATTERN = re.compile(r'(\.\./|\.\.\\|^/|^~)')
    DANGEROUS_PATHS = {'etc', 'usr', 'bin', 'sbin', 'var', 'proc', 'sys', 'dev'}

    # Content validation (configurable via environment variables)
    MAX_NAME_LENGTH = _get_env_int("CHATBOOKS_MAX_NAME_LENGTH", 255)
    MAX_DESCRIPTION_LENGTH = _get_env_int("CHATBOOKS_MAX_DESCRIPTION_LENGTH", 5000)
    MAX_TAGS = _get_env_int("CHATBOOKS_MAX_TAGS", 50)
    MAX_TAG_LENGTH = _get_env_int("CHATBOOKS_MAX_TAG_LENGTH", 50)

    # Maximum manifest size to prevent DoS via large manifest files
    MAX_MANIFEST_SIZE = _get_env_int("CHATBOOKS_MAX_MANIFEST_SIZE_MB", 10) * 1024 * 1024

    # Maximum compression ratio to detect potential zip bombs
    # Default 100:1; increase for text-heavy archives that compress well
    MAX_COMPRESSION_RATIO = _get_env_int("CHATBOOKS_MAX_COMPRESSION_RATIO", 100)

    @classmethod
    def _is_symlink_entry(cls, zip_info: zipfile.ZipInfo) -> bool:
        """
        Check if a ZIP archive entry is a symlink.

        This performs cross-platform detection using multiple methods:
        1. Unix mode in external_attr (most common on Unix systems)
        2. Check for stat.S_IFLNK in the file mode

        Args:
            zip_info: A ZipInfo object from the archive

        Returns:
            True if the entry appears to be a symlink, False otherwise
        """
        # Extract the Unix file mode from external_attr
        # The upper 16 bits of external_attr contain the Unix mode
        unix_mode = zip_info.external_attr >> 16

        # Check if external_attr has Unix mode data (non-zero)
        if unix_mode:
            # Use stat.S_ISLNK for portable symlink detection
            if stat.S_ISLNK(unix_mode):
                return True
            # Also check the raw mode value for symlinks (0xA000 is S_IFLNK)
            if (unix_mode & 0xF000) == 0xA000:
                return True

        # Additional check: ZIP files from some tools use 0xA1ED pattern
        # This is a fallback for Unix-created archives
        return zip_info.external_attr >> 16 == 41453

    @classmethod
    def validate_filename(cls, filename: str) -> tuple[bool, Optional[str], str]:
        """
        Validate and sanitize a filename.

        Args:
            filename: The filename to validate

        Returns:
            Tuple of (is_valid, error_message, sanitized_filename)
        """
        if not filename:
            return False, "No filename provided", ""

        # Normalize path separators and sanitize traversal by collapsing to basename.
        # API path guards still enforce containment after this.
        normalized_input = filename.replace("\\", "/")
        base_filename = os.path.basename(normalized_input).strip()
        if not base_filename:
            return False, "Invalid filename", ""

        # Check length
        if len(base_filename) > cls.MAX_NAME_LENGTH:
            return False, f"Filename too long (max {cls.MAX_NAME_LENGTH} characters)", ""

        # Check extension - must end with a valid extension (not just contain it)
        lower_name = base_filename.lower()
        has_valid_extension = any(lower_name.endswith(valid_ext) for valid_ext in cls.VALID_EXTENSIONS)

        if not has_valid_extension:
            return False, f"Invalid file type. Allowed: {', '.join(cls.VALID_EXTENSIONS)}", ""

        # Sanitize filename - remove dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9._\-]', '_', base_filename)

        # Prevent double extensions that could bypass filters
        if sanitized.count('.') > 1:
            # Keep only the last extension
            parts = sanitized.rsplit('.', 1)
            if len(parts) == 2:
                name_part = parts[0].replace('.', '_')
                sanitized = f"{name_part}.{parts[1]}"

        # Ensure it ends with a valid extension after sanitization
        if not any(sanitized.lower().endswith(ext) for ext in cls.VALID_EXTENSIONS):
            # Force it to end with .zip
            sanitized = sanitized.rsplit('.', 1)[0] + '.zip' if '.' in sanitized else sanitized + '.zip'

        return True, None, sanitized

    @classmethod
    def validate_file_size(cls, file_size: int, compressed: bool = True) -> tuple[bool, Optional[str]]:
        """
        Validate file size.

        Args:
            file_size: Size in bytes
            compressed: Whether this is compressed size

        Returns:
            Tuple of (is_valid, error_message)
        """
        max_size = cls.MAX_UPLOAD_SIZE if compressed else cls.MAX_UNCOMPRESSED_SIZE
        max_mb = max_size / (1024 * 1024)

        if file_size > max_size:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum: {max_mb:.0f}MB"

        return True, None

    @classmethod
    def validate_zip_file(cls, file_path: str) -> tuple[bool, Optional[str]]:
        """
        Comprehensive validation of a ZIP file.

        Args:
            file_path: Path to the ZIP file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_path_obj = Path(file_path)

            # Check file exists
            if not file_path_obj.exists():
                return False, "File does not exist"

            # Check file size
            file_size = file_path_obj.stat().st_size
            valid, error = cls.validate_file_size(file_size, compressed=True)
            if not valid:
                return False, error

            # Verify ZIP magic number
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic[:2] != b'PK':
                    return False, "Not a valid ZIP file"

            # Open and validate ZIP contents
            with zipfile.ZipFile(file_path, 'r') as zf:
                filelist = zf.filelist

                # Check total uncompressed size
                total_uncompressed = sum(info.file_size for info in filelist)
                valid, error = cls.validate_file_size(total_uncompressed, compressed=False)
                if not valid:
                    return False, error

                # Check for zip bomb - excessive compression ratio
                # Use total compressed size from archive (sum of compress_size) for consistency
                # Threshold is configurable via CHATBOOKS_MAX_COMPRESSION_RATIO env var
                total_compressed = sum(info.compress_size for info in filelist)
                if total_compressed > 0 and total_uncompressed > 0:
                    compression_ratio = total_uncompressed / total_compressed
                    if compression_ratio > cls.MAX_COMPRESSION_RATIO:
                        return False, f"Suspicious compression ratio ({compression_ratio:.1f}:1, max {cls.MAX_COMPRESSION_RATIO}:1) - possible zip bomb"
                elif total_uncompressed > 0 and total_compressed == 0:
                    # Edge case: uncompressed data with zero compressed size is suspicious
                    return False, "Invalid archive: uncompressed data with zero compressed size"

                # Validate each file in archive
                for info in filelist:
                    # Check for null bytes in filename (can cause issues)
                    if '\x00' in info.filename:
                        return False, "Archive contains files with invalid characters"

                    # Check for symlinks FIRST (they are never allowed)
                    # Use cross-platform symlink detection
                    if cls._is_symlink_entry(info):
                        return False, "Archive contains symlinks which are not allowed"

                    # Check for path traversal (most important security check)
                    if cls._is_path_traversal(info.filename):
                        # Don't expose the actual path in error message
                        return False, "Archive contains files with unsafe paths"

                    # Check for files without extensions (could indicate truncation or obfuscation)
                    # Skip the manifest.json and directory entries
                    if not info.filename.endswith('/') and info.filename != 'manifest.json':
                        if '.' not in os.path.basename(info.filename):
                            return False, "Archive contains files with invalid names"

                    # Check individual file size
                    if info.file_size > cls.MAX_FILE_IN_ARCHIVE:
                        info.file_size / (1024 * 1024)
                        max_mb = cls.MAX_FILE_IN_ARCHIVE / (1024 * 1024)
                        return False, f"Archive contains files larger than {max_mb:.0f}MB limit"

                    # Check for suspicious compression ratio per file (consistent with overall check)
                    if info.compress_size > 0 and info.file_size > 0:
                        per_file_ratio = info.file_size / info.compress_size
                        if per_file_ratio > cls.MAX_COMPRESSION_RATIO:
                            return False, "Archive contains files with suspicious compression ratios"

                    # Check for suspicious file types
                    if cls._is_dangerous_file(info.filename):
                        return False, "Archive contains potentially dangerous file types"

                # Check for required files
                if 'manifest.json' not in zf.namelist():
                    return False, "Invalid chatbook: manifest.json not found"

                # Check manifest size to prevent DoS via large manifest files
                manifest_info = zf.getinfo('manifest.json')
                if manifest_info.file_size > cls.MAX_MANIFEST_SIZE:
                    max_mb = cls.MAX_MANIFEST_SIZE / (1024 * 1024)
                    return False, f"Manifest file too large (max {max_mb:.0f}MB)"

                # Test ZIP integrity after size/structure checks (avoids zip-bomb decompression)
                result = zf.testzip()
                if result is not None:
                    return False, f"Corrupted ZIP file: {result}"

            return True, None

        except zipfile.BadZipFile:
            return False, "Invalid or corrupted ZIP file"
        except (RuntimeError, NotImplementedError) as e:
            # Encrypted entries or unsupported compression raise these
            return False, f"Unsupported ZIP format: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error validating ZIP file: {e}")
            raise

    @classmethod
    def validate_chatbook_metadata(cls,
                                 name: str,
                                 description: str,
                                 tags: list[str] = None,
                                 categories: list[str] = None) -> tuple[bool, Optional[str]]:
        """
        Validate chatbook metadata.

        Args:
            name: Chatbook name
            description: Chatbook description
            tags: List of tags
            categories: List of categories

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate name
        if not name or not name.strip():
            return False, "Name is required"

        if len(name) > cls.MAX_NAME_LENGTH:
            return False, f"Name too long (max {cls.MAX_NAME_LENGTH} characters)"

        if not cls._is_safe_text(name):
            return False, "Name contains invalid characters"

        # Validate description
        if description:
            if len(description) > cls.MAX_DESCRIPTION_LENGTH:
                return False, f"Description too long (max {cls.MAX_DESCRIPTION_LENGTH} characters)"
            if not cls._is_safe_text(description):
                return False, "Description contains invalid characters"

        # Validate tags
        if tags:
            if len(tags) > cls.MAX_TAGS:
                return False, f"Too many tags (max {cls.MAX_TAGS})"

            for tag in tags:
                if len(tag) > cls.MAX_TAG_LENGTH:
                    return False, f"Tag too long: {tag[:20]}... (max {cls.MAX_TAG_LENGTH} characters)"
                if not cls._is_safe_text(tag):
                    return False, f"Tag contains invalid characters: {tag}"

        # Validate categories
        if categories:
            if len(categories) > 20:  # Reasonable limit for categories
                return False, "Too many categories (max 20)"

            for category in categories:
                if len(category) > cls.MAX_TAG_LENGTH:
                    return False, f"Category too long: {category[:20]}..."
                if not cls._is_safe_text(category):
                    return False, f"Category contains invalid characters: {category}"

        return True, None

    @classmethod
    def validate_media_quality(cls, media_quality: str) -> tuple[bool, Optional[str]]:
        """
        Validate media quality value.

        Args:
            media_quality: Media quality setting

        Returns:
            Tuple of (is_valid, error_message)
        """
        valid_qualities = {'thumbnail', 'compressed', 'original'}
        if media_quality.lower() not in valid_qualities:
            return False, f"Invalid media quality '{media_quality}'. Allowed: {', '.join(valid_qualities)}"
        return True, None

    @classmethod
    def validate_author(cls, author: Optional[str]) -> tuple[bool, Optional[str]]:
        """
        Validate author field.

        Args:
            author: Author name

        Returns:
            Tuple of (is_valid, error_message)
        """
        if author is None:
            return True, None

        if len(author) > cls.MAX_NAME_LENGTH:
            return False, f"Author name too long (max {cls.MAX_NAME_LENGTH} characters)"

        if not cls._is_safe_text(author):
            return False, "Author name contains invalid characters"

        return True, None

    @classmethod
    def sanitize_path(cls, path: str) -> str:
        """
        Sanitize a path to prevent traversal attacks.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        # Handle Windows paths by converting backslashes to forward slashes
        normalized = path.replace('\\', '/')

        # Remove any parent directory references
        while '../' in normalized:
            normalized = normalized.replace('../', '')
        while '..' in normalized:
            normalized = normalized.replace('..', '')

        # Remove leading slashes and drive letters
        normalized = re.sub(r'^[A-Za-z]:', '', normalized)  # Remove drive letters
        normalized = normalized.lstrip('/')
        normalized = normalized.lstrip('~')

        # Remove any dangerous directory prefixes that are known system paths
        # This ensures we remove things like "windows/system32/" prefix
        parts = normalized.split('/')
        # Filter out system directory names from the path components
        filtered_parts = []
        for part in parts:
            if part.lower() not in {'windows', 'system32', 'system', 'etc', 'usr', 'bin', 'var', 'proc', 'sys', 'dev'}:
                filtered_parts.append(part)

        # Get only the final filename component from filtered parts
        safe_name = filtered_parts[-1] if filtered_parts else ''

        # Remove any remaining dangerous patterns - ensure all special chars are replaced
        # Note: The question mark needs escaping in the regex
        safe_name = re.sub(r'[<>:"|\?\*]', '_', safe_name)

        # Ensure no double dots
        safe_name = safe_name.replace('..', '_')

        return safe_name if safe_name else 'unnamed'

    @classmethod
    def validate_content_selections(cls,
                                   content_selections: dict[str, list[str]]) -> tuple[bool, Optional[str]]:
        """
        Validate content selection dictionary.

        Args:
            content_selections: Dictionary of content types to IDs

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content_selections:
            return False, "No content selected for export"

        # Validate structure
        if not isinstance(content_selections, dict):
            return False, "Invalid content selections format"

        # Check for valid content types and IDs
        # Must match ContentType enum values from chatbook_models.py
        valid_types = {'conversation', 'note', 'character', 'world_book',
                      'dictionary', 'generated_document', 'media', 'embedding',
                      'prompt', 'evaluation', 'explainer_session'}

        for content_type, ids in content_selections.items():
            if content_type not in valid_types:
                return False, f"Invalid content type: {content_type}"

            if not isinstance(ids, list):
                return False, f"IDs for {content_type} must be a list"

            # Validate each ID
            for item_id in ids:
                if not cls._is_valid_id(item_id):
                    return False, f"Invalid ID format: {item_id}"

        return True, None

    @classmethod
    def _is_path_traversal(cls, path: str) -> bool:
        """Check if path contains traversal attempts."""
        # Normalize path
        normalized = os.path.normpath(path)

        # Check for absolute paths (Unix style)
        if os.path.isabs(normalized):
            return True

        # Check for Windows absolute paths (C:\ or \\server\)
        if re.match(r'^[A-Za-z]:[/\\]', path) or path.startswith('\\\\'):
            return True

        # Check for parent directory references
        if '..' in normalized:
            return True

        # Check for home directory reference
        if normalized.startswith('~'):
            return True

        # Check against pattern
        if cls.PATH_TRAVERSAL_PATTERN.search(path):
            return True

        # Check for dangerous directory names
        parts = Path(normalized).parts
        return bool(any(part in cls.DANGEROUS_PATHS for part in parts))

    @classmethod
    def _is_dangerous_file(cls, filename: str) -> bool:
        """Check if file type is potentially dangerous."""
        dangerous_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.app',  # Executables
            '.sh', '.bat', '.cmd', '.ps1', '.vbs',     # Scripts
            '.com', '.scr', '.msi', '.jar',            # More executables
            '.lnk', '.url', '.website',                # Shortcuts
            '.reg', '.inf',                             # System files
        }

        extension = Path(filename).suffix.lower()
        return extension in dangerous_extensions

    @classmethod
    def _is_safe_text(cls, text: str) -> bool:
        """Check if text contains only safe characters.

        Excludes potentially dangerous characters:
        - < > : XSS risk in HTML contexts
        - | : Shell pipe injection risk
        - ; : Command separator injection risk
        - ` : Command substitution risk
        - ~ : Home directory expansion risk
        - # : SQL/URL comment injection risk
        - & : Shell command chaining, URL parameter
        - * : Glob/wildcard character
        - ^ : Regex special character
        """
        # Allow alphanumeric, spaces, and safe punctuation only
        # Explicitly excludes: < > | ; ` ~ / : # & * ^ = + (dangerous in various contexts)
        safe_pattern = re.compile(r'^[\w\s\-.,!?\'"()\[\]{}@]+$')
        return bool(safe_pattern.match(text))

    @classmethod
    def _is_valid_id(cls, item_id: str) -> bool:
        """Check if ID has valid format."""
        if not item_id:
            return False

        # Allow alphanumeric, hyphens, underscores (common ID formats)
        # Also allow UUIDs
        id_pattern = re.compile(r'^[\w\-]+$')
        return bool(id_pattern.match(str(item_id)))

    @classmethod
    def validate_job_id(cls, job_id: str) -> bool:
        """
        Validate job ID format (should be UUID).

        Args:
            job_id: Job ID to validate

        Returns:
            True if valid UUID format
        """
        uuid_pattern = re.compile(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            re.IGNORECASE
        )
        return bool(uuid_pattern.match(job_id))
