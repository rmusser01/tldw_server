import hashlib
import json
from datetime import datetime
from typing import Any, Optional


class ContentMetadataHandler:
    """Handles the addition and parsing of metadata for scraped content."""

    METADATA_START = "[METADATA]"
    METADATA_END = "[/METADATA]"
    _MAX_METADATA_JSON_NESTING = 64

    @staticmethod
    def _metadata_json_nesting_is_safe(metadata_text: str) -> bool:
        """Return whether the leading JSON value stays within the nesting limit."""
        depth = 0
        in_string = False
        escaped = False

        for char in metadata_text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
                if depth > ContentMetadataHandler._MAX_METADATA_JSON_NESTING:
                    return False
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    return True
                if depth < 0:
                    return False

        return False

    @staticmethod
    def format_content_with_metadata(
        url: str, content: str, pipeline: str = "Trafilatura", additional_metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Format content with metadata header.

        Args:
            url: The source URL
            content: The scraped content
            pipeline: The scraping pipeline used
            additional_metadata: Optional dictionary of additional metadata to include

        Returns:
            Formatted content with metadata header
        """
        metadata = {
            "url": url,
            "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "scraping_pipeline": pipeline,
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        formatted_content = f"""{ContentMetadataHandler.METADATA_START}
        {json.dumps(metadata, indent=2)}
        {ContentMetadataHandler.METADATA_END}

        {content}"""

        return formatted_content

    @staticmethod
    def _parse_metadata_envelope(content: str) -> Optional[tuple[dict[str, Any], str]]:
        """Parse a canonical leading metadata envelope and return its body."""
        if not isinstance(content, str):
            return None
        envelope = content.lstrip()
        if not envelope.startswith(ContentMetadataHandler.METADATA_START):
            return None

        metadata_text = envelope[len(ContentMetadataHandler.METADATA_START) :].lstrip()
        if not ContentMetadataHandler._metadata_json_nesting_is_safe(metadata_text):
            return None
        try:
            metadata, metadata_end = json.JSONDecoder().raw_decode(metadata_text)
        except (json.JSONDecodeError, RecursionError):
            return None
        if not isinstance(metadata, dict):
            return None

        remainder = metadata_text[metadata_end:].lstrip()
        if not remainder.startswith(ContentMetadataHandler.METADATA_END):
            return None
        clean_content = remainder[len(ContentMetadataHandler.METADATA_END) :].strip()
        return metadata, clean_content

    @staticmethod
    def extract_metadata(content: str) -> tuple[dict[str, Any], str]:
        """
        Extract metadata and content separately.

        Args:
            content: The full content including metadata

        Returns:
            Tuple of (metadata dict, clean content)
        """
        parsed = ContentMetadataHandler._parse_metadata_envelope(content)
        return parsed if parsed is not None else ({}, content)

    @staticmethod
    def has_metadata(content: str) -> bool:
        """
        Check if content contains metadata.

        Args:
            content: The content to check

        Returns:
            bool: True if metadata is present
        """
        return ContentMetadataHandler._parse_metadata_envelope(content) is not None

    @staticmethod
    def strip_metadata(content: str) -> str:
        """
        Remove metadata from content if present.

        Args:
            content: The content to strip metadata from

        Returns:
            Content without metadata
        """
        parsed = ContentMetadataHandler._parse_metadata_envelope(content)
        return parsed[1] if parsed is not None else content

    @staticmethod
    def get_content_hash(content: str) -> str:
        """
        Get hash of content without metadata.

        Args:
            content: The content to hash

        Returns:
            SHA-256 hash of the clean content
        """
        clean_content = ContentMetadataHandler.strip_metadata(content)
        return hashlib.sha256(clean_content.encode("utf-8")).hexdigest()

    @staticmethod
    def content_changed(old_content: str, new_content: str) -> bool:
        """
        Check if content has changed by comparing hashes.

        Args:
            old_content: Previous version of content
            new_content: New version of content

        Returns:
            bool: True if content has changed
        """
        old_hash = ContentMetadataHandler.get_content_hash(old_content)
        new_hash = ContentMetadataHandler.get_content_hash(new_content)
        return old_hash != new_hash
