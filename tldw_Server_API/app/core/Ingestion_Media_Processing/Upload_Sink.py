# Upload_Sink.py
# Description: Contains classes and functions to handle file uploads,
#              validate their safety and integrity, and perform sanitization.
#
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Set, Union, Tuple
#
# 3rd-party Libraries
import puremagic
import yara
import zipfile
#
# Local Imports (adjust path as per your project structure)
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.Utils.Utils import logging


# If the above import fails in a different context, fallback to standard logging:
# import logging
# logging.basicConfig(level=logging.INFO)


class FileValidationError(Exception):
    """Custom exception for critical validation/setup errors."""

    def __init__(self, message, issues: Optional[List[str]] = None):
        super().__init__(message)
        self.issues = issues if issues is not None else [message]


class ValidationResult:
    """Holds the outcome of a file validation process."""

    def __init__(self, is_valid: bool, issues: Optional[List[str]] = None,
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

# Default configurations for common media types
# These can be overridden or extended via FileValidator's constructor
DEFAULT_MEDIA_TYPE_CONFIG = {
    "audio": {
        "allowed_extensions": {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'},
        "allowed_mimetypes": {'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/aac', 'audio/ogg',
                              'audio/mp4', 'audio/x-m4a'},
        "max_size_mb": media_config.get('max_audio_file_size_mb', 500),
    },
    "video": {
        "allowed_extensions": {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'},
        "allowed_mimetypes": {'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm',
                              'video/x-flv'},
        "max_size_mb": media_config.get('max_video_file_size_mb', 1000),
    },
    "image": {
        "allowed_extensions": {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'},
        "allowed_mimetypes": {'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/svg+xml'},
        "max_size_mb": 20,
    },
    "document": {  # Generic documents
        "allowed_extensions": {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.txt', '.md', '.rtf',
                               '.csv'},
        "allowed_mimetypes": {
            'application/pdf', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-powerpoint',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/plain', 'text/markdown', 'text/rtf', 'text/csv', 'application/rtf', 'application/x-rtf',
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
    "html": {
        "allowed_extensions": {'.html', '.htm'},
        "allowed_mimetypes": {'text/html'},
        "max_size_mb": 5,
        "sanitize": True,  # Flag to indicate sanitization should be applied
    },
    "xml": {
        "allowed_extensions": {'.xml', '.opml'},
        "allowed_mimetypes": {'text/xml', 'application/xml'},
        "max_size_mb": 10,
        "sanitize": True,  # Flag to indicate sanitization should be applied
    },
    "archive": {
        "allowed_extensions": {'.zip'},  # Example: only zip initially
        "allowed_mimetypes": {'application/zip', 'application/x-zip-compressed'},
        "max_size_mb": media_config.get('max_archive_uncompressed_size_mb', 200),
        "scan_contents": True,  # Flag to indicate archive contents should be scanned
        "max_internal_files": media_config.get('max_archive_internal_files', 100),
        "max_internal_uncompressed_size_mb": media_config.get('max_archive_uncompressed_size_mb', 200),
    },
}

# Mapping from extension to a media_type_key (used by the dispatcher)
# This can be auto-generated or manually maintained for clarity.
EXT_TO_MEDIA_TYPE_KEY = {
    '.mp3': 'audio', '.wav': 'audio', '.flac': 'audio', '.aac': 'audio', '.ogg': 'audio', '.m4a': 'audio',
    '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.mkv': 'video', '.webm': 'video', '.flv': 'video',
    '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image', '.webp': 'image', '.bmp': 'image',
    '.svg': 'image',
    '.pdf': 'pdf',
    '.doc': 'document', '.docx': 'document', '.ppt': 'document', '.pptx': 'document', '.xls': 'document',
    '.xlsx': 'document',
    '.txt': 'document', '.md': 'document', '.rtf': 'document', '.csv': 'document',
    '.epub': 'ebook', '.mobi': 'ebook', '.azw': 'ebook',
    '.html': 'html', '.htm': 'html',
    '.xml': 'xml', '.opml': 'xml',
    '.zip': 'archive',
}


class FileValidator:
    def __init__(self, yara_rules_path: Optional[str] = None, custom_media_configs: Optional[Dict] = None):
        self.magic_available = bool(puremagic)
        self.yara_available = bool(yara)
        self.zipfile_available = bool(zipfile)

        self.compiled_yara_rules = None
        if self.yara_available and yara_rules_path:
            self._initialize_yara_scanner(yara_rules_path)
        elif yara_rules_path and not self.yara_available:
            logging.warning("Yara rules path provided, but yara-python is not installed. Yara scanning disabled.")

        if not self.magic_available:
            logging.warning("python-magic is not installed. MIME type validation will be limited/skipped. "
                            "Install with: pip install python-magic")

        self.media_configs = DEFAULT_MEDIA_TYPE_CONFIG.copy()
        if custom_media_configs:
            for media_type, config_val in custom_media_configs.items():
                if media_type in self.media_configs:
                    self.media_configs[media_type].update(config_val)
                else:
                    self.media_configs[media_type] = config_val

    def _compile_yara_rules(self, rules_path: str):
        if not self.yara_available: return None
        try:
            # Check if the rules file exists
            if not os.path.exists(rules_path):
                logging.warning(f"Yara rules file not found at {rules_path}. Yara scanning will be disabled.")
                return None
            rules = yara.compile(filepath=rules_path)
            logging.info(f"Yara rules compiled successfully from {rules_path}.")
            return rules
        except Exception as e:
            logging.error(f"An unexpected error occurred during Yara rule compilation from {rules_path}: {e}")
        return None

    def _initialize_yara_scanner(self, rules_path: str):
        if not self.yara_available: return
        self.compiled_yara_rules = self._compile_yara_rules(rules_path)
        if self.compiled_yara_rules is None:
            logging.warning("Yara rules not loaded. Yara scanning will be disabled for this validator instance.")

    def _scan_file_with_yara(self, file_path: Path) -> Tuple[bool, List[str]]:
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
        except Exception as e:
            logging.error(f"Unexpected error scanning file {file_path} with Yara: {e}")
            return False, [f"Unexpected Yara scanning error: {e}"]

    def get_media_config(self, media_type_key: Optional[str]) -> Optional[Dict]:
        if not media_type_key: return None
        return self.media_configs.get(media_type_key.lower())

    def validate_file(
            self,
            file_path: Union[str, Path],
            original_filename: Optional[str] = None,
            media_type_key: Optional[str] = None,
            allowed_extensions_override: Optional[Set[str]] = None,
            allowed_mimetypes_override: Optional[Set[str]] = None,
            max_size_mb_override: Optional[Union[int, float]] = None
    ) -> ValidationResult:
        issues: List[str] = []
        current_file_path = Path(file_path)

        # Determine original filename if not provided
        _original_filename = original_filename or current_file_path.name
        # Get extension from the actual file path on disk
        disk_file_ext = current_file_path.suffix.lower()

        # 1. Check existence and if it's a file
        if not current_file_path.exists():
            issues.append(f"File does not exist: {current_file_path}")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)
        if not current_file_path.is_file():
            issues.append(f"Path is not a file: {current_file_path}")
            return ValidationResult(False, issues, current_file_path, detected_extension=disk_file_ext)

        # Determine configuration
        config = self.get_media_config(media_type_key)
        cfg_allowed_extensions = config.get("allowed_extensions") if config else None
        cfg_allowed_mimetypes = config.get("allowed_mimetypes") if config else None
        cfg_max_size_mb = config.get("max_size_mb") if config else None

        final_allowed_extensions = allowed_extensions_override if allowed_extensions_override is not None else cfg_allowed_extensions
        final_allowed_mimetypes = allowed_mimetypes_override if allowed_mimetypes_override is not None else cfg_allowed_mimetypes
        final_max_size_mb = max_size_mb_override if max_size_mb_override is not None else cfg_max_size_mb
        final_max_size_bytes = (final_max_size_mb * 1024 * 1024) if final_max_size_mb is not None else None

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
            issues.append(f"File size {file_size / (1024 * 1024):.2f}MB "
                          f"exceeds limit of {final_max_size_bytes / (1024 * 1024):.2f}MB.")

        # 3. Validate extension (based on original_filename provided by user/client)
        claimed_ext = Path(_original_filename).suffix.lower()
        if final_allowed_extensions:
            if not claimed_ext:
                issues.append(f"Claimed filename '{_original_filename}' has no extension.")
            elif claimed_ext not in final_allowed_extensions:
                issues.append(f"Claimed extension '{claimed_ext}' from '{_original_filename}' is not allowed. "
                              f"Allowed: {final_allowed_extensions}")

        # 4. Validate MIME type
        detected_mime_type: Optional[str] = None
        if self.magic_available:
            try:
                detected_mime_type = puremagic.from_file(str(current_file_path), mime=True)
                if final_allowed_mimetypes:
                    if detected_mime_type not in final_allowed_mimetypes:
                        issues.append(
                            f"Detected MIME type '{detected_mime_type}' for file '{_original_filename}' is not allowed. "
                            f"Allowed: {final_allowed_mimetypes}")
                # TODO: Add more sophisticated MIME vs. Extension consistency check here if needed
                # For example, if claimed_ext is '.jpg', ensure detected_mime_type is 'image/jpeg'.
                # This requires a mapping. For now, relying on allowed_extensions and allowed_mimetypes.
            except Exception as e:  # Catch other errors during MIME detection
                issues.append(f"Unexpected error during MIME type detection for {_original_filename}: {e}")

        elif final_allowed_mimetypes:  # MIME types are restricted, but magic is unavailable
            issues.append(
                "MIME type validation skipped: python-magic not installed, but specific MIME types are required.")

        # 5. Malware Scan (Yara)
        is_safe_yara, yara_match_details = self._scan_file_with_yara(current_file_path)
        if not is_safe_yara:
            issues.append(f"Potential threat detected by Yara in '{_original_filename}'.")
            issues.extend([f"  Yara: {detail}" for detail in yara_match_details])

        if issues:
            logging.warning(f"Validation failed for '{_original_filename}' (path: {current_file_path}): {issues}")
            return ValidationResult(False, issues, current_file_path, detected_mime_type, disk_file_ext)

        logging.info(f"File '{_original_filename}' (path: {current_file_path}) validated successfully. "
                     f"MIME: {detected_mime_type}, Claimed Ext: {claimed_ext}, Disk Ext: {disk_file_ext}")
        return ValidationResult(True, file_path=current_file_path,
                                detected_mime_type=detected_mime_type, detected_extension=disk_file_ext)

    def validate_archive_contents(self, archive_path: Union[str, Path]) -> ValidationResult:
        """Validates an archive file and its contents."""
        archive_path_obj = Path(archive_path)
        issues: List[str] = []

        # Get config for "archive" type
        archive_config = self.get_media_config("archive")
        if not archive_config:
            issues.append("No configuration found for 'archive' type. Cannot validate archive contents.")
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

        if not self.zipfile_available:
            issues.append("zipfile module not available. Cannot scan archive contents.")
            logging.warning("zipfile module not found. Archive content scanning skipped.")
            # Return the validation of the archive file itself, but add this issue.
            archive_file_validation_result.is_valid = False  # Mark as invalid due to inability to scan
            archive_file_validation_result.issues.append("Cannot scan archive contents: zipfile module missing.")
            return archive_file_validation_result

        # Step 2: Securely extract and validate contents
        max_internal_files = archive_config.get("max_internal_files", 100)
        max_uncompressed_size = archive_config.get("max_internal_uncompressed_size_mb", 200) * 1024 * 1024
        extracted_count = 0
        total_extracted_size = 0

        try:
            with tempfile.TemporaryDirectory(prefix="archive_extract_") as extract_dir_str:
                extract_dir = Path(extract_dir_str)
                logging.debug(f"Extracting archive '{archive_path_obj.name}' to '{extract_dir}'")

                if archive_path_obj.suffix.lower() == ".zip":
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
                            if member.is_dir(): continue  # Skip directories

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

                            extracted_count += 1
                            if extracted_count > max_internal_files:  # Double check during extraction
                                issues.append(
                                    f"Exceeded max internal file limit ({max_internal_files}) during extraction.")
                                break  # Stop extraction

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
                                internal_ext = internal_file_path.suffix.lower()
                                internal_media_type_key = EXT_TO_MEDIA_TYPE_KEY.get(internal_ext)

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
                            except Exception as extract_err:
                                issues.append(
                                    f"Error extracting/validating internal file '{member.filename}': {extract_err}")
                                logging.error(f"Error extracting internal file '{member.filename}': {extract_err}",
                                              exc_info=True)
                                break  # Stop on extraction error for an internal file

                # Add handlers for other archive types (tar, etc.) here if needed
                else:
                    issues.append(f"Unsupported archive type for content scanning: {archive_path_obj.suffix}")

                if issues:  # If any issues occurred during extraction or internal validation
                    return ValidationResult(False, issues, archive_path_obj)

        except zipfile.BadZipFile:
            issues.append(f"Archive '{archive_path_obj.name}' is corrupted or not a valid ZIP file.")
        except Exception as e:
            issues.append(f"Error processing archive '{archive_path_obj.name}': {e}")
            logging.error(f"Error processing archive {archive_path_obj.name}: {e}", exc_info=True)

        if issues:
            return ValidationResult(False, issues, archive_path_obj)

        logging.info(f"Archive '{archive_path_obj.name}' and its contents validated successfully.")
        return ValidationResult(True, file_path=archive_path_obj)

    def sanitize_html_content(self, html_content: str, config: Optional[Dict] = None) -> str:
        # Placeholder for HTML sanitization using Bleach or similar
        # from bleach import clean
        # allowed_tags = config.get("allowed_tags", []) if config else []
        # allowed_attributes = config.get("allowed_attributes", {}) if config else {}
        # cleaned_html = clean(html_content, tags=allowed_tags, attributes=allowed_attributes, strip=True)
        # logging.info("HTML content sanitized.")
        # return cleaned_html
        logging.warning("HTML sanitization not yet implemented. Returning original content.")
        return html_content

    def sanitize_xml_content(self, xml_content: str, config: Optional[Dict] = None) -> str:
        # Placeholder for XML sanitization using lxml or similar
        # from lxml import etree
        # parser_config = config.get("xml_parser_options", {"resolve_entities": False, "no_network": True})
        # parser = etree.XMLParser(**parser_config)
        # try:
        #   root = etree.fromstring(xml_content.encode(), parser=parser)
        #   # Further processing to remove unwanted elements/attributes based on config
        #   cleaned_xml = etree.tostring(root).decode()
        #   logging.info("XML content sanitized.")
        #   return cleaned_xml
        # except etree.XMLSyntaxError as e:
        #   raise FileValidationError(f"Invalid XML syntax: {e}")
        logging.warning("XML sanitization not yet implemented. Returning original content.")
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
        file_ext = Path(_original_filename).suffix.lower()
        media_type_key = EXT_TO_MEDIA_TYPE_KEY.get(file_ext)

    if not media_type_key:
        logging.warning(f"Could not determine specific media type for '{_original_filename}' based on extension. "
                        "Performing generic validation.")
        # Perform a generic validation (no specific media_type_key implies less strict rules or default rules)
        return validator.validate_file(p_file_path, _original_filename, media_type_key=None)

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

        # Example of how sanitization could be triggered (content needs to be read and written back)
        # This part is more complex as it involves modifying the file or returning modified content.
        # For now, validate_file doesn't modify. Sanitization would be a separate step.
        # if media_config.get("sanitize"):
        #     if media_type_key == "html":
        #         # content = p_file_path.read_text()
        #         # sanitized_content = validator.sanitize_html_content(content, media_config)
        #         # p_file_path.write_text(sanitized_content) # Overwrites original, be careful
        #         logging.info(f"Sanitization would apply to HTML file: {_original_filename}")
        #     elif media_type_key == "xml":
        #         logging.info(f"Sanitization would apply to XML file: {_original_filename}")

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