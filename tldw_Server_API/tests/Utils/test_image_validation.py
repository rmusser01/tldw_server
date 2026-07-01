# test_image_validation.py
# Unit tests for image validation utilities

import base64
import pytest
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.core.Utils.image_validation import (
    validate_mime_type,
    estimate_decoded_size,
    validate_data_uri,
    safe_decode_base64_image,
    validate_image_url,
    ALLOWED_IMAGE_MIME_TYPES,
    MAX_BASE64_BYTES,
    MAX_BASE64_STRING_LENGTH,
)

PNG_BASE64_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
BROKEN_PNG_BASE64_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="


class TestValidateMimeType:
    """Test MIME type validation."""

    def test_allowed_mime_types(self):
        """Test that allowed MIME types pass validation."""
        assert validate_mime_type("image/png") is True
        assert validate_mime_type("image/jpeg") is True
        assert validate_mime_type("image/webp") is True

    def test_case_insensitive(self):
        """Test that MIME type validation is case-insensitive."""
        assert validate_mime_type("IMAGE/PNG") is True
        assert validate_mime_type("Image/Jpeg") is True
        assert validate_mime_type("IMAGE/WEBP") is True

    def test_disallowed_mime_types(self):
        """Test that disallowed MIME types fail validation."""
        assert validate_mime_type("image/gif") is False
        assert validate_mime_type("image/bmp") is False
        assert validate_mime_type("image/svg+xml") is False
        assert validate_mime_type("text/plain") is False
        assert validate_mime_type("application/pdf") is False

    def test_invalid_mime_types(self):
        """Test that invalid MIME types fail validation."""
        assert validate_mime_type("") is False
        assert validate_mime_type("not-a-mime-type") is False
        assert validate_mime_type("image/") is False


class TestEstimateDecodedSize:
    """Test base64 decoded size estimation."""

    def test_exact_sizes(self):
        """Test size estimation for known base64 strings."""
        # 4 base64 chars = 3 bytes
        assert estimate_decoded_size("AAAA") == 3
        assert estimate_decoded_size("AAAAAAAA") == 6
        assert estimate_decoded_size("AAAAAAAAAAAA") == 9

    def test_with_padding(self):
        """Test size estimation ignores padding."""
        assert estimate_decoded_size("AA==") == 1
        assert estimate_decoded_size("AAA=") == 2
        assert estimate_decoded_size("AAAA") == 3

    def test_empty_string(self):
        """Test size estimation for empty string."""
        assert estimate_decoded_size("") == 0

    def test_large_string(self):
        """Test size estimation for large strings."""
        # 1000 chars without padding
        base64_str = "A" * 1000
        estimated = estimate_decoded_size(base64_str)
        assert estimated == 750  # 1000 * 3 / 4


class TestValidateDataUri:
    """Test data URI validation."""

    def test_valid_png_data_uri(self):
        """Test validation of valid PNG data URI."""
        # Small valid PNG base64
        data_uri = f"data:image/png;base64,{PNG_BASE64_1X1}"

        is_valid, mime_type, base64_data = validate_data_uri(data_uri)
        assert is_valid is True
        assert mime_type == "image/png"
        assert base64_data == PNG_BASE64_1X1

    def test_valid_jpeg_data_uri(self):
        """Test validation of valid JPEG data URI."""
        jpeg_base64 = "SGVsbG8gV29ybGQ="  # Small test data
        data_uri = f"data:image/jpeg;base64,{jpeg_base64}"

        is_valid, mime_type, base64_data = validate_data_uri(data_uri)
        assert is_valid is True
        assert mime_type == "image/jpeg"
        assert base64_data == jpeg_base64

    def test_invalid_mime_type(self):
        """Test rejection of disallowed MIME type."""
        data_uri = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

        is_valid, mime_type, base64_data = validate_data_uri(data_uri)
        assert is_valid is False
        assert mime_type == "image/gif"
        assert base64_data is None

    def test_missing_data_prefix(self):
        """Test rejection of URI without data: prefix."""
        uri = "image/png;base64,SGVsbG8="

        is_valid, mime_type, base64_data = validate_data_uri(uri)
        assert is_valid is False
        assert mime_type is None
        assert base64_data is None

    def test_invalid_format(self):
        """Test rejection of malformed data URI."""
        # Missing base64 indicator
        uri = "data:image/png,SGVsbG8="

        is_valid, mime_type, base64_data = validate_data_uri(uri)
        assert is_valid is False
        assert mime_type is None
        assert base64_data is None

    def test_oversized_base64(self):
        """Test rejection of oversized base64 data."""
        # Create oversized base64 string
        oversized_base64 = "A" * (MAX_BASE64_STRING_LENGTH + 100)
        data_uri = f"data:image/png;base64,{oversized_base64}"

        is_valid, mime_type, base64_data = validate_data_uri(data_uri)
        assert is_valid is False
        assert mime_type == "image/png"
        assert base64_data is None

    def test_estimated_size_too_large(self):
        """Test rejection when estimated decoded size is too large."""
        # Create base64 that would decode to > MAX_BASE64_BYTES
        large_base64 = "A" * int((MAX_BASE64_BYTES + 1000) * 4 / 3)
        data_uri = f"data:image/png;base64,{large_base64}"

        is_valid, mime_type, base64_data = validate_data_uri(data_uri)
        assert is_valid is False


class TestSafeDecodeBase64Image:
    """Test safe base64 image decoding."""

    def test_valid_base64_decode(self):
        """Test successful decoding of valid image base64."""
        test_data = base64.b64decode(PNG_BASE64_1X1)

        decoded = safe_decode_base64_image(PNG_BASE64_1X1, "image/png")
        assert decoded == test_data

    def test_rejects_non_image_payload_for_allowed_mime_type(self):
        """Valid base64 bytes must still be a parseable image."""
        base64_data = base64.b64encode(b"Hello, World!").decode('utf-8')

        decoded = safe_decode_base64_image(base64_data, "image/png")

        assert decoded is None

    def test_rejects_broken_image_payload_without_raising(self):
        """Corrupt image bytes should fail validation instead of escaping parser errors."""
        decoded = safe_decode_base64_image(BROKEN_PNG_BASE64_1X1, "image/png")

        assert decoded is None

    def test_rejects_image_payload_that_does_not_match_declared_mime_type(self):
        """Declared MIME type must match the decoded image format."""
        decoded = safe_decode_base64_image(PNG_BASE64_1X1, "image/jpeg")

        assert decoded is None

    def test_invalid_mime_type_decode(self):
        """Test that invalid MIME type prevents decoding."""
        base64_data = base64.b64encode(b"test").decode('utf-8')

        decoded = safe_decode_base64_image(base64_data, "image/gif")
        assert decoded is None

    def test_invalid_base64(self):
        """Test handling of invalid base64 data."""
        invalid_base64 = "This is not valid base64!"

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            decoded = safe_decode_base64_image(invalid_base64, "image/png")
            assert decoded is None
            mock_logger.warning.assert_called()

    def test_oversized_decoded_data(self):
        """Test rejection of decoded data that's too large."""
        # Create data that's just over the limit
        large_data = b"A" * (MAX_BASE64_BYTES + 1)
        base64_data = base64.b64encode(large_data).decode('utf-8')

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            decoded = safe_decode_base64_image(base64_data, "image/png")
            assert decoded is None
            mock_logger.warning.assert_called()

    def test_valid_size_data(self):
        """Test acceptance of valid image data within size limits."""
        valid_data = base64.b64decode(PNG_BASE64_1X1)

        decoded = safe_decode_base64_image(PNG_BASE64_1X1, "image/png")
        assert decoded == valid_data

    def test_base64_with_validation_flag(self):
        """Test decoding with strict validation."""
        # Create base64 with incorrect padding
        bad_base64 = "SGVsbG8gV29ybGQ"  # Missing padding

        # Should fail with validate=True (default)
        decoded = safe_decode_base64_image(bad_base64, "image/png")
        assert decoded is None


class TestValidateImageUrl:
    """Test image URL validation."""

    def test_valid_data_uri(self):
        """Test validation of valid data URI."""
        test_data = base64.b64decode(PNG_BASE64_1X1)
        data_uri = f"data:image/png;base64,{PNG_BASE64_1X1}"

        is_valid, mime_type, decoded_bytes = validate_image_url(data_uri)
        assert is_valid is True
        assert mime_type == "image/png"
        assert decoded_bytes == test_data

    def test_invalid_data_uri(self):
        """Test rejection of invalid data URI."""
        data_uri = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///"

        is_valid, mime_type, decoded_bytes = validate_image_url(data_uri)
        assert is_valid is False
        assert mime_type == "image/gif"
        assert decoded_bytes is None

    def test_http_url_not_supported(self):
        """Test that HTTP URLs are not supported."""
        url = "http://example.com/image.png"

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            is_valid, mime_type, decoded_bytes = validate_image_url(url)
            assert is_valid is False
            assert mime_type is None
            assert decoded_bytes is None
            mock_logger.warning.assert_called()

    def test_https_url_not_supported(self):
        """Test that HTTPS URLs are not supported."""
        url = "https://example.com/image.png"

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            is_valid, mime_type, decoded_bytes = validate_image_url(url)
            assert is_valid is False
            assert mime_type is None
            assert decoded_bytes is None
            mock_logger.warning.assert_called()

    def test_external_url_logs_without_sensitive_query_values(self):
        """Rejected external URLs should not leak query strings into logs."""
        url = "https://example.com/image.png?token=secret-value"

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            is_valid, mime_type, decoded_bytes = validate_image_url(url)

        assert is_valid is False
        assert mime_type is None
        assert decoded_bytes is None
        logged = " ".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_logger.warning.call_args_list
        )
        assert "secret-value" not in logged
        assert "token=" not in logged

    def test_external_url_logs_without_userinfo_credentials(self):
        """Rejected external URLs should not leak username/password userinfo."""
        url = "https://user:secret-pass@example.com:8443/image.png"

        with patch('tldw_Server_API.app.core.Utils.image_validation.logger') as mock_logger:
            is_valid, mime_type, decoded_bytes = validate_image_url(url)

        assert is_valid is False
        assert mime_type is None
        assert decoded_bytes is None
        logged = " ".join(
            " ".join(str(arg) for arg in call.args)
            for call in mock_logger.warning.call_args_list
        )
        assert "user" not in logged
        assert "secret-pass" not in logged
        assert "@" not in logged
        assert "example.com:8443" in logged  # lgtm[py/incomplete-url-substring-sanitization] literal log-sanitization assertion

    def test_file_url_not_supported(self):
        """Test that file URLs are not supported."""
        url = "file:///path/to/image.png"

        is_valid, mime_type, decoded_bytes = validate_image_url(url)
        assert is_valid is False
        assert mime_type is None
        assert decoded_bytes is None

    def test_empty_url(self):
        """Test handling of empty URL."""
        is_valid, mime_type, decoded_bytes = validate_image_url("")
        assert is_valid is False
        assert mime_type is None
        assert decoded_bytes is None


class TestSecurityFeatures:
    """Test security features of image validation."""

    def test_prevents_dos_with_large_base64(self):
        """Test prevention of DoS via large base64 strings."""
        # Attempt to create a massive base64 string
        huge_base64 = "A" * (MAX_BASE64_STRING_LENGTH * 2)
        data_uri = f"data:image/png;base64,{huge_base64}"

        # Should reject before attempting decode
        is_valid, _, _ = validate_data_uri(data_uri)
        assert is_valid is False

    def test_size_check_before_decode(self):
        """Test that size is checked before decoding."""
        # Create base64 that would be expensive to decode
        large_size = int((MAX_BASE64_BYTES + 1000000) * 4 / 3)
        large_base64 = "A" * large_size
        data_uri = f"data:image/png;base64,{large_base64}"

        # Should reject based on estimated size without decoding
        with patch('base64.b64decode') as mock_decode:
            is_valid, _, _ = validate_data_uri(data_uri)
            assert is_valid is False
            # b64decode should never be called
            mock_decode.assert_not_called()

    def test_mime_type_whitelist_enforced(self):
        """Test that only whitelisted MIME types are allowed."""
        dangerous_types = [
            "application/javascript",
            "text/html",
            "application/x-shockwave-flash",
            "application/pdf",
            "image/svg+xml",  # Can contain scripts
        ]

        for mime_type in dangerous_types:
            data_uri = f"data:{mime_type};base64,SGVsbG8="
            is_valid, returned_mime, _ = validate_data_uri(data_uri)
            assert is_valid is False
            assert returned_mime == mime_type or returned_mime is None


class TestIntegration:
    """Integration tests for image validation workflow."""

    def test_full_validation_workflow_success(self):
        """Test successful validation of a complete data URI."""
        image_data = base64.b64decode(PNG_BASE64_1X1)
        data_uri = f"data:image/png;base64,{PNG_BASE64_1X1}"

        # Full validation workflow
        is_valid, mime_type, decoded = validate_image_url(data_uri)

        assert is_valid is True
        assert mime_type == "image/png"
        assert decoded == image_data

    def test_full_validation_workflow_failure_size(self):
        """Test rejection in full workflow due to size."""
        # Create oversized image
        large_data = b"X" * (MAX_BASE64_BYTES + 1)
        base64_data = base64.b64encode(large_data).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{base64_data}"

        # Should fail validation
        is_valid, mime_type, decoded = validate_image_url(data_uri)

        assert is_valid is False
        assert decoded is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
