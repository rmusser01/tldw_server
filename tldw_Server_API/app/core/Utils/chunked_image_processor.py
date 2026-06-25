# chunked_image_processor.py
# Description: Chunked image processing to prevent memory exhaustion
#
# Imports
import asyncio
import base64
import io
from collections.abc import AsyncIterator
from typing import Optional

from loguru import logger
from tldw_Server_API.app.core.Utils.image_validation import validate_mime_type

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, image processing will be limited")

#######################################################################################################################
#
# Constants:

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_IMAGE_DIMENSION = 4096  # Maximum width or height
MAX_IMAGE_PIXELS = 16777216  # 16 megapixels max

#######################################################################################################################
#
# Functions:

async def process_image_chunked(
    image_data: bytes,
    mime_type: str,
    max_size: Optional[tuple[int, int]] = None
) -> AsyncIterator[bytes]:
    """
    Process image in chunks to avoid loading entire image into memory.

    Args:
        image_data: Raw image bytes
        mime_type: MIME type of the image
        max_size: Optional maximum dimensions (width, height)

    Yields:
        Processed image data chunks
    """
    if not PIL_AVAILABLE:
        # If PIL not available, just yield the original data in chunks
        for i in range(0, len(image_data), CHUNK_SIZE):
            yield image_data[i:i + CHUNK_SIZE]
        return

    try:
        # Load image with PIL for processing
        image = Image.open(io.BytesIO(image_data))
        original_width, original_height = image.width, image.height
        original_pixels = original_width * original_height

        # Check pixel count before any resize so decompression work is bounded by
        # the original payload, not the thumbnail result.
        if original_pixels > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image too large: {original_pixels} pixels")

        # Validate image dimensions
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            # Resize if too large
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
            logger.info(
                f"Resized large image from {original_width}x{original_height} "
                f"to {image.width}x{image.height}"
            )

        # Check pixel count
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(f"Image too large: {image.width * image.height} pixels")

        # Apply max_size if specified
        if max_size:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary (for consistency)
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        # Save to bytes buffer
        output = io.BytesIO()
        format_map = {
            'image/jpeg': 'JPEG',
            'image/png': 'PNG',
            'image/webp': 'WebP'
        }
        save_format = format_map.get(mime_type, 'JPEG')

        # Use streaming save if possible
        image.save(output, format=save_format, optimize=True, quality=85)
        processed_data = output.getvalue()

        # Yield in chunks
        for i in range(0, len(processed_data), CHUNK_SIZE):
            yield processed_data[i:i + CHUNK_SIZE]
            # Small delay to prevent blocking
            await asyncio.sleep(0)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError(f"Failed to process image payload: {e}") from e


async def decode_base64_image_chunked(
    base64_str: str,
    chunk_size: int = CHUNK_SIZE
) -> AsyncIterator[bytes]:
    """
    Decode base64 image data in chunks to avoid memory spikes.

    Args:
        base64_str: Base64 encoded image string
        chunk_size: Size of chunks to process

    Yields:
        Decoded image data chunks
    """
    # Remove data URI prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',', 1)[1]

    # Clean the base64 string
    base64_str = ''.join(base64_str.split())

    # Process in chunks
    # Base64 encoding increases size by ~33%, so adjust chunk size
    b64_chunk_size = int(chunk_size * 1.34)

    remainder = ""
    for i in range(0, len(base64_str), b64_chunk_size):
        chunk = remainder + base64_str[i:i + b64_chunk_size]

        if len(chunk) < 4:
            remainder = chunk
            continue

        decode_len = (len(chunk) // 4) * 4
        to_decode = chunk[:decode_len]
        remainder = chunk[decode_len:]

        if to_decode:
            try:
                # Strict validation: reject invalid characters or incorrect padding
                decoded = base64.b64decode(to_decode, validate=True)
                yield decoded
            except Exception as e:
                # Tightened policy: abort on invalid data rather than silently skipping
                logger.error(f"Invalid base64 data encountered: {e}")
                raise

        # Small delay to prevent blocking
        await asyncio.sleep(0)

    if remainder:
        if len(remainder) == 1:
            logger.error("Invalid base64 data: incomplete trailing quartet")
            raise ValueError("Invalid base64 data: incomplete trailing quartet")
        padding = (4 - (len(remainder) % 4)) % 4
        padded = remainder + ("=" * padding)
        try:
            decoded = base64.b64decode(padded, validate=True)
            if decoded:
                yield decoded
        except Exception as e:
            logger.error(f"Invalid base64 data encountered: {e}")
            raise


async def validate_and_process_image_stream(
    image_stream: AsyncIterator[bytes],
    mime_type: str,
    max_size_bytes: int
) -> tuple[bool, Optional[bytes], str]:
    """
    Validate and process an image stream.

    Args:
        image_stream: Async iterator of image data chunks
        mime_type: MIME type of the image
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        Tuple of (is_valid, processed_data, error_message)
    """
    chunks = []
    total_size = 0

    try:
        async for chunk in image_stream:
            total_size += len(chunk)

            # Check size limit
            if total_size > max_size_bytes:
                return False, None, f"Image exceeds size limit: {total_size} > {max_size_bytes}"

            chunks.append(chunk)

        # Combine chunks
        image_data = b''.join(chunks)

        # Validate image if PIL available
        if PIL_AVAILABLE:
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # Verify it's a valid image
            except Exception as e:
                return False, None, f"Invalid image format: {e}"

        return True, image_data, ""

    except Exception as e:
        return False, None, f"Error processing image stream: {e}"


class StreamingImageProcessor:
    """
    Process images with streaming to minimize memory usage.
    """

    def __init__(self, max_memory_mb: int = 100):
        """
        Initialize the streaming processor.

        Args:
            max_memory_mb: Maximum memory to use for processing
        """
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_memory = 0
        self._lock = asyncio.Lock()

    async def process_image_url(
        self,
        image_url: str,
        max_size_bytes: int
    ) -> tuple[bool, Optional[bytes], Optional[str], str]:
        """
        Process an image from a data URL with streaming.

        Args:
            image_url: Data URL containing the image
            max_size_bytes: Maximum allowed size

        Returns:
            Tuple of (is_valid, image_data, mime_type, error_message)
        """
        if not image_url.startswith('data:'):
            return False, None, None, "Not a data URL"

        try:
            # Parse data URL
            header, base64_data = image_url.split(',', 1)
            mime_type = header.split(';')[0].split(':')[1].strip().lower()
            if not validate_mime_type(mime_type):
                return False, None, mime_type, f"Unsupported image MIME type: {mime_type}"

            # Check if we have memory available (wait politely without holding the lock)
            attempts = 0
            while True:
                async with self._lock:
                    if self.current_memory + max_size_bytes <= self.max_memory:
                        self.current_memory += max_size_bytes
                        break
                    if max_size_bytes > self.max_memory:
                        return False, None, None, "Requested image exceeds memory budget"
                attempts += 1
                if attempts >= 5:
                    return False, None, None, "Insufficient memory for processing"
                await asyncio.sleep(0.2)

            try:
                # Process image in chunks
                chunks = []
                async for chunk in decode_base64_image_chunked(base64_data):
                    chunks.append(chunk)

                image_data = b''.join(chunks)

                # Validate size
                if len(image_data) > max_size_bytes:
                    return False, None, None, f"Image too large: {len(image_data)} bytes"

                # Process image if needed
                if PIL_AVAILABLE:
                    try:
                        # Validate decoded payload is actually a parseable image.
                        with Image.open(io.BytesIO(image_data)) as img:
                            img.verify()
                    except Exception as e:
                        return False, None, mime_type, f"Invalid image format: {e}"

                    processed_chunks = []
                    async for chunk in process_image_chunked(image_data, mime_type):
                        processed_chunks.append(chunk)
                    image_data = b''.join(processed_chunks)

                return True, image_data, mime_type, ""

            finally:
                # Release memory
                async with self._lock:
                    self.current_memory -= max_size_bytes

        except Exception as e:
            logger.error(f"Error processing image URL: {e}")
            return False, None, None, str(e)

    async def batch_process_images(
        self,
        image_urls: list[str],
        max_size_bytes: int,
        max_concurrent: int = 3
    ) -> list[tuple[bool, Optional[bytes], Optional[str], str]]:
        """
        Process multiple images concurrently with memory management.

        Args:
            image_urls: List of image data URLs
            max_size_bytes: Maximum size per image
            max_concurrent: Maximum concurrent processing

        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(url):
            """Process one image URL while respecting the shared concurrency limit."""
            async with semaphore:
                return await self.process_image_url(url, max_size_bytes)

        tasks = [process_with_limit(url) for url in image_urls]
        return await asyncio.gather(*tasks)


# Global processor instance
_image_processor = StreamingImageProcessor()

def get_image_processor() -> StreamingImageProcessor:
    """Get the global image processor instance."""
    return _image_processor
