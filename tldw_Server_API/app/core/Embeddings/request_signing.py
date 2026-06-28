# request_signing.py
# Request signing for security - HMAC signatures for request validation

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Embeddings.audit_adapter import log_security_violation


class RequestSigner:
    """
    Handles request signing and verification using HMAC.
    Ensures requests are authentic and haven't been tampered with.
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = 'sha256',
        max_age_seconds: int = 300,
        require_timestamp: bool = True
    ):
        """
        Initialize request signer.

        Args:
            secret_key: Secret key for signing (generated if not provided)
            algorithm: Hash algorithm to use (sha256, sha512)
            max_age_seconds: Maximum age of valid signatures
            require_timestamp: Whether to require timestamps in signatures
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.max_age_seconds = max_age_seconds
        self.require_timestamp = require_timestamp

        # Get hash function
        self.hash_func = getattr(hashlib, algorithm, hashlib.sha256)

        # Statistics
        self.stats = {
            'signed': 0,
            'verified': 0,
            'failed': 0,
            'expired': 0
        }

        logger.info(f"Request signer initialized with {algorithm} algorithm")

    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key"""
        return base64.b64encode(secrets.token_bytes(32)).decode('utf-8')

    def sign_request(
        self,
        user_id: str,
        request_data: dict[str, Any],
        nonce: Optional[str] = None
    ) -> dict[str, str]:
        """
        Sign a request with HMAC.

        Args:
            user_id: User making the request
            request_data: Request data to sign
            nonce: Optional nonce for replay protection

        Returns:
            Dictionary with signature components
        """
        # Add timestamp
        timestamp = str(int(time.time()))

        # Generate nonce if not provided
        if nonce is None:
            nonce = secrets.token_hex(16)

        # Create signing payload
        payload = self._create_payload(user_id, request_data, timestamp, nonce)

        # Generate signature
        signature = self._generate_signature(payload)

        self.stats['signed'] += 1

        return {
            'signature': signature,
            'timestamp': timestamp,
            'nonce': nonce,
            'algorithm': self.algorithm
        }

    def verify_request(
        self,
        user_id: str,
        request_data: dict[str, Any],
        signature: str,
        timestamp: str,
        nonce: str
    ) -> tuple[bool, Optional[str]]:
        """
        Verify a request signature.

        Args:
            user_id: User who made the request
            request_data: Request data that was signed
            signature: The signature to verify
            timestamp: Request timestamp
            nonce: Request nonce

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check timestamp age
            if self.require_timestamp:
                request_time = int(timestamp)
                current_time = int(time.time())
                age = current_time - request_time

                if age > self.max_age_seconds:
                    self.stats['expired'] += 1
                    log_security_violation(user_id=user_id, action="request_signature_expired", metadata={'age_seconds': age})
                    return False, f"Signature expired (age: {age}s)"

                if age < -60:  # Allow 1 minute clock skew
                    return False, "Request timestamp is in the future"

            # Recreate payload
            payload = self._create_payload(user_id, request_data, timestamp, nonce)

            # Generate expected signature
            expected_signature = self._generate_signature(payload)

            # Compare signatures (constant time)
            is_valid = hmac.compare_digest(signature, expected_signature)

            if is_valid:
                self.stats['verified'] += 1
                return True, None
            else:
                self.stats['failed'] += 1
                log_security_violation(user_id=user_id, action="request_signature_invalid", metadata={'provided_signature_prefix': signature[:10] + '...'})
                return False, "Invalid signature"

        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            self.stats['failed'] += 1
            return False, f"Verification error: {str(e)}"

    def _create_payload(
        self,
        user_id: str,
        request_data: dict[str, Any],
        timestamp: str,
        nonce: str
    ) -> bytes:
        """Create the payload to sign"""
        # Sort request data for consistent ordering
        sorted_data = json.dumps(request_data, sort_keys=True, separators=(',', ':'))

        # Combine all components
        components = [
            user_id,
            sorted_data,
            timestamp,
            nonce
        ]

        # Join with delimiter
        payload_str = '|'.join(components)

        return payload_str.encode('utf-8')

    def _generate_signature(self, payload: bytes) -> str:
        """Generate HMAC signature for payload"""
        h = hmac.new(
            self.secret_key.encode('utf-8'),
            payload,
            self.hash_func
        )
        return base64.b64encode(h.digest()).decode('utf-8')

    def get_statistics(self) -> dict[str, Any]:
        """Get signing statistics"""
        total = self.stats['verified'] + self.stats['failed']
        success_rate = (self.stats['verified'] / total * 100) if total > 0 else 0

        return {
            'total_signed': self.stats['signed'],
            'total_verified': self.stats['verified'],
            'total_failed': self.stats['failed'],
            'total_expired': self.stats['expired'],
            'success_rate': f"{success_rate:.1f}%"
        }


class NonceManager:
    """
    Manages nonces to prevent replay attacks.
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize nonce manager.

        Args:
            ttl_seconds: Time to live for nonces
        """
        self.ttl_seconds = ttl_seconds
        self.used_nonces: dict[str, datetime] = {}
        self.last_cleanup = datetime.utcnow()
        self._lock = threading.Lock()

    def is_valid_nonce(self, nonce: str) -> bool:
        """
        Check if a nonce is valid (not used before) and consume it.
        Backward-compatible alias for consume_nonce().

        Args:
            nonce: Nonce to check

        Returns:
            True if valid (not used), False otherwise
        """
        return self.consume_nonce(nonce)

    def consume_nonce(self, nonce: str) -> bool:
        """
        Atomically validate and consume a nonce.

        Expired nonce entries are treated as reusable immediately, even if
        periodic cleanup has not run yet.
        """
        with self._lock:
            now = datetime.utcnow()

            # Cleanup old nonces periodically (use total_seconds to handle day rollover)
            if (now - self.last_cleanup).total_seconds() > 3600:
                self._cleanup_old_nonces_locked(now)

            # Check if nonce was used and still within TTL
            existing_timestamp = self.used_nonces.get(nonce)
            if existing_timestamp is not None:
                if self._is_nonce_expired(existing_timestamp, now):
                    self.used_nonces.pop(nonce, None)
                else:
                    return False

            # Mark as used
            self.used_nonces[nonce] = now
            return True

    def _cleanup_old_nonces(self):
        """Remove expired nonces"""
        with self._lock:
            self._cleanup_old_nonces_locked(datetime.utcnow())

    def _cleanup_old_nonces_locked(self, now: datetime):
        """Remove expired nonces. Caller must hold self._lock."""
        cutoff = now - timedelta(seconds=self.ttl_seconds)
        self.used_nonces = {
            nonce: timestamp
            for nonce, timestamp in self.used_nonces.items()
            if timestamp > cutoff
        }

        self.last_cleanup = now
        logger.debug(f"Cleaned up nonces, {len(self.used_nonces)} remaining")

    def _is_nonce_expired(self, timestamp: datetime, now: datetime) -> bool:
        """Check whether a nonce timestamp exceeds configured TTL."""
        return (now - timestamp).total_seconds() > self.ttl_seconds


class APIKeyManager:
    """
    Manages API keys for service-to-service authentication.
    """

    def __init__(self, keys_file: Optional[str] = None):
        """
        Initialize API key manager.

        Args:
            keys_file: Path to file containing API keys
        """
        self.keys_file = keys_file
        self.api_keys: dict[str, dict[str, Any]] = {}
        self._load_keys()

    def _load_keys(self):
        """Load API keys from file or generate defaults"""
        if self.keys_file:
            try:
                with open(self.keys_file) as f:
                    data = json.load(f)
                    self.api_keys = data.get('api_keys', {})
                    logger.info(f"Loaded {len(self.api_keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")

        if not self.api_keys:
            raise RuntimeError("Embeddings request signing API key file loaded no keys")

    def generate_api_key(self, name: str) -> str:
        """
        Generate a new API key.

        Args:
            name: Name for the key

        Returns:
            Generated API key
        """
        # Generate secure random key
        key = f"emb_{secrets.token_urlsafe(32)}"

        # Store key metadata
        self.api_keys[key] = {
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "permissions": ["read"],
            "rate_limit": "100/hour",
            "last_used": None,
            "usage_count": 0
        }

        # Save to file if configured
        if self.keys_file:
            self._save_keys()

        logger.info(f"Generated new API key for '{name}'")
        return key

    def validate_api_key(self, api_key: str) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Validate an API key.

        Args:
            api_key: Key to validate

        Returns:
            Tuple of (is_valid, key_metadata)
        """
        if api_key in self.api_keys:
            metadata = self.api_keys[api_key]

            # Update usage stats
            metadata['last_used'] = datetime.utcnow().isoformat()
            metadata['usage_count'] = metadata.get('usage_count', 0) + 1

            return True, metadata

        return False, None

    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.

        Args:
            api_key: Key to revoke

        Returns:
            True if revoked, False if not found
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]

            if self.keys_file:
                self._save_keys()

            logger.info(f"Revoked API key: {api_key[:10]}...")
            return True

        return False

    def _save_keys(self):
        """Save API keys to file"""
        try:
            data = {
                "api_keys": self.api_keys,
                "updated_at": datetime.utcnow().isoformat()
            }

            with open(self.keys_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")


# Global instances
_nonce_manager: Optional[NonceManager] = None
_api_key_manager: Optional[APIKeyManager] = None


def get_request_signer(secret_key: str | None = None) -> RequestSigner:
    """Create a request signer from explicit or environment configuration."""
    resolved_secret = (secret_key or os.getenv("EMBEDDINGS_REQUEST_SIGNING_SECRET") or "").strip()
    if not resolved_secret:
        raise RuntimeError("Embeddings request signing secret is not configured")
    return RequestSigner(secret_key=resolved_secret)


def get_nonce_manager() -> NonceManager:
    """Get or create the global nonce manager."""
    global _nonce_manager
    if _nonce_manager is None:
        _nonce_manager = NonceManager()
    return _nonce_manager


def get_api_key_manager() -> APIKeyManager:
    """Get or create the global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        keys_file = (os.getenv("EMBEDDINGS_REQUEST_SIGNING_KEYS_FILE") or "").strip()
        if not keys_file:
            raise RuntimeError("Embeddings request signing API key file is not configured")
        _api_key_manager = APIKeyManager(keys_file=keys_file)
    return _api_key_manager


# Middleware for request validation
def validate_signed_request(
    user_id: str,
    request_data: dict[str, Any],
    headers: dict[str, str]
) -> tuple[bool, Optional[str]]:
    """
    Validate a signed request from headers.

    Args:
        user_id: User making the request
        request_data: Request payload
        headers: Request headers containing signature

    Returns:
        Tuple of (is_valid, error_message)
    """
    signer = get_request_signer()
    nonce_manager = get_nonce_manager()

    # Extract signature components from headers
    signature = headers.get('X-Signature')
    timestamp = headers.get('X-Timestamp')
    nonce = headers.get('X-Nonce')

    if not all([signature, timestamp, nonce]):
        return False, "Missing signature headers"

    # Verify signature first to avoid consuming nonce on invalid signatures.
    is_valid_signature, error_message = signer.verify_request(
        user_id=user_id,
        request_data=request_data,
        signature=signature,
        timestamp=timestamp,
        nonce=nonce,
    )
    if not is_valid_signature:
        return False, error_message

    # Consume nonce only after signature verification succeeds.
    if not nonce_manager.consume_nonce(nonce):
        log_security_violation(user_id=user_id, action="request_replay_attempt", metadata={'nonce': nonce})
        return False, "Invalid or reused nonce"

    return True, None


# Example usage
def create_signed_headers(
    user_id: str,
    request_data: dict[str, Any]
) -> dict[str, str]:
    """
    Create headers with request signature.

    Args:
        user_id: User making the request
        request_data: Request payload

    Returns:
        Headers dict with signature
    """
    signer = get_request_signer()

    sig_data = signer.sign_request(user_id, request_data)

    return {
        'X-Signature': sig_data['signature'],
        'X-Timestamp': sig_data['timestamp'],
        'X-Nonce': sig_data['nonce'],
        'X-Algorithm': sig_data['algorithm']
    }
