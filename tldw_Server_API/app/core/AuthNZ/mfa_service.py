# mfa_service.py
# Description: Multi-Factor Authentication service with TOTP support
#
# Imports
import base64
import hashlib
import hmac
import json
import secrets
import string
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Optional

#
# 3rd-party imports
import pyotp
from cryptography.fernet import Fernet
from loguru import logger

try:
    import qrcode
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    qrcode = None  # type: ignore[assignment]
    _FALLBACK_QR_PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
else:
    _FALLBACK_QR_PNG = None
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.crypto_utils import derive_hmac_key_candidates
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.repos.mfa_repo import AuthnzMfaRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

_DB_POOL_NOT_INITIALIZED = "MFAService database pool not initialized"

#######################################################################################################################
#
# MFA Service Class
#

class MFAService:
    """
    Multi-Factor Authentication service supporting TOTP (Time-based One-Time Passwords)

    Features:
    - TOTP generation and validation
    - QR code generation for authenticator apps
    - Backup codes generation and validation
    - Recovery options
    """

    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None,
        repo: Optional[AuthnzMfaRepo] = None,
    ):
        """Initialize MFA service"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self._initialized = False
        self._repo: Optional[AuthnzMfaRepo] = repo
        self._cipher: Optional[Fernet] = None
        self._cipher_candidates: list[Fernet] = []
        self._cipher_key_material: tuple[bytes, ...] = ()

        # TOTP configuration
        self.issuer_name = self.settings.APP_NAME if hasattr(self.settings, 'APP_NAME') else "TLDW Server"
        self.totp_digits = 6
        self.totp_interval = 30  # seconds
        self.backup_codes_count = 8

        # Window for TOTP validation (allows for time drift)
        self.validation_window = 1  # Allow 1 interval before/after

    async def initialize(self):
        """Initialize MFA service"""
        if self._initialized:
            return

        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()

        self._initialized = True
        logger.info("MFAService initialized")

    def _ensure_db_pool(self) -> DatabasePool:
        """Ensure database pool is initialized, raising DatabaseError if not."""
        if not self.db_pool:
            raise DatabaseError(_DB_POOL_NOT_INITIALIZED)
        return self.db_pool

    def _get_repo(self) -> AuthnzMfaRepo:
        """
        Return the MFA repository, constructing it lazily when needed.

        This allows reusing a single repo instance and makes it easy to
        inject a fake or preconfigured repo in tests.
        """
        if self._repo is not None:
            return self._repo
        self._repo = AuthnzMfaRepo(self._ensure_db_pool())
        return self._repo

    def supports_backend(self) -> bool:
        """
        Return True when MFA storage is supported by the current DB backend.

        MFA persistence is implemented through ``AuthnzMfaRepo``, which supports
        both PostgreSQL and SQLite-backed ``DatabasePool`` instances once the
        pool has been initialized. Callers use this to decide whether MFA
        endpoints may be exposed for the active AuthNZ backend.
        """
        return self.db_pool is not None

    def _ensure_cipher_candidates(self) -> list[Fernet]:
        """Return Fernet instances for all active/legacy key materials."""
        key_candidates = tuple(derive_hmac_key_candidates(self.settings))
        if not key_candidates:
            raise ValueError("No HMAC key material available for MFA encryption")
        if self._cipher_key_material != key_candidates:
            cipher_list: list[Fernet] = []
            for material in key_candidates:
                cipher_list.append(Fernet(base64.urlsafe_b64encode(material)))
            self._cipher_candidates = cipher_list
            self._cipher_key_material = key_candidates
            self._cipher = cipher_list[0]
        return self._cipher_candidates

    def _get_cipher(self) -> Fernet:
        """Return the primary Fernet cipher (current key material)."""
        self._ensure_cipher_candidates()
        if self._cipher is None:
            raise ValueError("Primary cipher failed to initialize")
        return self._cipher

    def _encrypt_secret(self, secret: str) -> str:
        """Encrypt a TOTP secret for storage."""
        cipher = self._get_cipher()
        return cipher.encrypt(secret.encode("utf-8")).decode("utf-8")

    def _decrypt_secret(self, encrypted_secret: Optional[str]) -> Optional[str]:
        """Decrypt a stored TOTP secret."""
        if not encrypted_secret:
            return None
        last_exc: Optional[Exception] = None
        for idx, cipher in enumerate(self._ensure_cipher_candidates()):
            try:
                decrypted = cipher.decrypt(encrypted_secret.encode("utf-8"))
                return decrypted.decode("utf-8")
            except Exception as exc:
                last_exc = exc
                logger.debug(f"Failed to decrypt MFA secret with cipher index {idx}: {exc}")
        logger.error("Failed to decrypt MFA secret with available key material")
        raise DatabaseError("Failed to decrypt MFA secret") from last_exc

    def _normalize_backup_code(self, code: str) -> str:
        """Normalize backup codes for hashing/comparison."""
        return code.replace("-", "").replace(" ", "").strip().upper()

    def _hash_backup_code(self, user_id: int, code: str) -> str:
        """Hash backup codes using per-installation secret material."""
        candidates = self._hash_backup_code_candidates(user_id, code)
        if not candidates:
            raise ValueError("No HMAC key material available for backup codes")
        return candidates[0]

    def _hash_backup_code_candidates(self, user_id: int, code: str) -> list[str]:
        """Return hash candidates for a backup code across current/legacy keys."""
        digests: list[str] = []
        normalized = self._normalize_backup_code(code)
        message = f"{user_id}:{normalized}".encode()
        for key in derive_hmac_key_candidates(self.settings):
            digest = hmac.new(key, message, hashlib.sha256).hexdigest()
            if digest not in digests:
                digests.append(digest)
        return digests

    def generate_secret(self) -> str:
        """
        Generate a new TOTP secret

        Returns:
            Base32-encoded secret key
        """
        # Generate 20 bytes (160 bits) of random data
        random_bytes = secrets.token_bytes(20)
        # Encode as base32 for TOTP compatibility
        secret = base64.b32encode(random_bytes).decode('utf-8')
        return secret

    def generate_totp_uri(
        self,
        secret: str,
        username: str,
        issuer: Optional[str] = None
    ) -> str:
        """
        Generate TOTP URI for QR code

        Args:
            secret: Base32-encoded secret
            username: User's username/email
            issuer: Application name

        Returns:
            TOTP URI string
        """
        issuer = issuer or self.issuer_name
        totp = pyotp.TOTP(secret, issuer=issuer)
        return totp.provisioning_uri(
            name=username,
            issuer_name=issuer
        )

    def generate_qr_code(self, totp_uri: str) -> bytes:
        """
        Generate QR code image for TOTP URI

        Args:
            totp_uri: TOTP URI string

        Returns:
            PNG image bytes
        """
        if qrcode is None:  # pragma: no cover - optional dependency path
            logger.warning(
                "qrcode library unavailable; returning fallback placeholder QR image. "
                "Install the 'qrcode' extra for generated QR codes."
            )
            return _FALLBACK_QR_PNG or b""

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()

    def generate_backup_codes(self, count: Optional[int] = None) -> list[str]:
        """
        Generate backup codes for account recovery

        Args:
            count: Number of codes to generate

        Returns:
            List of backup codes
        """
        count = count or self.backup_codes_count
        codes = []

        for _ in range(count):
            # Generate 8-character alphanumeric codes
            code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(8))
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            codes.append(formatted_code)

        return codes

    def verify_totp(
        self,
        secret: str,
        token: str,
        window: Optional[int] = None
    ) -> bool:
        """
        Verify a TOTP token

        Args:
            secret: User's TOTP secret
            token: 6-digit token to verify
            window: Validation window (intervals before/after)

        Returns:
            True if token is valid
        """
        if not secret or not token:
            return False

        # Remove any spaces or hyphens from token
        token = token.replace(' ', '').replace('-', '')

        # Validate token format
        if not token.isdigit() or len(token) != self.totp_digits:
            return False

        try:
            totp = pyotp.TOTP(secret)
            # Use custom window or default
            validation_window = window if window is not None else self.validation_window

            # Verify with time window to account for clock drift
            return totp.verify(token, valid_window=validation_window)

        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False

    async def enable_mfa(
        self,
        user_id: int,
        secret: str,
        backup_codes: list[str]
    ) -> bool:
        """
        Enable MFA for a user

        Args:
            user_id: User's ID
            secret: TOTP secret
            backup_codes: List of backup codes

        Returns:
            True if successfully enabled
        """
        if not self._initialized:
            await self.initialize()

        try:
            encrypted_secret = self._encrypt_secret(secret)
            hashed_codes = [self._hash_backup_code(user_id, code) for code in backup_codes]
            backup_codes_json = json.dumps(hashed_codes)

            repo = self._get_repo()
            await repo.set_mfa_config(
                user_id=user_id,
                encrypted_secret=encrypted_secret,
                backup_codes_json=backup_codes_json,
                updated_at=datetime.now(timezone.utc),
            )

            logger.info(f"MFA enabled for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to enable MFA: {e}")
            return False

    async def disable_mfa(self, user_id: int) -> bool:
        """
        Disable MFA for a user

        Args:
            user_id: User's ID

        Returns:
            True if successfully disabled
        """
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            await repo.clear_mfa_config(
                user_id=user_id,
                updated_at=datetime.now(timezone.utc),
            )

            logger.info(f"MFA disabled for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to disable MFA: {e}")
            return False

    async def get_user_totp_secret(self, user_id: int) -> Optional[str]:
        """Return decrypted TOTP secret for a user if configured."""
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            encrypted = await repo.get_encrypted_totp_secret(user_id)
            return self._decrypt_secret(encrypted)
        except DatabaseError:
            raise
        except Exception as exc:
            if self.settings.PII_REDACT_LOGS:
                logger.error("Failed to load MFA secret (details redacted)")
            else:
                logger.error(f"Failed to load MFA secret for user {user_id}: {exc}")
            raise DatabaseError("Failed to load MFA secret") from exc

    async def get_user_mfa_status(self, user_id: int) -> dict[str, Any]:
        """
        Get MFA status for a user

        Args:
            user_id: User's ID

        Returns:
            Dictionary with MFA status information
        """
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()
            row = await repo.get_mfa_status_row(user_id)

            if row:
                enabled_raw = row.get("two_factor_enabled")
                has_secret_raw = row.get("has_secret")
                has_backup_raw = row.get("has_backup_codes")
                enabled = bool(enabled_raw)
                return {
                    "enabled": enabled,
                    "has_secret": bool(has_secret_raw),
                    "has_backup_codes": bool(has_backup_raw),
                    "method": "totp" if enabled else None,
                }

        except Exception as e:  # noqa: BLE001 - fail safe fallback for dashboard views
            logger.error(f"Failed to get MFA status: {e}")
            # This method is used for read-only status introspection. On
            # unexpected failures we intentionally fall back to a "MFA
            # disabled" snapshot instead of propagating the error, so that
            # dashboard/profile views are not blocked by transient MFA
            # backend issues.

        return {
            "enabled": False,
            "has_secret": False,
            "has_backup_codes": False,
            "method": None
        }

    async def verify_backup_code(
        self,
        user_id: int,
        code: str
    ) -> bool:
        """
        Verify and consume a backup code

        Args:
            user_id: User's ID
            code: Backup code to verify

        Returns:
            True if code is valid and was consumed
        """
        if not self._initialized:
            await self.initialize()

        try:
            repo = self._get_repo()

            # Get backup codes JSON
            backup_codes_json = await repo.get_backup_codes_json(user_id)

            if not backup_codes_json:
                return False

            backup_codes = json.loads(backup_codes_json)
            hash_candidates = self._hash_backup_code_candidates(user_id, code)
            normalized_input = self._normalize_backup_code(code)

            matched = False
            for digest in hash_candidates:
                if digest in backup_codes:
                    backup_codes.remove(digest)
                    matched = True
                    break
            if not matched:
                # Fallback for backward compatibility: older deployments stored
                # backup codes as plain-text strings. Compare normalized values
                # here and then re-save remaining codes as hashes below. This
                # path can be removed once all users have migrated.
                # TODO: Track migration progress and remove this fallback in a future release.
                for candidate in list(backup_codes):
                    if not isinstance(candidate, str):
                        continue
                    if self._normalize_backup_code(candidate) == normalized_input:
                        backup_codes.remove(candidate)
                        matched = True
                        logger.debug(f"User {user_id} used legacy plain-text backup code")
                        break
            if not matched:
                return False

            # Ensure remaining codes are stored as hashed values
            normalized_codes: list[str] = []
            for candidate in backup_codes:
                if not isinstance(candidate, str):
                    continue
                if len(candidate) == 64 and all(ch in string.hexdigits for ch in candidate):
                    normalized_codes.append(candidate.lower())
                else:
                    normalized_codes.append(self._hash_backup_code(user_id, candidate))

            updated_codes_json = json.dumps(normalized_codes)

            # Atomically persist updated codes only if the underlying value has not
            # changed since we loaded it, so the same backup code cannot be consumed
            # twice under concurrent requests.
            consumed = await repo.consume_backup_codes_json(
                user_id=user_id,
                expected_backup_codes_json=backup_codes_json,
                updated_backup_codes_json=updated_codes_json,
            )
            if not consumed:
                logger.info(
                    f"Backup code for user {user_id} could not be consumed (already used or refreshed)."
                )
                return False
        except Exception as e:  # noqa: BLE001 - fail closed on any unexpected error in auth path
            logger.error(f"Failed to verify backup code: {e}")
            return False
        else:
            logger.info(f"Backup code used for user {user_id}")
            return True

    async def regenerate_backup_codes(
        self,
        user_id: int
    ) -> Optional[list[str]]:
        """
        Generate new backup codes for a user

        Args:
            user_id: User's ID

        Returns:
            List of new backup codes or None on failure
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate new codes
            new_codes = self.generate_backup_codes()
            hashed_codes = [self._hash_backup_code(user_id, code) for code in new_codes]
            backup_codes_json = json.dumps(hashed_codes)

            repo = self._get_repo()
            await repo.set_backup_codes_with_timestamp(
                user_id=user_id,
                backup_codes_json=backup_codes_json,
                updated_at=datetime.now(timezone.utc),
            )

            logger.info(f"Regenerated backup codes for user {user_id}")
            return new_codes

        except Exception as e:
            logger.error(f"Failed to regenerate backup codes: {e}")
            return None


#######################################################################################################################
#
# Module Functions for convenience
#

# Global instance
_mfa_service: Optional[MFAService] = None


def get_mfa_service() -> MFAService:
    """Get MFA service singleton instance"""
    global _mfa_service
    if not _mfa_service:
        _mfa_service = MFAService()
    return _mfa_service


def generate_totp_secret() -> str:
    """Generate a new TOTP secret"""
    return get_mfa_service().generate_secret()


def verify_totp_token(secret: str, token: str) -> bool:
    """Verify a TOTP token"""
    return get_mfa_service().verify_totp(secret, token)


#
# End of mfa_service.py
#######################################################################################################################
