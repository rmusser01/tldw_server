# password_service.py
# Description: Password hashing and validation service using Argon2
#
# Imports
import re
import secrets
import string
import threading
from typing import Any, Optional

#
# 3rd-party imports
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import WeakPasswordError

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

#######################################################################################################################
#
# Password Service Class

class PasswordService:
    """Service for password hashing, validation, and strength checking using Argon2"""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize password service with Argon2 hasher"""
        self.settings = settings or get_settings()

        # Initialize Argon2 hasher with configured parameters
        self.hasher = PasswordHasher(
            time_cost=self.settings.ARGON2_TIME_COST,
            memory_cost=self.settings.ARGON2_MEMORY_COST,
            parallelism=self.settings.ARGON2_PARALLELISM,
            hash_len=32,
            salt_len=16
        )

        # Password requirements
        self.min_length = self.settings.PASSWORD_MIN_LENGTH

        # Common weak passwords to check against (based on NCSC/HaveIBeenPwned data)
        # This list includes the most commonly breached passwords
        self.common_passwords = {
            # Top breached passwords
            "password", "123456", "123456789", "12345678", "12345",
            "1234567", "1234567890", "qwerty", "abc123", "password1",
            "password123", "111111", "123123", "admin", "letmein",
            "welcome", "monkey", "dragon", "master", "qwertyuiop",
            "login", "passw0rd", "hello", "iloveyou", "trustno1",
            "sunshine", "princess", "football", "baseball", "shadow",
            "michael", "ashley", "654321", "superman", "qazwsx",
            "000000", "access", "batman", "bailey", "1q2w3e4r",
            "696969", "loveme", "mustang", "password2", "password12",
            # Keyboard patterns
            "qwerty123", "1qaz2wsx", "zxcvbnm", "asdfghjkl", "1q2w3e",
            "qwe123", "asd123", "zxc123", "!qaz2wsx", "1qazxsw2",
            "qweasdzxc", "asdfgh", "zxcvbn", "1234qwer", "qwer1234",
            # Year-based
            "2020", "2021", "2022", "2023", "2024", "2025",
            "pass2020", "pass2021", "pass2022", "pass2023", "pass2024",
            # Common word + number patterns
            "welcome1", "welcome123", "admin123", "admin1", "admin1234",
            "root", "toor", "pass", "pass123", "password1234",
            "test", "test123", "guest", "guest123", "changeme",
            "temp", "temp123", "default", "secret", "secret123",
            # Application-specific weak passwords
            "tldw", "tldw123", "tldwserver", "server", "server123",
            "apikey", "api123", "media", "media123", "video", "video123",
            # Additional common passwords from breach lists
            "jennifer", "hunter", "thomas", "charlie", "andrew",
            "joshua", "jessica", "amanda", "daniel", "matthew",
            "whatever", "freedom", "starwars", "jordan23", "cheese",
            "pepper", "guitar", "yankees", "ranger", "killer",
            "mercedes", "harley", "maverick", "ginger", "corvette",
            "computer", "internet", "database", "network",
        }

        logger.debug(
            f"PasswordService initialized with Argon2 (time_cost={self.settings.ARGON2_TIME_COST}, "
            f"memory_cost={self.settings.ARGON2_MEMORY_COST}KB, parallelism={self.settings.ARGON2_PARALLELISM})"
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a password using Argon2id

        Args:
            password: Plain text password to hash

        Returns:
            Argon2 hash string

        Raises:
            WeakPasswordError: If password doesn't meet requirements
        """
        # Validate password strength first
        self.validate_password_strength(password)

        try:
            # Hash the password
            password_hash = self.hasher.hash(password)
            logger.debug("Password hashed successfully")
            return password_hash

        except Exception as e:
            logger.error(f"Failed to hash password: {e}")
            raise

    def verify_password(self, password: str, password_hash: str) -> tuple[bool, bool]:
        """
        Verify a password against its hash

        Args:
            password: Plain text password to verify
            password_hash: Argon2 hash to verify against

        Returns:
            Tuple of (is_valid, needs_rehash)
            - is_valid: True if password matches
            - needs_rehash: True if hash parameters have changed
        """
        try:
            # Verify the password
            self.hasher.verify(password_hash, password)

            # Check if rehashing is needed (parameters changed)
            needs_rehash = self.hasher.check_needs_rehash(password_hash)

            if needs_rehash:
                logger.info("Password hash needs rehashing due to parameter changes")

            return True, needs_rehash

        except VerifyMismatchError:
            # Password doesn't match
            logger.debug("Password verification failed - mismatch")
            return False, False

        except VerificationError as e:
            # Hash is malformed or corrupted
            logger.error(f"Password verification error - invalid hash: {e}")
            return False, False

        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error during password verification: {e}")
            return False, False

    def validate_password_strength(self, password: str, username: Optional[str] = None) -> None:
        """
        Validate password meets security requirements

        Args:
            password: Password to validate
            username: Username to check password doesn't contain

        Raises:
            WeakPasswordError: If password doesn't meet requirements
        """
        errors = []

        # Check minimum length
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")

        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")

        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")

        # Check for at least one digit
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one number")

        # Check for at least one special character
        # Expanded set includes: ! @ # $ % ^ & * ( ) - _ = + [ ] { } | ; : ' " , . < > / ? ~ `
        if not re.search(r'[!@#$%^&*()\-_=+\[\]{}|;:\'",.<>/?~`\\]', password):
            errors.append("Password must contain at least one special character")

        # Check against common passwords
        if password.lower() in self.common_passwords:
            errors.append("Password is too common, please choose a more unique password")

        # Check password doesn't contain username
        if username and username.lower() in password.lower():
            errors.append("Password must not contain your username")

        # Check for sequential or repeated characters
        if self._has_sequential_chars(password) or self._has_repeated_chars(password):
            errors.append("Password must not contain sequential or repeated characters")

        # If there are any errors, raise exception
        if errors:
            raise WeakPasswordError("; ".join(errors))

    def _has_sequential_chars(self, password: str, max_sequence: int = 3) -> bool:
        """Check if password has sequential characters (e.g., 'abc', '123')"""

        for i in range(len(password) - max_sequence + 1):
            substring = password[i:i + max_sequence]

            # Check if all characters are digits or all are letters
            all_digits = all(c.isdigit() for c in substring)
            all_letters = all(c.isalpha() for c in substring)

            if all_digits or all_letters:
                # Check ascending sequence (e.g., "abc", "123")
                if all(ord(substring[j+1]) - ord(substring[j]) == 1 for j in range(len(substring) - 1)):
                    return True

                # Check descending sequence (e.g., "cba", "321")
                if all(ord(substring[j]) - ord(substring[j+1]) == 1 for j in range(len(substring) - 1)):
                    return True

        return False

    def _has_repeated_chars(self, password: str, max_repeat: int = 3) -> bool:
        """Check if password has repeated characters (e.g., 'aaa', '111')"""
        return any(len(set(password[i:i + max_repeat])) == 1 for i in range(len(password) - max_repeat + 1))

    def generate_secure_password(
        self,
        length: int = 16,
        username: Optional[str] = None,
    ) -> str:
        """
        Generate a cryptographically secure password accepted by the active policy.

        Args:
            length: Requested password length.
            username: Optional username to exclude from the generated password.

        Returns:
            A secure random password accepted by ``validate_password_strength``.

        Raises:
            RuntimeError: If no valid password can be generated within the retry bound.
        """
        length = max(length, 12, self.min_length)

        # Character sets to use
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"

        all_chars = lowercase + uppercase + digits + special
        for _ in range(32):
            # Ensure at least one character from each set.
            password = [
                secrets.choice(lowercase),
                secrets.choice(uppercase),
                secrets.choice(digits),
                secrets.choice(special),
            ]
            password.extend(secrets.choice(all_chars) for _ in range(length - 4))
            secrets.SystemRandom().shuffle(password)
            candidate = "".join(password)

            try:
                self.validate_password_strength(candidate, username)
            except WeakPasswordError:
                continue
            return candidate

        raise RuntimeError(
            "Unable to generate a password that satisfies the configured policy"
        )

    def generate_registration_code(self, length: int = 24) -> str:
        """
        Generate a secure registration code

        Args:
            length: Length of code to generate

        Returns:
            Secure random code string (alphanumeric)
        """
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    @staticmethod
    def _resolve_postgres_backend(
        db_connection: Any,
        *,
        is_postgres: Optional[bool] = None,
    ) -> bool:
        """Resolve backend mode from explicit caller input or known connection hints."""
        if is_postgres is not None:
            return bool(is_postgres)

        sqlite_hint = getattr(db_connection, "_is_sqlite", None)
        if isinstance(sqlite_hint, bool):
            return not sqlite_hint

        # SQLite shims in AuthNZ DatabasePool expose underlying connection as `_c`.
        if getattr(db_connection, "_c", None) is not None:
            return False

        module_name = getattr(type(db_connection), "__module__", "")
        if isinstance(module_name, str) and module_name.startswith("asyncpg"):
            return True

        # Last-resort interface hint for legacy tests/callers.
        return callable(getattr(db_connection, "fetch", None))

    async def check_password_history(
        self,
        user_id: int,
        new_password: str,
        db_connection,
        *,
        is_postgres: Optional[bool] = None,
    ) -> bool:
        """
        Check if password was recently used by the user

        Args:
            user_id: User ID to check history for
            new_password: New password to check
            db_connection: Database connection

        Returns:
            True if password is acceptable (not in history), False otherwise
        """
        try:
            # Get recent password hashes
            history_count = self.settings.PASSWORD_HISTORY_RETENTION_COUNT
            postgres_backend = self._resolve_postgres_backend(
                db_connection,
                is_postgres=is_postgres,
            )
            if postgres_backend:
                # PostgreSQL
                rows = await db_connection.fetch(
                    """
                    SELECT password_hash FROM password_history
                    WHERE user_id = $1
                    ORDER BY created_at DESC, id DESC
                    LIMIT $2
                    """,
                    user_id, history_count
                )
                password_hashes = [row['password_hash'] for row in rows]
            else:
                # SQLite
                cursor = await db_connection.execute(
                    """
                    SELECT password_hash FROM password_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                    """,
                    (user_id, history_count)
                )
                rows = await cursor.fetchall()
                password_hashes = [row['password_hash'] for row in rows]

            # Check if new password matches any in history
            # SECURITY: Use constant-time loop to prevent timing attacks
            # Always check ALL hashes before returning to avoid leaking information
            # about which position in history matched
            match_found = False
            for old_hash in password_hashes:
                is_match, _ = self.verify_password(new_password, old_hash)
                if is_match:
                    match_found = True
                    # Don't break early - continue checking all hashes for constant time

            if match_found:
                if self.settings.PII_REDACT_LOGS:
                    logger.warning("Authenticated user attempted to reuse a recent password")
                else:
                    logger.warning(f"User {user_id} attempted to reuse a recent password")
                return False

            return True

        except Exception as e:
            if self.settings.PII_REDACT_LOGS:
                logger.error("Error checking password history (details redacted)")
            else:
                logger.error(f"Error checking password history: {e}")
            # SECURITY: Fail closed - don't allow potentially reused password when history check fails.
            # This prevents password reuse attacks if the history database becomes unavailable.
            # Users may need to retry if there's a transient DB error.
            return False

    async def add_to_password_history(
        self,
        user_id: int,
        password_hash: str,
        db_connection,
        *,
        is_postgres: Optional[bool] = None,
    ) -> None:
        """
        Add a password hash to user's password history

        Args:
            user_id: User ID
            password_hash: Password hash to add to history
            db_connection: Database connection
            is_postgres: Optional explicit backend mode override
        """
        try:
            postgres_backend = self._resolve_postgres_backend(
                db_connection,
                is_postgres=is_postgres,
            )
            if postgres_backend:
                # PostgreSQL
                await db_connection.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES ($1, $2)
                    """,
                    user_id, password_hash
                )

                # Clean up old entries
                await db_connection.execute(
                    """
                    DELETE FROM password_history
                    WHERE user_id = $1 AND id NOT IN (
                        SELECT id FROM password_history
                        WHERE user_id = $1
                        ORDER BY created_at DESC, id DESC
                        LIMIT $2
                    )
                    """,
                    user_id, self.settings.PASSWORD_HISTORY_RETENTION_COUNT
                )
            else:
                # SQLite
                await db_connection.execute(
                    """
                    INSERT INTO password_history (user_id, password_hash)
                    VALUES (?, ?)
                    """,
                    (user_id, password_hash)
                )

                # Clean up old entries
                await db_connection.execute(
                    """
                    DELETE FROM password_history
                    WHERE user_id = ? AND id NOT IN (
                        SELECT id FROM password_history
                        WHERE user_id = ?
                        ORDER BY created_at DESC, id DESC
                        LIMIT ?
                    )
                    """,
                    (user_id, user_id, self.settings.PASSWORD_HISTORY_RETENTION_COUNT)
                )

            if self.settings.PII_REDACT_LOGS:
                logger.debug("Added password to history for authenticated user (details redacted)")
            else:
                logger.debug(f"Added password to history for user {user_id}")

        except Exception as e:
            logger.error(f"Error adding to password history: {e}")
            # Don't raise - this shouldn't block password changes


#######################################################################################################################
#
# Module Functions for convenience

# Global instance with thread-safe initialization
_password_service: Optional[PasswordService] = None
_password_service_lock = threading.Lock()


def get_password_service() -> PasswordService:
    """Get password service singleton instance (thread-safe)"""
    global _password_service
    if _password_service is None:
        with _password_service_lock:
            # Double-check locking pattern to avoid unnecessary lock acquisition
            if _password_service is None:
                _password_service = PasswordService()
    return _password_service


def hash_password(password: str) -> str:
    """Convenience function to hash a password"""
    return get_password_service().hash_password(password)


def verify_password(password: str, password_hash: str) -> tuple[bool, bool]:
    """Convenience function to verify a password"""
    return get_password_service().verify_password(password, password_hash)


def generate_secure_password(length: int = 16) -> str:
    """Convenience function to generate a secure password"""
    return get_password_service().generate_secure_password(length)


def generate_registration_code(length: int = 24) -> str:
    """Convenience function to generate a registration code"""
    return get_password_service().generate_registration_code(length)


#
# End of password_service.py
#######################################################################################################################
