# security_filters.py
"""
Security filters for RAG pipeline.

This module provides PII detection, content filtering, access control,
and audit logging for sensitive data handling.
"""

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any, Optional

from loguru import logger


class SensitivityLevel(Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"


@dataclass
class PIIMatch:
    """A detected PII match."""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.pii_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    id: str
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    sensitivity: SensitivityLevel
    pii_detected: list[PIIType]
    access_granted: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "sensitivity": self.sensitivity.value,
            "pii_detected": json.dumps([p.value for p in self.pii_detected]),
            "access_granted": int(self.access_granted),
            "metadata": json.dumps(self.metadata)
        }


class PIIDetector:
    """Detects PII in text using regex patterns and heuristics."""

    def __init__(self):
        """Initialize PII detector with patterns."""
        self.patterns: dict[PIIType, list[Pattern]] = {
            PIIType.EMAIL: [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ],
            PIIType.PHONE: [
                re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # US
                re.compile(r'\b(?:\+?44[-.\s]?)?(?:\d{4}[-.\s]?\d{6}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'),  # UK
                re.compile(r'\b(?:\+?[1-9]\d{0,2}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),  # International
            ],
            PIIType.SSN: [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b\d{9}\b'),
            ],
            PIIType.CREDIT_CARD: [
                re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b'),
            ],
            PIIType.IP_ADDRESS: [
                re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),  # IPv4
                re.compile(r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'),  # IPv6
            ],
            PIIType.PASSPORT: [
                re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),  # Generic passport pattern
            ],
            PIIType.BANK_ACCOUNT: [
                re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:\d{3})?\b'),  # IBAN
                re.compile(r'\b\d{8,17}\b'),  # Generic account number
            ],
        }

        # Common name patterns (simplified - in production use NER)
        self.name_indicators = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Lady",
            "Jr.", "Sr.", "III", "IV"
        ]

    def detect_pii(self, text: str) -> list[PIIMatch]:
        """
        Detect PII in text.

        Args:
            text: Text to scan

        Returns:
            List of PII matches
        """
        matches: list[PIIMatch] = []

        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Validate match based on type
                    if self._validate_match(pii_type, match.group()):
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=self._get_confidence(pii_type, match.group())
                        ))

        # Detect potential names (simplified)
        matches.extend(self._detect_names(text))

        # Remove duplicates
        unique_matches: list[PIIMatch] = []
        seen: set[tuple[PIIType, int, int]] = set()
        for pii_match in matches:
            key = (pii_match.pii_type, pii_match.start, pii_match.end)
            if key not in seen:
                seen.add(key)
                unique_matches.append(pii_match)

        return unique_matches

    def _validate_match(self, pii_type: PIIType, text: str) -> bool:
        """Validate if a match is likely real PII."""
        if pii_type == PIIType.SSN:
            # Check if it's not all same digit
            if len(set(text.replace("-", ""))) == 1:
                return False
            # Check if it starts with valid SSN prefix
            prefix = text[:3].replace("-", "")
            if prefix in ["000", "666"] or prefix.startswith("9"):
                return False

        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm validation
            return self._luhn_check(text)

        elif pii_type == PIIType.EMAIL:
            # Basic email validation
            parts = text.split("@")
            if len(parts) != 2:
                return False
            if not parts[0] or not parts[1]:
                return False

        return True

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in card_number if d.isdigit()]
            checksum = 0
            for i, digit in enumerate(reversed(digits[:-1])):
                if i % 2 == 0:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit
            return (checksum + digits[-1]) % 10 == 0
        except Exception as e:
            logger.debug(f"Luhn checksum validation failed: {type(e).__name__}")
            return False

    def _get_confidence(self, pii_type: PIIType, text: str) -> float:
        """Get confidence score for a PII match."""
        # Simple heuristic - in production use ML models
        if pii_type in [PIIType.EMAIL, PIIType.SSN, PIIType.CREDIT_CARD]:
            return 0.95
        elif pii_type in [PIIType.PHONE, PIIType.IP_ADDRESS]:
            return 0.85
        else:
            return 0.7

    def _detect_names(self, text: str) -> list[PIIMatch]:
        """Detect potential person names (simplified)."""
        matches: list[PIIMatch] = []

        # Look for name indicators
        for indicator in self.name_indicators:
            idx = text.find(indicator)
            if idx != -1:
                # Extract potential name after indicator
                end_idx = idx + len(indicator)
                remaining = text[end_idx:].strip()

                # Get next 2-3 words as potential name
                words = remaining.split()[:3]
                if words:
                    name = " ".join(words)
                    # Filter out common words
                    if len(name) > 2 and name[0].isupper():
                        matches.append(PIIMatch(
                            pii_type=PIIType.PERSON_NAME,
                            text=indicator + " " + name,
                            start=idx,
                            end=idx + len(indicator) + len(name) + 1,
                            confidence=0.6
                        ))

        return matches

    def mask_pii(self, text: str, matches: list[PIIMatch]) -> str:
        """
        Mask PII in text.

        Args:
            text: Original text
            matches: PII matches to mask

        Returns:
            Text with PII masked
        """
        if not matches:
            return text

        # Sort matches by position (reverse for replacement)
        sorted_matches = sorted(matches, key=lambda x: x.start, reverse=True)

        masked_text = text
        for match in sorted_matches:
            mask = self._generate_mask(match.pii_type, match.text)
            masked_text = masked_text[:match.start] + mask + masked_text[match.end:]

        return masked_text

    def _generate_mask(self, pii_type: PIIType, text: str) -> str:
        """Generate appropriate mask for PII type."""
        if pii_type == PIIType.EMAIL:
            parts = text.split("@")
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"

        elif pii_type == PIIType.PHONE:
            # Keep area code, mask rest
            digits = "".join(c for c in text if c.isdigit())
            if len(digits) >= 10:
                return f"{text[:5]}***-****"

        elif pii_type == PIIType.SSN:
            return "***-**-****"

        elif pii_type == PIIType.CREDIT_CARD:
            # Show last 4 digits
            digits = "".join(c for c in text if c.isdigit())
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"

        elif pii_type == PIIType.PERSON_NAME:
            words = text.split()
            if words:
                return f"{words[0]} " + " ".join("***" for _ in words[1:])

        # Default mask
        return f"[{pii_type.value.upper()}]"


class ContentFilter:
    """Filters content based on sensitivity and classification."""

    def __init__(self):
        """Initialize content filter."""
        self.sensitive_keywords = {
            SensitivityLevel.CONFIDENTIAL: [
                "password", "secret", "token", "api_key", "private_key",
                "confidential", "proprietary", "trade secret"
            ],
            SensitivityLevel.RESTRICTED: [
                "classified", "top secret", "restricted", "eyes only",
                "need to know", "sensitive compartmented"
            ]
        }

        self.content_categories = {
            "financial": ["bank", "account", "credit", "debit", "transaction", "payment"],
            "medical": ["patient", "diagnosis", "treatment", "prescription", "medical"],
            "legal": ["attorney", "lawyer", "lawsuit", "litigation", "contract"],
            "personal": ["birth", "death", "marriage", "divorce", "family"]
        }

    def classify_content(self, text: str) -> SensitivityLevel:
        """
        Classify content sensitivity level.

        Args:
            text: Text to classify

        Returns:
            Sensitivity level
        """
        text_lower = text.lower()

        # Check for restricted keywords
        for keyword in self.sensitive_keywords[SensitivityLevel.RESTRICTED]:
            if keyword in text_lower:
                return SensitivityLevel.RESTRICTED

        # Check for confidential keywords
        for keyword in self.sensitive_keywords[SensitivityLevel.CONFIDENTIAL]:
            if keyword in text_lower:
                return SensitivityLevel.CONFIDENTIAL

        # Check content categories
        category_matches = 0
        for _category, keywords in self.content_categories.items():
            if any(keyword in text_lower for keyword in keywords):
                category_matches += 1

        if category_matches >= 2:
            return SensitivityLevel.CONFIDENTIAL
        elif category_matches == 1:
            return SensitivityLevel.INTERNAL

        return SensitivityLevel.PUBLIC

    def filter_by_sensitivity(
        self,
        documents: list[dict[str, Any]],
        max_sensitivity: SensitivityLevel
    ) -> list[dict[str, Any]]:
        """
        Filter documents by sensitivity level.

        Args:
            documents: List of documents
            max_sensitivity: Maximum allowed sensitivity

        Returns:
            Filtered documents
        """
        sensitivity_order = {
            SensitivityLevel.PUBLIC: 0,
            SensitivityLevel.INTERNAL: 1,
            SensitivityLevel.CONFIDENTIAL: 2,
            SensitivityLevel.RESTRICTED: 3
        }

        max_level = sensitivity_order[max_sensitivity]
        filtered = []

        for doc in documents:
            content = doc.get("content", "")
            doc_sensitivity = self.classify_content(content)
            doc_level = sensitivity_order[doc_sensitivity]

            if doc_level <= max_level:
                filtered.append(doc)
            else:
                logger.debug(
                    f"Filtered document due to sensitivity: {doc_sensitivity.value} > {max_sensitivity.value}"
                )

        return filtered


class AccessController:
    """Controls access to documents based on user permissions."""

    def __init__(self):
        """Initialize access controller."""
        self.user_permissions = {}
        self.document_acls = {}
        self.role_permissions = {
            "admin": [SensitivityLevel.RESTRICTED],
            "manager": [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.INTERNAL],
            "employee": [SensitivityLevel.INTERNAL, SensitivityLevel.PUBLIC],
            "guest": [SensitivityLevel.PUBLIC]
        }

    def set_user_role(self, user_id: str, role: str):
        """Set user role."""
        if role not in self.role_permissions:
            raise ValueError(f"Invalid role: {role}")
        self.user_permissions[user_id] = {"role": role}

    def set_document_acl(self, document_id: str, allowed_users: list[str], allowed_roles: list[str]):
        """Set document access control list."""
        self.document_acls[document_id] = {
            "users": set(allowed_users),
            "roles": set(allowed_roles)
        }

    def check_access(
        self,
        user_id: str,
        document_id: str,
        sensitivity: SensitivityLevel
    ) -> bool:
        """
        Check if user has access to document.

        Args:
            user_id: User identifier
            document_id: Document identifier
            sensitivity: Document sensitivity level

        Returns:
            Access granted status
        """
        # Check user permissions
        user_info = self.user_permissions.get(user_id, {})
        user_role = user_info.get("role", "guest")

        # Check role-based sensitivity access
        allowed_sensitivities = self.role_permissions.get(user_role, [SensitivityLevel.PUBLIC])
        if sensitivity not in allowed_sensitivities:
            return False

        # Check document-specific ACL
        if document_id in self.document_acls:
            acl = self.document_acls[document_id]

            # Check user-specific access
            if user_id in acl["users"]:
                return True

            # Check role-based access
            if user_role in acl["roles"]:  # noqa: SIM103
                return True

            # If ACL exists but user not in it, deny
            return False

        # No specific ACL, allow based on sensitivity
        return True

    def filter_by_access(
        self,
        user_id: str,
        documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Filter documents by user access.

        Args:
            user_id: User identifier
            documents: List of documents

        Returns:
            Accessible documents
        """
        filtered = []

        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")

            # Classify sensitivity
            filter = ContentFilter()
            sensitivity = filter.classify_content(content)

            # Check access
            if self.check_access(user_id, doc_id, sensitivity):
                filtered.append(doc)
            else:
                logger.debug(f"Access denied for user {user_id} to document {doc_id}")

        return filtered


class SecurityAuditor:
    """Audit logger for security events."""

    def __init__(
        self,
        db_path: str = "security_audit.db",
        max_size_mb: int = 100,
        max_age_days: int = 90,
        rotation_check_interval: int = 3600
    ):
        """
        Initialize security auditor with rotation support.

        Args:
            db_path: Path to audit database
            max_size_mb: Maximum database size in MB before rotation
            max_age_days: Maximum age of records in days
            rotation_check_interval: Seconds between rotation checks
        """
        self.db_path = db_path
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days
        self.rotation_check_interval = rotation_check_interval
        self._last_rotation_check = time.time()
        self._init_database()
        self._check_rotation()

    def _init_database(self):
        """Initialize audit database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    pii_detected TEXT,
                    access_granted INTEGER,
                    metadata TEXT,
                    created_at REAL DEFAULT (unixepoch())
                )
            """)

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")

            conn.commit()

    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        sensitivity: SensitivityLevel,
        pii_detected: list[PIIType],
        access_granted: bool,
        metadata: Optional[dict[str, Any]] = None
    ):
        """Log security event."""
        entry = SecurityAuditEntry(
            id=self._generate_id(),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            sensitivity=sensitivity,
            pii_detected=pii_detected,
            access_granted=access_granted,
            metadata=metadata or {}
        )

        with sqlite3.connect(self.db_path) as conn:
            data = entry.to_dict()
            conn.execute(
                """
                INSERT INTO audit_log
                (id, timestamp, user_id, action, resource, sensitivity, pii_detected, access_granted, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["timestamp"],
                    data["user_id"],
                    data["action"],
                    data["resource"],
                    data["sensitivity"],
                    data["pii_detected"],
                    data["access_granted"],
                    data["metadata"]
                )
            )
            conn.commit()

    def _generate_id(self) -> str:
        """Generate unique audit ID."""
        content = f"{time.time()}:{hash(time.time())}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get audit trail."""
        # Check for rotation before querying
        self._check_rotation()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM audit_log WHERE 1=1"
            params: list[Any] = []

            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def _check_rotation(self):
        """Check if audit log needs rotation."""
        current_time = time.time()

        # Only check rotation periodically
        if current_time - self._last_rotation_check < self.rotation_check_interval:
            return

        self._last_rotation_check = current_time

        try:
            # Check database size
            if Path(self.db_path).exists():
                size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
                if size_mb > self.max_size_mb:
                    logger.info(f"Audit log size ({size_mb:.2f}MB) exceeds limit ({self.max_size_mb}MB), rotating...")
                    self._rotate_by_size()

            # Check for old records
            self._delete_old_records()

        except Exception as e:
            logger.error(f"Error checking rotation: {type(e).__name__}")

    def _rotate_by_size(self):
        """Rotate audit log when size limit exceeded."""
        try:
            # Create archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"{self.db_path}.{timestamp}.archive"

            # Rename current database to archive
            Path(self.db_path).rename(archive_path)

            # Create new database
            self._init_database()

            # Copy recent records from archive to new database
            with sqlite3.connect(archive_path) as archive_conn:
                with sqlite3.connect(self.db_path) as new_conn:
                    # Copy last 30 days of records
                    cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()

                    cursor = archive_conn.execute(
                        "SELECT * FROM audit_log WHERE timestamp >= ? ORDER BY timestamp",
                        (cutoff_date,)
                    )

                    records = cursor.fetchall()
                    if records:
                        new_conn.executemany(
                            """
                            INSERT INTO audit_log
                            (id, timestamp, user_id, action, resource, sensitivity, pii_detected, access_granted, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            records
                        )
                        new_conn.commit()

            logger.info(f"Audit log rotated to {archive_path}")

            # Compress archive if possible
            try:
                import gzip
                import shutil

                with open(archive_path, 'rb') as f_in, gzip.open(f"{archive_path}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                # Remove uncompressed archive
                Path(archive_path).unlink()
                logger.info(f"Archive compressed to {archive_path}.gz")

            except ImportError:
                logger.debug("gzip not available, archive not compressed")

        except Exception as e:
            logger.error(f"Error rotating audit log: {type(e).__name__}")

    def _delete_old_records(self):
        """Delete records older than max_age_days."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=self.max_age_days)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount

                if deleted_count > 0:
                    conn.execute("VACUUM")  # Reclaim space
                    logger.info(f"Deleted {deleted_count} old audit records")

        except Exception as e:
            logger.error(f"Error deleting old records: {type(e).__name__}")


class SecurityFilters:
    """Main security filtering system."""

    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_content_filtering: bool = True,
        enable_access_control: bool = True,
        enable_audit_logging: bool = True,
        audit_db_path: str = "security_audit.db"
    ):
        """
        Initialize security filters.

        Args:
            enable_pii_detection: Enable PII detection
            enable_content_filtering: Enable content filtering
            enable_access_control: Enable access control
            enable_audit_logging: Enable audit logging
            audit_db_path: Path to audit database
        """
        self.pii_detector = PIIDetector() if enable_pii_detection else None
        self.content_filter = ContentFilter() if enable_content_filtering else None
        self.access_controller = AccessController() if enable_access_control else None
        self.auditor = SecurityAuditor(audit_db_path) if enable_audit_logging else None

    def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        mask_pii: bool = True
    ) -> tuple[str, dict[str, Any]]:
        """
        Process query through security filters.

        Args:
            query: Query text
            user_id: User identifier
            mask_pii: Whether to mask detected PII

        Returns:
            Tuple of (processed_query, security_metadata)
        """
        metadata: dict[str, Any] = {
            "original_query": query,
            "user_id": user_id,
            "pii_detected": [],
            "sensitivity": None
        }
        pii_detected: list[dict[str, Any]] = []
        sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC

        processed_query = query

        # Detect PII
        if self.pii_detector:
            pii_matches = self.pii_detector.detect_pii(query)
            pii_detected = [match.to_dict() for match in pii_matches]
            metadata["pii_detected"] = pii_detected

            if mask_pii and pii_matches:
                processed_query = self.pii_detector.mask_pii(query, pii_matches)

        # Classify content
        if self.content_filter:
            sensitivity = self.content_filter.classify_content(query)
            metadata["sensitivity"] = sensitivity.value

        # Log access
        if self.auditor:
            self.auditor.log_access(
                user_id=user_id,
                action="query",
                resource="search",
                sensitivity=sensitivity,
                pii_detected=[PIIType(m["type"]) for m in pii_detected],
                access_granted=True,
                metadata=metadata
            )

        return processed_query, metadata

    def filter_documents(
        self,
        documents: list[dict[str, Any]],
        user_id: str = "anonymous",
        max_sensitivity: Optional[SensitivityLevel] = None,
        mask_pii: bool = False
    ) -> list[dict[str, Any]]:
        """
        Filter documents through security filters.

        Args:
            documents: List of documents
            user_id: User identifier
            max_sensitivity: Maximum allowed sensitivity
            mask_pii: Whether to mask PII in results

        Returns:
            Filtered documents
        """
        filtered = documents

        # Filter by sensitivity
        if self.content_filter and max_sensitivity:
            filtered = self.content_filter.filter_by_sensitivity(filtered, max_sensitivity)

        # Filter by access control
        if self.access_controller:
            filtered = self.access_controller.filter_by_access(user_id, filtered)

        # Process PII
        if self.pii_detector and mask_pii:
            for doc in filtered:
                content = doc.get("content", "")
                pii_matches = self.pii_detector.detect_pii(content)

                if pii_matches:
                    doc["content"] = self.pii_detector.mask_pii(content, pii_matches)
                    doc["pii_masked"] = True
                    doc["pii_types"] = list({m.pii_type.value for m in pii_matches})

        # Log access
        if self.auditor:
            for doc in filtered:
                self.auditor.log_access(
                    user_id=user_id,
                    action="view",
                    resource=doc.get("id", "unknown"),
                    sensitivity=self.content_filter.classify_content(doc.get("content", "")) if self.content_filter else SensitivityLevel.PUBLIC,
                    pii_detected=[],
                    access_granted=True,
                    metadata={"document_id": doc.get("id")}
                )

        return filtered


# Global instance
_security_filters: Optional[SecurityFilters] = None


def get_security_filters() -> SecurityFilters:
    """Get or create global security filters instance."""
    global _security_filters
    if _security_filters is None:
        _security_filters = SecurityFilters()
    return _security_filters


# Pipeline integration functions

async def apply_security_filters(context: Any, **kwargs) -> Any:
    """Apply security filters in RAG pipeline."""
    if not context.config.get("security", {}).get("enabled", False):
        return context

    security = get_security_filters()

    # Process query
    user_id = context.metadata.get("user_id", "anonymous")
    mask_pii = context.config.get("security", {}).get("mask_pii", True)

    processed_query, security_metadata = security.process_query(
        query=context.query,
        user_id=user_id,
        mask_pii=mask_pii
    )

    context.query = processed_query
    context.metadata["security"] = security_metadata

    # Filter documents if present
    if hasattr(context, "documents") and context.documents:
        max_sensitivity = context.config.get("security", {}).get("max_sensitivity")
        if max_sensitivity:
            max_sensitivity = SensitivityLevel(max_sensitivity)

        context.documents = security.filter_documents(
            documents=context.documents,
            user_id=user_id,
            max_sensitivity=max_sensitivity,
            mask_pii=mask_pii
        )

        logger.debug(f"Filtered to {len(context.documents)} documents after security checks")

    return context
