"""
Enhanced webhook security validation and permissions.

Provides comprehensive security validation for webhook URLs, domain filtering,
rate limiting, and permission management for webhook operations.
"""

import asyncio
import ipaddress
import re
import socket
import ssl
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from loguru import logger

from tldw_Server_API.app.core.Evaluations.config_manager import get_config
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch


class WebhookSecurityLevel(Enum):
    """Webhook security levels."""
    PERMISSIVE = "permissive"  # Allow most URLs (development)
    STANDARD = "standard"      # Basic security checks (staging)
    STRICT = "strict"          # Full security validation (production)


@dataclass
class WebhookValidationError:
    """Webhook validation error."""
    code: str
    message: str
    severity: str
    field: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "field": self.field
        }


@dataclass
class WebhookValidationResult:
    """Result of webhook validation."""
    valid: bool
    errors: list[WebhookValidationError]
    warnings: list[WebhookValidationError]
    security_score: float  # 0.0 to 1.0
    metadata: dict[str, Any]

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


async def _close_response(resp: Any) -> None:
    if resp is None:
        return
    close = getattr(resp, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(resp, "close", None)
    if callable(close):
        close()


class WebhookSecurityValidator:
    """Validates webhook URLs and configurations for security."""

    def __init__(self):
        """Initialize webhook security validator."""
        # Load security configuration
        self.security_level = WebhookSecurityLevel(
            get_config("webhooks.security.security_level", "standard")
        )

        # URL validation settings
        self.require_https = get_config("webhooks.security.require_https", False)
        self.validate_ssl = get_config("webhooks.security.validate_ssl_certificates", True)
        self.max_url_length = get_config("webhooks.security.max_url_length", 2048)

        # Domain filtering
        self.allowed_domains = set(get_config("webhooks.security.allowed_domains", []))
        self.blocked_domains = set(get_config("webhooks.security.blocked_domains", []))

        # Rate limiting
        self.max_webhooks_per_user = get_config("webhooks.registration_limits.per_user_max", 10)
        self.max_registrations_per_url = get_config("webhooks.registration_limits.per_url_max", 1)

        # Private network ranges to block (RFC 1918, RFC 4193, etc.)
        self.private_networks = [
            ipaddress.IPv4Network("10.0.0.0/8"),
            ipaddress.IPv4Network("172.16.0.0/12"),
            ipaddress.IPv4Network("192.168.0.0/16"),
            ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local
            ipaddress.IPv4Network("127.0.0.0/8"),    # Loopback
            ipaddress.IPv6Network("::1/128"),        # IPv6 loopback
            ipaddress.IPv6Network("fc00::/7"),       # IPv6 unique local
            ipaddress.IPv6Network("fe80::/10"),      # IPv6 link-local
        ]

        # Blocked ports
        self.blocked_ports = {
            22,    # SSH
            23,    # Telnet
            25,    # SMTP
            53,    # DNS
            110,   # POP3
            143,   # IMAP
            993,   # IMAPS
            995,   # POP3S
            1433,  # SQL Server
            3306,  # MySQL
            5432,  # PostgreSQL
            6379,  # Redis
            11211, # Memcached
            27017, # MongoDB
        }

    async def validate_webhook_url(
        self,
        url: str,
        user_id: str,
        check_connectivity: bool = True
    ) -> WebhookValidationResult:
        """
        Validate a webhook URL for security and accessibility.

        Args:
            url: Webhook URL to validate
            user_id: User ID for audit logging
            check_connectivity: Whether to test URL connectivity

        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        metadata = {}

        try:
            # Basic URL validation
            parsed_url = urlparse(url)
            metadata["parsed_url"] = {
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "hostname": parsed_url.hostname,
                "port": parsed_url.port,
                "path": parsed_url.path
            }

            # Length check
            if len(url) > self.max_url_length:
                errors.append(WebhookValidationError(
                    code="URL_TOO_LONG",
                    message=f"URL exceeds maximum length of {self.max_url_length} characters",
                    severity="error",
                    field="url"
                ))

            # Scheme validation
            if parsed_url.scheme not in ["http", "https"]:
                errors.append(WebhookValidationError(
                    code="INVALID_SCHEME",
                    message=f"Invalid URL scheme: {parsed_url.scheme}. Only HTTP/HTTPS allowed",
                    severity="error",
                    field="url"
                ))

            # HTTPS requirement
            if self.require_https and parsed_url.scheme != "https":
                if self.security_level == WebhookSecurityLevel.STRICT:
                    errors.append(WebhookValidationError(
                        code="HTTPS_REQUIRED",
                        message="HTTPS is required for webhook URLs in production",
                        severity="error",
                        field="url"
                    ))
                else:
                    warnings.append(WebhookValidationError(
                        code="HTTP_DISCOURAGED",
                        message="HTTP is discouraged. Consider using HTTPS",
                        severity="warning",
                        field="url"
                    ))

            # Hostname validation
            if not parsed_url.hostname:
                errors.append(WebhookValidationError(
                    code="MISSING_HOSTNAME",
                    message="URL must include a valid hostname",
                    severity="error",
                    field="url"
                ))
            else:
                await self._validate_hostname(parsed_url.hostname, errors, warnings, metadata)

            # Port validation
            port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
            if port in self.blocked_ports:
                errors.append(WebhookValidationError(
                    code="BLOCKED_PORT",
                    message=f"Port {port} is not allowed for webhook URLs",
                    severity="error",
                    field="url"
                ))

            # Domain filtering
            if parsed_url.hostname:
                domain_validation = self._validate_domain(parsed_url.hostname)
                errors.extend(domain_validation["errors"])
                warnings.extend(domain_validation["warnings"])
                metadata.update(domain_validation["metadata"])

            # Path validation
            self._validate_path(parsed_url.path, warnings)

            # Query parameter validation
            if parsed_url.query:
                self._validate_query_params(parsed_url.query, warnings)

            # Connectivity test
            if check_connectivity and not errors and parsed_url.hostname:
                connectivity_result = await self._test_connectivity(
                    url, parsed_url.hostname, port
                )
                metadata["connectivity"] = connectivity_result

                if not connectivity_result["reachable"]:
                    warnings.append(WebhookValidationError(
                        code="CONNECTIVITY_FAILED",
                        message=f"Unable to reach webhook URL: {connectivity_result.get('error', 'Unknown error')}",
                        severity="warning",
                        field="url"
                    ))

        except Exception as e:
            errors.append(WebhookValidationError(
                code="VALIDATION_ERROR",
                message=f"URL validation failed: {str(e)}",
                severity="error",
                field="url"
            ))

        # Calculate security score
        security_score = self._calculate_security_score(url, errors, warnings, metadata)

        # (Optional) Could emit a unified audit event here if desired

        return WebhookValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            security_score=security_score,
            metadata=metadata
        )

    async def _validate_hostname(
        self,
        hostname: str,
        errors: list[WebhookValidationError],
        warnings: list[WebhookValidationError],
        metadata: dict[str, Any]
    ):
        """Validate hostname for security issues."""
        try:
            # Check for localhost variations
            localhost_patterns = [
                # Validation denylist literals (not socket bind targets).
                "localhost", "127.0.0.1", "::1", "0.0.0.0",  # nosec B104
                "10.0.", "172.16.", "172.17.", "172.18.", "172.19.",
                "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                "172.30.", "172.31.", "192.168."
            ]

            hostname_lower = hostname.lower()
            for pattern in localhost_patterns:
                if hostname_lower.startswith(pattern):
                    # Block private networks at STANDARD and STRICT levels for security
                    if self.security_level in [WebhookSecurityLevel.STANDARD, WebhookSecurityLevel.STRICT]:
                        errors.append(WebhookValidationError(
                            code="PRIVATE_NETWORK",
                            message=f"Private network addresses are not allowed: {hostname}",
                            severity="error",
                            field="url"
                        ))
                    else:
                        warnings.append(WebhookValidationError(
                            code="PRIVATE_NETWORK_WARNING",
                            message=f"Private network address detected: {hostname}",
                            severity="warning",
                            field="url"
                        ))
                    break

            # DNS resolution and IP validation
            try:
                import socket
                ip_addresses = socket.getaddrinfo(hostname, None)
                resolved_ips = list({addr[4][0] for addr in ip_addresses})
                metadata["resolved_ips"] = resolved_ips

                for ip_str in resolved_ips:
                    try:
                        ip_addr = ipaddress.ip_address(ip_str)

                        # Check if IP is in private networks
                        for network in self.private_networks:
                            if ip_addr in network:
                                # Block private networks at STANDARD and STRICT levels for security
                                if self.security_level in [WebhookSecurityLevel.STANDARD, WebhookSecurityLevel.STRICT]:
                                    errors.append(WebhookValidationError(
                                        code="PRIVATE_IP",
                                        message=f"Hostname resolves to private IP: {ip_str}",
                                        severity="error",
                                        field="url"
                                    ))
                                else:
                                    warnings.append(WebhookValidationError(
                                        code="PRIVATE_IP_WARNING",
                                        message=f"Hostname resolves to private IP: {ip_str}",
                                        severity="warning",
                                        field="url"
                                    ))
                                break

                    except ValueError:
                        # Invalid IP address
                        pass

            except socket.gaierror as e:
                warnings.append(WebhookValidationError(
                    code="DNS_RESOLUTION_FAILED",
                    message=f"Failed to resolve hostname: {str(e)}",
                    severity="warning",
                    field="url"
                ))

        except Exception as e:
            logger.warning(f"Hostname validation error: {e}")

    def _validate_domain(self, hostname: str) -> dict[str, Any]:
        """Validate domain against allow/block lists."""
        errors = []
        warnings = []
        metadata = {"domain_status": "unknown"}

        # Check blocked domains
        if self.blocked_domains:
            for blocked_domain in self.blocked_domains:
                if hostname.endswith(blocked_domain) or hostname == blocked_domain:
                    errors.append(WebhookValidationError(
                        code="BLOCKED_DOMAIN",
                        message=f"Domain is blocked: {hostname}",
                        severity="error",
                        field="url"
                    ))
                    metadata["domain_status"] = "blocked"
                    break

        # Check allowed domains (if specified)
        if self.allowed_domains and metadata["domain_status"] != "blocked":
            domain_allowed = False
            for allowed_domain in self.allowed_domains:
                if hostname.endswith(allowed_domain) or hostname == allowed_domain:
                    domain_allowed = True
                    metadata["domain_status"] = "allowed"
                    break

            if not domain_allowed:
                errors.append(WebhookValidationError(
                    code="DOMAIN_NOT_ALLOWED",
                    message=f"Domain is not in allowed list: {hostname}",
                    severity="error",
                    field="url"
                ))
                metadata["domain_status"] = "not_allowed"

        if metadata["domain_status"] == "unknown":
            metadata["domain_status"] = "neutral"

        return {
            "errors": errors,
            "warnings": warnings,
            "metadata": metadata
        }

    def _validate_path(self, path: str, warnings: list[WebhookValidationError]):
        """Validate URL path for potential issues."""
        # Check for suspicious path patterns
        suspicious_patterns = [
            r'\.\./', r'%2e%2e%2f',  # Directory traversal
            r'<script', r'javascript:', r'vbscript:',  # Script injection
            r'file://', r'ftp://',  # Non-HTTP protocols
        ]

        path_lower = path.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, path_lower):
                warnings.append(WebhookValidationError(
                    code="SUSPICIOUS_PATH",
                    message="Potentially suspicious path pattern detected",
                    severity="warning",
                    field="url"
                ))
                break

    def _validate_query_params(self, query: str, warnings: list[WebhookValidationError]):
        """Validate URL query parameters."""
        try:
            params = parse_qs(query)

            # Check for excessive parameters
            if len(params) > 20:
                warnings.append(WebhookValidationError(
                    code="EXCESSIVE_PARAMS",
                    message="URL contains excessive query parameters",
                    severity="warning",
                    field="url"
                ))

            # Check for suspicious parameter names
            suspicious_param_names = ['eval', 'exec', 'system', 'cmd', 'shell']
            for param_name in params:
                if param_name.lower() in suspicious_param_names:
                    warnings.append(WebhookValidationError(
                        code="SUSPICIOUS_PARAMETER",
                        message=f"Suspicious parameter name detected: {param_name}",
                        severity="warning",
                        field="url"
                    ))
                    break

        except Exception:
            # Query parsing failed
            warnings.append(WebhookValidationError(
                code="INVALID_QUERY_PARAMS",
                message="Invalid query parameters in URL",
                severity="warning",
                field="url"
            ))

    async def _test_connectivity(
        self,
        url: str,
        hostname: str,
        port: int
    ) -> dict[str, Any]:
        """Test webhook URL connectivity."""
        result = {
            "reachable": False,
            "response_time_ms": None,
            "ssl_valid": None,
            "error": None
        }

        try:
            # Create SSL context
            ssl_context = ssl.create_default_context()
            if not self.validate_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            start_time = asyncio.get_event_loop().time()
            resp = None
            try:
                # Validate hostname and port before making request
                # Resolve the hostname once and use the IP directly to prevent DNS rebinding
                safe_ip = None
                try:
                    ip_addr = ipaddress.ip_address(hostname)
                    # Check if IP is in private networks
                    for network in self.private_networks:
                        if ip_addr in network:
                            result["error"] = f"Private network address not allowed: {hostname}"
                            return result
                    safe_ip = str(ip_addr)
                except ValueError:
                    # Not a direct IP, resolve it
                    try:
                        ip_addresses = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
                        if not ip_addresses:
                            result["error"] = "Failed to resolve hostname"
                            return result

                        # Check all resolved IPs
                        safe_ips = []
                        for addr_info in ip_addresses:
                            ip_str = addr_info[4][0]
                            try:
                                ip_addr = ipaddress.ip_address(ip_str)
                                # Check against private networks
                                is_private = False
                                for network in self.private_networks:
                                    if ip_addr in network:
                                        is_private = True
                                        break

                                if is_private:
                                    result["error"] = f"Hostname resolves to private IP: {ip_str}"
                                    return result

                                safe_ips.append(ip_str)
                            except ValueError:
                                continue

                        if not safe_ips:
                            result["error"] = "No valid IPs found for hostname"
                            return result

                        # Use the first safe IP
                        safe_ip = safe_ips[0]

                    except socket.gaierror as e:
                        result["error"] = f"DNS resolution failed: {str(e)}"
                        return result

                # Reconstruct URL with the resolved IP to prevent DNS rebinding
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(url)

                # Build new URL with IP address instead of hostname
                if parsed.port:
                    netloc = f"[{safe_ip}]:{parsed.port}" if ":" in safe_ip else f"{safe_ip}:{parsed.port}"
                else:
                    netloc = f"[{safe_ip}]" if ":" in safe_ip else safe_ip

                # Create the URL with IP
                ip_url = urlunparse((
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))

                # Set Host header to original hostname for virtual hosting
                headers = {'Host': hostname}

                resp = await afetch(
                    method="HEAD",
                    url=ip_url,  # Use IP-based URL to prevent DNS rebinding
                    headers=headers,
                    timeout=10,
                    allow_redirects=False,
                    retry=RetryPolicy(attempts=1),
                    verify=ssl_context,
                )
                end_time = asyncio.get_event_loop().time()
                result["response_time_ms"] = int((end_time - start_time) * 1000)
                result["reachable"] = True
                status_code = int(resp.status_code)
                result["status_code"] = status_code

                # Check for redirect attempts
                if status_code in [301, 302, 303, 307, 308]:
                    redirect_location = resp.headers.get('Location', '') or resp.headers.get('location', '')
                    if redirect_location:
                        # Parse and validate redirect location
                        try:
                            redirect_parsed = urlparse(redirect_location)
                            if redirect_parsed.hostname:
                                # Validate redirect hostname
                                try:
                                    redirect_ip = socket.getaddrinfo(redirect_parsed.hostname, None)[0][4][0]
                                    redirect_addr = ipaddress.ip_address(redirect_ip)
                                    for network in self.private_networks:
                                        if redirect_addr in network:
                                            result["error"] = f"Redirect to private network blocked: {redirect_location}"
                                            result["reachable"] = False
                                            return result
                                except (socket.gaierror, ValueError):
                                    pass
                            result["redirect_location"] = redirect_location[:200]  # Truncate for safety
                        except Exception as redirect_error:
                            logger.debug("Webhook redirect location parsing failed", exc_info=redirect_error)

                # Check SSL if HTTPS
                if url.startswith("https://"):
                    result["ssl_valid"] = True  # If we got here, SSL is valid
            finally:
                await _close_response(resp)
        except Exception as e:
            result["error"] = f"Connectivity test failed: {str(e)}"

        return result

    def _calculate_security_score(
        self,
        url: str,
        errors: list[WebhookValidationError],
        warnings: list[WebhookValidationError],
        metadata: dict[str, Any]
    ) -> float:
        """Calculate security score for the webhook URL."""
        score = 1.0  # Start with perfect score

        # Deduct for errors (major issues)
        for error in errors:
            if error.code in ["PRIVATE_NETWORK", "PRIVATE_IP", "BLOCKED_DOMAIN"]:
                score -= 0.4  # Major security issue
            elif error.code in ["HTTPS_REQUIRED", "BLOCKED_PORT"]:
                score -= 0.3  # Significant security issue
            else:
                score -= 0.2  # General error

        # Deduct for warnings (minor issues)
        for warning in warnings:
            if warning.code in ["PRIVATE_NETWORK_WARNING", "HTTP_DISCOURAGED"]:
                score -= 0.1  # Minor security concern
            else:
                score -= 0.05  # General warning

        # Bonus for good practices
        if url.startswith("https://"):
            score += 0.1

        if metadata.get("domain_status") == "allowed":
            score += 0.1

        if metadata.get("connectivity", {}).get("ssl_valid"):
            score += 0.05

        return max(0.0, min(1.0, score))  # Clamp between 0.0 and 1.0


class WebhookPermissionManager:
    """Manages webhook permissions and ownership."""

    def __init__(self, db_adapter):
        """Initialize permission manager with database adapter."""
        self.db_adapter = db_adapter

    async def check_webhook_permissions(
        self,
        user_id: str,
        webhook_id: Optional[int] = None,
        url: Optional[str] = None,
        action: str = "access"
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user has permission to perform action on webhook.

        Args:
            user_id: User ID
            webhook_id: Webhook ID (optional)
            url: Webhook URL (optional)
            action: Action to check (access, modify, delete)

        Returns:
            Tuple of (has_permission, error_message)
        """
        try:
            with self.db_adapter.transaction():
                if webhook_id:
                    # Check by webhook ID
                    row = self.db_adapter.fetch_one("""
                        SELECT user_id FROM webhook_registrations
                        WHERE id = ? AND active = ?
                    """, (webhook_id, True))
                    if row is None:
                        return False, "Webhook not found"

                    webhook_owner = row.get("user_id") if isinstance(row, dict) else row[0]
                    if webhook_owner != user_id:
                        # (Optional) Could emit a unified audit SECURITY_VIOLATION here
                        return False, "Access denied: not webhook owner"

                elif url:
                    # Check by URL - only check ownership for non-register actions
                    if action != "register":
                        row = self.db_adapter.fetch_one("""
                            SELECT user_id FROM webhook_registrations
                            WHERE user_id = ? AND url = ? AND active = ?
                        """, (user_id, url, True))

                        if not row:
                            return False, "Webhook not found or access denied"

                # Check rate limits for registration actions
                if action == "register":
                    user_webhook_count = self._get_user_webhook_count(user_id)
                    max_webhooks = get_config("webhooks.registration_limits.per_user_max", 10)

                    if user_webhook_count >= max_webhooks:
                        return False, f"Maximum webhook limit reached ({max_webhooks})"

                    if url:
                        url_registration_count = self._get_url_registration_count(url)
                        max_per_url = get_config("webhooks.registration_limits.per_url_max", 1)

                        if url_registration_count >= max_per_url:
                            return False, f"URL already has maximum registrations ({max_per_url})"

                return True, None

        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False, f"Permission check failed: {str(e)}"

    def _get_user_webhook_count(self, user_id: str) -> int:
        """Get number of active webhooks for user."""
        count = self.db_adapter.fetch_value("""
            SELECT COUNT(*) FROM webhook_registrations
            WHERE user_id = ? AND active = ?
        """, (user_id, True))
        return count or 0

    def _get_url_registration_count(self, url: str) -> int:
        """Get number of active registrations for URL."""
        count = self.db_adapter.fetch_value("""
            SELECT COUNT(*) FROM webhook_registrations
            WHERE url = ? AND active = ?
        """, (url, True))
        return count or 0


# Global instances
webhook_validator = WebhookSecurityValidator()
