"""
Configuration validation for Evaluations module.

Ensures proper configuration and environment variables are set
for production deployment.
"""

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class ConfigurationIssue:
    """Represents a configuration issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "security", "performance", "integration"
    message: str
    recommendation: str


class EvaluationConfigValidator:
    """Validates configuration for the Evaluations module."""

    def __init__(self):
        self.issues: list[ConfigurationIssue] = []

    def validate(self) -> dict[str, Any]:
        """
        Perform comprehensive configuration validation.

        Returns:
            Dict containing validation results and issues
        """
        self.issues = []

        # Run all validation checks
        self._validate_authentication()
        self._validate_api_keys()
        self._validate_database()
        self._validate_rate_limiting()
        self._validate_monitoring()
        self._validate_security()

        # Categorize issues
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        info = [i for i in self.issues if i.severity == "info"]

        # Determine overall status
        if errors:
            status = "error"
            message = f"Configuration validation failed with {len(errors)} errors"
        elif warnings:
            status = "warning"
            message = f"Configuration validation passed with {len(warnings)} warnings"
        else:
            status = "success"
            message = "Configuration validation passed"

        return {
            "status": status,
            "message": message,
            "issues": {
                "errors": [self._issue_to_dict(i) for i in errors],
                "warnings": [self._issue_to_dict(i) for i in warnings],
                "info": [self._issue_to_dict(i) for i in info]
            },
            "summary": {
                "total_issues": len(self.issues),
                "errors": len(errors),
                "warnings": len(warnings),
                "info": len(info)
            }
        }

    def _validate_authentication(self):
        """Validate authentication configuration."""
        # Check AUTH_MODE
        auth_mode = os.getenv("AUTH_MODE", "single_user")

        if auth_mode == "single_user":
            # Check for API key configuration
            api_key = os.getenv("API_BEARER") or os.getenv("SINGLE_USER_API_KEY")

            if not api_key:
                self.issues.append(ConfigurationIssue(
                    severity="error",
                    category="security",
                    message="No API key configured for single-user mode",
                    recommendation="Set API_BEARER or SINGLE_USER_API_KEY environment variable"
                ))
            elif api_key in {
                "default-secret-key-for-single-user",
                "test-api-key-12345",
                "change-me-in-production",
            }:
                self.issues.append(ConfigurationIssue(
                    severity="error",
                    category="security",
                    message="Using default API key is not secure",
                    recommendation=(
                        "Generate a secure API key, e.g. "
                        "`python -c \"import secrets; print(secrets.token_urlsafe(32))\"` "
                        "and set SINGLE_USER_API_KEY"
                    )
                ))

        elif auth_mode == "multi_user":
            # Check JWT configuration
            jwt_secret = os.getenv("JWT_SECRET_KEY")
            if not jwt_secret:
                self.issues.append(ConfigurationIssue(
                    severity="error",
                    category="security",
                    message="No JWT secret key configured for multi-user mode",
                    recommendation="Set JWT_SECRET_KEY environment variable"
                ))

    def _validate_api_keys(self):
        """Validate LLM API keys."""
        # Check for at least one LLM provider
        providers = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "GOOGLE_API_KEY": "Google",
            "COHERE_API_KEY": "Cohere"
        }

        configured_providers = []
        for env_var, provider in providers.items():
            if os.getenv(env_var):
                configured_providers.append(provider)

        if not configured_providers:
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="integration",
                message="No LLM provider API keys configured",
                recommendation="Set at least one API key (e.g., OPENAI_API_KEY) for evaluation features"
            ))

        # Check for hardcoded keys
        for env_var in providers:
            value = os.getenv(env_var, "")
            if value and ("test" in value.lower() or "demo" in value.lower()):
                self.issues.append(ConfigurationIssue(
                    severity="warning",
                    category="security",
                    message=f"Possible test/demo API key detected for {providers[env_var]}",
                    recommendation="Use production API keys for deployment"
                ))

    def _validate_database(self):
        """Validate database configuration."""
        # Check database path
        db_path = os.getenv("EVALUATIONS_DB_PATH")

        if db_path and ":memory:" in db_path:
            self.issues.append(ConfigurationIssue(
                severity="error",
                category="performance",
                message="In-memory database not suitable for production",
                recommendation="Use persistent database path"
            ))

        # Check for database migrations
        if not os.getenv("SKIP_DB_MIGRATIONS"):
            self.issues.append(ConfigurationIssue(
                severity="info",
                category="integration",
                message="Database migrations will run on startup",
                recommendation="Ensure database backup before deployment"
            ))

    def _validate_rate_limiting(self):
        """Validate rate limiting configuration."""
        from tldw_Server_API.app.core.config import rg_enabled as _rg_enabled_flag

        if not bool(_rg_enabled_flag(True)):
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="performance",
                message="Resource Governor rate limiting is disabled",
                recommendation=(
                    "Set RG_ENABLED=true and tune evals.default requests.rpm/"
                    "requests.burst in Config_Files/resource_governor_policies.yaml"
                )
            ))
            return

        policy_path = os.getenv("RG_POLICY_PATH")
        if policy_path and not os.path.exists(policy_path):
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="performance",
                message=f"Resource Governor policy file not found ({policy_path})",
                recommendation=(
                    "Set RG_POLICY_PATH to a readable policy file and tune "
                    "evals.default requests.rpm/requests.burst"
                )
            ))

    def _validate_monitoring(self):
        """Validate monitoring configuration."""
        # Check for monitoring endpoints
        metrics_enabled = os.getenv("METRICS_ENABLED", "false").lower() == "true"

        if not metrics_enabled:
            self.issues.append(ConfigurationIssue(
                severity="info",
                category="performance",
                message="Metrics collection disabled",
                recommendation="Set METRICS_ENABLED=true for production monitoring"
            ))

        # Check for log level
        log_level = os.getenv("LOG_LEVEL", "INFO")
        if log_level == "DEBUG":
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="performance",
                message="Debug logging enabled may impact performance",
                recommendation="Set LOG_LEVEL=INFO or WARNING for production"
            ))

    def _validate_security(self):
        """Validate security configuration."""
        # Check CORS settings
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        if cors_origins == "*":
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="security",
                message="CORS allows all origins",
                recommendation="Set CORS_ORIGINS to specific allowed domains"
            ))

        # Check HTTPS enforcement
        if not os.getenv("FORCE_HTTPS"):
            self.issues.append(ConfigurationIssue(
                severity="info",
                category="security",
                message="HTTPS not enforced",
                recommendation="Set FORCE_HTTPS=true for production"
            ))

        # Check session encryption
        if not os.getenv("SESSION_ENCRYPTION_KEY"):
            self.issues.append(ConfigurationIssue(
                severity="warning",
                category="security",
                message="Session encryption key not set",
                recommendation="Set SESSION_ENCRYPTION_KEY for secure sessions"
            ))

    def _issue_to_dict(self, issue: ConfigurationIssue) -> dict[str, str]:
        """Convert issue to dictionary."""
        return {
            "severity": issue.severity,
            "category": issue.category,
            "message": issue.message,
            "recommendation": issue.recommendation
        }


def validate_configuration() -> dict[str, Any]:
    """
    Convenience function to validate configuration.

    Returns:
        Validation results
    """
    validator = EvaluationConfigValidator()
    return validator.validate()


def check_production_readiness() -> bool:
    """
    Check if configuration is production-ready.

    Returns:
        True if no errors, False otherwise
    """
    results = validate_configuration()

    if results["status"] == "error":
        logger.error(f"Configuration not production-ready: {results['message']}")
        for error in results["issues"]["errors"]:
            logger.error(f"  - {error['message']}")
        return False

    if results["status"] == "warning":
        logger.warning(f"Configuration has warnings: {results['message']}")
        for warning in results["issues"]["warnings"]:
            logger.warning(f"  - {warning['message']}")

    return results["status"] != "error"
