# test_case_manager.py
# Test case management for Prompt Studio

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    ConflictError,
    DatabaseError,
    InputError,
    PromptStudioDatabase,
)

########################################################################################################################
# Data Classes

@dataclass
class TestCase:
    """Test case for prompt evaluation."""
    name: str
    inputs: dict[str, Any]
    expected_outputs: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tags: list[str] = field(default_factory=list)
    is_golden: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "tags": self.tags,
            "is_golden": self.is_golden,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Create from dictionary."""
        return cls(**data)

@dataclass
class TestResult:
    """Result of running a test case."""
    test_case_id: str
    actual_outputs: dict[str, Any]
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

########################################################################################################################
# Test Case Manager Class

class TestCaseManager:
    """Manages test cases for Prompt Studio projects."""

    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize TestCaseManager with database instance.

        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
        self.test_cases = {}
        self.results = {}

    ####################################################################################################################
    # CRUD Operations

    def create_test_case(self, project_id: int, name: Optional[str], inputs: dict[str, Any],
                        description: Optional[str] = None, expected_outputs: Optional[dict[str, Any]] = None,
                        tags: Optional[list[str]] = None, is_golden: bool = False,
                        signature_id: Optional[int] = None) -> dict[str, Any]:
        """
        Create a new test case.

        Args:
            project_id: Parent project ID
            name: Test case name
            description: Test case description
            inputs: Input data
            expected_outputs: Expected output data
            tags: Tags for categorization
            is_golden: Whether this is a golden test case
            signature_id: Associated signature ID

        Returns:
            Created test case record
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Test case name cannot be empty")

        try:
            return self.db.create_test_case(
                project_id,
                name.strip(),
                inputs=inputs,
                description=description,
                expected_outputs=expected_outputs,
                tags=tags,
                is_golden=is_golden,
                signature_id=signature_id,
                client_id=self.client_id,
            )
        except (ConflictError, InputError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to create test case: {exc}") from exc

    def get_test_case(self, test_case_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        """
        Get a test case by ID.

        Args:
            test_case_id: Test case ID
            include_deleted: Include soft-deleted test cases

        Returns:
            Test case record or None
        """
        return self.db.get_test_case(test_case_id, include_deleted=include_deleted)

    def list_test_cases(self, project_id: int, signature_id: Optional[int] = None,
                       is_golden: Optional[bool] = None, tags: Optional[list[str]] = None,
                       search: Optional[str] = None, include_deleted: bool = False,
                       page: int = 1, per_page: int = 20,
                       return_pagination: bool = False) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Delegate to the database layer for filtered test case retrieval."""

        try:
            return self.db.list_test_cases(
                project_id,
                signature_id=signature_id,
                is_golden=is_golden,
                tags=tags,
                search=search,
                include_deleted=include_deleted,
                page=page,
                per_page=per_page,
                return_pagination=return_pagination,
            )
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to list test cases: {exc}") from exc

    def update_test_case(self, test_case_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        """
        Update a test case.

        Args:
            test_case_id: Test case ID
            updates: Fields to update

        Returns:
            Updated test case record
        """
        try:
            return self.db.update_test_case(test_case_id, updates)
        except (ConflictError, InputError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to update test case {test_case_id}: {exc}") from exc

    def delete_test_case(self, test_case_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a test case.

        Args:
            test_case_id: Test case ID
            hard_delete: Permanently delete if True

        Returns:
            True if deleted
        """
        try:
            return self.db.delete_test_case(test_case_id, hard_delete=hard_delete)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to delete test case {test_case_id}: {exc}") from exc

    ####################################################################################################################
    # Bulk Operations

    def create_bulk_test_cases(self, project_id: int, test_cases: list[dict[str, Any]],
                              signature_id: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Create multiple test cases at once.

        Args:
            project_id: Project ID
            test_cases: List of test case data
            signature_id: Optional signature ID for all test cases

        Returns:
            List of created test case records
        """
        try:
            return self.db.create_bulk_test_cases(
                project_id,
                test_cases,
                signature_id=signature_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to create test cases in bulk: {exc}")
            raise DatabaseError(f"Bulk creation failed: {exc}") from exc

    ####################################################################################################################
    # Search and Filter

    def search_test_cases(self, project_id: int, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search test cases using FTS.

        Args:
            project_id: Project ID
            query: Search query
            limit: Maximum results

        Returns:
            List of matching test cases
        """
        try:
            return self.db.search_test_cases(project_id, query, limit=limit)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to search test cases: {exc}") from exc

    # Compatibility method used in integration tests via patching
    async def run_batch_tests(self,
                              prompt_id: int,
                              test_case_ids: list[int],
                              model: str = "gpt-3.5-turbo",
                              temperature: float = 0.7,
                              max_tokens: int = 1000,
                              provider: str = "openai",
                              app_config: Optional[dict[str, Any]] = None,
                              api_key_override: Optional[str] = None,
                              credentials_resolved: bool = False,
                              provider_credentials: ProviderCallCredentials | None = None,
                              timeout_seconds: Optional[float] = None,
                              strict_provider_errors: bool = False,
                              on_provider_success: Any = None,
                              ) -> list[dict[str, Any]]:
        """Run multiple test cases (async wrapper).

        Delegates to TestRunner.run_multiple_tests. Exists to match test patch targets.
        """
        try:
            from .test_runner import TestRunner
            runner = TestRunner(self.db)
            return await runner.run_multiple_tests(
                prompt_id=prompt_id,
                test_case_ids=test_case_ids,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                parallel=False,
                provider=provider,
                app_config=app_config,
                api_key_override=api_key_override,
                credentials_resolved=credentials_resolved,
                provider_credentials=provider_credentials,
                timeout_seconds=timeout_seconds,
                strict_provider_errors=strict_provider_errors,
                on_provider_success=on_provider_success,
            )
        except Exception as e:
            if strict_provider_errors:
                raise
            logger.error("run_batch_tests failed; error_type={}", type(e).__name__)
            return []

    def get_golden_test_cases(self, project_id: int) -> list[dict[str, Any]]:
        """
        Get all golden test cases for a project.

        Args:
            project_id: Project ID

        Returns:
            List of golden test cases
        """
        result = self.list_test_cases(
            project_id=project_id,
            is_golden=True,
            per_page=1000,  # Get all golden cases
            return_pagination=True
        )
        return result["test_cases"]

    def get_test_cases_by_signature(self, signature_id: int) -> list[dict[str, Any]]:
        """
        Get all test cases for a signature.

        Args:
            signature_id: Signature ID

        Returns:
            List of test cases
        """
        try:
            return self.db.get_test_cases_by_signature(signature_id)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch test cases for signature {signature_id}: {exc}") from exc

    ####################################################################################################################
    # Statistics

    def get_test_case_stats(self, project_id: int) -> dict[str, Any]:
        """
        Get statistics for test cases in a project.

        Args:
            project_id: Project ID

        Returns:
            Statistics dictionary
        """
        try:
            return self.db.get_test_case_stats(project_id)
        except Exception as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to compute test case statistics: {exc}") from exc
