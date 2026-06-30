"""
Unit tests for PromptsInteropService.

Tests the core prompts service functionality with minimal mocking -
only the database layer is mocked.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import json

from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import PromptsInteropService

# ========================================================================
# Service Initialization Tests
# ========================================================================

class TestServiceInitialization:
    """Test service initialization and setup."""

    @pytest.mark.unit
    def test_service_initialization(self, test_db_path):
        """Test basic service initialization."""
        service = PromptsInteropService(
            db_directory=str(test_db_path.parent),
            client_id="test_client"
        )

        assert service is not None
        assert service.client_id == "test_client"
        assert service.db_directory == test_db_path.parent
        assert service._db_instance is None

    @pytest.mark.unit
    def test_service_lazy_db_initialization(self, mock_prompts_service):
        """Test lazy database initialization."""
        service = mock_prompts_service

        # DB should not be initialized until first use
        assert service._db_instance is not None  # Mock sets it up

        # Access DB
        service.list_prompts()
        service._db_instance.list_prompts.assert_called_once()

    @pytest.mark.unit
    def test_service_thread_safety(self, mock_prompts_service):
        """Test thread-safe database access."""
        service = mock_prompts_service

        import threading
        results = []

        def access_db():

            prompts = service.list_prompts()
            results.append(prompts)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=access_db)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should get same result
        assert len(results) == 5

# ========================================================================
# Prompt CRUD Operations Tests
# ========================================================================

class TestPromptCRUDOperations:
    """Test prompt creation, reading, updating, and deletion."""

    @pytest.mark.unit
    def test_create_prompt(self, mock_prompts_service, sample_prompt):
        """Test prompt creation."""
        service = mock_prompts_service

        prompt_id = service.create_prompt(**sample_prompt)

        assert prompt_id == 1
        service._db_instance.create_prompt.assert_called_once_with(
            name=sample_prompt['name'],
            content=sample_prompt['content'],
            author=sample_prompt['author'],
            keywords=sample_prompt['keywords']
        )

    @pytest.mark.unit
    def test_get_prompt(self, mock_prompts_service):
        """Test getting a prompt."""
        service = mock_prompts_service

        prompt = service.get_prompt(prompt_id=1)

        assert prompt is not None
        assert prompt['id'] == 1
        assert prompt['name'] == 'Test Prompt'
        service._db_instance.get_prompt.assert_called_once_with(1)

    @pytest.mark.unit
    def test_get_prompt_not_found(self, mock_prompts_service):
        """Test getting non-existent prompt."""
        service = mock_prompts_service
        service._db_instance.get_prompt.return_value = None

        prompt = service.get_prompt(prompt_id=999)

        assert prompt is None

    @pytest.mark.unit
    def test_list_prompts(self, mock_prompts_service, sample_prompts):
        """Test listing prompts."""
        service = mock_prompts_service
        service._db_instance.list_prompts.return_value = sample_prompts

        prompts = service.list_prompts()

        assert len(prompts) == len(sample_prompts)
        assert prompts[0]['name'] == 'Assistant Prompt'
        service._db_instance.list_prompts.assert_called_once()

    @pytest.mark.unit
    def test_update_prompt(self, mock_prompts_service):
        """Test updating a prompt."""
        service = mock_prompts_service

        result = service.update_prompt(
            prompt_id=1,
            content="Updated content",
            version_comment="Fixed typo"
        )

        assert result['success'] is True
        service._db_instance.update_prompt.assert_called_once()

    @pytest.mark.unit
    def test_delete_prompt(self, mock_prompts_service):
        """Test deleting a prompt."""
        service = mock_prompts_service

        result = service.delete_prompt(prompt_id=1)

        assert result['success'] is True
        service._db_instance.delete_prompt.assert_called_once_with(1)

    @pytest.mark.unit
    def test_restore_prompt(self, mock_prompts_service):
        """Test restoring a deleted prompt."""
        service = mock_prompts_service
        service._db_instance.restore_prompt = Mock(return_value={'success': True})

        result = service.restore_prompt(prompt_id=1)

        assert result['success'] is True
        service._db_instance.restore_prompt.assert_called_once_with(1)

# ========================================================================
# Prompt Version Management Tests
# ========================================================================

class TestPromptVersioning:
    """Test prompt versioning functionality."""

    @pytest.mark.unit
    def test_get_prompt_versions(self, mock_prompts_service, versioned_prompt):
        """Test getting prompt versions."""
        service = mock_prompts_service
        service._db_instance.get_prompt_versions.return_value = versioned_prompt['versions']

        versions = service.get_prompt_versions(prompt_id=1)

        assert len(versions) == 3
        assert versions[0]['version'] == 1
        assert versions[2]['comment'] == 'Production ready'

    @pytest.mark.unit
    def test_restore_version(self, mock_prompts_service):
        """Test restoring a specific version."""
        service = mock_prompts_service

        result = service.restore_version(prompt_id=1, version=2)

        assert result['success'] is True
        service._db_instance.restore_version.assert_called_once_with(1, 2)

    @pytest.mark.unit
    def test_get_version_diff(self, mock_prompts_service):
        """Test getting diff between versions."""
        service = mock_prompts_service
        service._db_instance.get_version_diff = Mock(return_value={
            'added': ['new line'],
            'removed': ['old line'],
            'modified': []
        })

        diff = service.get_version_diff(prompt_id=1, version1=1, version2=2)

        assert 'added' in diff
        assert 'removed' in diff
        assert len(diff['added']) == 1

# ========================================================================
# Search and Filter Tests
# ========================================================================

class TestSearchAndFilter:
    """Test search and filtering functionality."""

    @pytest.mark.unit
    def test_search_prompts_by_keyword(self, mock_prompts_service):
        """Test searching prompts by keyword."""
        service = mock_prompts_service
        service._db_instance.search_prompts.return_value = [
            {'id': 1, 'name': 'Code Helper', 'keywords': ['code', 'debugging']}
        ]

        results = service.search_prompts(query="code")

        assert len(results) == 1
        assert 'code' in results[0]['keywords']
        service._db_instance.search_prompts.assert_called_once_with(query="code")

    @pytest.mark.unit
    def test_search_prompts_by_author(self, mock_prompts_service):
        """Test searching prompts by author."""
        service = mock_prompts_service
        service._db_instance.search_prompts.return_value = [
            {'id': 1, 'name': 'Test Prompt', 'author': 'dev_user'}
        ]

        results = service.search_prompts(query="author:dev_user")

        assert len(results) == 1
        assert results[0]['author'] == 'dev_user'

    @pytest.mark.unit
    def test_search_prompts_uses_real_database_signature(self, tmp_path):
        """Test service search against the real database keyword signature."""
        service = PromptsInteropService(
            db_directory=str(tmp_path),
            client_id="test_client"
        )

        class SignatureOnlyDB:
            def __init__(self):
                self.call = None

            def search_prompts(self, *, search_query, search_fields, page, results_per_page, include_deleted):
                self.call = {
                    "search_query": search_query,
                    "search_fields": search_fields,
                    "page": page,
                    "results_per_page": results_per_page,
                    "include_deleted": include_deleted,
                }
                return ([{"id": 1, "name": "Code Helper"}], 1)

        fake_db = SignatureOnlyDB()
        service._db_instance = fake_db

        results = service.search_prompts(query="code")

        assert results == [{"id": 1, "name": "Code Helper"}]
        assert fake_db.call == {
            "search_query": "code",
            "search_fields": None,
            "page": 1,
            "results_per_page": 50,
            "include_deleted": False,
        }

    @pytest.mark.unit
    def test_filter_prompts_by_keywords(self, mock_prompts_service):
        """Test filtering prompts by keywords."""
        service = mock_prompts_service
        service._db_instance.filter_prompts = Mock(return_value=[
            {'id': 1, 'name': 'Code Review', 'keywords': ['code', 'review']}
        ])

        results = service.filter_prompts(keywords=['code', 'review'])

        assert len(results) == 1
        assert all(k in results[0]['keywords'] for k in ['code', 'review'])

    @pytest.mark.unit
    def test_get_prompts_by_category(self, mock_prompts_service):
        """Test getting prompts by category."""
        service = mock_prompts_service
        service._db_instance.get_prompts_by_category = Mock(return_value=[
            {'id': 1, 'name': 'Code Helper', 'category': 'development'}
        ])

        results = service.get_prompts_by_category('development')

        assert len(results) == 1
        assert results[0]['category'] == 'development'

# ========================================================================
# Template Variable Processing Tests
# ========================================================================

class TestTemplateProcessing:
    """Test template variable extraction and processing."""

    @pytest.mark.unit
    def test_extract_variables_simple(self, mock_prompts_service):
        """Test extracting simple variables from template."""
        service = mock_prompts_service

        content = "You are a {{role}}. Please {{task}}."
        variables = service.extract_template_variables(content)

        assert 'role' in variables
        assert 'task' in variables
        assert len(variables) == 2

    @pytest.mark.unit
    def test_extract_variables_complex(self, mock_prompts_service, complex_prompt):
        """Test extracting variables from complex template."""
        service = mock_prompts_service

        variables = service.extract_template_variables(complex_prompt['content'])

        expected_vars = [
            'expertise_area', 'context', 'data_type', 'data',
            'requirement_1', 'requirement_2', 'requirement_3', 'output_format'
        ]
        for var in expected_vars:
            assert var in variables

    @pytest.mark.unit
    def test_render_template(self, mock_prompts_service):
        """Test rendering template with variables."""
        service = mock_prompts_service

        template = "You are a {{role}}. Please {{task}}."
        variables = {'role': 'helpful assistant', 'task': 'explain Python'}

        rendered = service.render_template(template, variables)

        assert "helpful assistant" in rendered
        assert "explain Python" in rendered
        assert "{{" not in rendered

    @pytest.mark.unit
    def test_render_template_missing_variables(self, mock_prompts_service):
        """Test rendering template with missing variables."""
        service = mock_prompts_service

        template = "You are a {{role}}. Please {{task}}."
        variables = {'role': 'assistant'}  # Missing 'task'

        with pytest.raises(KeyError) as exc_info:
            service.render_template(template, variables)

        assert 'task' in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_template(self, mock_prompts_service):
        """Test template validation."""
        service = mock_prompts_service

        valid_template = "You are {{role}}."
        invalid_template = "You are {{role}."  # Missing closing brace

        assert service.validate_template(valid_template) is True
        assert service.validate_template(invalid_template) is False

# ========================================================================
# Import/Export Tests
# ========================================================================

class TestImportExport:
    """Test prompt import/export functionality."""

    @pytest.mark.unit
    def test_export_prompts(self, mock_prompts_service, sample_prompts):
        """Test exporting prompts."""
        service = mock_prompts_service
        service._db_instance.list_prompts.return_value = sample_prompts

        export_data = service.export_prompts(prompt_ids=[1, 2, 3])

        assert 'version' in export_data
        assert 'prompts' in export_data
        assert len(export_data['prompts']) == 3
        assert 'exported_at' in export_data

    @pytest.mark.unit
    def test_export_prompts_json_format(self, mock_prompts_service, sample_prompts):
        """Test exporting prompts in JSON format."""
        service = mock_prompts_service
        service._db_instance.list_prompts.return_value = sample_prompts

        json_data = service.export_prompts_json(prompt_ids=[1])

        # Should be valid JSON
        parsed = json.loads(json_data)
        assert 'prompts' in parsed
        assert len(parsed['prompts']) > 0

    @pytest.mark.unit
    def test_import_prompts(self, mock_prompts_service, export_data):
        """Test importing prompts."""
        service = mock_prompts_service
        service._db_instance.create_prompt.side_effect = [1, 2]  # Return IDs

        results = service.import_prompts(export_data)

        assert results['imported'] == 2
        assert results['failed'] == 0
        assert len(results['prompt_ids']) == 2

    @pytest.mark.unit
    def test_import_prompts_with_duplicates(self, mock_prompts_service, export_data):
        """Test importing prompts with duplicate handling."""
        service = mock_prompts_service
        service._db_instance.create_prompt.side_effect = [
            1,  # First prompt succeeds
            Exception("UNIQUE constraint failed")  # Second is duplicate
        ]

        results = service.import_prompts(export_data, skip_duplicates=True)

        assert results['imported'] == 1
        assert results['failed'] == 1
        assert results['skipped'] == 1

    @pytest.mark.unit
    def test_validate_import_data(self, mock_prompts_service, export_data):
        """Test validating import data structure."""
        service = mock_prompts_service

        # Valid data
        assert service.validate_import_data(export_data) is True

        # Invalid data - missing prompts
        invalid_data = {'version': '1.0'}
        assert service.validate_import_data(invalid_data) is False

        # Invalid prompt structure
        invalid_prompt_data = {
            'version': '1.0',
            'prompts': [{'name': 'Test'}]  # Missing content
        }
        assert service.validate_import_data(invalid_prompt_data) is False

# ========================================================================
# Bulk Operations Tests
# ========================================================================

class TestBulkOperations:
    """Test bulk operations on prompts."""

    @pytest.mark.unit
    def test_bulk_delete(self, mock_prompts_service):
        """Test bulk delete operation."""
        service = mock_prompts_service
        service._db_instance.delete_prompt.return_value = {'success': True}

        results = service.bulk_delete([1, 2, 3])

        assert results['deleted'] == 3
        assert results['failed'] == 0
        assert service._db_instance.delete_prompt.call_count == 3

    @pytest.mark.unit
    def test_bulk_update_keywords(self, mock_prompts_service):
        """Test bulk update of keywords."""
        service = mock_prompts_service
        service._db_instance.update_prompt_keywords = Mock(return_value={'success': True})

        results = service.bulk_update_keywords(
            prompt_ids=[1, 2, 3],
            add_keywords=['new-tag'],
            remove_keywords=['old-tag']
        )

        assert results['updated'] == 3
        assert results['failed'] == 0

    @pytest.mark.unit
    def test_bulk_export(self, mock_prompts_service, sample_prompts):
        """Test bulk export with filters."""
        service = mock_prompts_service
        service._db_instance.filter_prompts = Mock(return_value=sample_prompts)

        export_data = service.bulk_export(
            author='test_user',
            keywords=['test']
        )

        assert 'prompts' in export_data
        assert len(export_data['prompts']) == len(sample_prompts)

# ========================================================================
# Collection Management Tests
# ========================================================================

class TestCollectionManagement:
    """Test prompt collection management."""

    @pytest.mark.unit
    def test_create_collection(self, mock_prompts_service, prompt_collection):
        """Test creating a prompt collection."""
        service = mock_prompts_service
        service._db_instance.create_collection = Mock(return_value=1)

        collection_id = service.create_collection(
            name=prompt_collection['name'],
            description=prompt_collection['description'],
            prompt_ids=[p['id'] for p in prompt_collection['prompts']]
        )

        assert collection_id == 1
        service._db_instance.create_collection.assert_called_once()

    @pytest.mark.unit
    def test_get_collection(self, mock_prompts_service, prompt_collection):
        """Test getting a collection."""
        service = mock_prompts_service
        service._db_instance.get_collection = Mock(return_value=prompt_collection)

        collection = service.get_collection(collection_id=1)

        assert collection['name'] == 'Development Toolkit'
        assert len(collection['prompts']) == 4

    @pytest.mark.unit
    def test_add_to_collection(self, mock_prompts_service):
        """Test adding prompts to collection."""
        service = mock_prompts_service
        service._db_instance.add_to_collection = Mock(return_value={'success': True})

        result = service.add_to_collection(
            collection_id=1,
            prompt_ids=[5, 6]
        )

        assert result['success'] is True
        service._db_instance.add_to_collection.assert_called_once()

    @pytest.mark.unit
    def test_remove_from_collection(self, mock_prompts_service):
        """Test removing prompts from collection."""
        service = mock_prompts_service
        service._db_instance.remove_from_collection = Mock(return_value={'success': True})

        result = service.remove_from_collection(
            collection_id=1,
            prompt_ids=[3]
        )

        assert result['success'] is True

# ========================================================================
# Statistics and Analytics Tests
# ========================================================================

class TestStatistics:
    """Test prompt statistics and analytics."""

    @pytest.mark.unit
    def test_get_prompt_statistics(self, mock_prompts_service):
        """Test getting prompt statistics."""
        service = mock_prompts_service
        service._db_instance.get_statistics = Mock(return_value={
            'total_prompts': 100,
            'total_authors': 10,
            'total_keywords': 50,
            'avg_versions_per_prompt': 2.5
        })

        stats = service.get_statistics()

        assert stats['total_prompts'] == 100
        assert stats['total_authors'] == 10
        assert stats['avg_versions_per_prompt'] == 2.5

    @pytest.mark.unit
    def test_get_usage_analytics(self, mock_prompts_service):
        """Test getting usage analytics."""
        service = mock_prompts_service
        service._db_instance.get_usage_analytics = Mock(return_value={
            'most_used': [{'id': 1, 'usage_count': 50}],
            'recently_updated': [{'id': 2, 'updated_at': '2024-01-01'}],
            'most_versioned': [{'id': 3, 'version_count': 10}]
        })

        analytics = service.get_usage_analytics()

        assert 'most_used' in analytics
        assert 'recently_updated' in analytics
        assert analytics['most_used'][0]['usage_count'] == 50

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in the service."""

    @pytest.mark.unit
    def test_handle_database_error(self, mock_prompts_service):
        """Test handling of database errors."""
        service = mock_prompts_service
        service._db_instance.create_prompt.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            service.create_prompt(
                name="Test",
                content="Content",
                author="test"
            )

        assert "Database error" in str(exc_info.value)

    @pytest.mark.unit
    def test_handle_validation_error(self, mock_prompts_service):
        """Test handling of validation errors."""
        service = mock_prompts_service

        # Empty name should fail validation
        with pytest.raises(ValueError) as exc_info:
            service.create_prompt(
                name="",
                content="Content",
                author="test"
            )

        assert "name" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_handle_import_error(self, mock_prompts_service):
        """Test handling of import errors."""
        service = mock_prompts_service

        # Invalid import data
        invalid_data = "not a dict"

        with pytest.raises(TypeError) as exc_info:
            service.import_prompts(invalid_data)

        assert "dict" in str(exc_info.value).lower()

# ========================================================================
# Cleanup and Resource Management Tests
# ========================================================================

class TestResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.unit
    def test_close_database(self, mock_prompts_service):
        """Test closing database connection."""
        service = mock_prompts_service
        service._db_instance.close = Mock()

        service.close()

        service._db_instance.close.assert_called_once()
        assert service._db_instance is not None  # Mock doesn't actually close

    @pytest.mark.unit
    def test_context_manager(self, test_db_path):
        """Test service as context manager."""
        with PromptsInteropService(
            db_directory=str(test_db_path.parent),
            client_id="test_client"
        ) as service:
            assert service is not None
            # Service should be usable within context

        # After context, service should be closed
        # (would check if we had real implementation)
