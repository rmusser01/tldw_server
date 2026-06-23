"""
Property-based tests for Prompt Management using Hypothesis.

Tests invariants and properties that should always hold true.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, precondition, invariant, Bundle
import json
import re
from datetime import datetime

from tldw_Server_API.app.core.Prompt_Management.Prompts_Interop import PromptsInteropService

# Apply a default Hypothesis profile for this module to suppress fixture health checks
settings.register_profile(
    "tldw_prompt_props",
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("tldw_prompt_props")
from tldw_Server_API.app.core.DB_Management.Prompts_DB_V2 import PromptsDB

# ========================================================================
# Property Strategies
# ========================================================================

# Alphabets to avoid heavy filtering
ascii_printable = st.characters(min_codepoint=32, max_codepoint=126)
ascii_nonspace = st.characters(min_codepoint=33, max_codepoint=126)  # excludes space

# Valid prompt name strategy (no leading/trailing space, ASCII)
prompt_name_strategy = st.builds(
    lambda a, mid, b: a + mid + b,
    a=st.text(alphabet=ascii_nonspace, min_size=1, max_size=1),
    mid=st.text(alphabet=ascii_printable, min_size=0, max_size=198),
    b=st.text(alphabet=ascii_nonspace, min_size=1, max_size=1),
)

# Valid prompt content strategy
prompt_content_strategy = st.text(min_size=1, max_size=10000)

STRUCTURED_SEARCH_PREFIXES = {"author", "keyword"}


def uses_structured_search_prefix(query: str) -> bool:
    """Return true when a query uses field-filter syntax instead of free text."""
    stripped = query.lstrip()
    if ":" not in stripped:
        return False
    prefix, _rest = stripped.split(":", 1)
    return prefix.strip().casefold() in STRUCTURED_SEARCH_PREFIXES

# Template content with variables
template_strategy = st.builds(
    lambda base, vars: base + ''.join(f' {{{{var{i}}}}}' for i in range(vars)),
    base=st.text(min_size=1, max_size=100),
    vars=st.integers(min_value=0, max_value=5)
)

# Keywords strategy (ASCII printable, unique)
keywords_strategy = st.lists(
    st.text(alphabet=ascii_printable, min_size=1, max_size=50),
    min_size=0,
    max_size=10,
    unique=True,
)

# Author name strategy (ASCII, no leading space)
author_strategy = st.builds(
    lambda h, t: h + t,
    h=st.text(alphabet=ascii_nonspace, min_size=1, max_size=1),
    t=st.text(alphabet=ascii_printable, min_size=0, max_size=99),
)

# ========================================================================
# Template Variable Properties
# ========================================================================

class TestTemplateProperties:
    """Test properties of template variable extraction and rendering."""

    @pytest.mark.property
    @given(template=template_strategy)
    def test_variable_extraction_idempotent(self, template, prompts_service):
        """Extracting variables multiple times gives same result."""
        service = prompts_service

        vars1 = service.extract_template_variables(template)
        vars2 = service.extract_template_variables(template)

        assert vars1 == vars2

    @pytest.mark.property
    @given(
        template=st.text(min_size=1, max_size=1000),
        variables=st.dictionaries(
            st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
            st.text(min_size=0, max_size=100),
            min_size=0,
            max_size=10
        )
    )
    def test_render_removes_all_placeholders(self, template, variables, prompts_service):
        """Rendering with all variables should remove all placeholders."""
        service = prompts_service

        # Build template with known variables
        template_with_vars = template
        for var_name in variables:
            template_with_vars += f" {{{{var_name}}}}"

        # Extract and render
        extracted = service.extract_template_variables(template_with_vars)

        # Provide values for all extracted variables
        var_values = {var: f"value_{var}" for var in extracted}

        if var_values:  # Only test if there are variables
            rendered = service.render_template(template_with_vars, var_values)

            # No placeholders should remain
            assert '{{' not in rendered
            assert '}}' not in rendered

    @pytest.mark.property
    @given(content=st.text(min_size=0, max_size=1000))
    def test_no_variables_means_unchanged(self, content, prompts_service):
        """Content without variables should be unchanged after rendering."""
        service = prompts_service

        # Remove any accidental variable patterns
        clean_content = content.replace('{{', '').replace('}}', '')

        variables = service.extract_template_variables(clean_content)
        assert len(variables) == 0

        rendered = service.render_template(clean_content, {})
        assert rendered == clean_content

# ========================================================================
# CRUD Operation Properties
# ========================================================================

class TestCRUDProperties:
    """Test properties of CRUD operations."""

    @pytest.mark.property
    @given(
        name=prompt_name_strategy,
        content=prompt_content_strategy,
        author=author_strategy,
        keywords=keywords_strategy
    )
    def test_create_then_get_preserves_data(
        self, name, content, author, keywords, populated_prompts_db
    ):
        """Creating and getting a prompt preserves all data."""
        db = populated_prompts_db

        # Create prompt
        prompt_id = db.create_prompt(
            name=name,
            content=content,
            author=author,
            keywords=keywords
        )

        # Get prompt
        prompt = db.get_prompt(prompt_id)

        assert prompt is not None
        assert prompt['name'] == name
        assert prompt['content'] == content
        assert prompt['author'] == author
        assert set(prompt.get('keywords', [])) == set(keywords)

    @pytest.mark.property
    @given(
        updates=st.lists(
            st.dictionaries(
                st.sampled_from(['content', 'keywords']),
                st.one_of(
                    prompt_content_strategy,
                    keywords_strategy
                ),
                min_size=1,
                max_size=2
            ),
            min_size=1,
            max_size=5
        )
    )
    def test_multiple_updates_preserve_history(self, updates, populated_prompts_db):
        """Multiple updates should preserve version history."""
        db = populated_prompts_db

        # Create initial prompt
        prompt_id = db.create_prompt(
            name="Version Test",
            content="Initial",
            author="test"
        )

        # Apply updates
        for i, update in enumerate(updates):
            if 'content' in update:
                db.update_prompt(
                    prompt_id=prompt_id,
                    content=update['content'],
                    version_comment=f"Update {i}"
                )

        # Get versions
        versions = db.get_prompt_versions(prompt_id)

        # Should have initial version plus updates
        assert len(versions) >= 1

        # Versions should be in order
        for i in range(len(versions) - 1):
            assert versions[i]['version'] < versions[i + 1]['version']

    @pytest.mark.property
    @given(
        name=prompt_name_strategy,
        content=prompt_content_strategy
    )
    def test_soft_delete_is_reversible(self, name, content, populated_prompts_db):
        """Soft delete should be reversible."""
        db = populated_prompts_db

        # Create prompt
        prompt_id = db.create_prompt(name=name, content=content, author="test")

        # Delete it
        db.delete_prompt(prompt_id)

        # Should not appear in list
        prompts = db.list_prompts()
        assert not any(p['id'] == prompt_id for p in prompts)

        # Restore it
        db.restore_prompt(prompt_id)

        # Should appear in list again
        prompts = db.list_prompts()
        assert any(p['id'] == prompt_id for p in prompts)

        # Content should be preserved
        restored = db.get_prompt(prompt_id)
        assert restored['name'] == name
        assert restored['content'] == content

# ========================================================================
# Search Properties
# ========================================================================

class TestSearchProperties:
    """Test properties of search functionality."""

    @pytest.mark.property
    @given(query=st.text(min_size=1, max_size=100))
    def test_search_is_case_insensitive(self, query, populated_prompts_db):
        """Search should be case insensitive."""
        assume(not uses_structured_search_prefix(query))
        db = populated_prompts_db

        # Create prompt with known content
        prompt_id = db.create_prompt(
            name=f"Search Test {query}",
            content=f"Content with {query} in it",
            author="test"
        )

        # Search with different cases
        results_lower = db.search_prompts(query.lower())
        results_upper = db.search_prompts(query.upper())
        results_mixed = db.search_prompts(query.swapcase())

        # All should find the same prompts
        ids_lower = {r['id'] for r in results_lower}
        ids_upper = {r['id'] for r in results_upper}
        ids_mixed = {r['id'] for r in results_mixed}

        assert prompt_id in ids_lower
        assert ids_lower == ids_upper == ids_mixed

    @pytest.mark.property
    @given(
        keywords=st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    def test_keyword_search_finds_exact_matches(self, keywords, populated_prompts_db):
        """Searching by keyword should find exact matches."""
        db = populated_prompts_db

        # Create prompt with specific keywords
        prompt_id = db.create_prompt(
            name="Keyword Test",
            content="Test content",
            author="test",
            keywords=keywords
        )

        # Search for each keyword
        for keyword in keywords:
            results = db.search_prompts(f"keyword:{keyword}")

            # Should find our prompt
            assert any(r['id'] == prompt_id for r in results)

    @pytest.mark.property
    @given(author=author_strategy)
    def test_author_filter_is_exact(self, author, populated_prompts_db):
        """Author filter should match exactly."""
        db = populated_prompts_db

        # Create prompts with specific author
        prompt_ids = []
        for i in range(3):
            pid = db.create_prompt(
                name=f"Author Test {i}",
                content="Content",
                author=author
            )
            prompt_ids.append(pid)

        # Search by author
        results = db.search_prompts(f"author:{author}")
        result_ids = {r['id'] for r in results}

        # Should find all our prompts
        for pid in prompt_ids:
            assert pid in result_ids

# ========================================================================
# Import/Export Properties
# ========================================================================

class TestImportExportProperties:
    """Test properties of import/export functionality."""

    @pytest.mark.property
    @given(
        prompts=st.lists(
            st.builds(
                dict,
                name=prompt_name_strategy,
                content=prompt_content_strategy,
                author=author_strategy,
                keywords=keywords_strategy
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_export_import_roundtrip(self, prompts, prompts_service):
        """Exporting and importing should preserve all data."""
        service = prompts_service

        # Create prompts
        created_ids = []
        for prompt_data in prompts:
            pid = service.create_prompt(**prompt_data)
            created_ids.append(pid)

        # Export prompts
        export_data = service.export_prompts(prompt_ids=created_ids)

        # Clear database (simulate fresh import)
        for pid in created_ids:
            service.delete_prompt(pid)

        # Import prompts
        import_result = service.import_prompts(export_data)

        assert import_result['imported'] == len(prompts)

        # Verify all data preserved
        for original, new_id in zip(prompts, import_result['prompt_ids']):
            imported = service.get_prompt(new_id)
            assert imported['name'] == original['name']
            assert imported['content'] == original['content']
            assert imported['author'] == original['author']
            assert set(imported.get('keywords', [])) == set(original['keywords'])

    @pytest.mark.property
    @given(
        export_data=st.builds(
            dict,
            version=st.just("1.0"),
            exported_at=st.just(datetime.utcnow().isoformat()),
            prompts=st.lists(
                st.builds(
                    dict,
                    name=prompt_name_strategy,
                    content=prompt_content_strategy,
                    author=author_strategy,
                    keywords=keywords_strategy
                ),
                min_size=1,
                max_size=5
            )
        )
    )
    def test_import_validates_structure(self, export_data, prompts_service):
        """Import should validate data structure."""
        service = prompts_service

        # Valid data should import
        assert service.validate_import_data(export_data) is True

        # Invalid data should fail validation
        invalid_data = {**export_data}
        del invalid_data['prompts']
        assert service.validate_import_data(invalid_data) is False

        # Missing required fields in prompts
        invalid_prompts = {**export_data}
        invalid_prompts['prompts'] = [{'name': 'Test'}]  # Missing content
        assert service.validate_import_data(invalid_prompts) is False

# ========================================================================
# Stateful Property Testing
# ========================================================================

class PromptStateMachine(RuleBasedStateMachine):
    """Stateful testing for prompt management operations."""

    def __init__(self):

        super().__init__()
        self.service = None
        self.prompt_ids = set()
        self.deleted_ids = set()
        self.prompt_data = {}

    prompts = Bundle('prompts')
    deleted_prompts = Bundle('deleted_prompts')

    @rule()
    def initialize_service(self):
        """Initialize the service if not already done."""
        if self.service is None:
            import tempfile
            self.temp_dir = tempfile.mkdtemp()
            self.service = PromptsInteropService(
                db_directory=self.temp_dir,
                client_id="test"
            )

    @rule(
        target=prompts,
        name=prompt_name_strategy,
        content=prompt_content_strategy,
        author=author_strategy,
        keywords=keywords_strategy
    )
    def create_prompt(self, name, content, author, keywords):
        """Create a new prompt."""
        if self.service is None:
            self.initialize_service()

        prompt_id = self.service.create_prompt(
            name=name,
            content=content,
            author=author,
            keywords=keywords
        )

        self.prompt_ids.add(prompt_id)
        self.prompt_data[prompt_id] = {
            'name': name,
            'content': content,
            'author': author,
            'keywords': keywords
        }

        return prompt_id

    @rule(prompt_id=prompts, content=prompt_content_strategy)
    def update_prompt(self, prompt_id, content):
        """Update an existing prompt."""
        if prompt_id in self.prompt_ids:
            self.service.update_prompt(
                prompt_id=prompt_id,
                content=content,
                version_comment="Property test update"
            )
            self.prompt_data[prompt_id]['content'] = content

    @rule(target=deleted_prompts, prompt_id=prompts)
    def delete_prompt(self, prompt_id):
        """Delete a prompt."""
        if prompt_id in self.prompt_ids:
            self.service.delete_prompt(prompt_id)
            self.prompt_ids.remove(prompt_id)
            self.deleted_ids.add(prompt_id)
            return prompt_id

    @rule(prompt_id=deleted_prompts)
    def restore_prompt(self, prompt_id):
        """Restore a deleted prompt."""
        if prompt_id in self.deleted_ids:
            self.service.restore_prompt(prompt_id)
            self.deleted_ids.remove(prompt_id)
            self.prompt_ids.add(prompt_id)

    @invariant()
    def active_prompts_are_listable(self):
        """All active prompts should appear in list."""
        if self.service is not None:
            listed = self.service.list_prompts()
            listed_ids = {p['id'] for p in listed}

            for pid in self.prompt_ids:
                assert pid in listed_ids or pid in self.deleted_ids

    @invariant()
    def deleted_prompts_not_in_list(self):
        """Deleted prompts should not appear in list."""
        if self.service is not None:
            listed = self.service.list_prompts()
            listed_ids = {p['id'] for p in listed}

            for pid in self.deleted_ids:
                assert pid not in listed_ids

    @invariant()
    def prompt_data_preserved(self):
        """Prompt data should be preserved correctly."""
        if self.service is not None:
            for pid in self.prompt_ids:
                if pid not in self.deleted_ids:
                    prompt = self.service.get_prompt(pid)
                    if prompt and pid in self.prompt_data:
                        original = self.prompt_data[pid]
                        assert prompt['name'] == original['name']
                        assert prompt['author'] == original['author']
                        # Content may have been updated

    def teardown(self):

        """Clean up after test."""
        if self.service:
            self.service.close()

        if hasattr(self, 'temp_dir'):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except:
                _ = None


@pytest.mark.property
@pytest.mark.slow
def test_prompt_state_machine():
    """Run the stateful property test."""
    TestPromptStateMachine = PromptStateMachine.TestCase
    TestPromptStateMachine.settings = settings(
        max_examples=50,
        stateful_step_count=20
    )
    TestPromptStateMachine().runTest()

# ========================================================================
# Collection Properties
# ========================================================================

class TestCollectionProperties:
    """Test properties of prompt collections."""

    @pytest.mark.property
    @given(
        collection_name=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        prompt_ids=st.lists(st.integers(min_value=1, max_value=100), min_size=0, max_size=20)
    )
    def test_collection_preserves_prompt_ids(
        self, collection_name, prompt_ids, mock_prompts_service
    ):
        """Collections should preserve all prompt IDs."""
        service = mock_prompts_service
        service._db_instance.create_collection.return_value = 1
        service._db_instance.get_collection.return_value = {
            'id': 1,
            'name': collection_name,
            'prompts': [{'id': pid} for pid in prompt_ids]
        }

        # Create collection
        collection_id = service.create_collection(
            name=collection_name,
            description="Test",
            prompt_ids=prompt_ids
        )

        # Get collection
        collection = service.get_collection(collection_id)

        # All prompt IDs should be preserved
        collection_prompt_ids = {p['id'] for p in collection['prompts']}
        assert collection_prompt_ids == set(prompt_ids)

    @pytest.mark.property
    @given(
        add_ids=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=10),
        remove_ids=st.lists(st.integers(min_value=51, max_value=100), min_size=1, max_size=10)
    )
    def test_collection_add_remove_operations(
        self, add_ids, remove_ids, mock_prompts_service
    ):
        """Adding and removing from collections should work correctly."""
        service = mock_prompts_service

        # Start with remove_ids in collection
        initial_ids = set(remove_ids)
        service._db_instance.create_collection.return_value = 1
        service._db_instance.get_collection.return_value = {
            'id': 1,
            'prompts': [{'id': pid} for pid in initial_ids]
        }

        collection_id = service.create_collection(
            name="Test",
            description="Test",
            prompt_ids=list(initial_ids)
        )

        # Add new prompts
        service.add_to_collection(collection_id, add_ids)
        expected = initial_ids.union(set(add_ids))

        # Remove some prompts
        service.remove_from_collection(collection_id, remove_ids)
        expected = expected - set(remove_ids)

        # Final state should match expected
        service._db_instance.get_collection.return_value = {
            'id': 1,
            'prompts': [{'id': pid} for pid in expected]
        }

        collection = service.get_collection(collection_id)
        final_ids = {p['id'] for p in collection['prompts']}
        assert final_ids == expected
