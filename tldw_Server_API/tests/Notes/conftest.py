# Tests/conftest.py
#
#
# Imports
import sys
from pathlib import Path
import pytest
import tempfile
import shutil
#
# Third-party imports
#
# Local imports
#
############################################################################################################################
#
# Functions:

class CharactersRAGDBErrorBase(Exception):
    """Base for DB errors to make mocking easier."""
    pass

class CharactersRAGDBError(CharactersRAGDBErrorBase):
    pass

class SchemaError(CharactersRAGDBErrorBase):
    pass

class InputError(CharactersRAGDBErrorBase):
    pass

class ConflictError(CharactersRAGDBErrorBase):
    def __init__(self, message, entity=None, entity_id=None):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id

# Make these dummy exceptions available for mocking targets if needed by tests
# This assumes the tests will mock 'tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB.InputError' etc.
# If that module and its exceptions are importable, these dummies are not strictly needed for the mocks themselves,
# but can be useful for `isinstance` checks in test assertions if you don't want to import the real ones in tests.

@pytest.fixture(scope="function")
def temp_db_dir():
    """Creates a temporary directory for database files."""
    dir_path = tempfile.mkdtemp(prefix="chachadb_test_")
    yield Path(dir_path)
    shutil.rmtree(dir_path)

#
# End of Notes conftest.py
########################################################################################################################
