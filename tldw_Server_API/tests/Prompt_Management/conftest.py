# tests/conftest.py
# Description:
#
# Imports
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import os
import shutil
#
# Third-party imports
#
# Local imports
from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
    get_prompts_db_for_user,
    _get_prompts_db_path_for_user,
    close_all_cached_prompts_db_instances
)
from tldw_Server_API.app.api.v1.endpoints.prompts import verify_token
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.config import settings
#
########################################################################################################################
#
# Functions

@pytest.fixture(scope="session")
def test_user():
    return User(id=1, username="testuser")


@pytest.fixture(scope="session")
def test_api_token():
    # This should match what your verify_token expects if not mocked away completely
    return "fixed_test_api_token_for_pytest"


@pytest.fixture(scope="module")
def client(test_user, test_api_token):
    # --- Override dependencies ---
    def override_get_request_user():
        return test_user

    async def override_verify_token():  # Make it async if original is
        return True  # Bypass token check for tests

    fml = ("Fuck_temp_test_user_dbs_prompts_api")
    # Store original settings to restore later if necessary
    original_user_db_base_dir = fml
    temp_test_user_db_base_dir = Path("fuck_temp_test_user_dbs_prompts_api")

    def setup_test_db_environment():
        settings.USER_DB_BASE_DIR = temp_test_user_db_base_dir
        if temp_test_user_db_base_dir.exists():
            shutil.rmtree(temp_test_user_db_base_dir)
        temp_test_user_db_base_dir.mkdir(parents=True, exist_ok=True)

        # Create the specific user's prompt DB path
        # This ensures the DB is fresh for the test module
        user_prompts_dir = temp_test_user_db_base_dir / str(test_user.id) / "prompts_user_dbs"
        user_prompts_dir.mkdir(parents=True, exist_ok=True)
        db_path = user_prompts_dir / "user_prompts_v2.sqlite"

        # Initialize a clean DB for the test user for this module
        # Note: get_prompts_db_for_user in Prompts_DB_Deps will handle the actual instance creation
        # This setup here is mostly about managing the base directory for settings.
        # The caching in Prompts_DB_Deps might mean the DB instance persists if not careful.
        # `close_all_cached_prompts_db_instances()` at the end is important.

    def teardown_test_db_environment():
        close_all_cached_prompts_db_instances()  # Important
        if temp_test_user_db_base_dir.exists():
            shutil.rmtree(temp_test_user_db_base_dir)

    setup_test_db_environment()
    fastapi_app.dependency_overrides[get_request_user] = override_get_request_user
    fastapi_app.dependency_overrides[verify_token] = override_verify_token

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()
    teardown_test_db_environment()

#
# End of conftest.py
########################################################################################################################
