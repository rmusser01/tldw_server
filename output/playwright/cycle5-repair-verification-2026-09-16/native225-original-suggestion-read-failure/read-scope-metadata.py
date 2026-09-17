"""Read only the owned UAT administrator's scope metadata; never emit credentials."""
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.umask(0o077)
logging.disable(logging.CRITICAL)
from loguru import logger
logger.remove()

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("owned_metadata", root / ".tmp/uat181-full-native-20260917/all-table-metadata.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
profile = module.read_private(module.PRIVATE / "pg-multi.profile.private.json")
os.environ.update(module.read_private(Path(profile["backendEnvPath"])))
from dotenv import load_dotenv
load_dotenv(profile["envPath"], override=True)
sys.path[:0] = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]

from tldw_Server_API.app.core.Sync.v2.factory import sync_v2_storage_exists_for_user
storage_exists = sync_v2_storage_exists_for_user("1")
target, fingerprint = module.content_target()
from psycopg.conninfo import make_conninfo
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
backend = DatabaseBackendFactory.create_backend(DatabaseConfig(
    backend_type=BackendType.POSTGRESQL,
    connection_string=make_conninfo(host=target["host"], port=str(target["port"]), dbname=target["database"], user=target["user"], password=target["password"], connect_timeout="5", options="-c default_transaction_read_only=on -c statement_timeout=3000"),
    pool_size=1, max_overflow=0, pool_timeout=5, echo=False,
))
try:
    assert backend.execute("SELECT current_setting('transaction_read_only') AS mode").rows[0]["mode"] == "on"
    counts = backend.execute(
        "SELECT COUNT(*) AS owner_scopes, COUNT(*) FILTER (WHERE dataset_id = %s) AS legacy_scopes FROM note_task_scope_authority WHERE owner_user_id = %s",
        ("legacy:1", "1"),
    ).rows[0]
    result = {"at": datetime.now(timezone.utc).isoformat(), "profile": "pg-multi", "ownerId": "1", "databaseIdentitySha256": fingerprint, "readOnlyVerified": True, "syncStorageExists": storage_exists, "ownerScopeCount": int(counts["owner_scopes"]), "legacyScopeCount": int(counts["legacy_scopes"]), "limits": "Read-only metadata and current profile-derived Sync storage lookup, not interpreter introspection."}
    with (Path(__file__).parent / "admin-scope-metadata.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))
finally:
    backend.get_pool().close_all()
