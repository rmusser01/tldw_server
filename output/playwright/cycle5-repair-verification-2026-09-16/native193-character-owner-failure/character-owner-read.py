"""Read only the owned UAT character identity fields through the existing backend."""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

helper_path = Path('.tmp/uat181-full-native-20260917/all-table-metadata.py').resolve()
spec = importlib.util.spec_from_file_location('uat181_metadata_helper', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def main():
    target, fingerprint = helper.content_target()
    from psycopg.conninfo import make_conninfo
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
    conninfo = make_conninfo(host=target['host'], port=str(target['port']), dbname=target['database'], user=target['user'], password=target['password'], connect_timeout='5', application_name='uat193_owner_diagnostic', options='-c default_transaction_read_only=on -c statement_timeout=3000')
    backend = DatabaseBackendFactory.create_backend(DatabaseConfig(backend_type=BackendType.POSTGRESQL, connection_string=conninfo, pool_size=1, max_overflow=0, pool_timeout=5, echo=False))
    try:
        identity = backend.execute("SELECT current_database() AS name, current_setting('transaction_read_only') AS read_only").rows[0]
        if identity['name'] != target['database'] or identity['read_only'] != 'on':
            raise ValueError('connection_boundary')
        rows = backend.execute('SELECT id, name, client_id, deleted FROM character_cards WHERE id IN (?, ?) ORDER BY id', (1, 2)).rows
        return {'at':datetime.now(timezone.utc).isoformat(), 'databaseSha256':fingerprint, 'readOnlyVerified':True, 'characters':[dict(row) for row in rows]}
    finally:
        backend.get_pool().close_all()

if __name__ == '__main__':
    try:
        print(json.dumps(main(), indent=2))
    except Exception as exc:
        print(json.dumps({'ok':False,'errorType':type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
