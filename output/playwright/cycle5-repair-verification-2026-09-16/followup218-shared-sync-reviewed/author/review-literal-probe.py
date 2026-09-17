"""Independent UAT218 guard probe on an official temporary PostgreSQL database."""
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


def operation_checks(backend):
    with backend.transaction() as conn:
        return backend.execute(
            "SELECT conname,pg_get_constraintdef(oid) AS definition FROM pg_constraint "
            "WHERE conrelid='sync_log'::regclass AND contype='c' ORDER BY conname",
            connection=conn,
        ).rows


@pytest.mark.parametrize('shape', ['embedded-space', 'trailing-space'])
def test_unknown_operation_literal_is_not_rewritten_on_reopen(pg_database_config, tmp_path, shape):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(str(tmp_path / 'media.db'), client_id='2', backend=backend)
    reopened = None
    try:
        media.close_connection()
        with backend.transaction() as conn:
            for row in operation_checks(backend):
                name = backend.escape_identifier(row['conname'])
                backend.execute(f'ALTER TABLE sync_log DROP CONSTRAINT {name}', connection=conn)
            statements = {
                'embedded-space': "ALTER TABLE sync_log ADD CONSTRAINT custom_operation CHECK (operation IN ('cre ate','update','delete'))",
                'trailing-space': "ALTER TABLE sync_log ADD CONSTRAINT custom_operation CHECK (operation IN ('create ','update','delete'))",
            }
            backend.execute(statements[shape], connection=conn)
        before = operation_checks(backend)
        error = None
        try:
            reopened = MediaDatabase(str(tmp_path / 'media.db'), client_id='2', backend=backend)
        except DatabaseError as exc:
            error = exc
        after = operation_checks(backend)
        Path(f'.tmp/uat218-independent-20260917/operation-literal-{shape}-receipt.json').write_text(
            json.dumps({'before': before, 'after': after, 'reopen_rejected': error is not None}, indent=2) + '\n'
        )
        assert error is not None, 'Unknown literal was accepted and its constraint rewritten'
        assert after == before
    finally:
        if reopened is not None:
            reopened.close_connection()
        media.close_connection()
        backend.get_pool().close_all()
