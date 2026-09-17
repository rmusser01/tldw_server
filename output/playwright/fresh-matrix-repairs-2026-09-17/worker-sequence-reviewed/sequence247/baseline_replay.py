"""Disposable process-only replay of the original sequence helper."""
from pathlib import Path
import runpy
import pytest
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

baseline = runpy.run_path(str(Path(__file__).with_name('baseline_postgres_sequence_maintenance.py')))

@pytest.fixture(autouse=True)
def old_sequence_helper(monkeypatch, request):
    monkeypatch.setattr(MediaDatabase, '_sync_postgres_sequences', baseline['sync_postgres_sequences'])
    if hasattr(request.module, 'sync_postgres_sequences'):
        monkeypatch.setattr(request.module, 'sync_postgres_sequences', baseline['sync_postgres_sequences'])
