"""Bounded sibling diagnosis, isolated real fixtures; no pack membership required."""
import json
from pathlib import Path

from fastapi.exceptions import ResponseValidationError
from fastapi.testclient import TestClient

from tldw_Server_API.tests.StudyPacks.test_study_pack_response_timestamps import pack_api  # noqa: F401
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation


def test_real_citation_assistant_response_retains_timestamps(pack_api):
    db, _jobs, outer_client, kind = pack_api
    with chacha_operation(independent=True):
        note = db.add_note(title='Citation probe source', content='Synthetic source evidence')
        card = db.add_flashcard({'front': 'Synthetic question', 'back': 'Synthetic answer'})
        db.add_flashcard_citations(card, [{'source_type': 'note', 'source_id': note, 'citation_text': 'Synthetic source evidence'}])
    receipt = {'content_backend': kind, 'has_study_pack': False}
    try:
        with TestClient(outer_client.app, raise_server_exceptions=True) as client:
            response = client.get(f'/api/v1/flashcards/{card}/assistant')
        receipt['status'] = response.status_code
        assert response.status_code == 200
        assert response.json()['study_pack'] is None
        assert response.json()['citations'][0]['source_id'] == note
    except ResponseValidationError as exc:
        receipt['errors'] = [{'location': list(row['loc']), 'type': row['type'], 'input_type': type(row['input']).__name__} for row in exc.errors()]
        raise
    finally:
        Path(f'.tmp/uat219-repair-20260917/citation-{kind}-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
