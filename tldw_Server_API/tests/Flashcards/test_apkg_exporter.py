import io
import json
import sqlite3
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Flashcards.apkg_exporter import export_apkg_from_rows
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@contextmanager
def _open_sqlite_from_apkg(apkg_bytes: bytes):
    import tempfile, os

    with zipfile.ZipFile(io.BytesIO(apkg_bytes)) as zf:
        assert 'collection.anki2' in zf.namelist()
        data = zf.read('collection.anki2')
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'collection.anki2')
            with open(p, 'wb') as f:
                f.write(data)
            conn = sqlite3.connect(p)
            try:
                yield conn, zf
            finally:
                conn.close()


def _import_rows_from_apkg(apkg_bytes: bytes):
    """Importer stub for tests: reconstruct note rows from APKG."""
    with _open_sqlite_from_apkg(apkg_bytes) as (conn, zf):
        models = json.loads(conn.execute("SELECT models FROM col").fetchone()[0])
        decks = json.loads(conn.execute("SELECT decks FROM col").fetchone()[0])
        mid_to_model = {int(k): v.get('name') for k, v in models.items()}
        did_to_deck = {int(k): v.get('name') for k, v in decks.items()}

        notes_rows = conn.execute("SELECT id, mid, tags, flds FROM notes").fetchall()
        result = []
        for nid, mid, tags_str, flds in notes_rows:
            model_name = mid_to_model.get(int(mid), '')
            # find cards for this note
            card_rows = conn.execute(
                "SELECT did, ord, type, queue, due, ivl, factor FROM cards WHERE nid=? ORDER BY ord ASC",
                (nid,)
            ).fetchall()
            did = card_rows[0][0] if card_rows else None
            ords = [r[1] for r in card_rows]
            deck_name = did_to_deck.get(int(did), '') if did is not None else ''
            cards = [
                {
                    'ord': int(r[1]),
                    'type': int(r[2]),
                    'queue': int(r[3]),
                    'due': int(r[4]),
                    'ivl': int(r[5]),
                    'factor': int(r[6]),
                    'did': int(r[0]),
                    'deck_name': did_to_deck.get(int(r[0]), ''),
                }
                for r in card_rows
            ]
            fields = flds.split('\x1f')
            if model_name == 'Basic':
                front = fields[0] if len(fields) > 0 else ''
                back = fields[1] if len(fields) > 1 else ''
                extra = fields[2] if len(fields) > 2 else ''
                notes = fields[3] if len(fields) > 3 else ''
                model_type = 'basic' if ords == [0] else 'basic_reverse'
            elif model_name == 'Cloze':
                front = fields[0] if len(fields) > 0 else ''
                back = ''
                extra = fields[1] if len(fields) > 1 else ''
                notes = fields[2] if len(fields) > 2 else ''
                model_type = 'cloze'
            else:
                front = fields[0] if fields else ''
                back = ''
                extra = ''
                notes = ''
                model_type = ''
            # Parse tags (space-delimited with leading/trailing spaces)
            tags = [t for t in tags_str.split(' ') if t]
            result.append({
                'nid': nid,
                'deck_name': deck_name,
                'model_type': model_type,
                'front': front,
                'back': back,
                'extra': extra,
                'notes': notes,
                'tags': tags,
                'ords': ords,
                'cards': cards,
            })
        return result


def test_apkg_basic_reverse_two_cards():
    rows = [
        {
            "deck_name": "TestDeck",
            "model_type": "basic_reverse",
            "front": "Question?",
            "back": "Answer",
            "extra": "Extra info",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
            "tags_json": json.dumps(["test", "rev"]),
            "reverse": True,
        }
    ]
    apkg = export_apkg_from_rows(rows, include_reverse=False)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        cur = conn.execute("SELECT id,guid,mid,flds FROM notes")
        notes = cur.fetchall()
        assert len(notes) == 1
        nid = notes[0][0]

        cur = conn.execute("SELECT nid, ord FROM cards")
        cards = cur.fetchall()
        assert len(cards) == 2
        assert all(c[0] == nid for c in cards)
        ords = sorted(c[1] for c in cards)
        assert ords == [0, 1]


def test_apkg_cloze_multi_generates_multiple_cards():
    # Text contains two clozes: c1 and c2
    text = "The capital of {{c1::France}} is {{c2::Paris}}."
    rows = [
        {
            "deck_name": "Geo",
            "model_type": "cloze",
            "front": text,
            "back": "",
            "extra": "Europe",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
            "tags_json": json.dumps(["geo"]),
        }
    ]
    apkg = export_apkg_from_rows(rows)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        cur = conn.execute("SELECT id, mid, flds FROM notes")
        notes = cur.fetchall()
        assert len(notes) == 1
        nid = notes[0][0]

        cur = conn.execute("SELECT nid, ord FROM cards ORDER BY ord ASC")
        cards = cur.fetchall()
        # Expect two cards for c1 and c2, ord 0 and 1
        assert len(cards) == 2
        assert all(c[0] == nid for c in cards)
        ords = [c[1] for c in cards]
        assert ords == [0, 1]


def test_apkg_media_data_uri_extracted_and_linked():
    # Small transparent PNG (1x1) as data URI
    data_uri = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    front_html = f'<div><img src="{data_uri}" /></div>'
    rows = [
        {
            "deck_name": "Media",
            "model_type": "basic",
            "front": front_html,
            "back": "B",
            "extra": "E",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
            "tags_json": json.dumps(["media"]),
        }
    ]
    apkg = export_apkg_from_rows(rows)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        # Check media mapping exists
        media_json = zf.read('media').decode('utf-8')
        media_map = json.loads(media_json)
        # there should be at least one media entry
        assert len(media_map) >= 1
        # verify the numbered file exists in zip and has content
        first_index = sorted(media_map.keys())[0]
        fname = media_map[first_index]
        # The numbered entry in the zip is the index (e.g., '0')
        assert first_index in zf.namelist()
        assert len(zf.read(first_index)) > 0

        # Ensure the note fields reference the mapped filename, not the data URI
        cur = conn.execute("SELECT flds FROM notes")
        flds = cur.fetchone()[0]
        assert 'data:image' not in flds
        assert fname in flds


def test_apkg_round_trip_models_and_fields():
    rows = [
        {
            "deck_name": "RoundDeck",
            "model_type": "basic_reverse",
            "front": "Front A",
            "back": "Back A",
            "extra": "Extra A",
        },
        {
            "deck_name": "RoundDeck",
            "model_type": "cloze",
            "front": "Cloze {{c1::one}} and {{c2::two}}",
            "extra": "Extra Cloze",
        },
    ]
    apkg = export_apkg_from_rows(rows)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        # Check models JSON
        models_json = conn.execute("SELECT models FROM col").fetchone()[0]
        models = json.loads(models_json)
        # Should contain two models: Basic and Cloze
        assert any(m.get('name') == 'Basic' for m in models.values())
        assert any(m.get('name') == 'Cloze' for m in models.values())
        basic = next(m for m in models.values() if m.get('name') == 'Basic')
        cloze = next(m for m in models.values() if m.get('name') == 'Cloze')
        assert len(basic.get('tmpls', [])) == 2
        assert len(cloze.get('tmpls', [])) == 1

        # Check decks JSON
        decks_json = conn.execute("SELECT decks FROM col").fetchone()[0]
        decks = json.loads(decks_json)
        assert any(d.get('name') == 'RoundDeck' for d in decks.values())

        # Validate notes fields shape
        notes = conn.execute("SELECT mid, flds FROM notes ORDER BY id ASC").fetchall()
        assert len(notes) == 2
        # First is basic_reverse
        mid0, flds0 = notes[0]
        fields0 = flds0.split('\x1f')
        # Basic has 4 fields with Notes preserved for tldw exports
        assert len(fields0) == 4
        # Second is cloze
        mid1, flds1 = notes[1]
        fields1 = flds1.split('\x1f')
        assert len(fields1) == 3

        # Cards ords: expect [0,1] for basic_reverse and [0,1] for cloze
        cards = conn.execute("SELECT nid, ord FROM cards ORDER BY nid, ord").fetchall()
        # Two notes, four cards total
        assert len(cards) == 4
        # Group by nid
        nid0 = cards[0][0]
        ords0 = [o for n, o in cards if n == nid0]
        nid1 = next(n for n, _ in cards if n != nid0)
        ords1 = [o for n, o in cards if n == nid1]
        assert ords0 == [0, 1]
        assert ords1 == [0, 1]


def test_apkg_importer_stub_round_trip_content():
    rows = [
        {
            "deck_name": "RTDeck",
            "model_type": "basic",
            "front": "F0",
            "back": "B0",
            "extra": "E0",
            "tags_json": json.dumps(["t0"]) ,
        },
        {
            "deck_name": "RTDeck",
            "model_type": "cloze",
            "front": "x {{c1::one}} y {{c2::two}}",
            "extra": "EC",
            "tags_json": json.dumps(["tc"]) ,
        },
    ]
    apkg = export_apkg_from_rows(rows)
    imported = _import_rows_from_apkg(apkg)
    # Expect two notes
    assert len(imported) == 2
    basic = next(r for r in imported if r['model_type'] in ('basic', 'basic_reverse'))
    cloze = next(r for r in imported if r['model_type'] == 'cloze')
    assert basic['front'] == 'F0' and basic['back'] == 'B0' and basic['extra'] == 'E0'
    assert basic['notes'] == ''
    assert cloze['front'].startswith('x') and '{{c1::' in rows[1]['front']
    assert cloze['extra'] == 'EC'
    assert cloze['notes'] == ''
    # Cloze should have two ords [0,1]
    assert cloze['ords'] == [0, 1]


def test_apkg_export_translates_managed_asset_refs_and_preserves_notes():
    rows = [
        {
            "deck_name": "ManagedDeck",
            "model_type": "basic",
            "front": "Question ![Slide](flashcard-asset://asset-1)",
            "back": "Answer",
            "extra": "",
            "notes": "Notes ![Notes image](flashcard-asset://asset-2)",
            "tags_json": json.dumps(["managed"]),
        }
    ]

    def asset_loader(asset_uuid: str):
        return {
            "content": b"\x89PNG\r\n\x1a\nmanaged-" + asset_uuid.encode("utf-8"),
            "mime_type": "image/png",
            "original_filename": f"{asset_uuid}.png",
    }

    apkg = export_apkg_from_rows(rows, asset_loader=asset_loader)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert len(media_map) == 2

        flds = conn.execute("SELECT flds FROM notes").fetchone()[0].split("\x1f")
        assert len(flds) == 4
        assert "flashcard-asset://" not in flds[0]
        assert "flashcard-asset://" not in flds[3]
        assert "<img" in flds[0]
        assert "<img" in flds[3]


def test_apkg_export_rejects_oversized_total_media():
    rows = [
        {
            "deck_name": "TooBigDeck",
            "model_type": "basic",
            "front": "![Slide](flashcard-asset://asset-1)",
            "back": "Answer",
        }
    ]

    def asset_loader(asset_uuid: str):
        return {
            "content": b"0123456789",
            "mime_type": "image/png",
            "original_filename": f"{asset_uuid}.png",
        }

    with pytest.raises(ValueError, match="APKG media exceeds max total size"):
        export_apkg_from_rows(rows, asset_loader=asset_loader, max_total_media_bytes=4)


def test_apkg_importer_stub_scheduling_and_decks():
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    rows = [
        # New
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q",
            "back": "A",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
        },
        # Learning
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q2",
            "back": "A2",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 1,
            "lapses": 0,
            "due_at": now_iso,
        },
        # Review
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q3",
            "back": "A3",
            "ef": 2.2,
            "interval_days": 10,
            "repetitions": 3,
            "lapses": 0,
            "due_at": None,
        },
    ]
    apkg = export_apkg_from_rows(rows)
    imported = _import_rows_from_apkg(apkg)
    # Helper to classify
    def classify(card):
        if card['type'] == 0 and card['queue'] == 0:
            return 'new'
        if card['type'] == 1 and card['queue'] == 1:
            return 'learning'
        if card['type'] == 2 and card['queue'] == 2:
            return 'review'
        return 'other'
    # Deck and classification checks
    assert len(imported) == 3
    assert all(n['deck_name'] == 'SchedDeck' for n in imported)
    classes = [classify(n['cards'][0]) for n in imported]
    assert set(classes) == {'new', 'learning', 'review'}


def test_migration_v6_to_v7_adds_reverse_column(tmp_path):
    # Build a DB at v6: apply V4 full schema, then v4->v5, then v5->v6.
    db_path = tmp_path / "ChaChaNotes.db"
    conn = sqlite3.connect(str(db_path))
    try:
        # Access migration scripts from class attributes
        v4 = CharactersRAGDB._FULL_SCHEMA_SQL_V4
        v45 = CharactersRAGDB._MIGRATION_SQL_V4_TO_V5
        v56 = CharactersRAGDB._MIGRATION_SQL_V5_TO_V6
        conn.executescript(v4)
        conn.executescript(v45)
        conn.executescript(v56)
        # At this point, version should be 6 and reverse column should NOT exist
        cur = conn.execute("SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'")
        ver = cur.fetchone()[0]
        assert ver == 6
        # Reverse column should not be in pragma table_info
        cols = [r[1] for r in conn.execute("PRAGMA table_info(flashcards)")]  # 1 is column name
        assert 'reverse' not in cols
    finally:
        conn.close()

    # Now instantiate CharactersRAGDB which should migrate v6 -> v7 automatically
    db = CharactersRAGDB(str(db_path), client_id="test")
    conn2 = db.get_connection()
    try:
        ver2 = conn2.execute("SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'").fetchone()[0]
        # After instantiation, DB should migrate at least to v7 (reverse flag) and may include later versions
        assert ver2 >= 7
        cols2 = [r[1] for r in conn2.execute("PRAGMA table_info(flashcards)")]
        assert 'reverse' in cols2
    finally:
        db.close_connection()


def test_migration_to_v6_adds_modeltype_and_extra(tmp_path):
    db_path = tmp_path / "migrate_v6.db"
    conn = sqlite3.connect(str(db_path))
    try:
        v4 = CharactersRAGDB._FULL_SCHEMA_SQL_V4
        v45 = CharactersRAGDB._MIGRATION_SQL_V4_TO_V5
        v56 = CharactersRAGDB._MIGRATION_SQL_V5_TO_V6
        conn.executescript(v4)
        conn.executescript(v45)
        conn.executescript(v56)
        ver = conn.execute("SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'").fetchone()[0]
        assert ver == 6
        cols = [r[1] for r in conn.execute("PRAGMA table_info(flashcards)")]
        assert 'model_type' in cols
        assert 'extra' in cols
    finally:
        conn.close()


def test_reverse_inference_model_type_and_flag(tmp_path):
    db_path = tmp_path / "ChaChaNotes.db"
    db = CharactersRAGDB(str(db_path), client_id="test")
    try:
        # Case 1: model_type basic_reverse only
        uuid1 = db.add_flashcard({
            "deck_id": None,
            "front": "Q1",
            "back": "A1",
            "model_type": "basic_reverse",
        })
        card1 = db.get_flashcard(uuid1)
        assert card1["model_type"] == "basic_reverse"
        assert bool(card1["reverse"]) is True

        # Case 2: reverse flag only
        uuid2 = db.add_flashcard({
            "deck_id": None,
            "front": "Q2",
            "back": "A2",
            "reverse": True,
        })
        card2 = db.get_flashcard(uuid2)
        assert card2["model_type"] == "basic_reverse"
        assert bool(card2["reverse"]) is True

        # Case 3: cloze ignores reverse for model_type
        uuid3 = db.add_flashcard({
            "deck_id": None,
            "front": "This is {{c1::cloze}}",
            "back": "",
            "is_cloze": True,
            "reverse": True,
        })
        card3 = db.get_flashcard(uuid3)
        assert card3["model_type"] == "cloze"
        # reverse may be stored but exporter uses it only for basic
        assert bool(card3["reverse"]) is True
    finally:
        db.close_connection()


def test_apkg_scheduling_mapping():
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    rows = [
        # New
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q",
            "back": "A",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 0,
            "lapses": 0,
            "due_at": None,
        },
        # Learning
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q2",
            "back": "A2",
            "ef": 2.5,
            "interval_days": 0,
            "repetitions": 1,
            "lapses": 0,
            "due_at": now_iso,
        },
        # Review
        {
            "deck_name": "SchedDeck",
            "model_type": "basic",
            "front": "Q3",
            "back": "A3",
            "ef": 2.2,
            "interval_days": 10,
            "repetitions": 3,
            "lapses": 0,
            "due_at": None,
        },
    ]
    apkg = export_apkg_from_rows(rows)
    with _open_sqlite_from_apkg(apkg) as (conn, zf):
        cur = conn.execute("SELECT type, queue, due, ivl, factor FROM cards ORDER BY id ASC")
        cards = cur.fetchall()
        assert len(cards) == 3
        # New
        t0, q0, due0, ivl0, f0 = cards[0]
        assert t0 == 0 and q0 == 0
        assert due0 >= 1 and ivl0 == 0
        # Learning
        t1, q1, due1, ivl1, f1 = cards[1]
        assert t1 == 1 and q1 == 1
        assert ivl1 == 0 and due1 > 0
        # Review
        t2, q2, due2, ivl2, f2 = cards[2]
        assert t2 == 2 and q2 == 2
        assert ivl2 == 10 and 2000 <= f2 <= 3000 and due2 > 0
