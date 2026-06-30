import os
import tempfile
from contextlib import contextmanager

import pytest
from hypothesis import given, settings, strategies as st

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)


@contextmanager
def _temp_chacha_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "ChaChaNotes.db")
        db = CharactersRAGDB(db_path, client_id="test")
        try:
            yield db
        finally:
            db.close_all_connections()


def test_writing_sessions_crud_and_clone():
    with _temp_chacha_db() as db:
        payload = {"prompt": "Once upon a time", "tokens": 12}
        session_id = db.add_writing_session(name="Draft One", payload=payload)
        session = db.get_writing_session(session_id)
        assert session is not None
        assert session["payload"] == payload
        assert session["version"] == 1

        listed = db.list_writing_sessions(limit=10, offset=0)
        assert any(item["id"] == session_id for item in listed)
        assert db.count_writing_sessions() == 1

        updated_payload = {"prompt": "Updated draft", "tokens": 24}
        db.update_writing_session(
            session_id,
            {
                "name": "Draft One Updated",
                "payload_json": db._serialize_writing_payload(updated_payload, "Session"),
            },
            expected_version=1,
        )
        updated = db.get_writing_session(session_id)
        assert updated is not None
        assert updated["name"] == "Draft One Updated"
        assert updated["payload"] == updated_payload
        assert updated["version"] == 2

        with pytest.raises(ConflictError):
            db.update_writing_session(
                session_id,
                {"name": "Stale Update"},
                expected_version=1,
            )

        cloned = db.clone_writing_session(session_id, name="Draft One Copy")
        assert cloned["name"] == "Draft One Copy"
        assert cloned["version_parent_id"] == session_id
        assert cloned["payload"] == updated_payload

        db.soft_delete_writing_session(session_id, expected_version=2)
        assert db.get_writing_session(session_id) is None


def test_writing_templates_and_themes_versioning():
    with _temp_chacha_db() as db:
        db.add_writing_template(name="Template A", payload={"preset": "alpha"}, is_default=False)
        template = db.get_writing_template_by_name("Template A")
        assert template is not None
        assert template["payload"] == {"preset": "alpha"}
        assert bool(template["is_default"]) is False
        assert template["version"] == 1

        db.update_writing_template(
            "Template A",
            {"is_default": True},
            expected_version=1,
        )
        updated = db.get_writing_template_by_name("Template A")
        assert updated is not None
        assert bool(updated["is_default"]) is True
        assert updated["version"] == 2

        db.soft_delete_writing_template("Template A", expected_version=2)
        assert db.get_writing_template_by_name("Template A") is None

        db.add_writing_theme(
            name="Theme A",
            class_name="theme-a",
            css=".theme-a { color: #111; }",
            order_index=3,
        )
        theme = db.get_writing_theme_by_name("Theme A")
        assert theme is not None
        assert theme["class_name"] == "theme-a"
        assert theme["css"] == ".theme-a { color: #111; }"
        assert theme["order_index"] == 3
        assert theme["version"] == 1

        db.update_writing_theme(
            "Theme A",
            {"order_index": 1},
            expected_version=1,
        )
        updated_theme = db.get_writing_theme_by_name("Theme A")
        assert updated_theme is not None
        assert updated_theme["order_index"] == 1
        assert updated_theme["version"] == 2

        db.soft_delete_writing_theme("Theme A", expected_version=2)
        assert db.get_writing_theme_by_name("Theme A") is None


def test_update_writing_session_payload_json_dict_serialized():
    with _temp_chacha_db() as db:
        session_id = db.add_writing_session(name="Draft", payload={"prompt": "Original"})
        update_payload = {"prompt": "Direct payload_json", "tokens": 5}
        db.update_writing_session(
            session_id,
            {"payload_json": update_payload},
            expected_version=1,
        )
        updated = db.get_writing_session(session_id)
        assert updated is not None
        assert updated["payload"] == update_payload
        assert updated["version"] == 2


def test_update_writing_template_payload_json_dict_serialized():
    with _temp_chacha_db() as db:
        db.add_writing_template(name="Template B", payload={"preset": "alpha"})
        update_payload = {"preset": "beta", "notes": "direct payload_json"}
        db.update_writing_template(
            "Template B",
            {"payload_json": update_payload},
            expected_version=1,
        )
        updated = db.get_writing_template_by_name("Template B")
        assert updated is not None
        assert updated["payload"] == update_payload
        assert updated["version"] == 2


_ASCII_CHARS = st.characters(min_codepoint=32, max_codepoint=126)
_PAYLOAD_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-100, max_value=100),
    st.text(alphabet=_ASCII_CHARS, max_size=20),
    st.lists(st.integers(min_value=-10, max_value=10), max_size=5),
)
_PAYLOAD_DICTS = st.dictionaries(
    keys=st.text(alphabet=_ASCII_CHARS, min_size=1, max_size=10),
    values=_PAYLOAD_VALUES,
    max_size=6,
)


@given(payload=_PAYLOAD_DICTS)
@settings(max_examples=25, deadline=None)
def test_writing_session_payload_round_trip(payload):
    with _temp_chacha_db() as db:
        session_id = db.add_writing_session(name="Round Trip", payload=payload)
        session = db.get_writing_session(session_id)
        assert session is not None
        assert session["payload"] == payload
