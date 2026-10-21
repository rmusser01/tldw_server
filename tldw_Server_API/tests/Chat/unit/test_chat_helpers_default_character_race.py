"""Concurrent first requests both try to create the default character (TASK-13861)."""

import asyncio
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Character_Chat.constants import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Chat.chat_helpers import get_or_create_character_context
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, ConflictError

pytestmark = pytest.mark.unit


async def test_the_request_that_loses_the_insert_race_reads_the_winners_default_character():
    winner = {"id": 7, "name": DEFAULT_CHARACTER_NAME}
    db = MagicMock()
    db.client_id = "1"
    # Fallback read, read inside the creator, then the read after the conflict.
    db.get_character_card_by_name.side_effect = [None, None, winner]
    db.add_character_card.side_effect = ConflictError("Character card already exists.", entity="character_cards")

    card, card_id = await get_or_create_character_context(db, None, asyncio.get_running_loop())

    assert (card, card_id) == (winner, 7)


async def test_a_failed_read_back_after_the_conflict_returns_no_character():
    db = MagicMock()
    db.client_id = "1"
    db.get_character_card_by_name.side_effect = [None, None, CharactersRAGDBError("connection lost")]
    db.add_character_card.side_effect = ConflictError("Character card already exists.", entity="character_cards")

    assert await get_or_create_character_context(db, None, asyncio.get_running_loop()) == (None, None)
