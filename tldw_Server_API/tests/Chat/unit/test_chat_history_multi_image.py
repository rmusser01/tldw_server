from contextlib import contextmanager

from tldw_Server_API.app.core.Chat.chat_history import save_chat_history_to_db_wrapper
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError


class DummyDB:
    def __init__(self):
        self.client_id = "client"
        self.added_messages = []

    def get_character_card_by_name(self, name):
        return {"id": 1, "name": name}

    def add_conversation(self, conv_data):
        return "conv-1"

    @contextmanager
    def transaction(self):
        yield

    def add_message(self, payload):
        self.added_messages.append(payload)
        return "msg-1"


def test_legacy_history_persists_multiple_images():
    db = DummyDB()
    history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YQ=="},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,Yg=="},
                },
            ],
        }
    ]

    conv_id, status = save_chat_history_to_db_wrapper(
        db=db,
        chatbot_history=history,
        conversation_id=None,
        media_content_for_char_assoc=None,
        media_name_for_char_assoc=None,
        character_name_for_chat=DEFAULT_CHARACTER_NAME,
    )

    assert conv_id == "conv-1"
    assert status
    assert db.added_messages
    payload = db.added_messages[0]
    images = payload.get("images") or []
    assert len(images) == 2


class ExistingConversationDB(DummyDB):
    def __init__(self):
        super().__init__()
        self.transaction_id = 0
        self.active_transaction_id = None
        self.fetch_transaction_ids = []
        self.soft_delete_transaction_ids = []
        self.add_message_transaction_ids = []

    @contextmanager
    def transaction(self):
        self.transaction_id += 1
        previous = self.active_transaction_id
        self.active_transaction_id = self.transaction_id
        try:
            yield
        finally:
            self.active_transaction_id = previous

    def get_conversation_by_id(self, _conversation_id):
        return {"id": "conv-1", "character_id": 1, "version": 1, "title": "Existing"}

    def get_character_card_by_id(self, _character_id):
        return {"id": 1, "name": DEFAULT_CHARACTER_NAME}

    def get_messages_for_conversation(self, *_args, **_kwargs):
        self.fetch_transaction_ids.append(self.active_transaction_id)
        return [{"id": "old-1", "version": 1}]

    def soft_delete_message(self, *_args, **_kwargs):
        self.soft_delete_transaction_ids.append(self.active_transaction_id)

    def add_message(self, payload):
        self.add_message_transaction_ids.append(self.active_transaction_id)
        return super().add_message(payload)

    def update_conversation(self, *_args, **_kwargs):
        return True


def test_legacy_history_replacement_deletes_and_inserts_in_one_transaction():
    db = ExistingConversationDB()

    conv_id, status = save_chat_history_to_db_wrapper(
        db=db,
        chatbot_history=[{"role": "user", "content": "replacement"}],
        conversation_id="conv-1",
        media_content_for_char_assoc=None,
        media_name_for_char_assoc=None,
        character_name_for_chat=DEFAULT_CHARACTER_NAME,
    )

    assert conv_id == "conv-1"
    assert status == "Chat history saved successfully!"
    assert db.fetch_transaction_ids == db.soft_delete_transaction_ids
    assert db.soft_delete_transaction_ids == db.add_message_transaction_ids


class FailingReplacementDB(ExistingConversationDB):
    def __init__(self):
        super().__init__()
        self.active_messages = [{"id": "old-1", "version": 1, "deleted": False}]

    @contextmanager
    def transaction(self):
        self.transaction_id += 1
        previous_transaction = self.active_transaction_id
        message_snapshot = [message.copy() for message in self.active_messages]
        added_snapshot = list(self.added_messages)
        fetch_snapshot = list(self.fetch_transaction_ids)
        soft_delete_snapshot = list(self.soft_delete_transaction_ids)
        add_message_snapshot = list(self.add_message_transaction_ids)
        self.active_transaction_id = self.transaction_id
        try:
            yield
        except Exception:
            self.active_messages = message_snapshot
            self.added_messages = added_snapshot
            self.fetch_transaction_ids = fetch_snapshot
            self.soft_delete_transaction_ids = soft_delete_snapshot
            self.add_message_transaction_ids = add_message_snapshot
            raise
        finally:
            self.active_transaction_id = previous_transaction

    def get_messages_for_conversation(self, *_args, **_kwargs):
        self.fetch_transaction_ids.append(self.active_transaction_id)
        return [
            {"id": message["id"], "version": message["version"]}
            for message in self.active_messages
            if not message["deleted"]
        ]

    def soft_delete_message(self, message_id, _version):
        self.soft_delete_transaction_ids.append(self.active_transaction_id)
        for message in self.active_messages:
            if message["id"] == message_id:
                message["deleted"] = True
                return True
        return False

    def add_message(self, payload):
        self.add_message_transaction_ids.append(self.active_transaction_id)
        raise CharactersRAGDBError("insert failed")


def test_legacy_history_replacement_rolls_back_deletes_when_insert_fails():
    db = FailingReplacementDB()

    conv_id, status = save_chat_history_to_db_wrapper(
        db=db,
        chatbot_history=[{"role": "user", "content": "replacement"}],
        conversation_id="conv-1",
        media_content_for_char_assoc=None,
        media_name_for_char_assoc=None,
        character_name_for_chat=DEFAULT_CHARACTER_NAME,
    )

    active_message_ids = [
        message["id"]
        for message in db.active_messages
        if not message["deleted"]
    ]
    assert conv_id == "conv-1"
    assert status.startswith("Error saving messages:")
    assert active_message_ids == ["old-1"]
    assert db.added_messages == []
