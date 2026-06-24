from contextlib import contextmanager

from tldw_Server_API.app.core.Chat.chat_history import save_chat_history_to_db_wrapper
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME


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
    assert db.soft_delete_transaction_ids == db.add_message_transaction_ids
