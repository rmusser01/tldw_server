# /app/services/ephemeral_store.py

# FIXME - File is dummy code, needs to be updated
import uuid

class EphemeralStorage:
    def __init__(self):
        self._store = {}

    def store_data(self, data):
        ephemeral_id = str(uuid.uuid4())
        self._store[ephemeral_id] = data
        return ephemeral_id

    def get_data(self, ephemeral_id):
        return self._store.get(ephemeral_id)

    def remove_data(self, ephemeral_id):
        if ephemeral_id in self._store:
            del self._store[ephemeral_id]

# Single global instance (thread-safety depends on your usage):
ephemeral_storage = EphemeralStorage()
