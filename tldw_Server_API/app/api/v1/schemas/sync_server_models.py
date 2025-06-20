# sync_server_models.py
# Description: This file contains the models used for the sync server API.
#
# Imports
from typing import List, Optional
#
# 3rd-party Libraries
from pydantic import BaseModel, Field, ConfigDict
#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Pydantic Models ---

class SyncLogEntry(BaseModel):
    """
    Represents a single entry from the sync_log table.
    Used both in client requests and server responses.
    """
    change_id: int = Field(..., description="The primary key ID of the log entry.")
    entity: str = Field(..., description="The type of entity that changed (e.g., 'Media', 'Keywords').")
    entity_uuid: str = Field(..., description="The UUID of the entity that changed.")
    operation: str = Field(..., description="The type of operation ('create', 'update', 'delete', 'link', 'unlink').")
    timestamp: str = Field(..., description="Timestamp when the change occurred (ISO 8601 format string or similar). From client for requests, potentially server authoritative for responses.")
    # Optional: Add server_timestamp if you explicitly want to send it back
    server_timestamp: Optional[str] = Field(None, description="Server's authoritative timestamp when the change was processed.")
    client_id: str = Field(..., description="The ID of the client that originated the change.")
    version: int = Field(..., description="The version number of the entity *after* this change was applied.")
    payload: str = Field(..., description="A JSON string containing the state of the entity after the change (or minimal info for deletes/links).")

    model_config = ConfigDict(
        # Example for generating schema documentation if using OpenAPI/Swagger
        json_schema_extra={
            "example": {
                "change_id": 12345,
                "entity": "Media",
                "entity_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "operation": "update",
                "timestamp": "2023-10-27T10:30:00Z",
                "client_id": "client_abc_123",
                "version": 6,
                "payload": '{"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "title": "Updated Title", "last_modified": "2023-10-27T10:30:00Z", "version": 6, "client_id": "client_abc_123", "deleted": 0, "prev_version": 5, ...}'
            }
        }
    )

class ClientChangesPayload(BaseModel):
    """
    The payload sent by a client containing its local changes.
    Corresponds to the request body for the /sync/send endpoint.
    """
    client_id: str = Field(..., description="The unique ID of the client device sending these changes.")
    changes: List[SyncLogEntry] = Field(..., description="A list of sync log entries representing local changes made on the client.")
    last_processed_server_id: int = Field(0, description="The 'change_id' of the last entry received from the server that this client successfully processed. Helps server determine delta.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "client_id": "client_xyz_789",
                "changes": [
                    {
                        "change_id": 55, # Client's local change_id
                        "entity": "Keywords",
                        "entity_uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                        "operation": "create",
                        "timestamp": "2023-10-27T11:00:00Z",
                        "client_id": "client_xyz_789",
                        "version": 1,
                        "payload": '{"uuid": "a1b2c3d4-e5f6-7890-1234-567890abcdef", "keyword": "new sync tag", ...}'
                    }
                ],
                "last_processed_server_id": 12340 # Last server change ID client processed
            }
        }
    )

class ServerChangesResponse(BaseModel):
    """
    The response sent by the server containing changes for the client.
    Corresponds to the response body for the /sync/get endpoint.
    """
    changes: List[SyncLogEntry] = Field(..., description="A list of sync log entries from the server's perspective for the requesting user, filtered to exclude changes originating from the requesting client.")
    latest_change_id: int = Field(..., description="The highest 'change_id' currently present in the user's sync log on the server. Used by the client to know the server's current state.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "changes": [
                     {
                        "change_id": 12346, # Server's change_id
                        "entity": "Media",
                        "entity_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                        "operation": "update",
                        "timestamp": "2023-10-27T10:35:00Z", # Server's authoritative timestamp
                        "client_id": "client_other_456", # Change originated from another client
                        "version": 7,
                        "payload": '{"uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "title": "Server Processed Title", ...}'
                    }
                ],
                "latest_change_id": 12350 # Highest ID on server for this user
            }
        }
    )

#
# End of sync_server_models.py
#######################################################################################################################
