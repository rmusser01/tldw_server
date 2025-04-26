# sync_library/transport.py
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime, timezone
import requests # Example dependency for HTTP
import logging

from .models import SyncLogEntry
from .exceptions import TransportError

logger = logging.getLogger(__name__)

class SyncTransport(ABC):
    """Abstract base class for sync transport layers."""

    @abstractmethod
    def fetch_changes(self, client_id: str, since: Optional[datetime]) -> List[SyncLogEntry]:
        """
        Fetches changes from the remote endpoint since a given timestamp.

        Args:
            client_id: The ID of this client (may be needed by remote).
            since: The UTC timestamp of the last successfully processed change.
                   If None, fetch all available changes (initial sync).

        Returns:
            A list of SyncLogEntry objects received from the remote.

        Raises:
            TransportError: If fetching fails.
        """
        pass

    @abstractmethod
    def send_changes(self, client_id: str, changes: List[SyncLogEntry]) -> bool:
        """
        Sends local changes to the remote endpoint.

        Args:
            client_id: The ID of this client.
            changes: A list of SyncLogEntry objects representing local changes.

        Returns:
            True if sending was acknowledged successfully, False otherwise.

        Raises:
            TransportError: If sending fails.
        """
        pass

class HttpApiTransport(SyncTransport):
    """Example implementation using HTTP requests."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        if not base_url.endswith('/'):
            base_url += '/'
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        logger.info(f"HTTP Transport initialized for URL: {self.base_url}")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}" # Example auth
        return headers

    def fetch_changes(self, client_id: str, since: Optional[datetime]) -> List[SyncLogEntry]:
        fetch_url = f"{self.base_url}sync/changes"
        params = {"client_id": client_id}
        if since:
            # Ensure UTC and ISO format for the API
            if since.tzinfo is None or since.tzinfo.utcoffset(since) != timezone.utc.utcoffset(None):
                 since = since.astimezone(timezone.utc)
            params["since"] = since.isoformat().replace('+00:00', 'Z')

        logger.debug(f"Fetching changes from {fetch_url} with params: {params}")
        try:
            response = requests.get(fetch_url, params=params, headers=self._get_headers(), timeout=self.timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx, 5xx)

            raw_changes = response.json()
            if not isinstance(raw_changes, list):
                raise TransportError(f"Invalid response format from fetch_changes: expected list, got {type(raw_changes)}")

            parsed_changes = []
            for change_dict in raw_changes:
                try:
                    parsed_changes.append(SyncLogEntry.from_dict(change_dict))
                except (ValueError, KeyError) as e:
                     logger.error(f"Failed to parse received change dict: {change_dict}. Error: {e}")
                     # Decide: skip this change or fail the whole fetch? Let's skip.
                     continue # Skip malformed entries

            logger.info(f"Fetched {len(parsed_changes)} remote changes.")
            return parsed_changes

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed during fetch_changes: {e}", exc_info=True)
            raise TransportError(f"Failed to fetch changes: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response during fetch_changes: {e}", exc_info=True)
            raise TransportError(f"Invalid JSON response received: {e}") from e

    def send_changes(self, client_id: str, changes: List[SyncLogEntry]) -> bool:
        if not changes:
            logger.debug("No local changes to send.")
            return True

        send_url = f"{self.base_url}sync/changes"
        payload = {
            "client_id": client_id,
            "changes": [change.to_dict() for change in changes]
        }
        logger.debug(f"Sending {len(changes)} changes to {send_url} for client {client_id}")

        try:
            response = requests.post(send_url, json=payload, headers=self._get_headers(), timeout=self.timeout)
            response.raise_for_status()

            # Check response content for success confirmation if applicable
            # response_data = response.json()
            # success = response_data.get('success', False)
            # if not success:
            #    logger.warning(f"Server indicated send_changes was not fully successful: {response_data}")
            #    return False

            logger.info(f"Successfully sent {len(changes)} local changes.")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed during send_changes: {e}", exc_info=True)
            raise TransportError(f"Failed to send changes: {e}") from e
        # except json.JSONDecodeError as e: # If checking response body
        #    logger.error(f"Failed to decode JSON response during send_changes: {e}", exc_info=True)
        #    raise TransportError(f"Invalid JSON response received after sending: {e}") from e