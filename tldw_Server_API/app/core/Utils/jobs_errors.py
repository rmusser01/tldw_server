"""Shared Jobs value errors without loading managers or storage implementations."""


class JobPayloadDecryptionError(RuntimeError):
    """Raised when an encrypted Jobs value cannot be safely decoded."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"Encrypted job {field_name} could not be decrypted")
        self.field_name = field_name
