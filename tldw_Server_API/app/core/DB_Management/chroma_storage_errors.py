"""Source-free failures shared by Chroma persistence and SDK adapters."""


class ChromaStorageError(RuntimeError):
    """Source-free storage failure; absence is a separate, proven condition."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
