from typing import get_type_hints

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module


def test_connection_lifecycle_helpers_use_named_raw_connection_alias() -> None:
    alias = getattr(chacha_module, "RawBackendConnection", None)

    assert alias is not None

    release_hints = get_type_hints(
        chacha_module.CharactersRAGDB._release_connection,
        globalns=vars(chacha_module),
    )
    open_hints = get_type_hints(
        chacha_module.CharactersRAGDB._open_new_connection,
        globalns=vars(chacha_module),
    )

    assert release_hints["connection"] == alias
    assert open_hints["return"] == alias
