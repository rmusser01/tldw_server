"""Embeddings_Create must import when its optional dependencies are absent."""

import builtins
import importlib
import sys

import pytest

MODULE_NAME = "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create"
PACKAGE_NAME = "tldw_Server_API.app.core.Embeddings.Embeddings_Server"


@pytest.mark.unit
def test_embeddings_create_imports_without_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing with onnxruntime and huggingface_hub blocked must still work.

    The re-import is undone before the test ends, and the assertions after the
    scope prove it. Importing the module with those dependencies blocked
    produces a *degraded* module object, and leaving it behind hands that object
    to everything downstream: code that imported the module earlier keeps the
    working one, code importing later gets the degraded one, and the two
    disagree about what the module's functions are.

    Leaving it behind cost four unrelated orchestrator-parity tests, which
    failed with "Embedding provider authentication failed" -- a symptom with no
    visible connection to optional imports.

    Both bindings are restored, because importlib rebinds ``sys.modules`` *and*
    the attribute on the parent package, and both are asserted against the saved
    original rather than merely against each other.

    Returns:
        None.
    """
    original = importlib.import_module(MODULE_NAME)
    package = importlib.import_module(PACKAGE_NAME)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """Stand in for __import__, refusing the two optional dependencies."""
        if name.startswith(("onnxruntime", "huggingface_hub")):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as blocked:
        # Recorded before the re-import, so leaving this scope puts the working
        # module back on both bindings.
        blocked.setitem(sys.modules, MODULE_NAME, original)
        blocked.setattr(package, "Embeddings_Create", original, raising=False)
        blocked.setattr(builtins, "__import__", guarded_import)

        # Force a clean import so the module-level optional-dependency handling
        # runs again under the block instead of being served from the cache.
        sys.modules.pop(MODULE_NAME, None)
        module = importlib.import_module(MODULE_NAME)

        assert module is not original, "served from cache, so nothing was re-imported"
        # Revision checks have to stay safe when huggingface_hub is unavailable.
        module._ensure_hf_revision("dummy/model", "deadbeef")

    assert sys.modules[MODULE_NAME] is original, (
        "the degraded module outlived this test, so every later import of "
        "Embeddings_Create resolves to it"
    )
    assert package.Embeddings_Create is original, (
        "the parent package still points at the degraded module; restoring "
        "sys.modules alone is not enough, importlib rebinds both"
    )
