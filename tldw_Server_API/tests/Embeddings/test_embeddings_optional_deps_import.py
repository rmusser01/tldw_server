"""Embeddings_Create must import when its optional dependencies are absent."""

import builtins
import importlib
import sys

MODULE_NAME = "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create"
PACKAGE_NAME = "tldw_Server_API.app.core.Embeddings.Embeddings_Server"


def test_embeddings_create_imports_without_optional_deps(monkeypatch):
    """Importing with onnxruntime and huggingface_hub blocked must still work.

    The re-import is undone afterwards. Importing the module with those
    dependencies blocked produces a *degraded* module object, and leaving it
    behind hands that object to everything downstream: code that imported the
    module earlier keeps the working one, code importing later gets the
    degraded one, and the two disagree about what the module's functions are.

    Leaving it behind cost four unrelated orchestrator-parity tests, which
    failed with "Embedding provider authentication failed" -- a symptom with no
    visible connection to optional imports. Both bindings are restored, because
    importlib rebinds ``sys.modules`` *and* the attribute on the parent package.
    """
    original = importlib.import_module(MODULE_NAME)
    package = importlib.import_module(PACKAGE_NAME)

    # Recorded before the re-import so teardown puts the working module back.
    monkeypatch.setitem(sys.modules, MODULE_NAME, original)
    monkeypatch.setattr(package, "Embeddings_Create", original, raising=False)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("onnxruntime", "huggingface_hub")):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    # Force a clean import so the module-level optional-dependency handling runs
    # again under the block, rather than being served from the import cache.
    sys.modules.pop(MODULE_NAME, None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module(MODULE_NAME)

    assert module is not original, "the module was served from cache, not re-imported"
    # Revision checks have to stay safe when huggingface_hub is unavailable.
    module._ensure_hf_revision("dummy/model", "deadbeef")


def test_the_degraded_import_does_not_outlive_this_module(monkeypatch):
    """Guard the restore above, which is the part that is easy to drop.

    Runs after it in file order, and would fail if the re-import leaked.
    """
    assert sys.modules[MODULE_NAME] is importlib.import_module(MODULE_NAME)
    package = importlib.import_module(PACKAGE_NAME)
    assert package.Embeddings_Create is sys.modules[MODULE_NAME], (
        "the parent package still points at a different module object than "
        "sys.modules does; a re-import was not fully undone"
    )
