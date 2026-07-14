"""Core deep research primitives."""

from importlib import import_module

__all__ = [
    "ResearchArtifactStore",
    "ResearchBroker",
    "ResearchLimits",
    "ResearchSynthesizer",
    "apply_checkpoint_patch",
    "build_initial_plan",
    "ensure_limit_available",
]

_EXPORTS = {
    "ResearchArtifactStore": (".artifact_store", "ResearchArtifactStore"),
    "ResearchBroker": (".broker", "ResearchBroker"),
    "ResearchLimits": (".limits", "ResearchLimits"),
    "ResearchSynthesizer": (".synthesizer", "ResearchSynthesizer"),
    "apply_checkpoint_patch": (".checkpoint_service", "apply_checkpoint_patch"),
    "build_initial_plan": (".planner", "build_initial_plan"),
    "ensure_limit_available": (".limits", "ensure_limit_available"),
}
_SUBMODULES = frozenset(
    {
        "artifact_store",
        "broker",
        "checkpoint_service",
        "limits",
        "models",
        "planner",
        "providers",
        "synthesizer",
    }
)


def __getattr__(name: str) -> object:
    """Resolve legacy public exports without importing unrelated modules."""
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery."""
    return sorted({*globals(), *__all__, *_SUBMODULES})
