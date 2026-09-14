# Moderation Shared Models Extraction Design

**Date:** 2026-08-01

**Status:** Approved

**Backlog:** TASK-13010

**Predecessor:** TASK-12992, merged by PR #2770; original stacked head `285773ea6b05318512f4b004375e394e969e425d`

## Summary

Move `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` from
`moderation_service.py` into a neutral canonical module at
`tldw_Server_API/app/core/Moderation/models.py`.

This is a structural refactor. Runtime moderation decisions, redaction, the
mapping returned by `ModerationPolicy.to_dict()`, exception behavior, caller
imports, and public API contracts remain unchanged. `moderation_service.py`
continues to expose the three names as compatibility re-exports of the exact
canonical class objects.

The only intentional observable metadata change is each moved class's
`__module__`, which becomes
`tldw_Server_API.app.core.Moderation.models`. Existing qualified names under
`moderation_service.py` continue to resolve to the same classes.

## Context

The Moderation refactor has already extracted deterministic policy assembly
into `PolicyCompiler` and policy evaluation/redaction into `PolicyEvaluator`.
The three shared dataclasses remain defined by `moderation_service.py`, so both
extracted components use deferred imports through `policy_types()` to avoid a
cycle:

```text
moderation_service.py -> policy_compiler.py
moderation_service.py -> policy_evaluator.py
policy_compiler.py    -> moderation_service.py (deferred)
policy_evaluator.py   -> moderation_service.py (deferred)
```

Moving only the shared dataclasses to a neutral module removes the import cycle
without changing the service facade or migrating established callers.

## Goals

1. Make `models.py` the canonical owner of exactly:
   - `ModerationPolicy`
   - `PatternRule`
   - `ModerationEvaluationResult`
2. Preserve imports of those names from `moderation_service.py` with exact
   class identity.
3. Let `PolicyCompiler` and `PolicyEvaluator` import the models directly under
   private runtime aliases while retaining type-checking-only annotation names.
4. Retain both `policy_types()` static methods and their internal dynamic
   dispatch while replacing deferred imports with direct canonical references.
5. Preserve dataclass constructors, fields, defaults, factories, equality,
   mutability, annotations, and `ModerationPolicy.to_dict()` behavior.
6. Prove that `models.py` imports only approved standard-library modules and
   does not load configuration, the service, the compiler, or the evaluator
   beyond unavoidable parent-package initialization.
7. Keep the change reviewable as a separate pull request after the
   `PolicyEvaluator` predecessor lands.

## Non-Goals

- No moderation decision, scan, snippet, count, or redaction behavior changes.
- No regex or ReDoS hardening.
- No endpoint, schema, configuration, persistence, logging, or caller contract
  changes.
- No migration of established production callers from the service import path.
- No removal of `policy_types()` or private `ModerationService` delegates.
- No move of compiler-specific contracts, `EvaluationLimits`, or
  `_ResolvedModerationServiceState`.
- No package-level re-exports from `Moderation/__init__.py`.
- No broad cleanup of nearby imports, annotations, exception tuples, or model
  APIs.

## Canonical Module

Create:

```text
tldw_Server_API/app/core/Moderation/models.py
```

The module begins with `from __future__ import annotations`. This is required
because `ModerationPolicy` retains its position before `PatternRule` and must
preserve its current string-based annotation representation.

The module owns the three dataclasses and the private values required by
`ModerationPolicy.to_dict()`. Apart from the future import, it imports only
these standard-library modules:

- `dataclasses`
- `json`
- `re`

It must not import project configuration, Loguru, `moderation_service`,
`policy_compiler`, or `policy_evaluator`.

The classes retain their current declaration order and definitions. In
particular:

- `ModerationPolicy.block_patterns` remains a fresh list per instance.
- `categories_enabled` and rule category sets retain their current shallow
  alias behavior.
- regex objects are borrowed without copying or recompilation.
- `ModerationEvaluationResult` remains mutable.
- no validation or normalization is added to constructors.

## Dependency Direction

After extraction, the dependency graph is acyclic:

```text
models.py
  ^       ^
  |       |
policy_compiler.py   policy_evaluator.py
          ^           ^
          |           |
          moderation_service.py
```

`moderation_service.py` imports the canonical classes at module scope before
using them. `PolicyCompiler` and `PolicyEvaluator` import the same classes from
`models.py` in two deliberately separate forms:

- normal names only inside `if TYPE_CHECKING:` for static annotations
- private runtime aliases such as `_ModerationPolicy` for `policy_types()`

The private aliases avoid adding `ModerationPolicy`, `PatternRule`, or
`ModerationEvaluationResult` to the compiler/evaluator public runtime
namespaces. This preserves current star-import behavior and the current
inability of runtime `typing.get_type_hints()` to resolve those
type-checking-only names. The new underscore-prefixed aliases remain visible
through direct private-name introspection such as `dir()` or `module.__dict__`;
private module metadata is outside the compatibility guarantee.

Existing production callers continue importing models from
`moderation_service.py`. This deliberately proves the compatibility facade and
keeps the pull request focused.

## Service Compatibility Re-Exports

`moderation_service.py` removes the three local class bodies and imports:

```python
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
```

Because the imported names remain module globals, existing code continues to
work:

```python
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
```

The compatibility contract requires:

```python
moderation_service.ModerationPolicy is models.ModerationPolicy
moderation_service.PatternRule is models.PatternRule
moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
```

No wrapper classes, aliases with separate identities, subclass shims, lazy
module attributes, or `__getattr__` hooks are introduced.

## `policy_types()` Compatibility

`PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` currently
form observable static callable boundaries. Existing tests lock the evaluator
descriptor and tuple, and internal code dynamically calls the methods.

Both methods remain `staticmethod` descriptors with unchanged signatures and
tuple ordering. Their bodies return the privately aliased canonical classes
instead of importing `moderation_service.py` at call time.

Internal compiler and evaluator methods continue calling `self.policy_types()`
or the existing equivalent dispatch. This preserves subclass and interception
behavior. Removing the methods or bypassing their internal dispatch is a
separate compatibility decision and is out of scope.

### Service-export rebinding boundary

Today, rebinding `moderation_service.ModerationPolicy`, `PatternRule`, or
`ModerationEvaluationResult` changes subsequent `policy_types()` results
because the methods resolve those names from the service at call time. After
extraction, `policy_types()` always returns the canonical classes from
`models.py`.

Runtime rebinding or monkeypatching of the service compatibility exports is
explicitly outside the compatibility guarantee. Preserving that interception
would retain the service dependency that this extraction removes. Subclass
overrides of `policy_types()` and internal dynamic dispatch through
`self.policy_types()` remain supported and tested.

## `ModerationPolicy.to_dict()`

`ModerationPolicy.to_dict()` remains an instance method with identical output,
ordering, fallbacks, and exception boundaries.

The current implementation reads two private service-module globals:

- `_MODERATION_NONCRITICAL_EXCEPTIONS`
- `ModerationService._UNCATEGORIZED_CATEGORY`

The neutral model cannot retain those dependencies without loading the service.
`models.py` therefore owns:

- a private model-local noncritical exception tuple containing exactly
  `OSError`, `ValueError`, `TypeError`, `KeyError`, `RuntimeError`,
  `AttributeError`, `ConnectionError`, `TimeoutError`,
  `json.JSONDecodeError`, and `re.error`
- a private uncategorized-category constant with the same value,
  `"uncategorized"`

The service keeps its existing private exception tuple for service-owned
behavior. The model-local tuple is an intentional narrow duplication used only
by `ModerationPolicy.to_dict()`; moving the service tuple into `models.py` would
give a neutral data module service-level error-handling ownership.

The implementation otherwise remains literal. It does not tighten error
handling, improve malformed objects, change category sorting, or normalize
rules.

### Private interception boundary

Monkeypatching private service globals will no longer alter `to_dict()` after
the move. That private interception behavior is explicitly outside the
compatibility guarantee because preserving it would require a runtime import
from the neutral model back into the service.

Actual `to_dict()` outputs and exception/fallback behavior remain covered by
characterization tests.

## Module Metadata

The canonical classes use their natural module metadata:

```python
ModerationPolicy.__module__ == "tldw_Server_API.app.core.Moderation.models"
```

The same applies to `PatternRule` and `ModerationEvaluationResult`.

The implementation must not override `__module__`. Such an override would make
the neutral classes appear service-owned, complicate models-only annotation
resolution, and preserve the old dependency conceptually.

Legacy qualified-name resolution remains supported because importing
`moderation_service.py` exposes the exact canonical classes. Repository search
found no production persistence of these classes with `pickle`.

The serialization compatibility guarantee applies only to the mapping returned
by `ModerationPolicy.to_dict()`, not arbitrary JSON encoders or byte-for-byte
pickle output. Existing pickles that name
`moderation_service.ModerationPolicy`, `PatternRule`, or
`ModerationEvaluationResult` remain loadable by the new release because those
qualified names still resolve through the service facade. Pickles produced by
the new release name `Moderation.models` and cannot be loaded by an older
release that does not contain that module. This forward-to-old-release
asymmetry is an accepted consequence of the approved natural `__module__`
change. The design does not add a pickle format or execute pickle payloads in
tests.

## Import Isolation

A source-level AST test parses `models.py` without importing the package. It
allows imports rooted only at:

- `__future__`
- `dataclasses`
- `json`
- `re`

This gate proves that `models.py` itself does not import Loguru or any project
module.

A clean-process runtime test first imports the parent package
`tldw_Server_API.app.core.Moderation`, records the package-initialization
baseline, and confirms that these modules are absent:

- `tldw_Server_API.app.core.config`
- `tldw_Server_API.app.core.Moderation.moderation_service`
- `tldw_Server_API.app.core.Moderation.policy_compiler`
- `tldw_Server_API.app.core.Moderation.policy_evaluator`

It then imports `tldw_Server_API.app.core.Moderation.models` and confirms those
modules remain absent. The runtime test intentionally does not assert that
`loguru` is absent: Python executes `tldw_Server_API/__init__.py` and
`tldw_Server_API/app/__init__.py` first, and those existing parent initializers
load Loguru independently of `models.py`.

Two additional clean-process tests prove removal of the deferred service
edges:

1. import `policy_compiler`, confirm `moderation_service` is absent, call
   `PolicyCompiler.policy_types()`, assert the canonical tuple and confirm the
   service remains absent
2. import `policy_evaluator`, confirm `moderation_service` is absent, call
   `PolicyEvaluator.policy_types()`, assert the canonical tuple and confirm the
   service remains absent

Further clean-process tests verify these complete import orders:

1. models, then service, compiler, evaluator
2. compiler, then service
3. evaluator, then service
4. service, then compiler and evaluator

Each order must exit successfully and resolve all legacy/canonical class names
to the same objects. Compiler/evaluator module assertions also confirm that the
public model names remain absent from their runtime namespaces.

## Dataclass Compatibility

Pre-extraction characterization locks the current service-owned classes before
production code moves. Coverage includes:

- exact `inspect.signature()` output
- `dataclasses.fields()` order
- each default and default factory
- fresh-list behavior for `block_patterns`
- shallow alias behavior for supplied lists and sets
- equality and mutability
- `__annotations__` and resolved type hints where stable
- regex identity
- `ModerationPolicy.to_dict()` for ordinary rules, legacy regex-like objects,
  uncategorized rules, malformed objects, and fallback paths
- exact `policy_types()` descriptors, tuple order, and internal dispatch

After extraction, canonical-module tests assert the same contracts plus exact
identity through the service import path. Assertions account explicitly for
the approved `__module__` change and must not freeze unrelated implementation
details. Separate compatibility-boundary tests assert that:

- rebinding service facade exports does not change canonical `policy_types()`
  tuples
- subclass overrides of `policy_types()` still control internal dynamic
  dispatch
- compiler/evaluator runtime modules do not expose public model names

## Test Strategy

### Stage 1: Pre-move characterization

Add a focused characterization file that imports the existing classes from
`moderation_service.py`. Run it green before adding `models.py`.

The characterization is an independent oracle. It must not import or duplicate
future implementation helpers.

### Stage 2: Canonical module and import graph

Add `models.py`, replace the service definitions with imports, and switch
compiler/evaluator type imports to `models.py`, using private aliases for
runtime references. Preserve `policy_types()` and its internal dispatch.

Add the source import allowlist, canonical identity, deferred-edge removal,
runtime namespace, and clean-process import-order tests.

### Stage 3: Focused compatibility gates

Run these exact suites:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation*.py -q
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py \
  tldw_Server_API/tests/Guardian/test_supervised_policy.py \
  -q
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  -q
python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  -k moderation_adapter \
  -q
python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled \
  -q
```

The first command already includes the endpoint sample; the explicit second
command intentionally pairs that public surface with Guardian compatibility.

### Static and security gates

- `py_compile` every touched Python file
- Black check for every clean touched Python file
- Ruff over every clean touched file, documenting only pre-existing ignores
- Bandit over `tldw_Server_API/app/core/Moderation`
- `git diff --check`
- clean worktree check
- production-scope audit
- current-`dev` mergeability check
- independent whole-branch review

## Production Scope

Production changes are limited to:

- add `tldw_Server_API/app/core/Moderation/models.py`
- modify `tldw_Server_API/app/core/Moderation/moderation_service.py`
- modify `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- modify `tldw_Server_API/app/core/Moderation/policy_evaluator.py`

Test changes are limited to focused model characterization, canonical identity,
import isolation/order, compiler/evaluator compatibility, and the named caller
regressions if an existing assertion needs to recognize the canonical module.

No endpoint or schema production file changes are permitted.

## Stacked Branch Rollout

The design branch starts from the verified predecessor head:

```text
285773ea6b05318512f4b004375e394e969e425d
```

The predecessor later merged through PR #2770. The original stacked branch is
retained as a recovery reference; the PR branch is created directly from
current `origin/dev` and receives only the commits after this boundary.

After the predecessor merges, fetch current `dev` and transplant only commits
created after the recorded predecessor:

```bash
git fetch origin dev
git cherry-pick --no-commit <post-predecessor-commits>
```

The consolidated transplant also replaces stale colliding Backlog records with
the current collision-free TASK-13010 and TASK-13011 records.

Before opening the models pull request:

1. verify `git merge-base HEAD origin/dev` is current `origin/dev`
2. inspect `git log origin/dev..HEAD`
3. inspect `git diff --name-status origin/dev...HEAD`
4. confirm no predecessor-only commits or unrelated production files remain
5. rerun all verification gates on the transplanted head

This procedure is required even if the predecessor was squash-merged.

## Rollback

Rollback is a normal pull-request revert. The service remains the established
facade, so reverting restores local class ownership without caller migration,
database repair, or configuration changes.

## Risks And Mitigations

### Accidental duplicate class identities

Risk: copied definitions remain in the service or wrapper subclasses are added.

Mitigation: remove the service class bodies and assert `is` identity across
both module paths.

### Hidden import cycle

Risk: `models.py` imports a project module that eventually imports the service.

Mitigation: an AST import allowlist plus package-baseline-aware clean-process
`sys.modules` assertions and import-order tests.

### `to_dict()` behavior drift

Risk: neutralizing service globals changes output, ordering, or fallback scope.

Mitigation: pre-move literal characterization and exact post-move assertions.

### Compatibility-shim erosion

Risk: direct imports lead implementation code to bypass or remove
`policy_types()` in the same pull request.

Mitigation: descriptor, tuple, and internal-dispatch tests remain binding.

### Accidental runtime namespace expansion

Risk: normal runtime imports expose model names from `policy_compiler.py` or
`policy_evaluator.py`, changing star imports, the public module namespace, and
annotation resolution.

Mitigation: private runtime aliases, type-checking-only public annotation
names, and clean-process namespace assertions.

### Scope contamination from the predecessor

Risk: a stacked branch opened directly against `dev` includes the full
`PolicyEvaluator` extraction.

Mitigation: record the exact predecessor SHA, use `rebase --onto`, and audit the
post-transplant commit and file ranges before PR creation.

## Follow-Up Work

Separate reviewed tasks may later:

- evaluate removal of `policy_types()` after repository and external usage are
  understood
- migrate selected internal callers to canonical model imports if that provides
  measurable value
- remove private `ModerationService` delegates after compatibility review
- harden long-text regex execution and redaction guardrails

None of those changes belong in this extraction.
