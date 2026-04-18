# ChaChaNotes Lifecycle-First Decomposition Design

- Date: 2026-04-17
- Project: tldw_server
- Topic: Wave 6 first-pass decomposition of `ChaChaNotes_DB.py` optimized for long-term stability and pragmatic delivery
- Mode: Design for wave-scoped implementation planning

## 1. Objective

Define the first decomposition pass for `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` in a way that improves maintainability without turning the wave into a broad rewrite.

This design must:

- reduce the operational risk around ChaChaNotes initialization, shutdown, and mixed-suite `503` behavior
- preserve the public DB contract relied on by current API surfaces
- isolate the highest-value lifecycle slice into reviewable modules
- improve developer velocity in the character/chat/session path without pretending the whole monolith is solved in one pass

## 2. Context

`ChaChaNotes_DB.py` is now a very large mixed-responsibility module that owns substantially more than the Wave 2 label suggests. The file contains character cards, conversations, messages, notes, keywords, collections, prompt presets, persona state, flashcards, moodboards, note-studio support, and other adjacent features. At the time of this design pass it is roughly 32k lines, making it one of the largest edit surfaces in the repository.

The highest-value seam is not simply "large DB file" as an abstract maintainability concern. The concrete risk path runs through:

- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- the character/conversation/message methods inside `ChaChaNotes_DB.py`

That seam is where startup, shutdown, executor reuse, cached DB instance behavior, default-character initialization, session lifecycle, restore/versioning behavior, and caller-visible `503` outcomes intersect.

The design therefore prioritizes decomposition that follows the existing runtime and endpoint contract instead of starting with generic cleanup or architecture-first extraction.

## 3. Problem Statement

The current structure creates three compounding problems:

1. Operational runtime behavior is mixed into the FastAPI dependency layer instead of being isolated as a bounded runtime concern.
2. The `CharactersRAGDB` facade owns too many unrelated feature areas, so changes in one slice are hard to reason about and easy to broaden accidentally.
3. The most important lifecycle-facing methods share a giant implementation file with unrelated domains, which raises review cost and regression risk for routine maintenance.

The first decomposition pass should fix those structural pressures only where they directly support risk reduction and delivery speed in the character/chat/session lifecycle. It should not attempt to normalize the whole file or migrate every feature domain at once.

## 4. Goals And Non-Goals

### Goals

- keep `CharactersRAGDB` as the public import target and caller-facing facade
- isolate ChaChaNotes runtime/init/shutdown/cache orchestration from the FastAPI dependency provider
- extract the character, conversation, and message lifecycle implementations behind stable facade methods
- preserve existing endpoint behavior unless a confirmed lifecycle defect requires a targeted correction
- add focused regression coverage around the extracted seams

### Non-Goals

- a full decomposition of every responsibility currently inside `ChaChaNotes_DB.py`
- schema redesign or broad persistence rewrites
- repo-wide renaming or direct caller migration to new internal modules
- refactoring notes, prompt presets, persona state, flashcards, moodboards, note-studio, or other non-lifecycle domains in the first pass
- abstract "cleanup" that does not materially improve safety or local maintainability

## 5. Approaches Considered

### Recommended: Lifecycle-First Facade-Preserving Split

Keep `CharactersRAGDB` stable as the public entrypoint, move runtime orchestration out of the FastAPI dependency layer, and extract character/conversation/message internals into a small set of focused store modules behind the existing facade.

Pros:

- aligns with the concrete Wave 2 risk seam
- minimizes caller churn
- allows strong targeted verification
- reduces file size and hidden coupling without forcing a broad rewrite

Cons:

- leaves substantial deferred scope in the remaining monolith
- requires discipline to avoid slowly pulling unrelated domains into the first pass

### Alternative: Generic Primitives First

Start by extracting transaction, connection, and backend-generic helpers before moving any feature slice.

Pros:

- conceptually clean
- can create reusable foundations early

Cons:

- broadest initial blast radius
- weakest alignment with the actual `503` and lifecycle risks
- most likely to produce cross-domain churn before behavior is stabilized

### Alternative: Notes/Prompts/Collections First

Start with a narrower non-lifecycle domain and defer the character/chat/session seam.

Pros:

- may be easier to scope in isolation
- avoids immediate contact with the busiest character chat surface

Cons:

- does not address the highest-value operational seam first
- weak fit for the approved Wave 2 objective

## 6. Locked Design Decisions

- The first decomposition pass is anchored on the character/chat/session lifecycle slice.
- `CharactersRAGDB` remains the only public DB facade in this pass.
- `ChaCha_Notes_DB_Deps.py` keeps the FastAPI dependency-provider role, but its runtime orchestration logic moves into a dedicated module under `app/core/DB_Management/chacha/`.
- Internal decomposition is limited to the slices that support lifecycle behavior directly:
  - runtime/init/shutdown/cache orchestration
  - character lifecycle
  - conversation/session lifecycle
  - message lifecycle and related metadata/citation reads
- Shared helpers move only when at least two extracted slices genuinely need them.
- Any extraction that starts pulling in notes, prompt presets, persona state, flashcards, moodboards, note studio, or similar adjacent domains is out of scope for the first pass and should be deferred.

## 7. Target Module Layout

The repository already has one useful precedent in `media_db/runtime`: responsibility-oriented internal modules behind a stable public entrypoint. This design uses that pattern lightly rather than copying its full granularity.

Recommended first-pass layout:

- `tldw_Server_API/app/core/DB_Management/chacha/runtime.py`
  - owns executor lifecycle, cache reuse, initialization coordination, shutdown behavior, health snapshotting, and default-character warmup orchestration currently living in `ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/core/DB_Management/chacha/character_store.py`
  - owns character CRUD, restore/versioning behavior, and default-character support paths currently centered in `ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
  - owns conversation/session CRUD, conversation settings, restore/search paths, and quota-relevant reads currently centered in `ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
  - owns message lifecycle, metadata, and citation-related reads currently centered in `ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/chacha/shared.py`
  - optional and strictly limited to narrowly shared helpers such as row normalization, version checks, or backend-aware query helpers that are proven common to the extracted slices

Caller rules:

- external callers continue importing and using `CharactersRAGDB`
- endpoints and deps do not import store modules directly
- stores do not depend on each other's private logic
- `shared.py` is not allowed to become a second monolith; if it starts collecting unrelated helpers, the first pass should stop rather than absorb more scope

## 8. Pass Order

The first implementation plan should follow this order:

1. Freeze current lifecycle behavior with focused tests.
2. Extract runtime/init/shutdown/cache/default-character orchestration out of `ChaCha_Notes_DB_Deps.py` with no intentional API contract changes.
3. Extract character lifecycle internals behind existing `CharactersRAGDB` methods.
4. Extract conversation/session lifecycle internals behind existing `CharactersRAGDB` methods.
5. Extract message lifecycle internals behind existing `CharactersRAGDB` methods.
6. Stop unless a clearly necessary shared helper must be factored for the extracted slices to stay coherent.

This order keeps the most failure-prone operational seam first, then moves into the feature behavior that depends on it, while preserving a clean stopping point.

## 9. Contract Boundaries

The first pass is successful only if the public contract remains substantially unchanged.

Required contract boundaries:

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` remains the public import path and facade
- public method names on the extracted lifecycle slice stay stable for callers in this pass
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` continues to expose the same dependency-provider role and caller-visible HTTP behavior
- endpoint code such as `character_chat_sessions.py` should not need to know that internal store modules now exist
- schema changes are excluded unless a confirmed lifecycle defect cannot be stabilized without one

The key architectural rule is that movement behind the facade is allowed; caller-visible contract churn is not the default tool for this wave.

## 10. Risk Controls And Stop Rules

The decomposition should stop or narrow when any of the following occurs:

- extracting a helper would require pulling in notes, persona, prompt preset, flashcard, moodboard, or other unrelated domain behavior
- a store begins depending on another store's private implementation rather than a narrow shared helper or facade context
- the shared helper layer starts accumulating unrelated logic and becoming another catch-all file
- runtime isolation alone does not fully explain the mixed-suite `503` behavior and the remaining issue looks like a separate defect rather than structural fallout
- verification burden grows beyond what can still be reviewed coherently as one wave

If one of those conditions is hit, the correct response is to split the overflow into an explicit follow-on task or follow-on wave work item rather than quietly widening the scope.

## 11. Verification Strategy

Verification should be contract-focused and targeted at the extracted seam.

Required verification areas:

- dependency/runtime tests covering repeated init, shutdown, cache reuse, executor recreation, initialization races, and deterministic `503` behavior
- focused DB tests for character lifecycle behavior, including versioning, restore semantics, and default-character handling
- focused DB tests for conversation/session/message lifecycle behavior, including settings and optimistic concurrency paths
- integration tests for `character_chat_sessions.py` proving session creation and downstream chat flows still work through the preserved `CharactersRAGDB` facade
- targeted regression tests for workspace-scoped conversation behavior where it depends on the extracted conversation slice
- Bandit against touched scope using the project virtual environment, per repo policy

Evidence threshold for a stable wave:

- the extracted runtime seam is isolated and tested
- lifecycle-facing caller behavior is unchanged except for explicitly fixed defects
- no new order-dependent failures appear in the targeted test suite
- the touched scope passes targeted security and regression checks

## 12. Definition Of Done

This design should be considered successfully implemented when:

- the ChaChaNotes runtime seam has been moved out of the FastAPI dependency layer into a reviewable core runtime module
- character, conversation, and message lifecycle implementations are isolated into focused internal modules behind the existing facade
- public contracts used by endpoints and tests remain stable
- targeted regression coverage is stronger than before in the startup/shutdown and lifecycle paths
- the remaining monolith is smaller in a way that clearly sets up the next decomposition pass without overstating completion

## 13. Implementation Planning Handoff

The next implementation plan should stay wave-scoped and avoid turning this design into a blanket monolith rewrite.

The plan should:

- treat runtime extraction as the highest-priority task because it carries the clearest operational risk
- batch character, conversation, and message extraction only as far as the verification set stays coherent
- name any deferred helpers or adjacent domains explicitly instead of leaving them as implicit stretch scope
- preserve a clear stop point after the lifecycle slice even if more decomposition opportunities are visible

The practical outcome of this design is a first-pass split that is boring for callers, safer for maintainers, and narrow enough to execute without destabilizing the broader ChaChaNotes surface.
