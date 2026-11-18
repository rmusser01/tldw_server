# Complete embedding input coverage — TASK-15505.7

## Confirmed defect and scope
The media Jobs and direct paths use word chunks (default1000). Local HuggingFace inference silently truncates to configured max_length (currently512). Actual PDF431/431 chunks exceed512tokens. Even128words leaves272 oversized URL/reference chunks. A ready index therefore represents only prefixes.

Keep existing APIs, Jobs and one input/one vector semantics. Refine media chunks before batching using the actual resolved local-HF tokenizer, revision and finite configured/model token limit. Special tokens and instruction prefixes count. Split original strings by character offsets, not decoded tokens; retain all characters and parent metadata, assign consecutive child indexes, and persist the refined artifacts/counts. Recompute if fallback changes model. Unknown/invalid limits fail explicitly; no guessed token ratios or silent truncation.

Raw embedding callers cannot receive multiple vectors for one input; reject oversized inputs before cache/inference instead. Existing permitted model loading and revision rules apply. Remote providers retain their current behavior. No new dependency or UI option.

## Architecture assessment
Reuse ADR-003 Jobs execution and existing embedding/provider configuration; ADR-053 still governs retrieval fusion. This repairs input fidelity within existing boundaries and introduces no new durable architecture rule, so no new ADR is required.

## Validation
Frozen tests cover dense URLs, a long single word, Unicode, exact offsets/complete character coverage, special/prompt overhead, finite-limit failure, Jobs/direct/fallback persistence, and strict raw input rejection. Property tests check partition coverage and bounds with a deterministic tokenizer; an offline actual cached MiniLM tokenizer audit verifies the real PDF without provider calls. Root owns subsequent real indexing and frozen retrieval replay.

## Implemented recovery and verification
Raw calls use the existing nonretryable input_too_long domain error/HTTP400 before circuit breaker, provider work, or padded tensor allocation. HF cache keys have a new complete-input policy marker. Legacy HF job embedding checkpoints are recomputed; a storage-only retry without verified coverage is refused with regeneration guidance. Existing stored indexes require explicit regeneration; there is no background migration.

Offline audit against canonical MediaDB text (2,620,220characters), actual cached MiniLM tokenizer and existing431wordchunks: refinement produced2261exact canonical substrings, all at most512tokens, and reconstructed every originalchunk completely. The model package recommends256tokens for SentenceTransformer use; this runtime uses AutoModel with the configured/tokenizer512limit. No model-global default or model identity changed.
