# Document Insights Service Prompt

Approved scope: TASK-13198, continuing the incremental Service Prompts rollout.

Register `media.document.insights` in the existing catalog and shared WebUI /
extension Settings. Expose literal analysis guidance (role and category guidance)
and presentation guidance (relevance, title and content style). Keep the JSON
envelope instruction, JSON-only instruction, document carrier and requested
category carrier fixed. Assemble the original default messages byte-for-byte.
Keep provider/model configuration, content limits and typed output normalization.

Resolve one immutable configuration using authenticated-owner prompt storage,
reading and closing its connection on the same worker. Resolution errors must
not fall back to cached results or silently use defaults. Include a fingerprint
of the assembled system prompt in the existing owner/database-scoped cache key;
edits and reset cannot retrieve results produced with different guidance. An
in-flight request retains its original prompt and cache key even if Settings
changes while the model is running. No new cache, storage table or endpoint.

Verify public HTTP/model-facing behavior with real prompt storage and cache
serialization: defaults, save/reset, owner isolation, mid-request edits, cache
hits/misses, provider controls, content/category carriers and malformed output.
Reuse the shared Settings editor and tests; no new component or dependency.
