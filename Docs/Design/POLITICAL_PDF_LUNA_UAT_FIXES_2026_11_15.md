# Luna walkthrough: review and citation repairs

Task: TASK-15505.10. These bounded repairs follow failures reproduced in the actual Luna walkthrough; they do not expand generation, retrieval ranking or human acceptance policy.

## Review completion (UX-PDF-65)

A freshly completed one-section job has identical local/server version 6, content and section data. The generation projection omits canonical version/review metadata and review status, so the existing review comparison rejects the local artifact. Carry the canonical mapper's metadata through the generation result and final completion projection. Keep account, version and payload equality checks intact. Verify the full generation-to-review path and retain drift refusal controls. Existing cached artifacts require a safe canonical refresh; retain review drafts on failed saves.

## Verifiable citations (UX-PDF-64)

The first Luna answers correctly state the two sampled proposals but contain no inline source markers. The default RAG answer prompt supplies numbered sources without asking the model to cite them. Add that instruction to both packaged backend and frontend defaults and their shared parity fixture; preserve explicit custom templates and never invent citations after generation. The server-owned default is the primary runtime owner.

Retrieved excerpts already carry canonical page spans, while cards display only the first page and saved conversation provenance drops the spans. Validate and display the full bounded page list/range and page-specific PDF actions. Preserve optional strict page spans through the existing shared Notes/conversation provenance contract, including its Python validator and export/import consumers. Existing single-page records remain valid; malformed ranges use the existing safe fallback. Cap optional spans at 20 using existing provenance bounds and do not change source quotations or IDs.

Connected live checks require two presentation callers to follow the same contract: opening a PDF citation disclosure stays in the chat so its explicit page links can be chosen, and complete-pack `offline/chat.html` renders every validated page through its existing bundled-PDF link helper. Preserve non-PDF source focus, telemetry, legacy one-page fallback and archive escaping/identity checks.

Some vector results lack Media type/URL, preventing a direct PDF action. The existing context expansion step already loads the authorized owning Media row. Enrich returned metadata from that verified row, preserving scope checks, hidden-parent filtering and exact source-offset validation. Do not infer PDF type from names, issue additional reads, reindex embeddings, or change ranking. This reaches old and new vectors through one production owner.

## Verification and architecture

Red-first regressions cover generation-to-review, real payload drift, default/custom prompt behavior, live/saved two-page citations, malformed spans, legacy provenance, round-trip serialization and authoritative vector metadata. Run connected frontend/provider-context and Python provenance/expansion suites, type checks, targeted lint and Bandit. Re-run actual cited QA, review save/reload and complete export after the production fixes load. Preserve first failures and raw drafts.

ADR required: no new durable architecture rule. [ADR-007](../ADR/007-research-workspace-canonical-first-slice-shell.md), [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md), [ADR-049](../ADR/049-chat-history-selection-ownership.md) and [ADR-065](../ADR/065-independent-notes-knowledge-provenance.md) govern existing owners. Optional provenance fields retain the same source authority and compatibility model; no new service, storage owner or review authority is introduced.
