# Knowledge QA Guide

Knowledge QA is the `/knowledge` workflow for asking questions against your personal library and reviewing grounded answers with citations. Use it when you want a cited answer from indexed documents, media, notes, conversations, task boards, or other selected Knowledge QA source categories.

Knowledge QA is not the flashcards workflow. Flashcards are handled by the separate flashcards route and are not part of `/knowledge`.

## Where To Open It

- WebUI: open `/knowledge`.
- Browser extension: open the extension options page and choose Knowledge QA.

The WebUI and extension use the same shared Knowledge QA interface. The extension can also be blocked by extension-specific setup, such as missing server URL, API key, host permission, or an allowlist/backend reachability problem.

## Setup Requirements

Before Knowledge QA can answer:

1. The tldw server must be reachable.
2. Credentials must be configured for the selected auth mode.
3. The server must expose the Knowledge QA/RAG endpoints.
4. At least one selected personal-library source category must be searchable, or web fallback must be available and enabled.
5. Documents, media, notes, or other library items must be indexed before local-only answers can cite them.

If setup is incomplete, Knowledge QA should show a recovery state instead of a blank page. Follow the visible action, such as finishing setup, reconnecting, adding or indexing sources, selecting source categories, or retrying the backend check.

## Ask A Question

1. Open Knowledge QA.
2. Confirm the page says `Ask Your Library`.
3. Review the source scope.
4. Enter a question in the search box.
5. Choose `Ask`.
6. Review the answer and citations.

Good questions are specific and library-grounded, for example:

- "What does my library say about the onboarding checklist?"
- "Which notes mention API key setup?"
- "Summarize the cited evidence about the release process."

## Source Scope

Source scope controls what Knowledge QA is allowed to search.

Use the source scope controls to:

- Select source categories such as documents/media or notes.
- Select specific documents or notes when available.
- Save and restore source/search profiles for repeated workflows.
- Enable or disable web fallback when the server supports web search.

If no source categories are selected and web fallback is off, Knowledge QA should block submission with recovery copy. If the server has no indexed library sources, add or index content first through source-owner surfaces such as Media, Notes, or Quick Ingest.

Web fallback is optional. Leave it off when you only want personal-library evidence. When it is on and the server uses external web results, Knowledge QA should label that evidence origin in the answer, evidence review, history, and export surfaces.

## Presets And Settings

Use presets when you want a quick tradeoff:

- `Fast`: quicker retrieval with lighter search.
- `Balanced`: normal default for most questions.
- `Deep` or `Thorough`: broader retrieval for harder questions.
- `Custom`: manual settings have diverged from a preset.

Use settings when you need more control:

- Basic settings cover common tuning.
- Expert settings expose retrieval details for power users.
- Reset defaults returns settings to the supported baseline.

Keep settings changes tied to a concrete question. If answers become noisy, reduce scope or return to `Balanced`.

## Answer Model

The answer model/provider menu controls which configured model generates the final answer. You can use the server default, select a configured provider/model, or enter a manual model name when the UI supports it.

Provider/model availability depends on server configuration. Knowledge QA should avoid exposing sensitive provider configuration details in UI errors.

## Citations And Evidence

Citations connect answer claims to retrieved evidence. Use them to decide whether the answer is grounded enough to trust.

Expected review flow:

1. Read the answer.
2. Open or inspect cited sources.
3. Compare citation snippets with the answer claims.
4. Use the evidence rail when more detail is needed.
5. Switch between `Sources` and `Details` evidence views when available.

Treat uncited or weakly cited claims as lower confidence. Narrow source scope, try a more specific question, or use a deeper preset if the evidence does not support the answer.

If an answer is labeled as unsupported, degraded, unknown, or missing citations, do not treat it as a normal grounded answer. Re-run with narrower sources, inspect the evidence rail, or change settings before relying on it.

## No Results And Recovery

When Knowledge QA returns no results:

- Check whether the correct source categories are selected.
- Select exact documents or notes if the search should be scoped.
- Try a broader query or a different phrase.
- Use web fallback only when you want external web evidence and the server supports it.
- Wait for indexing to finish if recently added sources are not searchable yet.

No-results recovery should be explicit and should not imply that the library was searched successfully when the backend was offline or no sources were selected.

## Export

Use export after reviewing the answer and evidence.

Observed export behavior includes:

- Markdown export for text reuse.
- PDF export through the browser print flow.
- Chatbook export for portable tldw conversation packaging.
- Save to Notes when exportable content is available.
- Share-link controls when the current thread supports read-only sharing.

Exported content should preserve the question, answer, citations, source scope, and relevant retrieval details when available.

Exports should also preserve trust and evidence labels. Markdown exports include a `Trust and Evidence` section when trust state or evidence origin is known. Unsupported drafts are labeled in export output; if the export dialog asks for confirmation, confirm only when you intentionally want to keep that lower-confidence answer with the warning attached.

## Relationship To Other Workflows

- Research Workspace: use it for broader research projects, source organization, and multi-step investigation. Knowledge QA can support focused cited questions.
- Chat: use it for general conversation or model interaction. Knowledge QA is for library-grounded answers with citations.
- Notes: notes can be searched as Knowledge QA sources when indexed and selected. Export can also save reviewed output back to Notes.
- Media: ingested and indexed media can be searched as Knowledge QA sources.
- Flashcards: use the separate flashcards route. `/knowledge` does not create decks, review cards, or run spaced repetition.

## WebUI Versus Extension Differences

Shared behavior:

- Search state, source scope, presets, model controls, settings, citations, evidence review, and export should follow the same product contract.
- Backend unavailable, no-source, no-results, and blocked-search states should show actionable recovery.

Extension-specific behavior:

- Extension setup can fail because of missing server configuration, missing API key, host permission, allowlist, or backend reachability.
- Extension options width can force compact layout sooner than the WebUI.
- Extension shared-link routing may differ from WebUI routing.

WebUI-specific behavior:

- WebUI readiness can be blocked by route-level backend readiness or server health checks.
- WebUI has more horizontal space for detailed layout and evidence review.

## Regression Commands

Run focused Knowledge QA checks from the relevant package directory.

Shared UI:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA
```

WebUI:

```bash
cd apps/tldw-frontend
npx playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts e2e/ux-audit/knowledge-qa-states.spec.ts e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium
```

Seeded live WebUI release gate:

```bash
source ../../.venv/bin/activate
python ../../Helper_Scripts/seed_knowledge_qa_uat.py \
  --server-url http://127.0.0.1:8000 \
  --api-key "$TLDW_E2E_API_KEY" \
  --manifest /tmp/knowledge-qa-uat.json

TLDW_WEB_AUTOSTART=false \
TLDW_WEB_URL=http://127.0.0.1:3000 \
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY="$TLDW_E2E_API_KEY" \
TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST=/tmp/knowledge-qa-uat.json \
bunx playwright test e2e/ux-audit/knowledge-qa-live-backend.spec.ts --project=chromium --reporter=line
```

Extension:

```bash
cd apps/extension
npx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts tests/e2e/knowledge-empty-recovery.spec.ts --project=chromium-extension
```

Seeded live extension release gate:

```bash
cd apps/extension
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY="$TLDW_E2E_API_KEY" \
TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST=/tmp/knowledge-qa-uat.json \
bunx playwright test tests/e2e/knowledge-qa-live-backend.spec.ts --project=chromium-extension --reporter=line
```

The seeded live gates are strict. Missing backend readiness, missing manifest, null seeded source IDs, API failure, no generated cited answer, or extension launch failure should be recorded as a failing release gate rather than a passing test.

Python backend checks are not required for documentation-only changes. If backend Knowledge QA/RAG code is touched, run the focused pytest paths and Bandit on the touched Python scope.
