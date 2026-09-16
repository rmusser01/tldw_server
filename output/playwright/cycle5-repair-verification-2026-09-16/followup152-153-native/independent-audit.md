# UAT153 independent native Settings acceptance

**UAT153 passes the bounded Custom-to-Balanced native acceptance. TASK-13260.91 remains In Progress because UAT152 real source-generation acceptance is still pending.** This review reads retained browser receipts; no browser/runtime/product/git actions or new inference were performed.

## Actual save/reload sequence

1. `custom-set.txt` records filling the sixth numeric field with10. The surrounding actual form snapshot identifies it as **RAG Request Timeout**, not the ordinary request timeout. `custom-before-save.txt` shows Custom, neither Balanced nor Extended checked, RAG10, Chat Request120 and Chat startup120 seconds.
2. `custom-saved.txt` records the real Save click and a successful server response notice. `custom-reload-start.txt` records `page.reload()`. `custom-reloaded-expanded.txt` shows the reloaded form still Custom with neither preset checked and RAG10. Thus ordinary Save/reload preserves the deliberate custom timeout.
3. `balanced-selected.txt` records an actual click on the displayed Balanced option. Balanced becomes checked and RAG changes to120; the two chat generation fields remain120. `balanced-saved.txt` records the actual Save click and successful response notice.
4. `balanced-reload-start.txt` records a second real reload. `balanced-reloaded-expanded.txt` confirms Balanced checked, Extended unchecked, no Custom indicator, and **Chat Request120, Chat startup120 and RAG120 seconds**. `balanced-fields-and-labels.txt` independently reads the DOM values `[10,15,120,120,15,120,60,60]` and confirms Balanced true/Extended false.

The surrounding labels map those values, in order, to ordinary request10, ordinary streaming15, chat request120, chat startup120, chat streaming15, RAG120, media60 and upload60 seconds. The generation budgets are distinct from the deliberately shorter general request/stream inactivity budgets. Nothing in these receipts implies every timeout becomes120.

## Scope and remaining acceptance

- These are actual UI Save operations followed by real reloads, not direct storage/config rewrites. The wrapper-redacted snapshots include the relevant commands and rendered form state. Reload persistence is observed through the UI; this audit does not claim a separate direct storage read.
- This closes the native UAT153 presentation/action/persistence gap supporting existing AC4. It does **not** establish UAT152's original real source RAG request exceeding10 seconds and completing successfully under the saved generation budget. AC3 remains unchecked and the shared task must stay In Progress until that retained native source check passes.
- Prior automated Settings/transport reviews and combined test reports in `../followup152-153/` and `../followup151-154-combined/` remain separate. No test suite was rerun for this artifact audit. Later timeout accessibility-label changes under UAT162 are also separate; this capture's numeric inputs lack associated labels/IDs, as its final DOM receipt truthfully records.
- No screenshot exists in this evidence set and no visual or clean-console claim is added. Public demo-key help copy is ordinary product text; actual API-key values in these snapshots are `[REDACTED]`.

## Retention

The exact ten receipts and this audit are enumerated in `retention-manifest.json`, with source and retained SHA256 values. All copies are byte-identical; trailing-whitespace normalization required no changes. All retained bytes were checked against26 known credential values from six local sources plus credential patterns with zero matches, without printing values. Credential files, profiles, private wrappers and runtime logs are excluded. Bandit is inapplicable: only text/JSON evidence and task metadata are added in this pass.
