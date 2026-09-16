# Prompt editor transparency and false Synced state — read-only diagnosis

Reviewed source HEAD `4bd4e2dfda246bd83650dc922789551032a5af52`. No repository files changed; no browser, runtime, or inference performed. Native observations below are the root agent's captured evidence. This investigation traced the production path and ran only an in-memory Tailwind compilation check, not a new behavioral test.

## 1. Transparent full-page editor: confirmed, bounded CSS-token defect

`apps/packages/ui/src/components/Option/Prompt/PromptFullPageEditor.tsx:413`, `:446`, `:491`, and `:705` use `bg-background` (quarantined recipe, recipe, standard editor, and mobile preview overlay). `apps/tldw-frontend/tailwind.config.js:13` defines `bg`, not `background`; the extension imports this same config. Light/dark `--color-bg` values exist in `apps/packages/ui/src/assets/tailwind-shared.css:74` and `:140`.

An independent in-memory PostCSS/Tailwind compile using the actual config and raw `bg-background bg-bg bg-surface` content produced `.bg-bg` and `.bg-surface`, and **no `.bg-background` rule**. Generated `.bg-bg` sets opacity 1 and `background-color: rgb(var(--color-bg) / var(--tw-bg-opacity, 1))`.

Native proof:

- `/private/tmp/cycle4-targeted-remaining-prompt-sync-failure.png`: underlying list/header visibly show through the full-page editor.
- `/private/tmp/cycle4-targeted-remaining-prompt-layout-observation.json`: the fixed `z-50 ... bg-background` editor has computed `backgroundColor: rgba(0, 0, 0, 0)`. The requested short name `/private/tmp/prompt-layout-observation.json` does not exist; this is the actual artifact.

**Smallest correction:** replace these four local classes with `bg-bg`. No Tailwind alias, z-index change, positioning redesign, or global theme change is needed.

Read official `backlog task 13124.4 --plain`: the earlier layout task concluded its overlap was stale UAT/readiness drift and changed tests/page-object waits, **not production CSS**. Its four-viewport geometry checks cover the main workspace sidebar/search, not overlay opacity. It does not contradict the present native evidence.

**Permanent regression:** render all three editor modes and mobile preview, verify the supported semantic background token, and assert its actual generated CSS is opaque. Extend the existing native workspace flow to open the editor and assert computed nontransparent background in light/dark modes; retain the existing 390/1365/1536 geometry controls. A class-name-only assertion cannot prove that Tailwind emits CSS.

## 2. Failed standard Prompt update retains Synced: confirmed persisted-state gap

The list is rereading stale **status data**, not failing to invalidate its query:

1. `usePromptEditor.tsx:225–243` calls the real local `updatePrompt`, awaits `syncPromptAfterLocalSave`, and invalidates `["fetchAllPrompts"]` on local-save success regardless of remote outcome.
2. `db/dexie/helpers.ts:692–785` forwards the content update; `db/dexie/chat.ts:559–617` merges it into the existing record. Since this payload does not change `syncStatus`, the prior `synced` survives with the new local text.
3. `usePromptSync.tsx:70–132` calls actual `autoSyncPrompt` and correctly shows the warning when its result is unsuccessful. This is the native “Sync failed / ... saved locally” toast.
4. `prompt-studio.ts:463–484` -> `api-send.ts:195–202` -> `request-core.ts:751–765`: a rejected fetch is normalized into `{ok:false,status:0,error:"Failed to fetch"}`. The outer request wrapper retains dispatch metadata (`request-core.ts:224–236`). It does **not** reject the service promise like the current outage test mock.
5. `prompt-sync.ts:752–761` treats this failed response as `invalid_server_payload`. Its `failure` helper (`:633–662`) **returns** `syncStatus:"pending"`, but writes state only for uncertain recipes (`error`), not this standard Prompt.
6. `autoSyncPrompt` (`prompt-sync.ts:942–956`) persists `pending` only when `failureKind === "transient"`. The normalized failure misses that branch. The returned pending status is not applied by the caller.
7. `Prompt/index.tsx:307–311` rereads local records. `usePromptFilteredData.tsx:344–362` forwards the stored `syncStatus`; `SyncStatusBadge.tsx:54–64` accurately renders the stale stored value as “Synced.” No later display writer repairs it.

**Native failure boundary:** `/private/tmp/cycle4-targeted-remaining-prompt-requests.txt:229` records request **579**, `PUT /api/v1/prompt-studio/prompts/update/1`, `ERR_CONNECTION_REFUSED`. Both `...prompt-sync-failure-visible.txt` and `...prompt-sync-status.txt` show the new text “Offline recovery revision 0528.” alongside **Synced#1**, after the failure warning. The earlier `...prompt-sync-failure-settled.txt` still contains the old list text plus an edited draft; it is a pre-save observation, not the decisive failed-save proof.

**Native success boundary:** the same requests artifact at line265 records request **615**, `PUT .../update/1`, **200**. `/private/tmp/cycle4-targeted-remaining-prompt-recovered-response.json` returns the new text, `version_number:2`, `id:2`, `parent_version_id:1`; `...prompt-sync-restored.txt` shows **Synced#2**. Important: the badge's `#1/#2` are **server IDs**, not displayed version numbers (`SyncStatusBadge.tsx:198–201`). Recovery demonstrates a later explicit save succeeded; it does not make the earlier Synced claim truthful or establish automatic background recovery.

**Smallest correction:** persist the already-determined unsuccessful standard Prompt status at the sync owner boundary, including a normalized network-failure response, before query refresh. Keep this in `services/prompt-sync.ts` unless the permanent production-boundary regression identifies a required adjacent writer. Do not cosmetically override the badge, treat every HTTP failure as transient, or weaken v2 recipe dispatch/uncertainty gates. Preserve local content, server linkage, and the last successful server version/time on failure; only a valid acknowledged response may restore Synced. No backend change is indicated by this evidence.

### Required permanent regression and controls

- Seed a real local standard Prompt linked to server1 and marked Synced, then edit through the mounted owner with the real local save, `usePromptSync`, `autoSyncPrompt`, `prompt-studio`/`apiSend`, and request-core. Mock **fetch rejection**, not `updateServerPrompt` rejection. Await settled query refresh and assert new local text survives, the stored and rendered state is Pending, and the row does not say Synced. Verify unchanged server link and last successful version/time.
- Restore the transport with the actual update response shape (`id:2`, `version_number:2`, `parent_version_id:1`), explicitly save/retry, and assert Synced only after acknowledgement. Keep one request per explicit action and no duplicate create.
- Keep thrown transport failure as a separate control. Current `prompt-sync.auto-sync.test.ts:532–573` only uses `mockRejectedValueOnce(new Error("offline"))`; that takes `failureKind:"transient"` and therefore misses the real normalized-response path.
- Include new local standard Prompt and existing standard Prompt controls; verify unknown/malformed responses cannot claim Synced. Preserve the existing auto-sync-disabled behavior, known 401/403/409/422 handling, and recipe unknown/dispatched/not-dispatched uncertainty tests. Do not expand this repair into retry orchestration or account changes.

## Scope recommendation

Two independently reviewable repairs: four token replacements plus opacity coverage in the full-page editor; sync status persistence plus a real local-save -> normalized transport failure -> query -> badge regression. Existing success toasts describe the local save and are not proof of remote success. No source changes, new executable regression, lint/typecheck, or native re-test were performed in this read-only unit.
