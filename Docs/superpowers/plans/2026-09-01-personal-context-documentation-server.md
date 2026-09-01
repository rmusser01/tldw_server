# Personal Context Profile Server Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish accurate tldw_server operator and developer documentation for Personal Context storage, APIs, and the currently shipped Sync-v2 boundary.

**Architecture:** Add one operator guide and one developer guide, update the existing API reference where it has fallen behind merged Sync-v2 behavior, and expose all three through the curated MkDocs structure. Treat canonical `Docs/` files as source, regenerate `Docs/Published/` deterministically, and distinguish protocol support from missing server-origin publication and purge-acknowledgement workflows.

**Tech Stack:** Markdown, MkDocs Material, Backlog.md, Git, GitHub, existing Python/pytest contract checks

**Backlog task:** TASK-13151

**Design specification:** `https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md`

---

## File map

**Create**

- `Docs/User_Guides/Server/Personal_Context_Profile.md` — operator lifecycle, setup, Chatbook linking, API use, export, purge warning, and troubleshooting.
- `Docs/Code_Documentation/Personal_Context_Developer_Guide.md` — shared contract, encrypted storage, service/API/Sync boundaries, current gaps, extension checklist, and test map.

**Modify**

- `Docs/API-related/Personal_Context_API.md` — preserve endpoint reference while correcting stale Sync wording and documenting publication/purge limits.
- `Docs/User_Guides/index.md` — add common-workflow and troubleshooting links.
- `Docs/Code_Documentation/index.md` — add the developer guide.
- `Docs/API-related/API_README.md` — link the API/operator/developer guides.
- `Docs/mkdocs.yml` — add User Wiki, API/contracts, and backend-guide entries.
- `Docs/Published/**` — generated only by `Helper_Scripts/refresh_docs_published.sh`.
- `Docs/superpowers/plans/2026-09-01-personal-context-documentation-server.md` — this executable plan.
- `backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md` — plan, acceptance criteria, evidence, ADR result, and implementation notes.

**Inspect but do not duplicate**

- `Docs/Design/2026-08-30-personal-context-profile-server-design.md`
- `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`
- `Docs/Code_Documentation/Docs_Site_Guide.md`
- `packages/tldw_profile_core/`
- `tldw_Server_API/app/core/Personalization/personal_context_*.py`
- `tldw_Server_API/app/core/Sync/v2/`

## Cross-repository execution prerequisite

Completed before server execution. The approved design is published at the stable Chatbook `dev` URL:

`https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md`

The publication record is `TASK-27016`. PR [#2292](https://github.com/rmusser01/tldw_chatbook/pull/2292) published the reviewed design, and correction PR [#2294](https://github.com/rmusser01/tldw_chatbook/pull/2294) fixed its publication record. GitHub API verification on 2026-09-01 confirmed both PRs are merged and the specification is present on Chatbook `dev`.

### Task 1: Rebase and establish merged server truth

**Files:**

- Inspect: all paths in the file map
- Modify: `backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md`

- [x] **Step 1: Rebase the isolated branch on current `dev`**

Run:

```bash
git fetch origin dev
git rebase origin/dev
```

Expected: docs describe only code merged to `dev`, never unmerged follow-up branch behavior.

- [x] **Step 2: Verify Backlog ownership, the reviewed specification, and workflow lessons**

The spec-only Chatbook PR must be merged into `dev` before server execution so the cross-repository reference is stable. Run:

```bash
backlog task 13151 --plain
rg -n "TASK-13151|Document Personal Context Profile server" \
  "backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"
gh api -X GET repos/rmusser01/tldw_chatbook/contents/Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md \
  -f ref=dev
sed -n '1,240p' backlog/docs/lessons-testing-evidence.md
```

The Backlog hygiene lesson is not present in this server tree. It was consulted read-only from the source Chatbook checkout at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/backlog/docs/lessons-backlog-hygiene.md`; it is recorded here as execution evidence, not as an in-repository command.

Expected: TASK-13151 resolves to this documentation file and is assigned to `@codex`; the reviewed specification on Chatbook `dev` returns file metadata; no duplicate task ID appears. Repeat the task-resolution check after every rebase. Backlog MCP is not available in this repository, so use the Backlog CLI throughout.

Run this all-ref/all-worktree collision sweep now and after the final rebase:

```bash
profile_task_matches=$(
  {
    git for-each-ref --format='%(refname)' refs/heads refs/remotes |
      while IFS= read -r profile_ref; do
        git grep -l '^id: TASK-13151$' "$profile_ref" -- 'backlog/tasks/*.md' 2>/dev/null || true
      done | sed 's/^[^:]*://'
    git worktree list --porcelain |
      awk '$1 == "worktree" { sub(/^worktree /, ""); print }' |
      while IFS= read -r profile_worktree; do
        rg -l '^id: TASK-13151$' "$profile_worktree/backlog/tasks" 2>/dev/null || true
      done
  } | awk -F/ '{ print $NF }' | sort -u
)
printf '%s\n' "$profile_task_matches"
test "$profile_task_matches" = "task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"
```

Expected: the only unique matching task filename is the intended TASK-13151 record.

- [x] **Step 3: Confirm Shared Core pin and parity authority**

Run:

```bash
rg -n "version = \"0\.1\.0\"" packages/tldw_profile_core/pyproject.toml
rg -n "digest|parity|tldw_profile_core" \
  tldw_Server_API/tests/Personalization/test_personal_context_contract.py \
  backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md \
  Docs/Design/2026-08-30-personal-context-profile-server-design.md
```

Expected: version `0.1.0` and current parity/digest authority are identifiable.

- [x] **Step 4: Confirm current REST, Sync, and purge boundaries**

Run:

```bash
rg -n "def (create_record|update_record|create_proposal|purge_profile)|/purge" \
  tldw_Server_API/app/api/v1/endpoints/personal_context.py
rg -n "personal_context\.(manifest|scope|record|proposal|purge)" \
  tldw_Server_API/app/core/Sync/v2 tldw_Server_API/tests/Sync
rg -n "purge_pending|synchronization acknowledgment|Sync endpoints" \
  Docs/API-related/Personal_Context_API.md \
  tldw_Server_API/app/core/Personalization \
  tldw_Server_API/app/core/Sync/v2
```

Expected: five protocol domains plus bootstrap/inbound adapters exist; no ordinary REST-to-Sync publication seam or purge-envelope publication/acknowledgement completion is found.

- [x] **Step 5: Record the plan and ADR result in TASK-13151**

Run:

```bash
backlog task edit 13151 --plan $'1. Rebase and inventory merged behavior.\n2. Add server operator guide.\n3. Add developer guide and correct API reference.\n4. Add indexes and MkDocs navigation.\n5. Final rebase, regenerate curated docs, strict validation.\n6. Complete notes and open docs-only PR.\n\nADR required: no\nADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md\nReason: Documentation only; the existing Personal Context authority, Sync, and encryption ADR applies.'
```

- [x] **Step 6: Commit the plan and task metadata**

Run:

```bash
git add \
  Docs/superpowers/plans/2026-09-01-personal-context-documentation-server.md \
  "backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"
git commit -m "docs: plan server Personal Context guides"
```

### Task 2: Add the operator and user guide

**Files:**

- Create: `Docs/User_Guides/Server/Personal_Context_Profile.md`

- [ ] **Step 1: Write purpose and product boundaries**

State that tldw_server is the authenticated home peer/canonical server copy, Chatbook is the current full editing/interview UI, no complete standalone server profile editor exists, and standalone peers become one canonical linked profile only after reviewed reconciliation and link completion. Link:

`https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/User_Guide/settings/personal-context-profile.md`

- [ ] **Step 2: Document prerequisites and safe key setup**

Cover successful API-key/JWT authentication; `TLDW_PERSONAL_CONTEXT_MASTER_KEY` as strict base64 for exactly 32 bytes; secure backup before profile creation; fail-closed behavior for missing/malformed/changed keys; and one manifest per authenticated user's `Personalization.db`. Include a standard-library generation example but no fixed key and no advice to commit it.

- [ ] **Step 3: Add setup and status workflow**

Document configuration before start, server status/capability confirmation, manifest inspection/creation when needed, reviewed linking from Chatbook before upload, and Chatbook editing for changes expected to use current linked Sync. Link endpoint details to `../../API-related/Personal_Context_API.md`.

- [ ] **Step 4: Add current sync/non-sync matrix**

Include this deliberately identical shared-contract block, with the markers retained so Chatbook can check it automatically:

```markdown
<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->
```

Follow it with this full matrix:

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

Then add these server-specific notes:

- Current flow accepts eligible Chatbook-originated manifest, scope, record/tombstone, and proposal changes.
- The home server wraps its Sync integrity key for authenticated registered Chatbook devices; this is not at-rest key sharing.
- Ordinary server REST record/proposal mutations are not currently published to linked Chatbook clients.

- [ ] **Step 5: Document export, local removal, and purge accurately**

Explain export confirmations/sensitivity; Chatbook ownership of local-copy removal; server rejection of `local_copy`; and `POST /purge` requiring `DELETE EVERYWHERE`, advancing a server-local fence, removing canonical bodies/runtime state, blocking mutations, and remaining `purge_pending`. Include the exact sentence `The server purge endpoint does not publish the protocol purge envelope, and acknowledgement completion is not wired.` Do not promise reconnecting devices clears the state.

- [ ] **Step 6: Add troubleshooting**

Use these exact seven failure-state labels and give a cause, safe next action, and current product limit for each:

1. **Profile locked**
2. **Offline or queued**
3. **Capability not negotiated**
4. **Version conflict**
5. **First-link semantic collision**
6. **Post-link semantic collision**
7. **Purge pending**

Also cover authentication failure, missing/changed key, schema/quota incompatibility, and a REST edit absent from Chatbook. Explicitly state when no resolver or completion path exists.

- [ ] **Step 7: Validate and commit**

Run:

```bash
test -f Docs/API-related/Personal_Context_API.md
profile_operator_guide=Docs/User_Guides/Server/Personal_Context_Profile.md
rg -Fq '<!-- shared-personal-context-contract:start -->' "$profile_operator_guide"
rg -Fq '<!-- shared-personal-context-contract:end -->' "$profile_operator_guide"
rg -Fq 'Ordinary server REST record/proposal mutations are not currently published to linked Chatbook clients.' \
  "$profile_operator_guide"
rg -Fq 'The server purge endpoint does not publish the protocol purge envelope, and acknowledgement completion is not wired.' \
  "$profile_operator_guide"
rg -Fq 'First-link semantic collision' "$profile_operator_guide"
rg -Fq 'Post-link semantic collision' "$profile_operator_guide"
git diff --check -- Docs/User_Guides/Server/Personal_Context_Profile.md
git add Docs/User_Guides/Server/Personal_Context_Profile.md
git commit -m "docs: add Personal Context server operator guide"
```

### Task 3: Add developer guide and correct API reference

**Files:**

- Create: `Docs/Code_Documentation/Personal_Context_Developer_Guide.md`
- Modify: `Docs/API-related/Personal_Context_API.md`

- [ ] **Step 1: Write contract, ownership, and crypto sections**

Cover `tldw-profile-core==0.1.0`, parity/digest authority, separate Sync envelopes, authenticated per-user `Personalization.db`, root/profile/object key hierarchy, fail-closed locked state, and peer-local at-rest keys versus wrapped server-owned Sync integrity key. Use stable GitHub links for source-only documents:

- `https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/2026-08-30-personal-context-profile-server-design.md`
- `https://github.com/rmusser01/tldw_server/blob/dev/backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md`

- [ ] **Step 2: Add exact component map**

Document:

- `tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py` — authenticated service dependency
- `tldw_Server_API/app/api/v1/schemas/personal_context.py` — HTTP schemas
- `tldw_Server_API/app/api/v1/endpoints/personal_context.py` — REST translation
- `tldw_Server_API/app/core/DB_Management/Personal_Context_Key_Store.py` — real root/profile key-custody owner
- `tldw_Server_API/app/core/Personalization/personal_context_key_provider.py` — compatibility re-export, not the custody owner
- `tldw_Server_API/app/core/Personalization/personal_context_crypto.py` — envelopes/wrapping
- `tldw_Server_API/app/core/Personalization/personal_context_repository.py` — versions, heads, fences
- `tldw_Server_API/app/core/Personalization/personal_context_service.py` — canonical business boundary
- `tldw_Server_API/app/core/Personalization/personal_context_export.py` — exports
- `tldw_Server_API/app/core/Personalization/personal_context_runtime_policy.py` — server-local runtime
- `tldw_Server_API/app/core/Sync/v2/profile.py` — bootstrap, binding, key wrapping, completion
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py` — transport validation/encryption
- `tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py` — inbound service projection

State that endpoints, agents, Sync, and future publishers never access profile tables directly.

- [ ] **Step 3: Document REST and Sync separately**

REST:

`authentication -> PersonalContextService -> encrypted repository -> response`

Sync/bootstrap:

`capability negotiation -> registered device -> reviewed Chatbook plan -> snapshot/wrapped integrity key -> completion -> inbound validation/materialization`

Name all five domains. Repeat the approved shared-contract statement between the exact `<!-- shared-personal-context-contract:start -->` and `<!-- shared-personal-context-contract:end -->` markers. Include these exact current-limit statements so validation can fail closed:

- `REST edits are not published to linked clients.`
- `Server purge does not publish the protocol purge envelope, and acknowledgement completion is absent.`
- `Reviewed first-link reconciliation handles first-link semantic collisions before completion.`
- `No dedicated post-link semantic-collision resolver exists.`

Repeat the full boundary matrix in developer terms so every shared and peer-local category is explicit:

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

- [ ] **Step 4: Add the complete extension checklist and test map**

Include all ten checklist items:

1. Decide whether the change affects the shared contract or only one peer.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability.
4. Route through the owning service; never access profile tables directly.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage in both repositories.
8. Add peer-specific migration, repository, service, API/UI, and recovery tests.
9. Update the governing ADR for storage, ownership, encryption, Sync, or authority changes.
10. Update both documentation sets whenever the shared contract changes.

Map these exact suites:

- `packages/tldw_profile_core/tests/tldw_profile_core/test_public_contract.py`
- `tldw_Server_API/tests/Personalization/test_personal_context_contract.py`
- `tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py`
- `tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py`
- `tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py`

- [ ] **Step 5: Correct the API Sync boundary**

Replace the blanket closing statement with `REST and Sync-v2 boundary`:

- REST and Sync V2 are separate surfaces over the canonical service/repository.
- Sync supports negotiation, bootstrap/link completion, and inbound Chatbook-originated domains.
- REST edits are not published to linked clients.
- Server purge does not publish the protocol purge envelope and remains pending because acknowledgement completion is absent.

- [ ] **Step 6: Validate and commit**

Run:

```bash
test -f packages/tldw_profile_core/pyproject.toml
test -f tldw_Server_API/app/core/DB_Management/Personal_Context_Key_Store.py
test -f tldw_Server_API/app/core/Personalization/personal_context_service.py
test -f tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py
test -f tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py
profile_developer_guide=Docs/Code_Documentation/Personal_Context_Developer_Guide.md
profile_api_reference=Docs/API-related/Personal_Context_API.md
rg -Fq '<!-- shared-personal-context-contract:start -->' "$profile_developer_guide"
rg -Fq '<!-- shared-personal-context-contract:end -->' "$profile_developer_guide"
rg -Fq 'REST edits are not published to linked clients.' "$profile_developer_guide"
rg -Fq 'Server purge does not publish the protocol purge envelope, and acknowledgement completion is absent.' \
  "$profile_developer_guide"
rg -Fq 'Reviewed first-link reconciliation handles first-link semantic collisions before completion.' \
  "$profile_developer_guide"
rg -Fq 'No dedicated post-link semantic-collision resolver exists.' "$profile_developer_guide"
rg -Fq 'REST edits are not published to linked clients.' "$profile_api_reference"
rg -Fq 'Server purge does not publish the protocol purge envelope and remains pending because acknowledgement completion is absent.' \
  "$profile_api_reference"
git diff --check -- Docs/Code_Documentation/Personal_Context_Developer_Guide.md Docs/API-related/Personal_Context_API.md
git add Docs/Code_Documentation/Personal_Context_Developer_Guide.md Docs/API-related/Personal_Context_API.md
git commit -m "docs: document Personal Context server internals"
```

### Task 4: Add indexes and MkDocs navigation

**Files:**

- Modify: `Docs/User_Guides/index.md`
- Modify: `Docs/Code_Documentation/index.md`
- Modify: `Docs/API-related/API_README.md`
- Modify: `Docs/mkdocs.yml`
- Inspect: `Docs/Code_Documentation/README.md`

- [ ] **Step 1: Add discovery links**

- Add Personal Context to user common workflows and troubleshooting.
- Add `Personal Context Profile: Personal_Context_Developer_Guide.md` under Chat & Knowledge.
- Replace the code-formatted API path with relative links to the API, operator, and developer guides.
- Leave `Docs/Code_Documentation/README.md` unchanged unless its organization clearly requires an entry; record the decision.

- [ ] **Step 2: Add concise MkDocs entries**

- User Wiki > Admin and Operations: `Personal Context Profile`.
- Developer Wiki > API and Contracts: `Personal Context API`.
- Developer Wiki > Backend Code Guides: `Personal Context Developer Guide`.

- [ ] **Step 3: Validate and commit**

Run:

```bash
test -f Docs/User_Guides/Server/Personal_Context_Profile.md
test -f Docs/API-related/Personal_Context_API.md
test -f Docs/Code_Documentation/Personal_Context_Developer_Guide.md
rg -n "Personal Context" Docs/User_Guides/index.md Docs/Code_Documentation/index.md Docs/API-related/API_README.md Docs/mkdocs.yml
git diff --check -- Docs/User_Guides/index.md Docs/Code_Documentation/index.md Docs/API-related/API_README.md Docs/mkdocs.yml
git add Docs/User_Guides/index.md Docs/Code_Documentation/index.md Docs/API-related/API_README.md Docs/mkdocs.yml
git commit -m "docs: publish Personal Context guide navigation"
```

### Task 5: Final rebase, generate, and strictly verify published docs

**Files:**

- Generate: `Docs/Published/**`
- Verify: all changed canonical docs

- [ ] **Step 1: Perform the final rebase before generation and task closeout**

Run:

```bash
git fetch origin dev
git rebase origin/dev
backlog task 13151 --plain
rg -n "TASK-13151|Document Personal Context Profile server" \
  "backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"
set -o pipefail
profile_task_matches=$(
  {
    git for-each-ref --format='%(refname)' refs/heads refs/remotes |
      while IFS= read -r profile_ref; do
        if profile_ref_match=$(git grep -l '^id: TASK-13151$' "$profile_ref" -- 'backlog/tasks/*.md' 2>/dev/null); then
          printf '%s\n' "$profile_ref_match"
        else
          profile_ref_status=$?
          test "$profile_ref_status" -eq 1 || exit "$profile_ref_status"
        fi
      done | sed 's/^[^:]*://'
    git worktree list --porcelain |
      awk '$1 == "worktree" { sub(/^worktree /, ""); print }' |
      while IFS= read -r profile_worktree; do
        # Old or prunable worktrees without a Backlog task directory cannot hold this task.
        if [ ! -d "$profile_worktree/backlog/tasks" ]; then
          continue
        fi
        if profile_worktree_match=$(rg -l '^id: TASK-13151$' "$profile_worktree/backlog/tasks" 2>/dev/null); then
          printf '%s\n' "$profile_worktree_match"
        else
          profile_worktree_status=$?
          test "$profile_worktree_status" -eq 1 || exit "$profile_worktree_status"
        fi
      done
  } | awk -F/ '{ print $NF }' | sort -u
) || {
  echo "TASK-13151 collision sweep failed"
  exit 1
}
printf '%s\n' "$profile_task_matches"
test "$profile_task_matches" = "task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"

# Re-inventory merged behavior after the rebase and before generating docs.
rg -q '^version = "0\.1\.0"$' packages/tldw_profile_core/pyproject.toml
rg -q '^EXPECTED_CONTRACT_DIGEST = "[0-9a-f]{64}"$' \
  tldw_Server_API/tests/Personalization/test_personal_context_contract.py
rg -q '^def test_server_pins_exact_chatbook_profile_core_contract' \
  tldw_Server_API/tests/Personalization/test_personal_context_contract.py
test "$(rg -c '^@router\.(get|post|patch|delete)' tldw_Server_API/app/api/v1/endpoints/personal_context.py)" -eq 19
for profile_rest_handler in create_record update_record create_proposal purge_profile; do
  rg -q "^def ${profile_rest_handler}\\(" \
    tldw_Server_API/app/api/v1/endpoints/personal_context.py
done
for profile_domain in \
  personal_context.manifest \
  personal_context.scope \
  personal_context.record \
  personal_context.proposal \
  personal_context.purge; do
  rg -Fq "\"$profile_domain\"" tldw_Server_API/app/core/Sync/v2/models.py
done
rg -q '^    def bootstrap_personal_context\(' tldw_Server_API/app/core/Sync/v2/profile.py
rg -q '^class PersonalContextMaterializer' \
  tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py
rg -Fq 'PURGE_PENDING = "purge_pending"' \
  tldw_Server_API/app/core/Personalization/personal_context_service.py
if rg -n 'publish|outbox|enqueue|append_envelope|create_envelope' \
  tldw_Server_API/app/api/v1/endpoints/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py; then
  echo "Unexpected Personal Context REST-to-Sync publication seam"
  exit 1
fi
if rg -n 'complete_.*purge|purge.*acknowledg|acknowledg.*purge' \
  tldw_Server_API/app/api/v1/endpoints/personal_context.py \
  tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py \
  tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py; then
  echo "Unexpected Personal Context purge acknowledgement completion seam"
  exit 1
fi
```

Expected: the branch is based on current `origin/dev`; TASK-13151 still resolves uniquely; Shared Core, REST, Sync bootstrap/inbound, and purge boundaries still match the guides' source claims. Any new REST publication or purge-completion seam stops execution for re-inventory. There must be no later rebase after the task is marked Done.

- [ ] **Step 2: Refresh and stage the curated tree**

Run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git add Docs/Published
```

- [ ] **Step 3: Prove a second refresh is idempotent**

Run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git diff --exit-code -- Docs/Published
git diff --check --cached
```

Expected: no unstaged generated diff and no whitespace errors in the staged generated output.

- [ ] **Step 4: Run public/private and strict MkDocs checks**

Run Steps 4 and 5 in the same shell. Select the checked host interpreter first, fall back to the worktree virtual environment, and fail if neither exists:

```bash
profile_python=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python
if [ ! -x "$profile_python" ]; then
  profile_python=.venv/bin/python
fi
if [ ! -x "$profile_python" ]; then
  echo "No executable project Python found at the host or worktree virtual environment"
  exit 1
fi
printf 'Using project Python: %s\n' "$profile_python"
"$profile_python" Helper_Scripts/docs/check_public_private_boundary.py
"$profile_python" -m mkdocs build --strict -f Docs/mkdocs.yml
```

Expected: boundary check reports OK; MkDocs builds with zero warnings.

- [ ] **Step 5: Run targeted Shared Core, endpoint, custody, composed-app, bootstrap, materializer, and transport tests**

Run:

```bash
test -x "${profile_python:-}" || {
  echo "profile_python is unavailable; run Task 5 Step 4 in this shell first"
  exit 1
}
"$profile_python" -m pytest -q \
  packages/tldw_profile_core/tests/tldw_profile_core/test_public_contract.py \
  tldw_Server_API/tests/Personalization/test_personal_context_contract.py \
  tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py \
  tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py \
  tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py
```

Expected: selected tests pass.

- [ ] **Step 6: Run claim, failure-state, source-link, and diff guards**

Run:

```bash
profile_operator_guide=Docs/User_Guides/Server/Personal_Context_Profile.md
profile_developer_guide=Docs/Code_Documentation/Personal_Context_Developer_Guide.md
profile_api_reference=Docs/API-related/Personal_Context_API.md
for profile_shared_doc in "$profile_operator_guide" "$profile_developer_guide"; do
  rg -Fq '<!-- shared-personal-context-contract:start -->' "$profile_shared_doc"
  rg -Fq '<!-- shared-personal-context-contract:end -->' "$profile_shared_doc"
done
rg -Fq 'Ordinary server REST record/proposal mutations are not currently published to linked Chatbook clients.' \
  "$profile_operator_guide"
rg -Fq 'The server purge endpoint does not publish the protocol purge envelope, and acknowledgement completion is not wired.' \
  "$profile_operator_guide"
rg -Fq 'First-link semantic collision' "$profile_operator_guide"
rg -Fq 'Post-link semantic collision' "$profile_operator_guide"
rg -Fq 'REST edits are not published to linked clients.' "$profile_developer_guide"
rg -Fq 'Server purge does not publish the protocol purge envelope, and acknowledgement completion is absent.' \
  "$profile_developer_guide"
rg -Fq 'Reviewed first-link reconciliation handles first-link semantic collisions before completion.' \
  "$profile_developer_guide"
rg -Fq 'No dedicated post-link semantic-collision resolver exists.' "$profile_developer_guide"
rg -Fq 'REST edits are not published to linked clients.' "$profile_api_reference"
rg -Fq 'Server purge does not publish the protocol purge envelope and remains pending because acknowledgement completion is absent.' \
  "$profile_api_reference"
for profile_label in \
  "Profile locked" \
  "Offline or queued" \
  "Capability not negotiated" \
  "Version conflict" \
  "First-link semantic collision" \
  "Post-link semantic collision" \
  "Purge pending"; do
  rg -Fq "$profile_label" Docs/User_Guides/Server/Personal_Context_Profile.md || {
    echo "Missing failure-state label: $profile_label"
    exit 1
  }
done
set +e
rg -n '\]\((\.\./)+(Docs/)?(Design|backlog)/' \
  Docs/Published/User_Guides/Server/Personal_Context_Profile.md \
  Docs/Published/Code_Documentation/Personal_Context_Developer_Guide.md
profile_link_guard_status=$?
set -e
if [ "$profile_link_guard_status" -eq 0 ]; then
  echo "Unexpected source-only relative link in published docs"
  exit 1
elif [ "$profile_link_guard_status" -gt 1 ]; then
  echo "Source-only link guard failed to execute"
  exit "$profile_link_guard_status"
else
  echo "Published docs contain no source-only relative links"
fi
profile_changed_paths=$(
  {
    git diff --name-only origin/dev...HEAD
    git diff --name-only
    git diff --cached --name-only
  } | sed '/^$/d' | sort -u
)
profile_unexpected_paths=$(
  printf '%s\n' "$profile_changed_paths" | awk '
    $0 == "Docs/API-related/API_README.md" { next }
    $0 == "Docs/API-related/Personal_Context_API.md" { next }
    $0 == "Docs/Code_Documentation/Personal_Context_Developer_Guide.md" { next }
    $0 == "Docs/Code_Documentation/index.md" { next }
    $0 == "Docs/User_Guides/Server/Personal_Context_Profile.md" { next }
    $0 == "Docs/User_Guides/index.md" { next }
    $0 == "Docs/mkdocs.yml" { next }
    $0 == "Docs/superpowers/plans/2026-09-01-personal-context-documentation-server.md" { next }
    $0 == "backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md" { next }
    index($0, "Docs/Published/") == 1 { next }
    NF { print }
  '
)
if [ -n "$profile_unexpected_paths" ]; then
  printf 'Unexpected changed paths:\n%s\n' "$profile_unexpected_paths"
  exit 1
fi
printf 'Allowed changed paths:\n%s\n' "$profile_changed_paths"
git diff --check origin/dev...HEAD
git diff --check --cached
git status --short
git diff --stat origin/dev...HEAD
git diff --stat --cached
```

Expected: each guide independently proves its required shared-contract and current-limit claims; the API independently proves REST publication and purge limits; all seven operator failure states are explicit; the expected no-match source-link guard succeeds; and the allowed-path assertion accepts only task/plan/canonical/generated docs.

- [ ] **Step 7: Commit generated output and recheck the committed diff**

Run:

```bash
git commit -m "docs: refresh published Personal Context guides"
git diff --check origin/dev...HEAD
git status --short
```

Expected: generated output is committed and the worktree is clean before task closeout.

### Task 6: Close TASK-13151 and open the server PR

**Files:**

- Modify: `backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md`

- [ ] **Step 1: Complete all ACs and DoD items, record evidence, and mark Done as the final repository mutation**

Run, replacing the bracketed evidence with the exact commands and results from Task 5. If any skip or blocker exists, replace `Known skips/blockers: none` with the actual list before running the second command that checks DoD item 6:

```bash
backlog task edit 13151 \
  --notes "Implemented the server Personal Context operator and developer guides, corrected API Sync wording, added discovery/navigation, regenerated published docs, and documented the exact shared-contract block, full boundary matrix, seven failure states, current limitations, and ten-item extension checklist. Verification: [exact Task 5 results]. ADR required: no. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: documentation only; the existing Personal Context authority, Sync, and encryption ADR applies. Bandit: not applicable because only documentation and task metadata changed. Lessons learned: [record a genuine lesson with its incident, or state none]. Known skips/blockers: none."
backlog task edit 13151 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Published accurate Personal Context operator, API, architecture, Sync-boundary, and troubleshooting documentation with reproducible generated output." \
  -s Done
backlog task 13151 --plain
git add "backlog/tasks/task-13151 - Document-Personal-Context-Profile-server-operations-and-architecture.md"
git diff --check --cached
git commit -m "docs: close server Personal Context documentation task"
```

Expected: every AC and DoD item is checked, evidence and Implementation Notes are present, and TASK-13151 is Done. Do not rebase or modify repository files after this commit.

- [ ] **Step 2: Push and open the PR against `dev`**

Prepare `/tmp/personal-context-server-pr.md` with summary, current limitations, and exact evidence, then run:

```bash
git push -u origin codex/personal-context-documentation
gh pr create --base dev --head codex/personal-context-documentation --title "docs: add Personal Context operator and developer guides" --body-file /tmp/personal-context-server-pr.md
```

Expected: a docs-only server PR against `dev`. Merge it before finalizing Chatbook's stable server links.

- [ ] **Step 3: Satisfy the human-authored Change summary merge gate**

After the PR is open, ask the requester to write the required `Change summary` in their own words, explaining both what changed and why these documentation choices were made. The generated PR body and any AI-authored draft do not satisfy this gate. Do not call the PR merge-ready or merge it until the requester has supplied that summary and all required checks/reviews are green.
