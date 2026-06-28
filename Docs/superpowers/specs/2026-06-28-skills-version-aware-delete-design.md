# Skills Version-Aware Delete Design

## Context

`TASK-530.10` continues the `/skills` Safe Operations sequence after import review and seed overwrite confirmation. The backend already has most of the destructive-delete protection:

- `DELETE /api/v1/skills/{skill_name}` accepts `If-Match`.
- `SkillsService.delete_skill()` checks the current registry version when `expected_version` is provided.
- Service unit tests cover stale delete conflicts and rollback behavior.

The gap is the end-to-end product path. The Skills manager only passes a skill name to `tldwClient.deleteSkill(name)`, and the client sends an unversioned `DELETE`. The list summary contract also omits `version`, so the table row cannot send `If-Match` even though the registry has a version.

`SkillSummary` is also reused by the `/skills/context` response. Any required summary-field change must keep that endpoint valid by either adding `version` to context summaries or by introducing a separate context summary schema. This slice should prefer adding `version` to context summaries because the service already has registry versions and the `context_text` prompt body can remain unchanged.

## Goal

Make single-skill delete version-aware when the frontend has a known row version, while preserving existing unversioned delete compatibility for older callers and unknown-version rows.

## Non-Goals

- No bulk delete.
- No undo/restore workflow.
- No delete preview endpoint.
- No permissions/model/tool metadata panel work.
- No export feedback work.
- No broader Skills manager redesign.
- No change to update semantics beyond any type sharing needed for the list summary.

## UX Design

The visible delete workflow stays the same for a current row:

1. User clicks the row delete icon.
2. The manager opens the existing destructive confirmation dialog.
3. User confirms.
4. The frontend sends `DELETE` with `If-Match: <row.version>`.
5. On success, the list refreshes and the existing `Skill deleted` notification appears.

If the delete receives a stale-version conflict, the manager shows recovery-specific feedback instead of the generic server error:

- Message: `Skill changed elsewhere`
- Description: `Reload skills before deleting this version.`

After a conflict, the manager invalidates the skills query so the table refreshes to the latest version. The skill is not deleted. The user can review the refreshed row and choose delete again if it is still intended.

If a row or caller has no known version, deletion remains compatible and sends no `If-Match` header. This path is intentionally preserved for older UI state, direct API consumers, and any compatibility caller that has only the skill name.

## Technical Design

### Backend Contract

Extend `SkillSummary` with:

```python
version: int = Field(..., description="Version for optimistic locking")
```

Update `_metadata_to_summary()` to include `metadata.version`. This exposes a value already present in `SkillMetadata` and the skill registry; it does not require a schema migration.

Because `SkillSummary` is reused by `SkillContextPayload.available_skills`, update `SkillsService._build_context_payload()` to include `version` in each `available_skills` item. Do not add version text to `context_text`; this is API metadata for optimistic operations, not model prompt content.

Keep the existing delete endpoint signature:

```python
expected_version: Optional[int] = Header(None, alias="If-Match")
```

Keep existing compatibility behavior: `expected_version=None` does not reject the delete.

Add API integration tests for:

- list summaries include `version`.
- delete with matching `If-Match` returns `204`.
- delete without `If-Match` still returns `204`.
- delete with stale `If-Match` returns `409` and leaves the skill available.
- context payload still returns `200` and includes `version` in `available_skills` without changing the prompt-oriented `context_text`.

### Frontend Types And Client

Add `version: number` to `SkillSummary` in `apps/packages/ui/src/types/skill.ts`.

Update `tldwClient.deleteSkill` in `workspace-api.ts`:

```ts
async deleteSkill(name: string, version?: number): Promise<void>
```

When `version` is a finite number, send:

```ts
headers: { "If-Match": String(version) }
```

When no valid version is supplied, keep the existing unversioned request shape.

### Skills Manager

Change the delete mutation input from `name: string` to a small payload:

```ts
{ name: string; version?: number }
```

Change `handleDelete` to accept the `SkillSummary` row or `{ name, version }`, and update the row action from:

```ts
onClick={() => handleDelete(record.name)}
```

to pass the row version.

In `onError`, detect conflicts through a small local helper rather than inline checks. It should handle the error shapes used elsewhere in the UI:

- `err.status === 409`
- `err.statusCode === 409`
- `err.response?.status === 409`
- stringified message containing `409`

For conflicts:

- invalidate `["skills"]`
- show the conflict-specific message and description

For non-conflict errors, keep the existing generic delete error behavior and sanitized description.

## Accessibility And Interaction Notes

- The destructive confirmation remains explicit and keyboard-accessible through Ant Design `Modal.confirm`.
- The row delete button keeps its current accessible label: `Delete {skillName}`.
- Conflict copy should name the recovery action in plain language: reload the list before deleting.
- Do not add a second confirmation after conflict; the refreshed row plus existing confirmation is enough for this slice.

## Testing

Backend focused tests:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  -k "list_skills or delete_skill" -v
```

Frontend focused tests:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/tldw/domains/__tests__/workspace-api.skills.test.ts \
  src/components/Option/Skills/__tests__/Manager.test.tsx \
  --reporter=dot
```

Add or update tests for:

- `workspaceApiMethods.deleteSkill` sends `If-Match` when version is provided.
- `workspaceApiMethods.deleteSkill` omits headers when version is not provided.
- `SkillsManager` calls `deleteSkill(name, version)` from a versioned row.
- `SkillsManager` still calls `deleteSkill(name, undefined)` or equivalent for an unknown-version row.
- `SkillsManager` shows `Skill changed elsewhere` and `Reload skills before deleting this version.` on a `409`.
- existing context endpoint tests, MCP integration tests, and fixtures remain valid after `SkillSummary.version` becomes required.

Run Bandit on touched backend Python files before completion:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/app/api/v1/schemas/skills_schemas.py \
  tldw_Server_API/app/core/Skills/skills_service.py \
  -f json -o /tmp/bandit_task_530_10.json
```

## Risks And Mitigations

- **Risk:** Adding `version` to `SkillSummary` could expose an extra field to older clients.
  **Mitigation:** This is additive JSON. Existing consumers should ignore unknown fields, and the frontend type will be updated.

- **Risk:** Some test fixtures create skill summaries without versions.
  **Mitigation:** Update focused Skills test fixtures. Where intentionally testing unknown-version compatibility, use a type cast or partial object locally rather than weakening the production type.

- **Risk:** Requiring `SkillSummary.version` could break `/skills/context` and async context integration tests, because those paths build or mock summaries from service dictionaries rather than `_metadata_to_summary()`.
  **Mitigation:** Include `version` in `_build_context_payload()` `available_skills` items, keep `context_text` unchanged, and update context endpoint/MCP integration tests or mocks that construct summary dictionaries.

- **Risk:** Conflict detection based only on status may miss errors wrapped by the browser proxy.
  **Mitigation:** Use a focused helper that checks common status locations plus the stringified message, matching existing Skills drawer conflict handling style while keeping the behavior local to delete handling.

- **Risk:** Invalidating the query after conflict could briefly leave stale row state visible.
  **Mitigation:** React Query invalidation matches existing manager cache patterns and is sufficient for this focused safety slice.

## Acceptance Criteria Mapping

- AC1: `SkillSummary.version` is exposed; row delete passes version; `deleteSkill` sends `If-Match` when valid.
- AC2: Existing backend endpoint/service behavior remains enforced and gains API-level stale conflict coverage.
- AC3: Manager conflict handling shows reload-before-delete recovery copy and refreshes the list.
- AC4: `deleteSkill(name)` without version keeps unversioned compatibility and has explicit test coverage.
- AC5: Focused backend API, frontend API-client, and manager tests cover success, compatibility, and stale conflict paths.
