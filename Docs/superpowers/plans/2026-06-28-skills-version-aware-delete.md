# Skills Version-Aware Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/skills` single-skill deletion send and enforce optimistic versions when the UI has a known skill version, while preserving unversioned delete compatibility.

**Architecture:** The backend already accepts `If-Match` for deletes and the service already enforces optimistic locking, so backend work is mostly response-contract exposure and API-level coverage. The frontend will carry `SkillSummary.version` from list rows into the existing delete confirmation flow, and the API client will include `If-Match` only for positive safe integer versions. Stale deletes get recovery-specific copy and refresh the existing React Query `["skills"]` cache prefix.

**Tech Stack:** FastAPI, Pydantic, pytest, React 18, TypeScript, TanStack Query, Ant Design `Modal.confirm`, Vitest, Testing Library.

**Spec:** `Docs/superpowers/specs/2026-06-28-skills-version-aware-delete-design.md`

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
  - Add `version: int` to `SkillSummary`.
- Modify `tldw_Server_API/app/api/v1/endpoints/skills.py`
  - Include `metadata.version` in `_metadata_to_summary()`.
- Modify `tldw_Server_API/app/core/Skills/skills_service.py`
  - Include `version` in `_build_context_payload()` `available_skills` dictionaries.
  - Keep `context_text` unchanged.
- Modify `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
  - Add list-summary version, delete `If-Match`, stale conflict, and context-summary version coverage.
- Modify `tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py` only if the focused MCP context tests fail after the schema change.
  - Keep any changes limited to test fixture summary dictionaries.
- Modify `apps/packages/ui/src/types/skill.ts`
  - Add `version: number` to `SkillSummary`.
- Modify `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
  - Extend `deleteSkill(name, version?)`.
  - Send `If-Match` only when `Number.isSafeInteger(version) && version > 0`.
- Modify `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`
  - Add API client delete header and invalid-version tests.
- Modify `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
  - Pass row versions to delete.
  - Add local conflict detection helper.
  - Show conflict-specific notification and invalidate `["skills"]` on stale delete conflicts.
- Modify `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
  - Update `makeSkill()` and focused fixtures with `version`.
  - Add versioned delete, unknown-version compatibility, and stale conflict recovery tests.
- Modify `backlog/tasks/task-530.10 - Implement-Skills-version-aware-delete-path.md`
  - Record plan link, implementation notes, verification results, and final summary.

## Task 1: Backend Summary Contract And Delete API Coverage

**Files:**
- Modify: `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify: `tldw_Server_API/app/core/Skills/skills_service.py`
- Maybe modify: `tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py`

- [ ] **Step 1: Add failing list-summary version test**

In `TestListSkills`, add a test near `test_list_skills_pagination`:

```python
    def test_list_skills_includes_version(self, client):
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "listed-version", "content": "content"},
        )
        assert create_resp.status_code == 201, create_resp.text

        r = client.get(f"{SKILLS_PREFIX}/?limit=50&offset=0")
        assert r.status_code == 200, r.text
        listed = {skill["name"]: skill for skill in r.json()["skills"]}
        assert listed["listed-version"]["version"] == create_resp.json()["version"]
```

- [ ] **Step 2: Add failing delete `If-Match` tests**

In `TestDeleteSkill`, keep `test_delete_skill_204` as the unversioned compatibility test and add:

```python
    def test_delete_skill_accepts_matching_if_match(self, client):
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "del-versioned", "content": "content"},
        )
        assert create_resp.status_code == 201, create_resp.text
        version = create_resp.json()["version"]

        r = client.delete(
            f"{SKILLS_PREFIX}/del-versioned",
            headers={"If-Match": str(version)},
        )
        assert r.status_code == 204, r.text

        missing = client.get(f"{SKILLS_PREFIX}/del-versioned")
        assert missing.status_code == 404

    def test_delete_skill_stale_if_match_returns_409_and_keeps_skill(self, client):
        create_resp = client.post(
            f"{SKILLS_PREFIX}/",
            json={"name": "del-stale", "content": "v1"},
        )
        assert create_resp.status_code == 201, create_resp.text

        update_resp = client.put(
            f"{SKILLS_PREFIX}/del-stale",
            json={"content": "v2"},
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["version"] == create_resp.json()["version"] + 1

        r = client.delete(
            f"{SKILLS_PREFIX}/del-stale",
            headers={"If-Match": str(create_resp.json()["version"])},
        )
        assert r.status_code == 409

        still_there = client.get(f"{SKILLS_PREFIX}/del-stale")
        assert still_there.status_code == 200, still_there.text
        assert still_there.json()["version"] == update_resp.json()["version"]
```

- [ ] **Step 3: Add failing context-summary version assertion**

Update `TestContextPayload.test_get_context_payload`:

```python
        listed = {skill["name"]: skill for skill in data["available_skills"]}
        assert listed["ctx-skill"]["version"] == 1
        assert "version" not in data["context_text"].lower()
```

If another existing context skill includes the word `version`, narrow the prompt assertion to the test skill line:

```python
        ctx_line = next(line for line in data["context_text"].splitlines() if "ctx-skill" in line)
        assert "version" not in ctx_line.lower()
```

- [ ] **Step 4: Run backend tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  -k "list_skills_includes_version or delete_skill_accepts_matching_if_match or delete_skill_stale_if_match_returns_409_and_keeps_skill or get_context_payload" -v
```

Expected before implementation: list/context version assertions fail because `SkillSummary` omits `version`. Delete tests may already pass because endpoint/service enforcement exists; keep them as API contract coverage.

- [ ] **Step 5: Add backend schema and endpoint implementation**

In `SkillSummary`, add:

```python
    version: int = Field(..., description="Version for optimistic locking")
```

In `_metadata_to_summary()`, add:

```python
        version=metadata.version,
```

In `_build_context_payload()`, add:

```python
                    "version": int(s.get("version") or 1),
```

Do not add version text to `context_text`.

- [ ] **Step 6: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py \
  -k "list_skills or delete_skill or get_context_payload or async_variant_uses_async_context_payload" -v
```

Expected: PASS. If `test_skill_mcp_integration.py` fails because a mocked `available_skills` item is now treated as a `SkillSummary`, update the mock item with the required fields and `version: 1`; do not broaden production schema optionality.

- [ ] **Step 7: Commit backend contract**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/skills_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/app/core/Skills/skills_service.py \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py
git commit -m "TASK-530.10 add skills delete version API coverage"
```

If `test_skill_mcp_integration.py` is unchanged, omit it from `git add`.

## Task 2: Frontend API Client Version Header

**Files:**
- Modify: `apps/packages/ui/src/types/skill.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`

- [ ] **Step 1: Add failing API client tests**

Add tests in `workspace-api.skills.test.ts`:

```ts
  it("sends If-Match when deleting a skill with a valid version", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
    }

    await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize", 3)

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize",
      method: "DELETE",
      headers: { "If-Match": "3" }
    })
  })

  it("omits If-Match when deleting a skill without a known version", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
    const clientCore = {
      resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
      fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
    }

    await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize")

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/skills/summarize",
      method: "DELETE"
    })
  })

  it.each([Number.NaN, 0, -1, 1.5, Number.POSITIVE_INFINITY])(
    "omits If-Match for invalid delete version %s",
    async (version) => {
      vi.mocked(bgRequest).mockResolvedValueOnce(undefined)
      const clientCore = {
        resolveApiPath: vi.fn().mockResolvedValue("/api/v1/skills/{name}"),
        fillPathParams: vi.fn().mockReturnValue("/api/v1/skills/summarize")
      }

      await workspaceApiMethods.deleteSkill.call(clientCore as any, "summarize", version)

      expect(bgRequest).toHaveBeenCalledWith({
        path: "/api/v1/skills/summarize",
        method: "DELETE"
      })
    }
  )
```

- [ ] **Step 2: Run API client tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot
```

Expected before implementation: FAIL because `deleteSkill` does not accept/send versions.

- [ ] **Step 3: Implement API client header guard**

In `SkillSummary`, add:

```ts
  version: number
```

In `workspace-api.ts`, change `deleteSkill` to:

```ts
  async deleteSkill(
    this: TldwApiClientCore,
    name: string,
    version?: number
  ): Promise<void> {
    const base = await this.resolveApiPath("skills.delete", [
      "/api/v1/skills/{name}",
      "/api/v1/skills/{name}/"
    ])
    const path = this.fillPathParams(base, name)
    const headers =
      Number.isSafeInteger(version) && Number(version) > 0
        ? { "If-Match": String(version) }
        : undefined
    await bgRequest<any>({
      path,
      method: "DELETE",
      ...(headers ? { headers } : {})
    })
  },
```

- [ ] **Step 4: Run API client tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot
```

Expected: PASS.

- [ ] **Step 5: Commit API client contract**

```bash
git add \
  apps/packages/ui/src/types/skill.ts \
  apps/packages/ui/src/services/tldw/domains/workspace-api.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts
git commit -m "TASK-530.10 send skill delete If-Match header"
```

## Task 3: Skills Manager Versioned Delete UX

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- Modify: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Update test fixture helper with versions**

In `makeSkill`, add:

```ts
  version: index + 1
```

For existing inline skill fixtures in this file, add `version` when TypeScript or assertions require it. Keep one delete test using a local cast to exercise unknown-version compatibility.

- [ ] **Step 2: Add failing versioned delete test**

Add a test near the seed/delete action tests:

```ts
  it("passes the row version when deleting a skill", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(2)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockResolvedValueOnce(undefined)

    try {
      renderManager()
      await screen.findByText("skill-2")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-2" }))

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith("skill-2", 3)
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })
```

- [ ] **Step 3: Add failing unknown-version compatibility test**

```ts
  it("keeps delete compatible when a row has no known version", async () => {
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const legacySkill = { ...makeSkill(4) } as Partial<ReturnType<typeof makeSkill>>
    delete legacySkill.version
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [legacySkill],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockResolvedValueOnce(undefined)

    try {
      renderManager()
      await screen.findByText("skill-4")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-4" }))

      await waitFor(() => {
        expect(tldwClientMock.deleteSkill).toHaveBeenCalledWith("skill-4", undefined)
      })
    } finally {
      confirmSpy.mockRestore()
    }
  })
```

If TypeScript rejects the partial row in `mockResolvedValueOnce`, cast only that `skills` value to `any` in the test. Do not make production `SkillSummary.version` optional.

- [ ] **Step 4: Add failing stale conflict recovery test**

```ts
  it("shows reload-before-delete guidance on stale delete conflict", async () => {
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    const confirmSpy = vi.spyOn(Modal, "confirm").mockImplementationOnce(
      (config) => {
        void config.onOk?.()
        return { destroy: vi.fn(), update: vi.fn() } as any
      }
    )
    const conflict = Object.assign(new Error("409 version conflict"), { status: 409 })
    tldwClientMock.listSkills.mockResolvedValueOnce({
      skills: [makeSkill(1)],
      count: 1,
      total: 1,
      limit: 10,
      offset: 0
    })
    tldwClientMock.deleteSkill.mockRejectedValueOnce(conflict)

    try {
      renderManager()
      await screen.findByText("skill-1")
      fireEvent.click(screen.getByRole("button", { name: "Delete skill-1" }))

      await waitFor(() => {
        expect(notificationMock.error).toHaveBeenCalledWith(
          expect.objectContaining({
            message: "Skill changed elsewhere",
            description: "Reload skills before deleting this version."
          })
        )
      })
      expect(invalidateSpy).toHaveBeenCalledWith(
        expect.objectContaining({ queryKey: ["skills"] })
      )
    } finally {
      confirmSpy.mockRestore()
      invalidateSpy.mockRestore()
    }
  })
```

- [ ] **Step 5: Run manager tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected before implementation: FAIL because delete calls omit version and stale conflicts use generic copy.

- [ ] **Step 6: Implement local conflict helper**

Near `getErrorDescription`, add:

```ts
const isConflictError = (error: unknown): boolean => {
  const candidate = error as {
    status?: unknown
    statusCode?: unknown
    response?: { status?: unknown }
    message?: unknown
  } | null
  if (!candidate) return false
  return candidate.status === 409
    || candidate.statusCode === 409
    || candidate.response?.status === 409
    || (typeof candidate.message === "string" && candidate.message.includes("409"))
}
```

- [ ] **Step 7: Implement mutation payload and conflict feedback**

Add:

```ts
interface DeleteSkillPayload {
  name: string
  version?: number
}
```

Change delete mutation:

```ts
  const deleteMutation = useMutation({
    mutationFn: ({ name, version }: DeleteSkillPayload) =>
      tldwClient.deleteSkill(name, version),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      setSuccessAction(null)
      notification.success({
        message: t("option:skills.deleteSuccess", { defaultValue: "Skill deleted" })
      })
    },
    onError: (err: unknown) => {
      if (isConflictError(err)) {
        queryClient.invalidateQueries({ queryKey: ["skills"] })
        notification.error({
          message: t("option:skills.deleteConflict", {
            defaultValue: "Skill changed elsewhere"
          }),
          description: t("option:skills.deleteConflictDesc", {
            defaultValue: "Reload skills before deleting this version."
          })
        })
        return
      }
      notification.error({
        message: t("option:skills.deleteError", { defaultValue: "Failed to delete skill" }),
        description: getErrorDescription(err)
      })
    }
  })
```

The reviewer advisory is satisfied because the component already uses `queryClient.invalidateQueries({ queryKey: ["skills"] })` for create/import/delete success paths, and TanStack Query prefix matching invalidates the paginated/filter-specific skills query keys.

- [ ] **Step 8: Pass row version through the confirmation**

Change `handleDelete`:

```ts
  const handleDelete = (skill: Pick<SkillSummary, "name"> & Partial<Pick<SkillSummary, "version">>) => {
    Modal.confirm({
      title: t("option:skills.deleteConfirmTitle", {
        defaultValue: "Delete skill?"
      }),
      content: t("option:skills.deleteConfirmContent", {
        defaultValue: `Are you sure you want to delete "${skill.name}"? This cannot be undone.`,
        name: skill.name
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => deleteMutation.mutateAsync({
        name: skill.name,
        version: Number.isSafeInteger(skill.version) && Number(skill.version) > 0
          ? skill.version
          : undefined
      })
    })
  }
```

Change row action:

```ts
onClick={() => handleDelete(record)}
```

- [ ] **Step 9: Run manager tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot
```

Expected: PASS.

- [ ] **Step 10: Commit manager UX**

```bash
git add \
  apps/packages/ui/src/components/Option/Skills/Manager.tsx \
  apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
git commit -m "TASK-530.10 add skills delete conflict recovery"
```

## Task 4: Full Verification And Task Finalization

**Files:**
- Modify: `backlog/tasks/task-530.10 - Implement-Skills-version-aware-delete-path.md`

- [ ] **Step 1: Run full focused backend verification**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Skills/integration/test_skills_api.py \
  tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py \
  -k "list_skills or delete_skill or get_context_payload or async_variant_uses_async_context_payload" -v
```

Expected: PASS.

- [ ] **Step 2: Run full focused frontend verification**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/tldw/domains/__tests__/workspace-api.skills.test.ts \
  src/components/Option/Skills/__tests__/Manager.test.tsx \
  --reporter=dot
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend Python files**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/skills.py \
  tldw_Server_API/app/api/v1/schemas/skills_schemas.py \
  tldw_Server_API/app/core/Skills/skills_service.py \
  -f json -o /tmp/bandit_task_530_10.json
```

Expected: command completes. Review `/tmp/bandit_task_530_10.json`; fix any new findings in touched code before proceeding.

- [ ] **Step 4: Update Backlog task through MCP**

Record:

- Plan link: `Docs/superpowers/plans/2026-06-28-skills-version-aware-delete.md`
- Files changed.
- Verification command results.
- Bandit output path.
- Any known skips or blockers.
- Final summary.

- [ ] **Step 5: Commit final task note**

```bash
git add "backlog/tasks/task-530.10 - Implement-Skills-version-aware-delete-path.md"
git commit -m "TASK-530.10 record skills delete verification"
```

- [ ] **Step 6: Final self-check before PR**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: clean worktree, commits are scoped to `TASK-530.10`.
