import { describe, expect, it, vi } from "vitest"
import {
  createOwnedWorkspaceDraftStore,
  ownedWorkspaceDraftKey,
  prepareOwnedWorkspaceActivation,
  type OwnedWorkspaceAttempt,
  type OwnedWorkspaceDraft,
  type OwnedWorkspaceScope
} from "../owned-workspace-state"
import type { OwnedWorkspaceBundle } from "../workspace-api"

const scope: OwnedWorkspaceScope = {
  serverBase: "https://research.test/tldw",
  principalId: "2",
  organizationId: null
}
const attempt: OwnedWorkspaceAttempt = {
  scope,
  workspaceId: "workspace-1",
  generation: 2
}
const date = "2026-09-13T12:00:00Z"
const bundle = (): OwnedWorkspaceBundle => ({
  workspace: {
    id: "workspace-1",
    name: "Canonical",
    archived: false,
    deleted: false,
    workspace_profile: "research",
    study_materials_policy: "general",
    version: 4,
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null,
    created_at: date,
    last_modified: date
  },
  sources: [
    {
      id: "source-1",
      workspace_id: "workspace-1",
      media_id: 3,
      title: "Source",
      source_type: "document",
      url: null,
      position: 0,
      selected: false,
      added_at: date,
      version: 1
    }
  ],
  artifacts: [],
  notes: [
    {
      id: 3,
      workspace_id: "workspace-1",
      title: "Server note",
      content: "Latest content",
      keywords_json: '["evidence"]',
      version: 2,
      created_at: date,
      last_modified: date
    }
  ]
})
const draft = (): OwnedWorkspaceDraft => ({
  schemaVersion: 1,
  scope: { ...scope },
  workspaceId: "workspace-1",
  notes: "Scratch text",
  composer: "Unsent question",
  currentNote: {
    title: "New note",
    content: "Unsaved draft",
    keywords: [],
    isDirty: true
  },
  sourceFolders: [],
  sourceFolderMemberships: [],
  selectedSourceFolderIds: [],
  activeFolderId: null,
  leftPaneCollapsed: false,
  rightPaneCollapsed: true,
  pendingChanges: {}
})
function memoryStorage() {
  const values = new Map<string, string>()
  return {
    values,
    get length() {
      return values.size
    },
    key: (index: number) => [...values.keys()][index] ?? null,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value)
    },
    removeItem: (key: string) => {
      values.delete(key)
    }
  }
}

describe("owned workspace scope and draft persistence", () => {
  it.each([
    "{",
    JSON.stringify({ ...draft(), scope: { ...scope, principalId: "foreign" } })
  ])(
    "rejects an invalid or foreign persisted migration baseline",
    (baseline) => {
      const storage = memoryStorage()
      createOwnedWorkspaceDraftStore(() => storage).save(draft())
      const [slot, raw] = [...storage.values.entries()][0]
      const value = JSON.stringify({
        ...JSON.parse(raw),
        legacyBaseline: baseline
      })
      storage.setItem(slot, value)
      expect(
        createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
          .status
      ).toBe("invalid")
      expect(storage.getItem(slot)).toBe(value)
    }
  )

  it("retains malformed foreign legacy bytes as an explicit conflict without adopting them", () => {
    const storage = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => storage)
    store.save(draft())
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    storage.setItem(key, "{")
    const loaded = store.load(scope, "workspace-1")
    expect(loaded).toMatchObject({
      status: "conflict",
      variants: expect.arrayContaining([
        { revisionId: "legacy", draft: null, raw: "{", durable: true }
      ])
    })
    expect(
      store.save({ ...draft(), composer: "Still recoverable" }).status
    ).toBe("conflict")
    expect(store.remove(scope, "workspace-1").status).toBe("conflict")
    expect(storage.getItem(key)).toBe("{")
  })

  it("reads pre-baseline journals but treats their legacy cache as ambiguous", () => {
    const storage = memoryStorage()
    createOwnedWorkspaceDraftStore(() => storage).save(draft())
    const [slot, raw] = [...storage.values.entries()][0]
    const revision = JSON.parse(raw)
    delete revision.legacyBaseline
    storage.setItem(slot, JSON.stringify(revision))
    const store = createOwnedWorkspaceDraftStore(() => storage)
    expect(store.load(scope, "workspace-1").status).toBe("ready")
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    storage.setItem(key, JSON.stringify(draft()))
    expect(store.load(scope, "workspace-1").status).toBe("conflict")
    expect(
      store.save({ ...draft(), composer: "Modern continued" }).status
    ).toBe("conflict")
    expect(storage.getItem(key)).toBe(JSON.stringify(draft()))
  })

  it.each(["load", "save", "unchanged-save", "remove"])(
    "retains external legacy edits as an explicit conflict on %s",
    (operation) => {
      const storage = memoryStorage()
      const store = createOwnedWorkspaceDraftStore(() => storage)
      const key = ownedWorkspaceDraftKey(scope, "workspace-1")
      store.save(draft())
      const legacy = JSON.stringify({ ...draft(), composer: "Old tab edit" })
      storage.setItem(key, legacy)
      const result =
        operation === "load"
          ? store.load(scope, "workspace-1")
          : operation === "remove"
            ? store.remove(scope, "workspace-1")
            : store.save({
                ...draft(),
                composer:
                  operation === "save" ? "Modern continued" : draft().composer
              })
      expect(result.status).toBe("conflict")
      expect(storage.getItem(key)).toBe(legacy)
      const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
        scope,
        "workspace-1"
      )
      expect(loaded.status).toBe("conflict")
      if (loaded.status !== "conflict") throw new Error("Missing conflict")
      expect(
        loaded.variants.map((item) => item.draft?.composer).sort()
      ).toEqual(
        [
          operation === "save" ? "Modern continued" : draft().composer,
          "Old tab edit"
        ].sort()
      )
      expect(loaded.variants.every((item) => item.durable)).toBe(true)
    }
  )

  it("does not adopt an existing legacy draft on save without a prior load", () => {
    const storage = memoryStorage()
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    const legacy = JSON.stringify({ ...draft(), composer: "Unobserved legacy" })
    storage.setItem(key, legacy)
    const store = createOwnedWorkspaceDraftStore(() => storage)
    expect(store.save(draft()).status).toBe("conflict")
    expect(storage.getItem(key)).toBe(legacy)
    expect(store.load(scope, "workspace-1").status).toBe("conflict")
  })

  it("keeps one exact migrated baseline per writer slot across edits and reload", () => {
    const storage = memoryStorage()
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    const baseline = JSON.stringify(draft(), null, 2)
    storage.setItem(key, baseline)
    const store = createOwnedWorkspaceDraftStore(() => storage)
    store.load(scope, "workspace-1")
    for (let index = 0; index < 20; index++)
      expect(
        store.save({ ...draft(), composer: `Modern ${index}` }).status
      ).toBe("saved")
    expect(storage.getItem(key)).toBe(baseline)
    expect(storage.length).toBe(2)
    const revision = JSON.parse(
      [...storage.values.entries()].find(([slot]) => slot !== key)![1]
    )
    expect(revision.legacyBaseline).toBe(baseline)
    const reloaded = createOwnedWorkspaceDraftStore(() => storage)
    expect(reloaded.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: { composer: "Modern 19" }
    })
    expect(reloaded.save({ ...draft(), composer: "Reload edit" }).status).toBe(
      "saved"
    )
    expect(reloaded.remove(scope, "workspace-1").status).toBe("saved")
    expect(storage.getItem(key)).toBe(baseline)
    expect(storage.length).toBe(3)
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toEqual({ status: "missing", deleted: true })
  })

  it.each(["save", "remove"])(
    "never mutates a legacy edit interleaved with journal %s",
    (operation) => {
      const storage = memoryStorage()
      const key = ownedWorkspaceDraftKey(scope, "workspace-1")
      const legacy = JSON.stringify({
        ...draft(),
        composer: "Interleaved legacy"
      })
      let interleave = false
      const store = createOwnedWorkspaceDraftStore(() => ({
        ...storage,
        setItem(slot, value) {
          if (interleave && slot !== key) storage.setItem(key, legacy)
          storage.setItem(slot, value)
        }
      }))
      store.save(draft())
      interleave = true
      const result =
        operation === "save"
          ? store.save({ ...draft(), composer: "Modern edit" })
          : store.remove(scope, "workspace-1")
      expect(result.status).toBe("conflict")
      expect(storage.getItem(key)).toBe(legacy)
      expect(
        createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
          .status
      ).toBe("conflict")
    }
  )

  it("retains failed modern edits in memory alongside foreign legacy and retries without adopting it", () => {
    const storage = memoryStorage()
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    let failing = false
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem(slot, value) {
        if (failing) throw new Error("Quota")
        storage.setItem(slot, value)
      }
    }))
    store.save(draft())
    const legacy = JSON.stringify({
      ...draft(),
      composer: "Legacy while failing"
    })
    storage.setItem(key, legacy)
    failing = true
    const edit = { ...draft(), composer: "Pending modern" }
    expect(store.save(edit).status).toBe("unavailable")
    const pending = store.load(scope, "workspace-1")
    expect(pending.status).toBe("conflict")
    if (pending.status !== "conflict") throw new Error("Missing conflict")
    expect(pending.variants).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ draft: edit, durable: false }),
        expect.objectContaining({ draft: JSON.parse(legacy), durable: true })
      ])
    )
    failing = false
    expect(store.save(edit).status).toBe("conflict")
    expect(store.hasPendingWrites()).toBe(false)
    expect(store.remove(scope, "workspace-1").status).toBe("conflict")
    expect(storage.getItem(key)).toBe(legacy)
  })

  it.each(["corrupt", "foreign", "wrong-writer", "future-parent"])(
    "fails closed on a %s journal slot without trusting the cache",
    (mode) => {
      const storage = memoryStorage()
      createOwnedWorkspaceDraftStore(() => storage).save(draft())
      const slot = [...storage.values.keys()].find((key) =>
        key.includes(":revisions:")
      )!
      const revision = JSON.parse(storage.getItem(slot)!)
      if (mode === "foreign") revision.draft.scope.principalId = "3"
      if (mode === "wrong-writer")
        revision.writerId = "00000000-0000-4000-8000-000000000000"
      if (mode === "future-parent")
        revision.parents[revision.writerId] = revision.sequence
      const raw = mode === "corrupt" ? "{" : JSON.stringify(revision)
      storage.setItem(slot, raw)
      expect(
        createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
          .status
      ).toBe("invalid")
      expect(storage.getItem(slot)).toBe(raw)
    }
  )

  it.each([
    { principalId: "3" },
    { organizationId: "other" },
    { serverBase: "https://other.test" }
  ])("keeps genuine conflicts scoped independently from %j", (change) => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    const second = createOwnedWorkspaceDraftStore(() => storage)
    first.save({ ...draft(), composer: "A" })
    second.save({ ...draft(), composer: "B" })
    const otherScope = { ...scope, ...change }
    first.save({ ...draft(), scope: otherScope, composer: "Other scope" })
    const reloaded = createOwnedWorkspaceDraftStore(() => storage)
    expect(reloaded.load(scope, "workspace-1").status).toBe("conflict")
    expect(reloaded.load(otherScope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: { composer: "Other scope" }
    })
  })

  it("refuses to remove a fork and never deletes another writer's journal slots", () => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    const second = createOwnedWorkspaceDraftStore(() => storage)
    first.save(draft())
    const firstSlots = [...storage.values.entries()].filter(([key]) =>
      key.includes(":revisions:")
    )
    second.load(scope, "workspace-1")
    second.save({ ...draft(), composer: "B" })
    expect(second.remove(scope, "workspace-1").status).toBe("saved")
    for (const [key, raw] of firstSlots) expect(storage.getItem(key)).toBe(raw)
    first.save({ ...draft(), composer: "Concurrent edit" })
    const before = [...storage.values.entries()]
    expect(second.remove(scope, "workspace-1").status).toBe("conflict")
    expect([...storage.values.entries()]).toEqual(before)
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("conflict")
    if (loaded.status !== "conflict") throw new Error("Missing conflict")
    expect(
      loaded.variants.map((variant) => variant.draft?.composer ?? null)
    ).toEqual(expect.arrayContaining([null, "Concurrent edit"]))
  })

  it("does not reparent pending retries onto a newer other-window edit", () => {
    const storage = memoryStorage()
    let failing = true
    const first = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem(slot, value) {
        storage.setItem(slot, value)
        if (failing) throw new Error("Journal commit acknowledgement failed")
      }
    }))
    expect(first.save(draft()).status).toBe("unavailable")
    const second = createOwnedWorkspaceDraftStore(() => storage)
    second.load(scope, "workspace-1")
    second.save({ ...draft(), composer: "Other window continuation" })
    first.load(scope, "workspace-1")
    failing = false
    first.save(draft())
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("conflict")
    if (loaded.status !== "conflict") throw new Error("Missing conflict")
    expect(
      loaded.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Other window continuation", "Unsent question"])
  })
  it("repersists an unchanged draft after another window clears storage", () => {
    const storage = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => storage)
    store.save(draft())
    storage.values.clear()
    expect(store.save(draft()).status).toBe("saved")
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ status: "ready", durable: true, draft: draft() })
  })

  it("retains legacy durable recovery when its deletion journal write fails", () => {
    const storage = memoryStorage()
    storage.setItem(
      ownedWorkspaceDraftKey(scope, "workspace-1"),
      JSON.stringify(draft())
    )
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem() {
        throw new Error("Quota")
      }
    }))
    expect(store.remove(scope, "workspace-1").status).toBe("unavailable")
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ status: "ready", durable: true, draft: draft() })
  })

  it("supports writer identities on HTTP origins without randomUUID", () => {
    const storage = memoryStorage()
    const randomUUID = vi.spyOn(crypto, "randomUUID").mockImplementation(() => {
      throw new Error("Unavailable on HTTP")
    })
    try {
      const store = createOwnedWorkspaceDraftStore(() => storage)
      expect(store.save(draft()).status).toBe("saved")
      expect(store.load(scope, "workspace-1")).toMatchObject({
        status: "ready",
        draft: draft()
      })
    } finally {
      randomUUID.mockRestore()
    }
  })
  it("does not fork when another window resaves its unchanged loaded draft", () => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    first.save(draft())
    const second = createOwnedWorkspaceDraftStore(() => storage)
    second.load(scope, "workspace-1")
    second.save(draft())
    first.save({ ...draft(), composer: "Only real edit" })
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ status: "ready", draft: { composer: "Only real edit" } })
  })

  it("does not discard memory recovery when a deletion journal write fails", () => {
    const storage = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem() {
        throw new Error("Quota")
      }
    }))
    store.save(draft())
    expect(store.remove(scope, "workspace-1").status).toBe("unavailable")
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      durable: false,
      draft: draft()
    })
  })
  it("retains independent edits as recoverable conflict heads after reload", () => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    const second = createOwnedWorkspaceDraftStore(() => storage)
    first.load(scope, "workspace-1")
    second.load(scope, "workspace-1")
    first.save({ ...draft(), composer: "Window A" })
    second.save({ ...draft(), composer: "Window B" })
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("conflict")
    if (loaded.status !== "conflict") throw new Error("Expected conflict")
    expect(
      loaded.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Window A", "Window B"])
    first.load(scope, "workspace-1")
    first.save({ ...draft(), composer: "Window A continued" })
    const retried = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(retried).toMatchObject({ status: "conflict" })
    if (retried.status !== "conflict") throw new Error("Expected conflict")
    expect(
      retried.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Window A continued", "Window B"])
  })

  it("preserves both edits when another store writes inside a storage write", () => {
    const storage = memoryStorage()
    const second = createOwnedWorkspaceDraftStore(() => storage)
    second.load(scope, "workspace-1")
    let interleave = true
    const first = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem(key, value) {
        if (interleave) {
          interleave = false
          second.save({ ...draft(), composer: "Interleaved B" })
        }
        storage.setItem(key, value)
      }
    }))
    first.load(scope, "workspace-1")
    first.save({ ...draft(), composer: "Interleaved A" })
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("conflict")
    if (loaded.status !== "conflict") throw new Error("Expected conflict")
    expect(
      loaded.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Interleaved A", "Interleaved B"])
  })

  it("keeps journal storage bounded by writers, not keystrokes, across reloads", () => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    first.save(draft())
    const initialSlots = storage.length
    for (let index = 0; index < 30; index++)
      first.save({ ...draft(), composer: `Edit ${index}` })
    expect(storage.length).toBe(initialSlots)
    const reloaded = createOwnedWorkspaceDraftStore(() => storage)
    expect(reloaded.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: { composer: "Edit 29" }
    })
    reloaded.save({ ...draft(), composer: "After reload" })
    expect(first.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: { composer: "After reload" }
    })
  })

  it("retains a failed journal write in memory and preserves a competing edit on retry", () => {
    const storage = memoryStorage()
    let failing = true
    const first = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem(key, value) {
        if (failing && key !== ownedWorkspaceDraftKey(scope, "workspace-1"))
          throw new Error("Journal quota")
        storage.setItem(key, value)
      }
    }))
    const second = createOwnedWorkspaceDraftStore(() => storage)
    first.load(scope, "workspace-1")
    second.load(scope, "workspace-1")
    const value = { ...draft(), composer: "Recover A" }
    expect(first.save(value).status).toBe("unavailable")
    expect(first.hasPendingWrite(scope, "workspace-1")).toBe(true)
    expect(first.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      durable: false,
      draft: value
    })
    second.save({ ...draft(), composer: "Durable B" })
    expect(first.load(scope, "workspace-1").status).toBe("conflict")
    failing = false
    expect(first.save(value).status).toBe("conflict")
    expect(first.hasPendingWrites()).toBe(false)
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("conflict")
    if (loaded.status !== "conflict") throw new Error("Expected conflict")
    expect(
      loaded.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Durable B", "Recover A"])
  })

  it("reads a legacy draft and then recovers the journal without its compatibility cache", () => {
    const storage = memoryStorage()
    const key = ownedWorkspaceDraftKey(scope, "workspace-1")
    storage.setItem(key, JSON.stringify(draft()))
    const store = createOwnedWorkspaceDraftStore(() => storage)
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: draft()
    })
    store.save({ ...draft(), composer: "Journal authoritative" })
    storage.removeItem(key)
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({
      status: "ready",
      draft: { composer: "Journal authoritative" }
    })
  })

  it("restores banner text but rejects image data in a persisted banner draft", () => {
    const storage = memoryStorage()
    const value = {
      ...draft(),
      bannerDraft: { title: "Draft", subtitle: "Subtitle", baseVersion: 2 }
    }
    expect(
      createOwnedWorkspaceDraftStore(() => storage).save(value).status
    ).toBe("saved")
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("ready")
    if (loaded.status !== "ready") throw new Error("Missing draft")
    expect(loaded.draft.bannerDraft).toEqual(value.bannerDraft)
    storage.values.clear()
    storage.setItem(
      ownedWorkspaceDraftKey(scope, "workspace-1"),
      JSON.stringify({
        ...value,
        bannerDraft: { ...value.bannerDraft, image: "data:image/png;base64,a" }
      })
    )
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
        .status
    ).toBe("invalid")
  })
  it("recovers assistant selection and base version but never persisted consent", () => {
    const storage = memoryStorage()
    const value = {
      ...draft(),
      assistantDraft: {
        assistantId: "persona-1",
        personaMemoryMode: "read_write" as const,
        baseVersion: 2
      }
    }
    expect(
      createOwnedWorkspaceDraftStore(() => storage).save(value).status
    ).toBe("saved")
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("ready")
    if (loaded.status !== "ready") throw new Error("Missing draft")
    expect(loaded.draft.assistantDraft).toEqual(value.assistantDraft)
    storage.values.clear()
    storage.setItem(
      ownedWorkspaceDraftKey(scope, "workspace-1"),
      JSON.stringify({
        ...value,
        assistantDraft: { ...value.assistantDraft, confirmed: true }
      })
    )
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
        .status
    ).toBe("invalid")
  })

  it("recovers a versioned rename draft without replaying it as canonical metadata", () => {
    const storage = memoryStorage()
    const value = {
      ...draft(),
      renameDraft: { name: "Unsent rename", baseVersion: 2 }
    }
    expect(
      createOwnedWorkspaceDraftStore(() => storage).save(value).status
    ).toBe("saved")
    const loaded = createOwnedWorkspaceDraftStore(() => storage).load(
      scope,
      "workspace-1"
    )
    expect(loaded.status).toBe("ready")
    if (loaded.status !== "ready") throw new Error("Missing draft")
    const activated = prepareOwnedWorkspaceActivation(
      attempt,
      attempt,
      bundle(),
      loaded.draft
    )
    expect(activated.status).toBe("activated")
    if (activated.status !== "activated") throw new Error("Not activated")
    expect(activated.draft?.renameDraft).toEqual(value.renameDraft)
    expect(activated.bundle.workspace.name).toBe("Canonical")
  })

  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "rejects an invalid persisted rename base version %s",
    (baseVersion) => {
      const storage = memoryStorage()
      storage.setItem(
        ownedWorkspaceDraftKey(scope, "workspace-1"),
        JSON.stringify({
          ...draft(),
          renameDraft: { name: "Draft", baseVersion }
        })
      )
      expect(
        createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
          .status
      ).toBe("invalid")
    }
  )

  it("normalizes equivalent origins and trailing slashes without merging deployment paths", () => {
    expect(ownedWorkspaceDraftKey(scope, "workspace-1")).toBe(
      ownedWorkspaceDraftKey(
        { ...scope, serverBase: "https://RESEARCH.test:443/tldw/" },
        "workspace-1"
      )
    )
    expect(ownedWorkspaceDraftKey(scope, "workspace-1")).not.toBe(
      ownedWorkspaceDraftKey(
        { ...scope, serverBase: "https://research.test/other" },
        "workspace-1"
      )
    )
  })

  it.each([
    { principalId: "3" },
    { organizationId: "org-1" },
    { serverBase: "https://other.test/tldw" }
  ])("isolates identical workspace IDs across %j", (change) => {
    const memory = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => memory)
    store.save(draft())
    expect(store.load({ ...scope, ...change }, "workspace-1").status).toBe(
      "missing"
    )
  })

  it.each([
    "https://user:secret@research.test/tldw",
    "https://research.test?token=secret",
    "https://research.test#token",
    "file:///tmp/project"
  ])("rejects credential-bearing or invalid server bases: %s", (serverBase) => {
    expect(() =>
      ownedWorkspaceDraftKey({ ...scope, serverBase }, "workspace-1")
    ).toThrow()
  })

  it("round-trips a draft across repository instances without legacy workspace writes", () => {
    const storage = memoryStorage()
    expect(
      createOwnedWorkspaceDraftStore(() => storage).save(draft()).status
    ).toBe("saved")
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ status: "ready", durable: true, draft: draft() })
    expect(
      [...storage.values.keys()].every((key) =>
        key.startsWith(ownedWorkspaceDraftKey(scope, "workspace-1"))
      )
    ).toBe(true)
  })

  it("preserves unsaved server-field changes for explicit reconciliation", () => {
    const storage = memoryStorage()
    const value = {
      ...draft(),
      pendingChanges: {
        workspaceName: "Unsaved title",
        selectedSourceIds: ["source-1"]
      }
    }
    createOwnedWorkspaceDraftStore(() => storage).save(value)
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ draft: value })
  })

  it("retains a detached in-memory draft when storage fails and retries persistence", () => {
    const storage = memoryStorage()
    let failing = true
    const store = createOwnedWorkspaceDraftStore(() => {
      if (failing) throw new Error("Storage unavailable")
      return storage
    })
    const value = draft()
    expect(store.save(value).status).toBe("unavailable")
    value.currentNote.content = "Later mutation"
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      durable: false,
      draft: { currentNote: { content: "Unsaved draft" } }
    })
    failing = false
    expect(store.save(draft()).status).toBe("saved")
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
        .status
    ).toBe("ready")
  })

  it("reads fresh persisted content after a successful save instead of keeping a stale tab cache", () => {
    const storage = memoryStorage()
    const first = createOwnedWorkspaceDraftStore(() => storage)
    const second = createOwnedWorkspaceDraftStore(() => storage)
    first.save(draft())
    second.load(scope, "workspace-1")
    second.save({ ...draft(), composer: "From another tab" })
    expect(first.load(scope, "workspace-1")).toMatchObject({
      draft: { composer: "From another tab" }
    })
  })

  it("prefers detached failed-write recovery over an older durable draft", () => {
    const storage = memoryStorage()
    createOwnedWorkspaceDraftStore(() => storage).save(draft())
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem: () => {
        throw new Error("Quota exceeded")
      }
    }))
    const newer = { ...draft(), composer: "Latest unsent question" }
    store.load(scope, "workspace-1")
    expect(store.save(newer).status).toBe("unavailable")
    newer.composer = "Later caller mutation"
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      durable: false,
      draft: { composer: "Latest unsent question" }
    })
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
    ).toMatchObject({ durable: true, draft: { composer: "Unsent question" } })
  })

  it("preserves a failed-write memory draft when deletion also fails", () => {
    const storage = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem: () => {
        throw new Error("Quota exceeded")
      },
      removeItem: () => {
        throw new Error("Denied")
      }
    }))
    expect(store.save(draft()).status).toBe("unavailable")
    expect(store.remove(scope, "workspace-1").status).toBe("unavailable")
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      durable: false,
      draft: draft()
    })
    expect(storage.values.size).toBe(0)
  })

  it.each([
    "{",
    JSON.stringify({ ...draft(), schemaVersion: 99 }),
    JSON.stringify({ ...draft(), scope: { ...scope, principalId: "3" } })
  ])(
    "rejects corrupt, unsupported, or wrong-owner persisted drafts without deleting them",
    (value) => {
      const storage = memoryStorage()
      storage.setItem(ownedWorkspaceDraftKey(scope, "workspace-1"), value)
      expect(
        createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
          .status
      ).toBe("invalid")
      expect(
        storage.getItem(ownedWorkspaceDraftKey(scope, "workspace-1"))
      ).toBe(value)
    }
  )

  it("retains recovery state when deletion fails", () => {
    const storage = memoryStorage()
    let failing = false
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      setItem: (key, value) => {
        if (failing) throw new Error("Denied")
        storage.setItem(key, value)
      }
    }))
    store.save(draft())
    failing = true
    expect(store.remove(scope, "workspace-1").status).toBe("unavailable")
    expect(store.load(scope, "workspace-1").status).toBe("ready")
  })

  it("keeps deletion authoritative without calling legacy removal", () => {
    const storage = memoryStorage()
    const store = createOwnedWorkspaceDraftStore(() => ({
      ...storage,
      removeItem() {
        throw new Error("Denied")
      }
    }))
    store.save(draft())
    expect(store.remove(scope, "workspace-1").status).toBe("saved")
    expect(
      createOwnedWorkspaceDraftStore(() => storage).load(scope, "workspace-1")
        .status
    ).toBe("missing")
  })

  it("round-trips through browser storage without authorizing or activating a workspace", () => {
    const store = createOwnedWorkspaceDraftStore(() => window.localStorage)
    const value = draft()
    expect(store.save(value).status).toBe("saved")
    expect(store.load(scope, "workspace-1")).toMatchObject({
      status: "ready",
      draft: value
    })
    expect(store.remove(scope, "workspace-1").status).toBe("saved")
    expect(store.load(scope, "workspace-1").status).toBe("missing")
  })

  it.each(["foreign", "cycle", "duplicate", "missing-parent"])(
    "rejects %s folder metadata without overwriting the prior draft",
    (mode) => {
      const storage = memoryStorage()
      const store = createOwnedWorkspaceDraftStore(() => storage)
      store.save(draft())
      const value = draft()
      const folder = {
        id: "folder-1",
        workspaceId: "workspace-1",
        name: "Folder",
        parentFolderId: null as string | null,
        createdAt: date,
        updatedAt: date
      }
      if (mode === "foreign") folder.workspaceId = "another-workspace"
      if (mode === "cycle") folder.parentFolderId = folder.id
      if (mode === "missing-parent") folder.parentFolderId = "gone"
      value.sourceFolders = mode === "duplicate" ? [folder, folder] : [folder]
      expect(store.save(value).status).toBe("invalid")
      expect(store.load(scope, "workspace-1")).toMatchObject({ draft: draft() })
    }
  )

  it("reports blocked reads separately from an empty draft store", () => {
    const store = createOwnedWorkspaceDraftStore(() => {
      throw new Error("Blocked")
    })
    expect(store.load(scope, "workspace-1").status).toBe("unavailable")
  })
})

describe("owned workspace activation preparation", () => {
  it("cannot activate after the attempt was invalidated", () => {
    expect(
      prepareOwnedWorkspaceActivation(attempt, null, bundle(), draft()).status
    ).toBe("stale")
  })

  it("opens canonical content without a recovered draft", () => {
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle())
    ).toMatchObject({ status: "activated", draft: null })
  })

  it("does not resurrect deleted or archived targets", () => {
    for (const field of ["deleted", "archived"] as const) {
      const server = bundle()
      server.workspace[field] = true
      expect(
        prepareOwnedWorkspaceActivation(attempt, attempt, server, draft())
          .status
      ).toBe("stale")
    }
  })

  it("keeps the canonical bundle separate from a recovered unsent draft", () => {
    const server = bundle()
    const result = prepareOwnedWorkspaceActivation(
      attempt,
      attempt,
      server,
      draft()
    )
    expect(result).toMatchObject({
      status: "activated",
      bundle: server,
      draft: {
        composer: "Unsent question",
        currentNote: { content: "Unsaved draft" }
      }
    })
    if (result.status !== "activated") throw new Error("Activation failed")
    result.bundle.workspace.name = "Mutated copy"
    expect(server.workspace.name).toBe("Canonical")
    expect(result.bundle.sources[0].selected).toBe(false)
  })

  it.each([
    { ...attempt, generation: 1 },
    { ...attempt, workspaceId: "workspace-2" },
    { ...attempt, scope: { ...scope, principalId: "3" } },
    { ...attempt, scope: { ...scope, organizationId: "org-1" } }
  ])("rejects stale generation, target, or scope", (stale) => {
    expect(
      prepareOwnedWorkspaceActivation(stale, attempt, bundle(), draft()).status
    ).toBe("stale")
  })

  it("does not accept a recovered draft from a different scope", () => {
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), {
        ...draft(),
        scope: { ...scope, principalId: "3" }
      }).status
    ).toBe("stale")
  })

  it("does not auto-apply pending canonical edits", () => {
    const value = {
      ...draft(),
      pendingChanges: { workspaceName: "Unsaved rename" }
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({
      status: "draft-conflict",
      reason: "server-edits",
      draft: value
    })
  })

  it("restores dirty note text only when the canonical note version still matches", () => {
    const value = {
      ...draft(),
      currentNote: { ...draft().currentNote, id: 3, version: 2 }
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value).status
    ).toBe("activated")
    value.currentNote.version = 1
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({
      status: "draft-conflict",
      reason: "note-changed",
      draft: value
    })
  })

  it("preserves a dirty draft for a deleted note as a conflict", () => {
    const value = {
      ...draft(),
      currentNote: { ...draft().currentNote, id: 9, version: 1 }
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({ status: "draft-conflict", reason: "note-missing" })
  })

  it("refreshes a clean selected note from canonical associations", () => {
    const value = {
      ...draft(),
      currentNote: { ...draft().currentNote, id: 3, version: 1, isDirty: false }
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({
      status: "activated",
      draft: {
        currentNote: {
          title: "Server note",
          content: "Latest content",
          version: 2,
          keywords: ["evidence"],
          isDirty: false
        }
      }
    })
  })

  it("does not restore stale clean note text when its association disappeared", () => {
    const value = {
      ...draft(),
      currentNote: { ...draft().currentNote, id: 9, version: 1, isDirty: false }
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({
      status: "activated",
      draft: { currentNote: { content: "", isDirty: false } }
    })
  })

  it("prunes folder membership references not present in the authorized source set", () => {
    const value: OwnedWorkspaceDraft = {
      ...draft(),
      sourceFolders: [
        {
          id: "folder-1",
          workspaceId: "workspace-1",
          name: "My folder",
          parentFolderId: null,
          createdAt: date,
          updatedAt: date
        }
      ],
      sourceFolderMemberships: [
        { folderId: "folder-1", sourceId: "source-1" },
        { folderId: "folder-1", sourceId: "gone" },
        { folderId: "gone", sourceId: "source-1" }
      ],
      selectedSourceFolderIds: ["folder-1", "gone"],
      activeFolderId: "gone"
    }
    expect(
      prepareOwnedWorkspaceActivation(attempt, attempt, bundle(), value)
    ).toMatchObject({
      status: "activated",
      draft: {
        sourceFolderMemberships: [
          { folderId: "folder-1", sourceId: "source-1" }
        ],
        selectedSourceFolderIds: ["folder-1"],
        activeFolderId: null
      }
    })
    expect(value.sourceFolderMemberships).toHaveLength(3)
  })
})
