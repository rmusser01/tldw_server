import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "../workspace"
import { WORKSPACE_STORAGE_KEY } from "../workspace-events"
import {
  createOwnedWorkspaceDraftStore,
  ownedWorkspaceDraftKey
} from "../owned-workspace-state"
import type { OwnedWorkspaceBundle } from "../workspace-api"

const scope = {
  serverBase: "https://research.test/tldw",
  principalId: "2",
  organizationId: null
}
const date = "2026-09-13T12:00:00Z"
const bundle = (id = "server-1"): OwnedWorkspaceBundle => ({
  workspace: {
    id,
    name: "Server title",
    archived: false,
    deleted: false,
    workspace_profile: "research",
    study_materials_policy: "general",
    version: 1,
    banner_title: "Banner",
    banner_subtitle: "Subtitle",
    banner_color: "red",
    audio_provider: "custom-provider",
    audio_model: "model",
    audio_voice: "voice",
    audio_speed: 1,
    created_at: date,
    last_modified: date,
    effectiveAssistantDefault: {
      status: "none",
      source: "none",
      assistantKind: null,
      assistantId: null,
      label: null,
      personaMemoryMode: null,
      degradedReason: null
    }
  },
  sources: [
    {
      id: "source-1",
      workspace_id: id,
      media_id: 3,
      title: "Canonical source",
      source_type: "document",
      url: null,
      position: 0,
      selected: true,
      added_at: date,
      version: 1
    }
  ],
  artifacts: [],
  notes: [
    {
      id: 3,
      workspace_id: id,
      title: "Note",
      content: "Canonical note",
      keywords_json: "[]",
      version: 2,
      created_at: date,
      last_modified: date
    }
  ]
})
const state = () => useWorkspaceStore.getState()
const open = (id = "server-1", account = scope) => {
  const attempt = state().beginOwnedWorkspace(account, id)
  expect(state().activateOwnedWorkspace(attempt, bundle(id))).toBe("activated")
}
const storedDraft = (id = "server-1", account = scope) => {
  const loaded = createOwnedWorkspaceDraftStore(() => localStorage).load(
    account,
    id
  )
  if (loaded.status !== "ready")
    throw new Error(`Expected durable draft, got ${loaded.status}`)
  expect(loaded.durable).toBe(true)
  return loaded.draft
}

beforeEach(() => {
  vi.restoreAllMocks()
  useWorkspaceStore.setState({ ...initialState, storeHydrated: true })
  localStorage.clear()
})
afterEach(() => vi.restoreAllMocks())

describe("owned workspace archive completion", () => {
  it("blocks archive preparation and retains modern and legacy edits after an old-tab write", () => {
    open()
    state().setOwnedWorkspaceComposer("Modern text")
    const loaded = createOwnedWorkspaceDraftStore(() => localStorage).load(
      scope,
      "server-1"
    )
    if (loaded.status !== "ready") throw new Error("Missing draft")
    const legacy = JSON.stringify({ ...loaded.draft, composer: "Old tab text" })
    const key = ownedWorkspaceDraftKey(scope, "server-1")
    localStorage.setItem(key, legacy)
    state().setOwnedWorkspaceComposer("Modern continued")
    expect(state().ownedWorkspaceDraftStatus).toBe("conflict")
    expect(state().prepareOwnedWorkspaceArchive(target())).toBe(false)
    expect(localStorage.getItem(key)).toBe(legacy)
    const recovered = createOwnedWorkspaceDraftStore(() => localStorage).load(
      scope,
      "server-1"
    )
    expect(recovered.status).toBe("conflict")
    if (recovered.status !== "conflict") throw new Error("Missing conflict")
    expect(
      recovered.variants.map((item) => item.draft?.composer).sort()
    ).toEqual(["Modern continued", "Old tab text"])
  })

  it("does not mistake an explicit draft deletion for cleared storage", () => {
    open()
    state().setOwnedWorkspaceComposer("Previously saved")
    const other = createOwnedWorkspaceDraftStore(() => localStorage)
    expect(other.remove(scope, "server-1").status).toBe("saved")
    open()
    expect(state().ownedWorkspaceComposer).toBe("")
  })

  it("preserves active text on same-target activation after another window clears storage", () => {
    open()
    state().setOwnedWorkspaceComposer("Still in this window")
    localStorage.clear()
    open()
    expect(state().ownedWorkspaceComposer).toBe("Still in this window")
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "server-1")
    ).toMatchObject({
      status: "ready",
      draft: { composer: "Still in this window" }
    })
  })

  it("fails closed on competing window drafts during same-target activation and reload", () => {
    open()
    const other = createOwnedWorkspaceDraftStore(() => localStorage)
    const loaded = other.load(scope, "server-1")
    if (loaded.status !== "ready") throw new Error("Missing initial draft")
    other.save({ ...loaded.draft, composer: "Window B edit" })
    state().setOwnedWorkspaceComposer("Window A edit")
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    expect(state().activateOwnedWorkspace(attempt, bundle())).toBe(
      "draft-conflict"
    )
    expect(state().ownedWorkspaceConflict).toBe("concurrent-edits")
    expect(state().ownedWorkspaceComposer).toBe("Window A edit")
    state().invalidateOwnedWorkspace()
    const reopened = state().beginOwnedWorkspace(scope, "server-1")
    expect(state().activateOwnedWorkspace(reopened, bundle())).toBe(
      "draft-conflict"
    )
    expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
    const recovered = createOwnedWorkspaceDraftStore(() => localStorage).load(
      scope,
      "server-1"
    )
    expect(recovered.status).toBe("conflict")
    if (recovered.status !== "conflict") throw new Error("Missing conflict")
    expect(
      recovered.variants.map((variant) => variant.draft?.composer).sort()
    ).toEqual(["Window A edit", "Window B edit"])
  })

  it("does not overwrite a newer other-window draft on unchanged departure", () => {
    open()
    const other = { ...storedDraft(), composer: "Newer text from window B" }
    const otherStore = createOwnedWorkspaceDraftStore(() => localStorage)
    otherStore.load(scope, "server-1")
    otherStore.save(other)
    state().invalidateOwnedWorkspace()
    expect(storedDraft().composer).toBe(other.composer)
    expect(
      createOwnedWorkspaceDraftStore(() => localStorage).load(scope, "server-1")
    ).toMatchObject({ status: "ready", draft: { composer: other.composer } })
  })

  it("does not persist store mutations before initial hydration completes", async () => {
    state().initializeWorkspace("Keep legacy")
    await new Promise((resolve) => setTimeout(resolve, 0))
    const storage = useWorkspaceStore.persist.getOptions().storage!
    useWorkspaceStore.persist.setOptions({
      storage: { ...storage, setItem: () => {} }
    })
    useWorkspaceStore.setState({ ...initialState, storeHydrated: false })
    useWorkspaceStore.persist.setOptions({ storage })
    const before = Object.fromEntries(
      Object.keys(localStorage).map((key) => [key, localStorage.getItem(key)])
    )
    useWorkspaceStore.setState({ workspaceName: "Unhydrated transient value" })
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(
      Object.fromEntries(
        Object.keys(localStorage).map((key) => [key, localStorage.getItem(key)])
      )
    ).toEqual(before)
  })
  const target = () => ({
    origin: state().activeWorkspaceOrigin,
    workspaceId: state().workspaceId
  })
  const receipt = () => ({
    ...target(),
    workspace: { ...bundle().workspace, archived: true, version: 2 }
  })
  it("preserves the latest composer and editor drafts before clearing an archived active view", () => {
    open()
    const input = receipt()
    state().setOwnedWorkspaceComposer("Unsent research")
    expect(state().prepareOwnedWorkspaceArchive(target())).toBe(true)
    state().setOwnedWorkspaceComposer("Newer unsent research")
    expect(state().completeOwnedWorkspaceArchive(input)).toBe(true)
    expect(state().workspaceId).toBe("")
    expect(state().ownedWorkspaceBundle).toBeNull()
    expect(storedDraft().composer).toBe("Newer unsent research")
    expect(state().archivedWorkspaces).toEqual([])
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    expect(
      state().activateOwnedWorkspace(attempt, {
        ...bundle(),
        workspace: { ...bundle().workspace, version: 3 }
      })
    ).toBe("activated")
    expect(state().ownedWorkspaceComposer).toBe("Newer unsent research")
  })
  it("refuses preparation and completion when draft persistence is unavailable", () => {
    open()
    const input = receipt()
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("quota")
      })
    state().setOwnedWorkspaceComposer("Keep me")
    expect(state().prepareOwnedWorkspaceArchive(target())).toBe(false)
    expect(state().completeOwnedWorkspaceArchive(input)).toBe(false)
    expect(state().workspaceId).toBe("server-1")
    expect(state().ownedWorkspaceComposer).toBe("Keep me")
    setItem.mockRestore()
    expect(state().completeOwnedWorkspaceArchive(input)).toBe(true)
    expect(storedDraft().composer).toBe("Keep me")
  })
  it("cannot clear another activation even with the same workspace id", () => {
    open()
    const input = receipt()
    open()
    expect(state().prepareOwnedWorkspaceArchive(input)).toBe(false)
    expect(state().completeOwnedWorkspaceArchive(input)).toBe(false)
    expect(state().workspaceId).toBe("server-1")
  })
  it("does not let another account's pending draft block a durable current draft", () => {
    open("server-1", { ...scope, principalId: "4" })
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("quota")
      })
    state().setOwnedWorkspaceComposer("Account A unsaved text")
    open("server-1", scope)
    setItem.mockRestore()
    state().setOwnedWorkspaceComposer("Account B durable text")
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    const prepared = state().prepareOwnedWorkspaceArchive(target())
    const completed = state().completeOwnedWorkspaceArchive(receipt())
    const currentDraft = storedDraft()
    open("server-1", { ...scope, principalId: "4" })
    expect(state().ownedWorkspaceComposer).toBe("Account A unsaved text")
    expect(prepared).toBe(true)
    expect(completed).toBe(true)
    expect(currentDraft.composer).toBe("Account B durable text")
  })
  it.each([
    { ...bundle().workspace, archived: false, version: 2 },
    { ...bundle().workspace, archived: true, deleted: true, version: 2 },
    { ...bundle().workspace, archived: true, version: 0 },
    { ...bundle().workspace, id: "other", archived: true, version: 2 }
  ])(
    "rejects invalid archive completion without discarding content",
    (workspace) => {
      open()
      expect(() =>
        state().completeOwnedWorkspaceArchive({ ...target(), workspace })
      ).toThrow()
      expect(state().workspaceId).toBe("server-1")
    }
  )
  it("refuses older archive status than an already accepted metadata update", () => {
    open()
    const input = receipt()
    useWorkspaceStore.setState({
      ownedWorkspaceBundle: {
        ...bundle(),
        workspace: { ...bundle().workspace, version: 3 }
      }
    })
    expect(state().completeOwnedWorkspaceArchive(input)).toBe(false)
    expect(state().workspaceId).toBe("server-1")
  })
})

describe("owned workspace banner receipts", () => {
  const edit = (
    title = "New banner",
    subtitle = "New subtitle",
    baseVersion = 1
  ) => {
    const draft = { title, subtitle, baseVersion }
    expect(
      state().setOwnedWorkspaceBannerDraft({
        origin: state().activeWorkspaceOrigin,
        workspaceId: state().workspaceId,
        expectedDraft: state().ownedWorkspaceBannerDraft,
        draft
      })
    ).toBe(true)
    return draft
  }
  const receipt = () => ({
    origin: state().activeWorkspaceOrigin,
    workspaceId: state().workspaceId,
    expectedWorkspace: state().ownedWorkspaceBundle!.workspace,
    submittedDraft: state().ownedWorkspaceBannerDraft!,
    editorSession: state().ownedWorkspaceBannerSession,
    workspace: {
      ...bundle().workspace,
      version: 2,
      banner_title: "New banner",
      banner_subtitle: "New subtitle"
    }
  })
  it("persists scoped text drafts with their base without changing canonical banner", () => {
    open()
    const draft = edit()
    expect(storedDraft().bannerDraft).toEqual(draft)
    expect(state().workspaceBanner).toEqual({
      title: "Banner",
      subtitle: "Subtitle",
      image: null
    })
    expect(storedDraft().pendingChanges).toEqual({})
    open("server-2")
    expect(state().ownedWorkspaceBannerDraft).toBeNull()
    open()
    expect(state().ownedWorkspaceBannerDraft).toEqual(draft)
    state().invalidateOwnedWorkspace()
    open("server-1", { ...scope, principalId: "3" })
    expect(state().ownedWorkspaceBannerDraft).toBeNull()
  })
  it("applies canonical banner text without clearing another editor", () => {
    open()
    edit()
    const rename = { name: "Unsent rename", baseVersion: 1 }
    state().setOwnedWorkspaceRenameDraft({
      ...receipt(),
      expectedDraft: null,
      draft: rename
    })
    expect(state().applyOwnedWorkspaceBannerReceipt(receipt())).toBe("saved")
    expect(state().workspaceBanner).toEqual({
      title: "New banner",
      subtitle: "New subtitle",
      image: null
    })
    expect(state().ownedWorkspaceBundle?.workspace.banner_color).toBe("red")
    expect(state().ownedWorkspaceRenameDraft).toBe(rename)
    expect(state().ownedWorkspaceBannerDraft).toBeNull()
    expect(storedDraft().pendingChanges).toEqual({})
  })
  it("keeps newer banner typing while advancing its base", () => {
    open()
    edit()
    const input = receipt()
    edit("Still editing")
    expect(state().applyOwnedWorkspaceBannerReceipt(input)).toBe("newer-draft")
    expect(state().ownedWorkspaceBannerDraft).toEqual({
      title: "Still editing",
      subtitle: "New subtitle",
      baseVersion: 2
    })
    expect(state().workspaceBanner.title).toBe("New banner")
  })
  it("retains a cancelled and reopened editor after a dispatched receipt", () => {
    open()
    edit()
    const input = receipt()
    state().setOwnedWorkspaceBannerDraft({
      ...input,
      expectedDraft: input.submittedDraft,
      draft: null
    })
    const reopened = edit("Reopened")
    expect(state().applyOwnedWorkspaceBannerReceipt(input)).toBe(
      "editor-changed"
    )
    expect(state().ownedWorkspaceBannerDraft).toBe(reopened)
    expect(state().workspaceBanner.title).toBe("New banner")
  })
  it("accepts explicit unchanged-version review but not an unchanged PATCH", () => {
    open()
    edit()
    const input = { ...receipt(), workspace: bundle().workspace }
    expect(() => state().applyOwnedWorkspaceBannerReceipt(input)).toThrow()
    expect(state().acceptOwnedWorkspaceBannerReview(input)).toBe("saved")
    expect(state().workspaceBanner.title).toBe("Banner")
  })
  it("resets text from a receipt while preserving server color", () => {
    open()
    edit("", "")
    const input = receipt()
    input.workspace.banner_title = ""
    input.workspace.banner_subtitle = ""
    expect(state().applyOwnedWorkspaceBannerReceipt(input)).toBe("saved")
    expect(state().workspaceBanner).toEqual({
      title: "",
      subtitle: "",
      image: null
    })
    expect(state().ownedWorkspaceBundle?.workspace.banner_color).toBe("red")
  })
  it("rejects receipts from an earlier activation", () => {
    open()
    edit()
    const input = receipt()
    open("server-2")
    open()
    expect(state().applyOwnedWorkspaceBannerReceipt(input)).toBe("stale")
    expect(state().workspaceBanner.title).toBe("Banner")
  })
  it("accepts a concurrent notes refresh without overwriting unrelated local edits", () => {
    open()
    edit()
    const input = receipt()
    state().setWorkspaceName("Local pending name")
    state().replaceOwnedWorkspaceNotes({
      ...input,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: []
    })
    expect(state().applyOwnedWorkspaceBannerReceipt(input)).toBe("saved")
    expect(state().workspaceName).toBe("Local pending name")
    expect(state().ownedWorkspaceBundle?.notes).toEqual([])
    expect(state().workspaceBanner.title).toBe("New banner")
  })
  it("rejects stale setters and image-bearing drafts", () => {
    open()
    const draft = edit()
    const input = receipt()
    expect(
      state().setOwnedWorkspaceBannerDraft({
        ...input,
        expectedDraft: null,
        draft: null
      })
    ).toBe(false)
    expect(() =>
      state().setOwnedWorkspaceBannerDraft({
        ...input,
        expectedDraft: draft,
        draft: { ...draft, image: "data:image/png;base64,a" } as typeof draft
      })
    ).toThrow()
    state().beginOwnedWorkspace(scope, "server-2")
    expect(
      state().setOwnedWorkspaceBannerDraft({
        ...input,
        expectedDraft: draft,
        draft: null
      })
    ).toBe(false)
  })
  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid draft base version %s",
    (baseVersion) => {
      open()
      const draft = edit()
      expect(() =>
        state().setOwnedWorkspaceBannerDraft({
          origin: state().activeWorkspaceOrigin,
          workspaceId: state().workspaceId,
          expectedDraft: draft,
          draft: { title: "", subtitle: "", baseVersion }
        })
      ).toThrow()
    }
  )
})

describe("owned workspace assistant receipts", () => {
  const edit = (assistantId = "persona-1", baseVersion = 1) => {
    const draft = {
      assistantId,
      personaMemoryMode: "read_only" as const,
      baseVersion
    }
    expect(
      state().setOwnedWorkspaceAssistantDraft({
        origin: state().activeWorkspaceOrigin,
        workspaceId: state().workspaceId,
        expectedDraft: state().ownedWorkspaceAssistantDraft,
        draft
      })
    ).toBe(true)
    return draft
  }
  const receipt = () => ({
    origin: state().activeWorkspaceOrigin,
    workspaceId: state().workspaceId,
    expectedWorkspace: state().ownedWorkspaceBundle!.workspace,
    submittedDraft: state().ownedWorkspaceAssistantDraft!,
    editorSession: state().ownedWorkspaceAssistantSession,
    workspace: {
      ...bundle().workspace,
      version: 2,
      assistant_defaults: {
        assistant_kind: "persona" as const,
        assistant_id: "persona-1",
        persona_memory_mode: "read_only" as const
      }
    }
  })
  it("persists scoped drafts and their base without changing the canonical default", () => {
    open()
    const draft = edit()
    expect(state().ownedWorkspaceAssistantDraft).toBe(draft)
    expect(storedDraft().assistantDraft).toEqual(draft)
    expect(state().assistantDefaults).toBeNull()
    expect(storedDraft().pendingChanges).toEqual({})
    open("server-2")
    expect(state().ownedWorkspaceAssistantDraft).toBeNull()
    open()
    expect(state().ownedWorkspaceAssistantDraft).toEqual(draft)
  })
  it("normalizes canonical defaults and clears only the saved editor", () => {
    open()
    edit()
    expect(state().applyOwnedWorkspaceAssistantReceipt(receipt())).toBe("saved")
    expect(state().assistantDefaults?.assistantId).toBe("persona-1")
    expect(state().ownedWorkspaceAssistantDraft).toBeNull()
    expect(storedDraft().pendingChanges).toEqual({})
  })
  it("preserves newer assistant selection and advances its base", () => {
    open()
    edit()
    const input = receipt()
    edit("persona-2")
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe(
      "newer-draft"
    )
    expect(state().ownedWorkspaceAssistantDraft).toMatchObject({
      assistantId: "persona-2",
      baseVersion: 2
    })
  })
  it("does not overwrite a cancelled and reopened assistant editor", () => {
    open()
    edit()
    const input = receipt()
    state().setOwnedWorkspaceAssistantDraft({
      ...input,
      expectedDraft: input.submittedDraft,
      draft: null
    })
    const reopened = edit("persona-2")
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe(
      "editor-changed"
    )
    expect(state().ownedWorkspaceAssistantDraft).toBe(reopened)
  })
  it("allows explicit unchanged-version review but requires PATCH advancement", () => {
    open()
    edit()
    const input = { ...receipt(), workspace: bundle().workspace }
    expect(() => state().applyOwnedWorkspaceAssistantReceipt(input)).toThrow()
    expect(state().acceptOwnedWorkspaceAssistantReview(input)).toBe("saved")
    expect(state().assistantDefaults).toBeNull()
  })
  it("rejects stale receipts across account/target reactivation", () => {
    open()
    edit()
    const input = receipt()
    open("server-2")
    open()
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe("stale")
  })
  it("keeps unrelated drafts and clean canonical metadata after notes refresh", () => {
    open()
    edit()
    const input = receipt()
    state().setWorkspaceBanner({ title: "Unsent banner" })
    state().replaceOwnedWorkspaceNotes({
      ...input,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: []
    })
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe("saved")
    expect(state().workspaceBanner.title).toBe("Unsent banner")
    expect(state().ownedWorkspaceBundle!.notes).toEqual([])
    expect(storedDraft().pendingChanges.workspaceBanner).toMatchObject({
      title: "Unsent banner"
    })
  })
  it("rejects persisted consent and malformed assistant drafts", () => {
    open()
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedDraft: null
    }
    expect(() =>
      state().setOwnedWorkspaceAssistantDraft({
        ...input,
        draft: {
          assistantId: "p",
          personaMemoryMode: "read_write",
          baseVersion: 1,
          confirmed: true
        } as never
      })
    ).toThrow()
    expect(() =>
      state().setOwnedWorkspaceAssistantDraft({
        ...input,
        draft: {
          assistantId: "p",
          personaMemoryMode: "read_only",
          baseVersion: 0
        }
      })
    ).toThrow()
    expect(state().ownedWorkspaceAssistantDraft).toBeNull()
  })

  it("commits a clear receipt without affecting an open rename draft", () => {
    open()
    edit()
    state().setOwnedWorkspaceRenameDraft({
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedDraft: null,
      draft: { name: "Unsent name", baseVersion: 1 }
    })
    const rename = state().ownedWorkspaceRenameDraft
    const input = {
      ...receipt(),
      workspace: { ...bundle().workspace, version: 2, assistant_defaults: null }
    }
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe("saved")
    expect(state().assistantDefaults).toBeNull()
    expect(state().ownedWorkspaceAssistantDraft).toBeNull()
    expect(state().ownedWorkspaceRenameDraft).toBe(rename)
  })

  it.each([
    { id: "foreign" },
    { archived: true },
    { deleted: true },
    { version: 0 }
  ])("rejects an invalid assistant receipt %j", (patch) => {
    open()
    edit()
    const input = receipt()
    Object.assign(input.workspace, patch)
    const before = state()
    expect(() => state().applyOwnedWorkspaceAssistantReceipt(input)).toThrow()
    expect(state()).toBe(before)
  })

  it("rejects stale assistant setters and pending-target receipts", () => {
    open()
    const draft = edit()
    const input = receipt()
    expect(
      state().setOwnedWorkspaceAssistantDraft({
        ...input,
        expectedDraft: null,
        draft: null
      })
    ).toBe(false)
    state().beginOwnedWorkspace(scope, "server-2")
    expect(
      state().setOwnedWorkspaceAssistantDraft({
        ...input,
        expectedDraft: draft,
        draft: null
      })
    ).toBe(false)
    expect(state().applyOwnedWorkspaceAssistantReceipt(input)).toBe("stale")
  })
})

describe("owned workspace rename receipts", () => {
  const edit = (name = "My name", baseVersion = 1) => {
    const draft = { name, baseVersion }
    expect(
      state().setOwnedWorkspaceRenameDraft({
        origin: state().activeWorkspaceOrigin,
        workspaceId: state().workspaceId,
        expectedDraft: state().ownedWorkspaceRenameDraft,
        draft
      })
    ).toBe(true)
    return draft
  }
  const receipt = () => ({
    origin: state().activeWorkspaceOrigin,
    workspaceId: state().workspaceId,
    expectedWorkspace: state().ownedWorkspaceBundle!.workspace,
    submittedDraft: state().ownedWorkspaceRenameDraft!,
    editorSession: state().ownedWorkspaceRenameSession,
    workspace: { ...bundle().workspace, name: "My name", version: 2 }
  })

  it("persists and recovers the draft without changing canonical metadata", () => {
    open()
    const draft = edit()
    expect(state().ownedWorkspaceRenameDraft).toBe(draft)
    expect(state().workspaceName).toBe("Server title")
    expect(storedDraft().renameDraft).toEqual(draft)
    expect(storedDraft().pendingChanges).toEqual({})
    open("server-2")
    expect(state().ownedWorkspaceRenameDraft).toBeNull()
    open()
    expect(state().ownedWorkspaceRenameDraft).toEqual(draft)
  })

  it("commits the canonical receipt and clears only the submitted draft", () => {
    open()
    edit()
    expect(state().applyOwnedWorkspaceRenameReceipt(receipt())).toBe("saved")
    expect(state().workspaceName).toBe("My name")
    expect(state().ownedWorkspaceBundle!.workspace.version).toBe(2)
    expect(state().ownedWorkspaceRenameDraft).toBeNull()
    expect(storedDraft().renameDraft).toBeNull()
    expect(storedDraft().pendingChanges).toEqual({})
  })

  it("accepts an explicitly inspected unchanged server name without weakening PATCH receipts", () => {
    open()
    edit()
    const input = { ...receipt(), workspace: bundle().workspace }
    expect(() => state().applyOwnedWorkspaceRenameReceipt(input)).toThrow()
    expect(state().acceptOwnedWorkspaceRenameReview(input)).toBe("saved")
    expect(state().workspaceName).toBe("Server title")
    expect(state().ownedWorkspaceBundle!.workspace.version).toBe(1)
    expect(state().ownedWorkspaceRenameDraft).toBeNull()
  })

  it("does not accept an inspected version older than the acknowledged draft base", () => {
    open()
    edit("Mine", 3)
    const before = state()
    expect(() => state().acceptOwnedWorkspaceRenameReview(receipt())).toThrow()
    expect(state()).toBe(before)
  })

  it("preserves newer typing and advances its base version", () => {
    open()
    edit()
    const input = receipt()
    edit("Still typing")
    expect(state().applyOwnedWorkspaceRenameReceipt(input)).toBe("newer-draft")
    expect(state().ownedWorkspaceRenameDraft).toEqual({
      name: "Still typing",
      baseVersion: 2
    })
    expect(state().workspaceName).toBe("My name")
  })

  it("does not rebase a cancelled and reopened editor on a late response", () => {
    open()
    edit()
    const input = receipt()
    state().setOwnedWorkspaceRenameDraft({
      ...input,
      expectedDraft: input.submittedDraft,
      draft: null
    })
    const reopened = edit("Reopened")
    expect(state().applyOwnedWorkspaceRenameReceipt(input)).toBe(
      "editor-changed"
    )
    expect(state().ownedWorkspaceRenameDraft).toBe(reopened)
    expect(state().workspaceName).toBe("My name")
  })

  it("accepts a receipt after a notes refresh and preserves unrelated local edits", () => {
    open()
    edit("My name", 3)
    const input = receipt()
    input.workspace.version = 4
    input.workspace.banner_title = "Remote banner"
    input.workspace.audio_voice = "Remote voice"
    state().setWorkspaceBanner({
      title: "Local banner",
      subtitle: "Subtitle",
      image: null
    })
    state().replaceOwnedWorkspaceNotes({
      ...input,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: []
    })
    expect(state().applyOwnedWorkspaceRenameReceipt(input)).toBe("saved")
    expect(state().workspaceBanner.title).toBe("Local banner")
    expect(state().audioSettings.voice).toBe("Remote voice")
    expect(state().ownedWorkspaceBaseline!.workspaceBanner.title).toBe(
      "Remote banner"
    )
    expect(state().ownedWorkspaceBundle!.notes).toEqual([])
    expect(storedDraft().pendingChanges.workspaceBanner).toMatchObject({
      title: "Local banner"
    })
  })

  it("rejects stale scope and replaced metadata receipts", () => {
    open()
    edit()
    const input = receipt()
    open("server-2")
    open()
    expect(state().applyOwnedWorkspaceRenameReceipt(input)).toBe("stale")
    const current = receipt()
    expect(state().applyOwnedWorkspaceRenameReceipt(current)).toBe("saved")
    expect(state().applyOwnedWorkspaceRenameReceipt(current)).toBe("stale")
  })

  it.each([
    { id: "foreign" },
    { archived: true },
    { deleted: true },
    { version: 1 },
    { version: Number.MAX_SAFE_INTEGER + 1 }
  ])("rejects invalid receipts without mutation: %j", (invalid) => {
    open()
    edit()
    const input = receipt()
    Object.assign(input.workspace, invalid)
    const before = state()
    expect(() => state().applyOwnedWorkspaceRenameReceipt(input)).toThrow()
    expect(state()).toBe(before)
  })

  it("rejects stale draft setters and invalid base versions", () => {
    open()
    const draft = edit()
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedDraft: null
    }
    expect(state().setOwnedWorkspaceRenameDraft({ ...input, draft })).toBe(
      false
    )
    expect(() =>
      state().setOwnedWorkspaceRenameDraft({
        ...input,
        expectedDraft: draft,
        draft: { name: "Bad", baseVersion: 0 }
      })
    ).toThrow()
    expect(state().ownedWorkspaceRenameDraft).toBe(draft)
  })

  it("blocks draft edits and receipts while another activation is pending", () => {
    open()
    const draft = edit()
    const input = receipt()
    state().beginOwnedWorkspace(scope, "server-2")
    expect(
      state().setOwnedWorkspaceRenameDraft({
        ...input,
        expectedDraft: draft,
        draft: null
      })
    ).toBe(false)
    expect(state().applyOwnedWorkspaceRenameReceipt(input)).toBe("stale")
    expect(state().ownedWorkspaceRenameDraft).toBe(draft)
  })

  it("retains a rename draft in memory and reports storage failure", () => {
    open()
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("Quota exceeded")
    })
    const draft = edit()
    expect(state().ownedWorkspaceRenameDraft).toBe(draft)
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    expect(state().workspaceName).toBe("Server title")
    vi.restoreAllMocks()
    state().setOwnedWorkspaceRenameDraft({
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedDraft: draft,
      draft: null
    })
    expect(state().ownedWorkspaceDraftStatus).toBe("saved")
  })
})

describe("owned workspace store integration", () => {
  const saveInput = () => ({
    origin: state().activeWorkspaceOrigin,
    workspaceId: state().workspaceId,
    editorSession: state().ownedNoteEditorSession,
    submitted: state().currentNote,
    note: {
      ...bundle().notes[0],
      title: "Saved title",
      content: "Saved content",
      keywords_json: '["evidence"]',
      version: 3
    }
  })

  it("marks only the submitted editor version clean after saving", () => {
    open()
    state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
    state().updateNoteContent("Saved content")
    expect(state().applyOwnedWorkspaceNoteReceipt(saveInput())).toBe("saved")
    expect(state().currentNote).toEqual({
      id: 3,
      title: "Saved title",
      content: "Saved content",
      keywords: ["evidence"],
      version: 3,
      isDirty: false
    })
    expect(state().ownedWorkspaceBundle?.notes[0].version).toBe(3)
    expect(storedDraft().currentNote.version).toBe(3)
  })

  it("preserves newer typing and adopts the saved version", () => {
    open()
    state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
    const input = saveInput()
    state().updateNoteTitle("New title")
    state().updateNoteContent("New unsent content")
    state().updateNoteKeywords(["new"])
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("newer-draft")
    expect(state().currentNote).toEqual({
      id: 3,
      title: "New title",
      content: "New unsent content",
      keywords: ["new"],
      version: 3,
      isDirty: true
    })
    expect(storedDraft().currentNote.content).toBe("New unsent content")
  })

  it("attaches a new note ID without discarding subsequent edits", () => {
    open()
    state().updateNoteContent("Initial draft")
    const input = saveInput()
    input.note = { ...input.note, id: 4, version: 1 }
    state().updateNoteContent("Keep typing")
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("newer-draft")
    expect(state().currentNote).toMatchObject({
      id: 4,
      version: 1,
      content: "Keep typing",
      isDirty: true
    })
    expect(state().ownedWorkspaceBundle?.notes.map((note) => note.id)).toEqual([
      3, 4
    ])
  })

  it.each(["clear", "reload", "replace"])(
    "does not overwrite a %s editor session on save completion",
    (action) => {
      open()
      state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
      const input = saveInput()
      if (action === "clear") state().clearCurrentNote()
      if (action === "reload")
        state().loadNote({
          id: 3,
          title: "Reopened",
          content: "Reopened",
          version: 2
        })
      if (action === "replace")
        state().setCurrentNote({
          ...state().currentNote,
          content: "Replacement",
          isDirty: true
        })
      const editor = state().currentNote
      expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe(
        "editor-changed"
      )
      expect(state().currentNote).toBe(editor)
      expect(state().ownedWorkspaceBundle?.notes[0].version).toBe(3)
    }
  )

  it.each(["account", "reopen", "invalidate", "pending"])(
    "rejects %s activation save completions",
    (action) => {
      open()
      state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
      const input = saveInput()
      if (action === "account")
        open("server-1", { ...scope, principalId: "other" })
      if (action === "reopen") open()
      if (action === "invalidate") state().invalidateOwnedWorkspace()
      if (action === "pending") state().beginOwnedWorkspace(scope, "server-1")
      const before = state()
      expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("stale")
      expect(state().currentNote).toBe(before.currentNote)
      expect(state().ownedWorkspaceBundle).toBe(before.ownedWorkspaceBundle)
    }
  )

  it("does not roll back a newer canonical note receipt", () => {
    open()
    state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
    const input = saveInput()
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("saved")
    const before = state()
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("stale")
    expect(state().currentNote).toBe(before.currentNote)
  })

  it("accepts a save after a concurrent refresh already loaded the same receipt", () => {
    open()
    state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
    const input = saveInput()
    expect(
      state().replaceOwnedWorkspaceNotes({
        origin: input.origin,
        workspaceId: input.workspaceId,
        expectedBundle: state().ownedWorkspaceBundle!,
        notes: [input.note]
      })
    ).toBe(true)
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("saved")
    expect(state().currentNote).toMatchObject({ version: 3, isDirty: false })
  })

  it("does not resurrect a note removed by a newer canonical refresh", () => {
    open()
    state().loadNote({ id: 3, title: "Draft", content: "Draft", version: 2 })
    const input = saveInput()
    state().replaceOwnedWorkspaceNotes({
      origin: input.origin,
      workspaceId: input.workspaceId,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: []
    })
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("stale")
    expect(state().ownedWorkspaceBundle?.notes).toEqual([])
  })

  it("refuses a foreign receipt without changing the draft", () => {
    open()
    const input = saveInput()
    input.note.workspace_id = "foreign"
    const before = state()
    expect(() => state().applyOwnedWorkspaceNoteReceipt(input)).toThrow(
      "Workspace notes invalid-response"
    )
    expect(state().currentNote).toBe(before.currentNote)
    expect(state().ownedWorkspaceBundle).toBe(before.ownedWorkspaceBundle)
  })

  it("refreshes notes only within the captured canonical bundle lifetime", () => {
    open()
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: [{ ...bundle().notes[0], version: 3 }]
    }
    const editor = state().currentNote
    expect(state().replaceOwnedWorkspaceNotes(input)).toBe(true)
    expect(state().ownedWorkspaceBundle?.notes[0].version).toBe(3)
    expect(state().currentNote).toBe(editor)
    expect(state().replaceOwnedWorkspaceNotes({ ...input, notes: [] })).toBe(
      false
    )
    expect(state().ownedWorkspaceBundle?.notes).toHaveLength(1)
  })

  it("refuses cross-account note refreshes and duplicate associations", () => {
    open()
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      expectedBundle: state().ownedWorkspaceBundle!,
      notes: [bundle().notes[0], bundle().notes[0]]
    }
    expect(() => state().replaceOwnedWorkspaceNotes(input)).toThrow(
      "Workspace notes invalid-response"
    )
    open("server-1", { ...scope, principalId: "other" })
    const before = state().ownedWorkspaceBundle
    expect(state().replaceOwnedWorkspaceNotes({ ...input, notes: [] })).toBe(
      false
    )
    expect(state().ownedWorkspaceBundle).toBe(before)
  })

  it("persists an uncertain create through reload without changing editor identity", async () => {
    open()
    state().updateNoteContent("A potentially saved draft")
    const editorSession = state().ownedNoteEditorSession
    const marked = state().setOwnedNoteCreateUncertain({
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      editorSession,
      value: true
    })
    expect(marked).toBe(state().currentNote)
    expect(state().ownedNoteEditorSession).toBe(editorSession)
    expect(storedDraft().currentNote.createUncertain).toBe(true)
    state().invalidateOwnedWorkspace()
    open()
    expect(state().currentNote).toMatchObject({
      content: "A potentially saved draft",
      createUncertain: true
    })
  })

  it("clears create uncertainty after a receipt while preserving newer typing", () => {
    open()
    state().updateNoteContent("Sent draft")
    state().setOwnedNoteCreateUncertain({
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      editorSession: state().ownedNoteEditorSession,
      value: true
    })
    const input = saveInput()
    input.note = { ...input.note, id: 4, version: 1 }
    state().updateNoteContent("Newer draft")
    expect(state().applyOwnedWorkspaceNoteReceipt(input)).toBe("newer-draft")
    expect(state().currentNote.createUncertain).toBeUndefined()
    expect(storedDraft().currentNote).toMatchObject({
      id: 4,
      content: "Newer draft"
    })
    expect(storedDraft().currentNote.createUncertain).toBeUndefined()
  })

  it("does not mark a different editor session or account uncertain", () => {
    open()
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      editorSession: state().ownedNoteEditorSession,
      value: true
    }
    state().clearCurrentNote()
    expect(state().setOwnedNoteCreateUncertain(input)).toBeNull()
    const sameSession = {
      ...input,
      editorSession: state().ownedNoteEditorSession
    }
    open("server-1", { ...scope, principalId: "other" })
    expect(state().setOwnedNoteCreateUncertain(sameSession)).toBeNull()
    expect(state().currentNote.createUncertain).toBeUndefined()
  })

  it("clears a definitely rejected create marker without discarding the draft", () => {
    open()
    state().updateNoteContent("Keep this draft")
    const input = {
      origin: state().activeWorkspaceOrigin,
      workspaceId: state().workspaceId,
      editorSession: state().ownedNoteEditorSession,
      value: true
    }
    state().setOwnedNoteCreateUncertain(input)
    expect(
      state().setOwnedNoteCreateUncertain({ ...input, value: false })
    ).toMatchObject({ content: "Keep this draft" })
    expect(state().currentNote.createUncertain).toBeUndefined()
    expect(storedDraft().currentNote.createUncertain).toBeUndefined()
  })

  it("authorizes a durable current marker without hiding another target's pending draft", () => {
    open("pending-note-target")
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(
            `${ownedWorkspaceDraftKey(scope, "pending-note-target")}:revisions:`
          )
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    try {
      state().updateNoteContent("Keep the failed draft")
      open("durable-note-target")
      spy.mockRestore()
      state().updateNoteContent("Create this note")
      expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
      expect(
        state().setOwnedNoteCreateUncertain({
          origin: state().activeWorkspaceOrigin,
          workspaceId: state().workspaceId,
          editorSession: state().ownedNoteEditorSession,
          value: true
        })
      ).toMatchObject({ content: "Create this note", createUncertain: true })
      expect(
        storedDraft("durable-note-target").currentNote.createUncertain
      ).toBe(true)
      expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    } finally {
      spy.mockRestore()
      open("pending-note-target")
      expect(state().currentNote.content).toBe("Keep the failed draft")
      state().updateNoteContent("Keep the failed draft")
    }
  })

  it("refuses to authorize a create while its own draft has concurrent edits", () => {
    open()
    state().updateNoteContent("Current note")
    const legacy = { ...storedDraft(), composer: "Another window's draft" }
    const key = ownedWorkspaceDraftKey(scope, "server-1")
    localStorage.setItem(key, JSON.stringify(legacy))
    expect(() =>
      state().setOwnedNoteCreateUncertain({
        origin: state().activeWorkspaceOrigin,
        workspaceId: state().workspaceId,
        editorSession: state().ownedNoteEditorSession,
        value: true
      })
    ).toThrow("recovery storage")
    expect(state().currentNote.createUncertain).toBeUndefined()
    expect(state().ownedWorkspaceDraftStatus).toBe("conflict")
    expect(localStorage.getItem(key)).toBe(JSON.stringify(legacy))
  })

  it("refuses to authorize a create when its recovery marker cannot be persisted", () => {
    open()
    state().updateNoteContent("Do not risk a duplicate")
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(
            `${ownedWorkspaceDraftKey(scope, "server-1")}:revisions:`
          )
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    try {
      expect(() =>
        state().setOwnedNoteCreateUncertain({
          origin: state().activeWorkspaceOrigin,
          workspaceId: state().workspaceId,
          editorSession: state().ownedNoteEditorSession,
          value: true
        })
      ).toThrow("recovery storage")
      expect(state().currentNote.content).toBe("Do not risk a duplicate")
      expect(state().currentNote.createUncertain).toBeUndefined()
      expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    } finally {
      spy.mockRestore()
      state().updateNoteContent("Do not risk a duplicate")
    }
  })

  it("retains outstanding draft-loss risk through local workspace rehydration", async () => {
    const local = state().initializeWorkspace("Local recovery")
    open("local-rehydrate-quota")
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(
            `${ownedWorkspaceDraftKey(scope, "local-rehydrate-quota")}:revisions:`
          )
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    state().updateNoteContent("Still only in memory")
    state().switchWorkspace(local)
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    await useWorkspaceStore.persist.rehydrate()
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    spy.mockRestore()
    open("local-rehydrate-quota")
    expect(state().currentNote.content).toBe("Still only in memory")
    expect(state().ownedWorkspaceDraftStatus).toBe("saved")
  })
  it("does not resurrect an account invalidated between merge and hydration completion", async () => {
    open("invalidated-hydration")
    const { storage, merge } = useWorkspaceStore.persist.getOptions()
    useWorkspaceStore.persist.setOptions({
      storage: { ...storage!, getItem: async () => null },
      merge: (persisted, current) => {
        const merged = merge!(persisted, current)
        queueMicrotask(() => state().invalidateOwnedWorkspace())
        return merged
      }
    })
    try {
      await useWorkspaceStore.persist.rehydrate()
      expect(state().workspaceId).toBe("")
      expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
      expect(state().sources).toEqual([])
    } finally {
      useWorkspaceStore.persist.setOptions({ storage, merge })
    }
  })
  it("revives scoped folders and prunes missing canonical source memberships", () => {
    open("folders-target")
    const folder = state().createSourceFolder("Evidence")
    state().assignSourceToFolders("source-1", [folder.id])
    state().setActiveFolder(folder.id)
    expect(
      typeof storedDraft("folders-target").sourceFolders[0].createdAt
    ).toBe("string")
    state().invalidateOwnedWorkspace()
    const attempt = state().beginOwnedWorkspace(scope, "folders-target")
    const updated = bundle("folders-target")
    updated.sources = []
    expect(state().activateOwnedWorkspace(attempt, updated)).toBe("activated")
    expect(state().sourceFolders[0].createdAt).toBeInstanceOf(Date)
    expect(state().activeFolderId).toBe(folder.id)
    expect(state().sourceFolderMemberships).toEqual([])
  })

  it("does not classify ingestion status projections as unsaved canonical edits", () => {
    open("status-target")
    state().setSourceStatusById("source-1", "processing", "Indexing")
    expect(storedDraft("status-target").pendingChanges).toEqual({})
    state().invalidateOwnedWorkspace()
    open("status-target")
  })

  it("retains true source removal as a pending canonical edit", () => {
    open("source-edit-target")
    state().removeSource("source-1")
    expect(storedDraft("source-edit-target").pendingChanges.sources).toEqual([])
  })

  it("keeps canonical source selection when an ingestion status reports failure", () => {
    open("failed-status-target")
    state().setSourceStatusById("source-1", "error", "Failed indexing")
    expect(state().selectedSourceIds).toEqual(["source-1"])
    expect(storedDraft("failed-status-target").pendingChanges).toEqual({})
    expect(state().getSelectedSources()).toEqual([])
  })
  it("keeps reload-risk visible when an incoming draft saves but an outgoing draft cannot", () => {
    open("outgoing-quota")
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(
            `${ownedWorkspaceDraftKey(scope, "outgoing-quota")}:revisions:`
          )
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    state().updateNoteContent("Only in memory")
    open("incoming-saved")
    expect(storedDraft("incoming-saved")).toBeDefined()
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    spy.mockRestore()
    open("outgoing-quota")
    expect(state().currentNote.content).toBe("Only in memory")
    expect(state().ownedWorkspaceDraftStatus).toBe("saved")
  })

  it("rejects invalid editor state without making account invalidation fail", () => {
    open("invalid-editor")
    state().updateNoteContent("Valid recovery")
    expect(() =>
      state().setCurrentNote({ ...state().currentNote, version: 0 })
    ).toThrow(/invalid/i)
    expect(state().currentNote.version).toBeUndefined()
    expect(() => state().invalidateOwnedWorkspace()).not.toThrow()
    expect(state().workspaceId).toBe("")
    expect(storedDraft("invalid-editor").currentNote.content).toBe(
      "Valid recovery"
    )
  })
  it("requires completed legacy hydration before beginning activation", () => {
    useWorkspaceStore.setState({ storeHydrated: false })
    expect(() => state().beginOwnedWorkspace(scope, "server-1")).toThrow(
      /hydrat/i
    )
  })

  it("commits canonical data once and captures local edits made during loading", () => {
    const local = state().initializeWorkspace("Local")
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    state().updateNoteContent("Typed while loading")
    const listener = vi.fn()
    const unsubscribe = useWorkspaceStore.subscribe(listener)
    expect(state().activateOwnedWorkspace(attempt, bundle())).toBe("activated")
    unsubscribe()
    expect(listener).toHaveBeenCalledTimes(1)
    expect(state().workspaceSnapshots[local].currentNote.content).toBe(
      "Typed while loading"
    )
    expect(state().activeWorkspaceOrigin.kind).toBe("server-owned")
    expect(state().sources[0].title).toBe("Canonical source")
    expect(state().selectedSourceIds).toEqual(["source-1"])
    expect(state().effectiveAssistantDefault).toEqual(
      bundle().workspace.effectiveAssistantDefault
    )
    expect(state().ownedWorkspaceBundle?.workspace.audio_provider).toBe(
      "custom-provider"
    )
    expect(state().workspaceSnapshots["server-1"]).toBeUndefined()
  })

  it("persists editor changes only under the verified account scope", () => {
    open()
    state().updateNoteContent("Private unsaved note")
    state().setOwnedWorkspaceComposer("Private question")
    state().saveCurrentWorkspace()
    expect(storedDraft()).toMatchObject({
      currentNote: { content: "Private unsaved note" },
      composer: "Private question"
    })
    const legacyPayload =
      useWorkspaceStore.persist.getOptions().partialize!(state())
    expect(JSON.stringify(legacyPayload)).not.toContain("Private")
    expect(legacyPayload.workspaceId).toBe("")
    expect(state().savedWorkspaces).toEqual([])
    expect(state().workspaceSnapshots).toEqual({})
  })

  it("quarantines canonical edits instead of adding legacy snapshots", () => {
    open()
    state().setWorkspaceName("Unsaved rename")
    expect(storedDraft().pendingChanges.workspaceName).toBe("Unsaved rename")
    expect(state().workspaceSnapshots).toEqual({})
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    expect(state().activateOwnedWorkspace(attempt, bundle())).toBe(
      "draft-conflict"
    )
    expect(state().workspaceName).toBe("Unsaved rename")
  })

  it("preserves an owned draft when switching to a local workspace and back", () => {
    const local = state().initializeWorkspace("Local")
    open()
    state().updateNoteContent("Owned draft")
    state().switchWorkspace(local)
    expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
    expect(state().workspaceSnapshots["server-1"]).toBeUndefined()
    expect(state().ownedWorkspaceBundle).toBeNull()
    open()
    expect(state().currentNote.content).toBe("Owned draft")
  })

  it("rejects stale attempts after a local switch or invalidation", () => {
    const local = state().initializeWorkspace("Local")
    const old = state().beginOwnedWorkspace(scope, "server-1")
    state().switchWorkspace(local)
    expect(state().activateOwnedWorkspace(old, bundle())).toBe("stale")
    const next = state().beginOwnedWorkspace(scope, "server-1")
    state().invalidateOwnedWorkspace()
    expect(state().activateOwnedWorkspace(next, bundle())).toBe("stale")
  })

  it("clears owned content on invalidation without deleting its draft or local records", () => {
    const local = state().initializeWorkspace("Local")
    open()
    state().updateNoteContent("Retain this")
    state().invalidateOwnedWorkspace()
    expect(state().workspaceId).toBe("")
    expect(state().sources).toEqual([])
    expect(state().currentNote.content).toBe("")
    expect(state().workspaceSnapshots[local]).toBeDefined()
    expect(storedDraft().currentNote.content).toBe("Retain this")
  })

  it("isolates identical target IDs across accounts", () => {
    open()
    state().updateNoteContent("Account two")
    state().invalidateOwnedWorkspace()
    open("server-1", { ...scope, principalId: "3" })
    expect(state().currentNote.content).toBe("")
    state().updateNoteContent("Account three")
    expect(storedDraft().currentNote.content).toBe("Account two")
  })

  it("retains dirty same-target note conflict rather than replacing it", () => {
    open()
    state().loadNote({
      id: 3,
      title: "Note",
      content: "Canonical note",
      keywords: [],
      version: 2
    })
    state().updateNoteContent("Dirty")
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    const changed = bundle()
    changed.notes[0].version = 3
    expect(state().activateOwnedWorkspace(attempt, changed)).toBe(
      "draft-conflict"
    )
    expect(state().currentNote.content).toBe("Dirty")
  })

  it("reports failed storage and recovers the in-memory draft", () => {
    open("quota-target")
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new Error("Quota")
      })
    state().updateNoteContent("Recover me")
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    state().invalidateOwnedWorkspace()
    spy.mockRestore()
    open("quota-target")
    expect(state().currentNote.content).toBe("Recover me")
  })

  it("does not replace live owned state with a later persistence hydration", async () => {
    state().initializeWorkspace("Local")
    const storage = useWorkspaceStore.persist.getOptions().storage!
    const legacy = await storage.getItem(WORKSPACE_STORAGE_KEY)
    expect(legacy?.state.workspaceId).toBe(state().workspaceId)
    open()
    state().updateNoteContent("Current editor")
    await storage.setItem(WORKSPACE_STORAGE_KEY, legacy!)
    await useWorkspaceStore.persist.rehydrate()
    expect(state().workspaceId).toBe("server-1")
    expect(state().currentNote.content).toBe("Current editor")
    expect(state().workspaceSnapshots["server-1"]).toBeUndefined()
  })

  it("retries a failed draft write on explicit save without another edit", () => {
    open("retry-target")
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (key.startsWith("tldw:research-workspace:owned-drafts:"))
          throw new Error("Quota")
        original.call(this, key, value)
      })
    state().updateNoteContent("Retry draft")
    expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    spy.mockRestore()
    state().saveCurrentWorkspace()
    expect(state().ownedWorkspaceDraftStatus).toBe("saved")
    expect(storedDraft("retry-target").currentNote.content).toBe("Retry draft")
  })

  it("keeps the outgoing local collection assignment on activation", () => {
    const local = state().initializeWorkspace("Local")
    const collection = state().createWorkspaceCollection("Collection")
    state().assignWorkspaceToCollection(local, collection.id)
    open()
    expect(
      state().savedWorkspaces.find((item) => item.id === local)?.collectionId
    ).toBe(collection.id)
  })

  it("invalidates pending activation when restoring a local undo snapshot", () => {
    state().initializeWorkspace("Local")
    const undo = state().captureUndoSnapshot()
    const attempt = state().beginOwnedWorkspace(scope, "server-1")
    state().restoreUndoSnapshot(undo)
    expect(state().activateOwnedWorkspace(attempt, bundle())).toBe("stale")
  })

  it("fails closed on a corrupt target draft without replacing it", () => {
    state().initializeWorkspace("Local")
    localStorage.setItem(ownedWorkspaceDraftKey(scope, "corrupt-target"), "{")
    const attempt = state().beginOwnedWorkspace(scope, "corrupt-target")
    expect(
      state().activateOwnedWorkspace(attempt, bundle("corrupt-target"))
    ).toBe("draft-conflict")
    expect(state().workspaceName).toBe("Local")
    expect(state().ownedWorkspaceConflict).toBe("invalid")
    expect(
      localStorage.getItem(ownedWorkspaceDraftKey(scope, "corrupt-target"))
    ).toBe("{")
  })

  it("removes old-account content as soon as another account starts opening", () => {
    open()
    state().updateNoteContent("Old account draft")
    state().beginOwnedWorkspace({ ...scope, principalId: "4" }, "server-1")
    expect(state().sources).toEqual([])
    expect(state().workspaceId).toBe("")
    expect(storedDraft().currentNote.content).toBe("Old account draft")
  })

  it.each(["create", "load", "import"])(
    "preserves drafts without unscoped owned copies on local %s",
    (action) => {
      state().initializeWorkspace("Export source")
      const exported = state().exportWorkspaceBundle()!
      open()
      state().updateNoteContent("Owned outgoing")
      if (action === "create") state().createNewWorkspace("New local")
      if (action === "load")
        state().loadWorkspace({
          id: "local-loaded",
          name: "Loaded",
          tag: "workspace:loaded",
          createdAt: new Date(date),
          updatedAt: new Date(date)
        })
      if (action === "import") state().importWorkspaceBundle(exported)
      expect(state().activeWorkspaceOrigin.kind).toBe("legacy-local")
      expect(state().workspaceSnapshots["server-1"]).toBeUndefined()
      expect(
        state().savedWorkspaces.some((item) => item.id === "server-1")
      ).toBe(false)
      expect(storedDraft().currentNote.content).toBe("Owned outgoing")
    }
  )

  it("does not overwrite a same-ID legacy snapshot when opening an owned workspace", () => {
    state().loadWorkspace({
      id: "server-1",
      name: "Legacy namesake",
      tag: "workspace:legacy",
      createdAt: new Date(date),
      updatedAt: new Date(date)
    })
    state().updateNoteContent("Legacy namesake draft")
    open()
    state().updateNoteContent("Server-owned draft")
    state().saveCurrentWorkspace()
    expect(state().workspaceSnapshots["server-1"].currentNote.content).toBe(
      "Legacy namesake draft"
    )
    state().switchWorkspace("server-1")
    expect(state().currentNote.content).toBe("Legacy namesake draft")
  })

  it("rejects a late asynchronous hydration without publishing old target content", async () => {
    state().initializeWorkspace("Old local")
    const storage = useWorkspaceStore.persist.getOptions().storage!
    const old = await storage.getItem(WORKSPACE_STORAGE_KEY)
    let release!: () => void
    useWorkspaceStore.persist.setOptions({
      storage: {
        ...storage,
        getItem: () =>
          new Promise((resolve) => {
            release = () => resolve(old)
          })
      }
    })
    try {
      const hydration = useWorkspaceStore.persist.rehydrate()
      open()
      state().updateNoteContent("Live owned content")
      const observed: string[] = []
      const unsubscribe = useWorkspaceStore.subscribe((value) =>
        observed.push(value.workspaceId)
      )
      release()
      await hydration
      unsubscribe()
      expect(observed.every((id) => id === "server-1")).toBe(true)
      expect(state().currentNote.content).toBe("Live owned content")
    } finally {
      useWorkspaceStore.persist.setOptions({ storage })
    }
  })

  it.each([
    "duplicateWorkspace",
    "archiveWorkspace",
    "deleteWorkspace",
    "captureUndoSnapshot",
    "exportWorkspaceBundle"
  ] as const)("rejects legacy %s for an owned target", (action) => {
    open()
    expect(() => state()[action]("server-1")).toThrow(/server-owned/i)
    expect(state().workspaceSnapshots).toEqual({})
  })
})
