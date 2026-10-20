import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { usePromptTemplates, type UsePromptTemplatesDeps } from "../hooks/usePromptTemplates"
import { createStartupTemplateBundle, parseStartupTemplateBundles, type StartupTemplateBundle } from "../startup-template-bundles"

const bundle = (ownerId?: string, linked = false) => createStartupTemplateBundle({
  name: "Saved private prompt", selectedModel: "model-a", systemPrompt: "ALICE-PRIVATE",
  ...(ownerId ? { ownerId } : {}),
  ...(linked ? { selectedSystemPromptId: "local-uuid", promptStudioPromptId: 1, promptSource: "prompt-studio" as const } : {})
}, { id: ownerId || (linked ? "ambiguous" : "shared"), now: 1 })

const deps = (overrides: Partial<UsePromptTemplatesDeps> = {}): UsePromptTemplatesDeps => ({
  startupTemplatesRaw: "[]", setStartupTemplatesRaw: vi.fn(), promptLibrary: [],
  selectedModel: "model-a", systemPrompt: "ALICE-PRIVATE", selectedSystemPrompt: null,
  selectedQuickPrompt: null, selectedCharacter: null, ragPinnedResults: [], currentChatModelSettings: {},
  setSelectedModel: vi.fn(), setSelectedSystemPrompt: vi.fn(), setSelectedQuickPrompt: vi.fn(), setSystemPrompt: vi.fn(),
  applyAssistantSelection: vi.fn(async () => undefined), setRagPinnedResults: vi.fn(), updateChatModelSettings: vi.fn(),
  compareModeActive: false, setCompareSelectedModels: vi.fn(), setModeAnnouncement: vi.fn(),
  t: (_key, fallback) => fallback || _key,
  ...overrides
})
const ownedDeps = (ownerId: string, overrides: Partial<UsePromptTemplatesDeps> = {}) => deps({
  promptOwnerId: ownerId, promptOwnerReady: true, promptOwnerBoundaryKey: ownerId, ...overrides
})

describe("saved startup template owner boundary", () => {
  it("preserves opaque owner and exact local/Studio identity through storage", () => {
    const saved = parseStartupTemplateBundles(JSON.stringify([bundle("alice", true)]))[0]
    expect([saved.ownerId, saved.selectedSystemPromptId, saved.promptStudioPromptId]).toEqual(["alice", "local-uuid", 1])
  })
  it("hides foreign and ambiguous linked copies while retaining explicit shared local templates", () => {
    const raw = JSON.stringify([bundle("alice", true), bundle("bob", true), bundle(undefined, true), bundle()])
    const { result } = renderHook(() => usePromptTemplates(ownedDeps("bob", { startupTemplatesRaw: raw })))
    expect(result.current.startupTemplates.map(entry => entry.id)).toEqual(["bob", "shared"])
  })
  it("does not expose private copies while owner resolution is pending", () => {
    const raw = JSON.stringify([bundle("alice", true), bundle()])
    const { result } = renderHook(() => usePromptTemplates(ownedDeps("alice", { startupTemplatesRaw: raw, promptOwnerReady: false })))
    expect(result.current.startupTemplates.map(entry => entry.id)).toEqual(["shared"])
  })
  it("stamps new startup and role-play copies and retains foreign raw entries", () => {
    const state = ownedDeps("alice", { startupTemplatesRaw: JSON.stringify([bundle("bob", true)]), selectedSystemPrompt: "private-id",
      promptLibrary: [{ id: "private-id", ownerId: "alice", title: "Private", content: "ALICE-PRIVATE", is_system: true, createdAt: 1, updatedAt: 1 }] })
    const { result } = renderHook(() => usePromptTemplates(state))
    act(() => result.current.handleSaveStartupTemplate())
    const saved = JSON.parse(vi.mocked(state.setStartupTemplatesRaw).mock.calls[0][0]) as StartupTemplateBundle[]
    expect(saved.map((entry) => entry.ownerId)).toEqual(["alice", "bob"])
    act(() => result.current.handleSaveRolePlaySetup({ name: "Role", rolePlay: { source: "role-play-setup", identity: null, behavior: { source: "custom", templateId: null, templateTitle: null, templateCategory: null, systemPrompt: "ALICE-PRIVATE", modified: false }, scene: null, generation: null, context: null } }))
    const rolePlaySaved = JSON.parse(vi.mocked(state.setStartupTemplatesRaw).mock.calls[1][0]) as StartupTemplateBundle[]
    expect(rolePlaySaved.map(entry => entry.ownerId)).toEqual(["alice", "bob"])
  })
  it.each(["startup", "role-play"] as const)("keeps a known shared local library %s save visible after storage reload", (kind) => {
    const foreign = bundle("bob", true)
    const ambiguous = bundle(undefined, true)
    const state = deps({ startupTemplatesRaw: JSON.stringify([foreign, ambiguous]), selectedSystemPrompt: "shared-local-uuid", systemPrompt: "SHARED-LOCAL",
      promptLibrary: [{ id: "shared-local-uuid", title: "Shared local", content: "SHARED-LOCAL", is_system: true, createdAt: 1, updatedAt: 1 }] })
    const { result, rerender } = renderHook(({ current }) => usePromptTemplates(current), { initialProps: { current: state } })
    act(() => kind === "startup" ? result.current.handleSaveStartupTemplate() : result.current.handleSaveRolePlaySetup({ name: "Shared role", rolePlay: { source: "role-play-setup", identity: null, behavior: null, scene: null, generation: null, context: null } }))
    const raw = vi.mocked(state.setStartupTemplatesRaw).mock.calls[0][0]
    expect(parseStartupTemplateBundles(raw).filter(entry => entry.id === foreign.id || entry.id === ambiguous.id)).toEqual([foreign, ambiguous])
    rerender({ current: { ...state, startupTemplatesRaw: raw } })
    expect(result.current.startupTemplates).toEqual([expect.objectContaining({ systemPrompt: "SHARED-LOCAL", selectedSystemPromptId: null, promptStudioPromptId: null, promptSource: "system-template" })])
  })
  it.each(["startup", "role-play"] as const)("retains authenticated local UUID and Studio identity on %s save", (kind) => {
    const state = ownedDeps("alice", { selectedSystemPrompt: "alice-local-uuid", promptLibrary: [{ id: "alice-local-uuid", ownerId: "alice", studioPromptId: 7, sourceSystem: "studio", title: "Alice", content: "ALICE-PRIVATE", is_system: true, createdAt: 1, updatedAt: 1 }] })
    const { result } = renderHook(() => usePromptTemplates(state))
    act(() => kind === "startup" ? result.current.handleSaveStartupTemplate() : result.current.handleSaveRolePlaySetup({ name: "Alice role", rolePlay: { source: "role-play-setup", identity: null, behavior: null, scene: null, generation: null, context: null } }))
    expect(parseStartupTemplateBundles(vi.mocked(state.setStartupTemplatesRaw).mock.calls[0][0])[0]).toMatchObject({ ownerId: "alice", selectedSystemPromptId: "alice-local-uuid", promptStudioPromptId: 7, promptSource: "prompt-studio" })
  })
  it("rejects save until configured owner resolves", () => {
    const state = ownedDeps("alice", { promptOwnerReady: false })
    const { result } = renderHook(() => usePromptTemplates(state))
    act(() => result.current.handleSaveStartupTemplate())
    expect(state.setStartupTemplatesRaw).not.toHaveBeenCalled()
  })
  it("rejects foreign direct apply before identity or composer writes", async () => {
    const state = ownedDeps("bob")
    const { result } = renderHook(() => usePromptTemplates(state))
    await act(async () => { await result.current.applyStartupTemplateBundle(bundle("alice", true)) })
    expect([vi.mocked(state.applyAssistantSelection).mock.calls.length, vi.mocked(state.setSystemPrompt).mock.calls.length]).toEqual([0, 0])
  })
  it("retires pending apply immediately on principal change and exposes no stale preview", async () => {
    let finish!: () => void
    const gate = new Promise<void>(resolve => { finish = resolve })
    const saved = bundle("alice", true)
    const state = ownedDeps("alice", { startupTemplatesRaw: JSON.stringify([saved]), applyAssistantSelection: vi.fn(() => gate) })
    const { result } = renderHook(() => usePromptTemplates(state))
    act(() => result.current.handleOpenStartupTemplatePreview(saved.id))
    let applying!: Promise<unknown>
    act(() => { applying = result.current.applyStartupTemplateBundle(saved) })
    await act(async () => { window.dispatchEvent(new Event("tldw:auth-principal-changed")); finish(); await applying })
    expect([result.current.startupTemplatePreview, vi.mocked(state.setSystemPrompt).mock.calls.length]).toEqual([null, 0])
  })
  it("preserves every foreign/ambiguous copy when renaming or removing an owned template", () => {
    const alice = bundle("alice", true)
    const bob = bundle("bob", true)
    const ambiguous = bundle(undefined, true)
    const state = ownedDeps("alice", { startupTemplatesRaw: JSON.stringify([alice, bob, ambiguous]) })
    const { result } = renderHook(() => usePromptTemplates(state))
    act(() => result.current.handleRenameStartupTemplate("alice", "Renamed"))
    const renamed = JSON.parse(vi.mocked(state.setStartupTemplatesRaw).mock.calls[0][0]) as StartupTemplateBundle[]
    expect(renamed.filter((entry) => entry.id !== "alice")).toEqual([bob, ambiguous])
    act(() => result.current.handleDeleteStartupTemplate("bob"))
    expect(state.setStartupTemplatesRaw).toHaveBeenCalledTimes(1)
    act(() => result.current.handleDeleteStartupTemplate("alice"))
    expect(JSON.parse(vi.mocked(state.setStartupTemplatesRaw).mock.calls[1][0])).toEqual([bob, ambiguous])
  })
  it("retires pending apply on a rerender to another owner with the same numeric Studio id", async () => {
    let finish!: () => void
    const gate = new Promise<void>(resolve => { finish = resolve })
    const state = ownedDeps("alice", { applyAssistantSelection: vi.fn(() => gate) })
    const { result, rerender } = renderHook(({ current }) => usePromptTemplates(current), { initialProps: { current: state } })
    let applying!: Promise<unknown>
    act(() => { applying = result.current.applyStartupTemplateBundle(bundle("alice", true)) })
    rerender({ current: ownedDeps("bob") })
    await act(async () => { finish(); await applying })
    expect(state.setSystemPrompt).not.toHaveBeenCalled()
  })
  it("keeps an equivalent configuration event from cancelling a valid same-owner apply", async () => {
    let finish!: () => void
    const gate = new Promise<void>(resolve => { finish = resolve })
    const state = ownedDeps("alice", { applyAssistantSelection: vi.fn(() => gate) })
    const { result } = renderHook(() => usePromptTemplates(state))
    let applying!: Promise<unknown>
    act(() => { applying = result.current.applyStartupTemplateBundle(bundle("alice")) })
    await act(async () => { window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })); finish(); await applying })
    expect(state.setSystemPrompt).toHaveBeenCalledWith("ALICE-PRIVATE")
  })
  it("retires a continuation guard synchronously before the owner facade rerenders", () => {
    const { result } = renderHook(() => usePromptTemplates(ownedDeps("alice")))
    const current = result.current.captureStartupTemplateGuard()
    act(() => { window.dispatchEvent(new Event("tldw:auth-principal-changed")) })
    expect(current()).toBe(false)
  })
  it("allows explicit shared local apply without owner and preserves literal content", async () => {
    const state = deps()
    const { result } = renderHook(() => usePromptTemplates(state))
    await act(async () => { await result.current.applyStartupTemplateBundle(bundle()) })
    expect(state.setSystemPrompt).toHaveBeenCalledWith("ALICE-PRIVATE")
  })
})
