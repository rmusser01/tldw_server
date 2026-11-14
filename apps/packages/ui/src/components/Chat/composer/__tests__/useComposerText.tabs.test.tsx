import React from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"
import { Storage as PlasmoStorage } from "@plasmohq/storage"
import { usePlaygroundSubmit, type UsePlaygroundSubmitDeps } from "@/components/Option/Playground/hooks/usePlaygroundSubmit"
import { useStoreMessageOption } from "@/store/option"
import { conversationDraftKey } from "@/utils/chat-draft-address"

const authority = vi.hoisted(() => ({ user: "alice" }))
vi.mock("@plasmohq/storage", () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => ({
    scopeKey: authority.user, userId: authority.user,
    config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual", orgId: 2 }
  })
}))

import { useComposerText } from "../hooks/useComposerText"

const makeTabStorage = (copy?: Storage): Storage => {
  const data = new Map<string, string>()
  for (let i = 0; copy && i < copy.length; i++) {
    const key = copy.key(i)!
    data.set(key, copy.getItem(key)!)
  }
  return {
    get length() { return data.size },
    key: index => Array.from(data.keys())[index] ?? null,
    getItem: key => data.get(key) ?? null,
    setItem: (key, value) => { data.set(key, value) },
    removeItem: key => { data.delete(key) },
    clear: () => data.clear()
  }
}
const mount = (tab: Storage, strictMode = false, draftKey = "tldw:playgroundChatDraft") => {
  vi.stubGlobal("sessionStorage", tab)
  return renderHook(({ draftKey }) => useComposerText({
    draftKey,
    tabScopedDraft: true,
    textareaRef: React.createRef<HTMLTextAreaElement>()
  }), { initialProps: { draftKey }, wrapper: strictMode ? React.StrictMode : undefined })
}
const settle = async () => {
  await act(async () => { await vi.dynamicImportSettled() })
  await act(async () => { await vi.advanceTimersByTimeAsync(COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1) })
}
const enter = async (view: ReturnType<typeof mount>, text: string) => {
  act(() => view.result.current.setMessageValue(text))
  await settle()
}

describe("Playground drafts in distinct native tab storage areas", () => {
  beforeEach(() => { vi.useFakeTimers(); localStorage.clear(); authority.user = "alice" })
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers() })

  it.each([false, true])("refuses WebUI submission before a destination draft hydrates (queue: %s)", async isSending => {
    const tab = makeTabStorage()
    vi.stubGlobal("sessionStorage", tab)
    useStoreMessageOption.getState().startNewChatDraft()
    const source = useStoreMessageOption.getState().chatDraftAddress!
    const sendMessage = vi.fn(async () => ({ status: "submitted" }))
    const queueSubmission = vi.fn()
    const view = renderHook(() => {
      const address = useStoreMessageOption(state => state.chatDraftAddress)!
      const composer = useComposerText({ draftKey: conversationDraftKey("submit-test", address), tabScopedDraft: true,
        textareaRef: React.createRef<HTMLTextAreaElement>() })
      const deps: UsePlaygroundSubmitDeps & { draftReady: boolean } = {
        ...composer, draftReady: composer.draftReady, isSending, isConnectionReady: true,
        webSearch: false, compareModeActive: false, compareSelectedModels: [], selectedModel: "test",
        fileRetrievalEnabled: false, ragPinnedResults: [], selectedDocuments: [], uploadedFiles: [],
        currentContextSnapshot: {}, conversationTokenCount: 0, characterContextTokenEstimate: 0,
        pinnedSourceTokenEstimate: 0, resolvedMaxContext: 1000, jsonMode: false, openUIRequestMode: false,
        sendMessage, queueSubmission, clearOpenUIRequestMode: () => {}, clearSelectedDocuments: () => {},
        clearUploadedFiles: () => {}, setLastSubmittedContext: () => {}, estimateTokensForText: () => 1,
        resolveSubmissionIntent: (message: string) => ({ message }), validateSelectedChatModelsAvailability: () => true,
        compareModelsSupportCapability: () => true, notificationApi: { info: vi.fn(), warning: vi.fn() },
        t: (key: string, fallback?: string | { defaultValue?: string }) => typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
      }
      return { ...composer, ...usePlaygroundSubmit(deps) }
    })
    await settle()
    act(() => view.result.current.setMessageValue("SOURCE DRAFT"))
    await settle()
    let finish!: () => void
    const held = new Promise<void>(resolve => { finish = resolve })
    const original = PlasmoStorage.prototype.get
    let destinationKey = ""
    const read = vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function (key) {
      if (destinationKey && String(key).includes(destinationKey)) await held
      return original.call(this, key)
    })
    try {
      act(() => {
        useStoreMessageOption.getState().startNewChatDraft()
        destinationKey = conversationDraftKey("submit-test", useStoreMessageOption.getState().chatDraftAddress!)
      })
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      expect(view.result.current.draftReady).toBe(false)
      act(() => view.result.current.submitForm())
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      expect(sendMessage).not.toHaveBeenCalled()
      expect(queueSubmission).not.toHaveBeenCalled()
      expect(view.result.current.form.values.message).toBe("SOURCE DRAFT")
      await act(async () => { finish(); await held })
      await settle()
      expect(view.result.current.form.values.message).toBe("")
      act(() => useStoreMessageOption.getState().setChatDraftAddress(source))
      await settle()
      expect(view.result.current.form.values.message).toBe("SOURCE DRAFT")
    } finally { finish?.(); read.mockRestore(); view.unmount() }
  })

  it("recovers the latest source text and collapse metadata after switching keys before debounce", async () => {
    const tab = makeTabStorage()
    const destination = mount(tab, false, "saved-B")
    await settle(); await enter(destination, "B DRAFT"); destination.unmount()
    vi.stubGlobal("sessionStorage", tab)
    let metadata = { wasExpanded: false, collapsedRange: { start: 10, end: 50 } }
    const view = renderHook(({ draftKey }) => useComposerText({ draftKey, tabScopedDraft: true, textareaRef: React.createRef<HTMLTextAreaElement>(),
      getDraftMetadata: () => metadata,
      restoreWithMetadata: (_text, restored) => { if (restored) metadata = restored as typeof metadata }
    }), { initialProps: { draftKey: "saved-A" } })
    await settle()
    act(() => view.result.current.setMessageValue("LATEST A"))
    view.rerender({ draftKey: "saved-B" })
    await settle()
    expect(view.result.current.form.values.message).toBe("B DRAFT")
    metadata = { wasExpanded: true, collapsedRange: { start: 20, end: 60 } }
    view.rerender({ draftKey: "saved-A" })
    await settle()
    expect(view.result.current.form.values.message).toBe("LATEST A")
    expect(metadata).toEqual({ wasExpanded: false, collapsedRange: { start: 10, end: 50 } })
  })

  it("binds first-ID acknowledgement without replacing the next draft or collapse state", async () => {
    const tab = makeTabStorage()
    vi.stubGlobal("sessionStorage", tab)
    let metadata = { wasExpanded: true, collapsedRange: { start: 10, end: 60 } }
    const view = renderHook(({ draftKey, bindFromDraftKey }) => useComposerText({
      draftKey, bindFromDraftKey, tabScopedDraft: true, textareaRef: React.createRef<HTMLTextAreaElement>(),
      getDraftMetadata: () => metadata,
      restoreWithMetadata: (_text, restored) => { metadata = restored as typeof metadata }
    }), { initialProps: { draftKey: "fresh-A", bindFromDraftKey: undefined as string | undefined } })
    await settle()
    metadata = { wasExpanded: true, collapsedRange: { start: 10, end: 60 } }
    act(() => view.result.current.setMessageValue("SUBMITTED"))
    await settle()
    act(() => view.result.current.beginPromptAssistReset())
    act(() => view.result.current.setMessageValue("NEXT DRAFT"))
    view.rerender({ draftKey: "saved-A", bindFromDraftKey: "fresh-A" })
    await settle()
    expect(view.result.current.form.values.message).toBe("NEXT DRAFT")
    expect(metadata).toEqual({ wasExpanded: true, collapsedRange: { start: 10, end: 60 } })
    view.unmount()
    const restored = mount(tab, false, "saved-A")
    await settle()
    expect(restored.result.current.form.values.message).toBe("NEXT DRAFT")
  })

  it("hydrates normally on a remount even when its previous bind marker remains", async () => {
    const tab = makeTabStorage()
    const saved = mount(tab, false, "saved-A")
    await settle(); await enter(saved, "RESTORED A"); saved.unmount()
    vi.stubGlobal("sessionStorage", tab)
    const restored = renderHook(() => useComposerText({
      draftKey: "saved-A", bindFromDraftKey: "fresh-A", tabScopedDraft: true,
      textareaRef: React.createRef<HTMLTextAreaElement>()
    }))
    await settle()
    expect(restored.result.current.form.values.message).toBe("RESTORED A")
  })

  it("does not replay a skipped bind from a different source key", async () => {
    const tab = makeTabStorage()
    const saved = mount(tab, false, "saved-A")
    await settle(); await enter(saved, "SAVED A"); saved.unmount()
    vi.stubGlobal("sessionStorage", tab)
    const view = renderHook(({ draftKey, bindFromDraftKey }) => useComposerText({
      draftKey, bindFromDraftKey, tabScopedDraft: true, textareaRef: React.createRef<HTMLTextAreaElement>()
    }), { initialProps: { draftKey: "other-B", bindFromDraftKey: undefined as string | undefined } })
    await settle()
    act(() => view.result.current.setMessageValue("OTHER B"))
    view.rerender({ draftKey: "saved-A", bindFromDraftKey: "fresh-A" })
    await settle()
    expect(view.result.current.form.values.message).toBe("SAVED A")
  })

  it("restores an absent destination as empty instead of copying source text", async () => {
    const view = mount(makeTabStorage(), false, "saved-A")
    await settle(); await enter(view, "SOURCE DRAFT")
    view.rerender({ draftKey: "empty-B" })
    await settle()
    expect(view.result.current.form.values.message).toBe("")
  })

  it("does not replace a new destination edit when its storage read finishes late", async () => {
    const destination = mount(makeTabStorage(), false, "held-B")
    await settle(); await enter(destination, "STORED B"); destination.unmount()
    const view = mount(makeTabStorage(), false, "saved-A")
    await settle(); await enter(view, "SOURCE DRAFT")
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    const original = PlasmoStorage.prototype.get
    const read = vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function (key) {
      if (String(key).includes("held-B")) await pending
      return original.call(this, key)
    })
    try {
      view.rerender({ draftKey: "held-B" })
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      act(() => view.result.current.setMessageValue("NEW B EDIT"))
      await act(async () => { finish(); await pending })
      await settle()
      expect(view.result.current.form.values.message).toBe("NEW B EDIT")
      expect(view.result.current.draftReady).toBe(true)
    } finally { finish?.(); read.mockRestore() }
  })

  it("retains a destination edit when leaving before hydration finishes", async () => {
    const tab = makeTabStorage()
    const view = mount(tab, false, "saved-A")
    await settle(); await enter(view, "SOURCE DRAFT")
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    const original = PlasmoStorage.prototype.get
    const read = vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function (key) {
      if (String(key).includes("held-B")) await pending
      return original.call(this, key)
    })
    try {
      view.rerender({ draftKey: "held-B" })
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      act(() => view.result.current.setMessageValue("NEW B BEFORE LEAVING"))
      expect(view.result.current.draftReady).toBe(false)
      view.unmount()
      await act(async () => { finish(); await pending })
      const restored = mount(tab, false, "held-B")
      await settle()
      expect(restored.result.current.form.values.message).toBe("NEW B BEFORE LEAVING")
    } finally { finish?.(); read.mockRestore() }
  })

  it("does not retain carried source text when an untouched destination read is pending", async () => {
    const tab = makeTabStorage()
    const view = mount(tab, false, "saved-A")
    await settle(); await enter(view, "SOURCE ONLY")
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    const original = PlasmoStorage.prototype.get
    const read = vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function (key) {
      if (String(key).includes("held-B")) await pending
      return original.call(this, key)
    })
    try {
      view.rerender({ draftKey: "held-B" })
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      view.unmount()
      await act(async () => { finish(); await pending })
      const restored = mount(tab, false, "held-B")
      await settle()
      expect(restored.result.current.form.values.message).toBe("")
      restored.unmount()
      const source = mount(tab, false, "saved-A")
      await settle()
      expect(source.result.current.form.values.message).toBe("SOURCE ONLY")
    } finally { finish?.(); read.mockRestore() }
  })

  it("retains collapse-only changes on departure after the text has saved", async () => {
    const tab = makeTabStorage()
    vi.stubGlobal("sessionStorage", tab)
    let metadata = { wasExpanded: false }
    const view = renderHook(({ draftKey }) => useComposerText({ draftKey, tabScopedDraft: true, textareaRef: React.createRef<HTMLTextAreaElement>(),
      getDraftMetadata: () => metadata,
      restoreWithMetadata: (_text, restored) => { if (restored) metadata = restored as typeof metadata }
    }), { initialProps: { draftKey: "saved-A" } })
    await settle()
    act(() => view.result.current.setMessageValue("LONG A DRAFT"))
    await settle()
    metadata = { wasExpanded: true }
    view.rerender({ draftKey: "saved-A" })
    view.rerender({ draftKey: "empty-B" })
    await settle()
    metadata = { wasExpanded: false }
    view.rerender({ draftKey: "saved-A" })
    await settle()
    expect(view.result.current.form.values.message).toBe("LONG A DRAFT")
    expect(metadata).toEqual({ wasExpanded: true })
  })

  it("restores destination text when only inherited collapse metadata changes during hydration", async () => {
    const destination = mount(makeTabStorage(), false, "held-B")
    await settle(); await enter(destination, "STORED B"); destination.unmount()
    vi.stubGlobal("sessionStorage", makeTabStorage())
    let metadata = { wasExpanded: false }
    const view = renderHook(({ draftKey }) => useComposerText({ draftKey, tabScopedDraft: true, textareaRef: React.createRef<HTMLTextAreaElement>(),
      getDraftMetadata: () => metadata
    }), { initialProps: { draftKey: "saved-A" } })
    await settle()
    act(() => view.result.current.setMessageValue("SOURCE A"))
    await settle()
    let finish!: () => void
    const pending = new Promise<void>(resolve => { finish = resolve })
    const original = PlasmoStorage.prototype.get
    const read = vi.spyOn(PlasmoStorage.prototype, "get").mockImplementation(async function (key) {
      if (String(key).includes("held-B")) await pending
      return original.call(this, key)
    })
    try {
      view.rerender({ draftKey: "held-B" })
      await act(async () => { await Promise.resolve(); await Promise.resolve() })
      metadata = { wasExpanded: true }
      view.rerender({ draftKey: "held-B" })
      await act(async () => { finish(); await pending })
      await settle()
      expect(view.result.current.form.values.message).toBe("STORED B")
    } finally { finish?.(); read.mockRestore() }
  })

  it("retains the next draft when an older submitted attempt finishes", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "SUBMITTED")
    const clearAfterSend = view.result.current.clearDraft
    let attempt!: number
    act(() => { attempt = view.result.current.beginPromptAssistReset() })
    await settle()
    act(() => view.result.current.setMessageValue("NEXT DRAFT"))
    act(() => { view.result.current.markPromptAssistAttemptSaved(attempt); clearAfterSend(attempt) })
    view.unmount()
    const restored = mount(tab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("NEXT DRAFT")
  })

  it("does not let an old clear callback remove its retained source after a key change", async () => {
    const tab = makeTabStorage()
    const view = mount(tab, false, "source-draft")
    await settle(); await enter(view, "RETAINED SOURCE")
    const clearSource = view.result.current.clearDraft
    view.rerender({ draftKey: "destination-draft" })
    await settle()
    act(() => clearSource())
    view.unmount()
    const source = mount(tab, false, "source-draft")
    await settle()
    expect(source.result.current.form.values.message).toBe("RETAINED SOURCE")
  })

  it.each(["", "PREVIOUSLY SAVED"])("preserves an immediate route departure after editing %j", async previous => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle()
    if (previous) await enter(view, previous)
    act(() => view.result.current.setMessageValue("LATEST BEFORE IN-APP NAVIGATION"))
    view.unmount()
    const restored = mount(tab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("LATEST BEFORE IN-APP NAVIGATION")
  })

  it("does not revive an explicitly cleared pending draft when the route unmounts", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "PREVIOUSLY SAVED")
    act(() => view.result.current.setMessageValue("PENDING EDIT"))
    act(() => { view.result.current.clearDraft(); view.result.current.setMessageValue("") })
    view.unmount()
    const restored = mount(tab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("")
  })

  it("preserves the latest edit on route unmount under StrictMode", async () => {
    const tab = makeTabStorage()
    const view = mount(tab, true)
    await settle(); await enter(view, "PREVIOUSLY SAVED")
    act(() => view.result.current.setMessageValue("FIRST PENDING EDIT"))
    act(() => view.result.current.setMessageValue("LATEST PENDING EDIT"))
    view.unmount()
    const restored = mount(tab, true)
    await settle()
    expect(restored.result.current.form.values.message).toBe("LATEST PENDING EDIT")
  })

  it("keeps the other native tab's draft separate when a pending route edit flushes", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    const first = mount(a)
    await settle(); await enter(first, "PRIMARY CHARACTER DRAFT")
    const second = mount(b)
    await settle()
    act(() => second.result.current.setMessageValue("SECONDARY NAVIGATION DRAFT"))
    second.unmount()
    const restoredSecond = mount(b)
    await settle()
    expect(restoredSecond.result.current.form.values.message).toBe("SECONDARY NAVIGATION DRAFT")
    first.unmount()
    const restoredFirst = mount(a)
    await settle()
    expect(restoredFirst.result.current.form.values.message).toBe("PRIMARY CHARACTER DRAFT")
  })

  it("does not flush a revoked owner's edit when account change also unmounts the route", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "ALICE SAVED")
    act(() => view.result.current.setMessageValue("ALICE PENDING"))
    act(() => {
      authority.user = "bob"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
      view.unmount()
    })
    const bob = mount(tab)
    await settle()
    expect(bob.result.current.form.values.message).toBe("")
    bob.unmount()
    authority.user = "alice"
    const alice = mount(tab)
    await settle()
    expect(alice.result.current.form.values.message).toBe("ALICE SAVED")
  })

  it.each(["pagehide", "hidden"])("restores the latest edit after %s before the save delay elapses", async event => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "PREVIOUSLY SAVED")
    act(() => view.result.current.setMessageValue("LATEST UNSENT EDIT"))
    const visibility = event === "hidden"
      ? vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
      : null
    act(() => {
      if (event === "hidden") document.dispatchEvent(new Event("visibilitychange"))
      else window.dispatchEvent(new Event("pagehide"))
    })
    // A reload can terminate async durable writes; only the pin already written
    // before the lifecycle handler returns is guaranteed to survive this reload.
    const reloadedTab = makeTabStorage(tab)
    visibility?.mockRestore()
    view.unmount()
    const restored = mount(reloadedTab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("LATEST UNSENT EDIT")
  })

  it("preserves a first draft when pagehide precedes the first debounce", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle()
    act(() => view.result.current.setMessageValue("FIRST UNSENT EDIT"))
    act(() => window.dispatchEvent(new Event("pagehide")))
    const reloadedTab = makeTabStorage(tab)
    view.unmount()
    const restored = mount(reloadedTab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("FIRST UNSENT EDIT")
  })

  it("does not resurrect an explicitly cleared pending edit on pagehide", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "PREVIOUSLY SAVED")
    act(() => view.result.current.setMessageValue("PENDING EDIT"))
    act(() => { view.result.current.clearDraft(); window.dispatchEvent(new Event("pagehide")) })
    const reloadedTab = makeTabStorage(tab)
    view.unmount()
    const restored = mount(reloadedTab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("")
  })

  it("does not flush invalidated owner text during a synchronous account change", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "ALICE SAVED")
    act(() => view.result.current.setMessageValue("ALICE PENDING"))
    act(() => {
      authority.user = "bob"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
      window.dispatchEvent(new Event("pagehide"))
    })
    await settle()
    expect(view.result.current.form.values.message).toBe("")
    act(() => {
      authority.user = "alice"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE SAVED")
  })

  it("reports draft readiness only after current-owner hydration", async () => {
    const view = mount(makeTabStorage())
    expect(view.result.current.draftReady).toBe(false)
    await settle()
    expect(view.result.current.draftReady).toBe(true)
    act(() => {
      authority.user = "bob"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    expect(view.result.current.draftReady).toBe(false)
    await settle()
    expect(view.result.current.draftReady).toBe(true)
  })

  it("restores each active tab's draft after the other tab saves", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    let normal = mount(a)
    await settle()
    await enter(normal, "NORMAL UNSENT")
    let character = mount(b)
    await settle()
    await enter(character, "CHARACTER UNSENT")
    normal.unmount(); character.unmount()
    normal = mount(a)
    await settle()
    expect(normal.result.current.form.values.message).toBe("NORMAL UNSENT")
    character = mount(b)
    await settle()
    expect(character.result.current.form.values.message).toBe("CHARACTER UNSENT")
  })

  it("keeps an explicit clear local to its tab when another tab writes later", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    let normal = mount(a)
    await settle(); await enter(normal, "NORMAL UNSENT")
    let character = mount(b)
    await settle(); await enter(character, "CHARACTER UNSENT")
    act(() => { normal.result.current.clearDraft(); normal.result.current.setMessageValue("") })
    await settle()
    character.unmount(); character = mount(b)
    await settle()
    expect(character.result.current.form.values.message).toBe("CHARACTER UNSENT")
    await enter(character, "CHARACTER NEWER")
    normal.unmount(); normal = mount(a)
    await settle()
    expect(normal.result.current.form.values.message).toBe("")
  })

  it("pins an initially empty tab instead of importing a later other-tab draft", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    let normal = mount(a)
    await settle()
    const character = mount(b)
    await settle(); await enter(character, "OTHER TAB")
    normal.unmount(); normal = mount(a)
    await settle()
    expect(normal.result.current.form.values.message).toBe("")
    normal.unmount(); character.unmount()
    const recovered = mount(makeTabStorage())
    await settle()
    expect(recovered.result.current.form.values.message).toBe("OTHER TAB")
  })

  it("preserves a newer durable draft when another tab's old record expires", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    let old = mount(a)
    await settle(); await enter(old, "OLD TAB")
    old.unmount()
    vi.setSystemTime(Date.now() + 31 * 24 * 60 * 60 * 1000)
    const newer = mount(b)
    await settle(); await enter(newer, "FRESH RECOVERY")
    old = mount(a)
    await settle()
    expect(old.result.current.form.values.message).toBe("")
    old.unmount(); newer.unmount()
    const recovered = mount(makeTabStorage())
    await settle()
    expect(recovered.result.current.form.values.message).toBe("FRESH RECOVERY")
  })

  it("does not resurrect a draft cleared by the tab that last saved it", async () => {
    const view = mount(makeTabStorage())
    await settle(); await enter(view, "CLEAR ME")
    await enter(view, "")
    view.unmount()
    const recovered = mount(makeTabStorage())
    await settle()
    expect(recovered.result.current.form.values.message).toBe("")
  })

  it("recovers the durable last draft in a new tab and keeps duplicated tabs independent", async () => {
    const a = makeTabStorage()
    const normal = mount(a)
    await settle(); await enter(normal, "RECOVERABLE DRAFT")
    const fresh = mount(makeTabStorage())
    await settle()
    expect(fresh.result.current.form.values.message).toBe("RECOVERABLE DRAFT")
    const duplicate = mount(makeTabStorage(a))
    await settle(); await enter(duplicate, "DUPLICATE EDIT")
    normal.unmount()
    const restored = mount(a)
    await settle()
    expect(restored.result.current.form.values.message).toBe("RECOVERABLE DRAFT")
  })

  it("keeps account ownership when each tab has a retained draft", async () => {
    const a = makeTabStorage()
    const view = mount(a)
    await settle(); await enter(view, "ALICE PRIVATE")
    act(() => {
      authority.user = "bob"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await settle()
    expect(view.result.current.form.values.message).toBe("")
    await enter(view, "BOB PRIVATE")
    act(() => {
      authority.user = "alice"
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE PRIVATE")
  })

  it("retains durable draft recovery when session storage rejects writes", async () => {
    const restricted = makeTabStorage()
    restricted.setItem = () => { throw new DOMException("Full", "QuotaExceededError") }
    const view = mount(restricted)
    await settle(); await enter(view, "DURABLE FALLBACK")
    view.unmount()
    const restored = mount(makeTabStorage())
    await settle()
    expect(restored.result.current.form.values.message).toBe("DURABLE FALLBACK")
  })

  it("does not restore an old pin after a later draft only saves durably", async () => {
    const tab = makeTabStorage()
    const view = mount(tab)
    await settle(); await enter(view, "OLD PIN")
    tab.setItem = () => { throw new DOMException("Full", "QuotaExceededError") }
    await enter(view, "NEW DURABLE DRAFT")
    view.unmount()
    const restored = mount(tab)
    await settle()
    expect(restored.result.current.form.values.message).toBe("NEW DURABLE DRAFT")
  })

  it("keeps another tab's recovery draft when clearing hits session quota", async () => {
    const a = makeTabStorage(), b = makeTabStorage()
    const first = mount(a)
    await settle(); await enter(first, "OLDER A")
    const second = mount(b)
    await settle(); await enter(second, "NEWER B")
    a.setItem = () => { throw new DOMException("Full", "QuotaExceededError") }
    act(() => { first.result.current.clearDraft(); first.result.current.setMessageValue("") })
    await settle()
    first.unmount(); second.unmount()
    const restored = mount(makeTabStorage())
    await settle()
    expect(restored.result.current.form.values.message).toBe("NEWER B")
  })
})
