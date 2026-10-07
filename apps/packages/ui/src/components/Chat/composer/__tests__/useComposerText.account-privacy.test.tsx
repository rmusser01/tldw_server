import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"
import { MemoryRouter } from "react-router-dom"
import { useSidepanelChatHandoffImport } from "../../../Option/Playground/hooks/useSidepanelChatHandoffImport"
import { encodeSidepanelChatWebUiHandoff, SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE } from "@/services/tldw/sidepanel-chat-webui-handoff"
import type { SidepanelChatHandoffPackage } from "@/services/sidepanel-chat-handoff"
import { Storage } from "@plasmohq/storage"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

const handoffIO = vi.hoisted(() => ({ read: vi.fn(), consume: vi.fn() }))
vi.mock("@/services/sidepanel-chat-handoff", () => ({ readSidepanelChatHandoff: handoffIO.read, consumeSidepanelChatHandoff: handoffIO.consume }))
const authority = vi.hoisted(() => ({ user: "alice" as string | null, org: null as number | null, error: null as unknown }))
vi.mock("@plasmohq/storage", () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  resolveServicePromptScope: async () => {
    if (authority.error) throw authority.error
    if (!authority.user) throw new Error("Signed out")
    return { scopeKey: authority.user, userId: authority.user,
      config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual", orgId: authority.org } }
  }
}))

import { useComposerText } from "../hooks/useComposerText"

const draftKey = "tldw:playgroundChatDraft"
const ownerRecordKeys = () => Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!)
  .filter(key => key.startsWith(`registry:draft:${draftKey}:owner:`))
const settle = async () => {
  await act(async () => { await vi.dynamicImportSettled() })
  await act(async () => { await vi.advanceTimersByTimeAsync(COMPOSER_CONSTANTS.DRAFT_SAVE_DEBOUNCE_MS + 1) })
}
const switchTo = async (user: string) => {
  act(() => {
    authority.user = null
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
  })
  await act(async () => {
    authority.user = user
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
  })
}

describe.each([false, true])("composer account ownership using the real draft registry (tabScoped=%s)", tabScopedDraft => {
  const mount = () => renderHook(() => useComposerText({ draftKey, tabScopedDraft, textareaRef: React.createRef<HTMLTextAreaElement>() }))
  beforeEach(() => { vi.useFakeTimers(); localStorage.clear(); sessionStorage.clear(); authority.user = "alice"; authority.org = null; authority.error = null })
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

  it("rejects unowned raw and registry drafts from earlier releases", async () => {
    localStorage.setItem(draftKey, "UNOWNED PRIVATE")
    localStorage.setItem(`registry:draft:${draftKey}`, JSON.stringify({ value: "UNOWNED PRIVATE", updatedAt: Date.now() }))
    const { result } = mount()
    await settle()
    expect(result.current.form.values.message).toBe("")
    expect(localStorage.getItem(draftKey)).toBeNull()
    expect(localStorage.getItem(`registry:draft:${draftKey}`)).toBeNull()
  })

  it("clears an active message and image on logout and never restores them for Bob", async () => {
    const { result } = mount()
    await settle()
    act(() => { result.current.setMessageValue("ALICE PRIVATE"); result.current.form.setFieldValue("image", "alice-image") })
    await settle()
    await switchTo("bob")
    await settle()
    expect(result.current.form.values).toEqual({ message: "", image: "" })
  })

  it("recovers each owner's draft on reload and reciprocal login", async () => {
    let view = mount()
    await settle()
    act(() => view.result.current.setMessageValue("ALICE PRIVATE"))
    await settle()
    expect(view.result.current.draftSaved).toBe(true)
    const ownerRecordKey = ownerRecordKeys()[0]
    expect(localStorage.getItem(ownerRecordKey)).toContain("ALICE PRIVATE")
    view.unmount()
    view = mount()
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE PRIVATE")
    await switchTo("bob")
    await settle()
    expect(localStorage.getItem(ownerRecordKey)).toContain("ALICE PRIVATE")
    expect(view.result.current.form.values.message).toBe("")
    act(() => view.result.current.setMessageValue("BOB PRIVATE"))
    await settle()
    expect(localStorage.getItem(ownerRecordKey)).toContain("ALICE PRIVATE")
    await switchTo("alice")
    await settle()
    expect(localStorage.getItem(ownerRecordKey)).toContain("ALICE PRIVATE")
    expect(view.result.current.form.values.message).toBe("ALICE PRIVATE")
  })

  it("keeps unsaved edits on an ordinary same-principal config update", async () => {
    const { result } = mount()
    await settle()
    act(() => result.current.setMessageValue("CURRENT EDIT"))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
    await settle()
    expect(result.current.form.values.message).toBe("CURRENT EDIT")
  })

  it.each([
    ["focus", new TypeError("Failed to fetch")],
    ["pageshow", { status: 500, message: "Temporarily unavailable" }]
  ])("retains pending same-owner text and image after a transient %s recheck", async (event, error) => {
    const { result } = mount()
    await settle()
    act(() => result.current.setMessageValue("ALICE SAVED"))
    await settle()
    const ownerKey = ownerRecordKeys()[0]
    act(() => {
      result.current.setMessageValue("ALICE PENDING")
      result.current.form.setFieldValue("image", "alice-image")
      authority.error = error
      window.dispatchEvent(new Event(event))
    })
    await act(async () => { await vi.dynamicImportSettled() })
    expect(result.current.form.values).toEqual({ message: "ALICE PENDING", image: "alice-image" })
    expect(localStorage.getItem(ownerKey)).toContain("ALICE SAVED")
    await settle()
    expect(localStorage.getItem(ownerKey)).toContain("ALICE PENDING")
    expect(ownerRecordKeys()).toEqual([ownerKey])
    authority.error = null
    act(() => window.dispatchEvent(new Event(event)))
    await settle()
    expect(result.current.form.values).toEqual({ message: "ALICE PENDING", image: "alice-image" })
  })

  it.each([
    { code: "service_prompt_scope_unresolved" },
    { status: 401 },
    createServicePromptScopeChangedError()
  ])("clears pending private content when revalidation confirms authority loss: %j", async error => {
    const { result } = mount()
    await settle()
    act(() => {
      result.current.setMessageValue("ALICE PENDING")
      result.current.form.setFieldValue("image", "alice-image")
      authority.error = error
      window.dispatchEvent(new Event("focus"))
    })
    await settle()
    expect(result.current.form.values).toEqual({ message: "", image: "" })
    expect(ownerRecordKeys()).toEqual([])
  })

  it.each([
    ["tldw:auth-principal-changed", { kind: "logout" }],
    ["tldw:auth-credentials-changed", {}],
    ["tldw:config-updated", { authorityChanged: true }]
  ])("clears pending private content at %s even while offline", async (event, detail) => {
    const { result } = mount()
    await settle()
    act(() => {
      result.current.setMessageValue("ALICE PENDING")
      result.current.form.setFieldValue("image", "alice-image")
      authority.error = new TypeError("Failed to fetch")
      window.dispatchEvent(new CustomEvent(event, { detail }))
    })
    await settle()
    expect(result.current.form.values).toEqual({ message: "", image: "" })
    expect(ownerRecordKeys()).toEqual([])
  })

  it("keeps organization changes separate even for the same account", async () => {
    const { result } = mount()
    await settle()
    act(() => result.current.setMessageValue("WORKSPACE-LESS DRAFT"))
    await settle()
    act(() => {
      authority.org = 2
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await settle()
    expect(result.current.form.values.message).toBe("")
    act(() => {
      authority.org = null
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    await settle()
    expect(result.current.form.values.message).toBe("WORKSPACE-LESS DRAFT")
  })

  it("ignores a delayed Alice read after Alice to Bob to Alice and a newer edit", async () => {
    let view = mount()
    await settle()
    act(() => view.result.current.setMessageValue("ALICE OLD DRAFT"))
    await settle()
    const key = ownerRecordKeys()[0]
    view.unmount()
    // A newly opened tab must read the durable fallback before it can pin it.
    sessionStorage.clear()
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let blocked = false
    const original = Storage.prototype.get
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function<T>(this: Storage, candidate: string): Promise<T | undefined> {
      const value = await original.call(this, candidate) as T | undefined
      if (candidate === key && !blocked) { blocked = true; await held }
      return value
    })
    view = mount()
    await settle()
    expect(blocked).toBe(true)
    await switchTo("bob")
    await settle()
    expect(view.result.current.form.values.message).toBe("")
    await switchTo("alice")
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE OLD DRAFT")
    act(() => view.result.current.setMessageValue("ALICE NEW EDIT"))
    await act(async () => { release() })
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE NEW EDIT")
  })

  it("keeps a storage write already in flight under its captured owner", async () => {
    const view = mount()
    await settle()
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let blocked = false
    const original = Storage.prototype.set
    vi.spyOn(Storage.prototype, "set").mockImplementation(async function<T>(this: Storage, key: string, value: T) {
      if (key.startsWith(`registry:draft:${draftKey}:owner:`) && !blocked) {
        blocked = true
        await held
      }
      return original.call(this, key, value)
    })
    act(() => view.result.current.setMessageValue("ALICE PENDING WRITE"))
    await settle()
    expect(blocked).toBe(true)
    await switchTo("bob")
    await settle()
    act(() => view.result.current.setMessageValue("BOB CURRENT DRAFT"))
    await settle()
    await act(async () => { release() })
    await settle()
    expect(view.result.current.form.values.message).toBe("BOB CURRENT DRAFT")
    await switchTo("alice")
    await settle()
    expect(view.result.current.form.values.message).toBe("ALICE PENDING WRITE")
  })
})


describe("copied drafts use the actual composer restoration and account watcher", () => {
  const copy: SidepanelChatHandoffPackage = {
    id: "owned-copy", source: "sidepanel-chat", createdAt: "2026-10-06T00:00:00Z", expiresAt: "2026-10-06T00:10:00Z",
    draft: { text: "COPIED TEXT  " }, pageContext: { title: "Copied page", snippets: [] }
  }
  const fragmentEntry = (query = false) => {
    const token = encodeSidepanelChatWebUiHandoff({ source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE, createdAt: Date.now(), draft: copy.draft.text })
    return query ? `/chat?handoff=${token}` : `/chat#handoff=${token}`
  }
  const mount = (entry = "/chat?handoff=owned-copy") => renderHook(() => {
    const text = useComposerText({ draftKey, textareaRef: React.useRef<HTMLTextAreaElement>(null) })
    const captureGuard = React.useCallback(() => text.isDraftOwnerCurrent, [text.isDraftOwnerCurrent])
    const handoff = useSidepanelChatHandoffImport({ ready: text.draftReady, destinationKey: text.draftOwnerKey ?? "unresolved",
      captureGuard, draftValue: text.form.values.message, getDraftRevision: text.form.getRevision, setMessageValue: text.setMessageValue,
      t: (_key, fallback) => typeof fallback === "string" ? fallback : _key })
    return { text, handoff }
  }, { wrapper: ({ children }) => <MemoryRouter initialEntries={[entry]}>{children}</MemoryRouter> })
  beforeEach(() => {
    vi.useFakeTimers(); localStorage.clear(); sessionStorage.clear(); authority.user = "alice"; authority.org = null; authority.error = null
    handoffIO.read.mockReset().mockResolvedValue(copy); handoffIO.consume.mockReset().mockResolvedValue(copy)
  })
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

  it.each(["local", "fragment", "encoded query"])("waits for actual cold durable restoration before reviewing a %s copy", async transport => {
    const seed = renderHook(() => useComposerText({ draftKey, textareaRef: React.createRef<HTMLTextAreaElement>() }))
    await settle()
    act(() => seed.result.current.setMessageValue("RESTORED DESTINATION  "))
    await settle()
    const key = ownerRecordKeys()[0]
    seed.unmount()
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const original = Storage.prototype.get
    vi.spyOn(Storage.prototype, "get").mockImplementation(async function<T>(this: Storage, candidate: string): Promise<T | undefined> {
      const value = await original.call(this, candidate) as T | undefined
      if (candidate === key) await held
      return value
    })
    const view = mount(transport === "local" ? undefined : fragmentEntry(transport === "encoded query"))
    await settle()
    expect(view.result.current.text.draftReady).toBe(false)
    expect(handoffIO.read).not.toHaveBeenCalled()
    await act(async () => { release() })
    await settle()
    expect(view.result.current.text.form.values.message).toBe("RESTORED DESTINATION  ")
    expect(view.result.current.handoff.conflict?.handoff.draft.text).toBe("COPIED TEXT  ")
    expect(handoffIO.consume).not.toHaveBeenCalled()
  })

  it("retires yielded fragment choices through the real synchronous account watcher", async () => {
    const view = mount(fragmentEntry())
    await settle()
    expect(view.result.current.text.draftReady).toBe(true)
    expect(view.result.current.handoff.conflict?.handoff.draft.text).toBe(copy.draft.text)
    const cancel = view.result.current.handoff.cancelHandoffImport
    let pending!: Promise<void>
    act(() => {
      pending = view.result.current.handoff.replaceWithHandoffDraft()
      authority.user = null
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    })
    await switchTo("bob")
    await settle()
    act(() => view.result.current.text.setMessageValue("BOB CURRENT  "))
    await act(async () => { await pending; cancel() })
    await settle()
    expect(view.result.current.text.form.values.message).toBe("BOB CURRENT  ")
    expect(view.result.current.handoff.importedContext).toBeNull()
    expect(view.result.current.handoff.conflict?.handoff.draft.text).toBe(copy.draft.text)
    expect(handoffIO.read).not.toHaveBeenCalled()
    expect(handoffIO.consume).not.toHaveBeenCalled()
  })

  it.each(["initial", "choice"])("retires a held %s read through the real synchronous account event", async mode => {
    let finish!: (value: SidepanelChatHandoffPackage) => void
    const pending = new Promise<SidepanelChatHandoffPackage>(resolve => { finish = resolve })
    if (mode === "initial") handoffIO.read.mockReturnValueOnce(pending)
    else {
      const seed = renderHook(() => useComposerText({ draftKey, textareaRef: React.createRef<HTMLTextAreaElement>() }))
      await settle(); act(() => seed.result.current.setMessageValue("ALICE DESTINATION  ")); await settle(); seed.unmount()
    }
    const view = mount()
    await settle()
    if (mode === "choice") {
      expect(view.result.current.handoff.conflict).not.toBeNull()
      handoffIO.read.mockReturnValueOnce(pending)
      act(() => { void view.result.current.handoff.replaceWithHandoffDraft() })
    }
    await switchTo("bob")
    await settle()
    act(() => view.result.current.text.setMessageValue("BOB CURRENT  "))
    await act(async () => { finish(copy) })
    await settle()
    expect(view.result.current.text.form.values.message).toBe("BOB CURRENT  ")
    expect(view.result.current.handoff.importedContext).toBeNull()
    expect(handoffIO.consume).not.toHaveBeenCalled()
  })
})


describe("owned original creation text uses its displayed source fence", () => {
  beforeEach(() => {
    vi.restoreAllMocks(); vi.useFakeTimers(); localStorage.clear(); sessionStorage.clear()
    authority.user = "alice"; authority.org = null; authority.error = null
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: vi.fn().mockResolvedValue(undefined) } })
    const tails = new Map<string, Promise<unknown>>()
    Object.defineProperty(navigator, "locks", { configurable: true, value: { request: (key: string, work: () => unknown) => {
      const next = (tails.get(key) ?? Promise.resolve()).then(work); tails.set(key, next.catch(() => undefined)); return next
    } } })
  })
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })
  const mountCreation = async (seed = true) => {
    const { useStoreMessageOption: store } = await import("@/store/option")
    store.getState().startNewChatDraft(false)
    const address = store.getState().chatDraftAddress!
    const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
    const { createChatWithRecovery } = await import("@/services/pending-chat-promotion")
    const requestScope = await (await import("@/services/service-prompts")).resolveServicePromptScope()
    const snapshot = { requestScope, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() } as unknown as import("@/services/service-prompts").ServicePromptSnapshot
    const retainCreation = () => createChatWithRecovery({ snapshot, handle: { address, ownerHash: chatDraftOwnerKey(requestScope) }, surface: "playground",
      inputText: "ALICE ORIGINAL ATTEMPT", setup: {}, isCurrent: () => true, create: async () => { throw new Error("lost ACK") } }).catch(() => undefined)
    if (seed) await retainCreation()
    const view = renderHook(() => useComposerText({ draftKey, draftAddress: address, recoverySurface: "playground", textareaRef: React.createRef<HTMLTextAreaElement>() }))
    await settle()
    expect(view.result.current.creationRecovery.active?.inputText).toBe(seed ? "ALICE ORIGINAL ATTEMPT" : undefined)
    return { view, store, address, retainCreation, copy: view.result.current.copyCreationText }
  }
  it.each(["new", "ABA", "logout"])("does not copy an old displayed attempt after %s before rerender", async boundary => {
    const { store, address, copy } = await mountCreation()
    await act(async () => {
      if (boundary === "logout") { authority.user = null; window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } })) }
      else { store.getState().startNewChatDraft(false); if (boundary === "ABA") store.getState().setChatDraftAddress(address) }
      expect(await copy()).toBe(false)
    })
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
  })
  it("observes a new uncertain creation after a same-object source ABA without a prior marker", async () => {
    const { store, address, view, retainCreation } = await mountCreation(false)
    act(() => { store.getState().startNewChatDraft(false); store.getState().setChatDraftAddress(address) })
    await settle()
    await act(async () => { await retainCreation() })
    await settle()
    expect(view.result.current.creationRecovery.active).toMatchObject({ state: "unconfirmed", inputText: "ALICE ORIGINAL ATTEMPT" })
    expect(await view.result.current.copyCreationText()).toBe(true)
  })
  it("copies the owned original text and suppresses a late clipboard error after New", async () => {
    const { store, copy } = await mountCreation()
    expect(await copy()).toBe(true)
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("ALICE ORIGINAL ATTEMPT")
    let reject!: (reason: Error) => void
    vi.mocked(navigator.clipboard.writeText).mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail }))
    const pending = copy()
    await act(async () => { await vi.dynamicImportSettled() })
    expect(reject).toBeTypeOf("function")
    act(() => store.getState().startNewChatDraft(false))
    reject(new Error("Clipboard unavailable"))
    expect(await pending).toBe(false)
  })
})
