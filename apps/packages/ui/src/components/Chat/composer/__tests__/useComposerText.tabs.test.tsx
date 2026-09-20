import React from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"

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
const mount = (tab: Storage) => {
  vi.stubGlobal("sessionStorage", tab)
  return renderHook(() => useComposerText({
    draftKey: "tldw:playgroundChatDraft",
    tabScopedDraft: true,
    textareaRef: React.createRef<HTMLTextAreaElement>()
  }))
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
