import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { COMPOSER_CONSTANTS } from "@/config/ui-constants"
import { Storage } from "@plasmohq/storage"

const authority = vi.hoisted(() => ({ user: "alice" as string | null, org: null as number | null }))
vi.mock("@plasmohq/storage", () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => {
    if (!authority.user) throw new Error("Signed out")
    return { scopeKey: authority.user, userId: authority.user,
      config: { serverUrl: "http://localhost:8000", authMode: "multi-user", authSource: "manual", orgId: authority.org } }
  }
}))

import { useComposerText } from "../hooks/useComposerText"

const draftKey = "tldw:playgroundChatDraft"
const ownerRecordKeys = () => Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)!)
  .filter(key => key.startsWith(`registry:draft:${draftKey}:owner:`))
const mount = () => renderHook(() => useComposerText({ draftKey, textareaRef: React.createRef<HTMLTextAreaElement>() }))
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

describe("composer account ownership using the real draft registry", () => {
  beforeEach(() => { vi.useFakeTimers(); localStorage.clear(); authority.user = "alice"; authority.org = null })
  afterEach(() => vi.useRealTimers())

  it("rejects unowned raw and registry drafts from earlier releases", async () => {
    localStorage.setItem(draftKey, "UNOWNED PRIVATE")
    localStorage.setItem(`registry:draft:${draftKey}`, JSON.stringify({ value: "UNOWNED PRIVATE", updatedAt: Date.now() }))
    const { result } = mount()
    await settle()
    expect(result.current.form.values.message).toBe("")
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
      await original.call(this, key, value)
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
