import { beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { captureChatDraftSource } from "../chat-draft-transition"

beforeEach(() => {
  usePlaygroundSessionStore.getState().clearSession()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  useStoreMessageOption.getState().startNewChatDraft(false)
})

describe("original draft source route acknowledgement", () => {
  it("accepts the exact next restore receipt and then retains that exact revision", () => {
    const source = captureChatDraftSource()
    const original = usePlaygroundSessionStore.getState().restoreRevision
    expect(source.isCurrent()).toBe(true)
    usePlaygroundSessionStore.getState().cancelPendingRestore()
    expect(source.isCurrent()).toBe(false)
    expect(source.acceptRestoreCancelled(original + 1)).toBe(true)
    expect(source.isCurrent()).toBe(true)
    expect(source.acceptRestoreCancelled(original + 1)).toBe(false)
    expect(source.isCurrent()).toBe(true)
    usePlaygroundSessionStore.getState().cancelPendingRestore()
    expect(source.isCurrent()).toBe(false)
  })

  it.each([
    "old receipt",
    "future receipt",
    "later cancellation",
    "New",
    "New then original address",
    "raw IDs",
    "original owner"
  ])("refuses %s without adopting current ownership", (retirement) => {
    let ownerCurrent = true
    const source = captureChatDraftSource(() => ownerCurrent)
    const original = usePlaygroundSessionStore.getState().restoreRevision
    const address = useStoreMessageOption.getState().chatDraftAddress
    usePlaygroundSessionStore.getState().cancelPendingRestore()
    if (retirement === "later cancellation")
      usePlaygroundSessionStore.getState().cancelPendingRestore()
    if (retirement === "New" || retirement === "New then original address") {
      useStoreMessageOption.getState().startNewChatDraft(false)
      if (retirement === "New then original address")
        useStoreMessageOption.getState().setChatDraftAddress(address)
    }
    if (retirement === "raw IDs")
      useStoreMessageOption.setState({ historyId: "replacement" })
    if (retirement === "original owner") ownerCurrent = false
    const receipt =
      retirement === "old receipt"
        ? original
        : retirement === "future receipt"
          ? original + 2
          : usePlaygroundSessionStore.getState().restoreRevision
    const destination = useStoreMessageOption.getState()
    expect(source.acceptRestoreCancelled(receipt)).toBe(false)
    expect(source.isCurrent()).toBe(false)
    expect(useStoreMessageOption.getState()).toBe(destination)
  })
})
