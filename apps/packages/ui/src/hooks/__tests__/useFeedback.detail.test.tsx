import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { NativeHistoryOwnerV1 } from "@/services/chat-history-selection"
import { useFeedback } from "../useFeedback"
import { useFeedbackStore } from "@/store/feedback"

const mocks = vi.hoisted(() => ({
  submitExplicitFeedback: vi.fn().mockResolvedValue({ ok: true, feedback_id: "fb-1" }),
  updateChatRating: vi.fn(),
  notificationError: vi.fn(),
  setServerChatVersion: vi.fn(),
  controller: null as {
    getCurrent: () => { owner: NativeHistoryOwnerV1 }
    fence: () => () => boolean
    getSignal: () => AbortSignal
    settingsMode: () => "ordinary"
  } | null,
  serverChatVersion: 4,
  core: { historyId: "H_1", serverChatId: "C_1", serverChatVersion: 4, chatDraftAddress: null, chatDraftRevision: 1, quickPromptRevision: 0 }
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "sess-test",
  submitExplicitFeedback: mocks.submitExplicitFeedback,
  updateChatRating: mocks.updateChatRating
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign((selector: (state: { serverChatVersion: number; setServerChatVersion: typeof mocks.setServerChatVersion }) => unknown) =>
    selector({ serverChatVersion: mocks.serverChatVersion, setServerChatVersion: mocks.setServerChatVersion }),
    { getState: () => mocks.core })
}))

vi.mock("../chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => mocks.controller
}))

vi.mock("../useAntdNotification", () => ({
  useAntdNotification: () => ({ error: mocks.notificationError })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

describe("useFeedback detail submission", () => {
  beforeEach(() => {
    mocks.submitExplicitFeedback.mockClear()
    mocks.updateChatRating.mockReset()
    mocks.controller = null
    mocks.core = { historyId: "H_1", serverChatId: "C_1", serverChatVersion: 4, chatDraftAddress: null, chatDraftRevision: 1, quickPromptRevision: 0 }
    mocks.setServerChatVersion.mockClear()
    mocks.notificationError.mockClear()
    useFeedbackStore.setState({ entries: {} })
    window.localStorage.clear()
  })

  const ownedRating = () => {
    let current = true
    const signal = new AbortController().signal
    const owner: NativeHistoryOwnerV1 = {
      kind: "native", conversation_id: "C_1",
      request_scope: { config: { serverUrl: "https://owned.test", authMode: "single-user" }, userId: 1 },
      scope: { type: "workspace", workspaceId: "W" }, validate_lease: () => current
    }
    mocks.controller = { getCurrent: () => ({ owner }), fence: () => () => current, getSignal: () => signal, settingsMode: () => "ordinary" }
    return { owner, signal, retire: () => { current = false } }
  }

  const renderFeedback = () => renderHook(() => useFeedback({ messageKey: "srv:M_1", conversationId: "C_1", messageId: "M_1" }))

  it("does not replay a stale advisory rating after explicit feedback succeeds", async () => {
    ownedRating()
    mocks.updateChatRating.mockRejectedValue(Object.assign(new Error("Version conflict"), { status: 409 }))
    const { result } = renderFeedback()
    await act(async () => { expect(await result.current.submitThumb("up")).toBe(true) })
    expect(mocks.updateChatRating).toHaveBeenCalledTimes(1)
    expect(mocks.setServerChatVersion).not.toHaveBeenCalled()
  })

  it("keeps the rating destination and version from the rendered native owner", async () => {
    const { owner, signal } = ownedRating()
    mocks.updateChatRating.mockResolvedValue({ id: "C_1", version: 5 })
    const { result } = renderFeedback()
    await act(async () => { await result.current.submitThumb("up") })
    expect(mocks.updateChatRating).toHaveBeenCalledWith("C_1", 5, 4, { requestScope: owner.request_scope, scope: owner.scope, signal })
    expect(mocks.setServerChatVersion).toHaveBeenCalledWith(5)
  })

  it("does not publish a held rating after its native owner retires", async () => {
    const { retire } = ownedRating()
    let release!: (value: { id: string; version: number }) => void
    let enter!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    mocks.updateChatRating.mockImplementation(() => { enter(); return new Promise<{ id: string; version: number }>(resolve => { release = resolve }) })
    const { result } = renderFeedback()
    await act(async () => {
      const pending = result.current.submitThumb("up")
      await entered
      retire()
      release({ id: "C_1", version: 5 })
      await pending
    })
    expect(mocks.setServerChatVersion).not.toHaveBeenCalled()
  })

  it.each(["newer-version", "same-id-restore"])("does not publish a held rating after %s", async (change) => {
    ownedRating()
    let release!: (value: { id: string; version: number }) => void
    let enter!: () => void
    const entered = new Promise<void>(resolve => { enter = resolve })
    mocks.updateChatRating.mockImplementation(() => { enter(); return new Promise<{ id: string; version: number }>(resolve => { release = resolve }) })
    const { result } = renderFeedback()
    await act(async () => {
      const pending = result.current.submitThumb("up")
      await entered
      if (change === "newer-version") mocks.core = { ...mocks.core, serverChatVersion: 6 }
      else mocks.core = { ...mocks.core, chatDraftRevision: mocks.core.chatDraftRevision + 1 }
      release({ id: "C_1", version: 5 })
      await pending
    })
    expect(mocks.setServerChatVersion).not.toHaveBeenCalled()
  })

  it("does not infer a metadata owner for an unqualified conversation", async () => {
    const { result } = renderFeedback()
    await act(async () => { expect(await result.current.submitThumb("up")).toBe(true) })
    expect(mocks.updateChatRating).not.toHaveBeenCalled()
  })

  it("submits rating + issues + notes in a single explicit request", async () => {
    const { result } = renderHook(() =>
      useFeedback({
        messageKey: "srv:M_1",
        conversationId: "C_1",
        messageId: "M_1",
        query: null
      })
    )

    let ok = false
    await act(async () => {
      ok = await result.current.submitDetail({
        rating: 3,
        issues: ["missing_details", "sources_unhelpful"],
        notes: "Needs source grounding."
      })
    })

    expect(ok).toBe(true)
    expect(mocks.submitExplicitFeedback).toHaveBeenCalledTimes(1)
    expect(mocks.submitExplicitFeedback).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_id: "C_1",
        message_id: "M_1",
        session_id: "sess-test",
        feedback_type: "relevance",
        relevance_score: 3,
        issues: ["missing_details", "sources_unhelpful"],
        user_notes: "Needs source grounding.",
        idempotency_key: expect.any(String)
      })
    )
  })
})
