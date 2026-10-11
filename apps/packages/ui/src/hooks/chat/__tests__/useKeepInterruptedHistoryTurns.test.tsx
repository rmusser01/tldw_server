import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { useKeepInterruptedHistoryTurns } from "../useKeepInterruptedHistoryTurns"
import { keepHistoryTurnRecovery, markHistoryTurnLive, releaseHistoryTurn, resetHistoryTurnRegistry } from "@/services/history-turn-keep"
import type { HistorySelectionController } from "../useHistorySelection"
import type { Message } from "@/store/option"

vi.mock("@/services/history-turn-keep", async original => ({
  ...await original<typeof import("@/services/history-turn-keep")>(),
  keepHistoryTurnRecovery: vi.fn().mockResolvedValue({ status: "skipped", reason: "view" })
}))

beforeEach(() => resetHistoryTurnRegistry())

it.each(["failed", "stopped"])("keeps original recovery after a %s replacement is released", status => {
  const entry = {
    scope: { profile_id: "profile-1", client_session_id: "session-1" },
    turn: { operation_id: "original", outcome: "interrupted", input_id: "question", assistant_id: "partial", result_text: "Original partial", created_at: 1 }
  }
  const dismiss = vi.fn()
  const selection = { status: "ready", owner: { kind: "native" }, recoveries: [entry], dismissRecovery: dismiss } as unknown as HistorySelectionController
  const mounted = renderHook(({ rows }) => useKeepInterruptedHistoryTurns({ selection, messages: rows, setMessages: vi.fn() }), { initialProps: { rows: [] as Message[] } })
  markHistoryTurnLive("original")
  mounted.rerender({ rows: [{ id: "question", isBot: false, message: "Question" }, { id: "replacement", isBot: true, message: `${status} partial` }] as Message[] })
  act(() => releaseHistoryTurn("original"))
  mounted.rerender({ rows: [{ id: "question", isBot: false, message: "Question" }, { id: "replacement", isBot: true, message: `${status} partial` }] as Message[] })
  expect(dismiss).not.toHaveBeenCalled()
  mounted.unmount()
})

it.each(["failed", "skipped"] as const)("leaves recovery controls visible when automatic Keep is %s", async status => {
  vi.mocked(keepHistoryTurnRecovery).mockResolvedValueOnce(status === "failed" ? { status, reason: "storage_unavailable" } : { status, reason: "not_ready" })
  const entry = {
    scope: { profile_id: "profile-1", client_session_id: "session-1" },
    turn: { operation_id: "unsaved", outcome: "complete", input_id: "question", assistant_id: "partial", result_text: "Unsaved reply", created_at: 1 }
  }
  const selection = { status: "ready", owner: { kind: "local" }, recoveries: [entry] } as unknown as HistorySelectionController
  const mounted = renderHook(() => useKeepInterruptedHistoryTurns({ selection, messages: [], setMessages: vi.fn() }))
  await act(async () => {})
  expect(mounted.result.current(entry as Parameters<typeof mounted.result.current>[0])).toBe(false)
  mounted.unmount()
})

it("shows an already kept original in recovery controls when a failed replacement occupies the transcript", async () => {
  vi.mocked(keepHistoryTurnRecovery).mockResolvedValueOnce({ status: "kept", resultId: "question", followed: false, retained: true })
  const entry = {
    scope: { profile_id: "profile-1", client_session_id: "session-1" },
    turn: { operation_id: "kept-original", outcome: "interrupted", input_id: "question", assistant_id: "original-partial", result_text: "Original partial", created_at: 1 }
  }
  const selection = { status: "ready", owner: { kind: "native" }, recoveries: [entry] } as unknown as HistorySelectionController
  const mounted = renderHook(({ rows }) => useKeepInterruptedHistoryTurns({ selection, messages: rows, setMessages: vi.fn() }), { initialProps: { rows: [] as Message[] } })
  await act(async () => {})
  mounted.rerender({ rows: [{ id: "question", isBot: false, message: "Question" }, { id: "replacement-partial", isBot: true, message: "Failed replacement" }] as Message[] })
  expect(mounted.result.current(entry as Parameters<typeof mounted.result.current>[0])).toBe(false)
  mounted.unmount()
})

it.each([false, true])("does not publish or dismiss a cached retained reply during replacement (new reply=%s)", (replacement) => {
  const entry = {
    scope: { profile_id: "profile-1", client_session_id: "session-1" },
    turn: { operation_id: "original", outcome: "interrupted", input_id: "question", assistant_id: "partial", result_text: "Original partial", created_at: 1 }
  }
  const dismiss = vi.fn()
  const selection = { status: "ready", owner: { kind: "native" }, recoveries: [entry], dismissRecovery: dismiss } as unknown as HistorySelectionController
  const setMessages = vi.fn()
  const messages = [{ id: "question", isBot: false, message: "Question" }, ...(replacement ? [{ id: "replacement", isBot: true, message: "Replacement reply" }] : [])] as Message[]
  const mounted = renderHook(({ rows }) => useKeepInterruptedHistoryTurns({ selection, messages: rows, setMessages }), { initialProps: { rows: [] as Message[] } })
  markHistoryTurnLive("original")
  mounted.rerender({ rows: messages })
  expect(dismiss).not.toHaveBeenCalled()
  expect(setMessages).not.toHaveBeenCalled()
  mounted.unmount()
})
