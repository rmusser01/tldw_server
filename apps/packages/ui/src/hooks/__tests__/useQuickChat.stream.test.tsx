import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useQuickChatStore } from "@/store/quick-chat"
const calls = vi.hoisted(() => ({ stream: vi.fn() }))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (_key: string, value: unknown) => [value] }))
vi.mock("@/hooks/chat/useSelectedModel", () => ({ useSelectedModel: () => ({ selectedModel: "plain-model" }) }))
vi.mock("@/services/tldw/TldwChat", () => ({ TldwChatService: class {
  streamMessage = (...args: unknown[]) => calls.stream(...args)
  cancelStream() {}
} }))
import { useQuickChat } from "../useQuickChat"
beforeEach(() => {
  useQuickChatStore.getState().clearMessages()
  calls.stream.mockReset()
})
describe("QuickChat text stream contract", () => {
  it("keeps ordinary token order and completes the plain QuickChat message", async () => {
    calls.stream.mockImplementation(async function* () { yield "Plain "; yield "answer" })
    const { result } = renderHook(() => useQuickChat())
    await act(async () => { await result.current.sendMessage("Question") })
    expect(useQuickChatStore.getState().messages.map(message => message.content)).toEqual(["Question", "Plain answer"])
    expect(useQuickChatStore.getState().isStreaming).toBe(false)
    expect(calls.stream.mock.calls[0][1]).toEqual({ model: "plain-model", stream: true })
  })
  it("does not render nontext stream observations as object strings", async () => {
    calls.stream.mockImplementation(async function* () {
      yield "Plain "
      yield { tldw_history_admission_v1: { version: 1 } }
      yield "answer"
    })
    const { result } = renderHook(() => useQuickChat())
    await act(async () => { await result.current.sendMessage("Question") })
    expect(useQuickChatStore.getState().messages.at(-1)?.content).toBe("Plain answer")
  })
})
