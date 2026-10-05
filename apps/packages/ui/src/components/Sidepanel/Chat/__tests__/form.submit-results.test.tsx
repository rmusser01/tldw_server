import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, renderHook } from "@testing-library/react"
import ts from "typescript"
import { describe, expect, it, vi } from "vitest"
import {
  throwIfChatSubmitUnsuccessful,
  type ChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import { useQueuedRequests } from "@/hooks/chat/useQueuedRequests"
import {
  buildQueuedRequest,
  type QueuedRequest
} from "@/utils/chat-request-queue"

const ast = ts.createSourceFile(
  "form.tsx",
  readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX
)
let callback: ts.Expression | undefined
const visit = (node: ts.Node) => {
  if (
    ts.isVariableDeclaration(node) &&
    node.name.getText(ast) === "sendQueuedRequest"
  )
    callback = node.initializer
  ts.forEachChild(node, visit)
}
visit(ast)
if (!callback) throw new Error("Missing actual compact queue callback")
const code = ts.transpileModule(
  `return function useCompactQueue(scope) {
  const { sendMessage, setQueue, queue, isConnectionReady = true, stopListening = async () => {} } = scope;
  const captureQueuedDispatchGuard = () => () => {};
  const validateQueuedRequest = () => null;
  const form = { setFieldError: () => {} };
  const setSelectedModel = () => {}, setChatMode = () => {}, setWebSearch = () => {}, setSelectedSystemPrompt = () => {}, setSelectedQuickPrompt = () => {}, setToolChoice = () => {}, setUseOCR = () => {};
  const stopServerDictation = () => {};
  const sendQueuedRequest = ${callback.getText(ast)};
  return useQueuedRequests({ isConnectionReady, isStreaming: false, queue, setQueue, sendQueuedRequest, stopStreamingRequest: () => {} });
}`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } }
).outputText
const useCompactQueue = new Function(
  "React",
  "useQueuedRequests",
  "throwIfChatSubmitUnsuccessful",
  code
)(React, useQueuedRequests, throwIfChatSubmitUnsuccessful) as (scope: {
  sendMessage: (payload: unknown) => Promise<ChatSubmitResult>
  isConnectionReady?: boolean
  stopListening?: () => Promise<void>
  queue: QueuedRequest[]
  setQueue: (
    value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])
  ) => void
}) => ReturnType<typeof useQueuedRequests>

describe("actual compact queued submission outcome", () => {
  it.each([
    {
      status: "failed",
      errorMessage: "Outcome not confirmed; review recovery"
    },
    { status: "skipped", reason: "Choose an available model" }
  ] as const)(
    "retains $status text instead of treating a resolved promise as completion",
    async (outcome) => {
      const first = buildQueuedRequest({ promptText: "KEEP FIRST" })
      const second = buildQueuedRequest({ promptText: "NEXT" })
      const queue = { current: [first, second] }
      const sendMessage = vi.fn(async () => outcome)
      const setQueue = (
        value:
          | QueuedRequest[]
          | ((previous: QueuedRequest[]) => QueuedRequest[])
      ) => {
        queue.current =
          typeof value === "function" ? value(queue.current) : value
      }
      const hook = renderHook(() =>
        useCompactQueue({ sendMessage, setQueue, queue: queue.current })
      )
      await act(async () => {
        await hook.result.current.flushNext()
      })
      expect(queue.current).toMatchObject([
        {
          id: first.id,
          promptText: "KEEP FIRST",
          status: "blocked",
          blockedReason:
            outcome.status === "failed" ? outcome.errorMessage : outcome.reason
        },
        { id: second.id, promptText: "NEXT", status: "queued" }
      ])
      await act(async () => {
        await hook.result.current.flushNext()
      })
      expect(sendMessage).toHaveBeenCalledOnce()
    }
  )

  it("removes only the acknowledged submitted item", async () => {
    const first = buildQueuedRequest({ promptText: "FIRST" })
    const second = buildQueuedRequest({ promptText: "NEXT" })
    const queue = { current: [first, second] }
    const setQueue = (
      value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])
    ) => {
      queue.current = typeof value === "function" ? value(queue.current) : value
    }
    const hook = renderHook(() =>
      useCompactQueue({
        sendMessage: async () => ({ status: "submitted" }),
        setQueue,
        queue: queue.current
      })
    )
    await act(async () => {
      await hook.result.current.flushNext()
    })
    expect(queue.current.map((item) => item.id)).toEqual([second.id])
  })
})

describe("actual compact queued preparation during a disconnect", () => {
  it.each([false, true])("blocks held dictation preparation even if reconnected=%s", async reconnect => {
    let release!: () => void
    const waiting = new Promise<void>(resolve => { release = resolve })
    const stopListening = vi.fn(() => waiting)
    const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))
    const first = buildQueuedRequest({ promptText: "KEEP PREPARED" })
    let queue = [first]
    const setQueue = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => {
      queue = typeof value === "function" ? value(queue) : value
    }
    const hook = renderHook(({ ready }) => useCompactQueue({ sendMessage, stopListening, setQueue, queue, isConnectionReady: ready }), { initialProps: { ready: true } })
    let pending!: Promise<unknown>
    act(() => { pending = hook.result.current.flushNext() })
    expect(stopListening).toHaveBeenCalledOnce()
    hook.rerender({ ready: false })
    if (reconnect) hook.rerender({ ready: true })
    await act(async () => { release(); await pending })
    expect(sendMessage).not.toHaveBeenCalled()
    expect(queue).toMatchObject([{ promptText: "KEEP PREPARED", status: "blocked", blockedReason: "connection_unavailable" }])
    hook.rerender({ ready: true })
    await act(async () => { await hook.result.current.runNow(first.id); await hook.result.current.flushNext() })
    expect(sendMessage).toHaveBeenCalledOnce()
    expect(queue).toEqual([])
  })
})
