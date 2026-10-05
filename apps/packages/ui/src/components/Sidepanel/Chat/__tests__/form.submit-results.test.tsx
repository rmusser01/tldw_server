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
  const { sendMessage, setQueue, queue } = scope;
  const captureQueuedDispatchGuard = () => () => {};
  const validateQueuedRequest = () => null;
  const form = { setFieldError: () => {} };
  const setSelectedModel = () => {}, setChatMode = () => {}, setWebSearch = () => {}, setSelectedSystemPrompt = () => {}, setSelectedQuickPrompt = () => {}, setToolChoice = () => {}, setUseOCR = () => {};
  const stopListening = async () => {}, stopServerDictation = () => {};
  const sendQueuedRequest = ${callback.getText(ast)};
  return useQueuedRequests({ isConnectionReady: true, isStreaming: false, queue, setQueue, sendQueuedRequest, stopStreamingRequest: () => {} });
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
