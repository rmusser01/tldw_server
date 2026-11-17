import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import ts from "typescript"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  throwIfChatSubmitUnsuccessful,
  type ChatSubmitResult
} from "@/hooks/chat/chat-action-utils"
import { useQueuedRequests } from "@/hooks/chat/useQueuedRequests"
import {
  buildQueuedRequest,
  type QueuedRequest
} from "@/utils/chat-request-queue"

import { captureActorSceneSend } from "@/hooks/chat/actor-scene-source"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"
import { createDefaultActorSettings } from "@/types/actor"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
const sceneIO = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat: sceneIO }))

type ActorSceneResult = Awaited<ReturnType<typeof sceneIO>>

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(complete => { resolve = complete })
  return { promise, resolve }
}
const requestScope = { config: { serverUrl: "https://compact-queue-scene.example", authMode: "single-user" as const }, userId: 1 }
const greetingHistorySelection = { getCurrent: () => ({ status: "idle", owner: null, view: null }), fence: () => () => true,
  draftAccount: { requestScope, isCurrent: () => true } } as unknown as HistorySelectionController
beforeEach(() => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ healthState: "healthy", chatTools: [] })
  sceneIO.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
})

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
  const { sendMessage, setQueue, queue, isConnectionReady = true, stopListening = async () => {}, setters = {} } = scope;
  const captureQueuedDispatchGuard = () => { const source = captureChatDraftSource(); const guard = () => { if (!source.isCurrent()) throw new Error("chat_source_changed"); }; guard.isSourceCurrent = source.isCurrent; return guard; };
  const validateQueuedRequest = () => null, chatModels = [], t = (key, fallback) => fallback ?? key;
  const form = { setFieldError: () => {} };
  const { setSelectedModel = () => {}, setChatMode = () => {}, setWebSearch = () => {}, setSelectedSystemPrompt = () => {}, setToolChoice = () => {}, setUseOCR = () => {} } = setters;
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
  "captureActorSceneSend",
  "captureChatDraftSource",
  "greetingHistorySelection",
  "captureChatDispatchSource",
  "useStoreChatModelSettings",
  "resolveModelSelectorSelection",
  "isSavedRoleplayConversation",
  code
)(React, useQueuedRequests, throwIfChatSubmitUnsuccessful, captureActorSceneSend, captureChatDraftSource, greetingHistorySelection, captureChatDispatchSource, useStoreChatModelSettings, resolveModelSelectorSelection, isSavedRoleplayConversation) as (scope: {
  sendMessage: (payload: unknown) => Promise<ChatSubmitResult>
  isConnectionReady?: boolean
  stopListening?: () => Promise<void>
  setters?: Record<string, unknown>
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
    await waitFor(() => expect(stopListening).toHaveBeenCalledOnce())
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

it("captures queued scene ownership before dictation and keeps it out of persisted overrides", async () => {
  const original = { ...createDefaultActorSettings(), notes: "ORIGINAL QUEUED SCENE" }
  let stored = original, release!: () => void
  sceneIO.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored), originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
  const stopListening = vi.fn(() => new Promise<void>(resolve => { release = resolve }))
  const sendMessage = vi.fn(async (_payload: unknown) => ({ status: "submitted" as const }))
  let queue = [buildQueuedRequest({ promptText: "QUEUED INPUT" })]
  const setQueue = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => { queue = typeof value === "function" ? value(queue) : value }
  const hook = renderHook(() => useCompactQueue({ sendMessage, stopListening, setQueue, queue }))
  let pending!: Promise<unknown>
  act(() => { pending = hook.result.current.flushNext() })
  await waitFor(() => expect(stopListening).toHaveBeenCalledOnce())
  const readsBeforeDictation = sceneIO.mock.calls.length
  stored = { ...original, notes: "REPLACEMENT SCENE" }
  await act(async () => { release(); await pending })
  expect(readsBeforeDictation).toBe(1)
  expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope, scene: expect.objectContaining({ settings: original }) }) }))
  const payload = sendMessage.mock.calls[0][0] as { requestOverrides: Record<string, unknown> }
  expect(payload.requestOverrides).not.toHaveProperty("actorSceneSend")
})

it("does not publish queued compact settings after the model setter synchronously opens New", async () => {
  const first = buildQueuedRequest({ promptText: "OLD COMPACT QUEUE", snapshot: { selectedModel: "original-queued-model", chatMode: "normal", webSearch: false, selectedSystemPrompt: "OLD SYSTEM", toolChoice: "auto", useOCR: false } });
  const queue = { current: [first] };
  const core = useStoreMessageOption.getState();
  const sendMessage = vi.fn(async () => ({ status: "submitted" as const }));
  const stopListening = vi.fn(async () => {});
  const setQueue = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => { queue.current = typeof value === "function" ? value(queue.current) : value; };
  const view = renderHook(() => useCompactQueue({ queue: queue.current, setQueue, sendMessage, stopListening, setters: {
    setSelectedModel: core.setSelectedModel, setChatMode: core.setChatMode, setWebSearch: core.setWebSearch, setSelectedSystemPrompt: core.setSelectedSystemPrompt, setToolChoice: core.setToolChoice, setUseOCR: core.setUseOCR } }));
  const replacement = { historyId: "replacement-compact-queue", serverChatId: null, selectedModel: "replacement-model", chatMode: "rag", webSearch: true, selectedSystemPrompt: "REPLACEMENT SYSTEM PROMPT", toolChoice: "none" as const, useOCR: true };
  const unsubscribe = useStoreMessageOption.subscribe(state => {
    if (state.selectedModel !== "original-queued-model") return;
    unsubscribe(); state.startNewChatDraft(); useStoreMessageOption.setState(replacement);
  });
  try {
    await act(async () => { await view.result.current.flushNext(); });
    expect(useStoreMessageOption.getState()).toMatchObject(replacement);
    expect(stopListening).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  } finally { unsubscribe(); }
});


it.each(["scene", "dictation"].flatMap(boundary => ["model settings", "model scope", "enabled tools"].map(setting => ({ boundary, setting }))))("keeps queued compact input when $setting changes during $boundary preparation", async ({ boundary, setting }) => {
  useStoreMessageOption.setState({ selectedModel: "ambient-model", toolChoice: "none" })
  useMcpToolsStore.setState({ chatTools: buildChatToolFilterState({ tools: [{ name: "queued-compact-tool", inputSchema: { type: "object" } }] }).chatTools })
  const scene = deferred<ActorSceneResult>()
  const dictation = deferred<void>()
  if (boundary === "scene") sceneIO.mockReturnValueOnce(scene.promise)
  const stopListening = vi.fn(() => boundary === "dictation" ? dictation.promise : Promise.resolve())
  const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))
  const first = buildQueuedRequest({ promptText: "KEEP COMPACT QUEUED INPUT", snapshot: { selectedModel: "queued-model", toolChoice: "auto" } })
  let queue = [first]
  const setQueue = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => { queue = typeof value === "function" ? value(queue) : value }
  const view = renderHook(() => useCompactQueue({ sendMessage, stopListening, setQueue, queue }))
  let pending!: Promise<unknown>
  act(() => { pending = view.result.current.flushNext() })
  await waitFor(() => expect(boundary === "scene" ? sceneIO : stopListening).toHaveBeenCalledOnce())
  act(() => {
    if (setting === "model settings") useStoreChatModelSettings.getState().updateSetting("temperature", 0.17)
    else if (setting === "model scope") useStoreChatModelSettings.getState().setActiveSettingsScope("external:scope")
    else useMcpToolsStore.setState({ chatTools: [] })
  })
  await act(async () => {
    if (boundary === "scene") scene.resolve({ target: sceneIO.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
    else dictation.resolve(undefined)
    await pending
  })
  expect(sendMessage).not.toHaveBeenCalled()
  expect(queue).toMatchObject([{ id: first.id, promptText: "KEEP COMPACT QUEUED INPUT", status: "blocked", blockedReason: expect.any(String) }])
  if (setting === "model settings") expect(useStoreChatModelSettings.getState().getEffectiveSettings().temperature).toBe(0.17)
  else if (setting === "model scope") expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("external:scope")
  else expect(useMcpToolsStore.getState().chatTools).toEqual([])
  view.unmount()
})

it("captures the compact queue's intended model scope before its first preparation wait", async () => {
  useStoreMessageOption.setState({ selectedModel: "openai:ambient-model", toolChoice: "none" })
  useStoreChatModelSettings.getState().setActiveSettingsScope("openai:ambient-model")
  useStoreChatModelSettings.getState().updateScopedSetting("openai:gpt-4o-mini", "temperature", 0.41)
  useMcpToolsStore.setState({ chatTools: buildChatToolFilterState({ tools: [{ name: "queued-compact-tool", inputSchema: { type: "object" } }] }).chatTools })
  let release!: (value: ActorSceneResult) => void
  const waiting = new Promise<ActorSceneResult>(resolve => { release = resolve })
  sceneIO.mockReturnValueOnce(waiting)
  const core = useStoreMessageOption.getState()
  const sendMessage = vi.fn(async () => ({ status: "submitted" as const }))
  let queue = [buildQueuedRequest({ promptText: "INTENDED COMPACT QUEUE", snapshot: { selectedModel: "openai:gpt-4o-mini", toolChoice: "auto" } })]
  const setQueue = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => { queue = typeof value === "function" ? value(queue) : value }
  const view = renderHook(() => useCompactQueue({ sendMessage, setQueue, queue, setters: { setSelectedModel: core.setSelectedModel, setToolChoice: core.setToolChoice } }))
  let pending!: Promise<unknown>
  act(() => { pending = view.result.current.flushNext() })
  await waitFor(() => expect(sceneIO).toHaveBeenCalledOnce())
  try {
    expect(useStoreChatModelSettings.getState().activeSettingsScope).toBe("openai:gpt-4o-mini")
    expect(useStoreChatModelSettings.getState().getEffectiveSettings().temperature).toBe(0.41)
    act(() => { useStoreChatModelSettings.getState().setActiveSettingsScope("openai:gpt-4o-mini") })
  } finally {
    await act(async () => {
      release({ target: sceneIO.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
      await pending
    })
    view.unmount()
  }
  expect(sendMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "INTENDED COMPACT QUEUE", requestOverrides: expect.objectContaining({ selectedModel: "openai:gpt-4o-mini", toolChoice: "auto" }) }))
  expect(queue).toEqual([])
})
