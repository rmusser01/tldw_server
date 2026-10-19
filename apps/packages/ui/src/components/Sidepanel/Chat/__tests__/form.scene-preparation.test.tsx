import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
// @vitest-environment jsdom
import { readFileSync } from "node:fs"
import path from "node:path"
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import ts from "typescript"
import { beforeEach, expect, it, vi } from "vitest"
import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { buildChatToolFilterState } from "@/utils/chat-tools"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { captureActorSceneSend } from "@/hooks/chat/actor-scene-source"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { createDefaultActorSettings } from "@/types/actor"
import type { UploadedFile } from "@/db/dexie/types"
import { documentProcessingSelectionKey } from "@/services/chat-document-processing"
import { normalizeChatSubmitResult, isChatSubmitSuccess } from "@/hooks/chat/chat-action-utils"

const sceneIO = vi.hoisted(() => vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>())
vi.mock("@/services/actor-settings", () => ({ getActorSceneForChat: sceneIO }))

type ActorSceneResult = Awaited<ReturnType<typeof sceneIO>>

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(complete => { resolve = complete })
  return { promise, resolve }
}
const ast = ts.createSourceFile("form.tsx", readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let command: ts.FunctionDeclaration | undefined
let preparationEffect: ts.ExpressionStatement | undefined
const visit = (node: ts.Node) => {
  if (ts.isFunctionDeclaration(node) && node.name?.text === "sendCurrentFormMessage") command = node
  if (ts.isExpressionStatement(node) && node.getText(ast).includes("const active = sidepanelDocumentPreparationRef.current")) preparationEffect = node
  ts.forEachChild(node, visit)
}
visit(ast)
if (!command || !preparationEffect) throw new Error("Missing actual compact direct command or preparation effect")
const code = ts.transpileModule(`return function useDirect(scope) {
  const { form, stopListening, sendMessage, greetingHistorySelection, contextFiles = [], setContextFiles = () => {} } = scope;
  const sourceDraftRevision = useStoreMessageOption(state => state.chatDraftRevision), isConnectionReady = true;
  const connectionReadyRef = { current: true }, connectionGenerationRef = { current: 0 }, knowledgeMentionRevision = { current: 0 };
  const sidepanelDocumentPreparationRef = React.useRef(null), currentDocumentSelectionKey = documentProcessingSelectionKey(contextFiles), currentDocumentSelectionKeyRef = React.useRef(currentDocumentSelectionKey);
  currentDocumentSelectionKeyRef.current = currentDocumentSelectionKey;
  ${preparationEffect.getText(ast)}
  const captureCompletionGuard = () => { const captured = useStoreMessageOption.getState(); return () => useStoreMessageOption.getState().chatDraftAddress === captured.chatDraftAddress && useStoreMessageOption.getState().chatDraftRevision === captured.chatDraftRevision; };
  const activeSubmissionRef = React.useRef(null), draftReady = true;
  const contextRevisionRef = React.useRef({ key: "fixture-context", revision: 0 });
  const contextKeyFor = () => "fixture-context";
  const captureDraftOwnership = () => {
    const revision = form.getRevision(), ownerIsCurrent = captureCompletionGuard();
    const isCurrent = () => ownerIsCurrent() && form.getRevision() === revision;
    return { isCurrent, isOwnerCurrent: ownerIsCurrent, normalizeMessage: message => isCurrent() && message === form.getValues().message };
  };
  const resolveSubmissionIntent = message => ({ message, combinedMessage: message, isImageCommand: false });
  const notification = { error: () => {} }, t = (key, fallback) => fallback ?? key;
  const selectedDocuments = [], selectedModel = "fixture-model", availableChatModelIds = new Set(["fixture-model"]), history = [];
  const normalizeChatModelId = id => id, findUnavailableChatModel = () => null;
  const ensureEmbeddingModelAvailable = async () => true;
  const conversationContextComposition = { composeForSend: async () => ({ requestOverrides: {} }) };
  const beginPromptAssistReset = () => { form.reset(); return 1 }, textAreaFocus = () => {};
  const markPromptAssistAttemptSaved = () => {}, clearDraft = () => {}, restorePromptAssistAttempt = () => {};
  const clearSubmittedDocuments = () => {}, setKnowledgeMentionActive = () => {};
  const submitDispatch = async (payload, callbacks) => { callbacks.beforeSend(); const result = await sendMessage(payload); callbacks.afterSend(result); return result; };
  ${command.getText(ast)}
  return sendCurrentFormMessage;
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const useDirect = new Function("React", "useStoreMessageOption", "captureActorSceneSend", "nativeCharacterPreflightMessage", "normalizeChatSubmitResult", "isChatSubmitSuccess", "captureChatDispatchSource", "documentProcessingSelectionKey", "isSavedRoleplayConversation", code)(
  React, useStoreMessageOption, captureActorSceneSend, () => null, normalizeChatSubmitResult, isChatSubmitSuccess, captureChatDispatchSource, documentProcessingSelectionKey, isSavedRoleplayConversation) as (scope: Record<string, unknown>) => (message: string, image: string) => Promise<void>
const requestScope = { config: { serverUrl: "https://compact-outer-scene.example", authMode: "single-user" as const }, userId: 1 }
const controller = { getCurrent: () => ({ status: "idle", owner: null, view: null }), fence: () => () => true,
  draftAccount: { requestScope, isCurrent: () => true } } as unknown as HistorySelectionController
const mount = (stopListening: () => Promise<void>) => {
  const sendMessage = vi.fn(async (_payload: unknown) => ({ status: "submitted" }))
  const view = renderHook(() => {
    const form = useSimpleForm({ initialValues: { message: "  ORIGINAL COMPACT INPUT  ", image: "" } })
    const [files, setFiles] = React.useState<UploadedFile[]>([])
    return { form, files, setFiles, send: useDirect({ form, contextFiles: files, setContextFiles: setFiles, sendMessage, stopListening, greetingHistorySelection: controller }) }
  })
  return { ...view, sendMessage }
}
beforeEach(() => {
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  sceneIO.mockReset().mockImplementation(async ({ target }) => ({ target: target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false }))
})
it("freezes the compact scene before dictation preparation and forwards original ownership", async () => {
  const original = { ...createDefaultActorSettings(), notes: "Original compact outer scene" }
  let stored = original, release!: () => void
  sceneIO.mockImplementation(async ({ target }) => ({ target: target!, settings: structuredClone(stored), originalRecord: { value: structuredClone(stored), updatedAt: 1 }, origin: "scene", profileUnavailable: false }))
  const stopListening = vi.fn(() => new Promise<void>(resolve => { release = resolve }))
  const view = mount(stopListening)
  let pending!: Promise<void>
  act(() => { pending = view.result.current.send(view.result.current.form.getValues().message, "") })
  await waitFor(() => expect(stopListening).toHaveBeenCalledOnce())
  const readsBeforeDictation = sceneIO.mock.calls.length
  stored = { ...original, notes: "Newer compact outer scene" }
  await act(async () => { release(); await pending })
  expect(readsBeforeDictation).toBe(1)
  expect(view.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ actorSceneSend: expect.objectContaining({ requestScope, scene: expect.objectContaining({ settings: original }) }) }))
})
it.each(["New", "edit", "edit then Undo"])("preserves compact input after %s during held dictation preparation", async boundary => {
  let release!: () => void
  const stopListening = vi.fn(() => new Promise<void>(resolve => { release = resolve }))
  const view = mount(stopListening)
  let pending!: Promise<void>
  act(() => { pending = view.result.current.send(view.result.current.form.getValues().message, "") })
  await waitFor(() => expect(stopListening).toHaveBeenCalledOnce())
  act(() => {
    if (boundary === "New") useStoreMessageOption.getState().startNewChatDraft()
    else { view.result.current.form.setFieldValue("message", "NEWER COMPACT INPUT"); if (boundary === "edit then Undo") view.result.current.form.setFieldValue("message", "  ORIGINAL COMPACT INPUT  ") }
  })
  await act(async () => { release(); await pending })
  expect(view.sendMessage).not.toHaveBeenCalled()
  expect(view.result.current.form.getValues().message).toBe(boundary === "edit" ? "NEWER COMPACT INPUT" : "  ORIGINAL COMPACT INPUT  ")
})

it.each(["scene", "dictation"].flatMap(wait => ["model settings", "enabled tools"].map(setting => ({ wait, setting }))))("refuses changed $setting during outer compact $wait preparation", async ({ wait, setting }) => {
  useStoreChatModelSettings.getState().reset();
  useStoreMessageOption.setState({ selectedModel: "fixture-model", toolChoice: "auto" });
  useMcpToolsStore.setState({ healthState: "healthy", chatTools: buildChatToolFilterState({ tools: [{ name: "compact-outer-tool", inputSchema: { type: "object" } }] }).chatTools });
  const scene = deferred<ActorSceneResult>();
  const dictation = deferred<void>();
  const stopListening = vi.fn(() => wait === "dictation" ? dictation.promise : Promise.resolve());
  if (wait === "scene") sceneIO.mockReturnValueOnce(scene.promise);
  const view = mount(stopListening);
  let pending!: Promise<void>;
  act(() => { pending = view.result.current.send(view.result.current.form.getValues().message, ""); });
  await waitFor(() => expect(wait === "scene" ? sceneIO : stopListening).toHaveBeenCalledOnce());
  act(() => {
    if (setting === "model settings") useStoreChatModelSettings.getState().updateSetting("temperature", 0.23);
    else useMcpToolsStore.setState({ chatTools: [] });
  });
  await act(async () => {
    if (wait === "scene") scene.resolve({ target: sceneIO.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false });
    else dictation.resolve(undefined);
    await pending;
  });
  expect(view.sendMessage).not.toHaveBeenCalled();
  expect(view.result.current.form.getValues().message).toBe("  ORIGINAL COMPACT INPUT  ");
});


it.each(["scene", "dictation"])("preserves newly attached files and input after empty-file compact %s preparation is cancelled", async (boundary) => {
  const scene = deferred<ActorSceneResult>()
  const dictation = deferred<void>()
  if (boundary === "scene") sceneIO.mockReturnValueOnce(scene.promise)
  const stopListening = vi.fn(() => boundary === "dictation" ? dictation.promise : Promise.resolve())
  const view = mount(stopListening)
  let pending!: Promise<void>
  act(() => { pending = view.result.current.send(view.result.current.form.getValues().message, "") })
  await waitFor(() => expect(boundary === "scene" ? sceneIO : stopListening).toHaveBeenCalledOnce())
  const newlyAttached: UploadedFile = { id: "compact-new-file", filename: "new.pdf", type: "application/pdf", content: "data:application/pdf;base64,abc", size: 3, uploadedAt: 1, processed: false }
  act(() => { view.result.current.setFiles([newlyAttached]) })
  await act(async () => {
    if (boundary === "scene") scene.resolve({ target: sceneIO.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
    else dictation.resolve(undefined)
    await pending
  })
  expect(view.sendMessage).not.toHaveBeenCalled()
  expect(view.result.current.files).toEqual([newlyAttached])
  expect(view.result.current.form.getValues().message).toBe("  ORIGINAL COMPACT INPUT  ")
  view.unmount()
})

it.each(["scene", "dictation"])("allows compact New to send before the old %s preparation returns", async (boundary) => {
  const scene = deferred<ActorSceneResult>()
  const dictation = deferred<void>()
  if (boundary === "scene") sceneIO.mockReturnValueOnce(scene.promise)
  const stopListening = vi.fn(async () => {}).mockImplementationOnce(() => boundary === "dictation" ? dictation.promise : Promise.resolve())
  const view = mount(stopListening)
  let pending!: Promise<void>
  act(() => { pending = view.result.current.send(view.result.current.form.getValues().message, "") })
  await waitFor(() => expect(boundary === "scene" ? sceneIO : stopListening).toHaveBeenCalledOnce())
  try {
    act(() => {
      useStoreMessageOption.getState().startNewChatDraft()
      view.result.current.form.setFieldValue("message", "NEW COMPACT INPUT")
    })
    await act(async () => { await view.result.current.send("NEW COMPACT INPUT", "") })
    expect(view.sendMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "NEW COMPACT INPUT" }))
  } finally {
    await act(async () => {
      if (boundary === "scene") scene.resolve({ target: sceneIO.mock.calls[0][0].target!, settings: createDefaultActorSettings(), originalRecord: null, origin: "default", profileUnavailable: false })
      else dictation.resolve(undefined)
      await pending
    })
    view.unmount()
  }
  expect(view.sendMessage).toHaveBeenCalledOnce()
})
