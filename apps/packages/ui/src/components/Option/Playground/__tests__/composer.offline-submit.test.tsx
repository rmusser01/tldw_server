import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"
import { usePlaygroundSubmit, type UsePlaygroundSubmitDeps } from "../hooks/usePlaygroundSubmit"
import { useComposerSubmit } from "@/components/Chat/composer/hooks/useComposerSubmit"
import { normalizeChatSubmitResult, isChatSubmitSuccess } from "@/hooks/chat/chat-action-utils"
import { normalizeChatModelId, findUnavailableChatModel } from "@/utils/chat-model-availability"
import { isSavedRoleplayConversation } from "@/hooks/chat/roleplay-resume-source"
import { captureChatDispatchSource } from "@/hooks/chat/chat-dispatch-source"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"

const t = (_key: string, fallback?: string | { defaultValue?: string }) => typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
const makeForm = () => {
  const form = {
    values: { message: "KEEP RAW  ", image: "KEEP IMAGE" },
    setFieldError: vi.fn(), setFieldValue: vi.fn(), reset: vi.fn(), getRevision: () => 0,
    onSubmit: (callback: (values: typeof form.values) => Promise<void>) => () => { pending = callback(form.values) }
  }
  let pending = Promise.resolve()
  return { form, wait: () => pending }
}
const makeWebDeps = (ready: boolean, busy: boolean, imageCommand = false) => {
  const test = makeForm()
  const deps: UsePlaygroundSubmitDeps = {
    form: test.form, beginPromptAssistReset: vi.fn(() => 1), restorePromptAssistAttempt: vi.fn(), markPromptAssistAttemptSaved: vi.fn(),
    isSending: busy, isConnectionReady: ready, draftReady: true, webSearch: false, compareModeActive: false, compareSelectedModels: [], selectedModel: "gpt-test",
    fileRetrievalEnabled: false, ragPinnedResults: [], selectedDocuments: [{ id: "keep-tab" }], uploadedFiles: [], currentContextSnapshot: {},
    conversationTokenCount: 0, characterContextTokenEstimate: 0, pinnedSourceTokenEstimate: 0, resolvedMaxContext: 8000, jsonMode: false,
    openUIRequestMode: false, sendMessage: vi.fn(async () => ({ status: "submitted" })), clearOpenUIRequestMode: vi.fn(), clearSelectedDocuments: vi.fn(), clearUploadedFiles: vi.fn(),
    textAreaFocus: vi.fn(), setLastSubmittedContext: vi.fn(), estimateTokensForText: () => 1,
    resolveSubmissionIntent: vi.fn(() => ({ message: "NORMALIZED", combinedMessage: "NORMALIZED", handled: imageCommand, isImageCommand: imageCommand, imageBackendOverride: "test" })),
    queueSubmission: vi.fn(() => ({ id: "queued" })), validateSelectedChatModelsAvailability: () => true, compareModelsSupportCapability: () => true,
    notificationApi: { info: vi.fn(), error: vi.fn(), warning: vi.fn() }, t
  }
  return { ...test, deps }
}

const ast = ts.createSourceFile("form.tsx", readFileSync(path.resolve(__dirname, "../../../Sidepanel/Chat/form.tsx"), "utf8"), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let callback: ts.Expression | undefined
const visit = (node: ts.Node) => {
  if (ts.isVariableDeclaration(node) && node.name.getText(ast) === "submitCurrentRequest") callback = node.initializer
  ts.forEachChild(node, visit)
}
visit(ast)
if (!callback) throw new Error("Missing actual compact submit callback")
const source = ts.transpileModule(`const {
  isConnectionReady, connectionReadyRef, connectionGenerationRef, isSending, draftReady, greetingStarting, form, notification, resolveSubmissionIntent, selectedDocuments,
  contextFiles, selectedModel, availableChatModelIds, queueSubmission, stopListening, stopServerDictation, sendCurrentFormMessageRef, t,
  greetingHistorySelection, isSavedRoleplayConversation
} = scope;
return ${callback.getText(ast)};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const compactSubmit = new Function("React", "normalizeChatModelId", "findUnavailableChatModel", "scope", source)
const makeCompact = (ready: boolean, busy: boolean, imageCommand: boolean) => {
  const web = makeWebDeps(ready, busy, imageCommand)
  const scope = {
    ...web.deps, isSavedRoleplayConversation, greetingHistorySelection: {getCurrent: () => ({owner:null})} as unknown as Pick<HistorySelectionController, "getCurrent">, greetingStarting: false, connectionReadyRef: { current: ready }, connectionGenerationRef: { current: 0 }, notification: web.deps.notificationApi, contextFiles: [],
    availableChatModelIds: new Set(["gpt-test"]), stopListening: vi.fn(async () => {}), stopServerDictation: vi.fn(), sendCurrentFormMessageRef: { current: web.deps.sendMessage }
  }
  return { ...web, scope, submit: compactSubmit({ useCallback: (fn: unknown) => fn }, normalizeChatModelId, findUnavailableChatModel, scope) as (message: string, image: string) => Promise<void> }
}

describe("unavailable composer submission retains the source", () => {
  it.each([[false, false], [true, false], [false, true], [true, true]])("full-page offline busy=%s imageCommand=%s does not normalize, queue, reset or dispatch", async (busy, imageCommand) => {
    const test = makeWebDeps(false, busy, imageCommand)
    const hook = renderHook(() => usePlaygroundSubmit(test.deps))
    await act(async () => { hook.result.current.submitForm(); await test.wait() })
    expect(test.deps.resolveSubmissionIntent).not.toHaveBeenCalled()
    expect(test.deps.queueSubmission).not.toHaveBeenCalled()
    expect(test.deps.sendMessage).not.toHaveBeenCalled()
    expect(test.form.reset).not.toHaveBeenCalled()
    expect(test.deps.beginPromptAssistReset).not.toHaveBeenCalled()
    expect(test.deps.clearSelectedDocuments).not.toHaveBeenCalled()
    expect(test.form.values).toEqual({ message: "KEEP RAW  ", image: "KEEP IMAGE" })
    expect(test.deps.notificationApi.info).toHaveBeenCalled()
  })
  it.each([[false, false], [true, false], [false, true], [true, true]])("actual compact callback offline busy=%s imageCommand=%s leaves the draft untouched", async (busy, imageCommand) => {
    const test = makeCompact(false, busy, imageCommand)
    await test.submit(test.form.values.message, test.form.values.image)
    expect(test.deps.resolveSubmissionIntent).not.toHaveBeenCalled()
    expect(test.deps.queueSubmission).not.toHaveBeenCalled()
    expect(test.deps.sendMessage).not.toHaveBeenCalled()
    expect(test.form.reset).not.toHaveBeenCalled()
    expect(test.deps.notificationApi.info).toHaveBeenCalled()
  })
  it("full-page connected busy submission still queues", async () => {
    const test = makeWebDeps(true, true)
    const hook = renderHook(() => usePlaygroundSubmit(test.deps))
    await act(async () => { hook.result.current.submitForm(); await test.wait() })
    expect(test.deps.queueSubmission).toHaveBeenCalledOnce()
    expect(test.deps.sendMessage).not.toHaveBeenCalled()
  })
  it("actual compact connected busy submission still queues", async () => {
    const test = makeCompact(true, true, false)
    await test.submit(test.form.values.message, test.form.values.image)
    expect(test.deps.queueSubmission).toHaveBeenCalledOnce()
    expect(test.deps.sendMessage).not.toHaveBeenCalled()
  })
})

let directHelper: ts.FunctionDeclaration | undefined
const findHelper = (node: ts.Node) => {
  if (ts.isFunctionDeclaration(node) && node.name?.text === "sendCurrentFormMessage") directHelper = node
  ts.forEachChild(node, findHelper)
}
findHelper(ast)
if (!directHelper) throw new Error("Missing actual compact preparation helper")
const helperSource = ts.transpileModule(`const {
  connectionReadyRef, connectionGenerationRef, sidepanelDocumentPreparationRef, resolveSubmissionIntent, notification, t, selectedDocuments, contextFiles,
  stopListening, selectedModel, availableChatModelIds, form, ensureEmbeddingModelAvailable, conversationContextComposition, history,
  submitDispatch, beginPromptAssistReset, textAreaFocus, markPromptAssistAttemptSaved, clearDraft, clearSelectedDocuments, setContextFiles,
  setKnowledgeMentionActive, restorePromptAssistAttempt, knowledgeMentionRevision, clearSubmittedDocuments, captureCompletionGuard,
  currentDocumentSelectionKeyRef, greetingHistorySelection, isSavedRoleplayConversation, captureChatDispatchSource, captureActorSceneSend,
  nativeCharacterPreflightMessage, chatMode, webSearch, selectedSystemPrompt, toolChoice, pageContextActive
} = scope;
${directHelper.getText(ast)}
return sendCurrentFormMessage;`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
const compactDirect = new Function("normalizeChatModelId", "findUnavailableChatModel", "normalizeChatSubmitResult", "isChatSubmitSuccess", "scope", helperSource)

it.each([false, true])("actual compact context preparation does not clear or send after disconnect, reconnected=%s", async reconnect => {
  let release!: () => void
  const waiting = new Promise<void>(resolve => { release = resolve })
  const test = makeWebDeps(true, false)
  const dispatcher = renderHook(() => useComposerSubmit({ sendMessage: test.deps.sendMessage }))
  const scope = {
    ...test.deps, notification: test.deps.notificationApi, contextFiles: [], history: [], availableChatModelIds: new Set(["gpt-test"]),
    connectionReadyRef: { current: true }, connectionGenerationRef: { current: 0 }, sidepanelDocumentPreparationRef: { current: null },
    stopListening: vi.fn(async () => {}), ensureEmbeddingModelAvailable: vi.fn(async () => true),
    conversationContextComposition: { composeForSend: vi.fn(() => waiting) }, submitDispatch: dispatcher.result.current.dispatch,
    currentDocumentSelectionKeyRef: {current:""}, greetingHistorySelection: {getCurrent:()=>({owner:null})},
    isSavedRoleplayConversation, captureChatDispatchSource, captureActorSceneSend: () => ({isCurrent:()=>true, read:async()=>({isCurrent:()=>true})}),
    nativeCharacterPreflightMessage: (error:unknown)=>String(error), chatMode:"normal", webSearch:false, selectedSystemPrompt:"", toolChoice:"auto", pageContextActive:false,
    clearDraft: vi.fn(), setContextFiles: vi.fn(), setKnowledgeMentionActive: vi.fn(), knowledgeMentionRevision: {current:0}, clearSubmittedDocuments:vi.fn(),captureCompletionGuard:()=>()=>true
  }
  const send = compactDirect(normalizeChatModelId, findUnavailableChatModel, normalizeChatSubmitResult, isChatSubmitSuccess, scope) as (text: string, image: string) => Promise<void>
  const pending = send(test.form.values.message, test.form.values.image)
  await act(async () => {})
  expect(scope.conversationContextComposition.composeForSend).toHaveBeenCalledOnce()
  scope.connectionReadyRef.current = false
  scope.connectionGenerationRef.current += 1
  if (reconnect) scope.connectionReadyRef.current = true
  await act(async () => { release(); await pending })
  expect(test.deps.sendMessage).not.toHaveBeenCalled()
  expect(test.deps.beginPromptAssistReset).not.toHaveBeenCalled()
  expect(test.deps.clearSelectedDocuments).not.toHaveBeenCalled()
  expect(scope.setContextFiles).not.toHaveBeenCalled()
  expect(test.form.values).toEqual({ message: "KEEP RAW  ", image: "KEEP IMAGE" })
})

it("retired full-page preparation preserves input before optimistic reset", async () => {
  const test = makeWebDeps(true, false);
  test.deps.captureCompletionGuard = () => () => false;
  const view = renderHook(() => usePlaygroundSubmit(test.deps));
  let error: unknown;
  await act(async () => {
    view.result.current.submitForm();
    try {
      await test.wait();
    } catch (caught) {
      error = caught;
    }
  });
  expect(test.form.reset).not.toHaveBeenCalled();
  expect(test.deps.beginPromptAssistReset).not.toHaveBeenCalled();
  expect(test.deps.clearSelectedDocuments).not.toHaveBeenCalled();
  expect(test.deps.sendMessage).not.toHaveBeenCalled();
  expect(error).toBeUndefined();
});


it("actual Compact saved Roleplay queue preserves literal text with no live model", async () => {
  const test = makeCompact(true, true, false)
  const conversation_id = "saved-compact-queue"
  useStoreMessageOption.setState({
    serverChatId: conversation_id,
    serverChatAssistantKind: "character",
    temporaryChat: false
  })
  test.scope.greetingHistorySelection = {
    getCurrent: () => ({ owner: { kind: "native", conversation_id } })
  } as unknown as Pick<HistorySelectionController, "getCurrent">
  test.scope.selectedModel = ""
  test.scope.selectedDocuments = []
  test.scope.resolveSubmissionIntent.mockReturnValue({
    message: "  Exact literal\n",
    combinedMessage: "  Exact literal\n",
    handled: false,
    isImageCommand: false,
    imageBackendOverride: "test"
  })
  const submit = compactSubmit(
    { useCallback: (fn: unknown) => fn },
    normalizeChatModelId,
    findUnavailableChatModel,
    test.scope
  ) as (message: string, image: string) => Promise<void>
  await submit("  Exact literal\n", "")
  expect(test.deps.queueSubmission).toHaveBeenCalledWith(
    expect.objectContaining({ promptText: "  Exact literal\n" })
  )
})

it("actual Compact saved Roleplay preparation uses literal input without reading live Scene, settings or embedding context", async () => {
  const test = makeWebDeps(true, false)
  test.deps.resolveSubmissionIntent.mockReturnValue({
    message: "  Exact literal\n",
    combinedMessage: "  Exact literal\n",
    handled: false,
    isImageCommand: false,
    imageBackendOverride: "test"
  })
  const conversation_id = "saved-compact-direct"
  useStoreMessageOption.setState({
    serverChatId: conversation_id,
    serverChatAssistantKind: "character",
    temporaryChat: false
  })
  const dispatcher = renderHook(() =>
    useComposerSubmit({ sendMessage: test.deps.sendMessage })
  )
  const scope = {
    ...test.deps,
    notification: test.deps.notificationApi,
    selectedDocuments: [],
    contextFiles: [],
    history: [],
    connectionReadyRef: { current: true },
    connectionGenerationRef: { current: 0 },
    sidepanelDocumentPreparationRef: { current: null },
    currentDocumentSelectionKeyRef: { current: "" },
    knowledgeMentionRevision: { current: 0 },
    form: test.form,
    greetingHistorySelection: {
      getCurrent: () => ({ owner: { kind: "native", conversation_id } })
    },
    isSavedRoleplayConversation,
    captureChatDispatchSource,
    captureActorSceneSend: vi.fn(() => {
      throw new Error("live Scene forbidden")
    }),
    nativeCharacterPreflightMessage: (error: unknown) => String(error),
    selectedModel: "",
    availableChatModelIds: new Set<string>(),
    chatMode: "normal",
    webSearch: false,
    selectedSystemPrompt: "",
    toolChoice: "auto",
    pageContextActive: false,
    stopListening: vi.fn(async () => {
      useStoreChatModelSettings.setState({ temperature: 1.9 })
    }),
    ensureEmbeddingModelAvailable: vi.fn(async () => true),
    conversationContextComposition: { composeForSend: vi.fn(async () => null) },
    captureCompletionGuard: () => () => true,
    submitDispatch: dispatcher.result.current.dispatch,
    clearDraft: vi.fn(),
    setContextFiles: vi.fn(),
    setKnowledgeMentionActive: vi.fn(),
    clearSubmittedDocuments: vi.fn()
  }
  const send = compactDirect(
    normalizeChatModelId,
    findUnavailableChatModel,
    normalizeChatSubmitResult,
    isChatSubmitSuccess,
    scope
  ) as (text: string, image: string) => Promise<void>
  await act(async () => {
    await send("  Exact literal\n", "")
  })
  expect(test.deps.sendMessage).toHaveBeenCalledWith(
    expect.objectContaining({ message: "  Exact literal\n" })
  )
  expect(scope.captureActorSceneSend).not.toHaveBeenCalled()
  expect(scope.ensureEmbeddingModelAvailable).not.toHaveBeenCalled()
  expect(
    scope.conversationContextComposition.composeForSend
  ).not.toHaveBeenCalled()
})


it.each(["streaming", "isProcessing"])(
  "actual Compact queues behind shared %s activity even when its own mutation is idle",
  async (activity) => {
    let declaration: ts.VariableDeclaration | undefined
    const find = (node: ts.Node) => {
      if (
        ts.isVariableDeclaration(node) &&
        ts.isIdentifier(node.name) &&
        node.name.text === "isSending"
      )
        declaration = node
      ts.forEachChild(node, find)
    }
    find(ast)
    const scope = {
      isMutationPending: false,
      streaming: activity === "streaming",
      isProcessing: activity === "isProcessing"
    }
    const busy = declaration?.initializer
      ? new Function(
          "scope",
          `const {isMutationPending, streaming, isProcessing} = scope; return ${declaration.initializer.getText(ast)};`
        )(scope)
      : false
    const test = makeCompact(true, busy, false)
    await test.submit(test.form.values.message, test.form.values.image)
    expect(test.deps.queueSubmission).toHaveBeenCalledOnce()
    expect(test.deps.sendMessage).not.toHaveBeenCalled()
  }
)
