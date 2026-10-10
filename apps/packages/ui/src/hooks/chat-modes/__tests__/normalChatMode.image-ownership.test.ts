// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { ActorSceneSend } from "@/hooks/chat/actor-scene-source"

// Retaining table IO only; mode, pipeline, scene read and scoped persistence are real.
const storage = vi.hoisted(() => {
  type Row = Record<string, unknown>
  const tables: Record<
    string,
    {
      rows: Map<string, Row>
      get: (id: unknown) => Promise<Row | undefined>
      add: (row: Row) => Promise<void>
      put: (row: Row) => Promise<void>
      update: (id: unknown, patch: Row) => Promise<void>
      toArray: () => Promise<Row[]>
    }
  > = {}
  for (const name of [
    "chatHistories",
    "messages",
    "modelNickname",
    "userSettings",
    "sessionFiles",
    "prompts",
    "compareStates"
  ]) {
    const rows = new Map<string, Row>()
    const key = (row: Row) =>
      JSON.stringify(row.id ?? row.sessionId ?? row.history_id)
    tables[name] = {
      rows,
      get: async (id) => structuredClone(rows.get(JSON.stringify(id))),
      add: async (row) => {
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
      },
      put: async (row) => {
        rows.set(key(row), structuredClone(row))
      },
      update: async (id, patch) => {
        const idKey = JSON.stringify(id)
        rows.set(idKey, { ...rows.get(idKey), ...structuredClone(patch) })
      },
      toArray: async () => structuredClone([...rows.values()])
    }
  }
  let tail = Promise.resolve()
  return {
    tables,
    db: {
      ...tables,
      transaction: (
        _mode: unknown,
        _tables: unknown,
        operation: (tx: { abort: () => void }) => Promise<unknown>
      ) => {
        const result = tail.then(async () => {
          const originals = Object.values(tables).map(
            (table) => new Map(table.rows)
          )
          try {
            return await operation({
              abort: () => {
                throw new Error("aborted")
              }
            })
          } catch (error) {
            Object.values(tables).forEach((table, index) => {
              table.rows.clear()
              originals[index].forEach((row, key) => table.rows.set(key, row))
            })
            throw error
          }
        })
        tail = result.then(
          () => {},
          () => {}
        )
        return result
      }
    }
  }
})
vi.mock("@/db/dexie/schema", () => ({ db: storage.db }))

import { normalChatMode } from "../normalChatMode"
import { captureActorSceneSendForTarget } from "@/hooks/chat/actor-scene-source"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { saveMessageOnSuccess, saveMessageOnError } from "@/hooks/chat-helper"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import * as imageGeneration from "@/services/image-generation"
import * as models from "@/models"
import { setTitleGenEnabled } from "@/services/title"
import {
  IMAGE_GENERATION_USER_MESSAGE_TYPE,
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
} from "@/utils/image-generation-chat"

const originalScope: ServicePromptSnapshot["requestScope"] = {
  config: { serverUrl: "https://images.test", authMode: "multi-user" },
  userId: "alice"
}
const backendConfigs: imageGeneration.ImageBackendConfigMap = {
  fluxklein: {
    format: "png",
    width: 512,
    height: 384,
    steps: 4,
    seed: 7,
    model: "original-image-model"
  }
}
const imageResponse: Awaited<
  ReturnType<typeof tldwClient.createImageArtifact>
> = {
  artifact: {
    file_id: 77,
    file_type: "image",
    title: "Original image",
    export: {
      content_b64: "aW1hZ2U=",
      content_type: "image/png",
      bytes: 5,
      format: "png"
    }
  }
}
const initialCore = useStoreMessageOption.getState()
const initialSession = usePlaygroundSessionStore.getState()
const initialNextData = Object.getOwnPropertyDescriptor(window, "__NEXT_DATA__")
const releases: Array<() => void> = []
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((accept) => {
    resolve = accept
  })
  return { promise, resolve }
}

beforeEach(async () => {
  vi.restoreAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  Object.values(storage.tables).forEach((table) => table.rows.clear())
  usePlaygroundSessionStore.setState(initialSession)
  useStoreMessageOption.setState({
    ...initialCore,
    messages: [],
    history: [],
    historyId: null,
    serverChatId: null,
    isProcessing: false,
    streaming: false,
    selectedModel: "fluxklein",
    toolChoice: "none"
  })
  useStoreMessageOption.getState().startNewChatDraft(false)
  // The WebUI route owns its title; asynchronous persistence uses the real Next guard.
  Object.defineProperty(window, "__NEXT_DATA__", {
    configurable: true,
    value: {}
  })
  document.title = "Image Chat"
  await setTitleGenEnabled(false)
})

afterEach(() => {
  releases.splice(0).forEach((release) => release())
  vi.restoreAllMocks()
  useStoreMessageOption.setState(initialCore)
  usePlaygroundSessionStore.setState(initialSession)
  if (initialNextData)
    Object.defineProperty(window, "__NEXT_DATA__", initialNextData)
  else Reflect.deleteProperty(window, "__NEXT_DATA__")
})

const fixture = async () => {
  const stop = new AbortController()
  const accountLifetime = new AbortController()
  const execution = new AbortController()
  const abortExecution = () => execution.abort()
  stop.signal.addEventListener("abort", abortExecution)
  accountLifetime.signal.addEventListener("abort", abortExecution)
  const stopWatching = watchChatAccountChanges((invalidated) => {
    if (invalidated) accountLifetime.abort()
  })
  releases.push(() => {
    stopWatching()
    stop.signal.removeEventListener("abort", abortExecution)
    accountLifetime.signal.removeEventListener("abort", abortExecution)
    accountLifetime.abort()
  })
  const account = Object.freeze({
    requestScope: structuredClone(originalScope),
    isCurrent: () => !accountLifetime.signal.aborted
  })
  const source = captureChatDraftSource(account.isCurrent)
  const address = useStoreMessageOption.getState().chatDraftAddress
  if (address?.kind !== "fresh" || address.temporary)
    throw new Error("fixture requires the original non-Temporary fresh address")
  const target = captureActorSettingsTarget({
    address,
    requestScope: account.requestScope,
    accountIsCurrent: account.isCurrent,
    isCurrent: source.isCurrent
  })
  const capture = captureActorSceneSendForTarget(
    target,
    target?.account,
    source.isCurrent
  )
  if (!capture)
    throw new Error("fixture requires the original qualified fresh scene")
  const actorSceneSend: ActorSceneSend = await capture.read()
  const snapshot: ServicePromptSnapshot = {
    scopeKey: "original-image-account",
    requestScope: structuredClone(account.requestScope),
    capability: "supported",
    definitions: {},
    scopeSignal: execution.signal,
    scopeInvalidatedSignal: accountLifetime.signal,
    release: () => {}
  }
  const configIO = vi
    .spyOn(imageGeneration, "getImageBackendConfigs")
    .mockResolvedValue(backendConfigs)
  const initializeIO = vi.spyOn(tldwClient, "initialize").mockResolvedValue()
  const imageIO = vi
    .spyOn(tldwClient, "createImageArtifact")
    .mockResolvedValue(imageResponse)
  const modelFactory = vi.spyOn(models, "pageAssistModel")
  const modelProvider = vi.spyOn(tldwClient, "streamChatCompletion")
  const saveSuccess = vi.fn(saveMessageOnSuccess)
  const saveError = vi.fn(saveMessageOnError)
  let abortOwner: AbortController | null = stop
  // Writes still execute through the core setters, under the original source fence.
  const params: Parameters<typeof normalChatMode>[6] = {
    selectedModel: "fluxklein",
    useOCR: false,
    selectedSystemPrompt: "",
    currentChatModelSettings: {},
    toolChoice: "none",
    imageBackendOverride: "fluxklein",
    imageGenerationRequest: { prompt: "original image request" },
    userMessageType: IMAGE_GENERATION_USER_MESSAGE_TYPE,
    assistantMessageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
    userMessageId: "image-user",
    assistantMessageId: "image-assistant",
    actorSceneSend,
    actorScene: actorSceneSend.scene,
    actorSettings: actorSceneSend.scene.settings,
    servicePromptSnapshot: snapshot,
    historyId: null,
    serverChatId: null,
    saveMessageOnSuccess: saveSuccess,
    saveMessageOnError: saveError,
    setMessages: (next) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setMessages(next)
    },
    setHistory: (next) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setHistory(next)
    },
    setHistoryId: (id) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setHistoryId(id)
    },
    setIsProcessing: (value) => {
      if (source.isCurrent())
        useStoreMessageOption.getState().setIsProcessing(value)
    },
    setStreaming: (value) => {
      if (source.isCurrent())
        useStoreMessageOption.getState().setStreaming(value)
    },
    setAbortController: (next) => {
      if (source.isCurrent()) abortOwner = next
    },
    ownsAbortController: (signal) =>
      abortOwner?.signal === signal && source.isCurrent(),
    releaseAbortControllerIfOwned: (signal) => {
      if (abortOwner?.signal !== signal) return false
      abortOwner = null
      return source.isCurrent()
    }
  }
  const replaceView = (retirement: "New" | "account") => {
    if (retirement === "New") {
      usePlaygroundSessionStore.getState().markSourceSelectionIntent()
      useStoreMessageOption.getState().startNewChatDraft(false)
    } else {
      window.dispatchEvent(new Event("tldw:auth-principal-changed"))
      expect(accountLifetime.signal.aborted).toBe(true)
    }
    useStoreMessageOption.setState({
      messages: [],
      history: [],
      historyId: null,
      serverChatId: null,
      streaming: false,
      isProcessing: false
    })
    document.title = retirement === "New" ? "New image draft" : "New account"
    return useStoreMessageOption.getState()
  }
  return {
    actorSceneSend,
    address,
    account,
    configIO,
    initializeIO,
    imageIO,
    modelFactory,
    modelProvider,
    saveSuccess,
    saveError,
    replaceView,
    send: () =>
      normalChatMode(
        "original image request",
        "",
        false,
        [],
        [],
        stop.signal,
        params
      )
  }
}

describe("normal image preflight original ownership", () => {
  it.each([
    ["configs", "New"],
    ["initialize", "New"],
    ["configs", "account"],
    ["initialize", "account"]
  ] as const)(
    "does not create an image after %s is held and %s retires its original source",
    async (stage, retirement) => {
      const origin = await fixture()
      expect(origin.actorSceneSend.scene).toMatchObject({
        origin: "default",
        originalRecord: null
      })
      expect(origin.actorSceneSend.requestScope).toEqual(originalScope)
      expect(origin.actorSceneSend.scene.target.address).toEqual(origin.address)
      const entered = deferred<void>()
      const release = deferred<void>()
      if (stage === "configs") {
        origin.configIO.mockImplementationOnce(async () => {
          entered.resolve()
          await release.promise
          return backendConfigs
        })
      } else {
        origin.initializeIO.mockImplementationOnce(async () => {
          entered.resolve()
          await release.promise
        })
      }
      const pending = origin.send()
      try {
        await Promise.race([
          entered.promise,
          pending.then(() => {
            throw new Error(
              "Image preparation ended before the held IO was entered"
            )
          })
        ])
        const replacement = origin.replaceView(retirement)
        expect(origin.actorSceneSend.isCurrent()).toBe(false)
        release.resolve()
        const result = await pending
        expect(origin.imageIO).not.toHaveBeenCalled()
        expect(origin.modelFactory).not.toHaveBeenCalled()
        expect(origin.modelProvider).not.toHaveBeenCalled()
        expect(origin.saveSuccess).not.toHaveBeenCalled()
        expect(origin.saveError).not.toHaveBeenCalled()
        expect(storage.tables.chatHistories.rows.size).toBe(0)
        expect(storage.tables.messages.rows.size).toBe(0)
        expect(result.status).toBe("skipped")
        expect(useStoreMessageOption.getState()).toBe(replacement)
        expect(document.title).toBe(
          retirement === "New" ? "New image draft" : "New account"
        )
      } finally {
        release.resolve()
        await pending.catch(() => {})
      }
    }
  )

  it("keeps the returned image durable for its original operation after creation entered and New retires the view", async () => {
    const origin = await fixture()
    const entered = deferred<void>()
    const returnedImage = deferred<typeof imageResponse>()
    origin.imageIO.mockImplementationOnce(async () => {
      entered.resolve()
      return returnedImage.promise
    })
    const pending = origin.send()
    try {
      await Promise.race([
        entered.promise,
        pending.then(() => {
          throw new Error(
            "Image preparation ended before createImageArtifact entered"
          )
        })
      ])
      expect(origin.imageIO).toHaveBeenCalledOnce()
      expect(origin.imageIO.mock.calls[0][1]).toMatchObject({
        requestScope: originalScope,
        signal: expect.any(AbortSignal)
      })
      expect(origin.imageIO.mock.calls[0][1]?.signal?.aborted).toBe(false)
      expect(origin.imageIO.mock.calls[0][0]).toMatchObject({
        backend: "fluxklein",
        prompt: "original image request",
        format: "png",
        width: 512,
        height: 384,
        steps: 4,
        seed: 7,
        model: "original-image-model"
      })
      const replacement = origin.replaceView("New")
      expect(origin.actorSceneSend.isCurrent()).toBe(false)
      expect(origin.account.isCurrent()).toBe(true)
      returnedImage.resolve(imageResponse)
      expect((await pending).status).toBe("submitted")
      expect(origin.saveSuccess).toHaveBeenCalledOnce()
      expect(origin.saveError).not.toHaveBeenCalled()
      expect(origin.modelFactory).not.toHaveBeenCalled()
      expect(origin.modelProvider).not.toHaveBeenCalled()
      const histories = await storage.tables.chatHistories.toArray()
      expect(histories).toHaveLength(1)
      expect(histories[0]).toMatchObject({
        title: "original image request",
        server_scope_key:
          '["https://images.test","multi-user","manual",null,"alice",null]'
      })
      const saved = await storage.tables.messages.toArray()
      expect(saved).toHaveLength(2)
      expect(saved.find((row) => row.role === "user")).toMatchObject({
        id: "image-user",
        history_id: histories[0].id,
        content: "original image request",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      })
      expect(saved.find((row) => row.role === "assistant")).toMatchObject({
        id: "image-assistant",
        history_id: histories[0].id,
        parent_message_id: "image-user",
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
        images: ["data:image/png;base64,aW1hZ2U="],
        generationInfo: {
          file_id: 77,
          image_generation: {
            request: {
              backend: "fluxklein",
              prompt: "original image request",
              width: 512,
              height: 384
            }
          }
        }
      })
      expect(useStoreMessageOption.getState()).toBe(replacement)
      expect(document.title).toBe("New image draft")
    } finally {
      returnedImage.resolve(imageResponse)
      await pending.catch(() => {})
    }
  })
})
