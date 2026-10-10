// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { TFunction } from "i18next"
import { useStoreMessageOption } from "@/store/option"
import { createDefaultActorSettings } from "@/types/actor"
import { saveActorSettingsForChat } from "@/services/actor-settings"
import {
  captureCurrentActorSceneTarget,
  captureActorSceneSend
} from "../actor-scene-source"
import { sendNativeHistoryCharacter } from "../native-history-character-send"
import {
  useHistorySelection,
  type HistorySelectionController
} from "../useHistorySelection"
import { captureChatDraftLoad } from "../chat-draft-transition"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { resolveHistorySelection } from "@/utils/history-selection"
import { loadHistoryTurnRecoveries } from "@/db/dexie/history-selection"
import type {
  HistorySelectionSnapshotV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const io = vi.hoisted(() => ({
  snapshot: vi.fn(),
  create: vi.fn(),
  bareSettings: vi.fn(),
  capture: vi.fn(),
  settings: vi.fn(),
  stream: vi.fn(),
  spec: vi.fn()
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: io.snapshot
}))
vi.mock("@/services/actor-settings", async (original) => ({
  ...(await original<typeof import("@/services/actor-settings")>()),
  getActorSettingsForChat: io.bareSettings
}))
vi.mock("@/services/tldw/TldwApiClient", async (original) => ({
  ...(await original<typeof import("@/services/tldw/TldwApiClient")>()),
  tldwClient: {
    createChat: io.create,
    captureHistorySelection: io.capture,
    getChatSettings: io.settings,
    streamChatCompletion: io.stream,
    getOpenAPISpec: io.spec
  }
}))

vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("./local-history-fixture")).memory
}))

const requestScope = {
  config: {
    serverUrl: "https://native-scene-command.example",
    authMode: "single-user" as const
  },
  userId: 1
}
const snapshot = (scope = requestScope): ServicePromptSnapshot => ({
  requestScope: scope,
  definitions: {},
  capability: "unchecked",
  scopeKey: "native-scene-command",
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: vi.fn()
})
const t = ((key: string) => key) as TFunction
const prepare = async (enabled = false) => {
  const controller = {
    getCurrent: () => ({ status: "idle", owner: null, view: null }),
    fence: () => () => true,
    draftAccount: { requestScope, isCurrent: () => true },
    loadConversation: vi.fn()
  } as unknown as HistorySelectionController
  const target = captureCurrentActorSceneTarget(controller)!
  expect(
    await saveActorSettingsForChat({
      target,
      historyId: null,
      serverChatId: null,
      expectedRecord: null,
      settings: {
        ...createDefaultActorSettings(),
        isEnabled: enabled,
        notes: "Original native command scene"
      }
    })
  ).toBe(true)
  const actorSceneSend = await captureActorSceneSend(controller)!.read()
  return { controller, actorSceneSend }
}
const dispatch = ({
  controller,
  actorSceneSend
}: Awaited<ReturnType<typeof prepare>>) =>
  sendNativeHistoryCharacter({
    controller,
    actorSceneSend,
    surface: "playground",
    originIsCurrent: () => true,
    signal: new AbortController().signal,
    temporary: false,
    characterId: "12",
    model: "openai:fixture-model",
    settings: { apiProvider: "openai" },
    message: "Original native input",
    image: "",
    t,
    setServerChatId: vi.fn(),
    setMessages: vi.fn(),
    releaseActivity: vi.fn()
  } as Parameters<typeof sendNativeHistoryCharacter>[0])

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: async (_key: string, work: () => Promise<unknown>) => work()
    }
  })
  useStoreMessageOption.getState().startNewChatDraft()
  useStoreMessageOption.setState({ historyId: null, serverChatId: null })
  io.bareSettings.mockResolvedValue(createDefaultActorSettings())
  io.create.mockRejectedValue(new Error("fixture stops before creation"))
  io.snapshot.mockResolvedValue(snapshot())
})

it("retains the command's original scope before held native prompt preparation", async () => {
  const prepared = await prepare()
  let release!: (value: ServicePromptSnapshot) => void
  io.snapshot.mockReturnValueOnce(
    new Promise((resolve) => {
      release = resolve
    })
  )
  const result = dispatch(prepared).catch((error) => error)
  await vi.waitFor(() => expect(io.snapshot).toHaveBeenCalledOnce())
  useStoreMessageOption.getState().startNewChatDraft()
  release(snapshot())
  await result
  expect(io.snapshot).toHaveBeenCalledWith(
    [],
    expect.objectContaining({ requestScope })
  )
  expect(io.bareSettings).not.toHaveBeenCalled()
  expect(io.create).not.toHaveBeenCalled()
})

it("refuses enabled client Actor context from the frozen scene before native creation", async () => {
  const result = await dispatch(await prepare(true)).catch((error) => error)
  expect(result).toMatchObject({
    message: "native_history_actor_context_unsupported"
  })
  expect(io.snapshot).not.toHaveBeenCalled()
  expect(io.create).not.toHaveBeenCalled()
  expect(io.bareSettings).not.toHaveBeenCalled()
})

it("refuses a native snapshot for another account before creation", async () => {
  io.snapshot.mockResolvedValueOnce(snapshot({ ...requestScope, userId: 2 }))
  const result = await dispatch(await prepare()).catch((error) => error)
  expect(result).toMatchObject({ message: "request_config_scope_changed" })
  expect(io.create).not.toHaveBeenCalled()
})

// Real H1/parser/store/scene/journal; capture and provider I/O use declared
// canonical rows. The transaction fixture is not an IndexedDB/API acceptance test.
describe("native admitted reasoning progress", () => {
  const mountAdmitted = async (detached = false) => {
    const { memory } = await import("./local-history-fixture")
    for (const table of Object.values(memory))
      if (table?.rows) table.rows.clear()
    sessionStorage.clear()
    await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
    const chatId = "native-reasoning-chat"
    const ownerKey = "native-reasoning-owner"
    const auth = new AbortController()
    const response = new AbortController()
    const canonical = [
      {
        id: "old-user",
        revision: "1",
        parent_id: null,
        role: "user",
        settled: true,
        text: "Original question"
      },
      {
        id: "old-assistant",
        revision: "1",
        parent_id: "old-user",
        role: "assistant",
        settled: true,
        text: "Original answer"
      }
    ]
    io.capture.mockImplementation(async (_id, request) => {
      const state: HistorySelectionSnapshotV1 = {
        version: 1,
        owner_key: ownerKey,
        conversation_id: chatId,
        fences: {
          conversation: "1",
          history: String(canonical.length),
          settings: "1"
        },
        nodes: canonical.map(({ text: _text, ...node }) => node),
        source_digest: "source",
        storage_context_digest: "storage",
        interpretation_status: { kind: "parent_graph_v1" }
      }
      const view: HistoryViewSelectionV1 = {
        ...request.view,
        owner_key: ownerKey
      }
      const resolved = resolveHistorySelection(state, view, request.purpose, "")
      if (resolved.status !== "ready") throw new Error(resolved.code)
      return {
        status: "captured",
        snapshot: state,
        view,
        rows: resolved.rows,
        purpose: request.purpose,
        selected_content: resolved.rows.map((row) => ({
          id: row.id,
          revision: row.revision,
          message: canonical.find((source) => source.id === row.id)!.text,
          images: []
        })),
        storage_context_digest: "storage"
      }
    })
    io.settings.mockResolvedValue({ conversation_id: chatId, settings: {} })
    io.spec.mockResolvedValue({
      paths: {
        "/api/v1/chat/completions": {
          post: {
            requestBody: {
              content: {
                "application/json": {
                  schema: {
                    type: "object",
                    properties: { tldw_history_branch: { type: "boolean" } }
                  }
                }
              }
            }
          }
        }
      }
    })
    io.snapshot.mockImplementation(async (_ids, options) => ({
      ...snapshot(),
      scopeSignal: options.signal,
      scopeInvalidatedSignal: auth.signal
    }))
    const selection = renderHook(() =>
      useHistorySelection({
        onCapture: (capture) =>
          useStoreMessageOption.setState(formatSelectedHistory(capture))
      })
    )
    const owner = {
      kind: "native" as const,
      owner_key: ownerKey,
      conversation_id: chatId,
      request_scope: requestScope,
      scope: { type: "global" as const },
      validate_lease: () => !auth.signal.aborted
    }
    await act(async () => {
      expect(await selection.result.current.open(owner, null)).toBe(true)
    })
    const controller: HistorySelectionController = {
      ...selection.result.current,
      draftAccount: {
        requestScope,
        isCurrent: owner.validate_lease,
        retryVerification: vi.fn()
      }
    }
    const origin = controller.getCurrent()
    expect(origin.status).toBe("ready")
    expect(origin.capture?.rows.map((row) => row.id)).toEqual([
      "old-user",
      "old-assistant"
    ])
    const bind = captureChatDraftLoad("switch")
    expect(bind({ owner: origin.owner!, view: origin.view! })).toBe(true)
    useStoreMessageOption.setState({
      historyId: null,
      serverChatId: chatId,
      temporaryChat: false
    })
    const target = captureCurrentActorSceneTarget(controller)!
    expect(
      await saveActorSettingsForChat({
        target,
        expectedRecord: null,
        settings: {
          ...createDefaultActorSettings(),
          isEnabled: false,
          notes: "Qualified native reasoning scene"
        }
      })
    ).toBe(true)
    const actorSceneSend = await captureActorSceneSend(controller)!.read()
    const originalRows = structuredClone(
      useStoreMessageOption.getState().messages
    )
    const originalCapture = controller.getCurrent().capture
    const follow = vi.spyOn(controller, "followResult")
    let release!: () => void,
      entered = false,
      editorCurrent = true
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.stream.mockImplementation(async function* (request) {
      const selection = request.tldw_history_selection_v1
      canonical.push({
        id: "new-user",
        revision: "1",
        parent_id: selection.messages.at(-1)?.id ?? null,
        role: "user",
        settled: true,
        text: detached ? "Edited question" : "Next question"
      })
      yield {
        tldw_history_admission_v1: {
          version: 1,
          owner_key: ownerKey,
          conversation_id: chatId,
          input_message_id: "new-user",
          input_message_revision: "1",
          selection_digest: selection.selection_digest,
          messages: selection.messages,
          originating_selection_revision: selection.selection_revision
        }
      }
      yield { choices: [{ delta: { reasoning_content: "Plan the reply" } }] }
      entered = true
      await held
      yield { choices: [{ delta: { content: "Answer" } }] }
      canonical.push({
        id: "new-assistant",
        revision: "1",
        parent_id: "new-user",
        role: "assistant",
        settled: true,
        text: "<think>Plan the reply</think>Answer"
      })
      yield { tldw_message_id: "new-assistant", tldw_conversation_id: chatId }
    })
    const params: Parameters<typeof sendNativeHistoryCharacter>[0] = {
      controller,
      actorSceneSend,
      surface: "playground",
      originIsCurrent: controller.fence(),
      sourceIsCurrent: actorSceneSend.isCurrent,
      signal: response.signal,
      temporary: false,
      serverChatId: chatId,
      characterId: "12",
      model: "openai:fixture-model",
      settings: { apiProvider: "openai" },
      message: detached ? "Edited question" : "Next question",
      image: "",
      t,
      ...(detached
        ? {
            historyEditResend: {
              owner_key: ownerKey,
              conversation_id: chatId,
              message_id: "old-user",
              message_revision: "1",
              isCurrent: () => editorCurrent,
              isOriginCurrent: () => true
            }
          }
        : {}),
      setServerChatId: useStoreMessageOption.getState().setServerChatId,
      setMessages: useStoreMessageOption.getState().setMessages,
      releaseActivity: vi.fn()
    }
    return {
      controller,
      selection,
      origin,
      originalRows,
      originalCapture,
      params,
      response,
      auth,
      follow,
      release,
      entered: () => entered,
      retireEditor: () => {
        editorCurrent = false
      },
      journal: () =>
        loadHistoryTurnRecoveries(origin.bookmarkScope!, origin.view!)
    }
  }

  it("publishes held Thinking progress after actual admission without claiming a saved assistant", async () => {
    const mounted = await mountAdmitted()
    const pending = sendNativeHistoryCharacter(mounted.params)
    void pending.catch(() => {})
    try {
      await waitFor(() => expect(mounted.entered()).toBe(true))
      expect(io.stream).toHaveBeenCalledOnce()
      expect(mounted.params.actorSceneSend!.isCurrent()).toBe(true)
      expect(mounted.controller.getCurrent().capture).toBe(
        mounted.originalCapture
      )
      expect((await mounted.journal()).map((entry) => entry.turn)).toEqual([
        expect.objectContaining({
          state: "accepted_unsent",
          input_id: "new-user"
        })
      ])
      expect((await mounted.journal())[0].turn.assistant_id).toBeUndefined()
      expect(mounted.follow).not.toHaveBeenCalled()
      expect(useStoreMessageOption.getState().messages.at(-1)?.message).toBe(
        "<think>Plan the reply"
      )
      expect(useStoreMessageOption.getState().messages.slice(0, 2)).toEqual(
        mounted.originalRows
      )
      await act(async () => {
        mounted.release()
        await pending
      })
      expect(mounted.follow).toHaveBeenCalledOnce()
      expect(useStoreMessageOption.getState().messages.at(-1)?.message).toBe(
        "<think>Plan the reply</think>Answer"
      )
    } finally {
      mounted.release()
      await pending.catch(() => {})
      mounted.selection.unmount()
    }
  })

  it.each(["Stop", "account"] as const)(
    "keeps reasoning-only %s recovery without inventing settlement",
    async (boundary) => {
      const mounted = await mountAdmitted()
      const pending = sendNativeHistoryCharacter(mounted.params)
      void pending.catch(() => {})
      try {
        await waitFor(() => expect(mounted.entered()).toBe(true))
        expect(io.stream).toHaveBeenCalledOnce()
        expect((await mounted.journal())[0].turn).toMatchObject({
          state: "accepted_unsent",
          input_id: "new-user"
        })
        if (boundary === "Stop") mounted.response.abort()
        else mounted.auth.abort()
        mounted.release()
        await pending.catch(() => {})
        const entries = await mounted.journal()
        expect(entries).toHaveLength(1)
        expect(entries[0].turn).toMatchObject({
          state: "generated_unsaved",
          input_id: "new-user",
          result_text: "<think>Plan the reply"
        })
        expect(entries[0].turn.assistant_id).toBeUndefined()
        expect(entries[0].turn.settled_message_id).toBeUndefined()
        expect(mounted.follow).not.toHaveBeenCalled()
      } finally {
        mounted.release()
        await pending.catch(() => {})
        mounted.selection.unmount()
      }
    }
  )

  it("keeps New's replacement rows while an admitted reasoning-only response settles", async () => {
    const mounted = await mountAdmitted()
    const pending = sendNativeHistoryCharacter(mounted.params)
    void pending.catch(() => {})
    try {
      await waitFor(() => expect(mounted.entered()).toBe(true))
      const replacement = [
        {
          id: "replacement",
          isBot: false,
          message: "Keep my new draft",
          sources: []
        }
      ]
      act(() => {
        useStoreMessageOption.getState().startNewChatDraft()
        useStoreMessageOption.setState({ messages: replacement })
      })
      await act(async () => {
        mounted.release()
        await pending
      })
      expect(useStoreMessageOption.getState().messages).toEqual(replacement)
      expect(mounted.follow).not.toHaveBeenCalled()
    } finally {
      mounted.release()
      await pending.catch(() => {})
      mounted.selection.unmount()
    }
  })

  it("keeps detached original cards and complete recovery when the editor cannot follow", async () => {
    const mounted = await mountAdmitted(true)
    const pending = sendNativeHistoryCharacter(mounted.params)
    void pending.catch(() => {})
    try {
      await waitFor(() => expect(mounted.entered()).toBe(true))
      expect(useStoreMessageOption.getState().messages).toEqual(
        mounted.originalRows
      )
      expect(mounted.follow).not.toHaveBeenCalled()
      mounted.retireEditor()
      await act(async () => {
        mounted.release()
        await pending
      })
      expect(useStoreMessageOption.getState().messages).toEqual(
        mounted.originalRows
      )
      const entries = await mounted.journal()
      expect(entries[0].turn).toMatchObject({
        action: "edit_resend",
        state: "generated_unsaved",
        outcome: "complete",
        settled_message_id: "new-assistant",
        result_text: "<think>Plan the reply</think>Answer"
      })
      expect(mounted.follow).not.toHaveBeenCalled()
    } finally {
      mounted.release()
      await pending.catch(() => {})
      mounted.selection.unmount()
    }
  })
})
