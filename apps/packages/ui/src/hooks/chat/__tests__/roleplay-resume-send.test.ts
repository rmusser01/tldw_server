// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { waitFor } from "@testing-library/react"
import type { TFunction } from "i18next"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { HistoryBookmark } from "@/db/dexie/types"
import type { HistorySelectionController } from "../useHistorySelection"
import type {
  HistorySelectionSnapshotV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"
import { sendNativeHistoryCharacter } from "../native-history-character-send"
import { RoleplayResumeResponseError } from "@/services/tldw/roleplay-resume"
import { sendSavedRoleplay } from "../roleplay-resume-send"
import { captureChatDraftLoad } from "../chat-draft-transition"
import { resolveHistorySelection } from "@/utils/history-selection"
import { useStoreMessageOption } from "@/store/option"
import {
  loadHistoryTurnRecoveries,
  saveHistoryTurnRecovery,
  dismissHistoryTurnRecovery
} from "@/db/dexie/history-selection"

const io = vi.hoisted(() => ({
  initialize: vi.fn(),
  config: vi.fn(),
  snapshot: vi.fn(),
  capabilities: vi.fn(),
  detail: vi.fn(),
  capture: vi.fn(),
  append: vi.fn(),
  complete: vi.fn(),
  legacy: vi.fn(),
  settings: vi.fn()
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: io.snapshot
}))
vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: io.capabilities
}))
vi.mock("@/services/tldw/TldwApiClient", async (original) => ({
  ...(await original<typeof import("@/services/tldw/TldwApiClient")>()),
  tldwClient: {
    initialize: io.initialize,
    ensureConfigForRequest: io.config,
    getChat: io.detail,
    captureHistorySelection: io.capture,
    appendRoleplayResumeInput: io.append,
    completeCharacterChatTurn: io.complete,
    streamChatCompletion: io.legacy,
    getChatSettings: io.settings
  }
}))
vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("./local-history-fixture")).memory
}))

const scope = {
  config: {
    serverUrl: "https://original-roleplay.test",
    authMode: "single-user" as const
  },
  userId: 1
}
const snapshotDigest = "sha256:" + "c".repeat(64)
const metadata = {
  id: "roleplay-chat",
  scope_type: "global",
  workspace_id: null,
  behavior_snapshot: {
    status: "valid",
    schema_version: 1,
    digest: snapshotDigest
  },
  resume_eligible: true,
  resume_ineligible_reason: null,
  settings_version: 4,
  history_version: 9,
  tail: { message_id: "prior-assistant", message_version: 2 }
}
const prepare = async (requestScope: ServicePromptRequestScope = scope) => {
  const { memory } = await import("./local-history-fixture")
  for (const table of Object.values(memory)) if (table?.rows) table.rows.clear()
  await memory.userSettings.put({ id: "main", history_profile_id: "profile" })
  const auth = new AbortController(),
    stop = new AbortController()
  let commandCurrent = true
  const owner = {
    kind: "native" as const,
    owner_key: "roleplay-owner",
    conversation_id: metadata.id,
    request_scope: requestScope,
    scope: { type: "global" as const },
    validate_lease: () => !auth.signal.aborted
  }
  const view: HistoryViewSelectionV1 = {
    owner_key: owner.owner_key,
    conversation_id: owner.conversation_id,
    view_session_id: "original-view",
    selection_revision: 1,
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "after_message", message_id: "prior-assistant" }
  }
  const canonical = [
    {
      id: "prior-user",
      revision: "opaque-user",
      parent_id: null,
      role: "user" as const,
      settled: true,
      text: "Prior question"
    },
    {
      id: "prior-assistant",
      revision: "opaque-assistant",
      parent_id: "prior-user",
      role: "assistant" as const,
      settled: true,
      text: "Prior answer"
    }
  ]
  io.capture.mockImplementation(async (_id, request) => {
    const snapshot: HistorySelectionSnapshotV1 = {
      version: 1,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      nodes: canonical.map(({ text: _text, ...node }) => node),
      fences: {
        conversation: "opaque-owner",
        history: "opaque-history",
        settings: "opaque-settings"
      },
      source_digest: "source",
      storage_context_digest: "storage",
      interpretation_status: { kind: "parent_graph_v1" }
    }
    const requested = { ...request.view, owner_key: owner.owner_key }
    const selected = resolveHistorySelection(
      snapshot,
      requested,
      request.purpose,
      ""
    )
    if (selected.status !== "ready") throw new Error(selected.code)
    return {
      status: "captured",
      snapshot,
      view: requested,
      purpose: request.purpose,
      rows: selected.rows,
      selected_content: selected.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: canonical.find((value) => value.id === row.id)!.text,
        images: []
      })),
      storage_context_digest: "storage"
    }
  })
  const capture = await io.capture(metadata.id, { view, purpose: "send" })
  let current = {
    status: "ready",
    owner,
    view,
    capture,
    bookmarkScope: { profile_id: "profile", client_session_id: "session" }
  }
  const follow = vi.fn(async (_origin, id) => {
    current = {
      ...current,
      view: {
        ...current.view,
        cursor: { kind: "after_message", message_id: id },
        selection_revision: 2
      }
    }
    return true
  })
  const controller = {
    getCurrent: () => current,
    fence: () => () => current.view === view,
    draftAccount: { requestScope, isCurrent: owner.validate_lease },
    followResult: follow,
    refreshRecovery: vi.fn(),
    takeBranchIntent: vi.fn(() => false)
  } as unknown as HistorySelectionController
  const bind = captureChatDraftLoad("switch")
  expect(bind({ owner, view })).toBe(true)
  useStoreMessageOption.setState({
    serverChatId: metadata.id,
    serverChatAssistantKind: "character",
    serverChatAssistantId: "snapshot:roleplay-chat",
    serverChatCharacterId: null,
    serverChatMetaLoaded: true,
    historyId: null,
    temporaryChat: false
  })
  io.snapshot.mockResolvedValue({
    requestScope,
    definitions: {},
    capability: "unchecked",
    scopeKey: "roleplay-owner",
    scopeSignal: stop.signal,
    scopeInvalidatedSignal: auth.signal,
    release: vi.fn()
  })
  io.capabilities.mockResolvedValue({ hasRoleplayResumeV1: true })
  io.detail.mockResolvedValue(metadata)
  io.append.mockImplementation(async (_id, input, options) => {
    options.onDispatch?.()
    return {
      version: 1,
      conversation_id: metadata.id,
      message_id: input.message_id,
      message_version: 1,
      parent_message_id: input.expected.tail.message_id,
      fence: {
        ...input.expected,
        history_version: input.expected.history_version + 1,
        tail: { message_id: input.message_id, message_version: 1 }
      }
    }
  })
  io.complete.mockImplementation(async (_id, body, options) => {
    options.onDispatch?.()
    const request = body.tldw_roleplay_resume_v1
    return {
      chat_id: metadata.id,
      character_id: 12,
      provider: "openai",
      model: "original-model",
      saved: true,
      user_message_id: request.input_message_id,
      assistant_message_id: request.assistant_message_id,
      assistant_content: "Original-roleplay reply",
      tldw_roleplay_resume_v1: {
        ...request,
        assistant_message_version: 1,
        fence: {
          ...request.expected,
          history_version: request.expected.history_version + 1,
          tail: { message_id: request.assistant_message_id, message_version: 1 }
        },
        unsaved_reason: null
      }
    }
  })
  const params: Parameters<typeof sendNativeHistoryCharacter>[0] = {
    controller,
    surface: "playground",
    originIsCurrent: controller.fence(),
    sourceIsCurrent: () => commandCurrent,
    signal: stop.signal,
    temporary: false,
    serverChatId: metadata.id,
    characterId: "snapshot:roleplay-chat",
    model: "",
    settings: {
      apiProvider: "changed-provider",
      temperature: 1.9,
      extraBody: "ignored current defaults"
    },
    message: "  Exact first reply\n",
    image: "",
    t: ((key: string) => key) as TFunction,
    setServerChatId: vi.fn(),
    setMessages: vi.fn(),
    releaseActivity: vi.fn()
  }
  return {
    controller,
    owner,
    view,
    follow,
    params,
    auth,
    stop,
    retire: () => {
      commandCurrent = false
    },
    earlier: () => {
      current = {
        ...current,
        view: {
          ...view,
          cursor: { kind: "before_message", message_id: "prior-assistant" }
        }
      }
    }
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  useStoreMessageOption.getState().startNewChatDraft()
})
describe("saved Roleplay user workflow", () => {
  it("accepts the real single-user credential scope for a linked saved character", async () => {
    io.config.mockResolvedValue({
      ...scope.config,
      apiKey: "roleplay-resume-test-key"
    })
    const real = await vi.importActual<
      typeof import("@/services/service-prompts")
    >("@/services/service-prompts")
    const verified = await real.resolveServicePromptScope()
    const requestScope = { config: verified.config, userId: verified.userId }
    expect(requestScope.userId).toBeNull()
    const mounted = await prepare(requestScope)
    io.snapshot.mockImplementation(real.loadServicePromptSnapshot)
    useStoreMessageOption
      .getState()
      .setHistoryId("linked-character-history", { preserveServerChatId: true })
    mounted.params.historyId = "linked-character-history"

    await sendNativeHistoryCharacter(mounted.params)

    expect(io.capabilities).toHaveBeenCalledWith(
      expect.objectContaining({ requestScope })
    )
    expect(io.append.mock.calls[0][1].content).toBe(mounted.params.message)
    expect(io.complete).toHaveBeenCalledOnce()
    expect(mounted.follow).toHaveBeenCalledOnce()
  })
  it("loads frozen authority and sends without a live model, card, Scene or sampler read", async () => {
    const mounted = await prepare()
    await sendNativeHistoryCharacter(mounted.params)
    expect(io.capabilities).toHaveBeenCalledWith(
      expect.objectContaining({ requestScope: scope })
    )
    expect(io.detail).toHaveBeenCalledWith(
      metadata.id,
      expect.objectContaining({
        requestScope: scope,
        requireRoleplayResume: true
      })
    )
    expect(io.append).toHaveBeenCalledOnce()
    expect(io.complete.mock.calls[0][1]).toEqual({
      tldw_roleplay_resume_v1: expect.objectContaining({
        expected: expect.objectContaining({
          snapshot_digest: snapshotDigest,
          settings_version: 4,
          history_version: 10
        })
      }),
      stream: false,
      save_to_db: true,
      include_character_context: true,
      append_user_message: false
    })
    expect(io.settings).not.toHaveBeenCalled()
    expect(io.legacy).not.toHaveBeenCalled()
    expect(mounted.follow).toHaveBeenCalledOnce()
    expect(
      await loadHistoryTurnRecoveries(
        { profile_id: "profile", client_session_id: "reopen" },
        mounted.view
      )
    ).toEqual([])
  })
  it("holds an unsupported server before input admission without legacy fallback", async () => {
    const mounted = await prepare()
    io.capabilities.mockResolvedValue({ hasRoleplayResumeV1: false })
    await expect(sendNativeHistoryCharacter(mounted.params)).rejects.toThrow()
    expect(io.append).not.toHaveBeenCalled()
    expect(io.legacy).not.toHaveBeenCalled()
  })
  it("holds a non-tail accepted view before input admission", async () => {
    const mounted = await prepare()
    mounted.earlier()
    await expect(
      sendNativeHistoryCharacter({
        ...mounted.params,
        originIsCurrent: () => true
      })
    ).rejects.toThrow()
    expect(io.append).not.toHaveBeenCalled()
  })
  it("does not append after New retires a held detail read", async () => {
    const mounted = await prepare()
    let release!: (value: typeof metadata) => void
    io.detail.mockReturnValue(
      new Promise((resolve) => {
        release = resolve
      })
    )
    const pending = sendNativeHistoryCharacter(mounted.params).catch(
      (error) => error
    )
    await vi.waitFor(() => expect(io.detail).toHaveBeenCalledOnce())
    mounted.retire()
    release(metadata)
    await pending
    expect(io.append).not.toHaveBeenCalled()
  })
  it("retains an acknowledged append but retires the next transport when New arrives", async () => {
    const mounted = await prepare()
    const admit = io.append.getMockImplementation()!
    io.append.mockImplementation(async (...args) => {
      const result = await admit(...args)
      mounted.retire()
      return result
    })
    await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
    expect(io.complete).not.toHaveBeenCalled()
    expect(mounted.follow).not.toHaveBeenCalled()
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn.state).toBe("accepted_unsent")
    expect(entry.turn.input_id).toBe(io.append.mock.calls[0][1].message_id)
    expect(entry.turn.admission).toBeUndefined()
  })
  it("keeps an original saved result when New retires its presentation during completion", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    io.complete.mockImplementation(async (...args) => {
      const result = await complete(...args)
      mounted.retire()
      return result
    })
    await sendNativeHistoryCharacter(mounted.params)
    expect(mounted.follow).not.toHaveBeenCalled()
    expect(mounted.params.setMessages).not.toHaveBeenCalled()
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn).toMatchObject({
      outcome: "complete",
      result_text: "Original-roleplay reply",
      settled_message_id:
        io.complete.mock.calls[0][1].tldw_roleplay_resume_v1
          .assistant_message_id
    })
  })
  it("keeps generated-but-unsaved text without a second persistence route", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    io.complete.mockImplementation(async (...args) => {
      const result = await complete(...args)
      return {
        ...result,
        saved: false,
        assistant_message_id: null,
        assistant_content: "x".repeat(1_000_001),
        tldw_roleplay_resume_v1: {
          ...result.tldw_roleplay_resume_v1,
          assistant_message_version: null,
          fence: null,
          unsaved_reason: "assistant_content_too_large"
        }
      }
    })
    await sendNativeHistoryCharacter(mounted.params)
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn.result_text).toHaveLength(1_000_001)
    expect(entry.turn.assistant_id).toBeUndefined()
    expect(mounted.follow).not.toHaveBeenCalled()
    expect(io.complete).toHaveBeenCalledOnce()
    expect(io.legacy).not.toHaveBeenCalled()
  })
  it("retries a lost append response with the exact original input and assistant IDs", async () => {
    const mounted = await prepare()
    const append = io.append.getMockImplementation()!
    let acknowledged: unknown
    io.append.mockImplementationOnce(async (...args) => {
      acknowledged = await append(...args)
      throw new Error("response lost")
    })
    await expect(sendNativeHistoryCharacter(mounted.params)).rejects.toThrow(
      "response lost"
    )
    const originalInput = structuredClone(io.append.mock.calls[0][1])
    const [original] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(original.turn.state).toBe("unknown")
    expect(original.turn.input_id).toBeUndefined()
    io.detail.mockRejectedValue(
      new Error("fresh detail must not replace retry authority")
    )
    io.append.mockImplementationOnce(async (_id, _input, options) => {
      options.onDispatch?.()
      return acknowledged
    })
    await sendNativeHistoryCharacter(mounted.params)
    expect(io.append.mock.calls[1][1]).toEqual(originalInput)
    expect(
      io.complete.mock.calls[0][1].tldw_roleplay_resume_v1.assistant_message_id
    ).toBe(original.turn.roleplay_resume!.intent.assistant_message_id)
    expect(io.detail).toHaveBeenCalledOnce()
  })
  it("retries a lost completion response without appending another user message", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    let acknowledged: unknown
    io.complete.mockImplementationOnce(async (...args) => {
      acknowledged = await complete(...args)
      throw new Error("completion response lost")
    })
    await expect(sendNativeHistoryCharacter(mounted.params)).rejects.toThrow(
      "completion response lost"
    )
    const originalRequest = structuredClone(io.complete.mock.calls[0][1])
    io.detail.mockRejectedValue(new Error("fresh detail forbidden"))
    io.complete.mockImplementationOnce(async (_id, _request, options) => {
      options.onDispatch?.()
      return acknowledged
    })
    await sendNativeHistoryCharacter(mounted.params)
    expect(io.append).toHaveBeenCalledOnce()
    expect(io.complete.mock.calls[1][1]).toEqual(originalRequest)
    expect(io.detail).toHaveBeenCalledOnce()
    expect(mounted.follow).toHaveBeenCalledOnce()
  })
  it.each(["Stop", "account change"])(
    "retains physical append acknowledgement after %s without dispatching completion",
    async (kind) => {
      const mounted = await prepare()
      const append = io.append.getMockImplementation()!
      io.append.mockImplementationOnce(async (...args) => {
        const receipt = await append(...args)
        if (kind === "Stop") mounted.stop.abort()
        else mounted.auth.abort()
        return receipt
      })
      await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
      const [entry] = await loadHistoryTurnRecoveries(
        { profile_id: "profile", client_session_id: "reopen" },
        mounted.view
      )
      expect(entry.turn.input_id).toBe(io.append.mock.calls[0][1].message_id)
      expect(entry.turn.state).toBe("accepted_unsent")
      expect(io.complete).not.toHaveBeenCalled()
      expect(mounted.follow).not.toHaveBeenCalled()
    }
  )
  it("retains a saved receipt when canonical following throws, and opens it without another provider call", async () => {
    const mounted = await prepare()
    mounted.follow.mockRejectedValueOnce(new Error("reload failed"))
    await sendNativeHistoryCharacter(mounted.params)
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn).toMatchObject({
      outcome: "complete",
      settled_message_id: expect.any(String),
      result_text: "Original-roleplay reply"
    })
    await sendSavedRoleplay({ ...mounted.params, retry: entry })
    expect(io.complete).toHaveBeenCalledOnce()
    expect(io.append).toHaveBeenCalledOnce()
    expect(mounted.follow).toHaveBeenCalledTimes(2)
  })
  it("does not erase earlier uncertainty when a replay receives a definitive conflict", async () => {
    const mounted = await prepare()
    io.append.mockImplementationOnce(async (_id, _input, options) => {
      options.onDispatch?.()
      throw new Error("lost append")
    })
    await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
    io.append.mockRejectedValueOnce({
      status: 409,
      details: {
        detail: { status: "resume_conflict", code: "history_changed" }
      }
    })
    await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn.state).toBe("unknown")
    expect(entry.turn.input_id).toBeUndefined()
    expect(io.complete).not.toHaveBeenCalled()
  })
  it("holds explicit Branch intent before appending", async () => {
    const mounted = await prepare()
    vi.mocked(mounted.controller.takeBranchIntent).mockReturnValueOnce(true)
    await expect(sendNativeHistoryCharacter(mounted.params)).rejects.toThrow()
    expect(io.append).not.toHaveBeenCalled()
    expect(io.complete).not.toHaveBeenCalled()
  })

  it("promotes an unsaved reply on explicit original retry while keeping the earlier reply", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    io.complete.mockImplementationOnce(async (...args) => {
      const response = await complete(...args)
      return {
        ...response,
        saved: false,
        assistant_message_id: null,
        assistant_content: "First unsaved reply",
        tldw_roleplay_resume_v1: {
          ...response.tldw_roleplay_resume_v1,
          assistant_message_version: null,
          fence: null,
          unsaved_reason: "persistence_unavailable"
        }
      }
    })
    await sendNativeHistoryCharacter(mounted.params)
    const [original] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    await sendSavedRoleplay({ ...mounted.params, retry: original })
    expect(io.append).toHaveBeenCalledOnce()
    expect(io.complete.mock.calls[1][1]).toEqual(io.complete.mock.calls[0][1])
    expect(mounted.follow).toHaveBeenCalledOnce()
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn).toMatchObject({
      outcome: "complete",
      result_text: "Original-roleplay reply",
      retained_result_texts: ["First unsaved reply"]
    })
  })

  it("requires explicit Retry after a known unsaved response instead of silently restarting generation", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    io.complete.mockImplementationOnce(async (...args) => {
      const response = await complete(...args)
      return {
        ...response,
        saved: false,
        assistant_message_id: null,
        assistant_content: "Known unsaved reply",
        tldw_roleplay_resume_v1: {
          ...response.tldw_roleplay_resume_v1,
          assistant_message_version: null,
          fence: null,
          unsaved_reason: "persistence_unavailable"
        }
      }
    })
    await sendNativeHistoryCharacter(mounted.params)
    await expect(sendNativeHistoryCharacter(mounted.params)).rejects.toThrow()
    expect(io.complete).toHaveBeenCalledOnce()
    expect(io.append).toHaveBeenCalledOnce()
  })

  it("does not reclassify a prewritten loaded dispatch as a fresh refusal", async () => {
    const mounted = await prepare()
    io.append.mockImplementationOnce(async (_id, _body, options) => {
      options.onDispatch?.()
      throw new Error("original response lost")
    })
    await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
    const [original] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    await saveHistoryTurnRecovery(original.scope, mounted.view, {
      ...original.turn,
      state: "dispatching"
    })
    io.append.mockRejectedValueOnce({
      status: 409,
      details: {
        detail: { status: "resume_conflict", code: "history_changed" }
      }
    })
    await sendNativeHistoryCharacter(mounted.params).catch(() => undefined)
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn.state).toBe("unknown")
    expect(entry.turn.input_id).toBeUndefined()
  })
  it("keeps unsaved text under its original owner when quota failure follows New", async () => {
    const mounted = await prepare()
    const complete = io.complete.getMockImplementation()!
    io.complete.mockImplementationOnce(async (...args) => {
      const response = await complete(...args)
      return {
        ...response,
        saved: false,
        assistant_message_id: null,
        assistant_content: "Quota original answer",
        tldw_roleplay_resume_v1: {
          ...response.tldw_roleplay_resume_v1,
          assistant_message_version: null,
          fence: null,
          unsaved_reason: "persistence_unavailable"
        }
      }
    })
    const { memory } = await import("./local-history-fixture")
    const originalPut = memory.historySelections.put
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const spy = vi
      .spyOn(memory.historySelections, "put")
      .mockImplementation(async (value: HistoryBookmark) => {
        if (
          Object.values(value.pending_turns ?? {}).some(
            (turn) => turn.result_text
          )
        ) {
          await held
          throw new DOMException("storage full", "QuotaExceededError")
        }
        return originalPut(value)
      })
    try {
      const pending = sendNativeHistoryCharacter(mounted.params).catch(
        (error) => error
      )
      await vi.waitFor(() =>
        expect(
          spy.mock.calls.some(([value]) =>
            Object.values((value as HistoryBookmark).pending_turns ?? {}).some(
              (turn) => turn.result_text
            )
          )
        ).toBe(true)
      )
      mounted.retire()
      release()
      await pending
      expect(mounted.params.setMessages).not.toHaveBeenCalled()
      const [entry] = await loadHistoryTurnRecoveries(
        { profile_id: "profile", client_session_id: "reopen" },
        mounted.view
      )
      expect(entry.turn.result_text).toBe("Quota original answer")
      expect(entry.turn.recovery_storage).toBe("memory")
      expect(
        await loadHistoryTurnRecoveries(
          { profile_id: "profile", client_session_id: "reopen" },
          { ...mounted.view, owner_key: "foreign-owner" }
        )
      ).toEqual([])
      spy.mockRestore()
      await dismissHistoryTurnRecovery(
        entry.scope,
        mounted.view,
        entry.turn.operation_id
      )
    } finally {
      spy.mockRestore()
    }
  })

  it("keeps new malformed reply text without contradicting an earlier valid unsaved receipt", async () => {
    const mounted = await prepare(),
      complete = io.complete.getMockImplementation()!
    io.complete.mockImplementationOnce(async (...args) => {
      const response = await complete(...args)
      return {
        ...response,
        saved: false,
        assistant_message_id: null,
        assistant_content: "First qualified unsaved reply",
        tldw_roleplay_resume_v1: {
          ...response.tldw_roleplay_resume_v1,
          assistant_message_version: null,
          fence: null,
          unsaved_reason: "persistence_unavailable"
        }
      }
    })
    await sendNativeHistoryCharacter(mounted.params)
    const [original] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    io.complete.mockRejectedValueOnce(
      new RoleplayResumeResponseError({
        assistant_content: "Second malformed reply"
      })
    )
    await sendSavedRoleplay({ ...mounted.params, retry: original }).catch(
      () => undefined
    )
    const [entry] = await loadHistoryTurnRecoveries(
      { profile_id: "profile", client_session_id: "reopen" },
      mounted.view
    )
    expect(entry.turn).toMatchObject({
      result_text: "First qualified unsaved reply",
      retained_result_texts: ["Second malformed reply"]
    })
    expect(entry.turn.roleplay_resume?.completion?.assistant_content).toBe(
      "First qualified unsaved reply"
    )
    expect(entry.turn.assistant_id).toBeUndefined()
  })
})


it.each(["request-scope", "handle-owner"])("private saved Roleplay rejects a foreign %s before transport", async boundary => {
  const mounted = await prepare()
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const handle = { address: freshChatDraftAddress(false), ownerHash: boundary === "handle-owner" ? "d".repeat(64) : chatDraftOwnerKey(scope) }
  const result = await sendNativeHistoryCharacter({ ...mounted.params,
    privateDraft: { handle, requestScope: boundary === "request-scope" ? { ...scope, userId: 2 } : scope, isCurrent: () => true }
  }).catch(error => error)
  expect(result).toMatchObject({ message: "request_config_scope_changed" })
  expect(io.detail).not.toHaveBeenCalled()
  expect(io.append).not.toHaveBeenCalled()
  expect(io.complete).not.toHaveBeenCalled()
  expect(mounted.owner.validate_lease()).toBe(true)
})


it("private saved Roleplay preserves its original view across a held creation-record read", async () => {
  const mounted = await prepare()
  const { freshChatDraftAddress } = await import("@/utils/chat-draft-address")
  const { chatDraftOwnerKey } = await import("@/hooks/useChatDraftOwner")
  const creation = await import("@/services/pending-chat-promotion")
  let entered!: () => void, release!: (value: undefined) => void
  const ready = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<undefined>(resolve => { release = resolve })
  const read = vi.spyOn(creation, "readChatCreationRecovery").mockImplementationOnce(() => { entered(); return held })
  const handle = { address: freshChatDraftAddress(false), ownerHash: chatDraftOwnerKey(scope) }
  const pending = sendNativeHistoryCharacter({ ...mounted.params,
    originIsCurrent: () => !mounted.stop.signal.aborted,
    privateDraft: { handle, requestScope: scope, isCurrent: () => !mounted.auth.signal.aborted }
  }).catch(error => error)
  try {
    await Promise.race([ready, pending.then(result => { throw new Error(`Sender finished before record read: ${String(result)}`) })])
    mounted.earlier()
    release(undefined)
    expect(await pending).toMatchObject({ message: "stale_selection" })
    expect(io.detail).not.toHaveBeenCalled()
    expect(io.append).not.toHaveBeenCalled()
    expect(io.complete).not.toHaveBeenCalled()
    expect(mounted.owner.validate_lease()).toBe(true)
  } finally {
    release(undefined)
    await pending
    read.mockRestore()
  }
})


describe("saved Roleplay Retry input receipt boundary", () => {
  it.each(["put", "read"] as const)(
    "acknowledges the confirmed input even if completed recovery %s fails",
    async (failure) => {
      const mounted = await prepare()
      io.complete.mockRejectedValueOnce(
        new Error("Original completion response lost")
      )
      await sendSavedRoleplay(mounted.params).catch(() => undefined)
      const [entry] = await loadHistoryTurnRecoveries(
        { profile_id: "profile", client_session_id: "reopen" },
        mounted.view
      )
      expect(entry.turn.roleplay_resume?.append).toBeTruthy()
      const { memory } = await import("./local-history-fixture")
      const originalPut = memory.historySelections.put
      const read = vi.spyOn(memory.historySelections, "get")
      let failed = false
      const put = vi
        .spyOn(memory.historySelections, "put")
        .mockImplementation(async (value: HistoryBookmark) => {
          const saved = Object.values(value.pending_turns ?? {}).some(
            (turn) => turn.roleplay_resume?.completion?.saved
          )
          if (saved && !failed) {
            failed = true
            if (failure === "put")
              throw new DOMException("storage full", "QuotaExceededError")
            const result = await originalPut(value)
            read.mockRejectedValueOnce(
              new Error("Completed recovery read unavailable")
            )
            return result
          }
          return originalPut(value)
        })
      const acknowledgeInput = vi.fn(() => true)
      try {
        expect(
          await sendSavedRoleplay({
            ...mounted.params,
            retry: entry,
            acknowledgeInput
          })
        ).toBe(true)
        expect(failed).toBe(true)
        expect(acknowledgeInput).toHaveBeenCalledOnce()
      } finally {
        put.mockRestore()
        read.mockRestore()
      }
    }
  )

  it("acknowledges the original selected input before its own canonical follow", async () => {
    const mounted = await prepare()
    const acknowledgeInput = vi.fn(() => {
      expect(mounted.controller.getCurrent().view).toBe(mounted.view)
      expect(mounted.follow).not.toHaveBeenCalled()
      return true
    })
    expect(
      await sendSavedRoleplay({ ...mounted.params, acknowledgeInput })
    ).toBe(true)
    expect(acknowledgeInput).toHaveBeenCalledOnce()
    expect(mounted.follow).toHaveBeenCalledOnce()
  })
  it.each(["boundary", "New", "account", "Stop"] as const)(
    "retains the composer when physical saved completion arrives after %s",
    async (boundary) => {
      const mounted = await prepare(),
        complete = io.complete.getMockImplementation()!,
        acknowledgeInput = vi.fn(() => true)
      io.complete.mockImplementationOnce(async (...args) => {
        const result = await complete(...args)
        if (boundary === "boundary") mounted.earlier()
        if (boundary === "New") mounted.retire()
        if (boundary === "account") mounted.auth.abort()
        if (boundary === "Stop") mounted.stop.abort()
        return result
      })
      await sendSavedRoleplay({ ...mounted.params, acknowledgeInput }).catch(
        () => undefined
      )
      expect(acknowledgeInput).not.toHaveBeenCalled()
    }
  )
})

describe("bound native roleplay feedback ownership", () => {
  const providerError = () => Object.assign(new Error("HTTP 401"), { status: 401,
    details: { detail: { error: { code: "provider_authentication_failed", type: "provider_authentication_failed" } } } })
  it("wraps current bound provider failure for settings recovery and retains one original accepted input", async () => {
    const mounted = await prepare()
    const failure = providerError()
    io.complete.mockRejectedValueOnce(failure)
    const error = await sendNativeHistoryCharacter(mounted.params).catch(cause => cause)
    const { NativeHistoryRoleplaySendError } = await import("../native-character-preflight")
    const { classifyCharacterChatFailureRecovery } = await import("../useCharacterChatMode")
    expect(error).toBeInstanceOf(NativeHistoryRoleplaySendError)
    expect(error.cause).toBe(failure)
    expect(error.isCurrent()).toBe(true)
    expect(classifyCharacterChatFailureRecovery(error)).toMatchObject({ kind: "provider_unconfigured", action: "open-model-settings" })
    const [entry] = await loadHistoryTurnRecoveries({ profile_id: "profile", client_session_id: "reopen" }, mounted.view)
    expect(entry.turn).toMatchObject({ input_text: mounted.params.message, state: "accepted_unsent", roleplay_resume: { append: { conversation_id: metadata.id } } })
    expect(io.append).toHaveBeenCalledOnce()
    expect(io.complete).toHaveBeenCalledOnce()
    expect(io.legacy).not.toHaveBeenCalled()
    expect(mounted.follow).not.toHaveBeenCalled()
    expect(mounted.params.releaseActivity).toHaveBeenCalledOnce()
  })
  it.each(["view", "source", "account", "draft", "Stop"] as const)("keeps a held bound provider failure retired after %s changes", async boundary => {
    const mounted = await prepare()
    let reject!: (error: Error) => void
    io.complete.mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail }))
    const pending = sendNativeHistoryCharacter(mounted.params).catch(cause => cause)
    await waitFor(() => expect(io.complete).toHaveBeenCalledOnce())
    if (boundary === "view") mounted.earlier()
    if (boundary === "source") mounted.retire()
    if (boundary === "account") mounted.auth.abort()
    if (boundary === "draft") useStoreMessageOption.getState().startNewChatDraft()
    if (boundary === "Stop") mounted.stop.abort()
    reject(providerError())
    const error = await pending
    const { NativeHistoryRoleplaySendError } = await import("../native-character-preflight")
    expect(error).toBeInstanceOf(NativeHistoryRoleplaySendError)
    expect(error.isCurrent()).toBe(false)
    if (boundary === "Stop") expect(mounted.controller.refreshRecovery).toHaveBeenCalledOnce()
    else expect(mounted.controller.refreshRecovery).not.toHaveBeenCalled()
    expect(mounted.follow).not.toHaveBeenCalled()
    expect(io.append).toHaveBeenCalledOnce()
    expect(io.complete).toHaveBeenCalledOnce()
    expect(mounted.params.releaseActivity).toHaveBeenCalledOnce()
  })
  it("preserves the bound typed settings/context refusal before any transport", async () => {
    const mounted = await prepare()
    const { NativeHistorySettingsReadError } = await import("../native-character-preflight")
    const error = await sendNativeHistoryCharacter({ ...mounted.params, image: "synthetic-image" }).catch(cause => cause)
    expect(error).toBeInstanceOf(NativeHistorySettingsReadError)
    expect(error.cause).toMatchObject({ message: "roleplay_resume_explicit_context_unsupported" })
    expect(io.append).not.toHaveBeenCalled()
    expect(io.complete).not.toHaveBeenCalled()
    expect(mounted.params.releaseActivity).toHaveBeenCalledOnce()
  })
})
