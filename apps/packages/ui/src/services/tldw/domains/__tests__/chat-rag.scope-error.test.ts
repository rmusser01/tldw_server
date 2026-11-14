import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import { chatRagMethods } from "../chat-rag"
import type { TldwApiClientCore } from "../../TldwApiClient"

describe("chat RAG scoped errors", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("preserves a structured request-scope rejection", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(scopeError)

    const request = chatRagMethods.ragSearch.call(
      {
        normalizeRagQuery: (query: string) => query
      } as any,
      "question",
      {
        requestScope: {
          config: {
            serverUrl: "https://server.test",
            authMode: "multi-user"
          },
          userId: 42
        }
      }
    )

    await expect(request).rejects.toBe(scopeError)
  })
})
it.each(["getChat", "getChatSettings", "updateChatSettings"] as const)("forwards scoped credentials, workspace and cancellation for %s", async method => {
  mocks.bgRequest.mockReset().mockResolvedValue({id: "child", conversation_id: "child", settings: {}})
  const signal = new AbortController().signal
  const options = {scope: {type: "workspace" as const, workspaceId: "original"}, signal,
    requestScope: {config: {serverUrl: "https://original.test", authMode: "multi-user" as const}, userId: "alice"}}
  const client = {normalizeChatSummary: (value: any) => value} as any
  if (method === "updateChatSettings") await chatRagMethods[method].call(client, "child", {authorNote: "explicit"}, options)
  else await chatRagMethods[method].call(client, "child", options)
  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: expect.stringContaining("scope_type=workspace&workspace_id=original"), abortSignal: signal,
    headers: expect.objectContaining({"X-TLDW-Expected-User-ID": "alice"}),
    servicePromptConfig: expect.objectContaining({serverUrl: "https://original.test", expectedUserId: "alice"})
  }))
})


describe("authoritative detail history versions", () => {
  const client = {
    normalizeChatSummary: chatRagMethods.normalizeChatSummary
  } as unknown as TldwApiClientCore
  it.each([0, 1, Number.MAX_SAFE_INTEGER])(
    "preserves safe history version %s",
    async (version) => {
      mocks.bgRequest.mockResolvedValueOnce({
        id: "child",
        history_version: version
      })
      const result = await chatRagMethods.getChat.call(client, "child", {
        requireHistoryVersion: true
      })
      expect(result.history_version).toBe(version)
    }
  )
  it.each([undefined, null, "4", -1, 1.5, NaN, Number.MAX_SAFE_INTEGER + 1])(
    "does not certify malformed history version %s",
    async (history_version) => {
      mocks.bgRequest.mockResolvedValueOnce({ id: "child", history_version })
      await expect(
        chatRagMethods.getChat.call(client, "child", {
          requireHistoryVersion: true
        })
      ).rejects.toThrow("unsupported_history_version")
    }
  )
  it.each([
    {},
    { id: "foreign", history_version: 3 },
    { chat_id: "child", history_version: 3 },
    { id: 1, history_version: 3 }
  ])("rejects unbound detail %j before normalization", async (body) => {
    mocks.bgRequest.mockResolvedValueOnce(body)
    await expect(
      chatRagMethods.getChat.call(client, "child", {
        requireHistoryVersion: true
      })
    ).rejects.toThrow("unsupported_history_version")
  })
  it("keeps an older detail readable without certifying a history version", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: "child" })
    const result = await chatRagMethods.getChat.call(client, "child")
    expect(result.history_version).toBeNull()
  })
})

it.each([
  { scope_type: "global", workspace_id: null },
  { scope_type: "workspace", workspace_id: "foreign" },
  { workspace_id: "original" }
])(
  "cannot use a history version from foreign or absent workspace metadata %j",
  async (scope) => {
    mocks.bgRequest.mockResolvedValueOnce({
      id: "child",
      history_version: 3,
      ...scope
    })
    await expect(
      chatRagMethods.getChat.call(
        {
          normalizeChatSummary: chatRagMethods.normalizeChatSummary
        } as unknown as TldwApiClientCore,
        "child",
        {
          requireHistoryVersion: true,
          scope: { type: "workspace", workspaceId: "original" }
        }
      )
    ).rejects.toThrow("unsupported_history_version")
  }
)


describe("definite native no-admission observer", () => {
  const selection = { conversation_id: "owned", messages: [{ id: "parent" }] }
  const options = {
    requestScope: {
      config: {
        serverUrl: "https://original.test",
        authMode: "single-user" as const,
        apiKey: "synthetic"
      }
    }
  }
  const client = {
    invalidateChatMessagesCache: vi.fn()
  } as unknown as TldwApiClientCore
  it("a captured strict-false user POST preserves acknowledged history after its exact no-admission409", async () => {
    const {
      resetServerChatSaveStatus,
      beginServerChatWrite,
      getServerChatSaveStatus
    } = await import("@/store/server-chat-save-status")
    resetServerChatSaveStatus()
    beginServerChatWrite("owned")("saved")
    const refusal = Object.assign(new Error("history_branch_changed"), {
      status: 409,
      details: {
        detail: {
          status: "stale_selection",
          code: "history_branch_changed",
          conversation_id: "owned",
          parent_message_id: "parent",
          leaf_ids: ["new-tip"],
          history_version: 4
        }
      }
    })
    mocks.bgRequest.mockRejectedValueOnce(refusal)
    await expect(
      chatRagMethods.addChatMessage.call(
        client,
        "owned",
        {
          role: "user",
          content: "Exact input.  \nNext line.",
          tldw_history_branch: false,
          tldw_history_selection_v1: selection
        },
        options
      )
    ).rejects.toBe(refusal)
    expect(getServerChatSaveStatus("owned")).toBe("saved")
    expect(client.invalidateChatMessagesCache).not.toHaveBeenCalled()
  })
  const refusal = () =>
    Object.assign(new Error("history_branch_changed"), {
      status: 409,
      details: {
        detail: {
          status: "stale_selection",
          code: "history_branch_changed",
          conversation_id: "owned",
          parent_message_id: "parent",
          leaf_ids: ["new-tip"],
          history_version: 4
        }
      }
    })
  it.each([
    { name: "explicit branch", payload: { tldw_history_branch: true } },
    {
      name: "missing strict flag",
      payload: { tldw_history_branch: undefined }
    },
    { name: "null flag", payload: { tldw_history_branch: null } },
    { name: "string flag", payload: { tldw_history_branch: "false" } },
    { name: "assistant settlement", payload: { role: "assistant" } },
    {
      name: "foreign conversation",
      payload: {
        tldw_history_selection_v1: { ...selection, conversation_id: "foreign" }
      }
    },
    { name: "no selection", payload: { tldw_history_selection_v1: undefined } },
    {
      name: "malformed messages",
      payload: { tldw_history_selection_v1: { ...selection, messages: [null] } }
    },
    {
      name: "sparse messages",
      payload: {
        tldw_history_selection_v1: { ...selection, messages: Array(1) }
      },
      error: {
        ...refusal(),
        details: {
          detail: { ...refusal().details.detail, parent_message_id: null }
        }
      }
    },
    { name: "no captured request scope", unscoped: true },
    { name: "transport error", error: new Error("connection lost") },
    {
      name: "unrelated409",
      error: { status: 409, details: { detail: { code: "version_conflict" } } }
    },
    {
      name: "foreign receipt",
      error: {
        ...refusal(),
        details: {
          detail: { ...refusal().details.detail, conversation_id: "foreign" }
        }
      }
    },
    {
      name: "wrong parent receipt",
      error: {
        ...refusal(),
        details: {
          detail: {
            ...refusal().details.detail,
            parent_message_id: "other-parent"
          }
        }
      }
    },
    {
      name: "missing revision receipt",
      error: {
        ...refusal(),
        details: {
          detail: { ...refusal().details.detail, history_version: undefined }
        }
      }
    }
  ])(
    "does not turn $name into an unchanged acknowledgement",
    async (change) => {
      const {
        resetServerChatSaveStatus,
        beginServerChatWrite,
        getServerChatSaveStatus
      } = await import("@/store/server-chat-save-status")
      resetServerChatSaveStatus()
      beginServerChatWrite("owned")("saved")
      const error = change.error ?? refusal()
      mocks.bgRequest.mockReset().mockRejectedValueOnce(error)
      await expect(
        chatRagMethods.addChatMessage.call(
          client,
          "owned",
          {
            role: "user",
            content: "Input",
            tldw_history_branch: false,
            tldw_history_selection_v1: selection,
            ...change.payload
          },
          change.unscoped ? undefined : options
        )
      ).rejects.toBe(error)
      expect(getServerChatSaveStatus("owned")).toBe("failed")
    }
  )
  it("a definite refusal keeps a newer overlapping failure rather than restoring a saved snapshot", async () => {
    const {
      resetServerChatSaveStatus,
      beginServerChatWrite,
      getServerChatSaveStatus
    } = await import("@/store/server-chat-save-status")
    resetServerChatSaveStatus()
    beginServerChatWrite("owned")("saved")
    const error = refusal()
    mocks.bgRequest.mockReset().mockImplementationOnce(async () => {
      beginServerChatWrite("owned")("failed")
      throw error
    })
    await expect(
      chatRagMethods.addChatMessage.call(
        client,
        "owned",
        {
          role: "user",
          content: "Input",
          tldw_history_branch: false,
          tldw_history_selection_v1: selection
        },
        options
      )
    ).rejects.toBe(error)
    expect(getServerChatSaveStatus("owned")).toBe("failed")
  })
})


describe("protected Edit save-status evidence", () => {
  it.each([
    "original",
    "foreign",
    "transport",
    "malformed",
    "missing precondition",
    "retired intent"
  ])("qualifies only its captured %s no-admission receipt", async (outcome) => {
    const {
      resetServerChatSaveStatus,
      beginServerChatWrite,
      getServerChatSaveStatus
    } = await import("@/store/server-chat-save-status")
    resetServerChatSaveStatus()
    beginServerChatWrite("owned")("saved")
    const target = {
      version: 1,
      message_id: "target",
      message_revision: "original"
    }
    const selection = {
      conversation_id: "owned",
      cursor: { kind: "before_message", message_id: "target" },
      selection_digest: "selection",
      messages: [{ id: "parent" }]
    }
    const refusal =
      outcome === "transport"
        ? new Error("disconnected")
        : {
            status: 409,
            details: {
              detail: {
                status: "stale_selection",
                code: "history_edit_target_changed",
                conversation_id: "owned",
                message_id: "target",
                message_revision: outcome === "foreign" ? "later" : "original",
                selection_digest: "selection"
              }
            }
          }
    let release!: () => void, entered!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const ready = new Promise<void>((resolve) => {
      entered = resolve
    })
    mocks.bgRequest.mockImplementationOnce(async () => {
      entered()
      await held
      throw refusal
    })
    const client = {
      invalidateChatMessagesCache: vi.fn()
    } as unknown as TldwApiClientCore
    const pending = chatRagMethods.addChatMessage.call(
      client,
      "owned",
      {
        role: "user",
        content: "Edited input",
        tldw_history_branch: true,
        tldw_history_selection_v1: selection,
        ...(outcome === "missing precondition"
          ? {}
          : {
              tldw_history_edit_v1:
                outcome === "malformed" ? { ...target, version: true } : target
            })
      },
      {
        requestScope: {
          config: { serverUrl: "https://original.test", authMode: "multi-user" },
          userId: "alice"
        }
      }
    )
    void pending.catch(() => {})
    await ready
    if (outcome === "retired intent") {
      target.message_revision = "later"
      selection.selection_digest = "later"
    }
    release()
    await expect(pending).rejects.toBe(refusal)
    expect(getServerChatSaveStatus("owned")).toBe(
      outcome === "original" || outcome === "retired intent" ? "saved" : "failed"
    )
  })
})
