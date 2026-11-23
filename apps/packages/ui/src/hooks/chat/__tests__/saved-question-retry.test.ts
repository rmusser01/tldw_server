import { afterEach, describe, expect, it, vi } from "vitest"
import {
  existingInputCurrentMessages,
  INSTRUCTIONS_UNSUPPORTED,
  isPlainSavedQuestion,
  savedQuestionChatKind,
  submitSavedQuestionRetry
} from "@/hooks/chat/saved-question-retry"
import { useStoreMessageOption } from "@/store/option"
import type { HistorySelectionCaptureV1, NativeCompletionCapabilityV1 } from "@/types/history-selection"

const capture = (native?: Partial<NativeCompletionCapabilityV1>) =>
  ({
    status: "captured",
    ...(native
      ? { native_completion_v1: { policy: "selected_history_v1", supported: true, storage_context_digest: "s", ...native } }
      : {})
  }) as unknown as HistorySelectionCaptureV1

const review = (overrides: Record<string, unknown> = {}) => ({
  id: "q1",
  revision: "1",
  text: "Saved question",
  model: "provider:model",
  isCurrent: () => true,
  instructionsSupported: true,
  ...overrides
})

describe("saved-question Retry rules (8B)", () => {
  const initial = useStoreMessageOption.getState()
  afterEach(() => {
    useStoreMessageOption.setState({
      serverChatId: initial.serverChatId,
      serverChatAssistantKind: initial.serverChatAssistantKind,
      serverChatCharacterId: initial.serverChatCharacterId
    })
  })

  it("tells a plain review (instructions, possibly none) from a character review", () => {
    expect(isPlainSavedQuestion({ instructions: [] })).toBe(true)
    expect(isPlainSavedQuestion({ instructions: ["Be brief."] })).toBe(true)
    expect(isPlainSavedQuestion({})).toBe(false)
    expect(isPlainSavedQuestion(undefined)).toBe(false)
  })

  it.each([
    [{ serverChatAssistantKind: null, serverChatCharacterId: null }, "plain"],
    [{ serverChatAssistantKind: "character", serverChatCharacterId: "7" }, "character"],
    [{ serverChatAssistantKind: null, serverChatCharacterId: "7" }, "character"],
    [{ serverChatAssistantKind: "persona", serverChatCharacterId: null }, "persona"]
  ] as const)("classifies the open chat %j as %s", (state, kind) => {
    useStoreMessageOption.setState({ serverChatId: "chat-1", ...state })
    expect(savedQuestionChatKind("chat-1")).toBe(kind)
    expect(savedQuestionChatKind("another-chat")).toBeNull()
  })

  describe("existingInputCurrentMessages", () => {
    it("sends nothing for a character Retry, whatever the server says", () => {
      expect(existingInputCurrentMessages(capture(), {})).toEqual([])
    })

    it("sends the reviewed instructions as system-only messages when the server accepts them", () => {
      expect(
        existingInputCurrentMessages(capture({ existing_input_system_messages_supported: true }), {
          instructions: ["Default instruction", "Answer like a pirate."]
        })
      ).toEqual([
        { role: "system", content: "Default instruction" },
        { role: "system", content: "Answer like a pirate." }
      ])
    })

    it("retries with no messages when no instruction is active, even on an older server", () => {
      expect(existingInputCurrentMessages(capture({}), { instructions: [] })).toEqual([])
    })

    it("refuses an active instruction on a server that doesn't accept it", () => {
      expect(() => existingInputCurrentMessages(capture({}), { instructions: ["Be brief."] })).toThrow(
        INSTRUCTIONS_UNSUPPORTED
      )
    })

    it.each([
      ["without native_completion_v1 (older server)", capture()],
      ["for a Sync v2 owner (supported: false)", capture({ supported: false, existing_input_system_messages_supported: true })]
    ])("keeps a plain Retry client-managed %s", (_label, captured) => {
      expect(() => existingInputCurrentMessages(captured, { instructions: [] })).toThrow("unsupported_history_capability")
    })
  })

  describe("submitSavedQuestionRetry", () => {
    it("submits a plain review once its instructions are still the current ones", async () => {
      const reviewed = review({ instructions: ["Be brief."] })
      const submit = vi.fn(async () => ({ status: "submitted" as const }))
      const result = await submitSavedQuestionRetry(reviewed, async () => ({ systemPrompts: ["Be brief."] }), submit)
      expect(result).toEqual({ status: "submitted" })
      expect(submit).toHaveBeenCalledWith({ message: "Saved question", image: "", historyServerExistingInput: reviewed })
    })

    it("refuses a plain review whose instructions changed since it was shown", async () => {
      const submit = vi.fn()
      const result = await submitSavedQuestionRetry(
        review({ instructions: ["Be brief."] }),
        async () => ({ systemPrompts: ["An edited default instruction"] }),
        submit
      )
      expect(result).toMatchObject({ status: "failed", errorMessage: "retry_context_changed" })
      expect(submit).not.toHaveBeenCalled()
    })

    it("refuses before reading anything when the server can't carry the active instructions", async () => {
      const getContext = vi.fn()
      const submit = vi.fn()
      const result = await submitSavedQuestionRetry(
        review({ instructions: ["Be brief."], instructionsSupported: false }),
        getContext,
        submit
      )
      expect(result).toMatchObject({ status: "failed", errorMessage: INSTRUCTIONS_UNSUPPORTED })
      expect(getContext).not.toHaveBeenCalled()
      expect(submit).not.toHaveBeenCalled()
    })

    it("allows a plain review with no instructions on an older server", async () => {
      const submit = vi.fn(async () => ({ status: "submitted" as const }))
      const result = await submitSavedQuestionRetry(
        review({ instructions: [], instructionsSupported: false }),
        async () => ({ systemPrompts: [] }),
        submit
      )
      expect(result).toEqual({ status: "submitted" })
    })

    it("reports a prompt that can no longer be read", async () => {
      const submit = vi.fn()
      const result = await submitSavedQuestionRetry(review({ instructions: [] }), async () => {
        throw new Error("retry_prompt_unavailable")
      }, submit)
      expect(result).toMatchObject({ status: "failed", errorMessage: "retry_prompt_unavailable" })
      expect(submit).not.toHaveBeenCalled()
    })

    it("refuses a review the conversation has moved on from", async () => {
      const submit = vi.fn()
      const result = await submitSavedQuestionRetry(review({ isCurrent: () => false }), async () => ({ systemPrompts: [] }), submit)
      expect(result).toMatchObject({ status: "failed", errorMessage: "history_selection_changed" })
      expect(submit).not.toHaveBeenCalled()
    })

    it("submits a character review without reading the current instructions", async () => {
      const getContext = vi.fn()
      const submit = vi.fn(async () => undefined)
      const result = await submitSavedQuestionRetry(review(), getContext, submit)
      expect(getContext).not.toHaveBeenCalled()
      expect(submit).toHaveBeenCalledOnce()
      expect(result).toMatchObject({ status: "failed", errorMessage: "native_history_existing_input_not_submitted" })
    })
  })
})

describe("send-time refusal copy (8B)", () => {
  it("explains a refused instruction in plain words and keeps the question", async () => {
    const { nativeCharacterPreflightMessage } = await import("@/hooks/chat/native-character-preflight")
    const t = (_key: string, fallback: string) => fallback
    expect(nativeCharacterPreflightMessage(new Error(INSTRUCTIONS_UNSUPPORTED), t)).toBe(
      "This server can't keep your System/Prompt instructions on a retry. Turn them off in the chat settings and review the saved question again, or update the server. Your question is kept."
    )
  })
})
