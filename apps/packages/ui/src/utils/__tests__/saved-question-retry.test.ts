import { describe, expect, it } from "vitest"
import type { HistorySelectionCaptureV1, NativeCompletionCapabilityV1 } from "@/types/history-selection"
import { encodeChatErrorPayload } from "@/utils/chat-error-message"
import {
  existingInputCurrentMessages,
  INSTRUCTIONS_UNSUPPORTED,
  isPlainSavedQuestion,
  plainSavedQuestionClientManaged,
  plainSavedQuestionRefusalMessage,
  SAVED_QUESTION_CLIENT_MANAGED,
  savedQuestionChatKindOf,
  savedQuestionReviewOffered
} from "@/utils/saved-question-retry"

type Row = { id: string; role: string; message: string }

/** A captured view ending at an unanswered question, after `earlier` rows. */
const capture = (native?: Partial<NativeCompletionCapabilityV1> | null, earlier: Row[] = []) => {
  const rows = [...earlier, { id: "q1", role: "user", message: "Saved question" }]
  return {
    status: "captured",
    rows: rows.map(({ id, role }) => ({ id, role, revision: "1", parent_id: null, settled: true })),
    selected_content: rows.map(({ id, message }) => ({ id, revision: "1", message, images: [] })),
    ...(native ? { native_completion_v1: { policy: "selected_history_v1", supported: true, storage_context_digest: "s", ...native } } : {})
  } as unknown as HistorySelectionCaptureV1
}

const t = (_key: string, fallback: string) => fallback

describe("saved-question Retry rules (8B)", () => {
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
    expect(savedQuestionChatKindOf({ serverChatId: "chat-1", ...state }, "chat-1")).toBe(kind)
    expect(savedQuestionChatKindOf({ serverChatId: "chat-1", ...state }, "another-chat")).toBeNull()
  })

  describe("which plain chats stay client-settled (like an ordinary send)", () => {
    const error = encodeChatErrorPayload({ summary: "Provider failed", hint: "Try again", detail: "" })
    it.each([
      ["no native completion (older server)", capture(null), true],
      ["a Sync v2 owner", capture({ supported: false }), true],
      ["plain history", capture({}, [{ id: "u0", role: "user", message: "Hi" }, { id: "a0", role: "assistant", message: "Hello" }]), false],
      ["a saved system row", capture({}, [{ id: "s0", role: "system", message: "Saved instruction" }]), true],
      ["a saved error reply", capture({}, [{ id: "u0", role: "user", message: "Hi" }, { id: "a0", role: "assistant", message: error }]), true],
      ["reasoning the server can't keep", capture({}, [{ id: "a0", role: "assistant", message: "<think>plan</think>Answer" }]), true],
      ["reasoning the server keeps", capture({ reasoning_history_supported: true }, [{ id: "a0", role: "assistant", message: "<think>plan</think>Answer" }]), false]
    ])("%s → client-settled: %s", (_label, captured, expected) => {
      expect(plainSavedQuestionClientManaged(captured)).toBe(expected)
    })
  })

  it.each([
    ["character", capture(null), true],
    ["plain", capture({}), true],
    ["plain", capture(null), false],
    ["plain", capture({}, [{ id: "s0", role: "system", message: "Saved instruction" }]), false],
    ["persona", capture({}), false],
    [null, capture({}), false]
  ] as const)("offers the review for a %s chat: %s", (kind, captured, offered) => {
    expect(savedQuestionReviewOffered(kind, captured)).toBe(offered)
  })

  describe("existingInputCurrentMessages", () => {
    it("sends nothing for a character Retry, whatever the server says", () => {
      expect(existingInputCurrentMessages(capture(null), {})).toEqual([])
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
      expect(() => existingInputCurrentMessages(capture({}), { instructions: ["Be brief."] })).toThrow(INSTRUCTIONS_UNSUPPORTED)
    })

    it("refuses a plain Retry whose history must stay client-settled", () => {
      const captured = capture({ existing_input_system_messages_supported: true }, [{ id: "s0", role: "system", message: "Saved" }])
      expect(() => existingInputCurrentMessages(captured, { instructions: [] })).toThrow(SAVED_QUESTION_CLIENT_MANAGED)
    })

    it.each([
      ["without native_completion_v1 (older server)", capture(null)],
      ["for a Sync v2 owner (supported: false)", capture({ supported: false, existing_input_system_messages_supported: true })]
    ])("keeps a plain Retry client-managed %s", (_label, captured) => {
      expect(() => existingInputCurrentMessages(captured, { instructions: [] })).toThrow("unsupported_history_capability")
    })
  })

  describe("plain-chat refusal copy", () => {
    it("names the setting a plain Retry can't use, without character wording", () => {
      const copy = plainSavedQuestionRefusalMessage("native_history_setting_unsupported:topK", t)
      expect(copy).toBe("topK can't be used when retrying a saved question. Clear it in the model settings and retry; your question is kept.")
    })

    it.each([INSTRUCTIONS_UNSUPPORTED, SAVED_QUESTION_CLIENT_MANAGED, "retry_context_changed", "native_history_client_tools_unsupported",
      "native_history_explicit_model_provider_required"])("has plain copy for %s", (code) => {
      const copy = plainSavedQuestionRefusalMessage(code, t)
      expect(copy).toEqual(expect.any(String))
      expect(copy).not.toMatch(/character/i)
    })

    it("points to the global settings for the default prompt and formatting guide", () => {
      expect(plainSavedQuestionRefusalMessage(INSTRUCTIONS_UNSUPPORTED, t)).toMatch(/default system prompt and output formatting guide in Settings/)
    })

    it("has no copy for an unknown reason", () => {
      expect(plainSavedQuestionRefusalMessage("secret-token=do-not-display", t)).toBeNull()
    })
  })
})
