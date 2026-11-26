import { afterEach, describe, expect, it, vi } from "vitest"
import { savedQuestionChatKind, submitSavedQuestionRetry } from "@/hooks/chat/saved-question-retry"
import { nativeCharacterPreflightMessage } from "@/hooks/chat/native-character-preflight"
import { useStoreMessageOption } from "@/store/option"
import {
  INSTRUCTIONS_UNSUPPORTED,
  PlainSavedQuestionRefusal,
  SAVED_QUESTION_RETRY_IN_PROGRESS
} from "@/utils/saved-question-retry"

const review = (overrides: Record<string, unknown> = {}) => ({
  id: "q1",
  revision: "1",
  text: "Saved question",
  model: "provider:model",
  isCurrent: () => true,
  instructionsSupported: true,
  ...overrides
})

describe("saved-question Retry submission (8B)", () => {
  const initial = useStoreMessageOption.getState()
  afterEach(() => {
    useStoreMessageOption.setState({
      serverChatId: initial.serverChatId,
      serverChatAssistantKind: initial.serverChatAssistantKind,
      serverChatCharacterId: initial.serverChatCharacterId
    })
  })

  it("reads the open chat's kind from the chat store", () => {
    useStoreMessageOption.setState({ serverChatId: "chat-1", serverChatAssistantKind: null, serverChatCharacterId: null })
    expect(savedQuestionChatKind("chat-1")).toBe("plain")
    useStoreMessageOption.setState({ serverChatAssistantKind: "persona" })
    expect(savedQuestionChatKind("chat-1")).toBe("persona")
    expect(savedQuestionChatKind("another-chat")).toBeNull()
  })

  it("submits a plain review once its instructions are still the current ones", async () => {
    const reviewed = review({ instructions: ["Be brief."] })
    const submit = vi.fn(async () => ({ status: "submitted" as const }))
    const result = await submitSavedQuestionRetry(reviewed, async () => ({ systemPrompts: ["Be brief."] }), submit)
    expect(result).toEqual({ status: "submitted" })
    expect(submit).toHaveBeenCalledWith({ message: "Saved question", image: "", historyServerExistingInput: reviewed })
  })

  it("answers one saved question once, even when Retry is pressed twice at once", async () => {
    let release!: () => void
    const held = new Promise<void>((resolve) => { release = resolve })
    const submit = vi.fn(async () => { await held; return { status: "submitted" as const } })
    const reviewed = review({ instructions: [] })
    const first = submitSavedQuestionRetry(reviewed, async () => ({ systemPrompts: [] }), submit)
    const second = submitSavedQuestionRetry(reviewed, async () => ({ systemPrompts: [] }), submit)
    expect(await second).toEqual({ status: "failed", errorMessage: SAVED_QUESTION_RETRY_IN_PROGRESS })
    release()
    expect(await first).toEqual({ status: "submitted" })
    expect(submit).toHaveBeenCalledOnce()
    // Once settled, a later deliberate Retry may run.
    expect(await submitSavedQuestionRetry(reviewed, async () => ({ systemPrompts: [] }), submit)).toEqual({ status: "submitted" })
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
    const result = await submitSavedQuestionRetry(review({ instructions: ["Be brief."], instructionsSupported: false }), getContext, submit)
    expect(result).toMatchObject({ status: "failed", errorMessage: INSTRUCTIONS_UNSUPPORTED })
    expect(getContext).not.toHaveBeenCalled()
    expect(submit).not.toHaveBeenCalled()
  })

  it("allows a plain review with no instructions on an older server", async () => {
    const submit = vi.fn(async () => ({ status: "submitted" as const }))
    const result = await submitSavedQuestionRetry(review({ instructions: [], instructionsSupported: false }), async () => ({ systemPrompts: [] }), submit)
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

describe("send-time refusal copy (8B)", () => {
  const t = (_key: string, fallback: string) => fallback

  it("explains a refused instruction in plain words and keeps the question", () => {
    expect(nativeCharacterPreflightMessage(new Error(INSTRUCTIONS_UNSUPPORTED), t)).toMatch(
      /^This server can't keep System\/Prompt instructions on a retry\..*Your question is kept\.$/
    )
  })

  it("uses plain-chat wording, not character wording, for a plain Retry's settings refusal", () => {
    const refused = new PlainSavedQuestionRefusal("native_history_setting_unsupported:topK")
    expect(nativeCharacterPreflightMessage(refused, t)).toBe(
      "The topK setting can't be used when retrying a saved question. Clear it in the model settings and retry; your question is kept."
    )
    // A character chat's own refusal keeps its character wording.
    expect(nativeCharacterPreflightMessage(new Error("native_history_setting_unsupported:topK"), t)).toMatch(/saved character chat/)
  })
})
