import { describe, expect, it } from "vitest"
import {
  resolveSavedGenerationMetadata,
  resolveMessageCostUsd,
  resolveMessageUsage
} from "../message-usage"

describe("message-usage", () => {
  it("resolves prompt/completion/total tokens from mixed generation payloads", () => {
    expect(
      resolveMessageUsage({
        prompt_eval_count: 120,
        eval_count: 55,
        usage: { total_tokens: 180 }
      })
    ).toEqual({
      promptTokens: 120,
      completionTokens: 55,
      totalTokens: 180
    })
  })

  it("falls back to usage fields when top-level values are absent", () => {
    expect(
      resolveMessageUsage({
        usage: {
          prompt_tokens: 80,
          completion_tokens: 20
        }
      })
    ).toEqual({
      promptTokens: 80,
      completionTokens: 20,
      totalTokens: 100
    })
  })

  it("resolves estimated cost from common cost fields", () => {
    expect(
      resolveMessageCostUsd({
        usage: {
          estimated_cost_usd: 0.0042
        }
      })
    ).toBe(0.0042)
    expect(
      resolveMessageCostUsd({
        pricing: {
          total_cost_usd: 0.0011
        }
      })
    ).toBe(0.0011)
  })
})


describe("saved generation metadata presentation", () => {
  it("reads only recorded generation fields without copying private recovery or cost flags", () => {
    const extra = {
      generation_status: "interrupted",
      model_id: " saved-model ",
      provider: "openai",
      finish_reason: "length",
      usage: {
        prompt_tokens: 11,
        completion_tokens: 2,
        total_tokens: 13,
        cost_usd: 99
      },
      interrupted: true,
      stopped: true,
      canRetry: true,
      api_key: "synthetic-secret"
    }
    const before = structuredClone(extra)
    expect(resolveSavedGenerationMetadata(extra)).toEqual({
      status: "interrupted",
      model: "saved-model",
      provider: "openai",
      finishReason: "length",
      usage: { prompt_tokens: 11, completion_tokens: 2, total_tokens: 13 }
    })
    expect(extra).toEqual(before)
  })
  it.each(["complete", "length", "interrupted", "stopped", "error"])(
    "keeps documented status %s as presentation",
    (status) => {
      expect(
        resolveSavedGenerationMetadata({ generation_status: status })
      ).toEqual({ status })
    }
  )
  it.each([
    null,
    [],
    "complete",
    { generation_status: "pending" },
    { model_id: "x".repeat(257) },
    { model_id: "bad\nmodel" },
    { provider: "bad provider" },
    { finish_reason: "x".repeat(65) }
  ])("ignores malformed or unrecognized metadata %#", (value) => {
    expect(resolveSavedGenerationMetadata(value)).toBeNull()
  })
  it("preserves zero and ignores invalid or unsafe reported token counts", () => {
    expect(
      resolveSavedGenerationMetadata({
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      })
    ).toEqual({
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    })
    expect(
      resolveSavedGenerationMetadata({
        usage: {
          prompt_tokens: -1,
          completion_tokens: Infinity,
          total_tokens: "13"
        }
      })
    ).toBeNull()
    expect(
      resolveSavedGenerationMetadata({
        usage: { prompt_tokens: 1.5, total_tokens: 2 ** 53 + 2 }
      })
    ).toBeNull()
  })
  it("keeps printable Unicode model names and the server token-count bound", () => {
    expect(
      resolveSavedGenerationMetadata({
        model_id: "模型/guide",
        usage: { total_tokens: 2 ** 53 }
      })
    ).toEqual({ model: "模型/guide", usage: { total_tokens: 2 ** 53 } })
  })
})
