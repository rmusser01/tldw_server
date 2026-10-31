import { describe, expect, it } from "vitest"
import {
  consumeStreamingChunk,
  extractStreamingChunkError
} from "../streaming-chunks"

describe("streaming chunk error handling", () => {
  it("extracts error details from explicit error payloads", () => {
    const details = extractStreamingChunkError({
      error: {
        code: "internal_error",
        message: "pony-alpha is not a valid model ID"
      }
    })

    expect(details).toEqual({
      code: "internal_error",
      message: "pony-alpha is not a valid model ID"
    })
  })

  it("treats event:error payloads as stream failures", () => {
    const details = extractStreamingChunkError({
      event: "error",
      message: "Selected model is unavailable"
    })

    expect(details).toEqual({
      code: undefined,
      message: "Selected model is unavailable"
    })
  })

  it("throws with provider message when stream emits an error chunk", () => {
    expect(() =>
      consumeStreamingChunk(
        { fullText: "", contentToSave: "", apiReasoning: false },
        { error: { code: "internal_error", message: "invalid model" } }
      )
    ).toThrowError("invalid model")
  })

  it("does not treat normal completion chunks as errors", () => {
    const state = consumeStreamingChunk(
      { fullText: "", contentToSave: "", apiReasoning: false },
      {
        choices: [{ delta: { content: "Hello" } }]
      }
    )

    expect(state.fullText).toBe("Hello")
    expect(state.contentToSave).toBe("Hello")
    expect(state.token).toBe("Hello")
  })
})


describe("reasoning and answer stream boundaries", () => {
  const empty = { fullText: "", contentToSave: "", apiReasoning: false }

  it("keeps reasoning open through empty metadata until an actual answer arrives", () => {
    let state = consumeStreamingChunk(empty, { choices: [{ delta: { reasoning_content: "First thought" } }] })
    for (const chunk of [{ choices: [{ delta: {} }] }, { choices: [] }, ""]) {
      state = consumeStreamingChunk(state, chunk)
      expect(state.fullText).toBe("<think>First thought")
      expect(state.apiReasoning).toBe(true)
    }
    state = consumeStreamingChunk(state, { choices: [{ delta: { reasoning_content: " then more", content: "Answer" } }] })
    expect(state.fullText).toBe("<think>First thought then more</think>Answer")
    expect(state.contentToSave).toBe(state.fullText)
    expect(state.apiReasoning).toBe(false)
  })

  it("starts a later reasoning segment without moving an earlier answer into it", () => {
    let state = consumeStreamingChunk(empty, { choices: [{ delta: { reasoning_content: "First" } }] })
    state = consumeStreamingChunk(state, "Answer one")
    state = consumeStreamingChunk(state, { additional_kwargs: { reasoning_content: "Second" } })
    state = consumeStreamingChunk(state, { choices: [{ delta: { content: "Answer two" } }] })
    expect(state.fullText).toBe("<think>First</think>Answer one<think>Second</think>Answer two")
    expect(state.contentToSave).toBe(state.fullText)
  })

  it("keeps ordinary strings exact and creates no empty reasoning segment", () => {
    let state = consumeStreamingChunk(empty, " Answer ")
    state = consumeStreamingChunk(state, { choices: [{ delta: { reasoning_content: "", content: " text\n" } }] })
    expect(state).toEqual({ fullText: " Answer  text\n", contentToSave: " Answer  text\n", apiReasoning: false, token: " text\n" })
  })
})
