import { describe, expect, it } from "vitest"

import { DEFAULT_RAG_SETTINGS, buildRagSearchRequest } from "../unified-rag"

describe("buildRagSearchRequest scope filters", () => {
  it("leaves reasoning room for selected DeepSeek QA unless the user sets a token cap", () => {
    const settings = { ...DEFAULT_RAG_SETTINGS, generation_provider: "deepseek", generation_model: "deepseek-flash", include_media_ids: [1] }
    expect(buildRagSearchRequest(settings).options.max_generation_tokens).toBe(4096)
    expect(buildRagSearchRequest({ ...settings, max_generation_tokens: 800 }).options.max_generation_tokens).toBe(800)
  })
  it("forwards the deadline to the server as well as the transport", () => {
    const request = buildRagSearchRequest({ ...DEFAULT_RAG_SETTINGS, timeout_seconds: 30 })
    expect(request.options.timeout_seconds).toBe(30)
    expect(request.timeoutMs).toBe(30000)
  })
  it.each(["llama", "llamacpp", "llama_cpp", "llama-cpp"])(
    "normalizes the catalog provider %s while preserving the selected model",
    (provider) => {
      const req = buildRagSearchRequest({
        ...DEFAULT_RAG_SETTINGS,
        query: "When does Project Juniper launch?",
        generation_provider: provider,
        generation_model: "local-model.gguf",
      })
      expect(req.options).toMatchObject({
        generation_provider: "llama.cpp",
        generation_model: "local-model.gguf",
      })
    }
  )
  it("preserves exact source scope filters in backend options", () => {
    const req = buildRagSearchRequest({
      ...DEFAULT_RAG_SETTINGS,
      query: "Only selected docs",
      sources: ["media_db", "notes"],
      include_media_ids: [42],
      include_note_ids: ["note-a"],
      enable_web_fallback: false,
    })

    expect(req.options.include_media_ids).toEqual([42])
    expect(req.options.include_note_ids).toEqual(["note-a"])
    expect(req.options.sources).toEqual(["media_db", "notes"])
    expect(req.options.enable_web_fallback).toBe(false)
  })
})
