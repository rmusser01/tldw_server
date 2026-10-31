import { describe, expect, it } from "vitest"

import { DEFAULT_RAG_SETTINGS, buildRagSearchRequest } from "../unified-rag"

describe("buildRagSearchRequest scope filters", () => {
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
