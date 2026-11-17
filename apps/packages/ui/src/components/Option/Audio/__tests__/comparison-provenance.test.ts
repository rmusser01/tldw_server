import { describe, expect, it } from "vitest"

import {
  buildSttComparisonConfig,
  buildTextPreview,
  buildTtsResultMetadata,
  formatByteSize,
  formatClientLatency,
  formatCreatedAt,
  hashTextForLocalComparison,
  normalizeSttResponse
} from "../comparison-provenance"

describe("comparison provenance helpers", () => {
  it.each([
    "[No speech detected]",
    { text: "[No speech detected]", duration: 3, segments: [{ text: "[No speech detected]" }] },
    { segments: [{ text: "[No speech detected]" }] }
  ])("B05 no speech status is not normalized as spoken words: %j", (response) => {
    const normalized = normalizeSttResponse(response)
    expect(normalized.text).toBe("")
    expect(normalized.metadata.wordCount).toBe(0)
    if (typeof response === "object" && "duration" in response) {
      expect(normalized.metadata.durationSeconds).toBe(3)
      expect(normalized.metadata.segmentCount).toBe(0)
    }
  })

  it("B05 no speech normalization preserves ordinary text that mentions the phrase", () => {
    const text = "The screen said [No speech detected], then I spoke."
    expect(normalizeSttResponse({ text }).text).toBe(text)
  })

  it("formats stable display metadata", () => {
    expect(formatByteSize(1536)).toBe("1.5 KB")
    expect(formatClientLatency(1234)).toBe("Client measured 1.2s")
    expect(formatCreatedAt("2026-03-06T14:05:09.000Z")).toBe(
      "2026-03-06 14:05:09 UTC"
    )
    expect(formatCreatedAt("2026-03-06T14:05:09.123Z")).toBe(
      "2026-03-06 14:05:09 UTC"
    )
  })

  it("creates privacy-aware text previews and local hashes", () => {
    const text = "hello ".repeat(30).trim()
    const preview = buildTextPreview(text, 24)

    expect(preview.inputTextLength).toBe(text.length)
    expect(preview.inputTextPreview).toBe("hello hello hello hello...")
    expect(preview.inputTextPreviewTruncated).toBe(true)
    expect(hashTextForLocalComparison("same text")).toBe(
      hashTextForLocalComparison("same text")
    )
    expect(hashTextForLocalComparison("same text")).toMatch(/^local-[0-9a-f]{8}$/)
  })

  it("builds STT request config from existing options", () => {
    expect(
      buildSttComparisonConfig(" whisper-large ", {
        language: " en ",
        task: " translate ",
        response_format: " verbose_json ",
        timestamp_granularities: [" word ", "segment", ""],
        segment: true
      })
    ).toEqual({
      model: "whisper-large",
      language: "en",
      task: "translate",
      responseFormat: "verbose_json",
      timestampGranularities: ["word", "segment"],
      segmentationEnabled: true
    })
  })

  it("normalizes available STT response metadata without inventing missing values", () => {
    const normalized = normalizeSttResponse({
      text: "hello world",
      language: "en",
      duration: 3.25,
      segments: [{ text: "hello" }, { text: "world" }],
      words: [{ word: "hello" }, { word: "world" }]
    })

    expect(normalized.text).toBe("hello world")
    expect(normalized.metadata).toMatchObject({
      language: "en",
      durationSeconds: 3.25,
      segmentCount: 2,
      wordCount: 2
    })

    expect(normalizeSttResponse({ transcript: "plain" }).metadata).toEqual({})
    expect(normalizeSttResponse({ text: "", duration: 0 }).metadata).toEqual({
      durationSeconds: 0
    })
  })

  it("preserves optional TTS gateway provenance without changing legacy metadata", () => {
    expect(
      buildTtsResultMetadata("hello", "2026-07-16T12:00:00.000Z", {
        requestedBackend: "gateway:company-proxy",
        actualBackend: "openrouter",
        fallbackUsed: true
      })
    ).toMatchObject({
      requestedBackend: "gateway:company-proxy",
      actualBackend: "openrouter",
      fallbackUsed: true
    })

    const legacy = buildTtsResultMetadata(
      "hello",
      "2026-07-16T12:00:00.000Z"
    )
    expect(legacy.requestedBackend).toBeUndefined()
    expect(legacy.actualBackend).toBeUndefined()
    expect(legacy.fallbackUsed).toBeUndefined()
  })
})
