import { describe, expect, it } from "vitest"

import {
  extractMediaDetailAnalysis,
  extractMediaDetailContent
} from "../media-detail-content"

describe("extractMediaDetailContent", () => {
  const source = "  indented\ttext\n\n\nFinal line\n"

  it.each([
    ["direct response", source],
    ["string content", { content: source }],
    ["nested content", { content: { text: source } }],
    ["legacy root", { raw_text: source }],
    ["latest version", { latest_version: { content: source } }],
    ["camel-case latest version", { latestVersion: { text: source } }],
    ["data content", { data: { content: { raw_text: source } } }],
    ["data root", { data: { rawText: source } }]
  ])("preserves source whitespace from %s", (_shape, detail) => {
    expect(extractMediaDetailContent(detail)).toBe(source)
  })

  it("skips whitespace-only candidates without trimming the chosen body", () => {
    expect(extractMediaDetailContent({
      content: " \n\t",
      text: source
    })).toBe(source)
    expect(extractMediaDetailContent(" \n\t")).toBe("")
  })

  it("extracts text from nested content object", () => {
    const detail = {
      content: {
        text: "Nested content text"
      }
    }

    expect(extractMediaDetailContent(detail)).toBe("Nested content text")
  })

  it("extracts direct string content returned by the media detail API", () => {
    expect(
      extractMediaDetailContent({
        content: "Direct media detail content"
      })
    ).toBe("Direct media detail content")
  })

  it("falls back to latest_version and data object content", () => {
    const latestDetail = {
      latest_version: {
        content: {
          text: "Latest version nested text"
        }
      }
    }
    const dataDetail = {
      data: {
        content: {
          raw_text: "Data nested raw text"
        }
      }
    }

    expect(extractMediaDetailContent(latestDetail)).toBe(
      "Latest version nested text"
    )
    expect(extractMediaDetailContent(dataDetail)).toBe("Data nested raw text")
  })

  it("supports legacy flat response fields", () => {
    const detail = {
      raw_text: "Legacy root content"
    }

    expect(extractMediaDetailContent(detail)).toBe("Legacy root content")
  })

  it("returns empty string when no text-like fields exist", () => {
    expect(extractMediaDetailContent({ content: { metadata: { a: 1 } } })).toBe("")
  })
})

describe("extractMediaDetailAnalysis", () => {
  it.each([
    { processing: { analysis: "  Analysis\n" } },
    { analyses: ["  Analysis\n"] },
    { analyses: [{ content: "  Analysis\n" }] },
    { latest_version: { analysis: "  Analysis\n" } },
    { summary: "  Analysis\n" }
  ])("retains analysis normalization for %j", (detail) => {
    expect(extractMediaDetailAnalysis(detail)).toBe("Analysis")
  })

  it("extracts the persisted processing analysis returned by the media detail API", () => {
    const detail = {
      processing: {
        analysis: "Persisted analysis from the real backend"
      },
      summary: "Older summary"
    }

    expect(extractMediaDetailAnalysis(detail)).toBe(
      "Persisted analysis from the real backend"
    )
  })

  it("supports root, analysis-list, and versioned response shapes", () => {
    expect(extractMediaDetailAnalysis({ analysis_content: "Root analysis" })).toBe(
      "Root analysis"
    )
    expect(
      extractMediaDetailAnalysis({ analyses: [{ text: "Analysis list entry" }] })
    ).toBe("Analysis list entry")
    expect(
      extractMediaDetailAnalysis({ latest_version: { analysis: "Version analysis" } })
    ).toBe("Version analysis")
  })
})
