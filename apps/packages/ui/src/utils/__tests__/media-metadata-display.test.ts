import { describe, expect, it } from "vitest"
import { stripMediaMetadata } from "../media-metadata-display"

describe("stripMediaMetadata", () => {
  it("removes a valid leading ingestion envelope and keeps the article body", () => {
    const metadata = { url: "https://example.com/", ingestion_date: "2026-10-05", content_hash: "fixture", scraping_pipeline: "web_scraping_v2" }
    expect(stripMediaMetadata(`  [METADATA]\n${JSON.stringify(metadata, null, 2)}\n[/METADATA]\n\nArticle body`)).toBe("Article body")
  })

  it("handles nested metadata and closing-marker text inside JSON strings", () => {
    const metadata = { title: 'A } [/METADATA] \"quoted\" title', extra: { tags: ["research"] } }
    expect(stripMediaMetadata(`[METADATA]${JSON.stringify(metadata)}[/METADATA] Body`)).toBe("Body")
  })

  it.each([
    "Ordinary article mentions [METADATA] and [/METADATA].",
    '[METADATA]{"title":"unterminated}[/METADATA]Body',
    '[METADATA]{"title":"valid"}Body',
    '[METADATA][][/METADATA]Body',
    '[METADATA]{broken}[/METADATA]Body',
    '[METADATA]{"nested":[}[/METADATA]Body',
    '[METADATA]{"title":"valid"} trailing text [/METADATA]Body'
  ])("preserves non-envelope or malformed source text: %s", content => {
    expect(stripMediaMetadata(content)).toBe(content)
  })

  it("only strips the first envelope, preserving metadata examples in the article", () => {
    const body = 'Example: [METADATA]{}[/METADATA]'
    expect(stripMediaMetadata(`[METADATA]{}[/METADATA]\n${body}`)).toBe(body)
  })
})
