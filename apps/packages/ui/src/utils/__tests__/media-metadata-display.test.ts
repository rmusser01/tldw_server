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
    ['[METADATA]{}[/METADATA]\n\n    print("hello")', '    print("hello")'],
    ['[METADATA]{}[/METADATA]\r\n\r\n    print("hello")  \r\n\t ', '    print("hello")  \r\n\t '],
    ['[METADATA]{}[/METADATA]\n\nArticle body  \n\n', 'Article body  \n\n'],
    ['[METADATA]{}[/METADATA]    print("hello")', '    print("hello")'],
    ['[METADATA]{}[/METADATA]\tprint("hello")', '\tprint("hello")'],
    ['[METADATA]{}[/METADATA]\n\n\n    print("hello")', '\n    print("hello")'],
    ['[METADATA]{}[/METADATA]\r\n\r\n\r\n\tprint("hello")', '\r\n\tprint("hello")']
  ])("preserves Markdown body whitespace after a valid envelope: %j", (content, body) => {
    expect(stripMediaMetadata(content)).toBe(body)
  })

  it.each([
    "Article body",
    '    print("hello")',
    '\tprint("hello")',
    '\n\n    print("hello")  \r\n\t ',
    '        print("hello")',
    " Article body",
    ""
  ])("removes only the exact legacy writer prefix and preserves its body: %j", body => {
    const metadata = { url: "https://example.com/" }
    const content = `[METADATA]\n        ${JSON.stringify(metadata, null, 2)}\n        [/METADATA]\n\n        ${body}`
    expect(stripMediaMetadata(content)).toBe(body)
  })

  it.each([
    '[METADATA]\n{}\n        [/METADATA]\n\n        Article body',
    '[METADATA]\n        {}\n[/METADATA]\n\n        Article body',
    '[METADATA]\n        {}\n        [/METADATA]\n        Article body'
  ])("preserves eight body spaces when the exact writer layout does not match: %j", content => {
    expect(stripMediaMetadata(content)).toBe("        Article body")
  })

  it.each([
    "",
    "Ordinary article mentions [METADATA] and [/METADATA].",
    '\n    print("hello")  \r\n\t ',
    '  [METADATA]{broken}[/METADATA]\r\n\r\n    print("hello")  \r\n\t ',
    '[METADATA]{"title":"unterminated}[/METADATA]Body',
    '[METADATA]{"title":"valid"}Body',
    '[METADATA][][/METADATA]Body',
    '[METADATA]{broken}[/METADATA]Body',
    '[METADATA]\n        {broken}\n        [/METADATA]\n\n        Article body',
    '[METADATA]{"nested":[}[/METADATA]Body',
    '[METADATA]{"title":"valid"} trailing text [/METADATA]Body'
  ])("preserves non-envelope or malformed source text: %s", content => {
    expect(stripMediaMetadata(content)).toBe(content)
  })

  it("returns an empty body when a valid envelope has no article content", () => {
    expect(stripMediaMetadata('[METADATA]{}[/METADATA]\n\n')).toBe("")
  })

  it("only strips the first envelope, preserving metadata examples in the article", () => {
    const body = 'Example: [METADATA]{}[/METADATA]'
    expect(stripMediaMetadata(`[METADATA]{}[/METADATA]\n${body}`)).toBe(body)
  })
})
