import { describe, expect, it } from "vitest"
import { markdownInlineToHtml } from "../notes-manager-utils"

// `sanitizeUrl` is private; exercise it through `markdownInlineToHtml`, which
// renders a markdown link only when the URL survives sanitization.
describe("markdownInlineToHtml URL sanitization", () => {
  it("keeps safe http(s) links", () => {
    const html = markdownInlineToHtml("[site](https://example.com)")
    expect(html).toContain('<a href="https://example.com">')
  })

  it("neutralizes javascript: links", () => {
    const html = markdownInlineToHtml("[x](javascript:alert(1))")
    expect(html).not.toContain("<a ")
    expect(html).not.toContain("javascript:")
  })

  it("neutralizes control-char obfuscated schemes (java\\tscript:)", () => {
    const html = markdownInlineToHtml("[x](java\tscript:alert(1))")
    expect(html).not.toContain("<a ")
    // The tab is stripped before scheme matching, so it is caught and dropped.
    expect(html).not.toContain("script:")
  })

  it("neutralizes newline obfuscated schemes", () => {
    const html = markdownInlineToHtml("[x](java\nscript:alert(1))")
    expect(html).not.toContain("<a ")
    expect(html).not.toContain("script:")
  })
})
