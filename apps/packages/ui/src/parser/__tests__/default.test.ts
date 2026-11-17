import { describe, expect, it } from "vitest"
import { defaultExtractContent } from "@/parser/default"

describe("defaultExtractContent fallback attribute stripping", () => {
  it("strips every attribute except href and src before markdown conversion", () => {
    const html = [
      "<html><body>",
      "<main>",
      '<h1 class="title" id="headline" data-track="hover">Headline</h1>',
      '<a href="https://example.com/story" title="Link title" class="btn" data-track="click" onclick="track()">Story link</a>',
      '<img src="https://example.com/pic.png" alt="Alt text" class="thumb" loading="lazy">',
      '<p style="color: red" data-testid="body">Body copy.</p>',
      "</main>",
      "</body></html>"
    ].join("")

    const markdown = defaultExtractContent(html)

    // The preserved set is exactly {href, src}: links and images survive.
    expect(markdown).toContain("[Story link](https://example.com/story)")
    expect(markdown).toContain("](https://example.com/pic.png)")
    // Every other attribute is stripped before Turndown runs: a surviving
    // link title would render as [text](url "Link title") and a surviving
    // image alt as ![Alt text](src).
    expect(markdown).not.toContain("Link title")
    expect(markdown).not.toContain("Alt text")
    // Content under stripped attributes still converts.
    expect(markdown).toContain("# Headline")
    expect(markdown).toContain("Body copy.")
  })
})
