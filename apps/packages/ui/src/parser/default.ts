import * as cheerio from "cheerio"
import TurndownService from "turndown"
import { Readability, isProbablyReaderable } from "@mozilla/readability"

// Attributes kept through the fallback strip: everything else is removed in
// a single traversal before the markdown conversion. The exact set is pinned
// by `src/parser/__tests__/default.test.ts`.
const PRESERVED_ATTRIBUTES: ReadonlySet<string> = new Set(["href", "src"])

export const defaultExtractContent = (html: string) => {
  const doc = new DOMParser().parseFromString(html, "text/html")
  if (isProbablyReaderable(doc)) {
    const reader = new Readability(doc)
    const article = reader.parse()
    if (article && article.content) {
      const $article = cheerio.load(article.content)
      $article("script, style, link, svg, [src^='data:image/']").remove()
      article.content = $article.html() || ""
    }
    const turndownService = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced"
    })
    return turndownService.turndown(article?.content || "").trim()
  }

  const $ = cheerio.load(html)

  $("script, style, link, svg, [src^='data:image/']").remove()

  // Single-pass strip: delete every non-preserved attribute directly from
  // each element's `attribs` map instead of a `removeAttr` round-trip per
  // attribute (cheerio's removeAttr does the same `delete` internally).
  $("*").each((_, element) => {
    if (!("attribs" in element)) return
    const { attribs } = element
    for (const attribute of Object.keys(attribs)) {
      if (!PRESERVED_ATTRIBUTES.has(attribute)) {
        delete attribs[attribute]
      }
    }
  })

  const mainContent =
    $('[role="main"]').html() || $("main").html() || $("body").html() || ""

  const turndownService = new TurndownService({
    headingStyle: "atx",
    codeBlockStyle: "fenced"
  })
  const markdown = turndownService.turndown(mainContent)

  return markdown.trim()
}
