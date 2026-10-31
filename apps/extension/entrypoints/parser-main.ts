import { defineUnlistedScript } from "wxt/utils/define-unlisted-script"
import { defaultExtractContent } from "@tldw/ui/parser/default"

/**
 * Web-accessible lazily-loaded chunk that carries the heavy default parser
 * (cheerio + @mozilla/readability + turndown). The every-page web-clipper
 * content script `import()`s this chunk on first capture request so none of
 * that weight ships inside the per-page IIFE bundle.
 *
 * Loaded via `import(browser.runtime.getURL("parser-main.js"))` — see
 * `packages/ui/src/services/web-clipper/content-extract.ts`.
 */
export default defineUnlistedScript(() => {
  ;(globalThis as Record<string, unknown>).__tldwParserDefaultExtractContent =
    defaultExtractContent
})
