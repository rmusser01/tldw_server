import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"

const entrypointDir = resolve(import.meta.dir, "../../entrypoints")

const readSource = (relative: string) =>
  readFileSync(resolve(entrypointDir, relative), "utf8")

describe("copilot content entrypoint", () => {
  test("stays a minimal top-frame stub that lazy-loads the heavy chunk", () => {
    const source = readSource("copilot-popup.content.tsx")

    expect(source).toMatch(
      /import\s*\{\s*defineContentScript\s*\}\s*from\s*["']wxt\/utils\/define-content-script["']/
    )
    // Top frame only: no all_frames registration for the every-page script.
    expect(source).not.toContain("allFrames")
    // The heavy module must NOT be bundler-analyzably imported here (that
    // would inline it into the every-page IIFE bundle).
    expect(source).not.toContain(
      'import("@tldw/ui/entries/copilot-popup.content")'
    )
    // It must load the web-accessible chunk through a runtime URL instead.
    expect(source).toContain('"copilot-popup-main.js"')
    expect(source).toMatch(/getURL\(/)
    expect(source).toMatch(/import\(/)
    // Readiness is marked synchronously at startup (E2E contract).
    expect(source).toContain("tldwCopilotPopupReady")
    // Only the popup-open message is claimed.
    expect(source).toContain("tldw:popup:open")
    expect(source).toMatch(/return Promise\.resolve\(\{ ok: true \}\)/)
  })

  test("keeps the heavy implementation in a web-accessible unlisted chunk", () => {
    const chunk = readSource("copilot-popup-main.ts")

    expect(chunk).toMatch(
      /import\s*\{\s*defineUnlistedScript\s*\}\s*from\s*["']wxt\/utils\/define-unlisted-script["']/
    )
    expect(chunk).toContain(
      'import { registerCopilotPopupHandler } from "@tldw/ui/entries/copilot-popup.content"'
    )
    expect(chunk).toContain("registerCopilotPopupHandler()")
  })

  test("keeps the heavy parser behind a web-accessible unlisted chunk", () => {
    const chunk = readSource("parser-main.ts")

    expect(chunk).toMatch(
      /import\s*\{\s*defineUnlistedScript\s*\}\s*from\s*["']wxt\/utils\/define-unlisted-script["']/
    )
    expect(chunk).toContain(
      'import { defaultExtractContent } from "@tldw/ui/parser/default"'
    )
    expect(chunk).toContain("__tldwParserDefaultExtractContent")
  })
})
