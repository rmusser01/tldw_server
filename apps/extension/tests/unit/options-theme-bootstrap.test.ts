import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const extensionRoot = path.resolve(__dirname, "..", "..")
const optionsHtmlPath = path.join(
  extensionRoot,
  "entrypoints",
  "options",
  "index.html",
)
const themeBootstrapPath = path.resolve(
  extensionRoot,
  "../packages/ui/src/public/theme-bootstrap.js",
)
const optionsHtml = readFileSync(optionsHtmlPath, "utf8")

describe("options theme bootstrap", () => {
  it("loads a same-origin external classic script synchronously from the head", () => {
    const head = optionsHtml.match(/<head\b[^>]*>([\s\S]*?)<\/head>/i)?.[1] ?? ""

    expect(head).toContain('<script src="/theme-bootstrap.js"></script>')
  })

  it("contains no executable inline script body", () => {
    const inlineScriptBodies = [...optionsHtml.matchAll(
      /<script\b([^>]*)>([\s\S]*?)<\/script>/gi,
    )]
      .filter(([, attributes, body]) => !/\bsrc\s*=/i.test(attributes) && body.trim())
      .map(([, , body]) => body.trim())

    expect(inlineScriptBodies).toEqual([])
  })

  it("ships the referenced public script", () => {
    expect(existsSync(themeBootstrapPath)).toBe(true)
  })
})
