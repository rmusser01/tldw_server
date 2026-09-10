import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { runInNewContext } from "node:vm"
import { load } from "cheerio"
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
const themeBootstrap = readFileSync(themeBootstrapPath, "utf8")

function inlineScriptBodies(html: string): string[] {
  const document = load(html)
  return document("script:not([src])")
    .toArray()
    .map((script) => document(script).text().trim())
    .filter(Boolean)
}

describe("options theme bootstrap", () => {
  it("loads a same-origin external classic script synchronously from the head", () => {
    const document = load(optionsHtml)
    const bootstrap = document('head > script[src="/theme-bootstrap.js"]')

    expect(bootstrap).toHaveLength(1)
    expect(bootstrap.is("[type], [async], [defer]")).toBe(false)
  })

  it("contains no executable inline script body", () => {
    expect(inlineScriptBodies(optionsHtml)).toEqual([])
  })

  it.each([
    ["closing-tag whitespace", "<script>boot()</script \t\n>"],
    ["mixed-case module tags", '<ScRiPt type="module">boot()</sCrIpT >'],
    ["an unrelated src attribute", '<script data-src="/external.js">boot()</script>'],
    ["src text inside an attribute", '<script title="src=external.js">boot()</script>'],
  ])("detects executable inline code with %s", (_name, html) => {
    expect(inlineScriptBodies(html)).toEqual(["boot()"])
  })

  it("excludes external scripts and empty inline bodies", () => {
    expect(inlineScriptBodies(`
      <script SRC="/theme-bootstrap.js">ignored()</script>
      <script src="">ignored()</script>
      <script> \t\n </script>
      <script>boot()</script>
    `)).toEqual(["boot()"])
  })

  it("ships the referenced public script", () => {
    expect(existsSync(themeBootstrapPath)).toBe(true)
  })

  it("applies the stored dark theme before application code runs", () => {
    const executionOrder: string[] = []

    runInNewContext(`${themeBootstrap}\napplicationCode()`, {
      localStorage: {
        getItem: (key: string) => key === "theme" ? "dark" : null,
      },
      window: {
        matchMedia: () => ({ matches: false }),
      },
      document: {
        documentElement: {
          classList: {
            add: (className: string) => executionOrder.push(`theme:${className}`),
          },
        },
      },
      applicationCode: () => executionOrder.push("application"),
    })

    expect(executionOrder).toEqual(["theme:dark", "application"])
  })

  it("falls back to the system theme when stored-theme access is blocked", () => {
    const executionOrder: string[] = []

    runInNewContext(`${themeBootstrap}\napplicationCode()`, {
      localStorage: {
        getItem: () => {
          throw new DOMException("Blocked", "SecurityError")
        },
      },
      window: {
        matchMedia: () => ({ matches: true }),
      },
      document: {
        documentElement: {
          classList: {
            add: (className: string) => executionOrder.push(`theme:${className}`),
          },
        },
      },
      applicationCode: () => executionOrder.push("application"),
    })

    expect(executionOrder).toEqual(["theme:dark", "application"])
  })

  it("surfaces unexpected stored-theme failures", () => {
    expect(() => runInNewContext(themeBootstrap, {
      localStorage: {
        getItem: () => {
          throw new Error("unexpected storage failure")
        },
      },
      window: {
        matchMedia: () => ({ matches: false }),
      },
      document: {
        documentElement: {
          classList: {
            add: () => undefined,
          },
        },
      },
    })).toThrow("unexpected storage failure")
  })
})
