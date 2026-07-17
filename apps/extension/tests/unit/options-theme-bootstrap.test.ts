import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { runInNewContext } from "node:vm"
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
