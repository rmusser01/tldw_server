import fs from "node:fs"
import path from "node:path"
import { createRequire } from "node:module"
import postcss from "postcss"
import tailwindcss from "tailwindcss"
import { describe, expect, it } from "vitest"

const require = createRequire(import.meta.url)

// The theme colour variables hold bare channel triples ("13 134 119"), not
// colours. An arbitrary Tailwind value such as `ring-[color:var(--color-focus)]`
// copies the triple straight into the property, which the browser rejects at
// computed-value time: the focus ring disappears, outlines fall back to
// currentColor and fills turn transparent (AX-08). The token utilities
// (`ring-focus`, `outline-focus`, `bg-primary`, ...) wrap the triple in rgb().
const BARE_COLOUR_VAR = /\[(?:color:)?var\(--(?:color|state)-[a-z0-9-]+\)\]/g

const appsRoot = path.resolve(__dirname, "../../../../..")
const sourceRoots = [
  "packages/ui/src",
  "tldw-frontend/components",
  "tldw-frontend/pages",
  "tldw-frontend/extension",
  "tldw-frontend/lib",
  "tldw-frontend/routes",
  "tldw-frontend/hooks",
  "extension/entrypoints"
]

const isScannedSource = (name: string) =>
  /\.(tsx?|jsx?)$/.test(name) && !/\.(test|spec)\.[jt]sx?$/.test(name)

function collectSources(dir: string, out: string[]) {
  if (!fs.existsSync(dir)) return
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "node_modules" || entry.name === "__tests__") continue
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) collectSources(full, out)
    else if (isScannedSource(entry.name)) out.push(full)
  }
}

async function compileUtilities(classes: string[]): Promise<string> {
  const config = require("../../../../../tldw-frontend/tailwind.config.js")
  const result = await postcss([
    tailwindcss({ ...config, content: [{ raw: classes.join(" ") }] })
  ]).process("@tailwind utilities;", { from: undefined })
  return result.css
}

describe("theme colour utilities", () => {
  it("never pass a bare colour variable through an arbitrary Tailwind value", () => {
    const files: string[] = []
    for (const root of sourceRoots) collectSources(path.join(appsRoot, root), files)
    expect(files.length).toBeGreaterThan(100)

    const offenders: string[] = []
    for (const file of files) {
      const lines = fs.readFileSync(file, "utf8").split("\n")
      lines.forEach((line, index) => {
        for (const match of line.match(BARE_COLOUR_VAR) ?? []) {
          offenders.push(`${path.relative(appsRoot, file)}:${index + 1} ${match}`)
        }
      })
    }

    expect(offenders).toEqual([])
  })

  it("explains the guard: arbitrary values emit an invalid colour, token utilities emit rgb()", async () => {
    const css = await compileUtilities([
      "focus:ring-[color:var(--color-focus)]",
      "focus:ring-focus",
      "focus-visible:outline-focus"
    ])

    expect(css).toMatch(/--tw-ring-color:\s*var\(--color-focus\)/)
    expect(css).toMatch(/--tw-ring-color:\s*rgb\(var\(--color-focus\)/)
    expect(css).toMatch(/outline-color:\s*rgb\(var\(--color-focus\)/)
  }, 30_000)
})
