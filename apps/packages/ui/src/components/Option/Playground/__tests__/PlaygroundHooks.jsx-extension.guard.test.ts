import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const resolveHookPath = (basename: string) => {
  const hooksDir = path.resolve(__dirname, "../hooks")
  const tsxPath = path.join(hooksDir, `${basename}.tsx`)
  const tsPath = path.join(hooksDir, `${basename}.ts`)
  if (fs.existsSync(tsxPath)) return tsxPath
  return tsPath
}

const hooksDir = path.resolve(__dirname, "../hooks")

// A closing tag or a self-closing element. TypeScript generics
// (`useRef<AbortController | null>`) produce neither, so this finds JSX
// without depending on which element a hook happens to render.
const JSX_MARKUP = /<\/[A-Za-z][\w.]*>|\/>/

describe("Playground hook JSX extension guard", () => {
  it("stores JSX-bearing hooks in .tsx modules", () => {
    const tsHooksWithJsx = fs
      .readdirSync(hooksDir)
      .filter((name) => name.endsWith(".ts"))
      .filter((name) =>
        JSX_MARKUP.test(fs.readFileSync(path.join(hooksDir, name), "utf8"))
      )

    expect(tsHooksWithJsx).toEqual([])
  })

  it("detects JSX markup but not TypeScript generics", () => {
    expect(JSX_MARKUP.test("return <button>Retry</button>")).toBe(true)
    expect(JSX_MARKUP.test("<React.Profiler id='x'>{child}</React.Profiler>")).toBe(true)
    expect(JSX_MARKUP.test("return <Spinner />")).toBe(true)
    expect(JSX_MARKUP.test("React.useRef<AbortController | null>(null)")).toBe(false)
    expect(JSX_MARKUP.test("const map = new Map<string, Array<number>>()")).toBe(false)
  })

  it("keeps the composer profiler hook in .tsx", () => {
    const jsxBearingHooks = [
      {
        path: resolveHookPath("useComposerInput"),
        marker: "<React.Profiler"
      }
    ]

    for (const hook of jsxBearingHooks) {
      const source = fs.readFileSync(hook.path, "utf8")
      expect(source).toContain(hook.marker)
      expect(hook.path.endsWith(".tsx")).toBe(true)
    }
  })
})
