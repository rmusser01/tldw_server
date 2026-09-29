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

describe("Playground hook JSX extension guard", () => {
  it("stores JSX-bearing hooks in .tsx modules", () => {
    const jsxBearingHooks = [
      {
        path: resolveHookPath("useComposerInput"),
        marker: "<React.Profiler"
      },
      {
        path: resolveHookPath("usePlaygroundPersistence"),
        marker: "<button"
      }
    ]

    for (const hook of jsxBearingHooks) {
      const source = fs.readFileSync(hook.path, "utf8")
      expect(source).toContain(hook.marker)
      expect(hook.path.endsWith(".tsx")).toBe(true)
    }
  })
})
