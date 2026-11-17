/**
 * Source guard for WP-21 (WebUI perf batch W2, Stage 6):
 * store mutation paths must not use `JSON.parse(JSON.stringify(...))`
 * deep clones. Use `structuredClone` (or a JSON-clone when the cloned shape
 * genuinely requires JSON semantics — document it here if so).
 *
 * The plan's success criterion: `grep -rn "JSON.parse(JSON.stringify" store/`
 * returns only test fixtures (plus the documented allowlist below).
 */
import { readFileSync, readdirSync } from "node:fs"
import { dirname, join, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = dirname(fileURLToPath(import.meta.url))
const storeDir = resolve(testDir, "..")

const PATTERN = "JSON.parse(JSON.stringify("

/**
 * Documented exceptions — file path (relative to src/store) -> reason.
 *
 * `workspace.ts`: `cloneWorkspaceValue` is structuredClone-first; the JSON
 * round-trip is a guarded fallback for runtimes without `structuredClone`
 * (the same defensive pattern as Watchlists `clone-utils.ts` and
 * `output-presets.ts`). It is not an active JSON mutation-path clone in any
 * supported target, so it stays on the allowlist. The structuredClone-first
 * guard is asserted below so the entry cannot silently rot.
 */
const ALLOWLIST: Record<string, string> = {
  "workspace.ts":
    "cloneWorkspaceValue: structuredClone-first guarded fallback (see reason above)"
}

const isTestDir = (path: string) => path.split("/").includes("__tests__")

const listSourceFiles = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true, recursive: true })
    .filter(
      (entry) =>
        entry.isFile() &&
        (entry.name.endsWith(".ts") || entry.name.endsWith(".tsx")) &&
        !isTestDir(entry.parentPath)
    )
    .map((entry) => join(entry.parentPath, entry.name))

describe("store source guard: no JSON deep clones in mutation paths", () => {
  it("src/store non-test files contain no JSON.parse(JSON.stringify( outside the allowlist", () => {
    const offenders: string[] = []

    for (const file of listSourceFiles(storeDir)) {
      const rel = relative(storeDir, file)
      const source = readFileSync(file, "utf8")
      if (source.includes(PATTERN) && !(rel in ALLOWLIST)) {
        offenders.push(rel)
      }
    }

    expect(offenders).toEqual([])
  })

  it("allowlisted workspace.ts still guards its JSON fallback with structuredClone-first", () => {
    const source = readFileSync(resolve(storeDir, "workspace.ts"), "utf8")
    // If the structuredClone-first guard is removed or the fallback line is
    // deleted, this allowlist entry must be revisited (fail loudly either way).
    expect(source).toContain("typeof structuredClone === \"function\"")
    expect(source).toContain(PATTERN)
  })
})
