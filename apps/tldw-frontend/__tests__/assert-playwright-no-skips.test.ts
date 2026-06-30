import { mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { spawnSync } from "node:child_process"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const scriptPath = path.join(appDir, "scripts", "assert-playwright-no-skips.mjs")

const runWithReport = (contents: string) => {
  const tempDir = mkdtempSync(path.join(tmpdir(), "playwright-no-skips-"))
  const reportPath = path.join(tempDir, "report.json")

  writeFileSync(reportPath, contents, "utf8")

  const result = spawnSync(process.execPath, [scriptPath, reportPath], {
    cwd: appDir,
    encoding: "utf8",
  })

  rmSync(tempDir, { force: true, recursive: true })

  return result
}

describe("assert-playwright-no-skips", () => {
  it("reports executed totals for a clean Playwright run", () => {
    const result = runWithReport(
      JSON.stringify({
        stats: {
          expected: 5,
          flaky: 0,
          skipped: 0,
          unexpected: 0,
        },
      })
    )

    expect(result.status).toBe(0)
    expect(result.stdout).toContain("executed=5 expected=5")
  })

  it("keeps all-failure Playwright runs distinguishable from empty runs", () => {
    const result = runWithReport(
      JSON.stringify({
        stats: {
          expected: 0,
          flaky: 0,
          skipped: 0,
          unexpected: 2,
        },
      })
    )

    expect(result.status).toBe(1)
    expect(result.stdout).toContain("executed=2 expected=0")
    expect(result.stderr).toContain("Found 2 unexpected failure(s).")
    expect(result.stderr).not.toContain("No tests executed")
  })

  it("handles invalid Playwright JSON reports without a Node stack trace", () => {
    const result = runWithReport("{ invalid json")

    expect(result.status).toBe(2)
    expect(result.stderr).toContain("Unable to parse Playwright JSON report")
    expect(result.stderr).not.toContain("SyntaxError:")
  })
})
