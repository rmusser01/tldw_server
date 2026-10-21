import { afterEach, beforeEach, describe, expect, it } from "vitest"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { execFile as execFileCallback } from "node:child_process"
import { promisify } from "node:util"
import { runDriftReport } from "../../../scripts/verify-design-system-drift.mjs"
import { REQUIRED_DEFINITION_PATHS, TOKEN_SOURCE_PATHS } from "../../../scripts/design-system-drift-tokens.mjs"

const execFile = promisify(execFileCallback)
const realRepo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../../../..")
const cli = path.join(realRepo, "apps/packages/ui/scripts/verify-design-system-drift.mjs")
const roots = ["apps/packages/ui/src", "apps/tldw-frontend", "apps/extension", "admin-ui"]
let repo: string
let output: string
const git = (...args: string[]) => execFile("git", ["-C", repo, ...args])
const write = async (relative: string, value: string) => {
  const absolute = path.join(repo, relative)
  await fs.mkdir(path.dirname(absolute), { recursive: true })
  await fs.writeFile(absolute, value)
}
const scan = () => runDriftReport({ repoRoot: repo, outputPath: output })

beforeEach(async () => {
  repo = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "tldw-drift-test-")))
  output = path.join(repo, "report.json")
  await git("init", "--quiet")
  await git("config", "user.email", "test@example.invalid")
  await git("config", "user.name", "Drift Test")
  for (const relative of new Set([...REQUIRED_DEFINITION_PATHS, ...TOKEN_SOURCE_PATHS])) {
    try { await write(relative, await fs.readFile(path.join(realRepo, relative), "utf8")) }
    catch (error) { if (REQUIRED_DEFINITION_PATHS.includes(relative) || error.code !== "ENOENT") throw error }
  }
  await write("apps/packages/ui/scripts/design-system-drift-exceptions.json", '{"schemaVersion":1,"exceptions":[]}')
  for (const root of roots) await write(`${root}/Notice.tsx`, 'export function Notice(){return <div style={{color:"red"}}/>}')
  await write(".gitignore", "ignored.tsx\n")
  await git("add", ".")
  await git("commit", "--quiet", "-m", "Fixture")
})

afterEach(async () => { await fs.rm(repo, { recursive: true, force: true }) })

describe("advisory drift CLI", () => {
  it("scans all four roots and exits successfully with findings", async () => {
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(0)
    expect(new Set(report.findings.map(f => f.surface))).toEqual(new Set(["shared-ui", "webui", "extension", "admin-ui"]))
    expect(report.dirtySource).toBe(false)
    expect(JSON.parse(await fs.readFile(output, "utf8"))).toEqual(report)
  })

  it.each(["staged", "unstaged", "untracked", "deleted"])("marks %s source dirty", async state => {
    const target = "apps/packages/ui/src/Notice.tsx"
    if (state === "deleted") await fs.unlink(path.join(repo, target))
    else await write(state === "untracked" ? "apps/packages/ui/src/Other notice.tsx" : target, 'export const Other=()=> <div className="bg-red-500"/>')
    if (state === "staged") await git("add", target)
    const { report } = await scan()
    expect(report.dirtySource).toBe(true)
    expect(report.untrackedSourceFileCount).toBe(state === "untracked" ? 1 : 0)
  })

  it("ignores unrelated, test, ignored-source, and report-output changes", async () => {
    await write("notes.md", "unrelated")
    await write("apps/packages/ui/src/ignored.tsx", "broken source")
    await write("apps/packages/ui/src/Notice.test.tsx", "broken test source")
    await fs.writeFile(output, "old report")
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(0)
    expect(report.dirtySource).toBe(false)
    expect(report.findings.some(f => f.path.endsWith("ignored.tsx"))).toBe(false)
  })

  it("excludes the documented upstream react-pdf stylesheet copy", async () => {
    await write("apps/packages/ui/src/assets/react-pdf.css", "invalid copied vendor CSS {")
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(0)
    expect(report.diagnostics.some(d => d.path.endsWith("react-pdf.css"))).toBe(false)
    expect(report.dirtySource).toBe(false)
  })

  it("includes definition-only and manifest edits in provenance", async () => {
    await write("apps/packages/ui/scripts/design-system-drift-exceptions.json", '{"schemaVersion":1,"exceptions":[]}\n')
    expect((await scan()).report.dirtySource).toBe(true)
  })

  it("retains a partial report when a root is missing", async () => {
    await fs.rm(path.join(repo, "admin-ui"), { recursive: true })
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(1)
    expect(report.status).toBe("incomplete")
    expect(report.findings.length).toBeGreaterThan(0)
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ code: "missing-root" }))
  })

  it("makes malformed source visibly incomplete", async () => {
    await write("apps/packages/ui/src/Broken.tsx", "export const = <")
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(1)
    expect(report.status).toBe("incomplete")
  })

  it("does not traverse source symlinks", async () => {
    await fs.symlink(path.join(repo, "notes.md"), path.join(repo, "apps/packages/ui/src/Linked.tsx"))
    const { report } = await scan()
    expect(report.findings.some(f => f.path.endsWith("Linked.tsx"))).toBe(false)
  })

  it("rejects output inside a scan root without overwriting source", async () => {
    const target = path.join(repo, "apps/packages/ui/src/Notice.tsx")
    const original = await fs.readFile(target, "utf8")
    await expect(runDriftReport({ repoRoot: repo, outputPath: target })).rejects.toThrow(/output/i)
    expect(await fs.readFile(target, "utf8")).toBe(original)
  })

  it("rejects tracked and symlink output destinations", async () => {
    await expect(runDriftReport({ repoRoot: repo, outputPath: path.join(repo, ".gitignore") })).rejects.toThrow(/output/i)
    const target = path.join(repo, "output-link")
    await fs.symlink(path.join(repo, "unrelated.json"), target)
    await expect(runDriftReport({ repoRoot: repo, outputPath: target })).rejects.toThrow(/output/i)
  })

  it("replaces a hard-linked report without truncating tracked content", async () => {
    await write("important.md", "keep this content")
    await git("add", "important.md")
    await git("commit", "--quiet", "-m", "Important content")
    await fs.link(path.join(repo, "important.md"), output)
    await scan()
    expect(await fs.readFile(path.join(repo, "important.md"), "utf8")).toBe("keep this content")
    expect(JSON.parse(await fs.readFile(output, "utf8")).status).toBe("complete")
  })

  it("marks unavailable skip-worktree source incomplete", async () => {
    const target = "apps/extension/Notice.tsx"
    await git("update-index", "--skip-worktree", target)
    await fs.unlink(path.join(repo, target))
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(1)
    expect(report.status).toBe("incomplete")
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ path: target, code: "unavailable-source", severity: "error" }))
  })

  it.each(["--skip-worktree", "--assume-unchanged"])("does not claim clean provenance when %s hides edits", async flag => {
    const target = "apps/extension/Notice.tsx"
    await git("update-index", flag, target)
    await write(target, 'export const Changed = () => <div className="bg-red-500"/>')
    const { report, exitCode } = await scan()
    expect(report.dirtySource).not.toBe(false)
    expect(exitCode).toBe(1)
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ code: "limited-git-provenance", severity: "error" }))
  })

  it("reports invalid manifest data as incomplete", async () => {
    await write("apps/packages/ui/scripts/design-system-drift-exceptions.json", "not json")
    const { report, exitCode } = await scan()
    expect(exitCode).toBe(1)
    expect(report.status).toBe("incomplete")
  })

  it("uses actual subprocess exit codes and rejects unknown flags", async () => {
    const result = await execFile(process.execPath, [cli, "--repo-root", repo, "--output", output])
    expect(result.stdout).toContain("COMPLETE")
    await expect(execFile(process.execPath, [cli, "--enforce"])).rejects.toMatchObject({ code: 1 })
    await fs.rm(path.join(repo, "admin-ui"), { recursive: true })
    await expect(execFile(process.execPath, [cli, "--repo-root", repo, "--output", output])).rejects.toMatchObject({ code: 1 })
  })
})
