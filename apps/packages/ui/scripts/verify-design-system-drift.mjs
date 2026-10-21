#!/usr/bin/env node
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { execFile as execFileCallback } from "node:child_process"
import { promisify } from "node:util"
import { buildTokenAuthority, REQUIRED_DEFINITION_PATHS, TOKEN_SOURCE_PATHS } from "./design-system-drift-tokens.mjs"
import { analyzeDriftSource } from "./design-system-drift-rules.mjs"
import { assembleDriftReport, formatDriftSummary } from "./design-system-drift-report.mjs"

const execFile = promisify(execFileCallback)
const scriptPath = fileURLToPath(import.meta.url)
const defaultRepo = path.resolve(path.dirname(scriptPath), "../../../..")
export const SOURCE_ROOTS = ["apps/packages/ui/src", "apps/tldw-frontend", "apps/extension", "admin-ui"]
const manifestPath = "apps/packages/ui/scripts/design-system-drift-exceptions.json"
const supportPaths = [".gitignore", "apps/bun.lock", "apps/package.json", "apps/packages/ui/package.json", manifestPath,
  "apps/packages/ui/scripts/design-system-drift-tokens.mjs", "apps/packages/ui/scripts/design-system-drift-rules.mjs",
  "apps/packages/ui/scripts/design-system-drift-report.mjs", "apps/packages/ui/scripts/verify-design-system-drift.mjs",
  ...TOKEN_SOURCE_PATHS]
const skippedDirectories = new Set(["node_modules", "dist", "build", ".next", ".wxt", ".output", ".turbo", ".cache", "coverage", "vendor", "vendors", "__tests__", "tests", "test", "__fixtures__", "fixtures", "__snapshots__", "scripts", "e2e", ".worktrees", ".superpowers"])
const limitations = [
  "Backend templates, static HTML and MDX are outside v1 syntax coverage.",
  "Runtime-generated styles and unresolved dynamic expressions are not evaluated.",
  "Layout values are review candidates, not confirmed visual defects.",
  "Definitions are read statically; runtime theme overrides and unresolved unit conversions require review."
]
const isWithin = (file, root) => file === root || file.startsWith(root + path.sep)
const rootFor = relative => SOURCE_ROOTS.find(root => relative.startsWith(root + "/"))
const errorDiagnostic = (relativePath, code, message) => ({ path: relativePath, code, message, severity: "error" })

function exclusion(relativePath) {
  if (!rootFor(relativePath)) return "outside-source-roots"
  const parts = relativePath.split("/")
  if (parts.some(part => skippedDirectories.has(part))) return "test-tooling-generated-or-vendor-directory"
  if (relativePath === "apps/packages/ui/src/assets/react-pdf.css") return "documented-upstream-vendor-copy"
  if (/\.(test|spec)\.[cm]?[jt]sx?$/.test(relativePath) || /\.(snap|d\.ts)$/.test(relativePath)) return "test-or-declaration-file"
  if (/\.min\.[cm]?js$/.test(relativePath) || /\/public\/pdf\.worker(?:\.[^/]*)?\.mjs$/.test(relativePath)) return "vendor-bundle"
  if (!/\.(?:[cm]?js|jsx|ts|tsx|css)$/.test(relativePath)) return "unsupported-file-type"
  if (/\/(?:eslint|vitest|playwright|postcss|prettier|webpack|next|wxt)\.config\.[^/]+$/.test(relativePath)) return "tooling-config"
  return null
}

const isInputPath = relative => supportPaths.includes(relative) || REQUIRED_DEFINITION_PATHS.includes(relative) || (rootFor(relative) && (relative.endsWith("/.gitignore") || exclusion(relative) === null))

async function git(repoRoot, args) {
  const result = await execFile("git", ["--literal-pathspecs", "-C", repoRoot, ...args], { maxBuffer: 64 * 1024 * 1024 })
  return result.stdout
}

async function safeSourceFile(repoRoot, relativePath) {
  const parts = relativePath.split("/")
  if (parts.some(part => part === ".." || part === "")) return "invalid"
  let absolute = repoRoot
  for (const part of parts) {
    absolute = path.join(absolute, part)
    try {
      const stat = await fs.lstat(absolute)
      if (stat.isSymbolicLink()) return "symlink"
    } catch (error) {
      if (error.code === "ENOENT") return "missing"
      throw error
    }
  }
  return (await fs.lstat(absolute)).isFile() ? "file" : "non-file"
}

async function validateOutput(repoRoot, outputPath) {
  const requested = path.resolve(repoRoot, outputPath)
  const parent = await fs.realpath(path.dirname(requested))
  const output = path.join(parent, path.basename(requested))
  for (const root of SOURCE_ROOTS) if (isWithin(output, path.join(repoRoot, root))) throw new Error("Report output must be outside source roots")
  const gitDirs = await Promise.all([git(repoRoot, ["rev-parse", "--absolute-git-dir"]), git(repoRoot, ["rev-parse", "--git-common-dir"])])
  if ([path.join(repoRoot, ".git"), ...gitDirs.map(dir => path.resolve(repoRoot, dir.trim()))].some(dir => isWithin(output, dir))) throw new Error("Report output must be outside Git metadata")
  try {
    const stat = await fs.lstat(requested)
    if (stat.isSymbolicLink() || !stat.isFile()) throw new Error("Report output must be an ordinary file")
  } catch (error) { if (error.code !== "ENOENT") throw error }
  const relative = path.relative(repoRoot, output).replaceAll(path.sep, "/")
  if (supportPaths.includes(relative) || REQUIRED_DEFINITION_PATHS.includes(relative)) throw new Error("Report output cannot overwrite reporter inputs")
  if (isWithin(output, repoRoot)) {
    const tracked = await git(repoRoot, ["ls-files", "-z", "--cached", "--", relative])
    if (tracked) throw new Error("Report output cannot overwrite tracked files")
  }
  return output
}

function statusEntries(status) {
  const paths = []
  const records = status.split("\0")
  for (let index = 0; index < records.length; index++) {
    const record = records[index]
    if (!record) continue
    const change = record.slice(0, 2)
    paths.push({ path: record.slice(3), change })
    if (/R|C/.test(change)) paths.push({ path: records[++index], change })
  }
  return paths.filter(entry => entry.path)
}

/** Scan owned UI source without evaluating application imports or enforcing findings. */
export async function runDriftReport({ repoRoot, outputPath }) {
  repoRoot = await fs.realpath(repoRoot)
  const output = await validateOutput(repoRoot, outputPath)
  const diagnostics = []
  const excluded = new Map()
  const exclude = reason => excluded.set(reason, (excluded.get(reason) ?? 0) + 1)
  const coverage = { roots: SOURCE_ROOTS.map(root => ({ path: root + "/", discovered: 0, analyzed: 0 })), excluded: [], limitations }
  const provenance = { sourceCommit: null, dirtySource: null, untrackedSourceFileCount: 0 }
  const deletedPaths = new Set()
  try {
    const status = await git(repoRoot, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ...SOURCE_ROOTS, ...supportPaths, ...REQUIRED_DEFINITION_PATHS])
    const changes = statusEntries(status)
    for (const entry of changes) if (entry.change.includes("D")) deletedPaths.add(entry.path)
    provenance.dirtySource = changes.some(({ path: relative }) => isInputPath(relative))
    const indexFlags = await git(repoRoot, ["ls-files", "-v", "-z", "--", ...SOURCE_ROOTS, ...supportPaths, ...REQUIRED_DEFINITION_PATHS])
    for (const record of indexFlags.split("\0").filter(Boolean)) {
      const flag = record[0]
      const relative = record.slice(2)
      if ((flag === "S" || flag === flag.toLowerCase()) && isInputPath(relative)) {
        provenance.dirtySource = null
        diagnostics.push(errorDiagnostic(relative, "limited-git-provenance", "Git skip-worktree/assume-unchanged flags prevent reliable source provenance"))
      }
    }
  } catch {
    provenance.dirtySource = null
    diagnostics.push(errorDiagnostic("", "git-provenance", "Unable to determine modified source inputs"))
  }
  let inventory = []
  let untracked = new Set()
  try {
    const [listed, untrackedOutput, commit] = await Promise.all([
      git(repoRoot, ["ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ...SOURCE_ROOTS]),
      git(repoRoot, ["ls-files", "-z", "--others", "--exclude-standard", "--", ...SOURCE_ROOTS]),
      git(repoRoot, ["rev-parse", "--verify", "HEAD"])
    ])
    inventory = [...new Set(listed.split("\0").filter(Boolean))].sort()
    untracked = new Set(untrackedOutput.split("\0").filter(Boolean))
    provenance.sourceCommit = commit.trim()
  } catch {
    diagnostics.push(errorDiagnostic("", "git-inventory", "Unable to establish Git source inventory and commit"))
  }
  const unavailableRoots = new Set()
  for (const root of SOURCE_ROOTS) {
    try {
      const stat = await fs.lstat(path.join(repoRoot, root))
      if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error("invalid-root")
    } catch {
      unavailableRoots.add(root)
      diagnostics.push(errorDiagnostic(root, "missing-root", "Expected source root is missing or is not an ordinary directory"))
    }
  }
  const candidates = []
  for (const relative of inventory) {
    const root = rootFor(relative)
    if (unavailableRoots.has(root)) continue
    const reason = exclusion(relative)
    if (reason) { exclude(reason); continue }
    const stats = coverage.roots.find(entry => entry.path === root + "/")
    stats.discovered++
    candidates.push(relative)
  }
  const sourcePaths = [...new Set([...candidates, ...REQUIRED_DEFINITION_PATHS, ...TOKEN_SOURCE_PATHS])].sort()
  const sources = []
  for (let offset = 0; offset < sourcePaths.length; offset += 16) {
    await Promise.all(sourcePaths.slice(offset, offset + 16).map(async relative => {
      if (unavailableRoots.has(rootFor(relative))) return
      try {
        const availability = await safeSourceFile(repoRoot, relative)
        if (availability !== "file") {
          if (REQUIRED_DEFINITION_PATHS.includes(relative)) diagnostics.push(errorDiagnostic(relative, "missing-definition-input", "Required definition input is missing or is a symlink"))
          else if (candidates.includes(relative)) {
            if (availability === "missing" && !deletedPaths.has(relative)) diagnostics.push(errorDiagnostic(relative, "unavailable-source", "Source is unavailable without a recorded Git deletion"))
            else exclude(availability === "missing" ? "deleted-source" : "symlink-or-non-file-source")
          }
          return
        }
        sources.push({ path: relative, source: await fs.readFile(path.join(repoRoot, relative), "utf8") })
        if (untracked.has(relative) && candidates.includes(relative)) provenance.untrackedSourceFileCount++
      } catch {
        diagnostics.push(errorDiagnostic(relative, "source-read", "Unable to read source file"))
      }
    }))
  }
  sources.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0)
  const authority = buildTokenAuthority({ sources })
  diagnostics.push(...authority.diagnostics)
  const analyses = []
  for (const input of sources) {
    if (!candidates.includes(input.path)) continue
    try {
      const analysis = analyzeDriftSource({ ...input, authority })
      analyses.push(analysis)
      if (!analysis.diagnostics.some(diagnostic => diagnostic.severity === "error")) {
        coverage.roots.find(entry => entry.path === rootFor(input.path) + "/").analyzed++
      }
    } catch {
      diagnostics.push(errorDiagnostic(input.path, "source-analysis", "Unable to analyze source file"))
    }
  }
  for (const entry of coverage.roots) {
    if (!unavailableRoots.has(entry.path.slice(0, -1)) && entry.analyzed === 0) diagnostics.push(errorDiagnostic(entry.path, "empty-root", "Source root contains no analyzed source files"))
  }
  let exceptions = { schemaVersion: 1, exceptions: [] }
  try {
    if (await safeSourceFile(repoRoot, manifestPath) !== "file") throw new Error("missing-manifest")
    exceptions = JSON.parse(await fs.readFile(path.join(repoRoot, manifestPath), "utf8"))
  } catch {
    diagnostics.push(errorDiagnostic(manifestPath, "exceptions-read", "Unable to read a valid exception manifest"))
  }
  coverage.excluded = [...excluded].map(([reason, count]) => ({ reason, count }))
  const report = assembleDriftReport({ analyses, exceptions, coverage, provenance, diagnostics })
  // Replace the directory entry, never truncate an inode that may have other hard links.
  const temporaryDirectory = await fs.mkdtemp(path.join(path.dirname(output), ".tldw-drift-"))
  try {
    const temporaryPath = path.join(temporaryDirectory, "report.json")
    await fs.writeFile(temporaryPath, JSON.stringify(report, null, 2) + "\n", { flag: "wx", mode: 0o600 })
    await fs.rename(temporaryPath, output)
  } finally {
    await fs.rm(temporaryDirectory, { recursive: true, force: true })
  }
  return { report, exitCode: report.status === "complete" ? 0 : 1 }
}

async function main(args) {
  const options = { repoRoot: defaultRepo, outputPath: path.join(os.tmpdir(), "tldw-design-system-drift.json") }
  const seen = new Set()
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index]
    if (!["--repo-root", "--output"].includes(flag) || seen.has(flag) || !args[index + 1] || args[index + 1].startsWith("--")) throw new Error("Usage: report:design-system-drift [--repo-root <path>] [--output <path>]")
    seen.add(flag)
    options[flag === "--repo-root" ? "repoRoot" : "outputPath"] = args[index + 1]
  }
  const { report, exitCode } = await runDriftReport(options)
  process.stdout.write(formatDriftSummary(report, { limit: 25, reportPath: path.resolve(options.repoRoot, options.outputPath) }))
  process.exitCode = exitCode
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main(process.argv.slice(2)).catch(error => {
    console.error(`UI drift scan failed: ${error.message}`)
    process.exitCode = 1
  })
}
