#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process"
import {
  appendFileSync,
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import path from "node:path"
import { tmpdir } from "node:os"
import { fileURLToPath } from "node:url"
import { redactText } from "../onboarding-uat/artifacts.mjs"
import { reservePorts } from "../onboarding-uat/ports.mjs"
import {
  spawnLoggedProcess,
  stopProcessTree,
  waitForHttpOk,
} from "../onboarding-uat/processes.mjs"
import { resolvePythonCommand } from "../onboarding-uat/run.mjs"
import { inventoryProjects } from "./inventory-api-mocks.mjs"
import { buildLiveTierBackendEnv, buildLiveTierProfile } from "./profile.mjs"
import {
  assertProjectAccounting,
  collectSkippedTests,
  parseListOutput,
  renderMarkdownReport,
  summarizePlaywrightReport,
} from "./report.mjs"

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRootDefault = path.resolve(moduleDir, "../..")
const repoRootDefault = path.resolve(frontendRootDefault, "../..")
const validProjects = new Set(["tier-1", "tier-2", "tier-3"])

function runIdNow() {
  return new Date().toISOString().replace(/[:.]/g, "-")
}

function requireNumber(value, flag) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${flag} requires a positive integer`)
  return parsed
}

export function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    projects: ["tier-1", "tier-2", "tier-3"],
    workers: 1,
    listOnly: false,
    grep: null,
    preserveRuntime: false,
    failOnSkip: true,
    runId: null,
    help: false,
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--help" || arg === "-h") options.help = true
    else if (arg === "--list-only") options.listOnly = true
    else if (arg === "--preserve-runtime") options.preserveRuntime = true
    else if (arg === "--allow-skips") options.failOnSkip = false
    else if (arg.startsWith("--projects=")) options.projects = arg.slice(11).split(",").filter(Boolean)
    else if (arg === "--projects") options.projects = (argv[++index] ?? "").split(",").filter(Boolean)
    else if (arg.startsWith("--workers=")) options.workers = requireNumber(arg.slice(10), "--workers")
    else if (arg === "--workers") options.workers = requireNumber(argv[++index], "--workers")
    else if (arg.startsWith("--grep=")) options.grep = arg.slice(7)
    else if (arg === "--grep") options.grep = argv[++index] ?? null
    else if (arg.startsWith("--run-id=")) options.runId = arg.slice(9)
    else if (arg === "--run-id") options.runId = argv[++index] ?? null
    else throw new Error(`Unknown option: ${arg}`)
  }

  if (!options.projects.length || options.projects.some((project) => !validProjects.has(project))) {
    throw new Error("--projects entries must be tier-1, tier-2, or tier-3")
  }
  if (options.runId && !/^[A-Za-z0-9._-]+$/.test(options.runId)) {
    throw new Error("--run-id may contain only letters, numbers, dots, underscores, and hyphens")
  }
  return options
}

export function formatUsage() {
  return [
    "Usage: bun run uat:live-tiers -- [options]",
    "",
    "  --projects=tier-1,tier-2,tier-3  Complete projects to list and run.",
    "  --workers=1                       Playwright worker count.",
    "  --list-only                       List and inventory without executing tests.",
    "  --grep=<pattern>                  Non-certifying bounded smoke selection.",
    "  --preserve-runtime                Keep the disposable backend profile.",
    "  --allow-skips                     Report skips without failing the run.",
    "  --run-id=<id>                     Stable artifact identifier.",
  ].join("\n")
}

export function isCertificationRun(options) {
  const certificationProjects = ["tier-1", "tier-2", "tier-3"]
  const selectedProjects = new Set(options.projects ?? [])
  const hasCompleteProjectSet =
    selectedProjects.size === certificationProjects.length &&
    certificationProjects.every((project) => selectedProjects.has(project))

  return !options.listOnly &&
    !options.grep &&
    options.failOnSkip &&
    options.workers === 1 &&
    hasCompleteProjectSet
}

function safeEnv(baseEnv, keys) {
  return Object.fromEntries(keys.flatMap((key) =>
    typeof baseEnv[key] === "string" ? [[key, baseEnv[key]]] : []
  ))
}

function withPythonPath(env, value) {
  return { ...env, PYTHONPATH: [value, env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
}

function playwrightProjectArgs(projects) {
  return projects.map((project) => `--project=${project}`)
}

export function buildCommands({
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  ports,
  profile,
  projects,
  workers,
  runId,
  grep = null,
  artifactRoot = profile.reportsDir,
  baseEnv = process.env,
}) {
  const python = resolvePythonCommand({ repoRoot, baseEnv })
  const backendEnv = buildLiveTierBackendEnv({ profile, mockPort: ports.mock, baseEnv })
  const apiKey = backendEnv.SINGLE_USER_API_KEY
  const backendUrl = `http://127.0.0.1:${ports.backend}`
  const webUrl = `http://localhost:${ports.web}`
  const mockUrl = `http://127.0.0.1:${ports.mock}`
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) {
    throw new Error("runId is not safe for the isolated Next build directory")
  }
  const nextDistDir = `.next-live-tier-${runId}`
  const nextDistPath = path.join(frontendRoot, nextDistDir)
  const baseKeys = ["PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "PYTHONPATH", "VIRTUAL_ENV"]
  const sharedFrontendEnv = {
    ...safeEnv(baseEnv, baseKeys),
    NEXT_PUBLIC_API_URL: backendUrl,
    NEXT_PUBLIC_API_VERSION: "v1",
    NEXT_PUBLIC_X_API_KEY: apiKey,
    TLDW_SERVER_URL: backendUrl,
    TLDW_E2E_SERVER_URL: backendUrl,
    TLDW_E2E_API_KEY: apiKey,
    TLDW_API_KEY: apiKey,
    TLDW_WEB_URL: webUrl,
    TLDW_WEB_AUTOSTART: "false",
    TLDW_E2E_ALLOW_OFFLINE: "0",
    TLDW_MOCK_OPENAI_URL: `${mockUrl}/v1`,
    TLDW_LIVE_TIER_UAT: "1",
    TLDW_LIVE_TIER_UAT_RUN_ID: runId,
    TLDW_E2E_ACP_WORKSPACE_ROOT_BASE: profile.acpWorkspaceRootBase,
    TLDW_E2E_INGESTION_SOURCE_ROOT: profile.fixtureRoot,
    TLDW_NEXT_DIST_DIR: nextDistDir,
  }
  const projectArgs = playwrightProjectArgs(projects)
  const jsonPath = path.join(artifactRoot, "playwright-results.json")
  const runArgs = [
    "playwright", "test", ...projectArgs, `--workers=${workers}`, "--retries=0", "--reporter=line,json",
  ]
  const grepArgs = grep ? ["--grep", grep] : []
  runArgs.push(...grepArgs)

  return {
    urls: { backend: backendUrl, web: webUrl, mock: mockUrl },
    backendEnv,
    mockOpenai: {
      name: "mock-openai",
      command: python,
      args: [
        "-m", "mock_openai.server", "--config",
        path.join(frontendRoot, "e2e/onboarding-uat/mock-openai/configs/local-success.json"),
        "--host", "127.0.0.1", "--port", String(ports.mock),
      ],
      cwd: path.join(frontendRoot, "e2e/onboarding-uat/mock-openai"),
      env: withPythonPath(safeEnv(baseEnv, baseKeys), path.join(repoRoot, "mock_openai_server")),
    },
    authInit: {
      name: "auth-init",
      command: python,
      args: ["-m", "tldw_Server_API.app.core.AuthNZ.initialize", "--non-interactive"],
      cwd: repoRoot,
      env: backendEnv,
    },
    backend: {
      name: "backend",
      command: python,
      args: ["-m", "uvicorn", "tldw_Server_API.app.main:app", "--host", "127.0.0.1", "--port", String(ports.backend)],
      cwd: repoRoot,
      env: backendEnv,
    },
    frontend: {
      name: "frontend",
      command: "bun",
      args: ["run", "dev", "--", "-p", String(ports.web)],
      cwd: frontendRoot,
      env: sharedFrontendEnv,
    },
    playwrightList: {
      name: "playwright-list",
      command: "bunx",
      args: ["playwright", "test", "--list", ...projectArgs, ...grepArgs],
      cwd: frontendRoot,
      env: sharedFrontendEnv,
    },
    playwrightRun: {
      name: "playwright-run",
      command: "bunx",
      args: runArgs,
      cwd: frontendRoot,
      env: { ...sharedFrontendEnv, PLAYWRIGHT_JSON_OUTPUT_NAME: jsonPath },
    },
    jsonPath,
    nextDistPath,
  }
}

function appendLog(logPath, chunk) {
  mkdirSync(path.dirname(logPath), { recursive: true })
  appendFileSync(logPath, redactText(chunk), "utf8")
}

/**
 * Run a one-shot command and wait for authoritative process-tree teardown on abort.
 *
 * @param {{ name?: string, command: string, args?: string[], cwd?: string, env?: NodeJS.ProcessEnv }} commandRecord
 * @param {string} logPath
 * @param {{ signal?: AbortSignal, stop?: (record: unknown) => Promise<void> }} [options]
 */
export async function runCommand(commandRecord, logPath, {
  signal,
  stop = stopProcessTree,
} = {}) {
  if (signal?.aborted) {
    return {
      code: null,
      signal: null,
      error: signal.reason,
      output: "",
      aborted: true,
    }
  }
  return new Promise((resolve) => {
    const child = spawn(commandRecord.command, commandRecord.args, {
      cwd: commandRecord.cwd,
      env: commandRecord.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    })
    const record = { ...commandRecord, child, pid: child.pid, loggingErrors: [] }
    let output = ""
    let settled = false
    let aborting = false
    const finish = (result) => {
      if (settled) return
      settled = true
      signal?.removeEventListener("abort", abort)
      resolve(result)
    }
    const abort = () => {
      if (settled || aborting) return
      aborting = true
      const reason = signal?.reason instanceof Error
        ? signal.reason
        : new Error(`${commandRecord.name ?? "Command"} aborted`)
      void stop(record).then(
        () => finish({ code: null, signal: null, error: reason, output, aborted: true }),
        (stopError) => finish({
          code: null,
          signal: null,
          error: new AggregateError([reason, stopError], `Failed to abort ${commandRecord.name ?? "command"}`),
          output,
          aborted: true,
        })
      )
    }
    const capture = (chunk) => {
      const text = String(chunk)
      output += text
      appendLog(logPath, text)
      process.stdout.write(redactText(text))
    }
    child.stdout?.on("data", capture)
    child.stderr?.on("data", capture)
    child.once("error", (error) => {
      if (!aborting) finish({ code: null, signal: null, error, output })
    })
    child.once("close", (code, childSignal) => {
      if (!aborting) finish({ code, signal: childSignal, output })
    })
    signal?.addEventListener("abort", abort, { once: true })
    if (signal?.aborted) abort()
  })
}

/**
 * @param {{
 *   processLike?: { on: (event: string, listener: () => void) => unknown, off: (event: string, listener: () => void) => unknown },
 *   controller?: AbortController,
 * }} [options]
 */
export function installTerminationHandlers({
  processLike = process,
  controller = new AbortController(),
} = {}) {
  const abortFor = (signalName) => {
    if (controller.signal.aborted) return
    const error = new Error(`Live Tier UAT interrupted by ${signalName}`)
    error.name = "AbortError"
    error.signalName = signalName
    controller.abort(error)
  }
  const handlers = {
    SIGINT: () => abortFor("SIGINT"),
    SIGTERM: () => abortFor("SIGTERM"),
  }
  processLike.on("SIGINT", handlers.SIGINT)
  processLike.on("SIGTERM", handlers.SIGTERM)

  return {
    controller,
    signal: controller.signal,
    dispose() {
      processLike.off("SIGINT", handlers.SIGINT)
      processLike.off("SIGTERM", handlers.SIGTERM)
    },
  }
}

export async function stopSpawnedProcesses(records, stop = stopProcessTree) {
  const errors = []
  for (const record of [...records].reverse()) {
    try {
      await stop(record)
    } catch (error) {
      errors.push(
        new Error(
          `Failed to stop ${record.name ?? "spawned process"}: ${error?.message ?? String(error)}`,
          { cause: error }
        )
      )
    }
  }
  if (errors.length) {
    throw new AggregateError(
      errors,
      `Failed to stop spawned processes: ${errors.map((error) => error.message).join("; ")}`
    )
  }
}

export function assertFreshRunTargets(targets, exists = existsSync) {
  for (const target of targets) {
    if (exists(target)) {
      throw new Error(`Live Tier run target already exists: ${target}`)
    }
  }
}

export function readPlaywrightReport(jsonPath, {
  exists = existsSync,
  read = readFileSync,
} = {}) {
  if (!exists(jsonPath)) {
    throw new Error(`Playwright JSON report was not produced: ${jsonPath}`)
  }
  return JSON.parse(read(jsonPath, "utf8"))
}

export function assertServicesStopped(stopped) {
  if (!stopped) {
    throw new Error("One or more spawned live-tier services did not stop")
  }
}

const retryableCleanupCodes = new Set(["EBUSY", "ENOTEMPTY", "EPERM"])

function makeGeneratedDirectoryTreeWritable(target) {
  if (!existsSync(target)) return
  const pending = [target]
  while (pending.length) {
    const current = pending.pop()
    let stats
    try {
      stats = lstatSync(current)
    } catch (error) {
      if (error?.code === "ENOENT") continue
      throw error
    }
    if (stats.isSymbolicLink() || !stats.isDirectory()) continue
    chmodSync(current, 0o700)
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        pending.push(path.join(current, entry.name))
      }
    }
  }
}

/**
 * Remove one explicitly-scoped generated directory, retrying transient
 * filesystem races without widening the cleanup target.
 *
 * @param {string} target
 * @param {{
 *   expectedParent: string,
 *   expectedPrefix: string,
 *   remove?: typeof rmSync,
 *   wait?: (delayMs: number) => Promise<void>,
 *   maxAttempts?: number,
 *   retryDelayMs?: number,
 * }} options
 * @returns {Promise<void>}
 */
export async function removeGeneratedPath(target, {
  expectedParent,
  expectedPrefix,
  remove = rmSync,
  wait = (delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs)),
  maxAttempts = 10,
  retryDelayMs = 100,
}) {
  const resolvedTarget = path.resolve(target)
  const resolvedParent = path.resolve(expectedParent)
  const basename = path.basename(resolvedTarget)
  if (
    path.dirname(resolvedTarget) !== resolvedParent ||
    !basename.startsWith(expectedPrefix) ||
    basename.length <= expectedPrefix.length
  ) {
    throw new Error(`Refusing unsafe generated cleanup target: ${resolvedTarget}`)
  }

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      makeGeneratedDirectoryTreeWritable(resolvedTarget)
      remove(resolvedTarget, { recursive: true, force: true })
      return
    } catch (error) {
      if (!retryableCleanupCodes.has(error?.code) || attempt === maxAttempts) {
        throw error
      }
      await wait(retryDelayMs)
    }
  }
}

export function assertNoMutableRepoDatabasePaths(logText, repoRoot) {
  const normalizedLog = String(logText).replaceAll("\\", "/")
  const normalizedRoot = path.resolve(repoRoot).replaceAll("\\", "/")
  const escapedRoots = [
    `${normalizedRoot}/Databases/`,
    `${normalizedRoot}/tldw_Server_API/Databases/`,
  ]
  const absoluteLeak = escapedRoots.find((root) => normalizedLog.includes(root))
  const relativeLeak = normalizedLog.match(/\b(?:db_path=|database:\s*)Databases\/[^\s]+\.db\b/)
  if (absoluteLeak || relativeLeak) {
    throw new Error(
      `Live Tier mutable database path escaped the disposable profile: ${absoluteLeak ?? relativeLeak[0]}`
    )
  }
}

export function assertOnlyLoopbackHttpRequests(logText) {
  const allowedHosts = new Set(["127.0.0.1", "::1", "localhost"])
  for (const line of String(logText).split(/\r?\n/)) {
    if (!/(?:url\.full|HTTP Request:)/.test(line)) continue
    for (const match of line.matchAll(/https?:\/\/[^\s"'<>]+/g)) {
      const candidate = match[0].replace(/[),.;]+$/, "")
      let hostname
      try {
        hostname = new URL(candidate).hostname.toLowerCase().replace(/^\[|\]$/g, "")
      } catch {
        continue
      }
      if (!allowedHosts.has(hostname)) {
        throw new Error(`Live Tier backend made a non-loopback HTTP request to ${hostname}`)
      }
    }
  }
}

async function probeStopped(url) {
  try {
    await fetch(url, { signal: AbortSignal.timeout(750) })
    return false
  } catch {
    return true
  }
}

function commitHash(repoRoot) {
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  return result.status === 0 ? result.stdout.trim() : "unknown"
}

export async function runLiveTierUat({
  options = parseArgs(),
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  baseEnv = process.env,
  signal,
} = {}) {
  if (options.help) {
    process.stdout.write(`${formatUsage()}\n`)
    return { status: "help" }
  }

  const runId = options.runId ?? runIdNow()
  const artifactRoot = path.join(frontendRoot, "test-results/live-tier-uat", runId)
  const profileRoot = path.join(tmpdir(), `tldw-onboarding-uat-${runId}`)
  const nextDistPath = path.join(frontendRoot, `.next-live-tier-${runId}`)
  const logs = {
    mock: path.join(artifactRoot, "mock-openai.log"),
    backend: path.join(artifactRoot, "backend.log"),
    frontend: path.join(artifactRoot, "frontend.log"),
    list: path.join(artifactRoot, "playwright-list.log"),
    run: path.join(artifactRoot, "playwright-run.log"),
  }
  const reportPath = path.join(artifactRoot, "report.md")
  const summaryPath = path.join(artifactRoot, "summary.json")
  const inventoryPath = path.join(artifactRoot, "api-interception-inventory.json")
  const processes = []
  const health = { before: false, after: false, stopped: false }
  let listed = {}
  let results = {}
  let inventory = []
  let skippedTests = []
  let status = "failed"
  let runnerError = null
  let ports = {}
  let profile = null
  let commands = null
  let ownsGeneratedTargets = false
  let ownsArtifactRoot = false

  try {
    assertFreshRunTargets([artifactRoot, profileRoot, nextDistPath])
    ownsGeneratedTargets = true
    ports = await reservePorts(["backend", "web", "mock"])
    mkdirSync(artifactRoot, { recursive: true })
    ownsArtifactRoot = true
    const pythonCommand = resolvePythonCommand({ repoRoot, baseEnv })
    profile = buildLiveTierProfile({
      repoRoot,
      frontendRoot,
      runId,
      mockPort: ports.mock,
      baseTmpDir: tmpdir(),
      pythonCommand,
    })
    commands = buildCommands({
      repoRoot,
      frontendRoot,
      ports,
      profile,
      projects: options.projects,
      workers: options.workers,
      runId,
      grep: options.grep,
      artifactRoot,
      baseEnv,
    })
    signal?.throwIfAborted()
    processes.push(spawnLoggedProcess({ ...commands.mockOpenai, logPath: logs.mock }))
    await waitForHttpOk(`${commands.urls.mock}/health`, { timeoutMs: 30_000, signal })

    const auth = await runCommand(commands.authInit, logs.backend, { signal })
    signal?.throwIfAborted()
    if (auth.code !== 0) throw new Error(`AuthNZ initialization failed with exit code ${auth.code}`)

    processes.push(spawnLoggedProcess({ ...commands.backend, logPath: logs.backend }))
    await waitForHttpOk(`${commands.urls.backend}/api/v1/health`, {
      timeoutMs: 120_000,
      headers: { "X-API-KEY": commands.backendEnv.SINGLE_USER_API_KEY },
      signal,
    })

    processes.push(spawnLoggedProcess({ ...commands.frontend, logPath: logs.frontend }))
    await waitForHttpOk(commands.urls.web, { timeoutMs: 120_000, signal })
    const startupBackendLog = readFileSync(logs.backend, "utf8")
    assertNoMutableRepoDatabasePaths(startupBackendLog, repoRoot)
    assertOnlyLoopbackHttpRequests(startupBackendLog)
    health.before = true

    const listResult = await runCommand(commands.playwrightList, logs.list, { signal })
    signal?.throwIfAborted()
    if (listResult.code !== 0) throw new Error(`Playwright list failed with exit code ${listResult.code}`)
    listed = parseListOutput(listResult.output)
    for (const project of options.projects) {
      if (!listed[project]) throw new Error(`Playwright listed no tests for ${project}`)
    }

    inventory = inventoryProjects(frontendRoot, options.projects)
    writeFileSync(inventoryPath, `${JSON.stringify(inventory, null, 2)}\n`, "utf8")

    if (options.listOnly) {
      status = "listed"
    } else {
      const testResult = await runCommand(commands.playwrightRun, logs.run, { signal })
      signal?.throwIfAborted()
      const playwrightReport = readPlaywrightReport(commands.jsonPath)
      results = summarizePlaywrightReport(playwrightReport)
      skippedTests = collectSkippedTests(playwrightReport)
      assertProjectAccounting({
        projects: options.projects,
        listed,
        results,
        allowSkips: !options.failOnSkip,
      })
      if (testResult.code !== 0) {
        status = "failed"
      } else if (options.failOnSkip && skippedTests.length) {
        status = "failed"
        runnerError = `Strict live-tier UAT rejected ${skippedTests.length} skipped test(s)`
      } else {
        status = "passed"
      }
    }

    await waitForHttpOk(`${commands.urls.backend}/api/v1/health`, {
      timeoutMs: 5_000,
      headers: { "X-API-KEY": commands.backendEnv.SINGLE_USER_API_KEY },
      signal,
    })
    await waitForHttpOk(commands.urls.web, { timeoutMs: 5_000, signal })
    const completedBackendLog = readFileSync(logs.backend, "utf8")
    assertNoMutableRepoDatabasePaths(completedBackendLog, repoRoot)
    assertOnlyLoopbackHttpRequests(completedBackendLog)
    health.after = true
  } catch (error) {
    runnerError = error?.stack ?? error?.message ?? String(error)
    status = "failed"
  } finally {
    try {
      await stopSpawnedProcesses(processes)
    } catch (error) {
      runnerError = [runnerError, error?.stack ?? error?.message ?? String(error)].filter(Boolean).join("\n")
      status = "failed"
    }

    try {
      health.stopped = commands
        ? await Promise.all([
            probeStopped(commands.urls.mock),
            probeStopped(commands.urls.backend),
            probeStopped(commands.urls.web),
          ]).then((values) => values.every(Boolean))
        : processes.length === 0
      assertServicesStopped(health.stopped)
    } catch (error) {
      runnerError = [runnerError, error?.stack ?? error?.message ?? String(error)].filter(Boolean).join("\n")
      status = "failed"
    }

    if (ownsGeneratedTargets) {
      for (const cleanup of [
        {
          target: nextDistPath,
          expectedParent: frontendRoot,
          expectedPrefix: ".next-live-tier-",
        },
        ...(!options.preserveRuntime ? [{
          target: profileRoot,
          expectedParent: tmpdir(),
          expectedPrefix: "tldw-onboarding-uat-",
        }] : []),
      ]) {
        try {
          await removeGeneratedPath(cleanup.target, cleanup)
        } catch (error) {
          runnerError = [
            runnerError,
            `Generated cleanup failed for ${cleanup.target}: ${error?.message ?? String(error)}`,
          ].filter(Boolean).join("\n")
          status = "failed"
        }
      }
    }

    if (ownsArtifactRoot) {
      const artifacts = {
        root: artifactRoot,
        playwrightJson: commands?.jsonPath ?? path.join(artifactRoot, "playwright-results.json"),
        inventory: inventoryPath,
        logs,
      }
      const report = renderMarkdownReport({
        runId,
        commit: commitHash(repoRoot),
        listed,
        results,
        inventory,
        health,
        artifacts,
        certification: isCertificationRun(options),
        skippedTests,
        error: runnerError,
      })
      writeFileSync(reportPath, report, "utf8")
      writeFileSync(summaryPath, `${JSON.stringify({ runId, status, ports, listed, results, skippedTests, health, artifacts, reportPath, error: runnerError }, null, 2)}\n`, "utf8")
    }
  }

  return { status, runId, ports, listed, results, skippedTests, inventory, health, artifactRoot, reportPath, summaryPath, error: runnerError }
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)
if (isEntrypoint) {
  const termination = installTerminationHandlers()
  const result = await runLiveTierUat({ signal: termination.signal })
  termination.dispose()
  if (result.error) process.stderr.write(`${redactText(result.error)}\n`)
  const signalName = termination.signal.reason?.signalName
  process.exitCode = signalName === "SIGINT"
    ? 130
    : signalName === "SIGTERM"
      ? 143
      : result.status === "passed" || result.status === "listed" || result.status === "help"
        ? 0
        : 1
}
