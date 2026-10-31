#!/usr/bin/env node

import { execFile, spawn } from "node:child_process"
import { mkdirSync, rmSync, writeFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import {
  spawnLoggedProcess,
  stopProcessTree,
} from "./onboarding-uat/processes.mjs"
import {
  MAX_RSS_BYTES,
  descendantUsage,
  evaluateRuntime,
  parseProcessTable,
} from "./dev-runtime-uat-lib.mjs"

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const defaultFrontendRoot = path.resolve(moduleDir, "..")
const allowedBundlers = new Set(["webpack", "turbopack"])
const safeBaseEnvKeys = new Set([
  "PATH",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "TMPDIR",
  "TEMP",
  "TMP",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "NODE_OPTIONS",
  "CI",
])

function optionValues(argv) {
  const values = new Map()
  for (const argument of argv) {
    const match = argument.match(/^--([^=]+)=(.*)$/)
    if (!match) throw new Error(`Unknown option syntax: ${argument}`)
    if (values.has(match[1])) throw new Error(`Duplicate option: --${match[1]}`)
    values.set(match[1], match[2])
  }
  return values
}

function requiredOption(values, name) {
  const value = values.get(name)?.trim()
  if (!value) throw new Error(`--${name} is required`)
  return value
}

function positiveIntegerOption(values, name) {
  const raw = requiredOption(values, name)
  if (!/^\d+$/.test(raw) || Number(raw) <= 0) {
    throw new Error(`--${name} must be a positive integer`)
  }
  return Number(raw)
}

export function parseArgs(argv = process.argv.slice(2)) {
  const values = optionValues(argv)
  const supportedOptions = new Set([
    "bundler",
    "port",
    "warm-idle-ms",
    "output",
  ])
  for (const name of values.keys()) {
    if (!supportedOptions.has(name)) throw new Error(`Unknown option: --${name}`)
  }

  const bundler = requiredOption(values, "bundler")
  if (!allowedBundlers.has(bundler)) {
    throw new Error("--bundler must be webpack or turbopack")
  }
  const port = positiveIntegerOption(values, "port")
  if (port > 65_535) {
    throw new Error("--port must be an integer between 1 and 65535")
  }

  return {
    bundler,
    port,
    warmIdleMs: positiveIntegerOption(values, "warm-idle-ms"),
    output: requiredOption(values, "output"),
    idleCheckIntervalMs: 30_000,
  }
}

export function resolveRuntimeEnvironment(env = process.env) {
  if (env.TLDW_E2E_ALLOW_OFFLINE !== "0") {
    throw new Error("TLDW_E2E_ALLOW_OFFLINE=0 is required")
  }
  const rawBackendUrl = env.TLDW_E2E_SERVER_URL?.trim()
  if (!rawBackendUrl) throw new Error("TLDW_E2E_SERVER_URL is required")

  let backendUrl
  try {
    const parsed = new URL(rawBackendUrl)
    if (!new Set(["http:", "https:"]).has(parsed.protocol)) {
      throw new Error("unsupported protocol")
    }
    parsed.pathname = parsed.pathname.replace(/\/+$/, "")
    backendUrl = parsed.toString().replace(/\/$/, "")
  } catch (error) {
    if (error?.message === "unsupported protocol") {
      throw new Error("TLDW_E2E_SERVER_URL must use http or https")
    }
    throw new Error("TLDW_E2E_SERVER_URL must be a valid URL")
  }

  const apiKey = env.TLDW_E2E_API_KEY?.trim()
  if (!apiKey) throw new Error("TLDW_E2E_API_KEY is required")
  return { backendUrl, apiKey }
}

function safeBaseEnv(baseEnv) {
  const env = {}
  for (const key of safeBaseEnvKeys) {
    if (typeof baseEnv[key] === "string") env[key] = baseEnv[key]
  }
  return env
}

export function buildRuntimeCommands({
  bundler,
  port,
  backendUrl,
  apiKey,
  baseEnv = process.env,
  frontendRoot = process.cwd(),
}) {
  const webUrl = `http://localhost:${port}`
  const env = {
    ...safeBaseEnv(baseEnv),
    NEXT_PUBLIC_API_URL: backendUrl,
    NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
    NEXT_PUBLIC_X_API_KEY: apiKey,
    TLDW_API_KEY: apiKey,
    TLDW_E2E_ALLOW_OFFLINE: "0",
    TLDW_E2E_API_KEY: apiKey,
    TLDW_E2E_SERVER_URL: backendUrl,
    TLDW_SERVER_URL: backendUrl,
    TLDW_WEB_AUTOSTART: "false",
    TLDW_WEB_URL: webUrl,
  }
  return {
    frontend: {
      name: `${bundler}-dev-server`,
      command: "bun",
      args: ["run", `dev:${bundler}`, "--", "-p", String(port)],
      cwd: frontendRoot,
      env,
    },
    fullTraversal: {
      name: "all-pages-traversal",
      command: "bun",
      args: ["run", "e2e:smoke:all-pages:gate"],
      cwd: frontendRoot,
      env,
    },
    secondPass: {
      name: "critical-route-second-pass",
      command: "bun",
      args: [
        "x",
        "playwright",
        "test",
        "e2e/smoke/all-pages.spec.ts",
        "--reporter=line",
        "--grep",
        "Smoke Tests - Key Navigation Targets",
        "--workers=1",
      ],
      cwd: frontendRoot,
      env,
    },
  }
}

export function idleWaitDurations(totalMs, intervalMs) {
  if (!Number.isInteger(totalMs) || totalMs <= 0) {
    throw new Error("total idle duration must be a positive integer")
  }
  if (!Number.isInteger(intervalMs) || intervalMs <= 0) {
    throw new Error("idle check interval must be a positive integer")
  }

  const durations = []
  let remaining = totalMs
  while (remaining > 0) {
    const duration = Math.min(intervalMs, remaining)
    durations.push(duration)
    remaining -= duration
  }
  return durations
}

function readProcessTable() {
  return new Promise((resolve, reject) => {
    execFile(
      "ps",
      ["-axo", "pid=,ppid=,rss=,%cpu=,command="],
      { maxBuffer: 16 * 1024 * 1024 },
      (error, stdout) => {
        if (error) reject(error)
        else resolve(stdout)
      },
    )
  })
}

function runCommand(command) {
  return new Promise((resolve) => {
    const child = spawn(command.command, command.args, {
      cwd: command.cwd,
      env: command.env,
      stdio: "inherit",
    })
    let settled = false
    const finish = (result) => {
      if (settled) return
      settled = true
      resolve(result)
    }
    child.once("error", (error) => finish({ code: null, signal: null, error }))
    child.once("close", (code, signal) => finish({ code, signal }))
  })
}

function writeReport(outputPath, content) {
  mkdirSync(path.dirname(outputPath), { recursive: true })
  writeFileSync(outputPath, content, "utf8")
}

async function probeHttpOk(url, { headers, timeoutMs = 5_000 } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  timer.unref?.()
  try {
    const response = await fetch(url, {
      headers,
      redirect: "manual",
      signal: controller.signal,
    })
    if (response.status < 200 || response.status >= 300) {
      throw new Error(`HTTP ${response.status} from ${url}`)
    }
    return response
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error(`Timed out after ${timeoutMs}ms waiting for ${url}`)
    }
    throw error
  } finally {
    clearTimeout(timer)
  }
}

export async function waitForStrictHttpOk(
  url,
  {
    headers,
    timeoutMs = 30_000,
    intervalMs = 250,
    requestTimeoutMs = 5_000,
    probe = probeHttpOk,
    now = () => Date.now(),
    sleep = (durationMs) => new Promise((resolve) => setTimeout(resolve, durationMs)),
  } = {},
) {
  const startedAt = now()
  let lastError = null

  while (now() - startedAt < timeoutMs) {
    const remainingMs = timeoutMs - (now() - startedAt)
    try {
      return await probe(url, {
        headers,
        timeoutMs: Math.min(requestTimeoutMs, remainingMs),
      })
    } catch (error) {
      lastError = error
    }

    const waitBudgetMs = timeoutMs - (now() - startedAt)
    if (waitBudgetMs <= 0) break
    await sleep(Math.min(intervalMs, waitBudgetMs))
  }

  throw new Error(
    `Timed out waiting for ${url}: ${lastError?.message ?? "no response"}`,
  )
}

const defaultOperations = {
  clearBuildOutput: (frontendRoot) => {
    rmSync(path.resolve(frontendRoot, ".next"), { recursive: true, force: true })
  },
  now: () => Date.now(),
  probeHttpOk,
  readProcessTable,
  runCommand,
  sleep: (durationMs) => new Promise((resolve) => setTimeout(resolve, durationMs)),
  spawnLoggedProcess,
  stopProcessTree,
  waitForHttpOk: waitForStrictHttpOk,
  writeReport,
}

async function sampleRuntime({ phase, rootPid, webUrl, operations }) {
  const processText = await operations.readProcessTable()
  const usage = descendantUsage(parseProcessTable(processText), rootPid)
  let responsive = true
  let healthError = null
  try {
    await operations.probeHttpOk(webUrl, { timeoutMs: 5_000 })
  } catch (error) {
    responsive = false
    healthError = error?.message ?? String(error)
  }

  return {
    phase,
    measuredAt: new Date(operations.now()).toISOString(),
    ...usage,
    responsive,
    ...(healthError ? { healthError } : {}),
  }
}

function commandFailure(command, result) {
  if (result?.error) {
    return `${command.name} failed to start: ${result.error.message}`
  }
  if (result?.code !== 0) {
    return `${command.name} exited with code ${result?.code ?? "unknown"}`
  }
  return null
}

function irreversibleSampleFailure(sample) {
  if (!sample.responsive) {
    return `${sample.phase} became unresponsive: ${sample.healthError ?? "health probe failed"}`
  }
  if (sample.rssBytes >= MAX_RSS_BYTES) {
    return `${sample.phase} reached the RSS guardrail (${sample.rssBytes} bytes)`
  }
  return null
}

async function collectRequiredSample({ phase, rootPid, webUrl, operations, samples }) {
  const sample = await sampleRuntime({ phase, rootPid, webUrl, operations })
  samples.push(sample)
  const irreversibleFailure = irreversibleSampleFailure(sample)
  if (irreversibleFailure) throw new Error(irreversibleFailure)
  return sample
}

export async function runDevRuntimeUat({
  options = parseArgs(),
  baseEnv = process.env,
  frontendRoot = defaultFrontendRoot,
  operations: operationOverrides = {},
} = {}) {
  const operations = { ...defaultOperations, ...operationOverrides }
  const { backendUrl, apiKey } = resolveRuntimeEnvironment(baseEnv)
  const commands = buildRuntimeCommands({
    ...options,
    backendUrl,
    apiKey,
    baseEnv,
    frontendRoot,
  })
  const outputPath = path.isAbsolute(options.output)
    ? options.output
    : path.resolve(frontendRoot, options.output)
  const webUrl = `http://localhost:${options.port}`
  const samples = []
  const startedAt = new Date(operations.now()).toISOString()
  let frontendRecord = null
  let failure = null
  let report

  try {
    operations.clearBuildOutput(frontendRoot)
    await operations.waitForHttpOk(`${backendUrl}/api/v1/health`, {
      timeoutMs: 30_000,
      headers: { "X-API-KEY": apiKey },
    })
    frontendRecord = operations.spawnLoggedProcess({
      ...commands.frontend,
      logPath: `${outputPath}.frontend.log`,
    })
    const rootPid = frontendRecord?.pid ?? frontendRecord?.child?.pid
    if (!rootPid) throw new Error("Frontend process did not expose a PID")

    await operations.waitForHttpOk(webUrl, { timeoutMs: 120_000 })
    await collectRequiredSample({
      phase: "initial",
      rootPid,
      webUrl,
      operations,
      samples,
    })

    const traversalResult = await operations.runCommand(commands.fullTraversal)
    const traversalFailure = commandFailure(commands.fullTraversal, traversalResult)
    if (traversalFailure) throw new Error(traversalFailure)
    await collectRequiredSample({
      phase: "post-traversal",
      rootPid,
      webUrl,
      operations,
      samples,
    })

    const waits = idleWaitDurations(
      options.warmIdleMs,
      options.idleCheckIntervalMs,
    )
    for (const [index, durationMs] of waits.entries()) {
      await operations.sleep(durationMs)
      const phase = index === waits.length - 1
        ? "post-idle"
        : `warm-idle-${index + 1}`
      await collectRequiredSample({
        phase,
        rootPid,
        webUrl,
        operations,
        samples,
      })
    }

    const secondPassResult = await operations.runCommand(commands.secondPass)
    const secondPassFailure = commandFailure(commands.secondPass, secondPassResult)
    if (secondPassFailure) throw new Error(secondPassFailure)
    await collectRequiredSample({
      phase: "second-pass",
      rootPid,
      webUrl,
      operations,
      samples,
    })
  } catch (error) {
    failure = error?.message ?? String(error)
  } finally {
    const evaluation = evaluateRuntime(samples)
    report = {
      schemaVersion: 1,
      bundler: options.bundler,
      status: !failure && evaluation.qualified ? "qualified" : "failed",
      startedAt,
      finishedAt: new Date(operations.now()).toISOString(),
      backendUrl,
      webUrl,
      warmIdleMs: options.warmIdleMs,
      idleCheckIntervalMs: options.idleCheckIntervalMs,
      commands: {
        frontend: [commands.frontend.command, ...commands.frontend.args],
        fullTraversal: [
          commands.fullTraversal.command,
          ...commands.fullTraversal.args,
        ],
        secondPass: [commands.secondPass.command, ...commands.secondPass.args],
      },
      samples,
      evaluation,
      failure,
    }
    try {
      await operations.writeReport(
        outputPath,
        `${JSON.stringify(report, null, 2)}\n`,
      )
    } finally {
      if (frontendRecord) await operations.stopProcessTree(frontendRecord)
    }
  }

  return report
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)
if (isEntrypoint) {
  try {
    const report = await runDevRuntimeUat()
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
    process.exitCode = report.status === "qualified" ? 0 : 1
  } catch (error) {
    process.stderr.write(`${error?.message ?? String(error)}\n`)
    process.exitCode = 1
  }
}
