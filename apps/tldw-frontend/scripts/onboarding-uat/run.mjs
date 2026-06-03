#!/usr/bin/env node

import { spawn } from "node:child_process"
import { appendFileSync, cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import {
  assertNoSecretLeaks,
  cleanupRunArtifacts,
  createRunArtifacts,
  redactText,
} from "./artifacts.mjs"
import { reservePorts } from "./ports.mjs"
import {
  buildBackendEnv,
  createRedactedProfileManifest,
  createRuntimeProfile,
} from "./profile.mjs"
import {
  spawnLoggedProcess,
  stopProcessTree,
  waitForHttpOk,
} from "./processes.mjs"

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRootDefault = path.resolve(moduleDir, "../..")
const repoRootDefault = path.resolve(frontendRootDefault, "../..")
const apiKey = "THIS-IS-A-SECURE-KEY-123-UAT"
const validViewports = new Set(["desktop", "mobile", "all"])
const mockConfigDir = "e2e/onboarding-uat/mock-openai/configs"
const playwrightConfig = "e2e/onboarding-uat/playwright.config.ts"
const scenarioMockConfigs = Object.freeze({
  "hosted-openai-first-chat": "hosted-success.json",
  "local-openai-discovered-model-first-chat": "local-success.json",
  "local-openai-first-chat": "local-success.json",
  "local-openai-manual-model-first-chat": "local-models-unavailable.json",
  "first-source-after-chat": "hosted-success.json",
  "provider-retry-recovery": "chat-fail-once.json",
  "model-unavailable-recovery": "model-unavailable.json",
  "local-openai-model-unavailable-recovery": "local-model-unavailable.json",
  "setup-endpoint-recovery": "local-success.json",
  "local-to-hosted-switch-state-isolated": "hosted-success.json",
})

export function formatUsage() {
  return [
    "Usage: bun run e2e:onboarding:uat -- [options]",
    "",
    "Options:",
    "  --scenario <id>          Run one onboarding UAT scenario.",
    "  --viewport <mode>        desktop, mobile, or all. Default: all.",
    "  --mock-config <name>     Mock config file name or path. Default: scenario config, otherwise hosted-success.json.",
    "  --preserve-runtime       Keep the isolated runtime profile after the run.",
    "  --preserve-artifacts     Keep artifacts after the run. Default: true.",
    "  --no-preserve-artifacts  Delete artifacts after the run if the run completes cleanup.",
    "  --reviewed-evidence      Copy reviewed evidence after Playwright completes.",
    "  --help                   Print this help and exit.",
  ].join("\n")
}

export function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    scenario: null,
    viewport: "all",
    mockConfig: null,
    preserveRuntime: false,
    preserveArtifacts: true,
    reviewedEvidence: false,
    help: false,
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--help" || arg === "-h") {
      options.help = true
    } else if (arg === "--scenario") {
      options.scenario = requireValue(argv, (index += 1), arg)
    } else if (arg === "--viewport") {
      const viewport = requireValue(argv, (index += 1), arg)
      if (!validViewports.has(viewport)) {
        throw new Error(`Invalid --viewport ${viewport}; expected desktop, mobile, or all`)
      }
      options.viewport = viewport
    } else if (arg === "--mock-config") {
      options.mockConfig = requireValue(argv, (index += 1), arg)
    } else if (arg === "--preserve-runtime") {
      options.preserveRuntime = true
    } else if (arg === "--preserve-artifacts") {
      options.preserveArtifacts = true
    } else if (arg === "--no-preserve-artifacts") {
      options.preserveArtifacts = false
    } else if (arg === "--reviewed-evidence") {
      options.reviewedEvidence = true
    } else {
      throw new Error(`Unknown option: ${arg}`)
    }
  }

  return options
}

function requireValue(argv, index, flag) {
  const value = argv[index]
  if (!value || value.startsWith("--")) {
    throw new Error(`${flag} requires a value`)
  }
  return value
}

function resolveMockConfig(frontendRoot, mockConfig) {
  if (path.isAbsolute(mockConfig)) {
    return mockConfig
  }
  return path.join(frontendRoot, mockConfigDir, mockConfig)
}

export function resolveEffectiveMockConfig({ scenario = null, mockConfig = null } = {}) {
  if (mockConfig) {
    return mockConfig
  }
  if (scenario && scenarioMockConfigs[scenario]) {
    return scenarioMockConfigs[scenario]
  }
  return "hosted-success.json"
}

function projectForViewport(viewport) {
  if (viewport === "desktop") {
    return "uat-desktop"
  }
  if (viewport === "mobile") {
    return "uat-mobile"
  }
  return null
}

/**
 * @param {NodeJS.ProcessEnv | Record<string, string | undefined>} baseEnv
 * @param {string[]} keys
 * @returns {Record<string, string>}
 */
function safeEnv(baseEnv, keys) {
  const env = /** @type {Record<string, string>} */ ({})
  for (const key of keys) {
    const value = baseEnv[key]
    if (typeof value === "string") {
      env[key] = value
    }
  }
  return env
}

/**
 * @param {{ repoRoot?: string, baseEnv?: NodeJS.ProcessEnv | Record<string, string | undefined> }} [options]
 * @returns {string}
 */
export function resolvePythonCommand({ repoRoot = repoRootDefault, baseEnv = process.env } = {}) {
  if (baseEnv.PYTHON) {
    return baseEnv.PYTHON
  }
  if (baseEnv.PYTHON3) {
    return baseEnv.PYTHON3
  }

  const candidates = [
    path.join(repoRoot, ".venv/bin/python"),
    path.join(repoRoot, ".venv/bin/python3"),
    path.join(repoRoot, ".venv/Scripts/python.exe"),
    path.resolve(repoRoot, "../..", ".venv/bin/python"),
    path.resolve(repoRoot, "../..", ".venv/bin/python3"),
    path.resolve(repoRoot, "../..", ".venv/Scripts/python.exe"),
  ]
  return candidates.find((candidate) => existsSync(candidate)) ?? "python"
}

function withPythonPath(env, paths) {
  const values = paths.filter(Boolean)
  if (env.PYTHONPATH) {
    values.push(env.PYTHONPATH)
  }
  return {
    ...env,
    PYTHONPATH: values.join(path.delimiter),
  }
}

function reviewedEvidencePath(repoRoot, runId) {
  if (!runId || !/^[A-Za-z0-9._-]+$/.test(runId)) {
    throw new Error(`Invalid reviewed evidence run id: ${runId}`)
  }
  return path.join(repoRoot, "Docs/Product/WebUI/evidence/onboarding_uat", runId)
}

export function copyReviewedEvidence({ artifacts, repoRoot = repoRootDefault }) {
  if (!artifacts?.root || !artifacts?.runId) {
    throw new Error("copyReviewedEvidence requires artifacts with root and runId")
  }
  if (!existsSync(artifacts.root)) {
    throw new Error(`Artifact root does not exist: ${artifacts.root}`)
  }

  assertNoSecretLeaks(artifacts.root)
  const destination = reviewedEvidencePath(repoRoot, artifacts.runId)
  rmSync(destination, { recursive: true, force: true })
  cpSync(artifacts.root, destination, { recursive: true })
  return destination
}

/**
 * @param {{
 *   repoRoot?: string,
 *   frontendRoot?: string,
 *   ports: { backend: number, web: number, mock: number },
 *   profile: Record<string, string>,
 *   mockConfig?: string,
 *   scenario?: string | null,
 *   viewport?: string,
 *   runId?: string,
 *   artifactRoot?: string,
 *   baseEnv?: NodeJS.ProcessEnv | Record<string, string | undefined>,
 * }} options
 */
export function buildCommands({
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  ports,
  profile,
  mockConfig = "hosted-success.json",
  scenario = null,
  viewport = "all",
  runId = "onboarding-uat",
  artifactRoot = "",
  baseEnv = process.env,
}) {
  if (!ports?.backend || !ports?.web || !ports?.mock) {
    throw new Error("buildCommands requires backend, web, and mock ports")
  }
  if (!profile) {
    throw new Error("buildCommands requires a runtime profile")
  }

  const mockConfigPath = resolveMockConfig(frontendRoot, mockConfig)
  const mockOpenaiRoot = path.join(frontendRoot, "e2e/onboarding-uat/mock-openai")
  const serverUrl = `http://127.0.0.1:${ports.backend}`
  const webUrl = `http://localhost:${ports.web}`
  const mockUrl = `http://127.0.0.1:${ports.mock}/v1`
  const pythonCommand = resolvePythonCommand({ repoRoot, baseEnv })
  const backendEnv = buildBackendEnv({ profile, mockPort: ports.mock, baseEnv })
  const mockEnv = withPythonPath(
    safeEnv(baseEnv, [
      "PATH",
      "HOME",
      "USER",
      "LOGNAME",
      "SHELL",
      "TMPDIR",
      "LANG",
      "PYTHONPATH",
      "VIRTUAL_ENV",
    ]),
    [path.join(repoRoot, "mock_openai_server")]
  )
  const frontendEnv = {
    ...safeEnv(baseEnv, [
      "PATH",
      "HOME",
      "USER",
      "LOGNAME",
      "SHELL",
      "TMPDIR",
      "LANG",
    ]),
    NEXT_PUBLIC_API_URL: serverUrl,
    NEXT_PUBLIC_API_VERSION: "v1",
    NEXT_PUBLIC_X_API_KEY: apiKey,
    TLDW_SERVER_URL: serverUrl,
    TLDW_WEB_URL: webUrl,
  }
  const playwrightArgs = [
    "playwright",
    "test",
    "-c",
    playwrightConfig,
    "--reporter=line",
  ]
  if (scenario) {
    playwrightArgs.push("--grep", scenario)
  }
  const project = projectForViewport(viewport)
  if (project) {
    playwrightArgs.push("--project", project)
  }

  return {
    mockOpenai: {
      name: "mock-openai",
      command: pythonCommand,
      args: [
        "-m",
        "mock_openai.server",
        "--config",
        mockConfigPath,
        "--host",
        "127.0.0.1",
        "--port",
        String(ports.mock),
      ],
      cwd: mockOpenaiRoot,
      env: mockEnv,
    },
    authInit: {
      name: "auth-init",
      command: pythonCommand,
      args: [
        "-m",
        "tldw_Server_API.app.core.AuthNZ.initialize",
        "--non-interactive",
      ],
      cwd: repoRoot,
      env: backendEnv,
    },
    backend: {
      name: "backend",
      command: pythonCommand,
      args: [
        "-m",
        "uvicorn",
        "tldw_Server_API.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(ports.backend),
      ],
      cwd: repoRoot,
      env: backendEnv,
    },
    frontend: {
      name: "frontend",
      command: "bun",
      args: ["run", "dev", "--", "-p", String(ports.web)],
      cwd: frontendRoot,
      env: frontendEnv,
    },
    playwright: {
      name: "playwright",
      command: "bunx",
      args: playwrightArgs,
      cwd: frontendRoot,
      env: {
        ...frontendEnv,
        TLDW_ONBOARDING_UAT: "1",
        TLDW_ONBOARDING_UAT_RUN_ID: runId,
        TLDW_ONBOARDING_UAT_ARTIFACT_ROOT: artifactRoot,
        TLDW_WEB_URL: webUrl,
        TLDW_SERVER_URL: serverUrl,
        TLDW_API_KEY: apiKey,
        TLDW_MOCK_OPENAI_URL: mockUrl,
      },
    },
  }
}

async function runCommand(commandRecord, { logPath }) {
  return new Promise((resolve) => {
    const child = spawn(commandRecord.command, commandRecord.args, {
      cwd: commandRecord.cwd,
      env: commandRecord.env,
      stdio: ["ignore", "pipe", "pipe"],
    })
    const write = (chunk) => {
      mkdirSync(path.dirname(logPath), { recursive: true })
      appendFileSync(logPath, redactText(chunk), "utf8")
    }
    child.stdout?.on("data", write)
    child.stderr?.on("data", write)
    child.once("error", (error) => {
      write(`[${commandRecord.name}] error: ${error.message}\n`)
      resolve({ code: null, signal: null, error })
    })
    child.once("close", (code, signal) => {
      resolve({ code, signal })
    })
  })
}

export async function runOnboardingUat({
  options = parseArgs(),
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  baseEnv = process.env,
} = {}) {
  if (options.help) {
    process.stdout.write(`${formatUsage()}\n`)
    return { status: "help" }
  }

  const ports = await reservePorts(["backend", "web", "mock"])
  const artifacts = createRunArtifacts({
    frontendRoot,
    preserve: options.preserveArtifacts,
  })
  const profile = createRuntimeProfile({
    repoRoot,
    frontendRoot,
    runId: artifacts.runId,
    mockPort: ports.mock,
  })
  writeFileSync(
    artifacts.runtimeProfileManifestPath,
    `${createRedactedProfileManifest(profile)}\n`,
    "utf8"
  )
  const effectiveMockConfig = resolveEffectiveMockConfig(options)
  const commands = buildCommands({
    repoRoot,
    frontendRoot,
    ports,
    profile,
    mockConfig: effectiveMockConfig,
    scenario: options.scenario,
    viewport: options.viewport,
    runId: artifacts.runId,
    artifactRoot: artifacts.root,
    baseEnv,
  })

  const processes = []
  try {
    if (!existsSync(resolveMockConfig(frontendRoot, effectiveMockConfig))) {
      throw new Error(`Mock config not found: ${effectiveMockConfig}`)
    }

    processes.push(
      spawnLoggedProcess({ ...commands.mockOpenai, logPath: artifacts.logs.mockOpenai })
    )
    await waitForHttpOk(`http://127.0.0.1:${ports.mock}/health`, { timeoutMs: 30_000 })

    const authInitResult = await runCommand(commands.authInit, {
      logPath: artifacts.logs.backend,
    })
    if (authInitResult.code !== 0) {
      throw new Error(`AuthNZ initialization failed with exit code ${authInitResult.code}`)
    }

    processes.push(
      spawnLoggedProcess({ ...commands.backend, logPath: artifacts.logs.backend })
    )
    await waitForHttpOk(`http://127.0.0.1:${ports.backend}/api/v1/health`, {
      timeoutMs: 120_000,
      headers: { "X-API-KEY": apiKey },
    })

    processes.push(
      spawnLoggedProcess({ ...commands.frontend, logPath: artifacts.logs.frontend })
    )
    await waitForHttpOk(`http://localhost:${ports.web}`, { timeoutMs: 120_000 })

    const result = await runCommand(commands.playwright, {
      logPath: artifacts.logs.runner,
    })
    const status = result.code === 0 ? "passed" : "failed"
    const reviewedEvidenceRoot = options.reviewedEvidence
      ? reviewedEvidencePath(repoRoot, artifacts.runId)
      : null
    writeFileSync(
      artifacts.summaryPath,
      `${JSON.stringify(
        {
          run_id: artifacts.runId,
          status,
          ports,
          mock_config: effectiveMockConfig,
          reviewed_evidence_root: reviewedEvidenceRoot,
        },
        null,
        2
      )}\n`,
      "utf8"
    )
    assertNoSecretLeaks(artifacts.root)
    const evidenceRoot = options.reviewedEvidence
      ? copyReviewedEvidence({ artifacts, repoRoot })
      : null
    return { status, artifacts, ports, commands, result, evidenceRoot }
  } finally {
    await Promise.allSettled(processes.reverse().map((record) => stopProcessTree(record)))
    if (!options.preserveRuntime) {
      rmSync(profile.root, { recursive: true, force: true })
    }
    cleanupRunArtifacts(artifacts)
  }
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)

if (isEntrypoint) {
  try {
    const result = await runOnboardingUat()
    process.exitCode = result.status === "failed" ? 1 : 0
  } catch (error) {
    process.stderr.write(`${redactText(error?.stack || error?.message || error)}\n`)
    process.exitCode = 1
  }
}
