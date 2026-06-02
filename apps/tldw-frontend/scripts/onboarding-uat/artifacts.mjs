import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const defaultFrontendRoot = path.resolve(moduleDir, "../..")
const scanExtensions = new Set([".json", ".log", ".txt", ".md", ".html"])

export const SYNTHETIC_SECRETS = [
  "sk-uat-mock-openai",
  "THIS-IS-A-SECURE-KEY-123-UAT",
]

function createRunId() {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-")
  const suffix = Math.random().toString(36).slice(2, 8)
  return `${stamp}-${suffix}`
}

function ensureFile(filePath, content = "") {
  mkdirSync(path.dirname(filePath), { recursive: true })
  if (!existsSync(filePath)) {
    writeFileSync(filePath, content, "utf8")
  }
}

export function redactText(value) {
  let out = String(value ?? "")
  for (const secret of SYNTHETIC_SECRETS) {
    out = out.split(secret).join("[REDACTED]")
  }
  out = out.replace(/Bearer\s+sk-[A-Za-z0-9._-]+/g, "Bearer [REDACTED]")
  out = out.replace(/x-api-key:\s*[A-Za-z0-9._-]+/gi, "x-api-key: [REDACTED]")
  return out
}

export function createRunArtifacts({ frontendRoot = defaultFrontendRoot, runId, preserve = false } = {}) {
  const resolvedRunId = runId ?? createRunId()
  const root = path.join(frontendRoot, "test-results/onboarding-uat", resolvedRunId)
  const logsDir = path.join(root, "logs")
  const browserDir = path.join(root, "browser")
  const screenshotsDir = path.join(root, "screenshots")
  const runtimeProfileDir = path.join(root, "runtime-profile")

  const artifacts = {
    runId: resolvedRunId,
    preserve,
    root,
    summaryPath: path.join(root, "summary.json"),
    logs: {
      backend: path.join(logsDir, "backend.log"),
      frontend: path.join(logsDir, "frontend.log"),
      mockOpenai: path.join(logsDir, "mock-openai.log"),
      runner: path.join(logsDir, "runner.log"),
    },
    browserDiagnosticsPath: path.join(browserDir, "console-and-network.json"),
    screenshotsDir,
    runtimeProfileManifestPath: path.join(runtimeProfileDir, "manifest.redacted.json"),
  }

  mkdirSync(screenshotsDir, { recursive: true })
  ensureFile(artifacts.summaryPath, "{}\n")
  ensureFile(artifacts.logs.backend)
  ensureFile(artifacts.logs.frontend)
  ensureFile(artifacts.logs.mockOpenai)
  ensureFile(artifacts.logs.runner)
  ensureFile(artifacts.browserDiagnosticsPath, "[]\n")
  ensureFile(artifacts.runtimeProfileManifestPath, "{}\n")

  return artifacts
}

function collectScannableFiles(root) {
  if (!existsSync(root)) {
    return []
  }

  const entries = readdirSync(root, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name)
    if (entry.isDirectory()) {
      files.push(...collectScannableFiles(fullPath))
    } else if (entry.isFile() && scanExtensions.has(path.extname(entry.name))) {
      files.push(fullPath)
    }
  }
  return files
}

export function assertNoSecretLeaks(root) {
  const leaks = []
  for (const file of collectScannableFiles(root)) {
    const content = readFileSync(file, "utf8")
    for (const secret of SYNTHETIC_SECRETS) {
      if (content.includes(secret)) {
        leaks.push(file)
        break
      }
    }
  }

  if (leaks.length > 0) {
    throw new Error(`Synthetic secret leak detected in artifacts: ${leaks.join(", ")}`)
  }
}

export function cleanupRunArtifacts(artifacts) {
  if (!artifacts?.preserve && artifacts?.root) {
    rmSync(artifacts.root, { recursive: true, force: true })
  }
}
