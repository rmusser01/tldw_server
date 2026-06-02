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
const artifactMarkerFile = ".onboarding-uat-artifacts"
const defaultMaxScanFiles = 5_000
const defaultMaxScanBytes = 5 * 1024 * 1024

export const SYNTHETIC_SECRETS = [
  "sk-uat-mock-openai",
  "THIS-IS-A-SECURE-KEY-123-UAT",
]

const genericSecretPatterns = [
  /\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\s*=\s*(?!\[REDACTED\]\b)[^\s]+/i,
  /"(?:[^"]*(?:api[_-]?key|token|secret)[^"]*)"\s*:\s*"(?!\[REDACTED\]")[^"]+"/i,
  /Bearer\s+sk-[A-Za-z0-9._-]+/i,
  /x-api-key:\s*(?!\[REDACTED\]\b)[A-Za-z0-9._-]+/i,
  /"x-api-key"\s*:\s*"(?!\[REDACTED\]")[^"]+"/i,
  /\bsk-[A-Za-z0-9._-]{8,}\b/,
  /\bgh[pousr]_[A-Za-z0-9_]{8,}\b/,
  /\bxox[baprs]-[A-Za-z0-9-]{8,}\b/,
  /\bAKIA[0-9A-Z]{12,}\b/,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
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
  out = out.replace(
    /("x-api-key"\s*:\s*")[^"]+"/gi,
    "$1[REDACTED]\""
  )
  out = out.replace(
    /("([^"]*(?:api[_-]?key|token|secret)[^"]*)"\s*:\s*")[^"]+"/gi,
    "$1[REDACTED]\""
  )
  out = out.replace(
    /\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET))\s*=\s*[^\s]+/gi,
    "$1=[REDACTED]"
  )
  out = out.replace(/\bsk-[A-Za-z0-9._-]{8,}\b/g, "[REDACTED]")
  out = out.replace(/\bgh[pousr]_[A-Za-z0-9_]{8,}\b/g, "[REDACTED]")
  out = out.replace(/\bxox[baprs]-[A-Za-z0-9-]{8,}\b/g, "[REDACTED]")
  out = out.replace(/\bAKIA[0-9A-Z]{12,}\b/g, "[REDACTED]")
  out = out.replace(/-----BEGIN [A-Z ]*PRIVATE KEY-----/g, "[REDACTED]")
  return out
}

export function createRunArtifacts({ frontendRoot = defaultFrontendRoot, runId, preserve = false } = {}) {
  const resolvedRunId = runId ?? createRunId()
  const root = path.join(frontendRoot, "test-results/onboarding-uat", resolvedRunId)
  const artifactBaseRoot = path.join(frontendRoot, "test-results/onboarding-uat")
  const logsDir = path.join(root, "logs")
  const browserDir = path.join(root, "browser")
  const screenshotsDir = path.join(root, "screenshots")
  const runtimeProfileDir = path.join(root, "runtime-profile")

  const artifacts = {
    runId: resolvedRunId,
    preserve,
    artifactBaseRoot,
    root,
    summaryPath: path.join(root, "summary.json"),
    markerPath: path.join(root, artifactMarkerFile),
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
  ensureFile(artifacts.markerPath, "onboarding-uat-artifacts\n")
  ensureFile(artifacts.summaryPath, "{}\n")
  ensureFile(artifacts.logs.backend)
  ensureFile(artifacts.logs.frontend)
  ensureFile(artifacts.logs.mockOpenai)
  ensureFile(artifacts.logs.runner)
  ensureFile(artifacts.browserDiagnosticsPath, "[]\n")
  ensureFile(artifacts.runtimeProfileManifestPath, "{}\n")

  return artifacts
}

function collectScannableFiles(root, { maxFiles = defaultMaxScanFiles } = {}) {
  if (!existsSync(root)) {
    return []
  }

  const entries = readdirSync(root, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const fullPath = path.join(root, entry.name)
    if (entry.isDirectory()) {
      files.push(...collectScannableFiles(fullPath, { maxFiles }))
    } else if (entry.isFile() && scanExtensions.has(path.extname(entry.name))) {
      files.push(fullPath)
    }
    if (files.length > maxFiles) {
      throw new Error(`Artifact leak scan exceeded ${maxFiles} files under ${root}`)
    }
  }
  return files
}

export function assertNoSecretLeaks(
  root,
  { additionalSecrets = [], maxFiles = defaultMaxScanFiles, maxBytes = defaultMaxScanBytes } = {}
) {
  const leaks = []
  const exactSecrets = [...SYNTHETIC_SECRETS, ...additionalSecrets].filter(Boolean)
  for (const file of collectScannableFiles(root, { maxFiles })) {
    const stat = existsSync(file) ? readFileSync(file) : null
    if (stat && stat.byteLength > maxBytes) {
      throw new Error(`Artifact leak scan exceeded ${maxBytes} bytes for ${file}`)
    }
    const content = stat ? stat.toString("utf8") : ""
    for (const secret of exactSecrets) {
      if (content.includes(secret)) {
        leaks.push(file)
        break
      }
    }
    if (genericSecretPatterns.some((pattern) => pattern.test(content))) {
      leaks.push(file)
    }
  }

  if (leaks.length > 0) {
    throw new Error(`Secret leak detected in artifacts: ${[...new Set(leaks)].join(", ")}`)
  }
}

function assertSafeArtifactRoot(artifacts) {
  const root = path.resolve(artifacts.root)
  const baseRoot = path.resolve(
    artifacts.artifactBaseRoot ?? path.join(defaultFrontendRoot, "test-results/onboarding-uat")
  )
  const markerPath = path.resolve(artifacts.markerPath ?? path.join(root, artifactMarkerFile))

  if (!root.startsWith(`${baseRoot}${path.sep}`)) {
    throw new Error(`Refusing to remove artifact root outside onboarding UAT results: ${root}`)
  }
  if (path.basename(root) !== artifacts.runId) {
    throw new Error(`Refusing to remove artifact root without matching run id: ${root}`)
  }
  if (markerPath !== path.join(root, artifactMarkerFile) || !existsSync(markerPath)) {
    throw new Error(`Refusing to remove artifact root without marker file: ${root}`)
  }
}

export function cleanupRunArtifacts(artifacts) {
  if (!artifacts?.preserve && artifacts?.root) {
    assertSafeArtifactRoot(artifacts)
    rmSync(artifacts.root, { recursive: true, force: true })
  }
}
