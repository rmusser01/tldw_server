import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { redactText } from "./artifacts.mjs"

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const defaultFrontendRoot = path.resolve(moduleDir, "../..")
const syntheticApiKey = "THIS-IS-A-SECURE-KEY-123-UAT"
const syntheticOpenAiKey = "sk-uat-mock-openai"
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
  "PYTHONPATH",
  "VIRTUAL_ENV",
  "SSL_CERT_FILE",
  "REQUESTS_CA_BUNDLE",
  "NODE_EXTRA_CA_CERTS",
  "SYSTEMROOT",
  "WINDIR",
])

function createRunId() {
  return new Date().toISOString().replace(/[:.]/g, "-")
}

function patchIniValue(text, sectionName, key, value, { addIfSectionExists = true } = {}) {
  const lines = text.split(/\r?\n/)
  let currentSection = null
  let sectionStart = -1
  let sectionEnd = lines.length
  let keyIndex = -1

  for (let index = 0; index < lines.length; index += 1) {
    const sectionMatch = lines[index].match(/^\s*\[([^\]]+)\]\s*$/)
    if (sectionMatch) {
      if (currentSection === sectionName && sectionEnd === lines.length) {
        sectionEnd = index
      }
      currentSection = sectionMatch[1]
      if (currentSection === sectionName) {
        sectionStart = index
        sectionEnd = lines.length
      }
      continue
    }

    if (currentSection === sectionName) {
      const keyMatch = lines[index].match(/^\s*([^#;][^=]*?)\s*=/)
      if (keyMatch && keyMatch[1].trim() === key) {
        keyIndex = index
        break
      }
    }
  }

  if (keyIndex >= 0) {
    lines[keyIndex] = `${key} = ${value}`
    return lines.join("\n")
  }

  if (addIfSectionExists && sectionStart >= 0) {
    lines.splice(sectionEnd, 0, `${key} = ${value}`)
    return lines.join("\n")
  }

  return text
}

function writeEnvFile(envPath, profileRoot, mockPort, fixtureRoot) {
  const databaseDir = path.join(profileRoot, "Databases")
  const usersDbPath = path.join(databaseDir, "users.db")
  const mockBaseUrl = `http://127.0.0.1:${mockPort}/v1`
  const lines = [
    "AUTH_MODE=single_user",
    `SINGLE_USER_API_KEY=${syntheticApiKey}`,
    "DEFAULT_LLM_PROVIDER=openai",
    `OPENAI_API_KEY=${syntheticOpenAiKey}`,
    `OPENAI_API_BASE_URL=${mockBaseUrl}`,
    `DATABASE_URL=sqlite:///${usersDbPath}`,
    `USER_DB_BASE_DIR_ALLOWED_ROOTS=${databaseDir}`,
    `TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS=${databaseDir}`,
    `INGESTION_SOURCE_ALLOWED_ROOTS=${fixtureRoot}`,
    `TLDW_INGESTION_SOURCE_ALLOWED_ROOTS=${fixtureRoot}`,
    "TLDW_SETUP_ALLOW_REMOTE=false",
    "WORKFLOWS_EGRESS_BLOCK_PRIVATE=false",
    `WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443,${mockPort}`,
    "",
  ]
  writeFileSync(envPath, lines.join("\n"), "utf8")
}

function safeBaseEnv(baseEnv) {
  const env = {}
  for (const key of safeBaseEnvKeys) {
    if (baseEnv[key] !== undefined) {
      env[key] = baseEnv[key]
    }
  }
  return env
}

export function createRuntimeProfile({
  repoRoot,
  frontendRoot = defaultFrontendRoot,
  runId,
  mockPort,
  baseTmpDir = tmpdir(),
}) {
  if (!repoRoot) {
    throw new Error("createRuntimeProfile requires repoRoot")
  }
  if (!mockPort) {
    throw new Error("createRuntimeProfile requires mockPort")
  }

  const resolvedRunId = runId ?? createRunId()
  const root = path.join(baseTmpDir, `tldw-onboarding-uat-${resolvedRunId}`)
  const configDir = path.join(root, "Config_Files")
  const databaseDir = path.join(root, "Databases")
  const userDatabasesDir = path.join(databaseDir, "user_databases")
  const uploadsDir = path.join(root, "uploads")
  const logsDir = path.join(root, "logs")
  const configPath = path.join(configDir, "config.txt")
  const envPath = path.join(configDir, ".env")
  const usersDbPath = path.join(databaseDir, "users.db")
  const sourceConfigPath = path.join(repoRoot, "tldw_Server_API/Config_Files/config.txt")

  mkdirSync(configDir, { recursive: true })
  mkdirSync(userDatabasesDir, { recursive: true })
  mkdirSync(uploadsDir, { recursive: true })
  mkdirSync(logsDir, { recursive: true })

  copyFileSync(sourceConfigPath, configPath)

  const mockBaseUrl = `http://127.0.0.1:${mockPort}/v1`
  const fixtureRoot = path.join(frontendRoot, "e2e/fixtures/media")
  let configText = readFileSync(configPath, "utf8")
  configText = patchIniValue(configText, "Setup", "enable_first_time_setup", "true")
  configText = patchIniValue(configText, "Setup", "setup_completed", "false")
  configText = patchIniValue(configText, "AuthNZ", "auth_mode", "single_user")
  configText = patchIniValue(configText, "AuthNZ", "single_user_api_key", syntheticApiKey)
  configText = patchIniValue(configText, "API", "openai_model", "gpt-4.1-mini")
  configText = patchIniValue(configText, "API", "custom_openai_api_ip", mockBaseUrl)
  configText = patchIniValue(configText, "API", "custom_openai_api_model", "local-uat-chat")
  configText = patchIniValue(configText, "Local-API", "ollama_api_IP", mockBaseUrl, {
    addIfSectionExists: false,
  })
  configText = patchIniValue(configText, "Local-API", "ollama_model", "llama3.2:3b", {
    addIfSectionExists: false,
  })
  configText = patchIniValue(
    configText,
    "TTS-Settings",
    "USER_DB_BASE_DIR",
    path.join(userDatabasesDir),
    { addIfSectionExists: false }
  )
  configText = patchIniValue(
    configText,
    "Files",
    "ingestion_source_allowed_roots",
    fixtureRoot
  )
  writeFileSync(configPath, configText, "utf8")
  writeEnvFile(envPath, root, mockPort, fixtureRoot)

  return {
    runId: resolvedRunId,
    root,
    configDir,
    configPath,
    envPath,
    databaseDir,
    usersDbPath,
    userDatabasesDir,
    uploadsDir,
    logsDir,
    fixtureRoot,
  }
}

export function buildBackendEnv({ profile, mockPort, baseEnv = process.env }) {
  if (!profile) {
    throw new Error("buildBackendEnv requires profile")
  }
  const mockBaseUrl = `http://127.0.0.1:${mockPort}/v1`
  const fixtureRoot =
    profile.fixtureRoot ?? path.join(defaultFrontendRoot, "e2e/fixtures/media")
  return {
    ...safeBaseEnv(baseEnv),
    TLDW_CONFIG_FILE: profile.configPath,
    TLDW_ENV_FILE: profile.envPath,
    DATABASE_URL: `sqlite:///${profile.usersDbPath}`,
    AUTH_MODE: "single_user",
    SINGLE_USER_API_KEY: syntheticApiKey,
    DEFAULT_LLM_PROVIDER: "openai",
    OPENAI_API_KEY: syntheticOpenAiKey,
    OPENAI_API_BASE_URL: mockBaseUrl,
    USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
    TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS: profile.databaseDir,
    INGESTION_SOURCE_ALLOWED_ROOTS: fixtureRoot,
    TLDW_INGESTION_SOURCE_ALLOWED_ROOTS: fixtureRoot,
    TLDW_SETUP_ALLOW_REMOTE: "false",
    WORKFLOWS_EGRESS_BLOCK_PRIVATE: "false",
    WORKFLOWS_EGRESS_ALLOWED_PORTS: `80,443,${mockPort}`,
  }
}

export function createRedactedProfileManifest(profile) {
  return redactText(JSON.stringify(profile, null, 2))
}
