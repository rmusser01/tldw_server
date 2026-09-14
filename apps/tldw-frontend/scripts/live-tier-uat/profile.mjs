import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import path from "node:path"
import {
  buildBackendEnv,
  createRuntimeProfile,
} from "../onboarding-uat/profile.mjs"

const deterministicModel = "local-uat-chat"

/**
 * Create the Tier UAT profile from the established onboarding isolation profile.
 */
export function buildLiveTierProfile({
  repoRoot,
  frontendRoot,
  runId,
  mockPort,
  baseTmpDir,
  pythonCommand,
}) {
  if (!pythonCommand) {
    throw new Error("buildLiveTierProfile requires pythonCommand")
  }
  const profile = createRuntimeProfile({
    repoRoot,
    frontendRoot,
    runId,
    mockPort,
    baseTmpDir,
  })
  const logsDir = path.join(profile.root, "logs", "live-tier-uat")
  const reportsDir = path.join(profile.root, "reports", "live-tier-uat")
  const evaluationsPath = path.join(
    profile.userDatabasesDir,
    "1",
    "evaluations",
    "evaluations.db"
  )
  const jobsPath = path.join(profile.databaseDir, "jobs.db")
  const acpSessionsPath = path.join(
    profile.userDatabasesDir,
    "1",
    "acp_sessions.db"
  )
  const acpWorkspaceRootBase = path.join(profile.root, "acp-workspaces")
  const watchlistTemplateDir = path.join(profile.root, "watchlist-templates")
  const acpRunnerConfigPath = path.join(
    profile.configDir,
    "acp_runner_home",
    ".tldw-agent",
    "config.yaml"
  )
  const mcpMediaPath = path.join(profile.userDatabasesDir, "1", "Media_DB_v2.db")
  const mcpDocsPath = path.join(profile.databaseDir, "mcp_docs.db")
  const acpAuditPath = path.join(profile.databaseDir, "acp_audit.db")
  const monitoringAlertsPath = path.join(profile.databaseDir, "monitoring_alerts.db")
  const systemLogFilePath = path.join(logsDir, "system_logs.jsonl")
  const mcpModulesConfigPath = path.join(profile.configDir, "mcp_modules.yaml")
  const sourceMcpModulesConfigPath = path.join(
    repoRoot,
    "tldw_Server_API/Config_Files/mcp_modules.yaml"
  )

  mkdirSync(logsDir, { recursive: true })
  mkdirSync(reportsDir, { recursive: true })
  mkdirSync(acpWorkspaceRootBase, { recursive: true })
  mkdirSync(watchlistTemplateDir, { recursive: true })
  mkdirSync(path.dirname(acpRunnerConfigPath), { recursive: true })
  const acpStubPath = path.join(repoRoot, "Helper_Scripts/acp_stub_agent.py")
  writeFileSync(
    acpRunnerConfigPath,
    [
      "agents:",
      "  default: opencode",
      "  agents:",
      "    - type: opencode",
      "      name: Live Tier ACP Stub",
      "      description: Deterministic downstream ACP process for isolated live-tier UAT.",
      `      command: ${JSON.stringify(pythonCommand)}`,
      "      args:",
      `        - ${JSON.stringify(acpStubPath)}`,
      "      env: []",
      "      entrypoint_strategy: native_acp",
      "",
    ].join("\n"),
    "utf8"
  )
  const isolatedMcpConfig = readFileSync(sourceMcpModulesConfigPath, "utf8")
    .replace(
      "db_path: Databases/user_databases/1/Media_DB_v2.db",
      `db_path: ${mcpMediaPath}`
    )
    .replace("db_path: Databases/mcp_docs.db", `db_path: ${mcpDocsPath}`)
  if (isolatedMcpConfig.includes("db_path: Databases/")) {
    throw new Error("Live Tier MCP module config still contains a mutable relative database path")
  }
  writeFileSync(mcpModulesConfigPath, isolatedMcpConfig, "utf8")

  return {
    ...profile,
    repoRoot,
    runDir: profile.root,
    logsDir,
    reportsDir,
    acpWorkspaceRootBase,
    watchlistTemplateDir,
    systemLogFilePath,
    acpRunnerConfigPath,
    mcpModulesConfigPath,
    databasePaths: {
      users: profile.usersDbPath,
      perUserRoot: profile.userDatabasesDir,
      evaluations: evaluationsPath,
      jobs: jobsPath,
      acpSessions: acpSessionsPath,
      mcpMedia: mcpMediaPath,
      mcpDocs: mcpDocsPath,
      acpAudit: acpAuditPath,
      monitoringAlerts: monitoringAlertsPath,
    },
  }
}

/**
 * Extend the established secret-safe backend environment with deterministic
 * provider aliases and an isolated evaluations database.
 *
 * @param {{
 *   profile: ReturnType<typeof buildLiveTierProfile>,
 *   mockPort: number,
 *   baseEnv?: NodeJS.ProcessEnv | Record<string, string | undefined>,
 * }} options
 * @returns {Record<string, string>}
 */
export function buildLiveTierBackendEnv({ profile, mockPort, baseEnv = process.env }) {
  const backendEnv = buildBackendEnv({ profile, mockPort, baseEnv })
  return {
    ...backendEnv,
    CUSTOM_OPENAI_API_IP: `http://127.0.0.1:${mockPort}/v1`,
    CUSTOM_OPENAI_API_KEY: backendEnv.OPENAI_API_KEY,
    CUSTOM_OPENAI_API_MODEL: deterministicModel,
    EVALUATIONS_TEST_DB_PATH: profile.databasePaths.evaluations,
    JOBS_DB_PATH: profile.databasePaths.jobs,
    USER_DB_BASE_DIR: profile.userDatabasesDir,
    ACP_SESSIONS_DB_PATH: profile.databasePaths.acpSessions,
    ACP_WORKSPACE_ALLOWED_BASE_PATHS: profile.acpWorkspaceRootBase,
    ACP_ALLOWED_SESSION_CWD_ROOTS: profile.acpWorkspaceRootBase,
    ACP_RUNNER_CWD: path.join(profile.repoRoot, "tools/tldw-agent"),
    MCP_MODULES_CONFIG: profile.mcpModulesConfigPath,
    WATCHLIST_TEMPLATE_DIR: profile.watchlistTemplateDir,
    ACP_AUDIT_DB_PATH: profile.databasePaths.acpAudit,
    MONITORING_ALERTS_DB: profile.databasePaths.monitoringAlerts,
    SYSTEM_LOG_FILE_PATH: profile.systemLogFilePath,
  }
}

export const LIVE_TIER_MODEL = deterministicModel
