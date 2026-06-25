export type CoreStatus = "unknown" | "checking" | "connected" | "failed"
export type RagStatus = "healthy" | "unhealthy" | "unknown" | "checking"
export type CoreIssueKind =
  | "missing_server_url"
  | "missing_api_key"
  | "invalid_api_key"
  | "unreachable"
  | "degraded"

type TranslateFn = (key: string, defaultValue: string) => string

export const getCoreStatusLabel = (t: TranslateFn, status: CoreStatus) => {
  switch (status) {
    case "checking":
      return t("settings:tldw.connection.coreChecking", "Core: checking…")
    case "connected":
      return t("settings:tldw.connection.coreOk", "Core: reachable")
    case "failed":
      return t("settings:tldw.connection.coreFailed", "Core: unreachable")
    default:
      return t(
        "settings:tldw.connection.coreUnknown",
        "Core: not checked yet"
      )
  }
}

export const getCoreIssueLabel = (
  t: TranslateFn,
  issue: CoreIssueKind
) => {
  switch (issue) {
    case "missing_server_url":
      return t(
        "settings:tldw.connection.issueMissingServerUrl",
        "Server URL missing"
      )
    case "missing_api_key":
      return t(
        "settings:tldw.connection.issueMissingApiKey",
        "API key missing"
      )
    case "invalid_api_key":
      return t(
        "settings:tldw.connection.issueInvalidApiKey",
        "API key invalid"
      )
    case "unreachable":
      return t(
        "settings:tldw.connection.issueUnreachable",
        "Server unreachable"
      )
    case "degraded":
      return t(
        "settings:tldw.connection.issueDegraded",
        "Feature checks degraded"
      )
  }
}

export const getRagStatusLabel = (t: TranslateFn, status: RagStatus) => {
  switch (status) {
    case "checking":
      return t("settings:tldw.connection.ragChecking", "RAG: checking…")
    case "healthy":
      return t("settings:tldw.connection.ragHealthy", "RAG: healthy")
    case "unhealthy":
      return t(
        "settings:tldw.connection.ragUnhealthy",
        "RAG: needs attention"
      )
    default:
      return t(
        "settings:tldw.connection.ragUnknown",
        "RAG: not checked yet"
      )
  }
}
