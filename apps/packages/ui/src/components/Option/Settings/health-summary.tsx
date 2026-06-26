import { useEffect, useState } from "react"
import { Drawer, Button, Tooltip } from "antd"

import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"
import { useTranslation } from "react-i18next"
import { getCoreIssueLabel } from "./tldw-connection-status"

export default function HealthSummary() {
  const { t } = useTranslation(["settings"])
  const {
    phase,
    isConnected,
    lastCheckedAt,
    knowledgeStatus,
    knowledgeLastCheckedAt
  } = useConnectionState()
  const { uxState, errorKind } = useConnectionUxState()
  const { checkOnce } = useConnectionActions()

  const [core, setCore] = useState<"unknown" | "ok" | "fail">("unknown")
  const [rag, setRag] = useState<"unknown" | "ok" | "fail">("unknown")

  const [coreCheckedAt, setCoreCheckedAt] = useState<number | null>(null)
  const [ragCheckedAt, setRagCheckedAt] = useState<number | null>(null)
  const [open, setOpen] = useState(false)
  const diagnosticsPanelId = "health-diagnostics-panel"

  // Keep core/server status in sync with the shared connection state.
  useEffect(() => {
    void checkOnce()
  }, [checkOnce])

  useEffect(() => {
    if (phase === ConnectionPhase.SEARCHING) {
      setCore("unknown")
    } else if (isConnected && phase === ConnectionPhase.CONNECTED) {
      setCore("ok")
    } else if (
      phase === ConnectionPhase.ERROR ||
      phase === ConnectionPhase.UNCONFIGURED
    ) {
      setCore("fail")
    }
  }, [phase, isConnected])

  useEffect(() => {
    if (lastCheckedAt != null) {
      setCoreCheckedAt(lastCheckedAt)
    }
  }, [lastCheckedAt])

  // Map shared knowledgeStatus into a simple dot state.
  useEffect(() => {
    if (
      knowledgeStatus === "ready" ||
      knowledgeStatus === "indexing" ||
      knowledgeStatus === "empty"
    ) {
      setRag("ok")
    } else if (knowledgeStatus === "offline") {
      setRag("fail")
    } else {
      setRag("unknown")
    }
  }, [knowledgeStatus])

  useEffect(() => {
    if (knowledgeLastCheckedAt != null) {
      setRagCheckedAt(knowledgeLastCheckedAt)
    }
  }, [knowledgeLastCheckedAt])

  const Dot = ({ status }: { status: "unknown" | "ok" | "fail" }) => (
    <span
      aria-hidden
      className={`inline-block w-2 h-2 rounded-full ${
        status === "ok"
          ? "bg-success"
          : status === "fail"
            ? "bg-danger"
            : "bg-border-strong"
      }`}
    />
  )

  let issueLabel: string | null = null
  let issueBody: string | null = null

  if (uxState === "configuring_url" || uxState === "unconfigured") {
    issueLabel = getCoreIssueLabel(t, "missing_server_url")
    issueBody = t(
      "healthSummary.issueMissingServerHint",
      "Add your tldw server URL in Settings → tldw server, then run diagnostics again."
    )
  } else if (uxState === "configuring_auth") {
    issueLabel = getCoreIssueLabel(t, "missing_api_key")
    issueBody = t(
      "healthSummary.issueMissingApiKeyHint",
      "Add your single-user API key in Settings → tldw server, then run diagnostics again."
    )
  } else if (uxState === "error_auth" || errorKind === "auth") {
    issueLabel = getCoreIssueLabel(t, "invalid_api_key")
    issueBody = t(
      "healthSummary.issueAuthHint",
      "Your server responded but the API key or login is invalid. Fix your credentials in Settings → tldw server, then retry."
    )
  } else if (uxState === "error_unreachable" || errorKind === "unreachable") {
    issueLabel = getCoreIssueLabel(t, "unreachable")
    issueBody = t(
      "healthSummary.issueConnectivityHint",
      "We couldn’t reach your tldw server. Check that it’s running, your browser has site access, and any proxies or firewalls allow the connection."
    )
  } else if (rag === "fail") {
    issueLabel = getCoreIssueLabel(t, "degraded")
    issueBody = t(
      "healthSummary.issueRagHint",
      "Chat is available, but the knowledge index looks offline. Re-run indexing or inspect RAG components in the detailed diagnostics."
    )
  }

  return (
    <div className="mb-3 flex items-center justify-between rounded border border-transparent bg-transparent p-2 transition-colors duration-150 hover:border-border hover:bg-surface2">
      <div className="flex items-center gap-4 text-sm text-text-muted">
        <span
          className="flex items-center gap-2"
          title={t(
            "healthSummary.coreAria",
            "Server: server/API health"
          )}
          aria-label={t(
            "healthSummary.coreAria",
            "Server: server/API health"
          )}>
          <Dot status={core} />{" "}
          {t("healthSummary.core", "Server")}
        </span>
        <span
          className="flex items-center gap-2"
          title={t(
            "healthSummary.ragAria",
            "Knowledge: knowledge index health"
          )}
          aria-label={t(
            "healthSummary.ragAria",
            "Knowledge: knowledge index health"
          )}>
          <Dot status={rag} />{" "}
          {t("healthSummary.rag", "Knowledge")}
        </span>
      </div>
      <Tooltip
        title={
          t(
            "healthSummary.diagnosticsTooltip",
            "Open detailed diagnostics to troubleshoot or inspect health checks."
          ) as string
        }>
        <Button
          size="small"
          type="link"
          className="text-primary"
          onClick={() => setOpen(true)}
          aria-expanded={open}
          aria-controls={diagnosticsPanelId}
        >
          {t('healthSummary.diagnostics', 'Health & diagnostics')}
        </Button>
      </Tooltip>
      <Drawer
        title={t("healthSummary.diagnostics", "Health & diagnostics")}
        placement="right"
        size={360}
        onClose={() => setOpen(false)}
        open={open}>
        <div id={diagnosticsPanelId} className="space-y-3 text-sm">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Dot status={core} /> {t("healthSummary.core", "Server")}
            </span>
            <span className="text-text-subtle">
              {coreCheckedAt ? new Date(coreCheckedAt).toLocaleString() : ""}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <Dot status={rag} /> {t("healthSummary.rag", "Knowledge")}
            </span>
            <span className="text-text-subtle">
              {ragCheckedAt ? new Date(ragCheckedAt).toLocaleString() : ""}
            </span>
          </div>
          {issueLabel && issueBody && (
            <div className="pt-2 text-xs text-text-muted">
              <div className="font-medium">
                {t(
                  "healthSummary.currentIssueLabel",
                  "Current focus"
                )}
                {": "}
                {issueLabel}
              </div>
              <div className="mt-1">
                {issueBody}
              </div>
            </div>
          )}
          <div className="pt-3 text-xs text-text-subtle">
            {t(
              "healthSummary.footerInfo",
              "These checks summarize the last successful ping to your tldw server and knowledge index."
            )}
          </div>
        </div>
      </Drawer>
    </div>
  )
}
