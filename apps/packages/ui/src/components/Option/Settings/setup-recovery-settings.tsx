import { Modal } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import { Link, useNavigate } from "react-router-dom"

import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"

type RowStatus = "ok" | "needs-action" | "unknown"

type RecoveryRow = {
  actionLabel: string
  description: string
  id: string
  label: string
  status: RowStatus
  to: string
}

const statusClassName: Record<RowStatus, string> = {
  ok: "border-success/40 bg-success/5",
  "needs-action": "border-warn/50 bg-warn/5",
  unknown: "border-border bg-surface"
}

export const SetupRecoverySettings = () => {
  const { t } = useTranslation("settings")
  const navigate = useNavigate()
  const connection = useConnectionState()
  const { errorKind, isChecking, uxState } = useConnectionUxState()
  const { restartOnboarding } = useConnectionActions()
  const { selectedModel, selectedModelIsLoading } = useSelectedModel()
  const [storedEmbeddingModel] = useStorage<string | null>(
    "defaultEmbeddingModel",
    null
  )

  const rows: RecoveryRow[] = [
    {
      actionLabel: t("setupRecovery.server.action", "Edit server"),
      description:
        connection.serverUrl ||
        t("setupRecovery.server.missing", "No server URL configured."),
      id: "server",
      label: t("setupRecovery.server.label", "Server connection"),
      status: connection.isConnected ? "ok" : "needs-action",
      to: "/settings/tldw"
    },
    {
      actionLabel: t("setupRecovery.auth.action", "Fix auth"),
      description:
        uxState === "error_auth" || errorKind === "auth"
          ? t("setupRecovery.auth.failed", "Authentication needs attention.")
          : t(
              "setupRecovery.auth.ready",
              "No auth issue detected from the current connection state."
            ),
      id: "auth",
      label: t("setupRecovery.auth.label", "Authentication"),
      status:
        uxState === "error_auth" || errorKind === "auth"
          ? "needs-action"
          : isChecking
            ? "unknown"
            : "ok",
      to: "/settings/tldw"
    },
    {
      actionLabel: t("setupRecovery.providers.action", "Provider keys"),
      description: t(
        "setupRecovery.providers.unknown",
        "Provider key status is checked on the provider keys page."
      ),
      id: "providers",
      label: t("setupRecovery.providers.label", "Provider keys"),
      status: "unknown",
      to: "/settings/provider-keys"
    },
    {
      actionLabel: t("setupRecovery.chatModel.action", "Model settings"),
      description:
        selectedModel ||
        t("setupRecovery.chatModel.missing", "No chat model selected."),
      id: "chat-model",
      label: t("setupRecovery.chatModel.label", "Default chat model"),
      status: selectedModel
        ? "ok"
        : selectedModelIsLoading
          ? "unknown"
          : "needs-action",
      to: "/settings/model"
    },
    {
      actionLabel: t(
        "setupRecovery.embeddingModel.action",
        "Embedding defaults"
      ),
      description:
        storedEmbeddingModel ||
        t(
          "setupRecovery.embeddingModel.unknown",
          "Embedding default not checked yet."
        ),
      id: "embedding-model",
      label: t("setupRecovery.embeddingModel.label", "Embedding model"),
      status: storedEmbeddingModel ? "ok" : "unknown",
      to: "/settings/rag"
    },
    {
      actionLabel: t("setupRecovery.health.action", "Full diagnostics"),
      description: t(
        "setupRecovery.health.description",
        "Open diagnostics for detailed server and subsystem checks."
      ),
      id: "health",
      label: t("setupRecovery.health.label", "Health checks"),
      status:
        connection.knowledgeStatus === "offline" ? "needs-action" : "unknown",
      to: "/settings/health"
    }
  ]

  const confirmRestartOnboarding = () => {
    Modal.confirm({
      cancelText: t("setupRecovery.restartOnboarding.cancel", "Cancel"),
      content: t(
        "setupRecovery.restartOnboarding.confirmDescription",
        "This reopens the initial setup flow so you can review server and authentication settings."
      ),
      okText: t("setupRecovery.restartOnboarding.confirm", "Restart"),
      onOk: async () => {
        await restartOnboarding()
        navigate("/")
      },
      title: t(
        "setupRecovery.restartOnboarding.confirmTitle",
        "Restart onboarding?"
      )
    })
  }

  return (
    <div className="space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("setupRecovery.title", "Setup & Recovery")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "setupRecovery.subtitle",
            "Check the basics first, then jump to the page that owns the fix."
          )}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <div className="space-y-3">
        {rows.map((row) => (
          <div
            className={`rounded-md border p-4 ${statusClassName[row.status]}`}
            key={row.id}
          >
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="font-medium text-text">{row.label}</div>
                <p className="mt-1 text-xs text-text-muted">
                  {row.description}
                </p>
              </div>
              <Link
                className="inline-flex rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
                to={row.to}
              >
                {row.actionLabel}
              </Link>
            </div>
          </div>
        ))}
      </div>

      <button
        className="rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
        onClick={confirmRestartOnboarding}
        type="button"
      >
        {t("setupRecovery.restartOnboarding.button", "Restart onboarding")}
      </button>
    </div>
  )
}

export default SetupRecoverySettings
