import type { FC } from "react"
import { Card, Empty, Tag } from "antd"
import { Alert } from "@/components/ui/primitives"
import { getDesignSystemState } from "@/design-system"
import { WritingPlaygroundResponseInspectorCard } from "./WritingPlaygroundResponseInspectorCard"
import { WritingPlaygroundTokenInspectorCard } from "./WritingPlaygroundTokenInspectorCard"
import { WritingPlaygroundWordcloudCard } from "./WritingPlaygroundWordcloudCard"
import type { WritingPlaygroundDiagnosticsPanelProps } from "./WritingPlaygroundDiagnostics.types"

const READY_STATE_LABEL = getDesignSystemState("ready").label

export const WritingPlaygroundDiagnosticsPanel: FC<
  WritingPlaygroundDiagnosticsPanelProps
> = ({
  title,
  t,
  status,
  showOffline,
  showUnsupported,
  hasActiveSession,
  response,
  token,
  wordcloud
}) => {
  const { enabled: responseEnabled, ...responseCardProps } = response
  const { enabled: tokenEnabled, ...tokenCardProps } = token
  const { enabled: wordcloudEnabled, ...wordcloudCardProps } = wordcloud

  return (
    <Card
      data-testid="writing-playground-diagnostics-card"
      title={title ?? t("option:writingPlayground.sidebarDiagnostics", "Diagnostics")}>
      <div className="flex flex-col gap-2">
        <Tag color={status === "warning" ? "gold" : status === "busy" ? "blue" : "green"}>
          {status === "warning"
            ? t("option:writingPlayground.diagnosticsWarning", "Warning")
            : status === "busy"
              ? t("option:writingPlayground.diagnosticsBusy", "Busy")
              : t("option:writingPlayground.diagnosticsReady", READY_STATE_LABEL)}
        </Tag>
        {showOffline ? (
          <Alert
            variant="warning"
            title={t("option:writingPlayground.offlineTitle", "Server required")}
          >
            {t(
              "option:writingPlayground.offlineBody",
              "Connect to your tldw server to load writing sessions and generate."
            )}
          </Alert>
        ) : null}
        {showUnsupported ? (
          <Alert
            variant="info"
            title={t(
              "option:writingPlayground.unavailableTitle",
              "Playground unavailable"
            )}
          >
            {t(
              "option:writingPlayground.unavailableBody",
              "This server does not advertise writing playground support yet."
            )}
          </Alert>
        ) : null}
        {!showOffline && !showUnsupported ? (
          hasActiveSession ? (
            <div className="flex flex-col gap-3">
              {responseEnabled ? (
                <WritingPlaygroundResponseInspectorCard
                  t={t}
                  {...responseCardProps}
                />
              ) : null}

              {tokenEnabled ? (
                <WritingPlaygroundTokenInspectorCard
                  t={t}
                  {...tokenCardProps}
                />
              ) : null}

              {wordcloudEnabled ? (
                <WritingPlaygroundWordcloudCard
                  t={t}
                  {...wordcloudCardProps}
                />
              ) : null}
            </div>
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={t(
                "option:writingPlayground.selectSession",
                "Select a session to begin."
              )}
            />
          )
        ) : null}
      </div>
    </Card>
  )
}
