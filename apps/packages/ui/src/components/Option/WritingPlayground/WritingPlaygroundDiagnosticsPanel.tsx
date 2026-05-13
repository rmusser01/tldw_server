import type { FC } from "react"
import { Alert, Card, Empty, Tag } from "antd"
import { getDesignSystemState } from "@/design-system"
import { WritingPlaygroundResponseInspectorCard } from "./WritingPlaygroundResponseInspectorCard"
import { WritingPlaygroundTokenInspectorCard } from "./WritingPlaygroundTokenInspectorCard"
import { WritingPlaygroundWordcloudCard } from "./WritingPlaygroundWordcloudCard"
import type { WritingPlaygroundDiagnosticsPanelProps } from "./WritingPlaygroundDiagnostics.types"

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
  const readyState = status === "ready" ? getDesignSystemState("ready") : null

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
              : t("option:writingPlayground.diagnosticsReady", readyState?.label ?? "")}
        </Tag>
        {showOffline ? (
          <Alert
            type="warning"
            showIcon
            title={t("option:writingPlayground.offlineTitle", "Server required")}
            description={t(
              "option:writingPlayground.offlineBody",
              "Connect to your tldw server to load writing sessions and generate."
            )}
          />
        ) : null}
        {showUnsupported ? (
          <Alert
            type="info"
            showIcon
            title={t(
              "option:writingPlayground.unavailableTitle",
              "Playground unavailable"
            )}
            description={t(
              "option:writingPlayground.unavailableBody",
              "This server does not advertise writing playground support yet."
            )}
          />
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
