import type { FC } from "react"
import { Button, Tag } from "antd"
import { Alert } from "@/components/ui/primitives"
import type { ResponseInspectorCardProps } from "./WritingPlaygroundDiagnostics.types"

export const WritingPlaygroundResponseInspectorCard: FC<
  ResponseInspectorCardProps
> = ({
  t,
  responseInspectorRowsCount,
  responseLogprobsCount,
  settingsLogprobsEnabled,
  settingsDisabled,
  responseLogprobRowsCount,
  responseLogprobTruncated,
  onCopyResponseInspectorJson,
  onExportResponseInspectorCsv,
  onClearResponseInspector
}) => (
  <div className="rounded-md border border-border bg-surface p-3">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
      <span className="text-xs font-medium text-text">
        {t("option:writingPlayground.responseInspectorTitle", "Response inspector")}
      </span>
      {responseInspectorRowsCount > 0 ? (
        <Tag color="blue">
          {t("option:writingPlayground.responseInspectorCount", "{{count}} rows", {
            count: responseInspectorRowsCount
          })}
        </Tag>
      ) : null}
    </div>
    <div className="mb-2 flex flex-wrap items-center gap-2">
      {responseLogprobsCount > 0 ? (
        <Button
          size="small"
          disabled={settingsDisabled}
          onClick={() => {
            void onCopyResponseInspectorJson()
          }}>
          {t("option:writingPlayground.responseInspectorCopyAction", "Copy JSON")}
        </Button>
      ) : null}
      {responseInspectorRowsCount > 0 ? (
        <Button
          size="small"
          disabled={settingsDisabled}
          onClick={onExportResponseInspectorCsv}>
          {t("option:writingPlayground.responseInspectorExportAction", "Export CSV")}
        </Button>
      ) : null}
      {responseLogprobsCount > 0 ? (
        <Button
          size="small"
          disabled={settingsDisabled}
          onClick={onClearResponseInspector}>
          {t("common:clear", "Clear")}
        </Button>
      ) : null}
    </div>
    {!settingsLogprobsEnabled ? (
      <Alert
        variant="info"
        title={t(
          "option:writingPlayground.responseInspectorDisabled",
          "Enable logprobs in generation settings to capture response token scores."
        )}
      />
    ) : null}
    {settingsLogprobsEnabled && responseLogprobsCount === 0 ? (
      <span className="text-xs text-text-muted">
        {t(
          "option:writingPlayground.responseInspectorEmpty",
          "No logprob data captured yet."
        )}
      </span>
    ) : null}
    {responseLogprobTruncated ? (
      <span className="text-xs text-text-muted">
        {t(
          "option:writingPlayground.responseInspectorTruncated",
          "Showing first {{count}} rows.",
          { count: responseLogprobRowsCount }
        )}
      </span>
    ) : null}
  </div>
)
