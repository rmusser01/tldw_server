import type { FC } from "react"
import { Button, Tag } from "antd"
import { Alert } from "@/components/ui/primitives"
import type { TokenInspectorCardProps } from "./WritingPlaygroundDiagnostics.types"

export const WritingPlaygroundTokenInspectorCard: FC<TokenInspectorCardProps> = ({
  t,
  tokenizerName,
  serverSupportsTokenCount,
  canCountTokens,
  isCountingTokens,
  onCountTokens,
  serverSupportsTokenize,
  canTokenizePreview,
  isTokenizingText,
  onTokenizePreview,
  hasTokenCountResult,
  tokenCountValue,
  hasTokenizeResult,
  tokenInspectorError,
  tokenInspectorBusy,
  tokenInspectorUnavailableReason,
  onClearTokenInspector,
  tokenPreviewRowsCount,
  tokenPreviewTotal
}) => (
  <div className="rounded-md border border-border bg-surface p-3">
    <div className="mb-2 flex items-center justify-between gap-2">
      <span className="text-xs font-medium text-text">
        {t("option:writingPlayground.tokenInspectorTitle", "Token inspector")}
      </span>
      {tokenizerName ? (
        <Tag color="blue">
          {t("option:writingPlayground.tokenInspectorTokenizer", "Tokenizer: {{tokenizer}}", {
            tokenizer: tokenizerName
          })}
        </Tag>
      ) : null}
    </div>
    <div className="mb-2 flex flex-wrap items-center gap-2">
      {serverSupportsTokenCount ? (
        <Button
          size="small"
          disabled={!canCountTokens}
          loading={isCountingTokens}
          onClick={() => {
            void onCountTokens()
          }}>
          {t("option:writingPlayground.countTokensAction", "Count tokens")}
        </Button>
      ) : null}
      {serverSupportsTokenize ? (
        <Button
          size="small"
          disabled={!canTokenizePreview}
          loading={isTokenizingText}
          onClick={() => {
            void onTokenizePreview()
          }}>
          {t("option:writingPlayground.tokenizePreviewAction", "Tokenize preview")}
        </Button>
      ) : null}
      {hasTokenCountResult || hasTokenizeResult || Boolean(tokenInspectorError) ? (
        <Button
          size="small"
          disabled={tokenInspectorBusy}
          onClick={onClearTokenInspector}>
          {t("common:clear", "Clear")}
        </Button>
      ) : null}
    </div>
    {tokenInspectorUnavailableReason ? (
<Alert variant="info" className="mb-2" title={tokenInspectorUnavailableReason} />
    ) : null}
    {tokenInspectorError ? (
<Alert variant="error" className="mb-2" title={tokenInspectorError} />
    ) : null}
    <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
      {hasTokenCountResult && tokenCountValue != null ? (
        <Tag color="blue">
          {t("option:writingPlayground.tokenInspectorCountLabel", "{{count}} tokens", {
            count: tokenCountValue
          })}
        </Tag>
      ) : null}
      {hasTokenizeResult ? (
        <Tag color="default">
          {t(
            "option:writingPlayground.tokenInspectorTruncated",
            "Showing first {{count}} of {{total}} tokens.",
            { count: tokenPreviewRowsCount, total: tokenPreviewTotal }
          )}
        </Tag>
      ) : null}
    </div>
  </div>
)
