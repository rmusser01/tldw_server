import type { FC } from "react"
import { Button, Tag } from "antd"
import { Alert } from "@/components/ui/primitives"
import type { WordcloudCardProps } from "./WritingPlaygroundDiagnostics.types"

const MAX_WORDCLOUD_WORDS_TO_DISPLAY = 12

export const WritingPlaygroundWordcloudCard: FC<WordcloudCardProps> = ({
  t,
  wordcloudStatus,
  wordcloudStatusColor,
  canGenerateWordcloud,
  isGeneratingWordcloud,
  onGenerateWordcloud,
  wordcloudError,
  onClearWordcloud,
  wordcloudWords
}) => (
  <div className="rounded-md border border-border bg-surface p-3">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
      <span className="text-xs font-medium text-text">
        {t("option:writingPlayground.wordcloudTitle", "Wordcloud")}
      </span>
      {wordcloudStatus ? <Tag color={wordcloudStatusColor}>{wordcloudStatus}</Tag> : null}
    </div>
    <div className="mb-2 flex flex-wrap items-center gap-2">
      <Button
        size="small"
        disabled={!canGenerateWordcloud}
        loading={isGeneratingWordcloud}
        onClick={() => {
          void onGenerateWordcloud()
        }}>
        {t("option:writingPlayground.wordcloudGenerateAction", "Generate wordcloud")}
      </Button>
      {wordcloudStatus || wordcloudError ? (
        <Button
          size="small"
          disabled={isGeneratingWordcloud}
          onClick={onClearWordcloud}>
          {t("common:clear", "Clear")}
        </Button>
      ) : null}
    </div>
    {wordcloudError ? <Alert variant="error" title={wordcloudError} /> : null}
    {wordcloudWords.length > 0 ? (
      <div className="max-h-40 overflow-y-auto rounded-md border border-border bg-background px-2 py-1">
        <div className="flex flex-col gap-1 text-xs">
          {wordcloudWords.slice(0, MAX_WORDCLOUD_WORDS_TO_DISPLAY).map((word) => (
            <div
              key={`${word.text}-${word.weight}`}
              className="flex items-center justify-between gap-2">
              <span className="truncate text-text">{word.text}</span>
              <span className="text-text-muted">{word.weight}</span>
            </div>
          ))}
        </div>
      </div>
    ) : wordcloudStatus === "ready" ? (
      <span className="text-xs text-text-muted">
        {t(
          "option:writingPlayground.wordcloudEmpty",
          "No words matched the current filters."
        )}
      </span>
    ) : null}
  </div>
)
