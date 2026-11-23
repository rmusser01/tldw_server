import React from "react"
import { Button, Tooltip } from "antd"
import { Check, Clock, Copy, Plus, Zap } from "lucide-react"
import { useTranslation } from "react-i18next"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

type GeneratedAnswerCardProps = {
  answer: string
  totalTime?: number
  cacheHit?: boolean
  onCopy: () => void
  onInsert: () => void
  /** Focus lands here when a requested answer arrives (CC-10 review). */
  headingRef?: React.Ref<HTMLHeadingElement>
}

/**
 * Card displaying the generated RAG answer with copy/insert actions.
 */
export const GeneratedAnswerCard: React.FC<GeneratedAnswerCardProps> =
  React.memo(({ answer, totalTime, cacheHit, onCopy, onInsert, headingRef }) => {
    const { t } = useTranslation(["sidepanel"])
    const [copied, setCopied] = React.useState(false)

    const handleCopy = () => {
      onCopy()
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }

    return (
      <div className="rounded-lg border border-accent/30 bg-accent/5 p-4">
        <div className="flex items-center justify-between mb-2">
          <h4
            ref={headingRef}
            tabIndex={-1}
            className="text-sm font-semibold text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            {t("sidepanel:qaSearch.generatedAnswer", "Generated Answer")}
          </h4>
          <div className="flex items-center gap-2 text-xs text-text-muted">
            {cacheHit && (
              <span className="flex items-center gap-1">
                <Zap className="h-3 w-3 text-warn" />
                {t("sidepanel:qaSearch.cached", "Cached")}
              </span>
            )}
            {typeof totalTime === "number" && totalTime > 0 && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {totalTime.toFixed(1)}s
              </span>
            )}
          </div>
        </div>

        <div className="mb-3 rounded-md bg-surface/30 px-2 py-2">
          <div className="prose prose-sm max-w-none break-words text-text dark:prose-invert prose-p:my-2 prose-pre:p-0">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {answer}
            </ReactMarkdown>
          </div>
        </div>

        <div className="flex items-center gap-1 pt-2 border-t border-accent/20">
          <Tooltip
            title={t("sidepanel:qaSearch.copyAnswer", "Copy answer")}
          >
            <Button
              type="text"
              size="small"
              onClick={handleCopy}
              icon={
                copied ? (
                  <Check className="h-3.5 w-3.5 text-success" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )
              }
              className="text-text-muted hover:text-accent"
            >
              {copied
                ? t("sidepanel:qaSearch.copied", "Copied")
                : t("sidepanel:qaSearch.copy", "Copy")}
            </Button>
          </Tooltip>

          <Tooltip
            title={t(
              "sidepanel:qaSearch.insertAnswer",
              "Insert answer into chat"
            )}
          >
            <Button
              type="text"
              size="small"
              onClick={onInsert}
              icon={<Plus className="h-3.5 w-3.5" />}
              className="text-text-muted hover:text-accent"
            >
              {t("sidepanel:rag.actions.insert", "Insert")}
            </Button>
          </Tooltip>
        </div>
      </div>
    )
  })

GeneratedAnswerCard.displayName = "GeneratedAnswerCard"
