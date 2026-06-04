import React from "react"
import type { TFunction } from "i18next"
import { Collapse } from "antd"
import { humanizeMilliseconds } from "@/utils/humanize-milliseconds"

const Markdown = React.lazy(() => import("../../Common/Markdown"))

type ReasoningBlockProps = {
  content: string
  isStreaming: boolean
  reasoningRunning?: boolean
  openReasoning?: boolean
  reasoningTimeTaken?: number
  assistantTextClass: string
  markdownBaseClasses: string
  searchQuery?: string
  t: TFunction
  enableMermaidDiagrams?: boolean
}

export function ReasoningBlock({
  content,
  isStreaming,
  reasoningRunning,
  openReasoning,
  reasoningTimeTaken,
  assistantTextClass,
  markdownBaseClasses,
  searchQuery,
  t,
  enableMermaidDiagrams
}: ReasoningBlockProps) {
  const isReasoningStreaming = isStreaming && Boolean(reasoningRunning)
  const shouldExpand = Boolean(openReasoning) || isReasoningStreaming

  return (
    <Collapse
      className="border-none text-text-muted !mb-3 "
      defaultActiveKey={shouldExpand ? "reasoning" : undefined}
      activeKey={isReasoningStreaming ? "reasoning" : undefined}
      items={[
        {
          key: "reasoning",
          label: isReasoningStreaming ? (
            <div className="flex items-center gap-2">
              <span className="italic shimmer-text">
                {t("reasoning.thinking", "Thinking…")}
              </span>
            </div>
          ) : (
            <span className="flex items-center gap-2">
              <span>
                {t("reasoning.thought", "Model's reasoning (optional)")}
              </span>
              {reasoningTimeTaken != null && (
                <span className="text-label text-text-subtle">
                  {humanizeMilliseconds(reasoningTimeTaken)}
                </span>
              )}
            </span>
          ),
          children: (
            <React.Suspense
              fallback={
                <p className={`text-body text-text-muted ${assistantTextClass}`}>
                  {t("reasoning.loading")}
                </p>
              }
            >
              <div>
                <Markdown
                  message={content}
                  className={`${markdownBaseClasses} ${assistantTextClass}`}
                  searchQuery={searchQuery}
                  codeBlockVariant="github"
                  enableMermaidDiagrams={
                    Boolean(enableMermaidDiagrams) && !isReasoningStreaming
                  }
                />
                {isReasoningStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-text-muted animate-pulse rounded-sm" />
                )}
              </div>
            </React.Suspense>
          )
        }
      ]}
    />
  )
}
