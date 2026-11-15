import React from "react"
import type { TFunction } from "i18next"
import { Collapse } from "antd"
import { Button } from "@/components/Common/Button"
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
  editor?: React.ReactNode
  onEdit?: () => void
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
  enableMermaidDiagrams,
  editor,
  onEdit
}: ReasoningBlockProps) {
  const isReasoningStreaming = isStreaming && Boolean(reasoningRunning)
  const editButton = React.useRef<HTMLButtonElement>(null)
  const isEditing = Boolean(editor)
  const wasEditing = React.useRef(false)

  React.useEffect(() => {
    if (!isEditing && wasEditing.current) editButton.current?.focus()
    wasEditing.current = isEditing
  }, [isEditing])

  return (
    <Collapse
      ghost
      className="!mb-3 rounded-md !border !border-solid !border-border !bg-surface"
      classNames={{ header: "!text-text", body: "border-t border-border !text-text" }}
      defaultActiveKey={openReasoning ? "reasoning" : undefined}
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
          children: editor ?? (
            <>
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
              {onEdit && (
                <Button
                  ref={editButton}
                  variant="text"
                  size="sm"
                  className="mt-2"
                  onClick={onEdit}>
                  {t("reasoning.edit", "Edit thinking")}
                </Button>
              )}
            </>
          )
        }
      ]}
    />
  )
}
