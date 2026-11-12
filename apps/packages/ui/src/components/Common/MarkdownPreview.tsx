import React from "react"
import { MarkdownErrorBoundary } from "@/components/Common/MarkdownErrorBoundary"

const Markdown = React.lazy(() => import("@/components/Common/Markdown"))

export interface MarkdownPreviewProps {
  content: string
  className?: string
  size?: "xs" | "sm" | "base"
}

export const MarkdownPreview: React.FC<MarkdownPreviewProps> = ({
  content,
  className = "",
  size = "sm"
}) => {
  const sizeClass = size === "xs" ? "prose-xs" : size === "sm" ? "prose-sm" : ""

  return (
    <MarkdownErrorBoundary fallbackText={content}>
      <React.Suspense
        fallback={<div className="whitespace-pre-wrap">{content}</div>}>
        <Markdown
          message={content}
          enableCodeArtifactActions={false}
          className={`prose ${sizeClass} break-words dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark [&_.task-list-item]:list-none [&_.task-list-item]:pl-0 ${className}`}
        />
      </React.Suspense>
    </MarkdownErrorBoundary>
  )
}

export default MarkdownPreview
