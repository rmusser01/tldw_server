import React from "react"
import { Markdown } from "@/components/Common/Markdown"
import { parseSections } from "@/components/Review/SectionNavigator"
import {
  hasLeadingTranscriptTimings,
  stripLeadingTranscriptTimings
} from "@/utils/media-transcript-display"

export type ContentType = "markdown" | "transcript" | "code" | "plain"

/**
 * Detect content type based on heuristics.
 */
export function detectContentType(content: string): ContentType {
  if (!content || content.trim().length === 0) return "plain"

  // Check for transcript-style content (timestamps at start of lines)
  if (hasLeadingTranscriptTimings(content)) return "transcript"

  // Check for markdown indicators
  const mdIndicators = [
    /^#{1,6}\s/m, // headings
    /\[.*?\]\(.*?\)/, // links
    /\*\*.*?\*\*/, // bold
    /```[\s\S]*?```/, // code fences
    /^\s*[-*+]\s/m, // unordered lists
    /^\s*\d+\.\s/m, // ordered lists
    /^\s*>\s/m, // blockquotes
    /\|.*\|.*\|/m // tables
  ]

  const matchCount = mdIndicators.filter((re) => re.test(content)).length
  if (matchCount >= 2) return "markdown"

  // Check for code-like content (high density of special chars)
  const codeIndicators = [
    /^(import|export|const|let|var|function|class|def|pub fn|package)\s/m,
    /[{};]\s*$/m,
    /^\s*(if|else|for|while|return|try|catch)\s*[({]/m
  ]
  if (codeIndicators.filter((re) => re.test(content)).length >= 2) return "code"

  return "plain"
}

interface ContentRendererProps {
  content: string
  contentType?: ContentType
  hideTranscriptTimings?: boolean
  searchQuery?: string
  className?: string
  headingAnchorIds?: string[]
}

/**
 * Renders content with appropriate formatting based on detected or specified type.
 * - markdown: Full markdown rendering with syntax highlighting
 * - transcript: Optionally strips timestamps, renders as markdown
 * - code: Wraps in code fence for syntax highlighting
 * - plain: Renders as pre-wrapped text
 */
export const ContentRenderer: React.FC<ContentRendererProps> = ({
  content,
  contentType,
  hideTranscriptTimings = false,
  searchQuery,
  className,
  headingAnchorIds
}) => {
  const detected = contentType ?? detectContentType(content)

  const processedContent = React.useMemo(() => {
    if (detected === "transcript" && hideTranscriptTimings) {
      return stripLeadingTranscriptTimings(content)
    }
    return content
  }, [content, detected, hideTranscriptTimings])

  const resolvedHeadingAnchorIds = React.useMemo(() => {
    if (headingAnchorIds && headingAnchorIds.length > 0) {
      return headingAnchorIds
    }
    return parseSections(processedContent)
      .filter((section) => section.id.startsWith("section-"))
      .map((section) => section.id)
  }, [headingAnchorIds, processedContent])

  if (detected === "plain") {
    return (
      <div
        className={`whitespace-pre-wrap break-words text-sm text-text leading-relaxed ${className ?? ""}`}
        data-testid="content-renderer-plain"
      >
        {processedContent}
      </div>
    )
  }

  if (detected === "code") {
    const wrapped = `\`\`\`\n${processedContent}\n\`\`\``
    return (
      <div data-testid="content-renderer-code" className={className}>
        <Markdown
          message={wrapped}
          searchQuery={searchQuery}
          codeBlockVariant="compact"
          className="prose prose-sm dark:prose-invert max-w-none"
          headingAnchorIds={resolvedHeadingAnchorIds}
        />
      </div>
    )
  }

  // markdown and transcript both render as markdown
  return (
    <div
      data-testid={`content-renderer-${detected}`}
      className={className}
    >
      <Markdown
        message={processedContent}
        searchQuery={searchQuery}
        codeBlockVariant="compact"
        className="prose prose-sm dark:prose-invert max-w-none"
        headingAnchorIds={resolvedHeadingAnchorIds}
      />
    </div>
  )
}
