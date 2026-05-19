import React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { MarkdownErrorBoundary } from "@/components/Common/MarkdownErrorBoundary"
import { cn } from "@/libs/utils"

const FLASHCARD_MARKDOWN_SNIPPET_CLASS =
  "[&_p]:m-0 [&_ul]:m-0 [&_ol]:m-0 [&_pre]:m-0 [&_pre]:bg-transparent [&_pre]:p-0 [&_h1]:m-0 [&_h2]:m-0 [&_h3]:m-0 [&_h4]:m-0 [&_h5]:m-0 [&_h6]:m-0 [&_img]:hidden"

interface FlashcardMarkdownSnippetProps {
  content: string
  className?: string
}

const stopSnippetLinkPropagation = (
  event: React.MouseEvent<HTMLDivElement>
) => {
  const target = event.target
  if (!(target instanceof Element)) return
  if (target.closest("a")) {
    event.stopPropagation()
  }
}

const FlashcardMarkdownSnippetInner: React.FC<FlashcardMarkdownSnippetProps> = ({
  content,
  className = ""
}) => (
  <MarkdownErrorBoundary fallbackText={content}>
    <div
      className={cn(
        "prose prose-xs max-w-none break-words text-inherit dark:prose-invert",
        FLASHCARD_MARKDOWN_SNIPPET_CLASS,
        className
      )}
      onClickCapture={stopSnippetLinkPropagation}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ className: anchorClassName, ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noopener noreferrer"
              className={cn("text-primary hover:underline", anchorClassName)}
            />
          ),
          img: () => null
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  </MarkdownErrorBoundary>
)

export const FlashcardMarkdownSnippet = React.memo(FlashcardMarkdownSnippetInner)

FlashcardMarkdownSnippet.displayName = "FlashcardMarkdownSnippet"

export default FlashcardMarkdownSnippet
