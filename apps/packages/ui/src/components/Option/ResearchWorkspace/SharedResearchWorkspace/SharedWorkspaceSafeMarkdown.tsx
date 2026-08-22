import React from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

type SharedWorkspaceSafeMarkdownProps = {
  compact?: boolean
  content: string
}

const safeLinkHref = (href: string | undefined): string | undefined => {
  if (!href) return undefined
  try {
    const url = new URL(href)
    return url.protocol === "https:" ||
      url.protocol === "http:" ||
      url.protocol === "mailto:"
      ? href
      : undefined
  } catch {
    return undefined
  }
}

export const SharedWorkspaceSafeMarkdown: React.FC<
  SharedWorkspaceSafeMarkdownProps
> = ({ compact = false, content }) => (
  <div
    className={
      compact
        ? "min-w-0 break-words text-xs leading-5 text-text-muted"
        : "min-w-0 break-words text-sm leading-6 text-text"
    }
  >
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      skipHtml
      components={{
        a: ({ children, href }) => {
          const safeHref = safeLinkHref(href)
          return safeHref ? (
            <a
              href={safeHref}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-focus"
            >
              {children}
            </a>
          ) : (
            <span>{children}</span>
          )
        },
        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l border-border pl-3 text-text-muted">
            {children}
          </blockquote>
        ),
        code: ({ children }) => (
          <code className="break-words font-mono text-[0.9em]">{children}</code>
        ),
        h1: ({ children }) => (
          <h3 className="mb-2 mt-3 text-base font-semibold">{children}</h3>
        ),
        h2: ({ children }) => (
          <h3 className="mb-2 mt-3 text-base font-semibold">{children}</h3>
        ),
        h3: ({ children }) => (
          <h3 className="mb-1.5 mt-3 text-sm font-semibold">{children}</h3>
        ),
        h4: ({ children }) => (
          <h4 className="mb-1.5 mt-2 text-sm font-semibold">{children}</h4>
        ),
        h5: ({ children }) => (
          <h5 className="mb-1.5 mt-2 text-sm font-semibold">{children}</h5>
        ),
        h6: ({ children }) => (
          <h6 className="mb-1.5 mt-2 text-sm font-semibold">{children}</h6>
        ),
        img: () => null,
        li: ({ children }) => <li className="my-0.5">{children}</li>,
        ol: ({ children }) => (
          <ol className="my-2 list-decimal space-y-0.5 pl-5">{children}</ol>
        ),
        p: ({ children }) => (
          <p className="my-1.5 first:mt-0 last:mb-0">{children}</p>
        ),
        pre: ({ children }) => (
          <pre className="my-2 max-w-full overflow-x-auto rounded-md border border-border bg-surface2 p-3 font-mono text-xs leading-5">
            {children}
          </pre>
        ),
        table: ({ children }) => (
          <div className="my-2 max-w-full overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              {children}
            </table>
          </div>
        ),
        td: ({ children }) => (
          <td className="border border-border px-2 py-1 align-top">{children}</td>
        ),
        th: ({ children }) => (
          <th className="border border-border bg-surface2 px-2 py-1 font-semibold">
            {children}
          </th>
        ),
        ul: ({ children }) => (
          <ul className="my-2 list-disc space-y-0.5 pl-5">{children}</ul>
        )
      }}
    >
      {content}
    </ReactMarkdown>
  </div>
)
