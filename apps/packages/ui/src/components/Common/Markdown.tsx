import "katex/dist/katex.min.css"

import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import ReactMarkdown from "react-markdown"
import rehypeKatex from "rehype-katex"
import { Highlight } from "prism-react-renderer"

import "property-information"
import React from "react"
import { CodeBlock } from "./CodeBlock"
import { TableBlock } from "./TableBlock"
import { ManagedMarkdownImage } from "./ManagedMarkdownImage"
import { MermaidDiagramBlock } from "./MermaidDiagramBlock"
import { preprocessLaTeX } from "@/utils/latex"
import { useStorage } from "@plasmohq/storage/hook"
import { highlightText } from "@/utils/text-highlight"
import {
  DEFAULT_CHAT_SETTINGS,
  type ChatRichTextMode
} from "@/types/chat-settings"
import { normalizeLanguage, resolveTheme, safeLanguage } from "@/utils/code-theme"
import {
  normalizeChatRichTextMode,
  renderStCompatMarkdownToHtml
} from "@/utils/chat-rich-text"
import {
  normalizeChatRichTextColor,
  normalizeChatRichTextFont,
  resolveChatRichTextStyleCssVars
} from "@/utils/chat-rich-text-style"

const RICH_TEXT_ELEMENT_STYLE_CLASS =
  "[&_em]:[color:var(--rt-italic-color)] [&_em]:[font-family:var(--rt-italic-font)] [&_strong]:[color:var(--rt-bold-color)] [&_strong]:[font-family:var(--rt-bold-font)] [&_blockquote]:[color:var(--rt-quote-text-color)] [&_blockquote]:[font-family:var(--rt-quote-font)] [&_blockquote]:[border-left-color:var(--rt-quote-border-color)] [&_blockquote]:[background-color:var(--rt-quote-bg-color)] [&_blockquote]:border-l-4 [&_blockquote]:rounded-md [&_blockquote]:px-3 [&_blockquote]:py-2"

const MANAGED_ASSET_MARKER = "flashcard-asset://"
const SAFE_URL_PROTOCOL = /^(https?:|mailto:|tel:|blob:)/i
const DATA_IMAGE_URL_PROTOCOL = /^data:image\//i
const FENCE_START = /^(\s*)(`{3,}|~{3,})([^\n]*)$/
const FENCE_CLOSE = /^\s*(`{3,}|~{3,})\s*$/

type ClosedMermaidFenceSource = {
  blockIndex: number
  closingLine: number
  openingLine: number
  source: string
}

const isManagedAssetReference = (url: string): boolean =>
  String(url || "").startsWith(MANAGED_ASSET_MARKER)

const parseManagedAssetReference = (url: string): string | null => {
  if (!isManagedAssetReference(url)) return null
  const assetUuid = String(url || "").slice(MANAGED_ASSET_MARKER.length).trim()
  return assetUuid || null
}

const transformMarkdownUrl = (url: string): string => {
  const normalizedUrl = String(url || "").trim()
  if (!normalizedUrl) return ""
  if (isManagedAssetReference(normalizedUrl)) return normalizedUrl
  if (DATA_IMAGE_URL_PROTOCOL.test(normalizedUrl)) return normalizedUrl
  if (
    normalizedUrl.startsWith("#") ||
    normalizedUrl.startsWith("/") ||
    normalizedUrl.startsWith("./") ||
    normalizedUrl.startsWith("../")
  ) {
    return normalizedUrl
  }
  if (SAFE_URL_PROTOCOL.test(normalizedUrl)) return normalizedUrl
  if (!/^[a-z][a-z0-9+.-]*:/i.test(normalizedUrl)) return normalizedUrl
  return ""
}

const getFenceLanguage = (infoString: string): string =>
  infoString.trim().split(/\s+/)[0]?.toLowerCase() || ""

const collectClosedMermaidFenceSources = (
  markdown: string
): ClosedMermaidFenceSource[] => {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n")
  const sources: ClosedMermaidFenceSource[] = []
  let fencedBlockIndex = 0

  for (let index = 0; index < lines.length; index += 1) {
    const startMatch = FENCE_START.exec(lines[index])
    if (!startMatch) continue

    const openingFence = startMatch[2]
    const fenceMarker = openingFence[0]
    const fenceLength = openingFence.length
    const language = getFenceLanguage(startMatch[3] || "")
    const sourceStartIndex = index + 1

    for (let closeIndex = sourceStartIndex; closeIndex < lines.length; closeIndex += 1) {
      const closeMatch = FENCE_CLOSE.exec(lines[closeIndex])
      const closingFence = closeMatch?.[1]
      if (
        !closingFence ||
        closingFence[0] !== fenceMarker ||
        closingFence.length < fenceLength
      ) {
        continue
      }

      if (language === "mermaid") {
        sources.push({
          blockIndex: fencedBlockIndex,
          closingLine: closeIndex + 1,
          openingLine: index + 1,
          source: lines
            .slice(sourceStartIndex, closeIndex)
            .join("\n")
            .replace(/\n$/, "")
        })
      }
      fencedBlockIndex += 1
      index = closeIndex
      break
    }
  }

  return sources
}

export function Markdown({
  message,
  className = "prose break-words dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark",
  searchQuery,
  codeBlockVariant = "default",
  allowExternalImages,
  richTextModeOverride,
  headingAnchorIds,
  enableMermaidDiagrams = false,
  enableMermaidArtifactActions = false,
  artifactContextId,
}: {
  message: string
  className?: string
  searchQuery?: string
  codeBlockVariant?: "default" | "plain" | "compact" | "github"
  allowExternalImages?: boolean
  richTextModeOverride?: ChatRichTextMode
  headingAnchorIds?: string[]
  enableMermaidDiagrams?: boolean
  enableMermaidArtifactActions?: boolean
  artifactContextId?: string
}) {
  const [checkWideMode] = useStorage("checkWideMode", false)
  const [codeTheme] = useStorage("codeTheme", "auto")
  const [allowExternalImagesSetting] = useStorage(
    "allowExternalImages",
    DEFAULT_CHAT_SETTINGS.allowExternalImages
  )
  const [chatRichTextModeRaw] = useStorage(
    "chatRichTextMode",
    DEFAULT_CHAT_SETTINGS.chatRichTextMode
  )
  const [chatRichItalicColorRaw] = useStorage(
    "chatRichItalicColor",
    DEFAULT_CHAT_SETTINGS.chatRichItalicColor
  )
  const [chatRichItalicFontRaw] = useStorage(
    "chatRichItalicFont",
    DEFAULT_CHAT_SETTINGS.chatRichItalicFont
  )
  const [chatRichBoldColorRaw] = useStorage(
    "chatRichBoldColor",
    DEFAULT_CHAT_SETTINGS.chatRichBoldColor
  )
  const [chatRichBoldFontRaw] = useStorage(
    "chatRichBoldFont",
    DEFAULT_CHAT_SETTINGS.chatRichBoldFont
  )
  const [chatRichQuoteTextColorRaw] = useStorage(
    "chatRichQuoteTextColor",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteTextColor
  )
  const [chatRichQuoteFontRaw] = useStorage(
    "chatRichQuoteFont",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteFont
  )
  const [chatRichQuoteBorderColorRaw] = useStorage(
    "chatRichQuoteBorderColor",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteBorderColor
  )
  const [chatRichQuoteBackgroundColorRaw] = useStorage(
    "chatRichQuoteBackgroundColor",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteBackgroundColor
  )
  const blockIndexRef = React.useRef(0)
  const headingIndexRef = React.useRef(0)
  // Reset index each render pass to assign sequential indices to code blocks.
  blockIndexRef.current = 0
  headingIndexRef.current = 0
  const resolvedClassName = React.useMemo(() => {
    if (!checkWideMode) return className
    return `${className} max-w-none`
  }, [checkWideMode, className])
  const resolvedAllowExternalImages =
    typeof allowExternalImages === "boolean"
      ? allowExternalImages
      : allowExternalImagesSetting
  const richTextMode = React.useMemo(
    () =>
      normalizeChatRichTextMode(
        richTextModeOverride ?? chatRichTextModeRaw,
        DEFAULT_CHAT_SETTINGS.chatRichTextMode
      ),
    [chatRichTextModeRaw, richTextModeOverride]
  )
  const richTextStyleVars = React.useMemo(
    () =>
      resolveChatRichTextStyleCssVars({
        chatRichItalicColor: normalizeChatRichTextColor(
          chatRichItalicColorRaw,
          DEFAULT_CHAT_SETTINGS.chatRichItalicColor
        ),
        chatRichItalicFont: normalizeChatRichTextFont(
          chatRichItalicFontRaw,
          DEFAULT_CHAT_SETTINGS.chatRichItalicFont
        ),
        chatRichBoldColor: normalizeChatRichTextColor(
          chatRichBoldColorRaw,
          DEFAULT_CHAT_SETTINGS.chatRichBoldColor
        ),
        chatRichBoldFont: normalizeChatRichTextFont(
          chatRichBoldFontRaw,
          DEFAULT_CHAT_SETTINGS.chatRichBoldFont
        ),
        chatRichQuoteTextColor: normalizeChatRichTextColor(
          chatRichQuoteTextColorRaw,
          DEFAULT_CHAT_SETTINGS.chatRichQuoteTextColor
        ),
        chatRichQuoteFont: normalizeChatRichTextFont(
          chatRichQuoteFontRaw,
          DEFAULT_CHAT_SETTINGS.chatRichQuoteFont
        ),
        chatRichQuoteBorderColor: normalizeChatRichTextColor(
          chatRichQuoteBorderColorRaw,
          DEFAULT_CHAT_SETTINGS.chatRichQuoteBorderColor
        ),
        chatRichQuoteBackgroundColor: normalizeChatRichTextColor(
          chatRichQuoteBackgroundColorRaw,
          DEFAULT_CHAT_SETTINGS.chatRichQuoteBackgroundColor
        )
      }),
    [
      chatRichBoldColorRaw,
      chatRichBoldFontRaw,
      chatRichItalicColorRaw,
      chatRichItalicFontRaw,
      chatRichQuoteBackgroundColorRaw,
      chatRichQuoteBorderColorRaw,
      chatRichQuoteFontRaw,
      chatRichQuoteTextColorRaw
    ]
  )
  const paragraphClass =
    codeBlockVariant === "plain" ||
    codeBlockVariant === "compact" ||
    codeBlockVariant === "github"
      ? "mb-2 last:mb-0 whitespace-pre-wrap"
      : "mb-2 last:mb-0"
  const renderHighlightedChildren = React.useCallback(
    (children: React.ReactNode): React.ReactNode => {
      if (!searchQuery) return children
      return React.Children.map(children, (child) => {
        if (typeof child === "string") {
          return highlightText(child, searchQuery)
        }
        if (React.isValidElement(child) && child.props?.children) {
          const nextChildren = renderHighlightedChildren(child.props.children)
          if (nextChildren === child.props.children) return child
          return React.cloneElement(child, { ...child.props }, nextChildren)
        }
        return child
      })
    },
    [searchQuery],
  )
  const processedMessage = React.useMemo(() => preprocessLaTeX(message), [message])
  const nextHeadingAnchorProps = React.useCallback(() => {
    const anchorId = headingAnchorIds?.[headingIndexRef.current]
    headingIndexRef.current += 1
    return anchorId ? { "data-section-anchor": anchorId, id: anchorId } : {}
  }, [headingAnchorIds])
  const stCompatHtml = React.useMemo(() => {
    if (richTextMode !== "st_compat") return ""
    return renderStCompatMarkdownToHtml(
      processedMessage,
      resolvedAllowExternalImages
    )
  }, [processedMessage, resolvedAllowExternalImages, richTextMode])
  const hasManagedAssetImages = React.useMemo(
    () => processedMessage.includes(MANAGED_ASSET_MARKER),
    [processedMessage]
  )
  const closedMermaidFenceSources = React.useMemo(
    () =>
      enableMermaidDiagrams
        ? collectClosedMermaidFenceSources(processedMessage)
        : [],
    [enableMermaidDiagrams, processedMessage]
  )
  const closedMermaidFenceCursorRef = React.useRef(0)
  closedMermaidFenceCursorRef.current = 0
  const shouldUseComponentMermaid = closedMermaidFenceSources.length > 0

  if (
    richTextMode === "st_compat" &&
    !hasManagedAssetImages &&
    !shouldUseComponentMermaid
  ) {
    return (
      <div
        className={`${resolvedClassName} ${RICH_TEXT_ELEMENT_STYLE_CLASS} [&_.st-inline-spoiler]:rounded-sm [&_.st-inline-spoiler]:bg-surface2 [&_.st-inline-spoiler]:px-1 [&_.st-inline-spoiler]:py-0.5 [&_.st-inline-spoiler]:font-medium [&_.st-spoiler]:my-2 [&_.st-spoiler]:rounded-md [&_.st-spoiler]:border [&_.st-spoiler]:border-border [&_.st-spoiler]:bg-surface2/70 [&_.st-spoiler]:px-3 [&_.st-spoiler]:py-2 [&_.st-spoiler_>summary]:cursor-pointer [&_.st-spoiler_>summary]:font-medium [&_.st-external-image-blocked]:inline-flex [&_.st-external-image-blocked]:items-center [&_.st-external-image-blocked]:gap-2 [&_.st-external-image-blocked]:rounded-md [&_.st-external-image-blocked]:border [&_.st-external-image-blocked]:border-border [&_.st-external-image-blocked]:bg-surface2 [&_.st-external-image-blocked]:px-2 [&_.st-external-image-blocked]:py-1 [&_.st-external-image-blocked]:text-[11px] [&_.st-external-image-blocked]:text-text-muted`}
        style={richTextStyleVars as React.CSSProperties}
        role="region"
        aria-label="Message content"
        dangerouslySetInnerHTML={{ __html: stCompatHtml }}
      />
    )
  }

  return (
    <div
      className={`${resolvedClassName} ${RICH_TEXT_ELEMENT_STYLE_CLASS}`}
      style={richTextStyleVars as React.CSSProperties}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={transformMarkdownUrl}
        components={{
          pre({ children, ...props }) {
            const childArray = React.Children.toArray(children)
            const codeChild = childArray.find((child) => {
              if (!React.isValidElement(child)) return false
              const element = child as React.ReactElement
              const className = element.props?.className as string | undefined
              const nodeTag = element.props?.node?.tagName
              return Boolean(className || nodeTag === "code")
            }) as React.ReactElement | undefined

            if (!codeChild) {
              return <pre {...props}>{children}</pre>
            }

            const codeClassName = codeChild.props?.className as string | undefined
            const match = /language-([^\s]+)/.exec(codeClassName || "")
            const blockIndex = blockIndexRef.current++
            const value = String(codeChild.props?.children ?? "")
              .replace(/\r\n?/g, "\n")
              .replace(/\n$/, "")
            const rawLanguage = match ? match[1] : ""
            const normalizedLanguage = normalizeLanguage(rawLanguage)
            const nodePosition = codeChild.props?.node?.position
            const startLine = Number(nodePosition?.start?.line)
            const endLine = Number(nodePosition?.end?.line)
            const hasNodeLinePosition =
              Number.isFinite(startLine) && Number.isFinite(endLine)
            const closedMermaidFenceIndex = closedMermaidFenceSources.findIndex(
              (fence, fenceIndex) => {
                if (fenceIndex < closedMermaidFenceCursorRef.current) return false
                if (fence.source !== value) return false
                if (hasNodeLinePosition) {
                  return (
                    fence.openingLine === startLine &&
                    fence.closingLine === endLine
                  )
                }
                return fence.blockIndex === blockIndex
              }
            )

            if (
              enableMermaidDiagrams &&
              rawLanguage.trim().toLowerCase() === "mermaid" &&
              normalizedLanguage === "mermaid" &&
              closedMermaidFenceIndex !== -1
            ) {
              closedMermaidFenceCursorRef.current = closedMermaidFenceIndex + 1
              return (
                <MermaidDiagramBlock
                  artifactContextId={artifactContextId}
                  blockIndex={blockIndex}
                  enableArtifactAction={enableMermaidArtifactActions}
                  source={value}
                />
              )
            }

            if (codeBlockVariant === "plain") {
              return <div className="my-2 rounded-lg border border-border bg-surface2/70 px-3 py-2 text-xs font-mono leading-relaxed text-text whitespace-pre overflow-x-auto">{value}</div>
            }
            if (codeBlockVariant === "compact") {
              const highlightLanguage = rawLanguage ? normalizedLanguage : "plaintext"
              return (
                <div className="not-prose my-2 rounded-lg border border-border bg-surface2/70 px-3 py-2 overflow-x-auto">
                  <Highlight code={value} language={safeLanguage(highlightLanguage)} theme={resolveTheme(codeTheme || "dracula")}>
                    {({ className: highlightClassName, style, tokens, getLineProps, getTokenProps }) => (
                      <pre
                        className={`${highlightClassName} m-0 text-xs font-mono leading-relaxed`}
                        style={{
                          ...style,
                          backgroundColor: "transparent",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        {tokens.map((line, i) => (
                          <div key={i} {...getLineProps({ line, key: i })}>
                            {line.map((token, key) => (
                              <span key={key} {...getTokenProps({ token, key })} />
                            ))}
                          </div>
                        ))}
                      </pre>
                    )}
                  </Highlight>
                </div>
              )
            }
            if (codeBlockVariant === "github") {
              return (
                <div className="not-prose my-2 overflow-x-auto rounded-md border border-border/80 bg-surface2/70 px-4 py-3">
                  <Highlight
                    code={value}
                    language={safeLanguage(normalizedLanguage)}
                    theme={resolveTheme("auto")}
                  >
                    {({
                      className: highlightClassName,
                      style,
                      tokens,
                      getLineProps,
                      getTokenProps
                    }) => (
                      <pre
                        className={`${highlightClassName} m-0 text-xs font-mono leading-relaxed`}
                        style={{
                          ...style,
                          backgroundColor: "transparent",
                          fontFamily: "var(--font-mono)"
                        }}
                      >
                        {tokens.map((line, i) => (
                          <div key={i} {...getLineProps({ line, key: i })}>
                            {line.map((token, key) => (
                              <span key={key} {...getTokenProps({ token, key })} />
                            ))}
                          </div>
                        ))}
                      </pre>
                    )}
                  </Highlight>
                </div>
              )
            }
            return <CodeBlock language={match ? match[1] : ""} value={value} blockIndex={blockIndex} />
          },
          code({ className, children, ...props }) {
            const mergedClassName = className ? `${className} font-semibold` : "font-semibold"
            return (
              <code className={mergedClassName} {...props}>
                {children}
              </code>
            )
          },
          a({ ...props }) {
            return (
              <a target="_blank" rel="noopener noreferrer" className="text-primary text-sm hover:underline" {...props}>
                {props.children}
              </a>
            )
          },
          img({ src, alt }) {
            const resolvedSrc = typeof src === "string" ? src : ""
            const managedAssetUuid = parseManagedAssetReference(resolvedSrc)
            const isExternal = /^https?:\/\//i.test(resolvedSrc) || /^\/\/[^/]/.test(resolvedSrc)
            const isAllowed = !isExternal || resolvedAllowExternalImages
            if (!resolvedSrc) return null
            if (managedAssetUuid) {
              return <ManagedMarkdownImage assetUuid={managedAssetUuid} alt={alt || ""} />
            }
            if (!isAllowed) {
              return (
                <span className="inline-flex items-center gap-2 rounded-md border border-border bg-surface2 px-2 py-1 text-[11px] text-text-muted">
                  <span>{alt ? `Image: ${alt}` : "External image blocked"}</span>
                  <a href={resolvedSrc} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                    Open
                  </a>
                </span>
              )
            }
            return <img src={resolvedSrc} alt={alt || ""} loading="lazy" referrerPolicy="no-referrer" className="my-2 max-w-full rounded-md border border-border" />
          },
          table({ children }) {
            return <TableBlock>{children}</TableBlock>
          },
          p({ children, className, ...props }) {
            const content = renderHighlightedChildren(children)
            const mergedClassName = [paragraphClass, className].filter(Boolean).join(" ")
            return (
              <p className={mergedClassName} {...props}>
                {content}
              </p>
            )
          },
          li({ children, ...props }) {
            return <li {...props}>{renderHighlightedChildren(children)}</li>
          },
          td({ children, ...props }) {
            return <td {...props}>{renderHighlightedChildren(children)}</td>
          },
          th({ children, ...props }) {
            return <th {...props}>{renderHighlightedChildren(children)}</th>
          },
          h1({ children, ...props }) {
            return <h1 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h1>
          },
          h2({ children, ...props }) {
            return <h2 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h2>
          },
          h3({ children, ...props }) {
            return <h3 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h3>
          },
          h4({ children, ...props }) {
            return <h4 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h4>
          },
          h5({ children, ...props }) {
            return <h5 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h5>
          },
          h6({ children, ...props }) {
            return <h6 {...props} {...nextHeadingAnchorProps()}>{renderHighlightedChildren(children)}</h6>
          },
        }}
      >
        {processedMessage}
      </ReactMarkdown>
    </div>
  )
}

export default Markdown
