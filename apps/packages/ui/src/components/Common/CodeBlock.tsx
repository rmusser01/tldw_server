import { programmingLanguages } from "@/utils/language-extension"
import { Tooltip } from "antd"
import {
  CopyCheckIcon,
  CopyIcon,
  DownloadIcon,
  EyeIcon,
  CodeIcon,
  ChevronDown,
  ChevronUp
} from "lucide-react"
import { FC, useState, useRef, useEffect, useCallback } from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { Highlight } from "prism-react-renderer"
import DOMPurify from "dompurify"
import { useUiModeStore } from "@/store/ui-mode"
import { useArtifactsStore } from "@/store/artifacts"
import { normalizeLanguage, resolveTheme, safeLanguage } from "@/utils/code-theme"

interface Props {
  language: string
  value: string
  blockIndex?: number
}

const PREVIEW_MAX_HEIGHT = 420

const generatePreviewToken = () => {
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) {
    const bytes = new Uint8Array(16)
    crypto.getRandomValues(bytes)
    return Array.from(bytes)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
  }
  return `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

const getOrCreateWindowMap = <K, V>(key: string): Map<K, V> => {
  if (typeof window === "undefined") {
    return new Map<K, V>()
  }
  const win = window as unknown as Record<string, Map<K, V> | undefined>
  if (!win[key]) {
    win[key] = new Map<K, V>()
  }
  return win[key] as Map<K, V>
}

export const CodeBlock: FC<Props> = ({ language, value, blockIndex }) => {
  const [isBtnPressed, setIsBtnPressed] = useState(false)
  const [previewValue, setPreviewValue] = useState(value)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewHeight, setPreviewHeight] = useState(PREVIEW_MAX_HEIGHT)
  const [isPreviewConstrained, setIsPreviewConstrained] = useState(true)
  const [previewToken, setPreviewToken] = useState(() => generatePreviewToken())
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const copyTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const previewFrameRef = useRef<HTMLIFrameElement | null>(null)
  const normalizedLanguage = normalizeLanguage(language)
  const lines = value ? value.split(/\r?\n/) : []
  const totalLines = lines.length
  const isLong = totalLines > 15
  const artifactAutoThreshold = 10
  const [codeTheme] = useStorage("codeTheme", "auto")
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"
  const { openArtifact, isPinned } = useArtifactsStore()
  const { t } = useTranslation("common")

  const isMermaidLanguage = normalizedLanguage === "mermaid"
  const artifactId = computeKey()
  const artifactKind = isMermaidLanguage ? "diagram" : "code"
  const artifactOriginId = `artifact-origin-${artifactId}`
  const viewLabel = isMermaidLanguage
    ? t("view", "View")
    : t("artifactsView", "View code")
  const downloadLabel = t("downloadCode", "Download code")
  const copyLabel = t("copyToClipboard", "Copy to clipboard")
  
  function computeKey() {
    const content = value ?? ""
    const base =
      typeof blockIndex === "number"
        ? `${normalizedLanguage}::${blockIndex}`
        : `${normalizedLanguage}::${content.length}::${content.slice(0, 200)}::${content.slice(-200)}`
    let hash = 0
    for (let i = 0; i < base.length; i++) {
      hash = (hash * 31 + base.charCodeAt(i)) >>> 0
    }
    return hash.toString(36)
  }
  const keyRef = useRef<string>(computeKey())
  const previewMapRef = useRef<Map<string, boolean> | null>(null)
  const collapsedMapRef = useRef<Map<string, boolean> | null>(null)
  const autoOpenMapRef = useRef<Map<string, boolean> | null>(null)

  if (!previewMapRef.current) {
    previewMapRef.current = getOrCreateWindowMap<string, boolean>(
      "__codeBlockPreviewState"
    )
  }

  if (!collapsedMapRef.current) {
    collapsedMapRef.current = getOrCreateWindowMap<string, boolean>(
      "__codeBlockCollapsedState"
    )
  }

  if (!autoOpenMapRef.current) {
    autoOpenMapRef.current = getOrCreateWindowMap<string, boolean>(
      "__artifactAutoOpenState"
    )
  }

  const previewStateMap = previewMapRef.current!
  const collapsedStateMap = collapsedMapRef.current!
  const autoOpenStateMap = autoOpenMapRef.current!

  const [showPreview, setShowPreview] = useState<boolean>(() => {
    return previewStateMap.get(keyRef.current) || false
  })

  const [collapsed, setCollapsed] = useState<boolean>(() => {
    const stored = collapsedStateMap.get(keyRef.current)
    if (typeof stored === "boolean") return stored
    return isLong
  })
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setIsBtnPressed(true)
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
      copyTimeoutRef.current = setTimeout(() => {
        setIsBtnPressed(false)
        copyTimeoutRef.current = null
      }, 4000)
    } catch {
      // Clipboard write failed - optionally show error feedback
    }
  }

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
        copyTimeoutRef.current = null
      }
    }
  }, [])

  const isPreviewable = ["html", "svg", "xml"].includes(
    normalizedLanguage
  )

  const buildPreviewDoc = useCallback(() => {
    const rawCode = previewValue || ""
    const sanitizeLanguages = ["html", "xml"]
    const shouldSanitize = sanitizeLanguages.includes(normalizedLanguage)
    const code =
      shouldSanitize
        ? DOMPurify.sanitize(rawCode, { WHOLE_DOCUMENT: false })
        : rawCode
    const tokenLiteral = JSON.stringify(previewToken)
    const readyScript =
      `<script>(function(){var token=${tokenLiteral};function sendHeight(){var doc=document.documentElement;var body=document.body;var height=Math.max(doc?doc.scrollHeight:0,body?body.scrollHeight:0);window.parent&&window.parent.postMessage({type:"tldw-preview-resize",height:height,token:token},"*");}window.addEventListener("load",sendHeight);window.addEventListener("resize",sendHeight);if(window.ResizeObserver){var ro=new ResizeObserver(sendHeight);ro.observe(document.documentElement);}window.addEventListener("message",function(event){if(event&&event.data&&event.data.type==="tldw-preview-ping"&&event.data.token===token){sendHeight();}});window.parent&&window.parent.postMessage({type:"tldw-preview-ready",token:token},"*");sendHeight();})();</script>`
    if (normalizedLanguage === "svg") {
      const hasSvgTag = /<svg[\s>]/i.test(code)
      let svgMarkup = hasSvgTag
        ? code
        : `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>${code}</svg>`
      
      const hasWidthHeight = /\s(width|height)\s*=/.test(svgMarkup)
      
      if (!hasWidthHeight && hasSvgTag) {
        svgMarkup = svgMarkup.replace(
          /<svg([^>]*?)>/i,
          '<svg$1 width="100%" height="100%" style="max-width: 100%; max-height: 100%;">'
        )
      }
      
      const sanitizedSvg = DOMPurify.sanitize(svgMarkup, {
        USE_PROFILES: { svg: true }
      })
      return `<!doctype html><html><head><meta charset='utf-8'/><style>html,body{margin:0;padding:0;display:flex;align-items:center;justify-content:center;background:#fff;height:100%;overflow:hidden;}svg{max-width:100%;max-height:100%;}</style></head><body>${sanitizedSvg}${readyScript}</body></html>`
    }
    return `<!doctype html><html><head><meta charset='utf-8'/></head><body>${code}${readyScript}</body></html>`
  }, [previewValue, normalizedLanguage, previewToken])

  useEffect(() => {
    if (!showPreview || !isPreviewable) {
      setPreviewUrl(null)
      setPreviewHeight(PREVIEW_MAX_HEIGHT)
      return
    }
    const doc = buildPreviewDoc()
    const blob = new Blob([doc], { type: "text/html" })
    const url = window.URL.createObjectURL(blob)
    setPreviewUrl(url)
    setPreviewHeight(PREVIEW_MAX_HEIGHT)
    return () => {
      window.URL.revokeObjectURL(url)
    }
  }, [buildPreviewDoc, isPreviewable, showPreview])

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const frameWindow = previewFrameRef.current?.contentWindow
      if (!frameWindow || event.source !== frameWindow) return
      if (event.origin !== "null") return
      const data = event.data as { type?: string; height?: number; token?: string } | null
      if (!data || typeof data !== "object") return
      if (data.token !== previewToken) return
      if (data.type === "tldw-preview-ready") {
        frameWindow.postMessage({ type: "tldw-preview-ping", token: previewToken }, "*")
        return
      }
      if (data.type === "tldw-preview-resize" && typeof data.height === "number") {
        const clamped = Math.min(Math.max(Math.ceil(data.height), 120), 1000)
        setPreviewHeight(clamped)
      }
    }

    window.addEventListener("message", handleMessage)
    return () => {
      window.removeEventListener("message", handleMessage)
    }
  }, [previewToken])

  const handleDownload = () => {
    const blob = new Blob([value], { type: "text/plain" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    const rawExtension =
      programmingLanguages[normalizedLanguage] || normalizedLanguage || "txt"
    const sanitizedExtension = rawExtension
      .toString()
      .trim()
      .replace(/^\.+/, "")
      .replace(/[^a-zA-Z0-9_-]+/g, "")
      .toLowerCase()
    const finalExtension = sanitizedExtension || "txt"
    a.download = `code_${new Date().toISOString().replace(/[:.]/g, "-")}.${finalExtension}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  useEffect(() => {
    previewStateMap.set(keyRef.current, showPreview)
  }, [showPreview, previewStateMap])

  useEffect(() => {
    collapsedStateMap.set(keyRef.current, collapsed)
  }, [collapsed, collapsedStateMap])

  useEffect(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current)
    }
    
    debounceTimeoutRef.current = setTimeout(() => {
      setPreviewValue(value)
    }, 300) 
    
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current)
      }
    }
  }, [value])

  useEffect(() => {
    const newKey = computeKey()
    if (newKey !== keyRef.current) {
      keyRef.current = newKey
      setPreviewToken(generatePreviewToken())
      setPreviewHeight(PREVIEW_MAX_HEIGHT)
      setIsPreviewConstrained(true)
      if (previewStateMap.has(newKey)) {
        const prev = previewStateMap.get(newKey)!
        if (prev !== showPreview) setShowPreview(prev)
      } else if (showPreview) {
        setShowPreview(false)
      }
      if (collapsedStateMap.has(newKey)) {
        const prevCollapsed = collapsedStateMap.get(newKey)!
        if (prevCollapsed !== collapsed) setCollapsed(prevCollapsed)
      } else if (collapsed !== isLong) {
        setCollapsed(isLong)
      }
    }
  }, [
    normalizedLanguage,
    value,
    blockIndex,
    previewStateMap,
    collapsedStateMap,
    showPreview,
    collapsed,
    isLong
  ])

  useEffect(() => {
    if (!isPreviewable && showPreview) setShowPreview(false)
  }, [isPreviewable, showPreview])

  useEffect(() => {
    if (!isProMode) {
      return
    }
    if (!isMermaidLanguage && totalLines <= artifactAutoThreshold) {
      return
    }
    if (autoOpenStateMap.get(artifactId)) {
      return
    }
    if (isPinned) {
      return
    }
    openArtifact(
      {
        id: artifactId,
        title:
          normalizedLanguage && normalizedLanguage !== "plaintext"
            ? normalizedLanguage
            : t("artifactsDefaultTitle", "Code"),
        content: value,
        language: normalizedLanguage,
        kind: artifactKind,
        lineCount: totalLines
      },
      { auto: true }
    )
    autoOpenStateMap.set(artifactId, true)
  }, [
    artifactId,
    artifactAutoThreshold,
    autoOpenStateMap,
    isMermaidLanguage,
    isPinned,
    isProMode,
    normalizedLanguage,
    openArtifact,
    t,
    totalLines,
    value
  ])

  const handleOpenArtifact = () => {
    openArtifact({
      id: artifactId,
      title:
        normalizedLanguage && normalizedLanguage !== "plaintext"
          ? normalizedLanguage
          : t("artifactsDefaultTitle", "Code"),
      content: value,
      language: normalizedLanguage,
      kind: artifactKind,
      lineCount: totalLines
    })
    autoOpenStateMap.set(artifactId, true)
  }

  const previewContainerHeight = isPreviewConstrained
    ? Math.min(previewHeight, PREVIEW_MAX_HEIGHT)
    : previewHeight
  const isPreviewOversized = previewHeight > PREVIEW_MAX_HEIGHT
  const isPreviewScrollable = isPreviewConstrained && isPreviewOversized

  return (
    <>
      <div
        id={artifactOriginId}
        data-artifact-origin={artifactId}
        className="not-prose"
      >
        <div className=" [&_div+div]:!mt-0 my-4 bg-surface rounded-xl">
          <div className="flex flex-row px-4 py-2 rounded-t-xl gap-3 bg-surface2 items-center justify-between">
            <div className="flex items-center gap-3">
              {isPreviewable && !collapsed && (
                <div className="flex rounded-md overflow-hidden border border-border">
                  <button
                    onClick={() => setShowPreview(false)}
                    className={`px-2 flex items-center gap-1 text-xs transition-colors ${
                      !showPreview
                        ? "bg-surface text-text"
                        : "bg-transparent text-text-muted hover:bg-surface"
                    }`}
                    aria-label={t("showCode") || "Code"}>
                    <CodeIcon className="size-3" />
                  </button>
                  <button
                    onClick={() => setShowPreview(true)}
                    className={`px-2 flex items-center gap-1 text-xs transition-colors ${
                      showPreview
                        ? "bg-surface text-text"
                        : "bg-transparent text-text-muted hover:bg-surface"
                    }`}
                    aria-label={t("preview") || "Preview"}>
                    <EyeIcon className="size-3" />
                  </button>
                </div>
              )}
              {showPreview && isPreviewable && !collapsed && isPreviewOversized && (
                <Tooltip
                  title={
                    isPreviewConstrained
                      ? t("previewToggleExpand", "Expand preview")
                      : t("previewToggleClamp", "Clamp preview")
                  }
                >
                  <button
                    onClick={() => setIsPreviewConstrained((prev) => !prev)}
                    aria-pressed={isPreviewConstrained}
                    className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-1 text-[11px] text-text-muted hover:text-text"
                  >
                    <span>
                      {isPreviewConstrained
                        ? t("previewMaxHeight", "Max height")
                        : t("previewFitHeight", "Fit height")}
                    </span>
                  </button>
                </Tooltip>
              )}

              <span className="font-mono text-xs">
                {normalizedLanguage || "text"}
              </span>
              {isLong && (
                <span className="text-[10px] text-text-subtle">
                  {totalLines} {t("lines", "lines")}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleOpenArtifact}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-1 text-[11px] font-medium text-text-muted hover:text-text">
                <CodeIcon className="size-3" />
                <span>{viewLabel}</span>
              </button>
              {isLong && (
                <button
                  onClick={() => setCollapsed((prev) => !prev)}
                  aria-label={
                    collapsed
                      ? t("expand", "Expand code")
                      : t("collapse", "Collapse code")
                  }
                  aria-expanded={!collapsed}
                  className="inline-flex items-center gap-1 text-[11px] text-text-muted hover:text-text">
                  {collapsed ? (
                    <>
                      <ChevronDown className="size-3" />
                      <span>{t("expand", "Expand")}</span>
                    </>
                  ) : (
                    <>
                      <ChevronUp className="size-3" />
                      <span>{t("collapse", "Collapse")}</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
          <div className="sticky top-9 md:top-[5.75rem]">
            <div className="absolute bottom-0 right-2 flex h-9 items-center gap-1">
              <Tooltip title={downloadLabel}>
                <button
                  onClick={handleDownload}
                  aria-label={downloadLabel}
                  className="flex gap-1.5 items-center rounded bg-none p-1 text-xs text-text-muted hover:bg-surface2 hover:text-text focus:outline-none">
                  <DownloadIcon className="size-4" />
                </button>
              </Tooltip>
              <Tooltip title={copyLabel}>
                <button
                  onClick={handleCopy}
                  aria-label={copyLabel}
                  className="flex gap-1.5 items-center rounded bg-none p-1 text-xs text-text-muted hover:bg-surface2 hover:text-text focus:outline-none">
                  {!isBtnPressed ? (
                    <CopyIcon className="size-4" />
                  ) : (
                    <CopyCheckIcon className="size-4 text-success" />
                  )}
                </button>
              </Tooltip>
            </div>
          </div>

          {collapsed ? (
            <div className="relative px-4 py-3">
              <pre className="text-xs font-mono text-text max-h-36 overflow-hidden whitespace-pre-wrap">
                {lines.slice(0, 3).join("\n")}
                {totalLines > 3 ? "…" : ""}
              </pre>
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent" />
            </div>
          ) : (
            <>
              {!showPreview && (
                <Highlight
                  code={value}
                  language={safeLanguage(normalizedLanguage)}
                  theme={resolveTheme(codeTheme || "dracula")}>
                  {({
                    className: highlightClassName,
                    style,
                    tokens,
                    getLineProps,
                    getTokenProps
                  }) => (
                    <pre
                      className={`${highlightClassName} m-0 w-full bg-transparent px-4 py-3 text-[0.9rem]`}
                      style={{
                        ...style,
                        fontFamily: "var(--font-mono)"
                      }}>
                      {tokens.map((line, i) => (
                        <div
                          key={i}
                          {...getLineProps({ line })}
                          className="table w-full">
                          <span className="table-cell select-none pr-4 text-right text-xs text-text-subtle">
                            {i + 1}
                          </span>
                          <span className="table-cell whitespace-pre-wrap">
                            {line.map((token, key) => (
                              <span
                                key={key}
                                {...getTokenProps({ token })}
                              />
                            ))}
                          </span>
                        </div>
                      ))}
                    </pre>
                  )}
                </Highlight>
              )}
              {showPreview && isPreviewable && (
                <div
                  className={`w-full bg-surface rounded-b-xl border-t border-border overflow-x-hidden ${
                    isPreviewScrollable ? "overflow-y-auto" : "overflow-hidden"
                  }`}
                  style={{ height: previewContainerHeight }}
                >
                  <iframe
                    title={t("preview", "Preview")}
                    ref={previewFrameRef}
                    src={previewUrl ?? "about:blank"}
                    className="w-full h-full border-0"
                    sandbox="allow-scripts"
                  />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  )
}
