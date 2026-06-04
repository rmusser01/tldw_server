import React, { useEffect, useRef, useState } from "react"

export type MermaidTheme =
  | "default"
  | "base"
  | "dark"
  | "forest"
  | "neutral"
  | "null"
export type MermaidRenderStatus = "idle" | "rendering" | "success" | "error"
export type MermaidRenderState = {
  status: MermaidRenderStatus
  svg?: string
  error?: string
  theme?: MermaidTheme
}

export type MermaidProps = {
  code: string
  className?: string
  ariaLabel?: string
  onRenderStateChange?: (state: MermaidRenderState) => void
}

const hashCode = (input: string) => {
  let hash = 0
  for (let i = 0; i < input.length; i++) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0
  }
  return hash.toString(36)
}

const resolveMermaidTheme = (): MermaidTheme => {
  try {
    if (typeof document !== "undefined") {
      const root = document.documentElement
      if (root.classList.contains("dark")) {
        return "dark"
      }
      if (root.classList.contains("light")) {
        return "default"
      }
    }
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "default"
    }
  } catch {
    return "default"
  }
  return "default"
}

export const Mermaid: React.FC<MermaidProps> = ({
  code,
  className,
  ariaLabel,
  onRenderStateChange
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [theme, setTheme] = useState<MermaidTheme>(() => resolveMermaidTheme())
  const renderIdRef = useRef(0)
  const renderSequenceRef = useRef(0)
  const initializedThemeRef = useRef<MermaidTheme | null>(null)
  const onRenderStateChangeRef =
    useRef<MermaidProps["onRenderStateChange"]>(undefined)
  onRenderStateChangeRef.current = onRenderStateChange

  useEffect(() => {
    if (typeof window === "undefined") return

    const updateTheme = () => {
      setTheme(resolveMermaidTheme())
    }

    updateTheme()

    const mql = window.matchMedia?.("(prefers-color-scheme: dark)")
    const handleChange = () => updateTheme()
    const cleanupMql = () => {
      if (!mql) return
      if (mql.removeEventListener) {
        mql.removeEventListener("change", handleChange)
      } else if (mql.removeListener) {
        mql.removeListener(handleChange)
      }
    }
    if (mql) {
      if (mql.addEventListener) {
        mql.addEventListener("change", handleChange)
      } else if (mql.addListener) {
        mql.addListener(handleChange)
      }
    }

    if (typeof document === "undefined" || typeof MutationObserver === "undefined") {
      return cleanupMql
    }

    const observer = new MutationObserver(handleChange)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"]
    })

    return () => {
      cleanupMql()
      observer.disconnect()
    }
  }, [])

  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }

    const renderSequence = renderSequenceRef.current + 1
    renderSequenceRef.current = renderSequence
    const isCurrentRender = () => renderSequenceRef.current === renderSequence
    const reportRenderState = (state: MermaidRenderState) => {
      if (isCurrentRender()) {
        onRenderStateChangeRef.current?.(state)
      }
    }

    const renderDiagram = async () => {
      if (!code?.trim()) {
        setError(null)
        if (containerRef.current) {
          containerRef.current.innerHTML = ""
        }
        reportRenderState({ status: "idle", theme })
        return
      }

      reportRenderState({ status: "rendering", theme })
      try {
        const mermaidModule = await import("mermaid")
        if (!isCurrentRender()) {
          return
        }
        const mermaid = mermaidModule.default

        if (initializedThemeRef.current !== theme) {
          mermaid.initialize({
            startOnLoad: false,
            theme,
            securityLevel: "strict"
          })
          initializedThemeRef.current = theme
        }
        const id = `mermaid-${hashCode(code)}-${renderIdRef.current++}`
        const { svg, bindFunctions } = await mermaid.render(id, code)

        if (!isCurrentRender() || !containerRef.current) {
          return
        }

        containerRef.current.innerHTML = svg
        bindFunctions?.(containerRef.current)
        setError(null)
        reportRenderState({ status: "success", svg, theme })
      } catch (err) {
        if (!isCurrentRender()) return
        const message =
          err instanceof Error ? err.message : "Unable to render diagram."
        setError(message)
        if (containerRef.current) {
          containerRef.current.textContent = code
        }
        reportRenderState({ status: "error", error: message, theme })
      }
    }

    renderDiagram()
    return () => {
      if (renderSequenceRef.current === renderSequence) {
        renderSequenceRef.current += 1
      }
    }
  }, [code, theme])

  if (typeof window === "undefined") {
    return (
      <pre className="whitespace-pre-wrap text-xs text-text">{code}</pre>
    )
  }

  return (
    <div className={className}>
      <div
        ref={containerRef}
        aria-label={ariaLabel ?? "Mermaid diagram"}
        role="img"
        className="flex items-center justify-center"
      />
      {error && (
        <div className="mt-2 rounded-md border border-border bg-surface px-3 py-2 text-xs text-text-muted">
          {error}
        </div>
      )}
    </div>
  )
}

export default Mermaid
