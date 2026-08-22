import React from "react"
import type { OnMount } from "@monaco-editor/react"

import {
  validateStandaloneHtmlSource,
  type AcceptedStandaloneHtmlSource
} from "./standalone-html-source"

const MonacoEditor = React.lazy(() => import("@monaco-editor/react"))

type StandaloneHtmlSourceEditorProps = {
  value: string
  onAcceptedChange: (source: AcceptedStandaloneHtmlSource) => void
  forceFallback?: boolean
  readOnly?: boolean
}

export type StandaloneHtmlSourceEditorHandle = {
  dispose: () => void
}

class StandaloneMonacoBoundary extends React.Component<
  { children: React.ReactNode; onError: () => void },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch() {
    this.props.onError()
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}

const editorOptions = {
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  wordWrap: "on" as const,
  fontSize: 13,
  lineNumbers: "on" as const,
  links: false,
  hover: { enabled: false },
  quickSuggestions: false,
  suggestOnTriggerCharacters: false,
  parameterHints: { enabled: false },
  lightbulb: { enabled: false },
  unicodeHighlight: { ambiguousCharacters: true, invisibleCharacters: true },
  ariaLabel: "HTML source"
}

export const StandaloneHtmlSourceEditor = React.forwardRef<
  StandaloneHtmlSourceEditorHandle,
  StandaloneHtmlSourceEditorProps
>(({ value, onAcceptedChange, forceFallback = false, readOnly = false }, ref) => {
  const editorId = React.useId()
  const labelId = React.useId()
  const overrideServices = React.useMemo(
    () => ({ openerService: { open: async () => false } }),
    []
  )
  const effectiveOptions = React.useMemo(
    () => ({ ...editorOptions, readOnly }),
    [readOnly]
  )
  const [useFallback, setUseFallback] = React.useState(false)
  const [reason, setReason] = React.useState<string | null>(null)
  const validationEpochRef = React.useRef(0)
  const editorRef = React.useRef<Parameters<OnMount>[0] | null>(null)
  const modelRef = React.useRef<ReturnType<Parameters<OnMount>[0]["getModel"]>>(null)

  const acceptCandidate = React.useCallback(
    (candidate: string) => {
      const epoch = ++validationEpochRef.current
      void validateStandaloneHtmlSource(candidate).then((result) => {
        if (epoch !== validationEpochRef.current) return
        if (result.ok === false) {
          setReason(result.message)
          return
        }
        setReason(null)
        onAcceptedChange(result)
      })
    },
    [onAcceptedChange]
  )

  const handleMount = React.useCallback<OnMount>(
    (editor) => {
      editorRef.current = editor
      modelRef.current = editor.getModel()
      editor.updateOptions(effectiveOptions as any)
      const input = editor.getDomNode()?.querySelector("textarea")
      input?.removeAttribute("name")
      input?.setAttribute("aria-labelledby", labelId)
      input?.setAttribute("spellcheck", "false")
      input?.setAttribute("autocorrect", "off")
      input?.setAttribute("autocapitalize", "off")
      input?.setAttribute("autocomplete", "off")
      input?.setAttribute("data-1p-ignore", "true")
      input?.setAttribute("data-lpignore", "true")
    },
    [effectiveOptions, labelId]
  )

  const dispose = React.useCallback(() => {
    validationEpochRef.current += 1
    const model = modelRef.current
    const editor = editorRef.current
    modelRef.current = null
    editorRef.current = null
    model?.dispose()
    editor?.dispose()
  }, [])

  React.useImperativeHandle(ref, () => ({ dispose }), [dispose])
  React.useEffect(() => dispose, [dispose])

  const fallback = (
    <textarea
      id={editorId}
      aria-label="HTML source"
      className="min-h-[28rem] w-full resize-y rounded-lg border border-border bg-bg p-3 font-mono text-sm text-text shadow-sm outline-none focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-focus motion-reduce:transition-none"
      value={value}
      readOnly={readOnly}
      spellCheck={false}
      autoCorrect="off"
      autoCapitalize="off"
      autoComplete="off"
      data-1p-ignore="true"
      data-lpignore="true"
      onChange={(event) => acceptCandidate(event.target.value)}
    />
  )

  return (
    <div className="space-y-2">
      <label id={labelId} htmlFor={editorId} className="block text-sm font-semibold text-text">
        HTML source
      </label>
      {forceFallback || useFallback ? (
        fallback
      ) : (
        <StandaloneMonacoBoundary onError={() => setUseFallback(true)}>
          <React.Suspense fallback={fallback}>
            <MonacoEditor
              defaultLanguage="plaintext"
              language="plaintext"
              value={value}
              onChange={(candidate) => acceptCandidate(candidate ?? "")}
              onMount={handleMount}
              height="28rem"
              keepCurrentModel
              wrapperProps={{ id: editorId, "aria-labelledby": labelId }}
              options={effectiveOptions as any}
              overrideServices={overrideServices}
            />
          </React.Suspense>
        </StandaloneMonacoBoundary>
      )}
      {reason ? (
        <p role="alert" className="text-sm text-danger">
          {reason}
        </p>
      ) : null}
    </div>
  )
})

StandaloneHtmlSourceEditor.displayName = "StandaloneHtmlSourceEditor"

export default StandaloneHtmlSourceEditor
