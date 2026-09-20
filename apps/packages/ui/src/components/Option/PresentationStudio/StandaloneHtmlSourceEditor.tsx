import React from "react"
import type { OnMount } from "@monaco-editor/react"

import {
  preflightStandaloneHtmlSource,
  validateStandaloneHtmlSource,
  type AcceptedStandaloneHtmlSource
} from "./standalone-html-source"

const MonacoEditor = React.lazy(() => import("@monaco-editor/react"))

type StandaloneHtmlSourceEditorProps = {
  value: string
  draftSeed?: string | null
  onAcceptedChange: (source: AcceptedStandaloneHtmlSource) => void
  onPreflightCandidate?: (source: string) => void
  onPendingChange?: (source: string | null) => void
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
  contextmenu: false,
  mouseMiddleClickAction: "default" as const,
  hover: { enabled: false },
  quickSuggestions: false,
  suggestOnTriggerCharacters: false,
  parameterHints: { enabled: false },
  lightbulb: { enabled: false },
  unicodeHighlight: { ambiguousCharacters: true, invisibleCharacters: true },
  ariaLabel: "HTML source"
}

type MonacoEditorLike = Parameters<OnMount>[0]

const disposeNoThrow = (value: { dispose: () => void } | null | undefined) => {
  try {
    value?.dispose()
  } catch {
    // Sensitive editor cleanup continues through every independently owned resource.
  }
}

const isNavigationKey = (event: KeyboardEvent): boolean =>
  event.key === "F12" || ((event.metaKey || event.ctrlKey) && event.key === "Enter")

export const installStandaloneHtmlMonacoGuards = (editor: MonacoEditorLike) => {
  editor.updateOptions(editorOptions as any)
  const root = editor.getDomNode()
  if (!root) return { dispose: () => undefined }

  const block = (event: Event) => {
    event.preventDefault()
    event.stopPropagation()
  }
  const blockNavigationClick = (event: MouseEvent) => {
    const target = event.target
    if (
      event.metaKey ||
      event.ctrlKey ||
      (target instanceof Element && Boolean(target.closest("a")))
    ) {
      block(event)
    }
  }
  const blockNavigationKey = (event: KeyboardEvent) => {
    if (isNavigationKey(event)) block(event)
  }
  const blockMiddlePointer = (event: MouseEvent) => {
    if (event.button === 1 || event.metaKey || event.ctrlKey) block(event)
  }
  root.addEventListener("click", blockNavigationClick, true)
  root.addEventListener("pointerdown", blockMiddlePointer, true)
  root.addEventListener("mousedown", blockMiddlePointer, true)
  root.addEventListener("mouseup", blockMiddlePointer, true)
  root.addEventListener("pointerup", blockMiddlePointer, true)
  root.addEventListener("auxclick", block, true)
  root.addEventListener("contextmenu", block, true)
  root.addEventListener("keydown", blockNavigationKey, true)
  return {
    dispose: () => {
      root.removeEventListener("click", blockNavigationClick, true)
      root.removeEventListener("pointerdown", blockMiddlePointer, true)
      root.removeEventListener("mousedown", blockMiddlePointer, true)
      root.removeEventListener("mouseup", blockMiddlePointer, true)
      root.removeEventListener("pointerup", blockMiddlePointer, true)
      root.removeEventListener("auxclick", block, true)
      root.removeEventListener("contextmenu", block, true)
      root.removeEventListener("keydown", blockNavigationKey, true)
    }
  }
}

export const StandaloneHtmlSourceEditor = React.forwardRef<
  StandaloneHtmlSourceEditorHandle,
  StandaloneHtmlSourceEditorProps
>(({
  value,
  draftSeed = null,
  onAcceptedChange,
  onPreflightCandidate,
  onPendingChange,
  forceFallback = false,
  readOnly = false
}, ref) => {
  const editorId = React.useId()
  const labelId = React.useId()
  const effectiveOptions = React.useMemo(
    () => ({ ...editorOptions, readOnly }),
    [readOnly]
  )
  const [useFallback, setUseFallback] = React.useState(false)
  const [draftValue, setDraftValue] = React.useState(
    () => draftSeed !== null ? draftSeed : value
  )
  const [reason, setReason] = React.useState<string | null>(null)
  const validationEpochRef = React.useRef(0)
  const pendingCallbackRef = React.useRef(onPendingChange)
  const appliedExternalValueRef = React.useRef(value)
  pendingCallbackRef.current = onPendingChange
  const externalValueRef = React.useRef(value)
  const acceptedValueRef = React.useRef(value)
  const rollingBackRef = React.useRef(false)
  const editorRef = React.useRef<Parameters<OnMount>[0] | null>(null)
  const modelRef = React.useRef<ReturnType<Parameters<OnMount>[0]["getModel"]>>(null)
  const guardRef = React.useRef<{ dispose: () => void } | null>(null)

  if (externalValueRef.current !== value) {
    externalValueRef.current = value
    acceptedValueRef.current = value
    validationEpochRef.current += 1
  }

  React.useEffect(() => {
    if (appliedExternalValueRef.current === value) return
    appliedExternalValueRef.current = value
    setDraftValue(value)
    setReason(null)
    pendingCallbackRef.current?.(null)
  }, [value])

  const rollBackEditor = React.useCallback(() => {
    const editor = editorRef.current
    const acceptedValue = acceptedValueRef.current
    setDraftValue(acceptedValue)
    if (!editor || editor.getValue() === acceptedValue) return
    rollingBackRef.current = true
    try {
      editor.setValue(acceptedValue)
    } finally {
      rollingBackRef.current = false
    }
  }, [])

  React.useEffect(() => {
    if (!readOnly) return
    validationEpochRef.current += 1
    pendingCallbackRef.current?.(null)
    rollBackEditor()
  }, [readOnly, rollBackEditor])

  const acceptCandidate = React.useCallback(
    (candidate: string) => {
      if (rollingBackRef.current) return
      if (readOnly) {
        onPendingChange?.(null)
        rollBackEditor()
        return
      }
      const preflight = preflightStandaloneHtmlSource(candidate)
      if (preflight.ok === false) {
        validationEpochRef.current += 1
        onPendingChange?.(null)
        setReason(preflight.message)
        rollBackEditor()
        return
      }
      setDraftValue(candidate)
      onPreflightCandidate?.(candidate)
      onPendingChange?.(candidate)
      const epoch = ++validationEpochRef.current
      void validateStandaloneHtmlSource(candidate).then((result) => {
        if (epoch !== validationEpochRef.current) return
        if (result.ok === false) {
          onPendingChange?.(null)
          setReason(result.message)
          rollBackEditor()
          return
        }
        setReason(null)
        onAcceptedChange(result)
        onPendingChange?.(null)
      })
    },
    [onAcceptedChange, onPendingChange, onPreflightCandidate, readOnly, rollBackEditor]
  )

  const handleMount = React.useCallback<OnMount>(
    (editor) => {
      editorRef.current = editor
      modelRef.current = editor.getModel()
      editor.updateOptions(effectiveOptions as any)
      const previousGuard = guardRef.current
      guardRef.current = null
      disposeNoThrow(previousGuard)
      guardRef.current = installStandaloneHtmlMonacoGuards(editor)
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

  const retireEditorSurface = React.useCallback(() => {
    const guard = guardRef.current
    guardRef.current = null
    const model = modelRef.current
    const editor = editorRef.current
    modelRef.current = null
    editorRef.current = null
    disposeNoThrow(guard)
    disposeNoThrow(model)
    disposeNoThrow(editor)
  }, [])

  const dispose = React.useCallback(() => {
    validationEpochRef.current += 1
    retireEditorSurface()
  }, [retireEditorSurface])

  const switchToFallback = React.useCallback(() => {
    retireEditorSurface()
    setUseFallback(true)
  }, [retireEditorSurface])

  React.useImperativeHandle(ref, () => ({ dispose }), [dispose])
  React.useEffect(() => dispose, [dispose])

  const fallback = (
    <textarea
      id={editorId}
      aria-label="HTML source"
      className="min-h-[28rem] w-full resize-y rounded-lg border border-border bg-bg p-3 font-mono text-sm text-text shadow-sm outline-none focus-visible:border-primary focus-visible:ring-2 focus-visible:ring-focus motion-reduce:transition-none"
      value={draftValue}
      readOnly={readOnly}
      spellCheck={false}
      autoCorrect="off"
      autoCapitalize="off"
      autoComplete="off"
      data-1p-ignore="true"
      data-lpignore="true"
      onChange={(event) => {
        acceptCandidate(event.target.value)
      }}
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
        <StandaloneMonacoBoundary onError={switchToFallback}>
          <React.Suspense fallback={fallback}>
            <MonacoEditor
              defaultLanguage="plaintext"
              language="plaintext"
              value={draftValue}
              onChange={(candidate) => acceptCandidate(candidate ?? "")}
              onMount={handleMount}
              height="28rem"
              keepCurrentModel
              saveViewState={false}
              wrapperProps={{ id: editorId, "aria-labelledby": labelId }}
              options={effectiveOptions as any}
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
