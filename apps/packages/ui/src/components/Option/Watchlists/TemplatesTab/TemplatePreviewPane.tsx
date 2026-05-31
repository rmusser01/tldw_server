import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Radio, Select, Spin } from "antd"
import DOMPurify from "dompurify"
import { marked } from "marked"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives/Alert"
import {
  flowCheckWatchlistTemplateSections,
  previewWatchlistTemplate,
  type TemplateComposerFlowCheckMode,
  type TemplateComposerFlowIssue,
  type TemplateComposerFlowSection
} from "@/services/watchlists"
import { trackWatchlistsPreventionTelemetry } from "@/utils/watchlists-prevention-telemetry"
import { FlowCheckDiffPanel } from "./FlowCheckDiffPanel"

interface TemplatePreviewPaneProps {
  content: string
  format: "md" | "html"
  /** Available runs to preview against (id + label) */
  runs?: Array<{ id: number; label: string }>
  sections?: TemplateComposerFlowSection[]
  onApplyFlowSections?: (sections: TemplateComposerFlowSection[]) => void
}

export const TemplatePreviewPane: React.FC<TemplatePreviewPaneProps> = ({
  content,
  format,
  runs,
  sections,
  onApplyFlowSections
}) => {
  const { t } = useTranslation(["watchlists"])
  const [mode, setMode] = useState<"static" | "live">("static")
  const [selectedRunId, setSelectedRunId] = useState<number | undefined>(undefined)
  const [liveRendered, setLiveRendered] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [warnings, setWarnings] = useState<string[]>([])
  const [flowMode, setFlowMode] = useState<TemplateComposerFlowCheckMode>("suggest_only")
  const [flowIssues, setFlowIssues] = useState<TemplateComposerFlowIssue[]>([])
  const [flowDiff, setFlowDiff] = useState("")
  const [flowSections, setFlowSections] = useState<TemplateComposerFlowSection[]>([])
  const [flowLoading, setFlowLoading] = useState(false)
  const [flowError, setFlowError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleModeChange = (nextMode: "static" | "live") => {
    setMode(nextMode)
    void trackWatchlistsPreventionTelemetry({
      type: "watchlists_template_preview_mode_changed",
      surface: "template_editor",
      mode: nextMode
    })
  }

  // Static preview: render markup locally (no Jinja2 evaluation)
  const staticHtml = React.useMemo(() => {
    if (!content) return ""
    if (format === "md") {
      const rendered = marked.parse(content)
      return DOMPurify.sanitize(String(rendered), { USE_PROFILES: { html: true } })
    }
    return DOMPurify.sanitize(content, { USE_PROFILES: { html: true } })
  }, [content, format])

  // Live preview: debounced server-side render
  const fetchLivePreview = useCallback(async () => {
    if (!selectedRunId || !content.trim()) {
      setLiveRendered("")
      setWarnings([])
      setError(null)
      return
    }
    // Cancel previous request
    if (abortRef.current) {
      abortRef.current.abort()
    }
    abortRef.current = new AbortController()

    setLoading(true)
    setError(null)
    try {
      const result = await previewWatchlistTemplate(content, selectedRunId, format, abortRef.current.signal)
      setLiveRendered(result.rendered)
      setWarnings(result.warnings || [])
      setError(null)
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_template_preview_rendered",
        surface: "template_editor",
        mode: "live",
        status: "success",
        warning_count: Array.isArray(result.warnings) ? result.warnings.length : 0,
        run_id: selectedRunId
      })
    } catch (err: any) {
      if (err?.name === "AbortError") return
      setError(err?.message || "Preview failed")
      setLiveRendered("")
      setWarnings([])
      void trackWatchlistsPreventionTelemetry({
        type: "watchlists_template_preview_rendered",
        surface: "template_editor",
        mode: "live",
        status: "error",
        warning_count: 0,
        run_id: selectedRunId
      })
    } finally {
      setLoading(false)
    }
  }, [content, selectedRunId, format])

  // Debounce live preview
  useEffect(() => {
    if (mode !== "live" || !selectedRunId) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      fetchLivePreview()
    }, 500)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [content, selectedRunId, mode, fetchLivePreview])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort()
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [])

  const runFlowCheck = useCallback(async () => {
    if (!selectedRunId) {
      return
    }

    setFlowLoading(true)
    setFlowError(null)

    try {
      const result = await flowCheckWatchlistTemplateSections({
        run_id: selectedRunId,
        mode: flowMode,
        sections: Array.isArray(sections) ? sections : []
      })

      setFlowIssues(Array.isArray(result.issues) ? result.issues : [])
      setFlowDiff(String(result.diff || ""))
      setFlowSections(Array.isArray(result.sections) ? result.sections : [])

      if (result.mode === "auto_apply" && Array.isArray(result.sections) && result.sections.length > 0) {
        onApplyFlowSections?.(result.sections)
      }
    } catch (err: any) {
      setFlowError(String(err?.message || "Flow-check failed"))
      setFlowIssues([])
      setFlowDiff("")
      setFlowSections([])
    } finally {
      setFlowLoading(false)
    }
  }, [flowMode, onApplyFlowSections, sections, selectedRunId])

  const acceptFlowDiff = useCallback(
    (_chunkId?: string) => {
      if (flowSections.length > 0) {
        onApplyFlowSections?.(flowSections)
      }
      setFlowDiff("")
      setFlowIssues([])
      setFlowSections([])
      setFlowError(null)
    },
    [flowSections, onApplyFlowSections]
  )

  const rejectFlowDiff = useCallback((_chunkId?: string) => {
    setFlowDiff("")
    setFlowIssues([])
    setFlowSections([])
    setFlowError(null)
  }, [])

  const livePreviewHtml = React.useMemo(() => {
    if (!liveRendered) return ""
    if (format === "md") {
      const rendered = marked.parse(liveRendered)
      return DOMPurify.sanitize(String(rendered), { USE_PROFILES: { html: true } })
    }
    return DOMPurify.sanitize(liveRendered, { USE_PROFILES: { html: true } })
  }, [liveRendered, format])

  const hasRuns = Array.isArray(runs) && runs.length > 0

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4">
        <Radio.Group
          value={mode}
          onChange={(e) => handleModeChange(e.target.value as "static" | "live")}
          size="small"
        >
          <Radio.Button value="static">
            {t("watchlists:templates.preview.mode.static", "Static markup")}
          </Radio.Button>
          <Radio.Button value="live">
            {t("watchlists:templates.preview.mode.live", "Live (render with run data)")}
          </Radio.Button>
        </Radio.Group>

        {mode === "live" && (
          <Select
            value={selectedRunId}
            onChange={setSelectedRunId}
            placeholder={t("watchlists:templates.preview.runPlaceholder", "Select a run…")}
            size="small"
            className="min-w-[200px]"
            allowClear
            options={(runs || []).map((r) => ({
              value: r.id,
              label: r.label,
            }))}
          />
        )}

        {loading && <Spin size="small" />}
      </div>

      <div className="text-xs text-text-muted" data-testid="template-preview-mode-note">
        {mode === "static"
          ? t(
              "watchlists:templates.preview.staticNote",
              "Static preview renders markdown/html locally and does not evaluate Jinja2 control flow."
            )
          : t(
              "watchlists:templates.preview.liveNote",
              "Live preview renders with data from a completed run to validate loops, variables, and conditionals."
            )}
      </div>

      {mode === "live" && !hasRuns && (
        <Alert
          variant="warning"
          title={t(
            "watchlists:templates.preview.noRunsTitle",
            "No completed runs available for live preview."
          )}
        >
          {t(
            "watchlists:templates.preview.noRunsDescription",
            "Run a monitor once from Activity, then return here to preview templates with real data."
          )}
        </Alert>
      )}

      {mode === "live" && hasRuns && !selectedRunId && (
        <Alert
          variant="info"
          title={t(
            "watchlists:templates.preview.selectRunTitle",
            "Select a run to preview the template with real data."
          )}
        />
      )}

      {error && (
        <Alert
          variant="error"
          title={t("watchlists:templates.preview.renderErrorTitle", "Live preview failed")}
        >
          <div>
            <div>
              {t(
                "watchlists:templates.preview.renderErrorHint",
                "Check template syntax or choose another run, then try live preview again."
              )}
            </div>
            <div className="mt-1 text-xs text-text-muted">{error}</div>
          </div>
        </Alert>
      )}

      {warnings.length > 0 && (
        <Alert
          variant="warning"
          title={t("watchlists:templates.preview.renderWarningsTitle", "Render warnings")}
        >
          <ul className="mb-0 pl-4">
            {warnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </Alert>
      )}

      {mode === "static" ? (
        staticHtml ? (
          <div
            className="prose dark:prose-invert max-w-none p-4 bg-surface rounded-lg border border-border overflow-auto max-h-96"
            dangerouslySetInnerHTML={{ __html: staticHtml }}
          />
        ) : (
          <div className="text-sm text-text-muted p-4">
            {t("watchlists:templates.preview.empty", "Nothing to preview yet.")}
          </div>
        )
      ) : (
        livePreviewHtml ? (
          <div
            className="prose dark:prose-invert max-w-none p-4 bg-surface rounded-lg border border-border overflow-auto max-h-96"
            dangerouslySetInnerHTML={{ __html: livePreviewHtml }}
          />
        ) : !loading && selectedRunId ? (
          <div className="text-sm text-text-muted p-4">
            {t(
              "watchlists:templates.preview.liveEmpty",
              "No preview content yet. The template will render after a short delay."
            )}
          </div>
        ) : null
      )}

      <div className="rounded-lg border border-border p-3 space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <div className="text-xs font-medium text-text-muted">Final flow-check</div>
          <Radio.Group
            value={flowMode}
            onChange={(event) => setFlowMode(event.target.value)}
            size="small"
            optionType="button"
          >
            <Radio.Button value="suggest_only">Suggest only</Radio.Button>
            <Radio.Button value="auto_apply">Auto apply</Radio.Button>
          </Radio.Group>
          <Button
            size="small"
            onClick={() => void runFlowCheck()}
            loading={flowLoading}
            disabled={!selectedRunId}
          >
            Run flow-check
          </Button>
        </div>

        {flowError ? <Alert variant="error" title={flowError} /> : null}

        {(flowDiff || flowIssues.length > 0) ? (
          <FlowCheckDiffPanel
            diff={flowDiff}
            mode={flowMode}
            issues={flowIssues}
            onModeChange={setFlowMode}
            onAcceptChunk={() => acceptFlowDiff()}
            onRejectChunk={() => rejectFlowDiff()}
            onRevertAll={() => rejectFlowDiff()}
          />
        ) : null}
      </div>
    </div>
  )
}

export default TemplatePreviewPane
