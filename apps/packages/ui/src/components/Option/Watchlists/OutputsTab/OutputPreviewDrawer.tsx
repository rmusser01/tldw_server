import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import {
  Button,
  Drawer,
  Empty,
  Segmented,
  Spin,
  Tag,
  Tooltip,
  message
} from "antd"
import { Download, ExternalLink, MessageSquare } from "lucide-react"
import DOMPurify from "dompurify"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { setSetting } from "@/services/settings"
import { DISCUSS_WATCHLIST_PROMPT_SETTING } from "@/services/settings/ui-settings"
import type { WatchlistChatHandoffPayload } from "@/services/tldw/watchlist-chat-handoff"
import { downloadWatchlistOutput, downloadWatchlistOutputBinary } from "@/services/watchlists"
import type { WatchlistOutput } from "@/types/watchlists"
import {
  getFocusableActiveElement,
  restoreFocusToElement
} from "../shared/focus-management"
import {
  type AudioArtifactSummary,
  getDeliveryStatusColor,
  getOutputArtifactLabel,
  getOutputFileExtension,
  getOutputAudioStatusSummary,
  getOutputDeliveryStatuses,
  getOutputMimeType,
  getOutputTemplateName,
  getOutputTemplateVersion,
  isAudioOutput
} from "./outputMetadata"
import { ReportEvidencePanel } from "./ReportEvidencePanel"

interface OutputPreviewDrawerProps {
  output: WatchlistOutput | null | undefined
  open: boolean
  onClose: () => void
}

const useSafeNavigate = () => {
  try {
    return useNavigate()
  } catch {
    return null
  }
}

export const OutputPreviewDrawer: React.FC<OutputPreviewDrawerProps> = ({
  output,
  open,
  onClose
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const navigate = useSafeNavigate()

  const [loading, setLoading] = useState(false)
  const [content, setContent] = useState<string | null>(null)
  const [audioObjectUrl, setAudioObjectUrl] = useState<string | null>(null)
  const audioObjectUrlRef = useRef<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<"rendered" | "source">("rendered")
  const outputIsAudio = useMemo(() => isAudioOutput(output), [output])
  const restoreFocusTargetRef = useRef<HTMLElement | null>(null)
  const wasOpenRef = useRef(false)

  const navigateHome = useCallback(() => {
    if (navigate) {
      navigate("/")
      return
    }

    if (typeof window !== "undefined") {
      window.location.hash = "#/"
    }
  }, [navigate])

  const handleChatAboutOutput = useCallback(() => {
    if (!output) return
    const payload: WatchlistChatHandoffPayload = {
      articles: [
        {
          title: output.title || `Output #${output.id}`,
          content: content || undefined,
          sourceType: "output",
          mediaId: output.media_item_id ?? undefined
        }
      ]
    }
    void setSetting(DISCUSS_WATCHLIST_PROMPT_SETTING, payload)
    window.dispatchEvent(
      new CustomEvent("tldw:discuss-watchlist", { detail: payload })
    )
    navigateHome()
  }, [output, content, navigateHome])

  useLayoutEffect(() => {
    if (open) {
      if (!wasOpenRef.current) {
        restoreFocusTargetRef.current = getFocusableActiveElement()
      }
      wasOpenRef.current = true
      return
    }

    if (wasOpenRef.current) {
      wasOpenRef.current = false
      restoreFocusToElement(restoreFocusTargetRef.current)
    }
  }, [open])

  const updateAudioObjectUrl = useCallback((nextUrl: string | null) => {
    if (audioObjectUrlRef.current && audioObjectUrlRef.current !== nextUrl) {
      URL.revokeObjectURL(audioObjectUrlRef.current)
    }
    audioObjectUrlRef.current = nextUrl
    setAudioObjectUrl(nextUrl)
  }, [])

  // Fetch content when drawer opens
  useEffect(() => {
    if (open && output) {
      setLoading(true)
      setError(null)
      if (outputIsAudio) {
        setContent(null)
        downloadWatchlistOutputBinary(output.id)
          .then((result) => {
            const blob = new Blob([result], { type: getOutputMimeType(output.format) })
            const nextUrl = URL.createObjectURL(blob)
            updateAudioObjectUrl(nextUrl)
          })
          .catch((err) => {
            console.error("Failed to fetch audio output content:", err)
            setError(err.message || "Failed to load content")
          })
          .finally(() => {
            setLoading(false)
          })
      } else {
        updateAudioObjectUrl(null)
        downloadWatchlistOutput(output.id)
          .then((result) => setContent(result))
          .catch((err) => {
            console.error("Failed to fetch output content:", err)
            setError(err.message || "Failed to load content")
          })
          .finally(() => {
            setLoading(false)
          })
      }
    } else {
      setContent(null)
      setError(null)
      updateAudioObjectUrl(null)
      setViewMode("rendered")
    }
  }, [open, output, outputIsAudio, updateAudioObjectUrl])

  useEffect(() => {
    return () => {
      if (audioObjectUrlRef.current) {
        URL.revokeObjectURL(audioObjectUrlRef.current)
      }
      audioObjectUrlRef.current = null
    }
  }, [])

  // Handle download
  const handleDownload = async () => {
    if (!output) return
    try {
      const mimeType = getOutputMimeType(output.format)
      const blob = outputIsAudio
        ? new Blob([await downloadWatchlistOutputBinary(output.id)], { type: mimeType })
        : new Blob([await downloadWatchlistOutput(output.id)], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${output.title || `output-${output.id}`}.${getOutputFileExtension(output)}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      message.success(t("watchlists:outputs.downloaded", "Output downloaded"))
    } catch (err) {
      console.error("Failed to download output:", err)
      message.error(t("watchlists:outputs.downloadError", "Failed to download output"))
    }
  }

  const sanitizedHtml = useMemo(() => {
    if (!content) return null
    return DOMPurify.sanitize(content, { USE_PROFILES: { html: true } })
  }, [content])

  const deliveryStatuses = useMemo(() => {
    return getOutputDeliveryStatuses(output?.metadata)
  }, [output?.metadata])

  const audioSummary = useMemo(() => {
    return getOutputAudioStatusSummary(output?.metadata)
  }, [output?.metadata])

  const templateName = useMemo(() => {
    return getOutputTemplateName(output?.metadata)
  }, [output?.metadata])

  const templateVersion = useMemo(() => {
    return getOutputTemplateVersion(output?.metadata)
  }, [output?.metadata])
  const artifactLabel = useMemo(() => {
    return getOutputArtifactLabel(output)
  }, [output])
  const hasReportEvidenceMetadata = useMemo(() => {
    const metadata = output?.metadata
    if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) {
      return false
    }
    return (
      "report_snapshot_path" in metadata ||
      "report_readiness" in metadata ||
      "report_schema_version" in metadata
    )
  }, [output?.metadata])

  // Open in new tab (for HTML)
  const handleOpenInNewTab = () => {
    if (!content || output?.format !== "html") return
    const safeHtml = sanitizedHtml || content
    const blob = new Blob([safeHtml], { type: "text/html" })
    const url = URL.createObjectURL(blob)
    window.open(url, "_blank")
    // Clean up after a delay
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const renderAudioArtifactGraph = () => {
    if (!audioSummary.requested) return null

    const renderArtifact = (artifact: AudioArtifactSummary, key: string) => (
      <div key={key}>
        <div className="font-medium text-text">{artifact.label}</div>
        {artifact.displayName ? <div>{artifact.displayName}</div> : null}
        {artifact.downloadUrl ? (
          <a
            href={artifact.downloadUrl}
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:underline"
            aria-label={t("watchlists:outputs.openAudioArtifactAria", "Open {{label}}", {
              label: artifact.label
            })}
          >
            {t("watchlists:outputs.openAudioArtifact", "Open")}
          </a>
        ) : null}
      </div>
    )

    return (
      <div className="space-y-2" data-testid="output-preview-audio-artifacts">
        <div className="flex flex-wrap items-center gap-2">
          <div className="text-sm font-medium text-text">
            {t("watchlists:outputs.audioArtifactsLabel", "Audio artifacts")}
          </div>
          <Tag color={audioSummary.statusColor}>{audioSummary.statusLabel}</Tag>
        </div>
        {audioSummary.fallbackReason && (
          <div className="text-xs text-warning">
            {t("watchlists:outputs.audioFallback", "Fallback: {{reason}}", {
              reason: audioSummary.fallbackReason
            })}
          </div>
        )}
        <div className="grid gap-2 text-xs text-text-muted sm:grid-cols-2">
          {audioSummary.scriptArtifact && (
            renderArtifact(audioSummary.scriptArtifact, "script")
          )}
          {audioSummary.speakerArtifacts.map((artifact, index) => (
            renderArtifact(artifact, `${artifact.speakerId || artifact.label}-${index}`)
          ))}
          {audioSummary.finalArtifact && (
            renderArtifact(audioSummary.finalArtifact, "final")
          )}
        </div>
      </div>
    )
  }

  return (
    <Drawer
      title={output?.title || t("watchlists:outputs.preview", "Output Preview")}
      placement="right"
      onClose={onClose}
      open={open}
      styles={{ wrapper: { width: 700 } }}
      extra={
        <div className="flex items-center gap-2">
          {output?.format === "html" && (
            <Tooltip title={t("watchlists:outputs.openInNewTab", "Open in new tab")}>
              <Button
                type="text"
                icon={<ExternalLink className="h-4 w-4" />}
                onClick={handleOpenInNewTab}
                disabled={!content}
              />
            </Tooltip>
          )}
          <Tooltip title={t("watchlists:outputs.chatAbout", "Chat about this")}>
            <Button
              type="text"
              icon={<MessageSquare className="h-4 w-4" />}
              onClick={handleChatAboutOutput}
              disabled={!content}
              data-testid="watchlists-output-chat-about"
            />
          </Tooltip>
          <Tooltip title={t("watchlists:outputs.download", "Download")}>
            <Button
              type="text"
              icon={<Download className="h-4 w-4" />}
              onClick={handleDownload}
            />
          </Tooltip>
        </div>
      }
    >
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : error ? (
        <div className="text-center py-12 text-danger">{error}</div>
      ) : outputIsAudio ? (
        <div className="space-y-4">
          {output && (
            <div className="rounded-lg border border-border p-3 space-y-2 bg-surface">
              {(templateName || templateVersion) && (
                <div className="text-sm text-text">
                  <span className="font-medium">
                    {t("watchlists:outputs.templateLabel", "Template")}:
                  </span>{" "}
                  {templateName || t("watchlists:outputs.templateUnknown", "Unknown")}
                  {templateVersion ? ` v${templateVersion}` : ""}
                </div>
              )}
              {output && (
                <div className="space-y-1" data-testid="output-preview-provenance">
                  <div className="text-sm font-medium text-text">
                    {t("watchlists:outputs.provenanceLabel", "Provenance")}
                  </div>
                  <div className="text-xs text-text-muted">
                    {t(
                      "watchlists:outputs.provenanceDescription",
                      "Monitor #{{job}} • Run #{{run}} • Artifact: {{artifact}}",
                      {
                        job: output.job_id,
                        run: output.run_id,
                        artifact: artifactLabel
                      }
                    )}
                  </div>
                </div>
              )}
              {deliveryStatuses.length > 0 && (
                <div className="space-y-1">
                  <div className="text-sm font-medium text-text">
                    {t("watchlists:outputs.deliveryStatusLabel", "Delivery status")}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {deliveryStatuses.map((delivery, index) => (
                      <Tooltip
                        key={`${delivery.channel}-${delivery.status}-${index}`}
                        title={delivery.detail}
                      >
                        <Tag color={getDeliveryStatusColor(delivery.status)}>
                          {delivery.channel}: {delivery.status}
                        </Tag>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              )}
              {renderAudioArtifactGraph()}
              {output?.chatbook_path && (
                <div className="text-xs text-text-muted">
                  Chatbook: {output.chatbook_path}
                </div>
              )}
              {output?.storage_path && (
                <div className="text-xs text-text-muted">
                  {t("watchlists:outputs.storagePath", "Stored file")}: {output.storage_path}
                </div>
              )}
            </div>
          )}
          {output && hasReportEvidenceMetadata && (
            <div className="rounded-lg border border-border bg-surface p-3">
              <ReportEvidencePanel
                outputId={output.id}
                compact
              />
            </div>
          )}

          {audioObjectUrl ? (
            <div className="rounded-lg border border-border bg-surface p-4">
              <div className="mb-2 text-sm text-text-muted">
                {t("watchlists:outputs.audioPlayerLabel", "Audio playback")}
              </div>
              <audio controls preload="metadata" className="w-full" src={audioObjectUrl}>
                {t(
                  "watchlists:outputs.audioPlayerUnsupported",
                  "Your browser does not support audio playback."
                )}
              </audio>
            </div>
          ) : (
            <Empty
              description={t("watchlists:outputs.noContent", "No content available")}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          )}
        </div>
      ) : content ? (
        <div className="space-y-4">
          {output && (
            <div className="rounded-lg border border-border p-3 space-y-2 bg-surface">
              {(templateName || templateVersion) && (
                <div className="text-sm text-text">
                  <span className="font-medium">
                    {t("watchlists:outputs.templateLabel", "Template")}:
                  </span>{" "}
                  {templateName || t("watchlists:outputs.templateUnknown", "Unknown")}
                  {templateVersion ? ` v${templateVersion}` : ""}
                </div>
              )}
              {output && (
                <div className="space-y-1" data-testid="output-preview-provenance">
                  <div className="text-sm font-medium text-text">
                    {t("watchlists:outputs.provenanceLabel", "Provenance")}
                  </div>
                  <div className="text-xs text-text-muted">
                    {t(
                      "watchlists:outputs.provenanceDescription",
                      "Monitor #{{job}} • Run #{{run}} • Artifact: {{artifact}}",
                      {
                        job: output.job_id,
                        run: output.run_id,
                        artifact: artifactLabel
                      }
                    )}
                  </div>
                </div>
              )}
              {deliveryStatuses.length > 0 && (
                <div className="space-y-1">
                  <div className="text-sm font-medium text-text">
                    {t("watchlists:outputs.deliveryStatusLabel", "Delivery status")}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {deliveryStatuses.map((delivery, index) => (
                      <Tooltip
                        key={`${delivery.channel}-${delivery.status}-${index}`}
                        title={delivery.detail}
                      >
                        <Tag color={getDeliveryStatusColor(delivery.status)}>
                          {delivery.channel}: {delivery.status}
                        </Tag>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              )}
              {output?.chatbook_path && (
                <div className="text-xs text-text-muted">
                  Chatbook: {output.chatbook_path}
                </div>
              )}
            </div>
          )}
          {output && hasReportEvidenceMetadata && (
            <div className="rounded-lg border border-border bg-surface p-3">
              <ReportEvidencePanel
                outputId={output.id}
                compact
              />
            </div>
          )}

          {/* View mode toggle for HTML */}
          {output?.format === "html" && (
            <div className="flex justify-end">
              <Segmented
                size="small"
                options={[
                  { value: "rendered", label: t("watchlists:outputs.rendered", "Rendered") },
                  { value: "source", label: t("watchlists:outputs.source", "Source") }
                ]}
                value={viewMode}
                onChange={(v) => setViewMode(v as "rendered" | "source")}
              />
            </div>
          )}

          {/* Content display */}
          {output?.format === "html" && viewMode === "rendered" ? (
            <div
              className="prose dark:prose-invert max-w-none p-4 bg-surface rounded-lg border border-border overflow-auto max-h-[calc(100vh-200px)]"
              dangerouslySetInnerHTML={{ __html: sanitizedHtml || "" }}
            />
          ) : output?.format === "html" && viewMode === "source" ? (
            <pre className="p-4 bg-bg text-text rounded-lg font-mono text-xs overflow-auto max-h-[calc(100vh-200px)] whitespace-pre-wrap border border-border">
              {content}
            </pre>
          ) : (
            // Markdown or other formats
            <pre className="p-4 bg-surface rounded-lg font-mono text-sm overflow-auto max-h-[calc(100vh-200px)] whitespace-pre-wrap border border-border">
              {content}
            </pre>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          {output && hasReportEvidenceMetadata && (
            <div className="rounded-lg border border-border bg-surface p-3">
              <ReportEvidencePanel
                outputId={output.id}
                compact
              />
            </div>
          )}
          <Empty
            description={t("watchlists:outputs.noContent", "No content available")}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        </div>
      )}
    </Drawer>
  )
}
