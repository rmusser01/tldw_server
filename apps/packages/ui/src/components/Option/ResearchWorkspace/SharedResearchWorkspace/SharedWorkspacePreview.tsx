import React from "react"
import { Drawer, Tooltip } from "antd"
import { ExternalLink, X } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { SharedSourcePreview } from "@/types/shared-workspace"
import type { SharedWorkspaceError } from "./shared-research-workspace-reducer"
import { SharedWorkspaceSafeMarkdown } from "./SharedWorkspaceSafeMarkdown"

type SharedWorkspacePreviewProps = {
  error: SharedWorkspaceError | null
  isMobile: boolean
  loading: boolean
  onClose: () => void
  open: boolean
  preview: SharedSourcePreview | null
}

export const SharedWorkspacePreview: React.FC<
  SharedWorkspacePreviewProps
> = ({ error, isMobile, loading, onClose, open, preview }) => {
  const { t } = useTranslation("playground")
  const removed =
    error?.code === "shared_workspace_not_found"

  return (
    <Drawer
      open={open}
      onClose={onClose}
      placement="right"
      closable={false}
      destroyOnHidden
      autoFocus
      aria-label={
        loading
          ? t("sharedWorkspace.loadingPreview", "Loading source preview")
          : t("sharedWorkspace.preview", "Source preview")
      }
      title={
        <div className="flex min-w-0 items-center justify-between gap-2">
          <span className="truncate text-sm font-semibold">
            {loading
              ? t("sharedWorkspace.loadingPreview", "Loading source preview")
              : t("sharedWorkspace.preview", "Source preview")}
          </span>
          <Tooltip
            title={t(
              "sharedWorkspace.closePreview",
              "Close source preview"
            )}
          >
            <button
              type="button"
              aria-label={t(
                "sharedWorkspace.closePreview",
                "Close source preview"
              )}
              onClick={onClose}
              className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md outline-none hover:bg-surface2 focus-visible:ring-2 focus-visible:ring-focus"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
          </Tooltip>
        </div>
      }
      styles={{
        wrapper: { width: isMobile ? "100%" : 520, maxWidth: "100%" },
        body: { padding: 0, overflow: "hidden" }
      }}
    >
      <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-surface">
        {removed ? (
          <p className="p-4 text-sm text-text-muted">
            {t(
              "sharedWorkspace.removedCitation",
              "This source is no longer shared."
            )}
          </p>
        ) : error ? (
          <p className="p-4 text-sm text-danger">{error.message}</p>
        ) : loading || !preview ? (
          <p className="p-4 text-sm text-text-muted" aria-live="polite">
            {t("sharedWorkspace.loadingPreview", "Loading source preview")}
          </p>
        ) : (
          <>
            <div className="shrink-0 border-b border-border px-4 py-3">
              <h2 className="text-base font-semibold">{preview.title}</h2>
              <div className="mt-1 flex min-w-0 items-center gap-2 text-xs text-text-muted">
                <span>{preview.source_type}</span>
                {preview.origin_url ? (
                  <a
                    href={preview.origin_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex min-w-0 items-center gap-1 text-primary outline-none focus-visible:ring-2 focus-visible:ring-focus"
                  >
                    <span className="truncate">
                      {preview.origin_host || preview.origin_url}
                    </span>
                    <ExternalLink
                      className="h-3 w-3 shrink-0"
                      aria-hidden="true"
                    />
                  </a>
                ) : null}
              </div>
            </div>
            <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
              {preview.text_preview ? (
                <SharedWorkspaceSafeMarkdown content={preview.text_preview} />
              ) : null}
              {preview.snippets.map((snippet, index) => (
                <section
                  key={snippet.kind + "-" + (snippet.chunk_index ?? index)}
                >
                  <h3 className="mb-1 text-xs font-semibold text-text-muted">
                    {snippet.chunk_index === null
                      ? t("sharedWorkspace.excerpt", "Excerpt")
                      : t("sharedWorkspace.chunk", "Chunk {{index}}", {
                          index: snippet.chunk_index
                        })}
                  </h3>
                  <SharedWorkspaceSafeMarkdown content={snippet.text} />
                </section>
              ))}
            </div>
          </>
        )}
      </div>
    </Drawer>
  )
}
