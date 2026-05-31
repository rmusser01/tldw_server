import React from "react"
import { Button, Modal, Skeleton } from "antd"
import { Alert } from "@/components/ui/primitives/Alert"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { formatRelativeTimestamp } from "./listUtils"

type DictionaryVersionHistoryModalProps = {
  open: boolean
  dictionary: any | null
  onClose: () => void
  onReverted?: () => void
}

type LoadingState = "idle" | "loading" | "success" | "error"

export function DictionaryVersionHistoryModal({
  open,
  dictionary,
  onClose,
  onReverted,
}: DictionaryVersionHistoryModalProps) {
  const dictionaryId = Number(dictionary?.id)
  const [versionsStatus, setVersionsStatus] = React.useState<LoadingState>("idle")
  const [detailStatus, setDetailStatus] = React.useState<LoadingState>("idle")
  const [versions, setVersions] = React.useState<any[]>([])
  const [selectedRevision, setSelectedRevision] = React.useState<number | null>(null)
  const [selectedDetail, setSelectedDetail] = React.useState<any | null>(null)
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null)
  const [revertMessage, setRevertMessage] = React.useState<string | null>(null)
  const [isReverting, setIsReverting] = React.useState(false)

  const loadVersions = React.useCallback(async () => {
    if (!Number.isFinite(dictionaryId) || dictionaryId <= 0) return
    setVersionsStatus("loading")
    setErrorMessage(null)
    try {
      const response = await tldwClient.dictionaryVersions(dictionaryId, {
        limit: 30,
        offset: 0,
      })
      const nextVersions = Array.isArray(response?.versions) ? response.versions : []
      setVersions(nextVersions)
      const fallbackRevision = Number(nextVersions[0]?.revision)
      setSelectedRevision((current) =>
        Number.isFinite(current) && current != null ? current : Number.isFinite(fallbackRevision) ? fallbackRevision : null
      )
      setVersionsStatus("success")
    } catch (error: any) {
      setVersionsStatus("error")
      setErrorMessage(error?.message || "Could not load dictionary versions.")
      setVersions([])
      setSelectedRevision(null)
    }
  }, [dictionaryId])

  React.useEffect(() => {
    if (!open) return
    void loadVersions()
  }, [loadVersions, open])

  React.useEffect(() => {
    if (!open || !Number.isFinite(dictionaryId) || dictionaryId <= 0 || selectedRevision == null) {
      setSelectedDetail(null)
      setDetailStatus("idle")
      return
    }

    let isCancelled = false
    const loadDetail = async () => {
      setDetailStatus("loading")
      try {
        const detail = await tldwClient.dictionaryVersionSnapshot(dictionaryId, selectedRevision)
        if (isCancelled) return
        setSelectedDetail(detail)
        setDetailStatus("success")
      } catch (error: any) {
        if (isCancelled) return
        setDetailStatus("error")
        setSelectedDetail(null)
        setErrorMessage(error?.message || "Could not load selected revision details.")
      }
    }

    void loadDetail()
    return () => {
      isCancelled = true
    }
  }, [dictionaryId, open, selectedRevision])

  const handleRevert = React.useCallback(async () => {
    if (!Number.isFinite(dictionaryId) || dictionaryId <= 0 || selectedRevision == null) return
    setIsReverting(true)
    setErrorMessage(null)
    setRevertMessage(null)
    try {
      const result = await tldwClient.revertDictionaryVersion(dictionaryId, selectedRevision)
      setRevertMessage(String(result?.message || `Reverted to revision ${selectedRevision}.`))
      onReverted?.()
      await loadVersions()
    } catch (error: any) {
      setErrorMessage(error?.message || "Failed to revert dictionary revision.")
    } finally {
      setIsReverting(false)
    }
  }, [dictionaryId, loadVersions, onReverted, selectedRevision])

  const footer = (
    <div className="flex items-center justify-between gap-2">
      <div className="text-xs text-text-muted">
        {selectedRevision != null ? `Selected revision: ${selectedRevision}` : "Select a revision"}
      </div>
      <div className="flex gap-2">
        <Button onClick={onClose}>Close</Button>
        <Button
          type="primary"
          danger
          disabled={selectedRevision == null}
          loading={isReverting}
          onClick={() => {
            void handleRevert()
          }}
        >
          {selectedRevision == null ? "Revert" : `Revert to revision ${selectedRevision}`}
        </Button>
      </div>
    </div>
  )

  return (
    <Modal
      title={`Dictionary Version History${dictionary?.name ? ` - ${dictionary.name}` : ""}`}
      open={open}
      onCancel={onClose}
      width={920}
      footer={footer}
    >
      {errorMessage ? (
        <Alert
          variant="error"
          className="mb-3"
          title="Version history failed"
        >
          {errorMessage}
        </Alert>
      ) : null}
      {revertMessage ? (
        <Alert
          variant="success"
          className="mb-3"
          title="Revision restored"
        >
          {revertMessage}
        </Alert>
      ) : null}

      {versionsStatus === "loading" ? (
        <Skeleton active paragraph={{ rows: 8 }} />
      ) : versions.length === 0 ? (
        <div className="rounded border border-border bg-surface2/40 p-3 text-sm text-text-muted">
          No version snapshots are available for this dictionary yet.
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-[260px_1fr]">
          <div className="max-h-[420px] overflow-y-auto rounded border border-border bg-surface2/20 p-2">
            {versions.map((version) => {
              const revision = Number(version?.revision)
              const isSelected = selectedRevision === revision
              return (
                <button
                  key={`dictionary-version-${revision}`}
                  type="button"
                  className={`mb-1 w-full rounded border px-2 py-2 text-left text-xs transition-colors ${
                    isSelected
                      ? "border-primary bg-primary/10 text-text"
                      : "border-border bg-surface hover:bg-surface2"
                  }`}
                  onClick={() => {
                    setSelectedRevision(revision)
                    setRevertMessage(null)
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono">r{revision}</span>
                    <span className="text-text-muted">
                      {formatRelativeTimestamp(version?.created_at)}
                    </span>
                  </div>
                  <div className="mt-1 text-text">{String(version?.change_type || "update")}</div>
                  <div className="text-text-muted">
                    {Number(version?.entry_count || 0)} entries
                    {version?.summary ? ` · ${String(version.summary)}` : ""}
                  </div>
                </button>
              )
            })}
          </div>

          <div className="rounded border border-border bg-surface2/20 p-3">
            {detailStatus === "loading" ? (
              <Skeleton active paragraph={{ rows: 6 }} />
            ) : selectedDetail ? (
              <div className="space-y-3 text-sm">
                <div className="grid gap-2 sm:grid-cols-2">
                  <div>
                    <div className="text-xs text-text-muted">Revision</div>
                    <div className="font-mono">r{selectedDetail?.revision}</div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted">Change Type</div>
                    <div>{String(selectedDetail?.change_type || "update")}</div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted">Captured</div>
                    <div>{formatRelativeTimestamp(selectedDetail?.created_at)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-text-muted">Entries</div>
                    <div>{Array.isArray(selectedDetail?.entries) ? selectedDetail.entries.length : 0}</div>
                  </div>
                </div>
                {selectedDetail?.summary ? (
                  <div className="rounded border border-border bg-surface p-2 text-xs">
                    {String(selectedDetail.summary)}
                  </div>
                ) : null}
                <div>
                  <div className="mb-1 text-xs text-text-muted">Dictionary Snapshot</div>
                  <div className="rounded border border-border bg-surface p-2 text-xs">
                    <div>Name: {String(selectedDetail?.dictionary?.name || "Untitled")}</div>
                    <div>
                      Active: {selectedDetail?.dictionary?.is_active ? "Yes" : "No"}
                    </div>
                    <div>
                      Category: {String(selectedDetail?.dictionary?.category || "None")}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-xs text-text-muted">Select a revision to view details.</div>
            )}
          </div>
        </div>
      )}
    </Modal>
  )
}
