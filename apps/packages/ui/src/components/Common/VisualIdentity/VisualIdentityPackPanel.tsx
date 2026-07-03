import React from "react"
import { Button, Tag } from "antd"
import { RefreshCw, Upload } from "lucide-react"

import { clearVisualIdentityResolverCaches } from "@/hooks/useVisualIdentityResolver"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  VisualIdentityActorKind,
  VisualIdentityAssetResponse,
  VisualIdentityCapabilitiesResponse,
  VisualIdentityDraftActivateRequest,
  VisualIdentityDraftResponse,
  VisualIdentityDraftSlotUpdate,
  VisualIdentityExpressionSlotResponse,
  VisualIdentityImportZipStartResponse,
  VisualIdentityPackResponse,
  VisualIdentityResolveResponse
} from "@/types/visual-identities"
import { VisualIdentityDraftReview } from "./VisualIdentityDraftReview"

const DRAFT_POLL_INTERVAL_MS = 1200
const DRAFT_POLL_ATTEMPTS = 60
const DRAFT_TERMINAL_STATUSES = new Set([
  "ready_for_review",
  "failed",
  "cancelled",
  "canceled",
  "quarantined",
  "activated",
  "abandoned"
])

type VisualIdentityApi = {
  getVisualIdentityCapabilities?: () => Promise<VisualIdentityCapabilitiesResponse>
  listVisualIdentityExpressionSlots?: () => Promise<VisualIdentityExpressionSlotResponse[]>
  listVisualIdentityPacks?: (params?: { status?: string | null }) => Promise<VisualIdentityPackResponse[]>
  resolveVisualIdentityBinding?: (request: {
    actor_kind: VisualIdentityActorKind
    actor_id: number | string
    expression_key?: string
  }) => Promise<VisualIdentityResolveResponse>
  startVisualIdentityZipImport?: (request: {
    archive: { name?: string; type?: string; data: ArrayBuffer | Uint8Array | number[] }
    title?: string
    pack_id?: number | null
    idempotency_key: string
  }) => Promise<VisualIdentityImportZipStartResponse>
  getVisualIdentityDraft?: (draftId: number) => Promise<VisualIdentityDraftResponse>
  activateVisualIdentityDraft?: (
    draftId: number,
    request: VisualIdentityDraftActivateRequest
  ) => Promise<VisualIdentityDraftResponse>
  uploadVisualIdentityPackAsset?: (
    packId: number,
    request: {
      expression_key: string
      draft_id?: number | null
      file: { name?: string; type?: string; data: ArrayBuffer | Uint8Array | number[] }
    }
  ) => Promise<VisualIdentityAssetResponse>
  updateVisualIdentityDraftSlot?: (
    draftId: number,
    slotKey: string,
    request: VisualIdentityDraftSlotUpdate
  ) => Promise<VisualIdentityDraftResponse>
  getVisualIdentityAssetContentPath?: (packId: number, assetId: number) => string
}

export type VisualIdentityPackPanelProps = {
  actorKind: VisualIdentityActorKind
  actorId: number | string
  actorName?: string
  className?: string
  client?: VisualIdentityApi
}

const makeIdempotencyKey = (): string => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `visual-identity-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const fileToUploadData = async (file: File) => ({
  name: file.name,
  type: file.type || "application/octet-stream",
  data: await file.arrayBuffer()
})

const wait = (delayMs: number) =>
  new Promise((resolve) => setTimeout(resolve, delayMs))

const hasVisualIdentityApi = (api: VisualIdentityApi): boolean =>
  typeof api.getVisualIdentityCapabilities === "function" &&
  typeof api.listVisualIdentityExpressionSlots === "function" &&
  typeof api.resolveVisualIdentityBinding === "function"

const isTerminalDraftStatus = (status: string): boolean =>
  DRAFT_TERMINAL_STATUSES.has(status)

export const VisualIdentityPackPanel: React.FC<VisualIdentityPackPanelProps> = ({
  actorKind,
  actorId,
  actorName,
  className = "",
  client = tldwClient
}) => {
  const archiveInputRef = React.useRef<HTMLInputElement | null>(null)
  const isMountedRef = React.useRef(true)
  const [capabilities, setCapabilities] =
    React.useState<VisualIdentityCapabilitiesResponse | null>(null)
  const [expressionSlots, setExpressionSlots] = React.useState<
    VisualIdentityExpressionSlotResponse[]
  >([])
  const [resolved, setResolved] = React.useState<VisualIdentityResolveResponse | null>(null)
  const [packs, setPacks] = React.useState<VisualIdentityPackResponse[]>([])
  const [draft, setDraft] = React.useState<VisualIdentityDraftResponse | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [importing, setImporting] = React.useState(false)
  const [activating, setActivating] = React.useState(false)
  const [uploadingSlotKey, setUploadingSlotKey] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [statusMessage, setStatusMessage] = React.useState<string | null>(null)

  const actorLabel = actorName || String(actorId)
  const activePack = React.useMemo(
    () => packs.find((pack) => pack.id === resolved?.pack_id) || null,
    [packs, resolved?.pack_id]
  )
  const apiReady = hasVisualIdentityApi(client)
  const canImportArchive =
    apiReady && typeof client.startVisualIdentityZipImport === "function"

  const buildAssetUrl = React.useCallback(
    (asset: VisualIdentityAssetResponse): string => {
      if (asset.pack_id == null) return ""
      if (typeof client.getVisualIdentityAssetContentPath !== "function") return ""
      return client.getVisualIdentityAssetContentPath(asset.pack_id, asset.id)
    },
    [client]
  )

  const loadPanel = React.useCallback(async () => {
    if (!apiReady) {
      setError("Visual Identity API is not available in this client build.")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const [nextCapabilities, nextSlots, nextResolved, nextPacks] =
        await Promise.all([
          client.getVisualIdentityCapabilities!(),
          client.listVisualIdentityExpressionSlots!(),
          client.resolveVisualIdentityBinding!({
            actor_kind: actorKind,
            actor_id: actorId,
            expression_key: "neutral"
          }),
          typeof client.listVisualIdentityPacks === "function"
            ? client.listVisualIdentityPacks({ status: "active" })
            : Promise.resolve([])
        ])
      setCapabilities(nextCapabilities)
      setExpressionSlots(nextSlots)
      setResolved(nextResolved)
      setPacks(nextPacks)
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load expression packs.")
    } finally {
      setLoading(false)
    }
  }, [actorId, actorKind, apiReady, client])

  React.useEffect(() => {
    isMountedRef.current = true
    void loadPanel()
    return () => {
      isMountedRef.current = false
    }
  }, [loadPanel])

  const refreshDraft = React.useCallback(
    async (draftId: number) => {
      if (typeof client.getVisualIdentityDraft !== "function") return
      setError(null)
      try {
        setDraft(await client.getVisualIdentityDraft(draftId))
      } catch (draftError) {
        setError(draftError instanceof Error ? draftError.message : "Failed to refresh draft.")
      }
    },
    [client]
  )

  const handleArchiveSelected = async (file: File) => {
    if (typeof client.startVisualIdentityZipImport !== "function") return
    setImporting(true)
    setError(null)
    try {
      const archive = await fileToUploadData(file)
      const started = await client.startVisualIdentityZipImport({
        archive,
        title: `${actorLabel} expression pack`,
        pack_id: activePack?.id ?? null,
        idempotency_key: makeIdempotencyKey()
      })
      setStatusMessage(`Import ${started.status}.`)
      if (started.draft_id && typeof client.getVisualIdentityDraft === "function") {
        let latestDraft = await client.getVisualIdentityDraft(started.draft_id)
        if (!isMountedRef.current) return
        setDraft(latestDraft)
        for (
          let attempt = 0;
          !isTerminalDraftStatus(latestDraft.status) &&
          attempt < DRAFT_POLL_ATTEMPTS &&
          isMountedRef.current;
          attempt += 1
        ) {
          await wait(DRAFT_POLL_INTERVAL_MS)
          latestDraft = await client.getVisualIdentityDraft(started.draft_id)
          if (!isMountedRef.current) return
          setDraft(latestDraft)
          if (isTerminalDraftStatus(latestDraft.status)) break
        }
        if (latestDraft.status === "importing" && isMountedRef.current) {
          setStatusMessage("Import is still processing. Refresh the draft to check again.")
        }
      }
    } catch (importError) {
      if (isMountedRef.current) {
        setError(importError instanceof Error ? importError.message : "Failed to import expression ZIP.")
      }
    } finally {
      if (isMountedRef.current) setImporting(false)
    }
  }

  const handleActivate = async (
    draftId: number,
    request: VisualIdentityDraftActivateRequest
  ) => {
    if (typeof client.activateVisualIdentityDraft !== "function") return
    setActivating(true)
    setError(null)
    try {
      const activated = await client.activateVisualIdentityDraft(draftId, request)
      clearVisualIdentityResolverCaches()
      setDraft(activated)
      setStatusMessage("Expression pack activated.")
      await loadPanel()
    } catch (activateError) {
      setError(
        activateError instanceof Error
          ? activateError.message
          : "Failed to activate expression pack."
      )
    } finally {
      setActivating(false)
    }
  }

  const handleUploadAsset = async (slotKey: string, file: File) => {
    if (!draft?.pack_id || typeof client.uploadVisualIdentityPackAsset !== "function") {
      setError("Save or import a pack draft before uploading expression assets.")
      return
    }
    if (typeof client.updateVisualIdentityDraftSlot !== "function") {
      setError("Draft slot update API is not available in this client build.")
      return
    }
    setUploadingSlotKey(slotKey)
    setError(null)
    try {
      const uploaded = await client.uploadVisualIdentityPackAsset(draft.pack_id, {
        expression_key: slotKey,
        draft_id: draft.id,
        file: await fileToUploadData(file)
      })
      const updated = await client.updateVisualIdentityDraftSlot(draft.id, slotKey, {
        asset_id: uploaded.id,
        expression_key: slotKey,
        display_label: uploaded.display_label || slotKey
      })
      clearVisualIdentityResolverCaches()
      setDraft(updated)
      setStatusMessage(`${slotKey} asset uploaded.`)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Failed to upload asset.")
    } finally {
      setUploadingSlotKey(null)
    }
  }

  const handleClearSlot = async (slotKey: string) => {
    if (!draft || typeof client.updateVisualIdentityDraftSlot !== "function") return
    setError(null)
    try {
      const updated = await client.updateVisualIdentityDraftSlot(draft.id, slotKey, {
        asset_id: null,
        expression_key: slotKey
      })
      clearVisualIdentityResolverCaches()
      setDraft(updated)
    } catch (clearError) {
      setError(clearError instanceof Error ? clearError.message : "Failed to clear expression slot.")
    }
  }

  return (
    <section
      data-testid="visual-identity-pack-panel"
      className={`rounded-lg border border-border bg-surface p-3 ${className}`}
      aria-label="Expression packs"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            Expression packs
          </div>
          <div className="mt-1 text-sm font-medium text-text">{actorLabel}</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
            <Tag color={resolved?.pack_id ? "green" : "default"}>
              {resolved?.pack_id ? "bound" : "no binding"}
            </Tag>
            {activePack ? <span>{activePack.title}</span> : null}
            {activePack?.active_version_id ? (
              <span>{`v${activePack.active_version_id}`}</span>
            ) : null}
            {activePack?.default_expression_key ? (
              <span>{`default ${activePack.default_expression_key}`}</span>
            ) : null}
            {resolved?.fallback_reason ? <span>{resolved.fallback_reason}</span> : null}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="small"
            icon={<RefreshCw className="h-3.5 w-3.5" />}
            disabled={loading}
            onClick={() => void loadPanel()}
          >
            {loading ? "Refreshing" : "Refresh"}
          </Button>
          <input
            ref={archiveInputRef}
            type="file"
            accept=".zip,application/zip"
            className="sr-only"
            aria-label="Import expression pack ZIP"
            disabled={!canImportArchive || importing}
            onChange={(event) => {
              const file = event.currentTarget.files?.[0]
              event.currentTarget.value = ""
              if (file) void handleArchiveSelected(file)
            }}
          />
          <Button
            size="small"
            type="primary"
            icon={<Upload className="h-3.5 w-3.5" />}
            loading={importing}
            disabled={!canImportArchive}
            onClick={() => archiveInputRef.current?.click()}
          >
            Import ZIP
          </Button>
        </div>
      </div>

      {capabilities ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-text-muted">
          <Tag>{Math.round(capabilities.upload_max_bytes / 1024 / 1024)} MB asset max</Tag>
          <Tag>{capabilities.supported_mime_types.join(", ")}</Tag>
          {capabilities.avif_enabled ? <Tag color="blue">AVIF</Tag> : null}
        </div>
      ) : null}

      {error ? (
        <div className="mt-3 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {error}
        </div>
      ) : null}
      {statusMessage ? (
        <div className="mt-3 rounded-md border border-border bg-bg px-3 py-2 text-xs text-text-muted">
          {statusMessage}
        </div>
      ) : null}

      {draft ? (
        <div className="mt-3">
          <VisualIdentityDraftReview
            actorKind={actorKind}
            actorId={actorId}
            draft={draft}
            expressionSlots={expressionSlots}
            activating={activating}
            uploadingSlotKey={uploadingSlotKey}
            embedded
            buildAssetUrl={buildAssetUrl}
            onActivate={(draftId, request) => void handleActivate(draftId, request)}
            onRefreshDraft={(draftId) => void refreshDraft(draftId)}
            onUploadAsset={(slotKey, file) => void handleUploadAsset(slotKey, file)}
            onClearSlot={(slotKey) => void handleClearSlot(slotKey)}
          />
        </div>
      ) : (
        <div className="mt-3 rounded-md border border-dashed border-border bg-bg px-3 py-3 text-xs text-text-muted">
          Import a SillyTavern-style expressions ZIP to create a review draft. Activating
          the draft binds it to this {actorKind} by default.
        </div>
      )}
    </section>
  )
}
