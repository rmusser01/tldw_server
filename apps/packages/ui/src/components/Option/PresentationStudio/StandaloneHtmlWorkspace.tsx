import React from "react"
import { useNavigate } from "react-router-dom"

import { Button } from "@/components/Common/Button"
import { usePresentationPrincipalScope, type PresentationPrincipalBoundaryKind } from "@/hooks/usePresentationPrincipalScope"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useSlidesCapabilities } from "@/hooks/useSlidesCapabilities"
import {
  tldwClient,
  type PresentationDetailResult,
  type StandalonePresentationDetailResult
} from "@/services/tldw/TldwApiClient"
import { StandaloneHtmlDownloadManager } from "./standalone-html-download"
import {
  StandaloneHtmlOutlineController
} from "./standalone-html-outline-client"
import type { StandaloneHtmlOutline } from "./standalone-html-outline.worker"
import {
  clearStandaloneHtmlRecovery,
  readStandaloneHtmlRecovery,
  writeStandaloneHtmlRecovery,
  type PresentationPrincipalScope,
  type StandaloneHtmlRecoveryRecord
} from "./standalone-html-recovery"
import {
  validateStandaloneHtmlSource,
  type AcceptedStandaloneHtmlSource
} from "./standalone-html-source"
import { StandaloneHtmlSafeOutline } from "./StandaloneHtmlSafeOutline"
import {
  StandaloneHtmlSourceEditor,
  type StandaloneHtmlSourceEditorHandle
} from "./StandaloneHtmlSourceEditor"

type SaveStatus = "Saved" | "Saving" | "Not saved" | "Conflict"
type LoadStatus = "loading" | "ready" | "error"

type LoadedStandalone = {
  title: string
  acceptedSource: AcceptedStandaloneHtmlSource
  etag: string
}

type AvailableRecovery = {
  record: StandaloneHtmlRecoveryRecord
  acceptedSource: AcceptedStandaloneHtmlSource
}

const isStrongEtag = (value: unknown): value is string =>
  typeof value === "string" && value.length >= 2 && value.startsWith('"') && value.endsWith('"')

const errorStatus = (error: unknown): number | null => {
  const status = error && typeof error === "object" ? (error as { status?: unknown }).status : null
  return typeof status === "number" && Number.isFinite(status) ? status : null
}

const validateStandaloneDetail = async (
  result: PresentationDetailResult | StandalonePresentationDetailResult,
  presentationId: string,
  scope: PresentationPrincipalScope
): Promise<LoadedStandalone> => {
  const record = result.record
  const etag = result.etag
  if (
    record.content_kind !== "standalone_html" ||
    record.id !== presentationId ||
    String(record.client_id ?? "") !== scope.principalId ||
    !isStrongEtag(etag)
  ) {
    throw new Error("Presentation detail could not be verified")
  }
  const acceptedSource = await validateStandaloneHtmlSource(record.html_document)
  if (
    !acceptedSource.ok ||
    acceptedSource.digest !== record.html_sha256 ||
    acceptedSource.byteLength !== record.html_bytes
  ) {
    throw new Error("Presentation source could not be verified")
  }
  return { title: record.title, acceptedSource, etag }
}

export const StandaloneHtmlWorkspace: React.FC<{ presentationId: string }> = ({
  presentationId
}) => {
  const navigate = useNavigate()
  const online = useServerOnline()
  const slides = useSlidesCapabilities()
  const mountedRef = React.useRef(true)
  const scopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  const lastTrustedScopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  const acceptedRef = React.useRef<AcceptedStandaloneHtmlSource | null>(null)
  const baseRef = React.useRef<LoadedStandalone | null>(null)
  const loadControllerRef = React.useRef<AbortController | null>(null)
  const saveControllerRef = React.useRef<AbortController | null>(null)
  const sourceEditorRef = React.useRef<StandaloneHtmlSourceEditorHandle | null>(null)
  const outlineControllerRef = React.useRef<StandaloneHtmlOutlineController | null>(null)
  const downloadManagerRef = React.useRef<StandaloneHtmlDownloadManager | null>(null)

  const [loadStatus, setLoadStatus] = React.useState<LoadStatus>("loading")
  const [title, setTitle] = React.useState("Standalone HTML presentation")
  const [accepted, setAccepted] = React.useState<AcceptedStandaloneHtmlSource | null>(null)
  const [saveStatus, setSaveStatus] = React.useState<SaveStatus>("Saved")
  const [message, setMessage] = React.useState<string | null>(null)
  const [recoveryWarning, setRecoveryWarning] = React.useState<string | null>(null)
  const [recovery, setRecovery] = React.useState<AvailableRecovery | null>(null)
  const [outline, setOutline] = React.useState<StandaloneHtmlOutline | null>(null)
  const [outlineStatus, setOutlineStatus] = React.useState<"current" | "stale" | "failed">("stale")
  const [activeTab, setActiveTab] = React.useState<"code" | "outline">("code")
  const [confirmRecoveryDiscard, setConfirmRecoveryDiscard] = React.useState(false)
  const [confirmLeave, setConfirmLeave] = React.useState(false)
  const [confirmOverwrite, setConfirmOverwrite] = React.useState(false)
  const [confirmServerDiscard, setConfirmServerDiscard] = React.useState(false)
  const codeTabId = React.useId()
  const outlineTabId = React.useId()
  const codePanelId = React.useId()
  const outlinePanelId = React.useId()

  const dirty = Boolean(accepted && baseRef.current && accepted.digest !== baseRef.current.acceptedSource.digest)
  const dirtyRef = React.useRef(dirty)
  dirtyRef.current = dirty

  const disposeSensitive = React.useCallback(
    (kind: PresentationPrincipalBoundaryKind = "reauthenticate") => {
      sourceEditorRef.current?.dispose()
      loadControllerRef.current?.abort()
      loadControllerRef.current = null
      saveControllerRef.current?.abort()
      saveControllerRef.current = null
      outlineControllerRef.current?.dispose()
      outlineControllerRef.current = null
      downloadManagerRef.current?.dispose()
      downloadManagerRef.current = null
      if (kind === "logout" || kind === "switch" || kind === "mismatch") {
        const previousScope = lastTrustedScopeRef.current
        if (previousScope) {
          clearStandaloneHtmlRecovery(sessionStorage, previousScope, presentationId)
        }
        lastTrustedScopeRef.current = null
      }
      acceptedRef.current = null
      baseRef.current = null
      scopeRef.current = null
      setAccepted(null)
      setOutline(null)
      setOutlineStatus("stale")
      setRecovery(null)
      setMessage(null)
      setLoadStatus("loading")
    },
    [presentationId]
  )

  const principal = usePresentationPrincipalScope({ onBoundary: disposeSensitive })

  const ensureOutlineController = React.useCallback(() => {
    if (!outlineControllerRef.current) {
      outlineControllerRef.current = new StandaloneHtmlOutlineController({
        onState: (state) => {
          if (mountedRef.current) setOutlineStatus(state.status)
        },
        onOutline: (nextOutline) => {
          if (mountedRef.current) setOutline(nextOutline)
        }
      })
    }
    return outlineControllerRef.current
  }, [])

  const publishAccepted = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      acceptedRef.current = next
      setAccepted(next)
      ensureOutlineController().request({ source: next.source, digest: next.digest })
    },
    [ensureOutlineController]
  )

  const ensureDownloadManager = React.useCallback(() => {
    if (!downloadManagerRef.current) {
      downloadManagerRef.current = new StandaloneHtmlDownloadManager()
    }
    return downloadManagerRef.current
  }, [])

  const adoptServer = React.useCallback(
    (loaded: LoadedStandalone) => {
      baseRef.current = loaded
      publishAccepted(loaded.acceptedSource)
      setTitle(loaded.title)
      setSaveStatus("Saved")
      setMessage(null)
      setConfirmOverwrite(false)
      setConfirmServerDiscard(false)
      const currentScope = scopeRef.current
      if (currentScope) {
        clearStandaloneHtmlRecovery(sessionStorage, currentScope, presentationId)
      }
      setRecovery(null)
    },
    [presentationId, publishAccepted]
  )

  React.useEffect(() => {
    if (!online || principal.status !== "ready" || !principal.scope) return
    const scope = principal.scope
    scopeRef.current = scope
    lastTrustedScopeRef.current = scope
    const controller = new AbortController()
    loadControllerRef.current?.abort()
    loadControllerRef.current = controller
    let cancelled = false
    setLoadStatus("loading")
    setMessage(null)

    void (async () => {
      let detail: PresentationDetailResult | StandalonePresentationDetailResult | null = null
      try {
        detail = await tldwClient.getPresentation(presentationId, {
          abortSignal: controller.signal
        })
        if (
          cancelled ||
          controller.signal.aborted ||
          scopeRef.current?.principalScope !== scope.principalScope
        ) {
          return
        }
        const loaded = await validateStandaloneDetail(detail, presentationId, scope)
        detail = null
        if (
          cancelled ||
          controller.signal.aborted ||
          scopeRef.current?.principalScope !== scope.principalScope
        ) {
          return
        }
        const recovered = await readStandaloneHtmlRecovery(
          sessionStorage,
          scope,
          presentationId
        )
        if (cancelled || controller.signal.aborted) return
        baseRef.current = loaded
        publishAccepted(loaded.acceptedSource)
        setTitle(loaded.title)
        setSaveStatus("Saved")
        if (recovered.kind === "available") {
          if (recovered.acceptedSource.digest === loaded.acceptedSource.digest) {
            clearStandaloneHtmlRecovery(sessionStorage, scope, presentationId)
            setRecovery(null)
          } else {
            setRecovery(recovered)
          }
        } else {
          setRecovery(null)
        }
        setLoadStatus("ready")
      } catch {
        if (cancelled || controller.signal.aborted) return
        acceptedRef.current = null
        baseRef.current = null
        setAccepted(null)
        setLoadStatus("error")
        setMessage("This standalone HTML presentation could not be loaded safely.")
      } finally {
        detail = null
        if (loadControllerRef.current === controller) loadControllerRef.current = null
      }
    })()

    return () => {
      cancelled = true
      controller.abort()
      if (loadControllerRef.current === controller) loadControllerRef.current = null
    }
  }, [online, presentationId, principal.scope, principal.status, publishAccepted])

  const persistCurrentDraft = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      const scope = scopeRef.current
      const base = baseRef.current
      if (!scope || !base) return
      if (next.digest === base.acceptedSource.digest) {
        clearStandaloneHtmlRecovery(sessionStorage, scope, presentationId)
        setRecovery(null)
        return
      }
      const result = writeStandaloneHtmlRecovery(sessionStorage, scope, {
        presentationId,
        baseEtag: base.etag,
        baseDigest: base.acceptedSource.digest,
        acceptedSource: next,
        updatedAt: Date.now()
      })
      if (result.ok === false) setRecoveryWarning(result.message)
    },
    [presentationId]
  )

  const handleAcceptedChange = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      publishAccepted(next)
      persistCurrentDraft(next)
      const base = baseRef.current
      setSaveStatus(base && next.digest === base.acceptedSource.digest ? "Saved" : "Not saved")
      setMessage(null)
    },
    [persistCurrentDraft, publishAccepted]
  )

  const performSave = React.useCallback(
    async (ifMatch: string, reconcileAmbiguous = true) => {
      const local = acceptedRef.current
      const scope = scopeRef.current
      if (!local || !scope || !isStrongEtag(ifMatch)) return
      saveControllerRef.current?.abort()
      const controller = new AbortController()
      saveControllerRef.current = controller
      setSaveStatus("Saving")
      setMessage(null)
      try {
        let response: StandalonePresentationDetailResult | null = await tldwClient.saveStandaloneHtmlSource(
          presentationId,
          local.source,
          { ifMatch, abortSignal: controller.signal }
        )
        const loaded = await validateStandaloneDetail(response, presentationId, scope)
        response = null
        if (loaded.acceptedSource.digest !== local.digest) {
          throw new Error("Saved source could not be confirmed")
        }
        if (!controller.signal.aborted && acceptedRef.current?.digest === local.digest) {
          adoptServer(loaded)
        }
      } catch (error) {
        if (controller.signal.aborted) return
        if (errorStatus(error) === 412) {
          setSaveStatus("Conflict")
          setMessage("The server version changed. Choose how to continue.")
          return
        }
        if (reconcileAmbiguous) {
          try {
            let detail: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
              abortSignal: controller.signal
            })
            const loaded = await validateStandaloneDetail(detail, presentationId, scope)
            detail = null
            if (
              loaded.acceptedSource.digest === local.digest &&
              acceptedRef.current?.digest === local.digest
            ) {
              adoptServer(loaded)
              return
            }
          } catch {
            // The local source remains authoritative until an exact server digest is observed.
          }
        }
        setSaveStatus("Not saved")
        setMessage("Save could not be confirmed. Your local draft is preserved.")
      } finally {
        if (saveControllerRef.current === controller) saveControllerRef.current = null
      }
    },
    [adoptServer, presentationId]
  )

  const handleSave = React.useCallback(() => {
    const base = baseRef.current
    if (base) void performSave(base.etag)
  }, [performSave])

  const handleOverwrite = React.useCallback(async () => {
    const scope = scopeRef.current
    if (!scope) return
    setConfirmOverwrite(false)
    saveControllerRef.current?.abort()
    const controller = new AbortController()
    saveControllerRef.current = controller
    try {
      let fresh: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
        abortSignal: controller.signal
      })
      const loaded = await validateStandaloneDetail(fresh, presentationId, scope)
      fresh = null
      if (
        controller.signal.aborted ||
        scopeRef.current?.principalScope !== scope.principalScope
      ) {
        return
      }
      await performSave(loaded.etag, false)
    } catch {
      if (controller.signal.aborted) return
      setSaveStatus("Conflict")
      setMessage("The current server version could not be verified. Your draft is preserved.")
    } finally {
      if (saveControllerRef.current === controller) saveControllerRef.current = null
    }
  }, [performSave, presentationId])

  const handleDiscardAndLoad = React.useCallback(async () => {
    const scope = scopeRef.current
    if (!scope) return
    setConfirmServerDiscard(false)
    loadControllerRef.current?.abort()
    const controller = new AbortController()
    loadControllerRef.current = controller
    try {
      let fresh: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
        abortSignal: controller.signal
      })
      const loaded = await validateStandaloneDetail(fresh, presentationId, scope)
      fresh = null
      if (
        controller.signal.aborted ||
        scopeRef.current?.principalScope !== scope.principalScope
      ) {
        return
      }
      adoptServer(loaded)
    } catch {
      if (controller.signal.aborted) return
      setSaveStatus("Conflict")
      setMessage("The server version could not be loaded. Your draft is preserved.")
    } finally {
      if (loadControllerRef.current === controller) loadControllerRef.current = null
    }
  }, [adoptServer, presentationId])

  const downloadSource = React.useCallback(
    async (source: string) => {
      try {
        await ensureDownloadManager().download({ presentationId, source })
      } catch {
        setMessage("Download could not be prepared. Your draft is preserved.")
      }
    },
    [ensureDownloadManager, presentationId]
  )

  const discardRecovery = React.useCallback(() => {
    const scope = scopeRef.current
    if (scope) clearStandaloneHtmlRecovery(sessionStorage, scope, presentationId)
    setRecovery(null)
    setConfirmRecoveryDiscard(false)
  }, [presentationId])

  React.useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirtyRef.current) return
      event.preventDefault()
      event.returnValue = ""
    }
    const pagehide = () => {
      const local = acceptedRef.current
      const base = baseRef.current
      const scope = scopeRef.current
      if (local && base && scope && local.digest !== base.acceptedSource.digest) {
        writeStandaloneHtmlRecovery(sessionStorage, scope, {
          presentationId,
          baseEtag: base.etag,
          baseDigest: base.acceptedSource.digest,
          acceptedSource: local,
          updatedAt: Date.now()
        })
      }
      disposeSensitive("reauthenticate")
    }
    window.addEventListener("beforeunload", beforeUnload)
    window.addEventListener("pagehide", pagehide)
    return () => {
      window.removeEventListener("beforeunload", beforeUnload)
      window.removeEventListener("pagehide", pagehide)
    }
  }, [disposeSensitive, presentationId])

  React.useEffect(
    () => () => {
      mountedRef.current = false
      sourceEditorRef.current?.dispose()
      loadControllerRef.current?.abort()
      saveControllerRef.current?.abort()
      outlineControllerRef.current?.dispose()
      downloadManagerRef.current?.dispose()
      acceptedRef.current = null
      baseRef.current = null
      scopeRef.current = null
      lastTrustedScopeRef.current = null
    },
    []
  )

  if (!online) {
    return (
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-text-muted">Server is offline. Your in-memory draft has not been sent.</p>
      </section>
    )
  }

  if (principal.status === "guarded") {
    return (
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-danger">Current server and account could not be confirmed.</p>
        <Button size="lg" variant="secondary" onClick={() => void principal.retry()} className="mt-4">
          Retry
        </Button>
      </section>
    )
  }

  if (principal.status === "loading" || loadStatus === "loading") {
    return (
      <section className="rounded-xl border border-border bg-surface p-6" aria-live="polite">
        <p className="text-sm text-text-muted">Confirming current server and account…</p>
      </section>
    )
  }

  if (loadStatus === "error" || !accepted) {
    return (
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-danger">{message ?? "Presentation unavailable."}</p>
      </section>
    )
  }

  const canSave = slides.canEditStandalone && dirty && saveStatus !== "Saving" && Boolean(baseRef.current)
  const canDownload = slides.canDraftStandalone

  return (
    <section className="space-y-4">
      <header className="rounded-xl border border-border bg-surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">Standalone HTML</p>
            <h1 className="mt-1 text-2xl font-semibold text-text">{title}</h1>
            <p className="mt-2 max-w-2xl text-sm text-text-muted">
              Studio never runs this code. Downloading and opening the file leaves tldw&apos;s security boundary.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="lg" variant="ghost" onClick={() => {
              if (dirty) setConfirmLeave(true)
              else navigate("/presentation-studio")
            }}>
              Back to presentations
            </Button>
            <Button
              size="lg"
              variant="secondary"
              disabled={!canDownload}
              onClick={() => void downloadSource(accepted.source)}
            >
              Download current draft
            </Button>
            <Button size="lg" variant="primary" disabled={!canSave} onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
        <div
          data-testid="standalone-html-save-status"
          role="status"
          aria-live="polite"
          className="mt-3 text-sm font-medium text-text"
        >
          {saveStatus}
        </div>
        {!slides.canEditStandalone ? (
          <p className="mt-2 text-sm text-warning">Saving is unavailable</p>
        ) : null}
        {message ? <p role="alert" className="mt-2 text-sm text-danger">{message}</p> : null}
        {recoveryWarning ? <p role="alert" className="mt-2 text-sm text-warning">{recoveryWarning}</p> : null}
      </header>

      {recovery ? (
        <section aria-label="Recovered draft" role="region" className="rounded-xl border border-warning/40 bg-warning/10 p-4">
          <h2 className="font-semibold text-text">Recovered draft</h2>
          <p className="mt-1 text-sm text-text-muted">A different draft was saved in this tab. It has not been applied.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="lg" variant="secondary" onClick={() => handleAcceptedChange(recovery.acceptedSource)}>
              Restore recovered draft
            </Button>
            <Button size="lg" variant="secondary" onClick={() => void downloadSource(recovery.acceptedSource.source)}>
              Download recovered draft
            </Button>
            <Button size="lg" variant="danger" onClick={() => setConfirmRecoveryDiscard(true)}>
              Discard recovered draft
            </Button>
          </div>
          {confirmRecoveryDiscard ? (
            <div className="mt-3 rounded-lg border border-danger/30 bg-surface p-3">
              <p className="text-sm text-text">Confirm discarding the recovered draft. This cannot be undone.</p>
              <Button size="lg" variant="danger" className="mt-2" onClick={discardRecovery}>
                Confirm discard recovered draft
              </Button>
            </div>
          ) : null}
        </section>
      ) : null}

      {saveStatus === "Conflict" ? (
        <section className="rounded-xl border border-warning/40 bg-warning/10 p-4" aria-label="Save conflict">
          <h2 className="font-semibold text-text">Conflict</h2>
          <p className="mt-1 text-sm text-text-muted">Your draft is unchanged. Choose an explicit next step.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button size="lg" variant="danger" onClick={() => setConfirmServerDiscard(true)}>
              Discard my changes and load server version
            </Button>
            <Button size="lg" variant="primary" onClick={() => setConfirmOverwrite(true)}>
              Overwrite server with my draft
            </Button>
            <Button size="lg" variant="secondary" onClick={() => void downloadSource(accepted.source)}>
              Download my draft
            </Button>
          </div>
          {confirmOverwrite ? (
            <div className="mt-3 rounded-lg border border-warning/40 bg-surface p-3">
              <p className="text-sm text-text">Confirm replacing the current server version with your local draft.</p>
              <Button size="lg" variant="danger" className="mt-2" onClick={() => void handleOverwrite()}>
                Confirm overwrite
              </Button>
            </div>
          ) : null}
          {confirmServerDiscard ? (
            <div className="mt-3 rounded-lg border border-danger/30 bg-surface p-3">
              <p className="text-sm text-text">Confirm discarding your local changes and loading the server version.</p>
              <Button size="lg" variant="danger" className="mt-2" onClick={() => void handleDiscardAndLoad()}>
                Confirm discard and load server version
              </Button>
            </div>
          ) : null}
        </section>
      ) : null}

      {confirmLeave ? (
        <section className="rounded-xl border border-danger/30 bg-surface p-4">
          <h2 className="font-semibold text-text">Leave without saving?</h2>
          <p className="mt-1 text-sm text-text-muted">Your in-memory draft will close. Scoped recovery may still be available in this tab.</p>
          <div className="mt-3 flex gap-2">
            <Button size="lg" variant="danger" onClick={() => navigate("/presentation-studio")}>
              Leave presentation
            </Button>
            <Button size="lg" variant="secondary" onClick={() => setConfirmLeave(false)}>
              Keep editing
            </Button>
          </div>
        </section>
      ) : null}

      <div role="tablist" aria-label="Standalone HTML workspace views" className="flex gap-2 md:hidden">
        <button
          id={codeTabId}
          type="button"
          role="tab"
          aria-controls={codePanelId}
          aria-selected={activeTab === "code"}
          onClick={() => setActiveTab("code")}
          className="min-h-[44px] rounded-md px-4 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          Code
        </button>
        <button
          id={outlineTabId}
          type="button"
          role="tab"
          aria-controls={outlinePanelId}
          aria-selected={activeTab === "outline"}
          onClick={() => setActiveTab("outline")}
          className="min-h-[44px] rounded-md px-4 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          Outline
        </button>
      </div>

      <div className="grid min-w-0 gap-4 md:grid-cols-2">
        <section
          id={codePanelId}
          role="tabpanel"
          aria-labelledby={codeTabId}
          className={`${activeTab === "code" ? "block" : "hidden"} min-w-0 rounded-xl border border-border bg-surface p-4 md:block`}
        >
          <StandaloneHtmlSourceEditor
            ref={sourceEditorRef}
            value={accepted.source}
            onAcceptedChange={handleAcceptedChange}
          />
        </section>
        <section
          id={outlinePanelId}
          role="tabpanel"
          aria-labelledby={outlineTabId}
          className={`${activeTab === "outline" ? "block" : "hidden"} min-w-0 rounded-xl border border-border bg-surface p-4 md:block`}
        >
          <StandaloneHtmlSafeOutline status={outlineStatus} outline={outline} />
        </section>
      </div>
    </section>
  )
}

export default StandaloneHtmlWorkspace
