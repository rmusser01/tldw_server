import React from "react"
import { flushSync } from "react-dom"
import { useNavigate } from "react-router-dom"

import { ProjectWorkspace } from "./ProjectWorkspace"
import { PresentationStudioIndex } from "./PresentationStudioIndex"
import { StandaloneHtmlWorkspace } from "./StandaloneHtmlWorkspace"
import { VisualStylePicker } from "./VisualStylePicker"
import { VisualStyleManager } from "./VisualStyleManager"
import {
  buildPresentationVisualStyleSnapshot,
  tldwClient,
  type PresentationDetailResult,
  type VisualStyleRecord
} from "@/services/tldw/TldwApiClient"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useServerOnline } from "@/hooks/useServerOnline"
import { usePresentationStudioStore } from "@/store/presentation-studio"
import { isExtensionRuntime } from "@/utils/browser-runtime"

type PresentationStudioPageProps = {
  mode?: "index" | "new" | "detail"
  projectId?: string | null
  embedded?: boolean
}

const formatEtag = (version: number | null | undefined): string | null =>
  typeof version === "number" && Number.isFinite(version) ? `W/"v${version}"` : null

const toErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message || "Failed to load presentation." : "Failed to load presentation."

const createBlankSlideId = (): string =>
  globalThis.crypto?.randomUUID?.() ||
  `slide-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`

const DEFAULT_VISUAL_STYLE_ID = "minimal-academic"

type InFlightProjectRequest = {
  projectId: string
  authorityEpoch: number
  controller: AbortController
  promise: Promise<DetailLoadResult | null>
}

type DetailLoadResult =
  | { kind: "structured"; detail: PresentationDetailResult }
  | { kind: "standalone_html" }
  | { kind: "unsupported"; contentKind: string | null }
  | { kind: "metadata_unavailable" }

type DetailSurfaceState =
  ({ kind: "structured" } | Exclude<DetailLoadResult, { kind: "structured" }>) & {
    projectId: string
    authorityEpoch: number
  }

type BufferedDetailOutcome = {
  projectId: string
  authorityEpoch: number
  outcome:
    | { kind: "result"; result: DetailLoadResult }
    | { kind: "error"; message: string }
}

const detailOutcomeRequiresRelease = (buffered: BufferedDetailOutcome): boolean =>
  buffered.outcome.kind === "error" || buffered.outcome.result.kind !== "standalone_html"

const errorStatus = (error: unknown): number | null => {
  const status = error && typeof error === "object" ? (error as { status?: unknown }).status : null
  return typeof status === "number" && Number.isFinite(status) ? status : null
}

const encodeVisualStyleValue = (styleId: string | null, styleScope: string | null): string =>
  styleId && styleScope ? `${styleScope}::${styleId}` : ""

const parseVisualStyleValue = (
  value: string
): { visualStyleId: string | null; visualStyleScope: string | null } => {
  if (!value) {
    return { visualStyleId: null, visualStyleScope: null }
  }
  const separatorIndex = value.indexOf("::")
  if (separatorIndex === -1) {
    return { visualStyleId: null, visualStyleScope: null }
  }
  const visualStyleScope = value.slice(0, separatorIndex).trim()
  const visualStyleId = value.slice(separatorIndex + 2).trim()
  if (!visualStyleScope || !visualStyleId) {
    return { visualStyleId: null, visualStyleScope: null }
  }
  return { visualStyleId, visualStyleScope }
}

const getDefaultVisualStyleValue = (styles: VisualStyleRecord[]): string => {
  const preferred =
    styles.find((style) => style.id === DEFAULT_VISUAL_STYLE_ID && style.scope === "builtin") ||
    styles[0]
  return preferred ? encodeVisualStyleValue(preferred.id, preferred.scope) : ""
}

const resolveThemeFromVisualStyle = (
  style: VisualStyleRecord | null,
  fallbackTheme: string
): string => {
  if (style?.scope !== "builtin") {
    return fallbackTheme
  }
  const resolvedTheme = style.appearance_defaults?.theme
  return typeof resolvedTheme === "string" && resolvedTheme.trim().length > 0
    ? resolvedTheme.trim()
    : fallbackTheme
}

export const PresentationStudioPage: React.FC<PresentationStudioPageProps> = ({
  mode = "index",
  projectId = null,
  embedded = false
}) => {
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const { capabilities, loading } = useServerCapabilities()
  const loadProject = usePresentationStudioStore((state) => state.loadProject)
  const title = usePresentationStudioStore((state) => state.title)
  const slides = usePresentationStudioStore((state) => state.slides)
  const currentProjectId = usePresentationStudioStore((state) => state.projectId)
  const theme = usePresentationStudioStore((state) => state.theme)
  const visualStyleId = usePresentationStudioStore((state) => state.visualStyleId)
  const visualStyleScope = usePresentationStudioStore((state) => state.visualStyleScope)
  const visualStyleName = usePresentationStudioStore((state) => state.visualStyleName)
  const updateProjectMeta = usePresentationStudioStore((state) => state.updateProjectMeta)
  const [isProjectLoading, setIsProjectLoading] = React.useState(mode === "detail")
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [availableStyles, setAvailableStyles] = React.useState<VisualStyleRecord[]>([])
  const [stylesLoading, setStylesLoading] = React.useState(mode === "new")
  const [stylesError, setStylesError] = React.useState<string | null>(null)
  const [draftTitle, setDraftTitle] = React.useState("Untitled Presentation")
  const [draftVisualStyleValue, setDraftVisualStyleValue] = React.useState("")
  const [isCreatingProject, setIsCreatingProject] = React.useState(false)
  const [detailState, setDetailState] = React.useState<DetailSurfaceState | null>(null)
  const [authorityEpoch, setAuthorityEpoch] = React.useState(0)
  const [kindAuthorityReleaseRequired, setKindAuthorityReleaseRequired] =
    React.useState(false)
  const detailRequestRef = React.useRef<InFlightProjectRequest | null>(null)
  const bufferedDetailOutcomeRef = React.useRef<BufferedDetailOutcome | null>(null)
  const authorityEpochRef = React.useRef(authorityEpoch)
  const authoritySuspendedRef = React.useRef(false)
  const standaloneAuthoritySettlementRef = React.useRef<{
    authorityEpoch: number
    releaseSafe: boolean
  } | null>(null)
  const detailStateRef = React.useRef(detailState)
  detailStateRef.current = detailState
  const detailContextRef = React.useRef({ mode, projectId, authorityEpoch })
  const detailErrorContextRef = React.useRef<string | null>(null)
  const detailContextKey = JSON.stringify([projectId, authorityEpoch])
  const currentDetailState =
    detailState?.projectId === projectId && detailState.authorityEpoch === authorityEpoch
      ? detailState
      : null
  const retainedStandaloneState =
    mode === "detail" &&
    projectId &&
    detailState?.projectId === projectId &&
    detailState.kind === "standalone_html"
      ? detailState
      : null
  const canRetainStandaloneSurface =
    Boolean(retainedStandaloneState) && !isExtensionRuntime()
  const standaloneKindRevalidationPending = Boolean(
    canRetainStandaloneSurface &&
    retainedStandaloneState &&
    retainedStandaloneState.authorityEpoch !== authorityEpoch
  )
  const currentDetailError = detailErrorContextRef.current === detailContextKey ? loadError : null

  const invalidateDetailRequest = React.useCallback(() => {
    detailRequestRef.current?.controller.abort()
    detailRequestRef.current = null
    bufferedDetailOutcomeRef.current = null
    setKindAuthorityReleaseRequired(false)
  }, [])

  React.useLayoutEffect(() => {
    const previous = detailContextRef.current
    if (
      previous.mode !== mode ||
      previous.projectId !== projectId ||
      previous.authorityEpoch !== authorityEpoch
    ) {
      invalidateDetailRequest()
      if (authoritySuspendedRef.current) setKindAuthorityReleaseRequired(true)
      standaloneAuthoritySettlementRef.current = null
      detailContextRef.current = { mode, projectId, authorityEpoch }
    }
  }, [authorityEpoch, invalidateDetailRequest, mode, projectId])

  const isStandaloneKindAuthorityCurrent = React.useCallback(
    (capturedAuthorityEpoch: number | null, candidatePresentationId: string) => {
      const committedContext = detailContextRef.current
      return (
        !authoritySuspendedRef.current &&
        capturedAuthorityEpoch !== null &&
        capturedAuthorityEpoch === authorityEpochRef.current &&
        committedContext.mode === "detail" &&
        committedContext.projectId === candidatePresentationId &&
        committedContext.authorityEpoch === capturedAuthorityEpoch
      )
    },
    []
  )

  React.useEffect(() => {
    const reserveAuthorityBoundary = () => {
      const nextEpoch = authorityEpochRef.current + 1
      authorityEpochRef.current = nextEpoch
      standaloneAuthoritySettlementRef.current = null
      return nextEpoch
    }
    const commitAuthorityBoundary = (nextEpoch: number) => {
      invalidateDetailRequest()
      detailErrorContextRef.current = null
      setLoadError(null)
      if (mode === "detail") setIsProjectLoading(true)
      setAuthorityEpoch(nextEpoch)
    }
    const handleAuthorityBoundary = () => {
      const nextEpoch = reserveAuthorityBoundary()
      commitAuthorityBoundary(nextEpoch)
    }
    const handleRestorationBoundary = () => {
      if (authoritySuspendedRef.current) {
        const nextEpoch = reserveAuthorityBoundary()
        authoritySuspendedRef.current = false
        flushSync(() => commitAuthorityBoundary(nextEpoch))
        return
      }
      const retained = detailStateRef.current
      const context = detailContextRef.current
      if (
        retained?.kind === "standalone_html" &&
        retained.projectId === context.projectId &&
        retained.authorityEpoch !== authorityEpochRef.current
      ) {
        return
      }
      const nextEpoch = reserveAuthorityBoundary()
      flushSync(() => commitAuthorityBoundary(nextEpoch))
    }
    const handlePagehideBoundary = () => {
      if (authoritySuspendedRef.current) return
      authoritySuspendedRef.current = true
      const nextEpoch = reserveAuthorityBoundary()
      flushSync(() => {
        commitAuthorityBoundary(nextEpoch)
        setKindAuthorityReleaseRequired(true)
      })
    }
    const handleVisibilityBoundary = () => {
      if (document.visibilityState === "visible") handleRestorationBoundary()
    }
    window.addEventListener("tldw:config-updated", handleAuthorityBoundary)
    window.addEventListener("tldw:auth-principal-changed", handleAuthorityBoundary)
    window.addEventListener("tldw:slides-scope-mismatch", handleAuthorityBoundary)
    window.addEventListener("pagehide", handlePagehideBoundary, true)
    window.addEventListener("pageshow", handleRestorationBoundary, true)
    window.addEventListener("focus", handleRestorationBoundary, true)
    document.addEventListener("visibilitychange", handleVisibilityBoundary, true)
    return () => {
      window.removeEventListener("tldw:config-updated", handleAuthorityBoundary)
      window.removeEventListener("tldw:auth-principal-changed", handleAuthorityBoundary)
      window.removeEventListener("tldw:slides-scope-mismatch", handleAuthorityBoundary)
      window.removeEventListener("pagehide", handlePagehideBoundary, true)
      window.removeEventListener("pageshow", handleRestorationBoundary, true)
      window.removeEventListener("focus", handleRestorationBoundary, true)
      document.removeEventListener("visibilitychange", handleVisibilityBoundary, true)
    }
  }, [invalidateDetailRequest, mode])

  const refreshVisualStyles = React.useCallback(async (): Promise<VisualStyleRecord[]> => {
    const styles = await tldwClient.listVisualStyles()
    setAvailableStyles(Array.isArray(styles) ? styles : [])
    return Array.isArray(styles) ? styles : []
  }, [])

  const adoptDetailOutcome = React.useCallback(
    (buffered: BufferedDetailOutcome) => {
      const context = detailContextRef.current
      if (
        context.mode !== "detail" ||
        context.projectId !== buffered.projectId ||
        context.authorityEpoch !== buffered.authorityEpoch ||
        authorityEpochRef.current !== buffered.authorityEpoch
      ) {
        return
      }
      bufferedDetailOutcomeRef.current = null
      setKindAuthorityReleaseRequired(false)
      if (buffered.outcome.kind === "error") {
        detailErrorContextRef.current = JSON.stringify([
          buffered.projectId,
          buffered.authorityEpoch
        ])
        setLoadError(buffered.outcome.message)
        setDetailState(null)
        setIsProjectLoading(false)
        return
      }
      detailErrorContextRef.current = null
      setLoadError(null)
      const result = buffered.outcome.result
      if (result.kind === "structured") {
        loadProject(result.detail.record, {
          etag: result.detail.etag
        })
        setDetailState({
          projectId: buffered.projectId,
          authorityEpoch: buffered.authorityEpoch,
          kind: "structured"
        })
      } else {
        setDetailState({
          projectId: buffered.projectId,
          authorityEpoch: buffered.authorityEpoch,
          ...result
        })
      }
      setIsProjectLoading(false)
    },
    [loadProject]
  )

  const queueOrAdoptDetailOutcome = React.useCallback(
    (buffered: BufferedDetailOutcome) => {
      const retained = detailStateRef.current
      const settlement = standaloneAuthoritySettlementRef.current
      const waitsForRetainedWorkspace =
        retained?.projectId === buffered.projectId &&
        retained.kind === "standalone_html" &&
        retained.authorityEpoch !== buffered.authorityEpoch &&
        (settlement?.authorityEpoch !== buffered.authorityEpoch ||
          (detailOutcomeRequiresRelease(buffered) && !settlement.releaseSafe))
      if (waitsForRetainedWorkspace) {
        bufferedDetailOutcomeRef.current = buffered
        setKindAuthorityReleaseRequired(detailOutcomeRequiresRelease(buffered))
        return
      }
      adoptDetailOutcome(buffered)
    },
    [adoptDetailOutcome]
  )

  const handleStandaloneAuthoritySettled = React.useCallback(
    (settledEpoch: number, releaseSafe: boolean) => {
      if (authorityEpochRef.current !== settledEpoch) return
      const retained = detailStateRef.current
      if (
        retained?.projectId !== detailContextRef.current.projectId ||
        retained.kind !== "standalone_html" ||
        retained.authorityEpoch === settledEpoch
      ) {
        return
      }
      standaloneAuthoritySettlementRef.current = {
        authorityEpoch: settledEpoch,
        releaseSafe
      }
      const buffered = bufferedDetailOutcomeRef.current
      if (
        buffered?.authorityEpoch === settledEpoch &&
        (!detailOutcomeRequiresRelease(buffered) || releaseSafe)
      ) {
        adoptDetailOutcome(buffered)
      }
    },
    [adoptDetailOutcome]
  )

  React.useEffect(() => {
    const shouldLoadStyles =
      mode === "new" ||
      (mode === "detail" &&
        currentDetailState?.kind === "structured" &&
        currentProjectId === projectId &&
        !isProjectLoading)
    if (!shouldLoadStyles || !isOnline) {
      return
    }

    let cancelled = false
    setStylesLoading(true)
    setStylesError(null)
    void refreshVisualStyles()
      .then((styles) => {
        if (cancelled) {
          return
        }
        setDraftVisualStyleValue((currentValue) => currentValue || getDefaultVisualStyleValue(styles))
      })
      .catch((error) => {
        if (cancelled) {
          return
        }
        setAvailableStyles([])
        setStylesError(toErrorMessage(error))
        setDraftVisualStyleValue("")
      })
      .finally(() => {
        if (cancelled) {
          return
        }
        setStylesLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [currentDetailState?.kind, currentProjectId, isOnline, isProjectLoading, mode, projectId, refreshVisualStyles])

  React.useEffect(() => {
    if (authoritySuspendedRef.current || mode !== "detail" || !projectId) {
      return
    }
    let cancelled = false
    setIsProjectLoading(true)
    setLoadError(null)
    detailErrorContextRef.current = null
    if (
      !detailRequestRef.current ||
      detailRequestRef.current.projectId !== projectId ||
      detailRequestRef.current.authorityEpoch !== authorityEpoch
    ) {
      const controller = new AbortController()
      const request: InFlightProjectRequest = {
        projectId,
        authorityEpoch,
        controller,
        promise: Promise.resolve(null)
      }
      detailRequestRef.current = request
      const requestIsCurrent = () =>
        detailRequestRef.current === request && !controller.signal.aborted
      request.promise = (async () => {
          try {
            const metadata = await tldwClient.getPresentationMetadata(projectId)
            if (!requestIsCurrent()) return null
            if (metadata.record.id !== projectId) {
              throw new Error("Presentation metadata could not be verified.")
            }
            if (metadata.record.content_kind === "structured_slides") {
              const detail = await tldwClient.getPresentation(projectId, {
                abortSignal: controller.signal
              })
              if (!requestIsCurrent()) return null
              if (
                detail.record.content_kind !== "structured_slides" ||
                detail.record.id !== projectId
              ) {
                throw new Error("Structured presentation could not be verified.")
              }
              return { kind: "structured", detail }
            }
            if (metadata.record.content_kind === "standalone_html") {
              return { kind: "standalone_html" }
            }
            return {
              kind: "unsupported",
              contentKind: metadata.record.content_kind === "unsupported"
                ? metadata.record.unsupported_content_kind
                : null
            }
          } catch (error) {
            if (!requestIsCurrent()) return null
            if (errorStatus(error) !== 404) throw error
            try {
              await tldwClient.getSlidesCapabilities({ abortSignal: controller.signal })
              if (!requestIsCurrent()) return null
              return { kind: "metadata_unavailable" }
            } catch (capabilityError) {
              if (!requestIsCurrent()) return null
              if (errorStatus(capabilityError) !== 404) {
                return { kind: "metadata_unavailable" }
              }
              if (isExtensionRuntime()) {
                return { kind: "metadata_unavailable" }
              }
              const detail = await tldwClient.getPresentation(projectId, {
                abortSignal: controller.signal
              })
              if (!requestIsCurrent()) return null
              if (
                detail.record.content_kind !== "structured_slides" ||
                detail.record.id !== projectId
              ) {
                return { kind: "metadata_unavailable" }
              }
              return { kind: "structured", detail }
            }
          }
        })()
    }

    const request = detailRequestRef.current
    void request.promise
      .then((result) => {
        if (cancelled || detailRequestRef.current !== request || !result) return
        queueOrAdoptDetailOutcome({
          projectId,
          authorityEpoch,
          outcome: { kind: "result", result }
        })
        if (detailRequestRef.current === request) detailRequestRef.current = null
      })
      .catch((error) => {
        if (cancelled || detailRequestRef.current !== request) return
        detailRequestRef.current = null
        queueOrAdoptDetailOutcome({
          projectId,
          authorityEpoch,
          outcome: { kind: "error", message: toErrorMessage(error) }
        })
      })
    return () => {
      cancelled = true
    }
  }, [authorityEpoch, detailContextKey, mode, projectId, queueOrAdoptDetailOutcome])

  const styleOptions = React.useMemo(() => {
    const options = [...availableStyles]
    if (
      visualStyleId &&
      visualStyleScope &&
      !options.some((style) => style.id === visualStyleId && style.scope === visualStyleScope)
    ) {
      options.unshift({
        id: visualStyleId,
        scope: visualStyleScope,
        name: visualStyleName || `${visualStyleScope}:${visualStyleId}`,
        description: "This style is no longer available, but this deck still retains its snapshot.",
        category: null,
        guide_number: null,
        tags: [],
        best_for: [],
        generation_rules: {},
        artifact_preferences: [],
        appearance_defaults: {},
        fallback_policy: {},
        version: null
      })
    }
    return options
  }, [availableStyles, visualStyleId, visualStyleName, visualStyleScope])

  const selectedDraftStyle = React.useMemo(() => {
    const { visualStyleId: nextStyleId, visualStyleScope: nextStyleScope } =
      parseVisualStyleValue(draftVisualStyleValue)
    return (
      styleOptions.find(
        (style) => style.id === nextStyleId && style.scope === nextStyleScope
      ) || null
    )
  }, [draftVisualStyleValue, styleOptions])

  const selectedPresentationStyle = React.useMemo(
    () =>
      styleOptions.find(
        (style) => style.id === visualStyleId && style.scope === visualStyleScope
      ) || null,
    [styleOptions, visualStyleId, visualStyleScope]
  )
  const selectedCustomStyle = React.useMemo(() => {
    const candidate = mode === "new" ? selectedDraftStyle : selectedPresentationStyle
    return candidate?.scope === "user" ? candidate : null
  }, [mode, selectedDraftStyle, selectedPresentationStyle])

  const applySelectedStyle = React.useCallback(
    (style: VisualStyleRecord | null) => {
      if (mode === "new") {
        setDraftVisualStyleValue(
          style ? encodeVisualStyleValue(style.id, style.scope) : ""
        )
        return
      }
      updateProjectMeta({
        visualStyleId: style?.id ?? null,
        visualStyleScope: style?.scope ?? null,
        visualStyleName: style?.name ?? null,
        visualStyleVersion: style?.version ?? null,
        visualStyleSnapshot: style ? buildPresentationVisualStyleSnapshot(style) : null
      })
    },
    [mode, updateProjectMeta]
  )

  const handleDraftStyleChange = React.useCallback((nextValue: string) => {
    setDraftVisualStyleValue(nextValue)
  }, [])

  const handleCreateProject = React.useCallback(async () => {
    if (isCreatingProject) {
      return
    }
    const { visualStyleId: selectedVisualStyleId, visualStyleScope: selectedVisualStyleScope } =
      parseVisualStyleValue(draftVisualStyleValue)
    const selectedStyle =
      styleOptions.find(
        (style) =>
          style.id === selectedVisualStyleId && style.scope === selectedVisualStyleScope
      ) || null
    const blankSlideId = createBlankSlideId()
    setIsCreatingProject(true)
    setLoadError(null)
    try {
      const project = await tldwClient.createPresentation({
        title: draftTitle.trim() || "Untitled Presentation",
        description: null,
        visual_style_id: selectedVisualStyleId,
        visual_style_scope: selectedVisualStyleScope,
        visual_style_name: selectedStyle?.name ?? null,
        visual_style_version: selectedStyle?.version ?? null,
        visual_style_snapshot: selectedStyle
          ? buildPresentationVisualStyleSnapshot(selectedStyle)
          : null,
        studio_data: {
          origin: "blank",
          entry_surface: "webui_new"
        },
        slides: [
          {
            order: 0,
            layout: "title",
            title: "Title slide",
            content: "",
            speaker_notes: "",
            metadata: {
              studio: {
                slideId: blankSlideId,
                transition: "fade",
                timing_mode: "auto",
                manual_duration_ms: null,
                audio: { status: "missing" },
                image: { status: "missing" }
              }
            }
          }
        ]
      })
      loadProject(project, {
        etag: formatEtag(project.version)
      })
      navigate(`/presentation-studio/${project.id}`, {
        replace: true
      })
    } catch (error) {
      setLoadError(toErrorMessage(error))
    } finally {
      setIsCreatingProject(false)
    }
  }, [draftTitle, draftVisualStyleValue, isCreatingProject, loadProject, navigate, styleOptions])

  const handleDetailStyleChange = React.useCallback(
    (nextValue: string) => {
      const { visualStyleId: nextStyleId, visualStyleScope: nextStyleScope } =
        parseVisualStyleValue(nextValue)
      const selectedStyle =
        styleOptions.find(
          (style) => style.id === nextStyleId && style.scope === nextStyleScope
        ) || null
      const nextTheme = resolveThemeFromVisualStyle(selectedStyle, theme)
      updateProjectMeta({
        visualStyleId: nextStyleId,
        visualStyleScope: nextStyleScope,
        visualStyleName: selectedStyle?.name ?? null,
        visualStyleVersion: selectedStyle?.version ?? null,
        visualStyleSnapshot: selectedStyle
          ? buildPresentationVisualStyleSnapshot(selectedStyle)
          : null,
        theme: selectedStyle?.scope === "builtin" ? nextTheme : undefined
      })
    },
    [styleOptions, theme, updateProjectMeta]
  )

  const retainedStandaloneWorkspace = canRetainStandaloneSurface && projectId ? (
    <StandaloneHtmlWorkspace
      key={projectId}
      presentationId={projectId}
      kindAuthorityPending={standaloneKindRevalidationPending}
      kindAuthorityEpoch={authorityEpoch}
      kindAuthorityReleaseRequired={kindAuthorityReleaseRequired}
      onKindAuthoritySettled={handleStandaloneAuthoritySettled}
      isKindAuthorityCurrent={isStandaloneKindAuthorityCurrent}
    />
  ) : null

  if (retainedStandaloneWorkspace && (!isOnline || standaloneKindRevalidationPending)) {
    return retainedStandaloneWorkspace
  }

  if (!isOnline) {
    const Heading = embedded ? "h2" : "h1"
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <Heading className="text-2xl font-semibold text-slate-900">Presentation Studio</Heading>
        <p className="mt-2 text-sm text-slate-600">
          Server is offline. Connect to use Presentation Studio.
        </p>
      </section>
    )
  }

  if (!loading && capabilities && !capabilities.hasPresentationStudio) {
    const Heading = embedded ? "h2" : "h1"
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <Heading className="text-2xl font-semibold text-slate-900">Presentation Studio</Heading>
        <p className="mt-2 text-sm text-slate-600">
          Presentation Studio is not available on this server.
        </p>
      </section>
    )
  }

  if (mode === "index") {
    return <PresentationStudioIndex />
  }

  if (mode === "new") {
    const Heading = embedded ? "h2" : "h1"
    return (
      <section className="space-y-4">
        <header className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="space-y-2">
            <Heading className="text-2xl font-semibold text-slate-900">
              {embedded ? "Structured presentation setup" : "Presentation Studio"}
            </Heading>
            <p className="max-w-2xl text-sm text-slate-600">
              Start a new deck with a reusable visual style preset. The selected style
              sets the default strategy for future generated slides.
            </p>
          </div>
        </header>

        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_280px]">
            <div className="space-y-4">
              <div>
                <label
                  className="mb-1 block text-sm font-medium text-slate-700"
                  htmlFor="presentation-studio-title"
                >
                  Presentation title
                </label>
                <input
                  id="presentation-studio-title"
                  value={draftTitle}
                  onChange={(event) => setDraftTitle(event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 shadow-sm outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-100"
                  placeholder="Untitled Presentation"
                />
              </div>

              <div>
                <VisualStylePicker
                  label="Choose visual style"
                  value={draftVisualStyleValue}
                  styles={styleOptions}
                  onChange={handleDraftStyleChange}
                  disabled={stylesLoading || isCreatingProject}
                  loading={stylesLoading}
                  description="Built-ins stay read-only. Updates deck appearance defaults and future generated slides. Existing slide content stays unchanged. Custom styles stay editable below."
                />
              </div>

              {stylesError && <p className="text-sm text-rose-600">{stylesError}</p>}
              {loadError && <p className="text-sm text-rose-600">{loadError}</p>}
            </div>

            <aside className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h2 className="text-sm font-semibold text-slate-900">Style coverage</h2>
              <p className="mt-2 text-sm text-slate-600">
                Built-ins include academic, exam-focused, infographic, timeline,
                data-heavy, storytelling, diagram-first, and high-contrast revision
                presets.
              </p>
              <p className="mt-4 text-xs uppercase tracking-wide text-slate-500">
                Built-in and per-user custom styles are listed together here.
              </p>
            </aside>
          </div>

          <div className="mt-6 flex items-center justify-end">
            <button
              type="button"
              data-testid="presentation-studio-create-button"
              onClick={() => void handleCreateProject()}
              disabled={isCreatingProject}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {isCreatingProject ? "Creating presentation…" : "Create presentation"}
            </button>
          </div>
        </section>

        <VisualStyleManager
          selectedCustomStyle={selectedCustomStyle}
          refreshVisualStyles={refreshVisualStyles}
          onStyleSelected={applySelectedStyle}
        />
      </section>
    )
  }

  if (retainedStandaloneWorkspace) return retainedStandaloneWorkspace

  if (currentDetailError) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio</h1>
        <p className="mt-2 text-sm text-rose-600">{currentDetailError}</p>
      </section>
    )
  }

  if (isProjectLoading || !currentDetailState) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <p className="text-sm text-slate-600">Loading presentation…</p>
      </section>
    )
  }

  if (currentDetailState.kind === "standalone_html" && isExtensionRuntime()) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-slate-600">
          Standalone HTML editing is available only in the WebUI.
        </p>
      </section>
    )
  }

  if (currentDetailState.kind === "unsupported") {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Unsupported presentation kind</h1>
        <p className="mt-2 text-sm text-slate-600">
          This presentation type is read only in this version.
        </p>
        {currentDetailState.contentKind ? <code className="mt-3 block break-all text-sm text-slate-700">{currentDetailState.contentKind}</code> : null}
      </section>
    )
  }

  if (currentDetailState.kind === "metadata_unavailable") {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation metadata is unavailable</h1>
        <p className="mt-2 text-sm text-slate-600">
          Try again after the server can identify this presentation type.
        </p>
      </section>
    )
  }

  return (
    <section className="space-y-4">
      <header className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio</h1>
            <p className="mt-1 text-sm text-slate-600">
              {title || "Untitled Presentation"} · {slides.length} slide
              {slides.length === 1 ? "" : "s"}
            </p>
          </div>
          <div className="min-w-[240px] flex-1 sm:max-w-sm">
            <VisualStylePicker
              label="Choose visual style"
              value={encodeVisualStyleValue(visualStyleId, visualStyleScope)}
              styles={styleOptions}
              onChange={handleDetailStyleChange}
              disabled={stylesLoading}
              loading={stylesLoading}
              description="Updates deck appearance defaults and future generated slides. Existing slide content stays unchanged."
            />
          </div>
        </div>
      </header>

      <VisualStyleManager
        selectedCustomStyle={selectedCustomStyle}
        refreshVisualStyles={refreshVisualStyles}
        onStyleSelected={applySelectedStyle}
      />

      <ProjectWorkspace canRender={Boolean(capabilities?.hasPresentationRender)} />
    </section>
  )
}
