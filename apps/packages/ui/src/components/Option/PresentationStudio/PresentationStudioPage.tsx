import React from "react"
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
  projectId: string | null
  promise: Promise<DetailLoadResult>
}

type DetailLoadResult =
  | { kind: "structured"; detail: PresentationDetailResult }
  | { kind: "standalone_html" }
  | { kind: "unsupported"; contentKind: string | null }
  | { kind: "metadata_unavailable" }

type DetailSurfaceState =
  | { kind: "structured" }
  | Exclude<DetailLoadResult, { kind: "structured" }>

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
  const detailRequestRef = React.useRef<InFlightProjectRequest | null>(null)

  const refreshVisualStyles = React.useCallback(async (): Promise<VisualStyleRecord[]> => {
    const styles = await tldwClient.listVisualStyles()
    setAvailableStyles(Array.isArray(styles) ? styles : [])
    return Array.isArray(styles) ? styles : []
  }, [])

  React.useEffect(() => {
    const shouldLoadStyles =
      mode === "new" ||
      (mode === "detail" &&
        detailState?.kind === "structured" &&
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
  }, [currentProjectId, detailState?.kind, isOnline, isProjectLoading, mode, projectId, refreshVisualStyles])

  React.useEffect(() => {
    if (mode !== "detail" || !projectId) {
      return
    }
    // This store accepts structured records only, so an exact ID hit is already a
    // source-free kind decision and cannot bypass the standalone metadata boundary.
    if (currentProjectId === projectId) {
      setDetailState({ kind: "structured" })
      setIsProjectLoading(false)
      return
    }
    let cancelled = false
    setIsProjectLoading(true)
    setLoadError(null)
    setDetailState(null)
    if (!detailRequestRef.current || detailRequestRef.current.projectId !== projectId) {
      detailRequestRef.current = {
        projectId,
        promise: (async () => {
          try {
            const metadata = await tldwClient.getPresentationMetadata(projectId)
            if (metadata.record.content_kind === "structured_slides") {
              return { kind: "structured", detail: await tldwClient.getPresentation(projectId) }
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
            if (errorStatus(error) !== 404) throw error
            try {
              await tldwClient.getSlidesCapabilities()
              return { kind: "metadata_unavailable" }
            } catch (capabilityError) {
              if (errorStatus(capabilityError) !== 404) {
                return { kind: "metadata_unavailable" }
              }
              if (isExtensionRuntime()) {
                return { kind: "metadata_unavailable" }
              }
              const detail = await tldwClient.getPresentation(projectId)
              if (detail.record.content_kind !== "structured_slides") {
                return { kind: "metadata_unavailable" }
              }
              return { kind: "structured", detail }
            }
          }
        })() as Promise<DetailLoadResult>
      }
    }

    void detailRequestRef.current.promise
      .then((result) => {
        if (cancelled) {
          return
        }
        if (result.kind === "structured") {
          if (result.detail.record.content_kind !== "structured_slides") {
            throw new Error("Structured presentation required")
          }
          loadProject(result.detail.record, {
            etag: result.detail.etag
          })
          setDetailState({ kind: "structured" })
        } else {
          setDetailState(result)
        }
        setIsProjectLoading(false)
        detailRequestRef.current = null
      })
      .catch((error) => {
        if (cancelled) {
          return
        }
        detailRequestRef.current = null
        setLoadError(toErrorMessage(error))
        setIsProjectLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [currentProjectId, loadProject, mode, projectId])

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

  if (isProjectLoading) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <p className="text-sm text-slate-600">Loading presentation…</p>
      </section>
    )
  }

  if (loadError) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Presentation Studio</h1>
        <p className="mt-2 text-sm text-rose-600">{loadError}</p>
      </section>
    )
  }

  if (detailState?.kind === "standalone_html" && isExtensionRuntime()) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-slate-600">
          Standalone HTML editing is available only in the WebUI.
        </p>
      </section>
    )
  }

  if (detailState?.kind === "standalone_html" && projectId) {
    return <StandaloneHtmlWorkspace presentationId={projectId} />
  }

  if (detailState?.kind === "unsupported") {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h1 className="text-2xl font-semibold text-slate-900">Unsupported presentation kind</h1>
        <p className="mt-2 text-sm text-slate-600">
          This presentation type is read only in this version.
        </p>
        {detailState.contentKind ? <code className="mt-3 block break-all text-sm text-slate-700">{detailState.contentKind}</code> : null}
      </section>
    )
  }

  if (detailState?.kind === "metadata_unavailable") {
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
