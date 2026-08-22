import { createWithEqualityFn } from "zustand/traditional"

import type {
  PresentationVisualStyleSnapshot,
  StructuredPresentationStudioRecord,
  PresentationStudioSlide
} from "@/services/tldw/TldwApiClient"
import { clonePresentationVisualStyleSnapshot } from "@/services/tldw/TldwApiClient"

export type PresentationStudioAssetStatus =
  | "missing"
  | "ready"
  | "stale"
  | "generating"
  | "failed"

export type PresentationStudioTransition = "fade" | "cut" | "wipe" | "zoom"
export type PresentationStudioTimingMode = "auto" | "manual"

export type PresentationStudioSlideStudioMeta = {
  slideId: string
  transition: PresentationStudioTransition
  timing_mode: PresentationStudioTimingMode
  manual_duration_ms: number | null
  audio: {
    status: PresentationStudioAssetStatus
    asset_ref?: string | null
    duration_ms?: number | null
  }
  image: {
    status: PresentationStudioAssetStatus
    asset_ref?: string | null
  }
  [key: string]: any
}

export type PresentationStudioEditorSlide = Omit<PresentationStudioSlide, "metadata"> & {
  metadata: Record<string, any> & {
    studio: PresentationStudioSlideStudioMeta
  }
}

type PresentationStudioSlideUpdate = Partial<Omit<PresentationStudioEditorSlide, "metadata">> & {
  metadata?: Record<string, any> & {
    studio?: Partial<PresentationStudioSlideStudioMeta>
  }
}

export type PresentationStudioPatchPayload = {
  title: string
  description: string | null
  theme: string
  visual_style_id: string | null
  visual_style_scope: string | null
  visual_style_name: string | null
  visual_style_version: number | null
  visual_style_snapshot: PresentationVisualStyleSnapshot | null
  studio_data: Record<string, any> | null
  slides: PresentationStudioSlide[]
}

type AutosaveState = "idle" | "saving" | "error"
type SlideMoveDirection = "earlier" | "later"

type PresentationStudioStore = {
  projectId: string | null
  title: string
  description: string
  theme: string
  visualStyleId: string | null
  visualStyleScope: string | null
  visualStyleName: string | null
  visualStyleVersion: number | null
  visualStyleSnapshot: PresentationVisualStyleSnapshot | null
  studioData: Record<string, any> | null
  slides: PresentationStudioEditorSlide[]
  selectedSlideId: string | null
  etag: string | null
  isDirty: boolean
  autosaveState: AutosaveState
  autosaveError: string | null
  reset: () => void
  initializeBlankProject: () => void
  loadProject: (
    project: StructuredPresentationStudioRecord,
    options?: { etag?: string | null; preserveDirty?: boolean }
  ) => void
  updateProjectMeta: (updates: {
    title?: string
    description?: string
    theme?: string
    visualStyleId?: string | null
    visualStyleScope?: string | null
    visualStyleName?: string | null
    visualStyleVersion?: number | null
    visualStyleSnapshot?: PresentationVisualStyleSnapshot | null
    studioData?: Record<string, any> | null
  }) => void
  selectSlide: (slideId: string) => void
  addSlide: () => void
  duplicateSlide: (slideId: string) => void
  removeSlide: (slideId: string) => void
  moveSlide: (slideId: string, direction: SlideMoveDirection) => void
  reorderSlides: (fromIndex: number, toIndex: number) => void
  updateSlide: (
    slideId: string,
    updates: PresentationStudioSlideUpdate
  ) => void
  setAutosaveState: (state: AutosaveState, error?: string | null) => void
  markPersisted: (etag?: string | null, project?: StructuredPresentationStudioRecord) => void
  buildPatchPayload: () => {
    title: string
    description: string | null
    theme: string
    visual_style_id: string | null
    visual_style_scope: string | null
    visual_style_name: string | null
    visual_style_version: number | null
    visual_style_snapshot: PresentationVisualStyleSnapshot | null
    studio_data: Record<string, any> | null
    slides: PresentationStudioSlide[]
  }
}

const createSlideId = (): string =>
  globalThis.crypto?.randomUUID?.() ||
  `slide-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`

const normalizeAssetStatus = (
  candidate: unknown,
  fallback: PresentationStudioAssetStatus
): PresentationStudioAssetStatus => {
  const normalized = String(candidate || "").trim().toLowerCase()
  return ["missing", "ready", "stale", "generating", "failed"].includes(normalized)
    ? (normalized as PresentationStudioAssetStatus)
    : fallback
}

const normalizeTransition = (
  candidate: unknown,
  fallback: PresentationStudioTransition = "fade"
): PresentationStudioTransition => {
  const normalized = String(candidate || "").trim().toLowerCase()
  return ["fade", "cut", "wipe", "zoom"].includes(normalized)
    ? (normalized as PresentationStudioTransition)
    : fallback
}

const normalizeManualDuration = (candidate: unknown): number | null => {
  if (candidate === null || candidate === undefined || candidate === "") {
    return null
  }

  const numericValue =
    typeof candidate === "number"
      ? candidate
      : typeof candidate === "string"
        ? Number(candidate)
        : NaN

  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return null
  }

  return Math.round(numericValue)
}

const normalizeTimingMode = (
  candidate: unknown,
  fallback: PresentationStudioTimingMode = "auto"
): PresentationStudioTimingMode => {
  const normalized = String(candidate || "").trim().toLowerCase()
  return ["auto", "manual"].includes(normalized)
    ? (normalized as PresentationStudioTimingMode)
    : fallback
}

const resolveTimingMode = (
  candidate: unknown,
  manualDurationMs: number | null,
  fallback: PresentationStudioTimingMode = "auto"
): PresentationStudioTimingMode => {
  if (!manualDurationMs) {
    return "auto"
  }

  return normalizeTimingMode(candidate, fallback === "manual" ? "manual" : "auto")
}

const normalizeSlide = (
  slide: Partial<PresentationStudioSlide> & { metadata?: Record<string, any> },
  order: number
): PresentationStudioEditorSlide => {
  const metadata =
    slide.metadata && typeof slide.metadata === "object" && !Array.isArray(slide.metadata)
      ? { ...slide.metadata }
      : {}
  const studio =
    metadata.studio && typeof metadata.studio === "object" && !Array.isArray(metadata.studio)
      ? { ...metadata.studio }
      : {}
  const audioState =
    studio.audio && typeof studio.audio === "object" && !Array.isArray(studio.audio)
      ? { ...studio.audio }
      : {}
  const imageState =
    studio.image && typeof studio.image === "object" && !Array.isArray(studio.image)
      ? { ...studio.image }
      : {}
  const slideId =
    typeof studio.slideId === "string" && studio.slideId.trim().length > 0
      ? studio.slideId
      : createSlideId()
  const manualDurationMs = normalizeManualDuration(studio.manual_duration_ms)
  const timingMode = resolveTimingMode(
    studio.timing_mode,
    manualDurationMs,
    manualDurationMs ? "manual" : "auto"
  )

  return {
    order,
    layout: String(slide.layout || "content"),
    title: slide.title ?? "",
    content: slide.content ?? "",
    speaker_notes: slide.speaker_notes ?? "",
    metadata: {
      ...metadata,
      studio: {
        ...studio,
        slideId,
        transition: normalizeTransition(studio.transition),
        timing_mode: timingMode,
        manual_duration_ms: manualDurationMs,
        audio: {
          ...audioState,
          status: normalizeAssetStatus(
            audioState.status,
            audioState.asset_ref ? "ready" : "missing"
          )
        },
        image: {
          ...imageState,
          status: normalizeAssetStatus(
            imageState.status,
            imageState.asset_ref ? "ready" : "missing"
          )
        }
      }
    }
  }
}

const buildPatchPayloadFromSlides = (input: {
  title: string
  description?: string | null
  theme: string
  visual_style_id?: string | null
  visual_style_scope?: string | null
  visual_style_name?: string | null
  visual_style_version?: number | null
  visual_style_snapshot?: PresentationVisualStyleSnapshot | null
  studio_data?: Record<string, any> | null
  slides: Array<PresentationStudioSlide | PresentationStudioEditorSlide>
}): PresentationStudioPatchPayload => ({
  title: input.title,
  description: input.description || null,
  theme: input.theme,
  visual_style_id: input.visual_style_id ?? null,
  visual_style_scope: input.visual_style_scope ?? null,
  visual_style_name: input.visual_style_name ?? null,
  visual_style_version: input.visual_style_version ?? null,
  visual_style_snapshot: clonePresentationVisualStyleSnapshot(input.visual_style_snapshot),
  studio_data: input.studio_data ?? null,
  slides: input.slides.map((slide, index) => ({
    order: index,
    layout: slide.layout,
    title: slide.title ?? "",
    content: slide.content,
    speaker_notes: slide.speaker_notes ?? "",
    metadata: slide.metadata
  }))
})

const mergeSlideMetadata = (
  remoteMetadata: Record<string, any> | undefined,
  localMetadata: Record<string, any> | undefined
): Record<string, any> => {
  const remote = remoteMetadata && typeof remoteMetadata === "object" ? remoteMetadata : {}
  const local = localMetadata && typeof localMetadata === "object" ? localMetadata : {}
  const remoteStudio =
    remote.studio && typeof remote.studio === "object" && !Array.isArray(remote.studio)
      ? remote.studio
      : {}
  const localStudio =
    local.studio && typeof local.studio === "object" && !Array.isArray(local.studio)
      ? local.studio
      : {}
  const remoteManualDurationMs = normalizeManualDuration(remoteStudio.manual_duration_ms)
  const localHasManualDuration = Object.prototype.hasOwnProperty.call(
    localStudio,
    "manual_duration_ms"
  )
  const mergedManualDurationMs = localHasManualDuration
    ? normalizeManualDuration(localStudio.manual_duration_ms)
    : remoteManualDurationMs
  const remoteAudio =
    remoteStudio.audio && typeof remoteStudio.audio === "object" && !Array.isArray(remoteStudio.audio)
      ? remoteStudio.audio
      : {}
  const localAudio =
    localStudio.audio && typeof localStudio.audio === "object" && !Array.isArray(localStudio.audio)
      ? localStudio.audio
      : {}
  const remoteImage =
    remoteStudio.image && typeof remoteStudio.image === "object" && !Array.isArray(remoteStudio.image)
      ? remoteStudio.image
      : {}
  const localImage =
    localStudio.image && typeof localStudio.image === "object" && !Array.isArray(localStudio.image)
      ? localStudio.image
      : {}

  return {
    ...remote,
    ...local,
    images:
      Object.prototype.hasOwnProperty.call(local, "images") && Array.isArray(local.images)
        ? local.images
        : remote.images,
    studio: {
      ...remoteStudio,
      ...localStudio,
      slideId:
        typeof localStudio.slideId === "string" && localStudio.slideId.trim().length > 0
          ? localStudio.slideId
          : typeof remoteStudio.slideId === "string" && remoteStudio.slideId.trim().length > 0
            ? remoteStudio.slideId
            : createSlideId(),
      transition: normalizeTransition(
        localStudio.transition ?? remoteStudio.transition,
        normalizeTransition(remoteStudio.transition)
      ),
      timing_mode: resolveTimingMode(
        localStudio.timing_mode ?? remoteStudio.timing_mode,
        mergedManualDurationMs,
        mergedManualDurationMs ? "manual" : "auto"
      ),
      manual_duration_ms: mergedManualDurationMs,
      audio: {
        ...remoteAudio,
        ...localAudio,
        status: normalizeAssetStatus(
          localAudio.status ?? remoteAudio.status,
          localAudio.asset_ref || remoteAudio.asset_ref ? "ready" : "missing"
        )
      },
      image: {
        ...remoteImage,
        ...localImage,
        status: normalizeAssetStatus(
          localImage.status ?? remoteImage.status,
          localImage.asset_ref || remoteImage.asset_ref ? "ready" : "missing"
        )
      }
    }
  }
}

export const buildPresentationStudioPatchPayloadFromRecord = (
  project: Pick<
    StructuredPresentationStudioRecord,
    | "title"
    | "description"
    | "theme"
    | "visual_style_id"
    | "visual_style_scope"
    | "visual_style_name"
    | "visual_style_version"
    | "visual_style_snapshot"
    | "studio_data"
    | "slides"
  >
): PresentationStudioPatchPayload =>
  buildPatchPayloadFromSlides({
    title: project.title,
    description: project.description,
    theme: project.theme,
    visual_style_id: project.visual_style_id ?? null,
    visual_style_scope: project.visual_style_scope ?? null,
    visual_style_name: project.visual_style_name ?? null,
    visual_style_version: project.visual_style_version ?? null,
    visual_style_snapshot: project.visual_style_snapshot ?? null,
    studio_data: project.studio_data ?? null,
    slides: project.slides
  })

export const mergePresentationStudioDraftWithRemote = (
  latest: StructuredPresentationStudioRecord,
  localDraft: PresentationStudioPatchPayload
): StructuredPresentationStudioRecord => {
  const localHasVisualStyleName = Object.prototype.hasOwnProperty.call(
    localDraft,
    "visual_style_name"
  )
  const localHasVisualStyleVersion = Object.prototype.hasOwnProperty.call(
    localDraft,
    "visual_style_version"
  )
  const localHasVisualStyleSnapshot = Object.prototype.hasOwnProperty.call(
    localDraft,
    "visual_style_snapshot"
  )
  const remoteSlides = (latest.slides || []).map((slide, index) => normalizeSlide(slide, index))
  const localSlides = (localDraft.slides || []).map((slide, index) => normalizeSlide(slide, index))
  const remoteBySlideId = new Map(
    remoteSlides.map((slide) => [slide.metadata.studio.slideId, slide] as const)
  )

  const mergedSlides: PresentationStudioSlide[] = []
  for (const localSlide of localSlides) {
    const slideId = localSlide.metadata.studio.slideId
    const remoteSlide = remoteBySlideId.get(slideId)
    if (remoteSlide) {
      remoteBySlideId.delete(slideId)
    }
    mergedSlides.push({
      order: mergedSlides.length,
      layout: localSlide.layout,
      title: localSlide.title ?? remoteSlide?.title ?? "",
      content: localSlide.content ?? remoteSlide?.content ?? "",
      speaker_notes: localSlide.speaker_notes ?? remoteSlide?.speaker_notes ?? "",
      metadata: mergeSlideMetadata(remoteSlide?.metadata, localSlide.metadata)
    })
  }

  for (const remoteSlide of remoteSlides) {
    if (!remoteBySlideId.has(remoteSlide.metadata.studio.slideId)) {
      continue
    }
    mergedSlides.push({
      order: mergedSlides.length,
      layout: remoteSlide.layout,
      title: remoteSlide.title ?? "",
      content: remoteSlide.content,
      speaker_notes: remoteSlide.speaker_notes ?? "",
      metadata: remoteSlide.metadata
    })
  }

  return {
    ...latest,
    title: localDraft.title,
    description: localDraft.description,
    theme: localDraft.theme,
    visual_style_id: localDraft.visual_style_id,
    visual_style_scope: localDraft.visual_style_scope,
    visual_style_name:
      localHasVisualStyleName
        ? localDraft.visual_style_name ?? null
        : latest.visual_style_name ?? latest.visual_style_snapshot?.name ?? null,
    visual_style_version: localHasVisualStyleVersion
      ? localDraft.visual_style_version ?? null
      : latest.visual_style_version ?? null,
    visual_style_snapshot:
      localHasVisualStyleSnapshot
        ? clonePresentationVisualStyleSnapshot(localDraft.visual_style_snapshot)
        : clonePresentationVisualStyleSnapshot(latest.visual_style_snapshot),
    studio_data: localDraft.studio_data,
    slides: mergedSlides
  }
}

const createBlankSlide = (order: number): PresentationStudioEditorSlide =>
  normalizeSlide(
    {
      order,
      layout: order === 0 ? "title" : "content",
      title: order === 0 ? "Title slide" : `Slide ${order + 1}`,
      content: "",
      speaker_notes: "",
      metadata: {}
    },
    order
  )

const createDuplicatedSlide = (
  slide: PresentationStudioEditorSlide,
  order: number
): PresentationStudioEditorSlide =>
  normalizeSlide(
    {
      ...slide,
      order,
      title: slide.title ? `${slide.title} copy` : `Slide ${order + 1}`,
      metadata: {
        ...slide.metadata,
        studio: {
          ...slide.metadata.studio,
          slideId: createSlideId(),
          audio: {
            ...slide.metadata.studio.audio
          },
          image: {
            ...slide.metadata.studio.image
          }
        }
      }
    },
    order
  )

const createInitialState = () => ({
  projectId: null,
  title: "Untitled Presentation",
  description: "",
  theme: "black",
  visualStyleId: null as string | null,
  visualStyleScope: null as string | null,
  visualStyleName: null as string | null,
  visualStyleVersion: null as number | null,
  visualStyleSnapshot: null as PresentationVisualStyleSnapshot | null,
  studioData: { origin: "blank" } as Record<string, any> | null,
  slides: [] as PresentationStudioEditorSlide[],
  selectedSlideId: null as string | null,
  etag: null as string | null,
  isDirty: false,
  autosaveState: "idle" as AutosaveState,
  autosaveError: null as string | null
})

const assertStructuredPresentation = (project: StructuredPresentationStudioRecord): void => {
  const candidate = project as unknown as Record<string, unknown>
  if (
    (candidate.content_kind != null && candidate.content_kind !== "structured_slides") ||
    Object.prototype.hasOwnProperty.call(candidate, "html_document")
  ) {
    throw new Error("Structured presentation required")
  }
}

export const usePresentationStudioStore = createWithEqualityFn<PresentationStudioStore>(
  (set, get) => ({
    ...createInitialState(),

    reset: () => set(createInitialState()),

    initializeBlankProject: () => {
      const firstSlide = createBlankSlide(0)
      set({
        ...createInitialState(),
        slides: [firstSlide],
        selectedSlideId: firstSlide.metadata.studio.slideId
      })
    },

    loadProject: (project, options) => {
      assertStructuredPresentation(project)
      set((state) => {
      const slides = (project.slides || []).map((slide, index) => normalizeSlide(slide, index))
      const previousSelected = state.selectedSlideId
      const selectedSlideId =
        slides.find((slide) => slide.metadata.studio.slideId === previousSelected)?.metadata.studio.slideId ||
        slides[0]?.metadata.studio.slideId ||
        null
      const etag = Object.prototype.hasOwnProperty.call(options ?? {}, "etag")
        ? options?.etag ?? null
        : `W/"v${project.version}"`
      return {
        projectId: project.id,
        title: project.title || "Untitled Presentation",
        description: project.description || "",
        theme: project.theme || "black",
        visualStyleId: project.visual_style_id ?? null,
        visualStyleScope: project.visual_style_scope ?? null,
        visualStyleName:
          project.visual_style_name ?? project.visual_style_snapshot?.name ?? null,
        visualStyleVersion: project.visual_style_version ?? null,
        visualStyleSnapshot: clonePresentationVisualStyleSnapshot(project.visual_style_snapshot),
        studioData:
          project.studio_data && typeof project.studio_data === "object"
            ? { ...project.studio_data }
            : null,
        slides,
        selectedSlideId,
        etag,
        isDirty: Boolean(options?.preserveDirty),
        autosaveState: options?.preserveDirty ? state.autosaveState : "idle",
        autosaveError: options?.preserveDirty ? state.autosaveError : null
      }
    })
    },

    updateProjectMeta: (updates) =>
      set((state) => ({
        title: updates.title ?? state.title,
        description: updates.description ?? state.description,
        theme: updates.theme ?? state.theme,
        visualStyleId:
          updates.visualStyleId !== undefined ? updates.visualStyleId : state.visualStyleId,
        visualStyleScope:
          updates.visualStyleScope !== undefined
            ? updates.visualStyleScope
            : state.visualStyleScope,
        visualStyleName:
          updates.visualStyleName !== undefined ? updates.visualStyleName : state.visualStyleName,
        visualStyleVersion:
          updates.visualStyleVersion !== undefined
            ? updates.visualStyleVersion
            : state.visualStyleVersion,
        visualStyleSnapshot:
          updates.visualStyleSnapshot !== undefined
            ? clonePresentationVisualStyleSnapshot(updates.visualStyleSnapshot)
            : state.visualStyleSnapshot,
        studioData: updates.studioData ?? state.studioData,
        isDirty: true
      })),

    selectSlide: (slideId) => set({ selectedSlideId: slideId }),

    addSlide: () =>
      set((state) => {
        const slide = createBlankSlide(state.slides.length)
        return {
          slides: [...state.slides, slide],
          selectedSlideId: slide.metadata.studio.slideId,
          isDirty: true
        }
      }),

    duplicateSlide: (slideId) =>
      set((state) => {
        const sourceIndex = state.slides.findIndex(
          (slide) => slide.metadata.studio.slideId === slideId
        )
        if (sourceIndex === -1) {
          return state
        }
        const duplicate = createDuplicatedSlide(state.slides[sourceIndex]!, sourceIndex + 1)
        const slides = [
          ...state.slides.slice(0, sourceIndex + 1),
          duplicate,
          ...state.slides.slice(sourceIndex + 1)
        ].map((slide, index) => ({
          ...slide,
          order: index
        }))

        return {
          slides,
          selectedSlideId: duplicate.metadata.studio.slideId,
          isDirty: true
        }
      }),

    removeSlide: (slideId) =>
      set((state) => {
        if (state.slides.length <= 1) {
          return state
        }
        const sourceIndex = state.slides.findIndex(
          (slide) => slide.metadata.studio.slideId === slideId
        )
        if (sourceIndex === -1) {
          return state
        }
        const slides = state.slides
          .filter((slide) => slide.metadata.studio.slideId !== slideId)
          .map((slide, index) => ({
            ...slide,
            order: index
          }))
        const nextSelectedSlideId =
          state.selectedSlideId === slideId
            ? slides[Math.max(0, sourceIndex - 1)]?.metadata.studio.slideId ||
              slides[0]?.metadata.studio.slideId ||
              null
            : state.selectedSlideId

        return {
          slides,
          selectedSlideId: nextSelectedSlideId,
          isDirty: true
        }
      }),

    moveSlide: (slideId, direction) =>
      set((state) => {
        const sourceIndex = state.slides.findIndex(
          (slide) => slide.metadata.studio.slideId === slideId
        )
        if (sourceIndex === -1) {
          return state
        }

        const targetIndex = direction === "earlier" ? sourceIndex - 1 : sourceIndex + 1
        if (targetIndex < 0 || targetIndex >= state.slides.length) {
          return state
        }

        const slides = [...state.slides]
        const [movedSlide] = slides.splice(sourceIndex, 1)
        slides.splice(targetIndex, 0, movedSlide!)

        return {
          slides: slides.map((slide, index) => ({
            ...slide,
            order: index
          })),
          selectedSlideId: slideId,
          isDirty: true
        }
      }),

    reorderSlides: (fromIndex, toIndex) =>
      set((state) => {
        if (
          fromIndex < 0 ||
          toIndex < 0 ||
          fromIndex >= state.slides.length ||
          toIndex >= state.slides.length ||
          fromIndex === toIndex
        ) {
          return state
        }

        const slides = [...state.slides]
        const [movedSlide] = slides.splice(fromIndex, 1)
        slides.splice(toIndex, 0, movedSlide!)

        return {
          slides: slides.map((slide, index) => ({
            ...slide,
            order: index
          })),
          selectedSlideId: movedSlide?.metadata.studio.slideId || state.selectedSlideId,
          isDirty: true
        }
      }),

    updateSlide: (slideId, updates) =>
      set((state) => {
        const slides = state.slides.map((slide, index) => {
          if (slide.metadata.studio.slideId !== slideId) {
            return slide
          }
          const studioUpdates =
            updates.metadata?.studio &&
            typeof updates.metadata.studio === "object" &&
            !Array.isArray(updates.metadata.studio)
              ? updates.metadata.studio
              : null
          const hasManualDurationUpdate = Boolean(
            studioUpdates &&
              Object.prototype.hasOwnProperty.call(studioUpdates, "manual_duration_ms")
          )
          const hasTimingModeUpdate = Boolean(
            studioUpdates && Object.prototype.hasOwnProperty.call(studioUpdates, "timing_mode")
          )
          const nextManualDurationMs = hasManualDurationUpdate
            ? normalizeManualDuration(studioUpdates?.manual_duration_ms)
            : slide.metadata.studio.manual_duration_ms
          const nextTransition = normalizeTransition(
            studioUpdates?.transition,
            slide.metadata.studio.transition
          )
          const nextTimingMode = hasTimingModeUpdate
            ? normalizeTimingMode(
                studioUpdates?.timing_mode,
                nextManualDurationMs ? "manual" : "auto"
              )
            : hasManualDurationUpdate
              ? nextManualDurationMs
                ? "manual"
                : "auto"
              : slide.metadata.studio.timing_mode
          const nextMetadata =
            updates.metadata && typeof updates.metadata === "object"
              ? {
                  ...slide.metadata,
                  ...updates.metadata,
                  studio: {
                    ...slide.metadata.studio,
                    ...(studioUpdates || {}),
                    transition: nextTransition,
                    timing_mode: nextTimingMode,
                    manual_duration_ms: nextManualDurationMs,
                    audio: studioUpdates?.audio
                      ? {
                          ...slide.metadata.studio.audio,
                          ...studioUpdates.audio,
                          status: normalizeAssetStatus(
                            studioUpdates.audio.status ?? slide.metadata.studio.audio.status,
                            studioUpdates.audio.asset_ref ||
                              slide.metadata.studio.audio.asset_ref
                              ? "ready"
                              : "missing"
                          )
                        }
                      : slide.metadata.studio.audio,
                    image: studioUpdates?.image
                      ? {
                          ...slide.metadata.studio.image,
                          ...studioUpdates.image,
                          status: normalizeAssetStatus(
                            studioUpdates.image.status ?? slide.metadata.studio.image.status,
                            studioUpdates.image.asset_ref ||
                              slide.metadata.studio.image.asset_ref
                              ? "ready"
                              : "missing"
                          )
                        }
                      : slide.metadata.studio.image
                  }
                }
              : slide.metadata
          const nextSlide: PresentationStudioEditorSlide = {
            ...slide,
            ...updates,
            order: index,
            metadata: nextMetadata
          }
          if (
            Object.prototype.hasOwnProperty.call(updates, "speaker_notes") &&
            updates.speaker_notes !== slide.speaker_notes
          ) {
            nextSlide.metadata = {
              ...nextSlide.metadata,
              studio: {
                ...nextSlide.metadata.studio,
                audio: {
                  ...nextSlide.metadata.studio.audio,
                  status: "stale"
                }
              }
            }
          }
          return nextSlide
        })
        return {
          slides,
          isDirty: true
        }
      }),

    setAutosaveState: (autosaveState, autosaveError = null) =>
      set({ autosaveState, autosaveError }),

    markPersisted: (etag, project) => {
      if (project) {
        get().loadProject(project, {
          etag: etag ?? `W/"v${project.version}"`
        })
        return
      }
      set({
        etag: etag ?? get().etag,
        isDirty: false,
        autosaveState: "idle",
        autosaveError: null
      })
    },

    buildPatchPayload: () => {
      const state = get()
      return buildPatchPayloadFromSlides({
        title: state.title,
        description: state.description,
        theme: state.theme,
        visual_style_id: state.visualStyleId,
        visual_style_scope: state.visualStyleScope,
        visual_style_name: state.visualStyleName,
        visual_style_version: state.visualStyleVersion,
        visual_style_snapshot: state.visualStyleSnapshot,
        studio_data: state.studioData,
        slides: state.slides
      })
    }
  })
)
