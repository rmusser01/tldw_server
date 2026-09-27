import { beforeEach, describe, expect, it } from "vitest"

import {
  mergePresentationStudioDraftWithRemote,
  type PresentationStudioPatchPayload,
  usePresentationStudioStore
} from "@/store/presentation-studio"

const sampleProject = {
  id: "pres-123",
  content_kind: "structured_slides" as const,
  title: "Deck",
  description: null,
  theme: "black",
  marp_theme: null,
  template_id: null,
  visual_style_id: "minimal-academic",
  visual_style_scope: "builtin",
  visual_style_name: "Minimal Academic",
  visual_style_version: 1,
  visual_style_snapshot: {
    id: "minimal-academic",
    scope: "builtin",
    name: "Minimal Academic",
    category: "Educational and Explainer",
    guide_number: 1,
    tags: ["study", "notes"],
    best_for: ["exam prep", "course notes"],
    appearance_defaults: { theme: "white" }
  },
  settings: null,
  studio_data: { origin: "blank" },
  slides: [
    {
      order: 0,
      layout: "title",
      title: "Intro",
      content: "Welcome",
      speaker_notes: "Original narration",
      metadata: {
        studio: {
          slideId: "slide-1",
          audio: {
            status: "ready",
            asset_ref: "output:1"
          }
        }
      }
    }
  ],
  custom_css: null,
  source_type: "manual",
  source_ref: null,
  source_query: null,
  created_at: "2026-03-13T00:00:00Z",
  last_modified: "2026-03-13T00:00:00Z",
  deleted: false,
  client_id: "1",
  version: 1
}

const sequenceProject = {
  ...sampleProject,
  slides: [
    sampleProject.slides[0],
    {
      order: 1,
      layout: "content",
      title: "Problem",
      content: "Explain the current gap.",
      speaker_notes: "Describe the pain point.",
      metadata: {
        studio: {
          slideId: "slide-2",
          audio: {
            status: "stale"
          },
          image: {
            status: "missing"
          }
        }
      }
    },
    {
      order: 2,
      layout: "content",
      title: "Outcome",
      content: "Describe the result.",
      speaker_notes: "Close with the value.",
      metadata: {
        studio: {
          slideId: "slide-3",
          audio: {
            status: "ready",
            asset_ref: "output:3"
          },
          image: {
            status: "ready",
            asset_ref: "output:4"
          }
        }
      }
    }
  ]
}

describe("presentation studio store", () => {
  beforeEach(() => {
    usePresentationStudioStore.getState().reset()
  })

  it("initializes blank slides with default transition and auto timing", () => {
    const store = usePresentationStudioStore.getState()

    store.initializeBlankProject()

    const slide = usePresentationStudioStore.getState().slides[0]
    expect(slide?.metadata?.studio?.transition).toBe("fade")
    expect(slide?.metadata?.studio?.timing_mode).toBe("auto")
    expect(slide?.metadata?.studio?.manual_duration_ms).toBeNull()
  })

  it("marks audio stale when speaker_notes change", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)
    store.updateSlide("slide-1", { speaker_notes: "New narration" })

    expect(usePresentationStudioStore.getState().slides[0]?.metadata?.studio?.audio?.status).toBe(
      "stale"
    )
  })

  it("loads and persists presentation-level visual style metadata", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)

    let state = usePresentationStudioStore.getState()
    expect(state.visualStyleId).toBe("minimal-academic")
    expect(state.visualStyleScope).toBe("builtin")
    expect(state.visualStyleName).toBe("Minimal Academic")
    expect(state.visualStyleVersion).toBe(1)
    expect(state.visualStyleSnapshot).toEqual(
      expect.objectContaining({
        id: "minimal-academic",
        name: "Minimal Academic",
        category: "Educational and Explainer",
        guide_number: 1,
        tags: ["study", "notes"],
        best_for: ["exam prep", "course notes"]
      })
    )

    store.updateProjectMeta({
      visualStyleId: "timeline",
      visualStyleScope: "builtin",
      visualStyleName: "Timeline",
      visualStyleVersion: 1,
      visualStyleSnapshot: {
        id: "timeline",
        scope: "builtin",
        name: "Timeline",
        appearance_defaults: { theme: "beige" }
      }
    })

    state = usePresentationStudioStore.getState()
    expect(state.visualStyleId).toBe("timeline")
    expect(state.visualStyleScope).toBe("builtin")
    expect(state.visualStyleName).toBe("Timeline")
    expect(state.visualStyleSnapshot).toEqual(
      expect.objectContaining({
        id: "timeline",
        appearance_defaults: { theme: "beige" }
      })
    )

    expect(state.buildPatchPayload()).toEqual(
      expect.objectContaining({
        visual_style_id: "timeline",
        visual_style_scope: "builtin",
        visual_style_name: "Timeline",
        visual_style_version: 1,
        visual_style_snapshot: expect.objectContaining({
          id: "timeline"
        })
      })
    )
  })

  it("preserves explicit visual-style clears when merging remote updates", () => {
    const localDraft: PresentationStudioPatchPayload = {
      title: sampleProject.title,
      description: sampleProject.description,
      theme: sampleProject.theme,
      visual_style_id: null,
      visual_style_scope: null,
      visual_style_name: null,
      visual_style_version: null,
      visual_style_snapshot: null,
      studio_data: sampleProject.studio_data,
      slides: sampleProject.slides
    }

    const merged = mergePresentationStudioDraftWithRemote(sampleProject, localDraft)

    expect(merged.visual_style_id).toBeNull()
    expect(merged.visual_style_scope).toBeNull()
    expect(merged.visual_style_name).toBeNull()
    expect(merged.visual_style_version).toBeNull()
    expect(merged.visual_style_snapshot).toBeNull()
  })

  it("duplicates and removes slides while preserving selection and order", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)

    store.duplicateSlide("slide-1")

    const duplicatedSlides = usePresentationStudioStore.getState().slides
    expect(duplicatedSlides).toHaveLength(2)
    expect(duplicatedSlides.map((slide) => slide.order)).toEqual([0, 1])
    expect(duplicatedSlides[1]?.title).toBe("Intro copy")
    expect(duplicatedSlides[1]?.metadata?.studio?.slideId).not.toBe("slide-1")
    expect(usePresentationStudioStore.getState().selectedSlideId).toBe(
      duplicatedSlides[1]?.metadata?.studio?.slideId
    )

    store.removeSlide(duplicatedSlides[1]!.metadata.studio.slideId)

    const remainingSlides = usePresentationStudioStore.getState().slides
    expect(remainingSlides).toHaveLength(1)
    expect(remainingSlides[0]?.metadata?.studio?.slideId).toBe("slide-1")
    expect(usePresentationStudioStore.getState().selectedSlideId).toBe("slide-1")
  })

  it("moves slides earlier and later while keeping selection on the moved slide", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sequenceProject)
    store.selectSlide("slide-2")

    store.moveSlide("slide-2", "earlier")

    let slides = usePresentationStudioStore.getState().slides
    expect(slides.map((slide) => slide.metadata.studio.slideId)).toEqual([
      "slide-2",
      "slide-1",
      "slide-3"
    ])
    expect(slides.map((slide) => slide.order)).toEqual([0, 1, 2])
    expect(usePresentationStudioStore.getState().selectedSlideId).toBe("slide-2")

    store.moveSlide("slide-2", "later")
    store.moveSlide("slide-2", "later")

    slides = usePresentationStudioStore.getState().slides
    expect(slides.map((slide) => slide.metadata.studio.slideId)).toEqual([
      "slide-1",
      "slide-3",
      "slide-2"
    ])
    expect(slides.map((slide) => slide.order)).toEqual([0, 1, 2])
    expect(usePresentationStudioStore.getState().selectedSlideId).toBe("slide-2")
  })

  it("reorders slides directly between arbitrary indexes", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sequenceProject)

    store.reorderSlides(0, 2)

    const slides = usePresentationStudioStore.getState().slides
    expect(slides.map((slide) => slide.metadata.studio.slideId)).toEqual([
      "slide-2",
      "slide-3",
      "slide-1"
    ])
    expect(slides.map((slide) => slide.order)).toEqual([0, 1, 2])
  })

  it("preserves manual timing and transition metadata on slide updates", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)

    store.updateSlide("slide-1", {
      metadata: {
        studio: {
          transition: "wipe",
          timing_mode: "manual",
          manual_duration_ms: 45_000
        } as any
      }
    })

    const slide = usePresentationStudioStore.getState().slides[0]
    expect(slide?.metadata?.studio?.transition).toBe("wipe")
    expect(slide?.metadata?.studio?.timing_mode).toBe("manual")
    expect(slide?.metadata?.studio?.manual_duration_ms).toBe(45_000)
  })

  it("keeps manual timing selected while the duration is being authored", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)

    store.updateSlide("slide-1", {
      metadata: {
        studio: {
          timing_mode: "manual"
        } as any
      }
    })

    const slide = usePresentationStudioStore.getState().slides[0]
    expect(slide?.metadata?.studio?.timing_mode).toBe("manual")
    expect(slide?.metadata?.studio?.manual_duration_ms).toBeNull()
  })

  it("switches back to auto timing when manual duration is cleared", () => {
    const store = usePresentationStudioStore.getState()
    store.loadProject(sampleProject)

    store.updateSlide("slide-1", {
      metadata: {
        studio: {
          timing_mode: "manual",
          manual_duration_ms: 45_000
        } as any
      }
    })
    store.updateSlide("slide-1", {
      metadata: {
        studio: {
          manual_duration_ms: null
        } as any
      }
    })

    const slide = usePresentationStudioStore.getState().slides[0]
    expect(slide?.metadata?.studio?.manual_duration_ms).toBeNull()
    expect(slide?.metadata?.studio?.timing_mode).toBe("auto")
  })
})
