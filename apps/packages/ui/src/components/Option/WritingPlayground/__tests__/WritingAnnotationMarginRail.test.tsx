import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { WritingAnnotationMarginRail } from "../WritingAnnotationMarginRail"
import { WritingAnnotationList } from "../WritingAnnotationList"
import type { WritingEditorAdapter } from "../writing-editor-adapter"
import type { ManuscriptAnnotationResponse } from "@/services/writing-playground"

const makeAnnotation = (
  overrides: Partial<ManuscriptAnnotationResponse> = {}
): ManuscriptAnnotationResponse => ({
  id: "annotation-1",
  project_id: "project-1",
  target_type: "scene",
  target_id: "scene-1",
  status: "open",
  category: "clarity",
  tags: [],
  source: "user",
  body: "Tighten this sentence.",
  suggested_fix: null,
  followup_note: null,
  metadata: {},
  scene_version: 4,
  anchor_start: 0,
  anchor_end: 7,
  selected_text: "Opening",
  anchor_status: "attached",
  derived_start: null,
  derived_end: null,
  scene_level: false,
  created_at: "2026-06-25T00:00:00Z",
  last_modified: "2026-06-25T00:00:00Z",
  deleted: false,
  client_id: "test",
  version: 1,
  ...overrides
})

const makeAdapter = (
  measurements: Record<string, { top: number; bottom: number; height: number } | null>
): WritingEditorAdapter => ({
  getSelection: () => ({ start: 0, end: 0 }),
  setSelection: vi.fn(),
  getSelectedText: () => "",
  focus: vi.fn(),
  measureRange: vi.fn((selection) => measurements[`${selection.start}:${selection.end}`] ?? null)
})

const renderRail = (
  props: Partial<React.ComponentProps<typeof WritingAnnotationMarginRail>> = {}
) => {
  const adapter =
    props.adapter ??
    makeAdapter({
      "0:7": { top: 40, bottom: 60, height: 20 }
    })

  return render(
    <WritingAnnotationMarginRail
      annotations={[makeAnnotation()]}
      adapter={adapter}
      activeAnnotationId={null}
      onActiveAnnotationChange={vi.fn()}
      {...props}
    />
  )
}

describe("WritingAnnotationMarginRail", () => {
  it("sorts cards by anchor top, then created_at, then id", () => {
    const annotations = [
      makeAnnotation({
        id: "third",
        anchor_start: 20,
        anchor_end: 25,
        created_at: "2026-06-25T00:00:03Z"
      }),
      makeAnnotation({
        id: "second",
        anchor_start: 10,
        anchor_end: 15,
        created_at: "2026-06-25T00:00:02Z"
      }),
      makeAnnotation({
        id: "first",
        anchor_start: 0,
        anchor_end: 5,
        created_at: "2026-06-25T00:00:02Z"
      })
    ]
    renderRail({
      annotations,
      adapter: makeAdapter({
        "0:5": { top: 25, bottom: 40, height: 15 },
        "10:15": { top: 25, bottom: 40, height: 15 },
        "20:25": { top: 10, bottom: 25, height: 15 }
      })
    })

    expect(screen.getAllByTestId("writing-annotation-margin-card")).toHaveLength(3)
    expect(screen.getAllByRole("button", { name: /Focus annotation/ }).map((button) => button.textContent)).toEqual([
      expect.stringContaining("third"),
      expect.stringContaining("first"),
      expect.stringContaining("second")
    ])
  })

  it("pushes later cards down by a fixed gap to avoid collisions", () => {
    renderRail({
      annotations: [
        makeAnnotation({ id: "top", anchor_start: 0, anchor_end: 5 }),
        makeAnnotation({ id: "pushed", anchor_start: 6, anchor_end: 10 })
      ],
      adapter: makeAdapter({
        "0:5": { top: 20, bottom: 30, height: 10 },
        "6:10": { top: 40, bottom: 50, height: 10 }
      })
    })

    const [top, pushed] = screen.getAllByTestId("writing-annotation-margin-card")
    expect(top).toHaveStyle({ top: "20px" })
    expect(pushed).toHaveStyle({ top: "124px" })
  })

  it("remeasures stable annotations when editor layout changes", () => {
    const measurements = {
      "0:7": { top: 40, bottom: 60, height: 20 }
    }
    const annotations = [makeAnnotation({ id: "layout-sensitive" })]
    const adapter = makeAdapter(measurements)
    const { rerender } = render(
      <WritingAnnotationMarginRail
        annotations={annotations}
        adapter={adapter}
        activeAnnotationId={null}
        onActiveAnnotationChange={vi.fn()}
        measurementVersion={0}
      />
    )

    expect(screen.getByTestId("writing-annotation-margin-card")).toHaveStyle({
      top: "40px"
    })

    measurements["0:7"] = { top: 88, bottom: 108, height: 20 }
    rerender(
      <WritingAnnotationMarginRail
        annotations={annotations}
        adapter={adapter}
        activeAnnotationId={null}
        onActiveAnnotationChange={vi.fn()}
        measurementVersion={1}
      />
    )

    expect(adapter.measureRange).toHaveBeenCalledTimes(2)
    expect(screen.getByTestId("writing-annotation-margin-card")).toHaveStyle({
      top: "88px"
    })
  })

  it("expands the active card and pushes following cards down", () => {
    renderRail({
      activeAnnotationId: "top",
      annotations: [
        makeAnnotation({ id: "top", anchor_start: 0, anchor_end: 5 }),
        makeAnnotation({ id: "pushed", anchor_start: 6, anchor_end: 10 })
      ],
      adapter: makeAdapter({
        "0:5": { top: 20, bottom: 30, height: 10 },
        "6:10": { top: 40, bottom: 50, height: 10 }
      })
    })

    const [, pushed] = screen.getAllByTestId("writing-annotation-margin-card")
    expect(pushed).toHaveStyle({ top: "180px" })
  })

  it("expands the active card with status, anchor state, follow-up, and suggested-fix action", () => {
    renderRail({
      activeAnnotationId: "active-card",
      annotations: [
        makeAnnotation({
          id: "active-card",
          anchor_start: 0,
          anchor_end: 7,
          followup_note: "Check the scene transition after this line.",
          suggested_fix: "Use a sharper verb here."
        })
      ],
      adapter: makeAdapter({
        "0:7": { top: 40, bottom: 60, height: 20 }
      })
    })

    expect(screen.getByTestId("writing-annotation-card")).toHaveClass("h-[152px]")
    expect(screen.getByTestId("writing-annotation-card").className).not.toContain("height")
    expect(screen.getByText("open")).toBeInTheDocument()
    expect(screen.getByText("attached")).toBeInTheDocument()
    expect(screen.getByText(/Follow-up: Check the scene transition/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create revision" })).toBeInTheDocument()
  })

  it("hands stable suggested fixes to the revision callback", () => {
    const onReviewSuggestedFix = vi.fn()
    const annotation = makeAnnotation({
      id: "fix-card",
      suggested_fix: "Use a sharper verb here."
    })
    renderRail({
      activeAnnotationId: "fix-card",
      annotations: [annotation],
      onReviewSuggestedFix
    })

    fireEvent.click(screen.getByRole("button", { name: "Create revision" }))

    expect(onReviewSuggestedFix).toHaveBeenCalledWith(annotation)
  })

  it("falls back to manual copy guidance when suggested-fix anchors need review", () => {
    const onReviewSuggestedFix = vi.fn()
    const onCopySuggestedFix = vi.fn()
    const annotation = makeAnnotation({
      id: "manual-card",
      anchor_status: "needs_review",
      suggested_fix: "Use a sharper verb here."
    })
    renderRail({
      activeAnnotationId: "manual-card",
      annotations: [annotation],
      onReviewSuggestedFix,
      onCopySuggestedFix
    })

    fireEvent.click(screen.getByRole("button", { name: "Copy fix manually" }))

    expect(onReviewSuggestedFix).not.toHaveBeenCalled()
    expect(onCopySuggestedFix).toHaveBeenCalledWith(annotation)
  })

  it("links margin cards to inspector rows with stable ids", () => {
    const annotation = makeAnnotation({ id: "linked-card" })
    render(
      <>
        <WritingAnnotationMarginRail
          annotations={[annotation]}
          adapter={makeAdapter({
            "0:7": { top: 40, bottom: 60, height: 20 }
          })}
          activeAnnotationId={null}
          onActiveAnnotationChange={vi.fn()}
        />
        <WritingAnnotationList
          annotations={[annotation]}
          onUpdate={vi.fn()}
          onDelete={vi.fn()}
        />
      </>
    )

    expect(screen.getByTestId("writing-annotation-card")).toHaveAttribute(
      "id",
      "writing-annotation-margin-card-linked-card"
    )
    expect(screen.getByRole("button", { name: "Focus annotation linked-card" })).toHaveAttribute(
      "aria-controls",
      "writing-annotation-inspector-row-linked-card"
    )
    expect(screen.getByTestId("writing-annotation-inspector-row")).toHaveAttribute(
      "id",
      "writing-annotation-inspector-row-linked-card"
    )
  })

  it("hides when measurement is unavailable", () => {
    const { container } = renderRail({
      adapter: makeAdapter({ "0:7": null })
    })

    expect(container.firstChild).toBeNull()
  })

  it("excludes resolved comments by default", () => {
    renderRail({
      annotations: [
        makeAnnotation({ id: "open", status: "open" }),
        makeAnnotation({ id: "resolved", status: "resolved", anchor_start: 10, anchor_end: 15 })
      ],
      adapter: makeAdapter({
        "0:7": { top: 10, bottom: 20, height: 10 },
        "10:15": { top: 30, bottom: 40, height: 10 }
      })
    })

    expect(screen.getAllByTestId("writing-annotation-margin-card")).toHaveLength(1)
    expect(screen.queryByText("resolved")).not.toBeInTheDocument()
  })

  it("syncs focus actions back to the editor selection", () => {
    const adapter = makeAdapter({
      "2:8": { top: 40, bottom: 60, height: 20 }
    })
    renderRail({
      annotations: [makeAnnotation({ id: "focus-me", anchor_start: 2, anchor_end: 8 })],
      adapter
    })

    fireEvent.click(screen.getByRole("button", { name: "Focus annotation focus-me" }))

    expect(adapter.focus).toHaveBeenCalled()
    expect(adapter.setSelection).toHaveBeenCalledWith({ start: 2, end: 8 })
  })

  it("marks needs_review cards as unattached warnings when measured", () => {
    renderRail({
      annotations: [
        makeAnnotation({
          id: "review-me",
          anchor_status: "needs_review",
          anchor_start: 2,
          anchor_end: 8
        })
      ],
      adapter: makeAdapter({
        "2:8": { top: 40, bottom: 60, height: 20 }
      })
    })

    expect(screen.getByText("needs review")).toBeInTheDocument()
    expect(screen.getByTestId("writing-annotation-margin-card")).toHaveAttribute(
      "data-anchor-status",
      "needs_review"
    )
  })
})
