import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const messageError = vi.hoisted(() => vi.fn())

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      ...actual.message,
      error: messageError
    }
  }
})

import { WritingAnnotationsTab } from "../WritingAnnotationsTab"
import type { UseWritingAnnotationsResult } from "../hooks/useWritingAnnotations"
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

const callbacks = {
  createAnnotation: vi.fn(),
  updateAnnotation: vi.fn(),
  deleteAnnotation: vi.fn(),
  reviewSelection: vi.fn(),
  reviewScene: vi.fn()
}

const baseHookResult = (
  annotations: ManuscriptAnnotationResponse[] = []
): UseWritingAnnotationsResult => ({
  annotations,
  isLoading: false,
  isFetching: false,
  error: null,
  createAnnotation: callbacks.createAnnotation,
  updateAnnotation: callbacks.updateAnnotation,
  deleteAnnotation: callbacks.deleteAnnotation,
  reviewSelection: callbacks.reviewSelection,
  reviewScene: callbacks.reviewScene,
  isCreating: false,
  isUpdating: false,
  isDeleting: false,
  isReviewingSelection: false,
  isReviewingScene: false
})

const renderTab = (
  overrides: Partial<React.ComponentProps<typeof WritingAnnotationsTab>> = {}
) =>
  render(
    <WritingAnnotationsTab
      annotationsHook={baseHookResult()}
      projectId="project-1"
      activeChapterId="chapter-1"
      activeSceneId="scene-1"
      activeSceneVersion={4}
      activeSceneText="Opening 😀 line"
      selectedModel="gpt-test"
      apiProvider="openai"
      selection={{ start: 0, end: 10 }}
      canCreateRangeAnnotation
      isSceneDirty={false}
      {...overrides}
    />
  )

beforeEach(() => {
  vi.clearAllMocks()
  messageError.mockClear()
  callbacks.createAnnotation.mockResolvedValue(makeAnnotation())
  callbacks.updateAnnotation.mockResolvedValue(makeAnnotation())
  callbacks.deleteAnnotation.mockResolvedValue(undefined)
  callbacks.reviewSelection.mockResolvedValue(makeAnnotation({ source: "ai_selected_text" }))
  callbacks.reviewScene.mockResolvedValue({ job_id: 1, status: "queued" })
})

describe("WritingAnnotationsTab", () => {
  it("requires a saved scene binding for scene range comments", () => {
    renderTab({ canCreateRangeAnnotation: false, isSceneDirty: true })

    expect(
      screen.getByRole("button", { name: "Add range comment" })
    ).toBeDisabled()
    expect(screen.getByText(/save the selected scene/i)).toBeInTheDocument()
  })

  it("creates chapter and project notes without a range selection", async () => {
    renderTab({ selection: null, canCreateRangeAnnotation: false })

    fireEvent.change(screen.getByLabelText("Annotation body"), {
      target: { value: "Chapter-level pacing note." }
    })
    fireEvent.click(screen.getByRole("radio", { name: "Chapter" }))
    fireEvent.click(screen.getByRole("button", { name: "Add note" }))

    await waitFor(() => {
      expect(callbacks.createAnnotation).toHaveBeenCalledWith(
        expect.objectContaining({
          target_type: "chapter",
          target_id: "chapter-1",
          body: "Chapter-level pacing note."
        })
      )
    })

    fireEvent.change(screen.getByLabelText("Annotation body"), {
      target: { value: "Project-level continuity note." }
    })
    fireEvent.click(screen.getByRole("radio", { name: "Project" }))
    fireEvent.click(screen.getByRole("button", { name: "Add note" }))

    await waitFor(() => {
      expect(callbacks.createAnnotation).toHaveBeenLastCalledWith(
        expect.objectContaining({
          target_type: "project",
          target_id: "project-1",
          body: "Project-level continuity note."
        })
      )
    })
  })

  it("calls resolve, reopen, and update callbacks from list actions", async () => {
    renderTab({
      annotationsHook: baseHookResult([
        makeAnnotation({ id: "open-note", status: "open", version: 2 }),
        makeAnnotation({ id: "resolved-note", status: "resolved", version: 3 })
      ])
    })

    fireEvent.click(screen.getByRole("button", { name: "Resolve open-note" }))
    fireEvent.click(screen.getByRole("button", { name: "Reopen resolved-note" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit open-note" }))
    fireEvent.change(screen.getByLabelText("Edit annotation body"), {
      target: { value: "Updated annotation body" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save open-note" }))

    await waitFor(() => {
      expect(callbacks.updateAnnotation).toHaveBeenCalledWith(
        "open-note",
        { status: "resolved" },
        2
      )
      expect(callbacks.updateAnnotation).toHaveBeenCalledWith(
        "resolved-note",
        { status: "open" },
        3
      )
      expect(callbacks.updateAnnotation).toHaveBeenCalledWith(
        "open-note",
        { body: "Updated annotation body" },
        2
      )
    })
  })

  it("routes rejected list actions through the annotation error handler", async () => {
    callbacks.updateAnnotation.mockRejectedValue(new Error("Version conflict"))
    renderTab({
      annotationsHook: baseHookResult([
        makeAnnotation({ id: "open-note", status: "open", version: 2 })
      ])
    })

    fireEvent.click(screen.getByRole("button", { name: "Resolve open-note" }))

    await waitFor(() => {
      expect(messageError).toHaveBeenCalledWith("Version conflict")
    })
  })

  it("shows needs_review anchor status in rows", () => {
    renderTab({
      annotationsHook: baseHookResult([
        makeAnnotation({ anchor_status: "needs_review" })
      ])
    })

    expect(screen.getByText("needs_review")).toBeInTheDocument()
  })

  it("disables AI review actions when provider or model is unavailable", () => {
    renderTab({ selectedModel: "", apiProvider: undefined })

    expect(
      screen.getByRole("button", { name: "Review selection with AI" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Review scene with AI" })
    ).toBeDisabled()
  })

  it("keeps selected-text AI review disabled while the scene has unsaved edits", () => {
    renderTab({ isSceneDirty: true })

    expect(
      screen.getByRole("button", { name: "Review selection with AI" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Review scene with AI" })
    ).toBeDisabled()
  })

  it("keeps AI selected-text review disabled for invalid selections", () => {
    renderTab({
      selection: { start: 0, end: 0 },
      selectedModel: "gpt-test",
      apiProvider: "openai"
    })

    expect(
      screen.getByRole("button", { name: "Review selection with AI" })
    ).toBeDisabled()
  })

  it("reviews selected text with provider, model, scene version, offsets, and text", async () => {
    renderTab({
      activeSceneText: "Opening 😀 line",
      selection: { start: 0, end: 10 },
      selectedModel: "gpt-test",
      apiProvider: "openai",
      activeSceneVersion: 4
    })

    fireEvent.click(screen.getByRole("button", { name: "Review selection with AI" }))

    await waitFor(() => {
      expect(callbacks.reviewSelection).toHaveBeenCalledWith({
        sceneId: "scene-1",
        provider: "openai",
        model: "gpt-test",
        scene_version: 4,
        start: 0,
        end: 9,
        selected_text: "Opening 😀",
        category_hints: ["other"]
      })
    })
  })

  it("shows the queued scene review job id and status", async () => {
    callbacks.reviewScene.mockResolvedValue({ job_id: 77, status: "queued" })
    renderTab({
      selectedModel: "gpt-test",
      apiProvider: "openai",
      activeSceneVersion: 4
    })

    fireEvent.click(screen.getByRole("button", { name: "Review scene with AI" }))

    await waitFor(() => {
      expect(callbacks.reviewScene).toHaveBeenCalledWith({
        sceneId: "scene-1",
        provider: "openai",
        model: "gpt-test",
        scene_version: 4,
        max_comments: 8,
        category_filters: ["other"]
      })
    })
    expect(screen.getByText("Scene review job 77 queued")).toBeInTheDocument()
  })

  it("routes rejected annotation creation through visible error messaging", async () => {
    callbacks.createAnnotation.mockRejectedValue(new Error("Create failed"))
    renderTab()

    fireEvent.change(screen.getByLabelText("Annotation body"), {
      target: { value: "Scene range note." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add range comment" }))

    await waitFor(() => {
      expect(messageError).toHaveBeenCalledWith("Create failed")
    })
  })

  it("clears queued scene review job state when the scene version changes", async () => {
    callbacks.reviewScene.mockResolvedValue({ job_id: 88, status: "queued" })
    const view = renderTab({
      selectedModel: "gpt-test",
      apiProvider: "openai",
      activeSceneVersion: 4
    })

    fireEvent.click(screen.getByRole("button", { name: "Review scene with AI" }))

    await waitFor(() => {
      expect(screen.getByText("Scene review job 88 queued")).toBeInTheDocument()
    })
    view.rerender(
      <WritingAnnotationsTab
        annotationsHook={baseHookResult()}
        projectId="project-1"
        activeChapterId="chapter-1"
        activeSceneId="scene-1"
        activeSceneVersion={5}
        activeSceneText="Opening 😀 line"
        selectedModel="gpt-test"
        apiProvider="openai"
        selection={{ start: 0, end: 10 }}
        canCreateRangeAnnotation
        isSceneDirty={false}
      />
    )

    expect(screen.queryByText("Scene review job 88 queued")).not.toBeInTheDocument()
  })

  it("ignores queued scene review jobs that resolve after the scene version changes", async () => {
    let resolveReview: (job: { job_id: number; status: "queued" }) => void = () => {}
    callbacks.reviewScene.mockReturnValue(
      new Promise((resolve) => {
        resolveReview = resolve
      })
    )
    const view = renderTab({
      selectedModel: "gpt-test",
      apiProvider: "openai",
      activeSceneVersion: 4
    })

    fireEvent.click(screen.getByRole("button", { name: "Review scene with AI" }))
    view.rerender(
      <WritingAnnotationsTab
        annotationsHook={baseHookResult()}
        projectId="project-1"
        activeChapterId="chapter-1"
        activeSceneId="scene-1"
        activeSceneVersion={5}
        activeSceneText="Opening 😀 line"
        selectedModel="gpt-test"
        apiProvider="openai"
        selection={{ start: 0, end: 10 }}
        canCreateRangeAnnotation
        isSceneDirty={false}
      />
    )

    await act(async () => {
      resolveReview({ job_id: 99, status: "queued" })
    })

    expect(screen.queryByText("Scene review job 99 queued")).not.toBeInTheDocument()
  })
})
