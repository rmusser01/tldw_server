import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

import {
  createManuscriptAnnotation,
  deleteManuscriptAnnotation,
  getManuscriptAnnotation,
  listManuscriptAnnotations,
  reviewManuscriptScene,
  reviewManuscriptSelection,
  updateManuscriptAnnotation,
  type ManuscriptAnnotationCreateInput,
  type ManuscriptAnnotationListFilters,
  type ManuscriptAnnotationResponse,
  type ManuscriptAnnotationUpdateInput,
  type ManuscriptSceneAnnotationReviewRequest,
  type ManuscriptSceneAnnotationReviewJobResponse,
  type ManuscriptSelectedTextAnnotationReviewRequest
} from "@/services/writing-playground"

const annotation: ManuscriptAnnotationResponse = {
  id: "annotation-1",
  project_id: "project-1",
  target_type: "scene",
  target_id: "scene-1",
  status: "open",
  category: "clarity",
  tags: ["dialogue"],
  source: "user",
  body: "Clarify the speaker's intent.",
  suggested_fix: "Add a direct reaction beat.",
  followup_note: null,
  metadata: { severity: "medium" },
  scene_version: 7,
  anchor_start: 12,
  anchor_end: 28,
  selected_text: "the selected text",
  anchor_status: "attached",
  derived_start: 12,
  derived_end: 28,
  scene_level: false,
  created_at: "2026-06-23T10:00:00Z",
  last_modified: "2026-06-23T10:05:00Z",
  deleted: false,
  client_id: "client-1",
  version: 3
}

const expectNoAlternateProviderFields = (body: unknown) => {
  expect(body).not.toHaveProperty("api_provider")
  expect(body).not.toHaveProperty("apiProvider")
  expect(body).not.toHaveProperty("llm_provider")
  expect(body).not.toHaveProperty("llmProvider")
  expect(body).not.toHaveProperty("provider_name")
  expect(body).not.toHaveProperty("providerName")
}

describe("writing-playground manuscript annotation service wiring", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("lists project annotations with exact filter query parameters", async () => {
    const response = {
      annotations: [annotation],
      total: 1,
      limit: 25,
      offset: 5,
      has_more: false,
      next_offset: null,
      pagination: {
        mode: "offset" as const,
        limit: 25,
        offset: 5,
        total: 1,
        has_more: false,
        next_offset: null
      }
    }
    const filters: ManuscriptAnnotationListFilters = {
      target_type: "scene",
      target_id: "scene-1",
      status: "open",
      category: "clarity",
      source: "user",
      anchor_status: "attached",
      limit: 25,
      offset: 5
    }
    mocks.bgRequest.mockResolvedValueOnce(response)

    const result = await listManuscriptAnnotations("project-1", filters)

    expect(result).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/projects/project-1/annotations?target_type=scene&target_id=scene-1&status=open&category=clarity&source=user&anchor_status=attached&limit=25&offset=5",
      method: "GET"
    })
  })

  it("creates a user annotation without sending a source field", async () => {
    const input: ManuscriptAnnotationCreateInput = {
      target_type: "scene",
      target_id: "scene-1",
      category: "clarity",
      body: "Clarify the speaker's intent.",
      tags: ["dialogue"],
      suggested_fix: "Add a direct reaction beat.",
      followup_note: null,
      metadata: { severity: "medium" },
      scene_version: 7,
      start: 12,
      end: 28,
      selected_text: "the selected text"
    }
    mocks.bgRequest.mockResolvedValueOnce(annotation)

    const result = await createManuscriptAnnotation(input)

    expect(result).toEqual(annotation)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/annotations",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: input
    })
    expect(mocks.bgRequest.mock.calls[0]?.[0]?.body).not.toHaveProperty("source")
  })

  it("gets one annotation by id", async () => {
    mocks.bgRequest.mockResolvedValueOnce(annotation)

    const result = await getManuscriptAnnotation("annotation-1")

    expect(result).toEqual(annotation)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/annotations/annotation-1",
      method: "GET"
    })
  })

  it("updates annotation fields with the expected version header", async () => {
    const input: ManuscriptAnnotationUpdateInput = {
      status: "resolved",
      category: "style",
      body: "Resolved after rewrite.",
      tags: [],
      suggested_fix: null,
      followup_note: "Check the next pass.",
      metadata: { reviewed: true }
    }
    mocks.bgRequest.mockResolvedValueOnce({ ...annotation, ...input, version: 4 })

    const result = await updateManuscriptAnnotation("annotation-1", input, 3)

    expect(result.version).toBe(4)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/annotations/annotation-1",
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        "expected-version": "3"
      },
      body: input
    })
  })

  it("deletes an annotation with the expected version header", async () => {
    mocks.bgRequest.mockResolvedValueOnce(undefined)

    await deleteManuscriptAnnotation("annotation-1", 3)

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/annotations/annotation-1",
      method: "DELETE",
      headers: {
        "expected-version": "3"
      }
    })
  })

  it("reviews selected manuscript text with provider and model only", async () => {
    const input: ManuscriptSelectedTextAnnotationReviewRequest = {
      provider: "openai",
      model: "gpt-4.1-mini",
      scene_version: 7,
      start: 12,
      end: 28,
      selected_text: "the selected text",
      category_hints: ["clarity", "style"],
      instruction: "Focus on dialogue clarity."
    }
    mocks.bgRequest.mockResolvedValueOnce({
      ...annotation,
      source: "ai_selected_text"
    })

    const result = await reviewManuscriptSelection("scene-1", input)

    expect(result.source).toBe("ai_selected_text")
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/scenes/scene-1/annotations/review-selection",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: input
    })
    expect(mocks.bgRequest.mock.calls[0]?.[0]?.body).toHaveProperty("provider", "openai")
    expect(mocks.bgRequest.mock.calls[0]?.[0]?.body).toHaveProperty("model", "gpt-4.1-mini")
    expectNoAlternateProviderFields(mocks.bgRequest.mock.calls[0]?.[0]?.body)
  })

  it("queues a scene review job with provider and model only", async () => {
    const input: ManuscriptSceneAnnotationReviewRequest = {
      provider: "anthropic",
      model: "claude-sonnet-4-5",
      scene_version: 7,
      max_comments: 5,
      category_filters: ["continuity", "character"],
      review_focus: "Look for continuity breaks."
    }
    const response: ManuscriptSceneAnnotationReviewJobResponse = {
      job_id: 42,
      job_uuid: "job-uuid-42",
      status: "queued",
      job_type: "writing_scene_annotation_review",
      project_id: "project-1",
      scene_id: "scene-1",
      scene_version: 7
    }
    mocks.bgRequest.mockResolvedValueOnce(response)

    const result = await reviewManuscriptScene("scene-1", input)

    expect(result).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/writing/manuscripts/scenes/scene-1/annotations/review-scene",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: input
    })
    expect(mocks.bgRequest.mock.calls[0]?.[0]?.body).toHaveProperty("provider", "anthropic")
    expect(mocks.bgRequest.mock.calls[0]?.[0]?.body).toHaveProperty("model", "claude-sonnet-4-5")
    expectNoAlternateProviderFields(mocks.bgRequest.mock.calls[0]?.[0]?.body)
  })
})
