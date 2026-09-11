import { describe, expect, it } from "vitest"
import {
  createPromptAssistInitialState,
  reducePromptAssist,
  type PromptAssistOperation
} from "../prompt-assist-state"
import type { PromptImproveResponse } from "@/services/prompt-improvement"

const route = {
  selected_model: "openai/gpt-5-mini",
  provider_hint: "openai"
}

const operation = (overrides: Partial<PromptAssistOperation> = {}): PromptAssistOperation => ({
  operationId: "11111111-1111-4111-8111-111111111111",
  target: "system",
  mode: "improve_now",
  originalText: "Be helpful.",
  revision: "r1",
  route,
  ...overrides
})

const response = (overrides: Partial<PromptImproveResponse> = {}): PromptImproveResponse => ({
  schema_version: 1,
  operation_id: "11111111-1111-4111-8111-111111111111",
  status: "improved",
  improved_text: "Be helpful and concise.",
  findings: [],
  review_required: false,
  warnings: [],
  resolved_model: {
    provider: "openai",
    model: "gpt-5-mini",
    display_name: "GPT-5 mini"
  },
  meta_prompt_version: "prompt-improvement-v1",
  ...overrides
})

const start = (overrides: Partial<PromptAssistOperation> = {}) =>
  reducePromptAssist(createPromptAssistInitialState(), {
    type: "request_started",
    operation: operation(overrides)
  })

describe("prompt assist state", () => {
  it("captures the request operation and enters analyzing", () => {
    expect(start()).toMatchObject({
      status: "analyzing",
      operation: operation(),
      undo: null
    })
  })

  it("ignores a response superseded by a newer operation", () => {
    const analyzing = start()
    expect(
      reducePromptAssist(analyzing, {
        type: "response_received",
        operationId: "older",
        response: response({ operation_id: "older" }),
        liveText: "Be helpful.",
        liveRevision: "r1",
        liveRoute: route,
        autoApplied: false
      })
    ).toEqual(analyzing)
  })

  it("records an eligible automatic application and exact undo snapshot", () => {
    const undoSnapshot = { override: undefined as string | undefined }
    const applied = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: true,
      undoSnapshot
    })

    expect(applied).toMatchObject({
      status: "applied",
      candidate: "Be helpful and concise.",
      undo: { target: "system", snapshot: undoSnapshot }
    })
  })

  it.each([
    ["draft_changed", "Edited live", "r2", route],
    [
      "route_changed",
      "Be helpful.",
      "r1",
      { selected_model: "anthropic/claude", provider_hint: "anthropic" }
    ]
  ] as const)("forces review when %s", (notice, liveText, liveRevision, liveRoute) => {
    const reviewing = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText,
      liveRevision,
      liveRoute,
      autoApplied: false
    })

    expect(reviewing).toMatchObject({
      status: "reviewing",
      notice,
      candidate: "Be helpful and concise."
    })
  })

  it("always sends review mode and server review-required results to review", () => {
    const manual = reducePromptAssist(start({ mode: "review_changes" }), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })
    const forced = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response({ review_required: true }),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })

    expect(manual).toMatchObject({ status: "reviewing", notice: null })
    expect(forced).toMatchObject({
      status: "reviewing",
      notice: "review_required"
    })
  })

  it("reports no change without a candidate or mutation state", () => {
    const noChange = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response({ status: "no_change", improved_text: null }),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })

    expect(noChange).toEqual({
      status: "idle",
      notice: "no_change",
      undo: null
    })
  })

  it("records stable failures including preservation failure", () => {
    const failed = reducePromptAssist(start(), {
      type: "request_failed",
      operationId: operation().operationId,
      error: {
        code: "preservation_failed",
        message: "The candidate could not be preserved safely.",
        retryable: false
      }
    })

    expect(failed).toMatchObject({
      status: "failed",
      error: { code: "preservation_failed", retryable: false }
    })
  })

  it("edits a review candidate without changing the captured original", () => {
    const reviewing = reducePromptAssist(start({ mode: "review_changes" }), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })
    const edited = reducePromptAssist(reviewing, {
      type: "candidate_edited",
      candidate: "My reviewed candidate"
    })

    expect(edited).toMatchObject({
      status: "reviewing",
      candidate: "My reviewed candidate",
      operation: { originalText: "Be helpful." }
    })
  })

  it("blocks normal Apply after a live edit and requires explicit replacement", () => {
    const reviewing = reducePromptAssist(start({ mode: "review_changes" }), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })
    const blocked = reducePromptAssist(reviewing, {
      type: "review_apply_requested",
      liveText: "New live text",
      fresh: false,
      applied: false
    })

    expect(blocked).toMatchObject({
      status: "reviewing",
      notice: "draft_changed",
      replaceConfirmationRequired: true
    })
  })

  it("records normal Apply and confirmed replacement snapshots", () => {
    const reviewing = reducePromptAssist(start({ mode: "review_changes" }), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: false
    })
    if (reviewing.status !== "reviewing") {
      throw new Error("expected review state")
    }
    const normalSnapshot = { override: "old" }
    const applied = reducePromptAssist(reviewing, {
      type: "review_apply_requested",
      liveText: "Be helpful.",
      fresh: true,
      applied: true,
      undoSnapshot: normalSnapshot
    })
    const replacementSnapshot = { override: "new live state" }
    const replaced = reducePromptAssist(
      {
        ...reviewing,
        notice: "draft_changed",
        replaceConfirmationRequired: true
      },
      {
        type: "replace_confirmed",
        undoSnapshot: replacementSnapshot
      }
    )

    expect(applied).toMatchObject({
      status: "applied",
      undo: { snapshot: normalSnapshot }
    })
    expect(replaced).toMatchObject({
      status: "applied",
      undo: { snapshot: replacementSnapshot }
    })
  })

  it("replaces one-step undo on a new apply and clears it for lifecycle events", () => {
    const first = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: true,
      undoSnapshot: { raw: "first" }
    })
    const secondStart = reducePromptAssist(first, {
      type: "request_started",
      operation: operation({ operationId: "22222222-2222-4222-8222-222222222222" })
    })
    const second = reducePromptAssist(secondStart, {
      type: "response_received",
      operationId: "22222222-2222-4222-8222-222222222222",
      response: response({
        operation_id: "22222222-2222-4222-8222-222222222222",
        improved_text: "Second candidate"
      }),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: true,
      undoSnapshot: { raw: "second" }
    })

    expect(second).toMatchObject({ undo: { snapshot: { raw: "second" } } })
    expect(
      reducePromptAssist(second, { type: "target_edited" })
    ).toMatchObject({ undo: null })
    expect(
      reducePromptAssist(second, { type: "lifecycle_cleared" })
    ).toEqual(createPromptAssistInitialState())
  })

  it("keeps the exact opaque raw system snapshot for restoration", () => {
    const rawSystemState = { override: undefined, selectedPromptId: "saved-1" }
    const applied = reducePromptAssist(start(), {
      type: "response_received",
      operationId: operation().operationId,
      response: response(),
      liveText: "Be helpful.",
      liveRevision: "r1",
      liveRoute: route,
      autoApplied: true,
      undoSnapshot: rawSystemState
    })

    expect(applied.undo?.snapshot).toBe(rawSystemState)
  })
})
