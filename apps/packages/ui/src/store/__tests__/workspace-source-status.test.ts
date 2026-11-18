import { describe, expect, it } from "vitest"
import * as status from "../workspace-source-status"
import type { WorkspaceSource } from "@/types/workspace"

describe("source status availability", () => {
  it.each([
    new TypeError("Failed to fetch"),
    new Error("fetch failed"),
    new Error("Connection refused"),
    new Error("Backend unavailable")
  ])("keeps transport interruptions retryable: %s", error => {
    expect(status.isTransientSourceStatusError(error)).toEqual({ transient: true, message: error.message })
  })

  it.each([500, 502, 503, 504, 507])("keeps server-side status lookup failure %s retryable", code => {
    const error = Object.assign(new Error("Internal Server Error"), { status: code })
    expect(status.isTransientSourceStatusError(error)).toEqual({ transient: true, message: error.message })
  })

  it("does not hide a definitive permission failure", () => {
    expect(status.isTransientSourceStatusError({ status: 403, message: "Forbidden" }).transient).toBe(false)
  })

  it.each([
    ["processing", "partially_queryable", "vector_index_failed", false],
    ["processing", "partially_queryable", "vector_index_pending", true],
    ["processing", "indexing", "vector_index_pending", true],
    ["processing", undefined, undefined, true],
    ["ready", "queryable", "source_queryable", false],
    ["error", "failed", "ingestion_failed", false]
  ])("projects %s/%s/%s as active work=%s", (sourceStatus, lifecycleState, statusReason, expected) => {
    const source = { id: "source", mediaId: 1, title: "PDF", type: "pdf", addedAt: new Date(),
      status: sourceStatus, statusDetails: { lifecycleState, statusReason } } as WorkspaceSource
    expect(status.isWorkspaceSourceActivelyProcessing?.(source)).toBe(expected)
  })
})
