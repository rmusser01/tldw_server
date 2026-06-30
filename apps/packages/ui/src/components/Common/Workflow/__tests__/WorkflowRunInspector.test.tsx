import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { WorkflowRunInvestigation } from "@/types/workflow-editor"

import { WorkflowRunInspector } from "../WorkflowRunInspector"

const investigation: WorkflowRunInvestigation = {
  run_id: "run-42",
  status: "failed",
  schema_version: 1,
  derived_from_event_seq: 3,
  failed_step: {
    step_run_id: "run-42:s1:1",
    step_id: "s1",
    name: "Deliver webhook",
    type: "webhook",
    status: "failed",
    attempt_count: 2,
    latest_failure: {
      reason_code_core: "runtime_error",
      category: "runtime",
      blame_scope: "external_dependency",
      retryable: true,
      retry_recommendation: "conditional",
      error_summary: "Gateway timeout"
    }
  },
  primary_failure: {
    reason_code_core: "runtime_error",
    category: "runtime",
    blame_scope: "external_dependency",
    retryable: true,
    retry_recommendation: "conditional",
    error_summary: "Gateway timeout",
    internal_detail: { event_count: 3 }
  },
  attempts: [
    {
      attempt_id: "attempt-1",
      step_run_id: "run-42:s1:1",
      step_id: "s1",
      attempt_number: 1,
      status: "failed",
      started_at: "2026-03-11T10:00:00Z",
      metadata: { retry_recommendation: "conditional" }
    },
    {
      attempt_id: "attempt-2",
      step_run_id: "run-42:s1:1",
      step_id: "s1",
      attempt_number: 2,
      status: "failed",
      started_at: "2026-03-11T10:00:02Z",
      metadata: { retry_recommendation: "conditional" }
    }
  ],
  evidence: {
    events: [{ event_type: "step_failed" }],
    artifacts: [{ type: "log", uri: "file:///tmp/workflow.log" }]
  },
  recommended_actions: [
    "Retry after verifying downstream connectivity",
    "Inspect webhook stderr excerpt"
  ]
}

describe("WorkflowRunInspector", () => {
  it("uses the canonical empty state when diagnostics are unavailable", () => {
    render(<WorkflowRunInspector investigation={null} />)

    expect(
      screen
        .getByText("No run diagnostics available")
        .closest('[data-ds-component="EmptyState"]')
    ).toBeInTheDocument()
  })

  it("renders failure summary and attempts", () => {
    render(<WorkflowRunInspector investigation={investigation} />)

    expect(screen.getByText("Failure summary")).toBeInTheDocument()
    expect(
      screen.getByText("runtime_error").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(
      screen.getByText("Retryable").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(
      screen.getAllByText("failed")[0]?.closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Attempt 2")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /evidence/i })).toBeInTheDocument()
    expect(
      screen.getByText("Retry after verifying downstream connectivity")
    ).toBeInTheDocument()
  })
})
