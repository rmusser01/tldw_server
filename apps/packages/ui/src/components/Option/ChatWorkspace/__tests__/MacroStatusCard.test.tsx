import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { MacroStatusCard } from "../MacroStatusCard"

const runningDetail = {
  run: {
    run_id: "run-1",
    macro_name: "wrapup",
    macro_command: "wrapup",
    status: "running",
    output_profile: "default"
  },
  branches: [
    { branch_id: "b1", label: "Summary", status: "completed", output_name: "summary" },
    { branch_id: "b2", label: "Risks", status: "running", output_name: "risks" }
  ]
}

describe("MacroStatusCard", () => {
  it("renders running macro status with branch count, output profile, and cancel action", () => {
    const onCancel = vi.fn()

    render(
      <MacroStatusCard
        metadata={{
          run_id: "run-1",
          name: "wrapup",
          command: "wrapup",
          status: "running",
          output_profile: "default"
        }}
        runDetail={runningDetail}
        onCancel={onCancel}
      />
    )

    expect(screen.getByText("/wrapup")).toBeInTheDocument()
    expect(screen.getByText("running")).toBeInTheDocument()
    expect(screen.getByText("2 branches")).toBeInTheDocument()
    expect(screen.getByText("default")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Cancel macro run" }))
    expect(onCancel).toHaveBeenCalledWith("run-1")
  })

  it("renders failed branch summaries without raw provider secret details", () => {
    render(
      <MacroStatusCard
        metadata={{
          run_id: "run-2",
          name: "wrapup",
          command: "wrapup",
          status: "failed",
          output_profile: "default"
        }}
        runDetail={{
          run: {
            run_id: "run-2",
            macro_name: "wrapup",
            macro_command: "wrapup",
            status: "failed",
            output_profile: "default"
          },
          branches: [
            {
              branch_id: "b1",
              label: "Risks",
              status: "failed",
              output_name: "risks",
              error_code: "provider_error",
              error: "Authorization: Bearer sk-secret-provider-token"
            }
          ]
        }}
      />
    )

    expect(screen.getByText("provider_error")).toBeInTheDocument()
    expect(screen.queryByText(/sk-secret-provider-token/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Authorization: Bearer/i)).not.toBeInTheDocument()
  })
})
