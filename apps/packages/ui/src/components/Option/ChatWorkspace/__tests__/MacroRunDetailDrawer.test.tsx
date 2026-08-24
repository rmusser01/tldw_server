import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getChatMacroRun: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  getChatMacroRun: (...args: unknown[]) => mocks.getChatMacroRun(...args)
}))

import { MacroRunDetailDrawer } from "../MacroRunDetailDrawer"

describe("MacroRunDetailDrawer", () => {
  beforeEach(() => {
    mocks.getChatMacroRun.mockReset()
    mocks.getChatMacroRun.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        run: {
          run_id: "run-1",
          macro_name: "wrapup",
          macro_command: "wrapup",
          status: "completed",
          output_profile: "default"
        },
        branches: [
          {
            branch_id: "b1",
            label: "Summary",
            status: "completed",
            output_name: "summary",
            output: "Useful summary"
          }
        ]
      }
    })
  })

  it("fetches run detail lazily when opened", async () => {
    render(<MacroRunDetailDrawer runId="run-1" open onClose={vi.fn()} />)

    await waitFor(() => expect(mocks.getChatMacroRun).toHaveBeenCalledWith("run-1"))
    expect(await screen.findByText("Macro run detail")).toBeInTheDocument()
    expect(screen.getByText("completed")).toBeInTheDocument()
    expect(screen.getByText("Summary")).toBeInTheDocument()
    expect(screen.getByText("Useful summary")).toBeInTheDocument()
  })

  it("does not fetch when closed", () => {
    render(<MacroRunDetailDrawer runId="run-1" open={false} onClose={vi.fn()} />)

    expect(mocks.getChatMacroRun).not.toHaveBeenCalled()
  })

  it("redacts API and branch errors before rendering", async () => {
    mocks.getChatMacroRun.mockResolvedValueOnce({
      ok: true,
      status: 200,
      data: {
        run: {
          run_id: "run-1",
          macro_name: "wrapup",
          macro_command: "wrapup",
          status: "failed"
        },
        branches: [
          {
            branch_id: "b1",
            step_id: "summary",
            status: "failed",
            error: "provider rejected api_key=sk-live-secret-value"
          }
        ]
      }
    })

    render(<MacroRunDetailDrawer runId="run-1" open onClose={vi.fn()} />)

    expect(await screen.findByText(/provider rejected/)).toHaveTextContent(
      "provider rejected api_key=[redacted-secret]"
    )
    expect(screen.queryByText(/sk-live-secret-value/)).not.toBeInTheDocument()
  })

  it("redacts API failure text before storing it in UI state", async () => {
    mocks.getChatMacroRun.mockResolvedValueOnce({
      ok: false,
      status: 500,
      error: "Authorization: Bearer secret-token-value"
    })

    render(<MacroRunDetailDrawer runId="run-1" open onClose={vi.fn()} />)

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "[redacted bearer token]"
    )
    expect(screen.queryByText(/secret-token-value/)).not.toBeInTheDocument()
  })
})
