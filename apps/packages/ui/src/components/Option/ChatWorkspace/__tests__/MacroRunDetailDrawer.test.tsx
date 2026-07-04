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
})
