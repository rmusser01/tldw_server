import "@testing-library/jest-dom/vitest"
import { Component, type ReactNode } from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

const transport = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: transport.request,
  bgUpload: vi.fn()
}))

import { listChecklists } from "@/services/kanban"
import { ChecklistSection } from "../ChecklistSection"

class CardBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  render() {
    return this.state.failed ? (
      <div role="alert">Card detail crashed</div>
    ) : (
      this.props.children
    )
  }
}

async function renderLoadedChecklists() {
  const data = await listChecklists(7)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  client.setQueryData(["kanban-card-checklists", 7], data)
  render(
    <QueryClientProvider client={client}>
      <CardBoundary>
        <ChecklistSection cardId={7} />
      </CardBoundary>
    </QueryClientProvider>
  )
  return client
}

beforeEach(() => {
  transport.request.mockReset()
})

describe("ChecklistSection canonical API data", () => {
  it("keeps the add control usable when the API returns an empty envelope", async () => {
    transport.request.mockResolvedValue({ checklists: [] })
    await renderLoadedChecklists()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add Checklist" })).toBeEnabled()
  })

  it("renders the API checklist name and checked item text", async () => {
    const checklist = {
      id: 21,
      uuid: "checklist-21",
      card_id: 7,
      name: "Launch checks",
      position: 0,
      created_at: "2026-10-03T04:00:00Z",
      updated_at: "2026-10-03T04:00:00Z"
    }
    transport.request.mockImplementation(async ({ path }) => {
      if (path === "/api/v1/kanban/cards/7/checklists")
        return { checklists: [checklist] }
      if (path === "/api/v1/kanban/checklists/21")
        return {
          ...checklist,
          items: [
            {
              id: 31,
              uuid: "item-31",
              checklist_id: 21,
              name: "Confirm alignment",
              checked: true,
              checked_at: "2026-10-03T04:01:00Z",
              position: 0,
              created_at: "2026-10-03T04:00:00Z",
              updated_at: "2026-10-03T04:01:00Z"
            }
          ],
          total_items: 1,
          checked_items: 1,
          progress_percent: 100
        }
      throw new Error(`Unexpected request: ${path}`)
    })
    await renderLoadedChecklists()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getByText("Launch checks")).toBeInTheDocument()
    expect(screen.getByText("Confirm alignment")).toBeInTheDocument()
    expect(screen.getByRole("checkbox")).toBeChecked()
  })
  it("shows a load error instead of implying the card has no checklists", async () => {
    transport.request.mockRejectedValue(new Error("Checklist unavailable"))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    render(
      <QueryClientProvider client={client}>
        <ChecklistSection cardId={7} />
      </QueryClientProvider>
    )
    await waitFor(() =>
      expect(client.getQueryState(["kanban-card-checklists", 7])?.status).toBe(
        "error"
      )
    )
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Unable to load checklists."
    )
    expect(screen.getByRole("button", { name: "Add Checklist" })).toBeDisabled()
  })
  it("blocks Enter submission if an open add form loses its checklist data", async () => {
    transport.request.mockResolvedValue({ checklists: [] })
    const client = await renderLoadedChecklists()
    fireEvent.click(screen.getByRole("button", { name: "Add Checklist" }))
    const input = screen.getByPlaceholderText("Checklist title")
    fireEvent.change(input, { target: { value: "New checks" } })
    transport.request.mockRejectedValue(new Error("Checklist unavailable"))
    await act(async () => {
      await client.invalidateQueries({
        queryKey: ["kanban-card-checklists", 7]
      })
    })
    await screen.findByRole("alert")
    expect(screen.getByRole("button", { name: /^Add$/ })).toBeDisabled()
    await act(async () => {
      fireEvent.keyDown(input, { key: "Enter", code: "Enter", keyCode: 13 })
    })
    expect(
      transport.request.mock.calls.filter(
        ([request]) => request.method === "POST"
      )
    ).toEqual([])
  })
})
