import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import type { Card, ListWithCards } from "@/types/kanban"
import { CardDetailPanel } from "../CardDetailPanel"
import { formatKanbanDateTimeLocalValue } from "../kanbanDateTime"

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn(),
  useMutation: vi.fn(),
  useQueryClient: vi.fn()
}))

vi.mock("@/services/kanban", () => ({
  copyCard: vi.fn(),
  listComments: vi.fn(),
  createComment: vi.fn(),
  generateClientId: vi.fn(() => "test-client-id")
}))

vi.mock("../LabelManager", () => ({
  LabelManager: () => <div data-testid="label-manager" />
}))

vi.mock("../ChecklistSection", () => ({
  ChecklistSection: () => <div data-testid="checklist-section" />
}))

const makeCard = (overrides: Partial<Card> = {}): Card => ({
  id: 11,
  uuid: "card-uuid",
  title: "Initial title",
  description: "Initial description",
  board_id: 1,
  list_id: 2,
  client_id: "client-1",
  position: 0,
  due_date: new Date(2026, 3, 3, 12, 34).toISOString(),
  due_complete: false,
  priority: "medium",
  archived: false,
  created_at: "2026-04-01T10:00:00Z",
  updated_at: "2026-04-02T10:00:00Z",
  deleted: false,
  version: 1,
  labels: [],
  ...overrides
})

const lists: ListWithCards[] = [
  {
    id: 2,
    uuid: "list-uuid",
    name: "Todo",
    board_id: 1,
    client_id: "list-client",
    position: 0,
    archived: false,
    created_at: "2026-04-01T10:00:00Z",
    updated_at: "2026-04-01T10:00:00Z",
    deleted: false,
    version: 1,
    cards: []
  }
]

const renderPanel = ({
  card = makeCard(),
  onSave = vi.fn()
}: {
  card?: Card
  onSave?: (cardId: number, data: any) => void
} = {}) => {
  render(
    <CardDetailPanel
      card={card}
      boardId={1}
      lists={lists}
      open
      onClose={vi.fn()}
      onSave={onSave}
      onDelete={vi.fn()}
      onMove={vi.fn()}
    />
  )
  return { card, onSave }
}

describe("CardDetailPanel due-date editing", () => {
  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useQueryClient).mockReturnValue({
      invalidateQueries: vi.fn()
    } as any)
    vi.mocked(useQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
    vi.mocked(useMutation).mockReturnValue({
      mutate: vi.fn(),
      isPending: false
    } as any)
  })

  it("renders the due date as a native datetime-local input", () => {
    const { card } = renderPanel()

    const input = screen.getByLabelText("Due Date") as HTMLInputElement

    expect(input.type).toBe("datetime-local")
    expect(input).toHaveValue(formatKanbanDateTimeLocalValue(card.due_date))
  })

  it("saves a changed due date as an ISO timestamp", () => {
    const onSave = vi.fn()
    const { card } = renderPanel({ onSave })
    const input = screen.getByLabelText("Due Date")

    fireEvent.change(input, { target: { value: "2026-05-09T14:45" } })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))

    expect(onSave).toHaveBeenCalledWith(card.id, {
      due_date: new Date(2026, 4, 9, 14, 45).toISOString()
    })
  })

  it("saves a cleared due date as null", () => {
    const onSave = vi.fn()
    const { card } = renderPanel({ onSave })

    fireEvent.change(screen.getByLabelText("Due Date"), {
      target: { value: "" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))

    expect(onSave).toHaveBeenCalledWith(card.id, { due_date: null })
  })

  it("does not rewrite an unchanged due date when saving another field", () => {
    const onSave = vi.fn()
    const { card } = renderPanel({ onSave })

    fireEvent.change(screen.getByPlaceholderText("Card title"), {
      target: { value: "Updated title" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))

    expect(onSave).toHaveBeenCalledWith(card.id, { title: "Updated title" })
  })

  it("does not keep resending due date after a due-date save in the same drawer session", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)
    const { card } = renderPanel({ onSave })

    fireEvent.change(screen.getByLabelText("Due Date"), {
      target: { value: "2026-05-09T14:45" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1))

    fireEvent.change(screen.getByPlaceholderText("Card title"), {
      target: { value: "Updated title" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }))

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(2))
    expect(onSave).toHaveBeenNthCalledWith(2, card.id, {
      title: "Updated title"
    })
  })
})
