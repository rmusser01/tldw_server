import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Form } from "antd"
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query"
import { WorldBookEntryManager } from "../WorldBookEntryManager"

const {
  confirmDangerMock,
  tldwClientMock
} = vi.hoisted(() => ({
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listWorldBooks: vi.fn(),
    listWorldBookEntries: vi.fn(),
    addWorldBookEntry: vi.fn(),
    deleteWorldBookEntry: vi.fn(),
    updateWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({}))
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn(), success: vi.fn() })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

const ParentSummary = () => {
  const { data = [] } = useQuery({
    queryKey: ["tldw:listWorldBooks"],
    queryFn: async () => {
      await tldwClientMock.initialize()
      const response = await tldwClientMock.listWorldBooks(false)
      return response.world_books
    }
  })

  return (
    <>
      {data.map((worldBook: { id: number; entry_count: number }) => (
        <output key={worldBook.id} aria-label={`Parent entry count ${worldBook.id}`}>
          {worldBook.entry_count}
        </output>
      ))}
    </>
  )
}

const SummaryHarness = ({ worldBooks = [{ id: 1, name: "Arcana" }] }: { worldBooks?: Array<{ id: number; name: string }> }) => {
  const [form] = Form.useForm()
  React.useEffect(() => {
    form.setFieldsValue({ keywords: ["lore"], content: "Entry content" })
  }, [form])

  return (
    <>
      <ParentSummary />
      <WorldBookEntryManager
        worldBookId={1}
        worldBookName="Arcana"
        tokenBudget={500}
        worldBooks={worldBooks}
        form={form}
      />
    </>
  )
}

describe("WorldBookEntryManager parent summary invalidation", () => {
  it("refreshes the visible parent entry count after successful add and delete without reload", async () => {
    vi.clearAllMocks()
    let entries: Array<{ entry_id: number; keywords: string[]; content: string; enabled: boolean }> = []
    tldwClientMock.listWorldBooks.mockImplementation(async () => ({
      world_books: [{ id: 1, name: "Arcana", entry_count: entries.length }]
    }))
    tldwClientMock.listWorldBookEntries.mockImplementation(async () => ({
      entries,
      total: entries.length
    }))
    tldwClientMock.addWorldBookEntry.mockImplementation(async () => {
      entries = [{ entry_id: 1, keywords: ["lore"], content: "Entry content", enabled: true }]
      return {}
    })
    tldwClientMock.deleteWorldBookEntry.mockImplementation(async () => {
      entries = []
      return {}
    })

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <SummaryHarness />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("0")
    })

    await user.click(screen.getByRole("button", { name: "Add Entry" }))
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("1")
    })

    await user.click(await screen.findByRole("button", { name: "Delete entry" }))
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("0")
    })

    queryClient.clear()
  }, 15000)

  it("keeps the visible parent count unchanged when an add mutation fails", async () => {
    vi.clearAllMocks()
    tldwClientMock.listWorldBooks.mockResolvedValue({
      world_books: [{ id: 1, name: "Arcana", entry_count: 0 }]
    })
    tldwClientMock.listWorldBookEntries.mockResolvedValue({ entries: [], total: 0 })
    tldwClientMock.addWorldBookEntry.mockRejectedValue(new Error("entry save failed"))

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <SummaryHarness />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("0")
    })
    await user.click(screen.getByRole("button", { name: "Add Entry" }))
    await waitFor(() => {
      expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledTimes(1)
    })

    expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("0")
    expect(tldwClientMock.listWorldBooks).toHaveBeenCalledTimes(1)
    queryClient.clear()
  }, 15000)

  it("refreshes the destination parent count after a copied move when source deletion fails", async () => {
    vi.clearAllMocks()
    let sourceEntries = [
      { entry_id: 1, keywords: ["move"], content: "Move me", enabled: true, priority: 50 }
    ]
    let destinationEntries: Array<{ entry_id: number; keywords: string[]; content: string; enabled: boolean }> = []
    tldwClientMock.listWorldBooks.mockImplementation(async () => ({
      world_books: [
        { id: 1, name: "Arcana", entry_count: sourceEntries.length },
        { id: 2, name: "Archive", entry_count: destinationEntries.length }
      ]
    }))
    tldwClientMock.listWorldBookEntries.mockImplementation(async (id: number) => ({
      entries: id === 1 ? sourceEntries : destinationEntries,
      total: id === 1 ? sourceEntries.length : destinationEntries.length
    }))
    tldwClientMock.addWorldBookEntry.mockImplementation(async (id: number, entry: { keywords: string[]; content: string; enabled: boolean }) => {
      if (id === 2) destinationEntries = [{ entry_id: 2, ...entry }]
      return {}
    })
    tldwClientMock.bulkWorldBookEntries.mockRejectedValue(new Error("source delete failed"))

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const user = userEvent.setup()
    render(
      <QueryClientProvider client={queryClient}>
        <SummaryHarness worldBooks={[{ id: 1, name: "Arcana" }, { id: 2, name: "Archive" }]} />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 1" })).toHaveTextContent("1")
      expect(screen.getByRole("status", { name: "Parent entry count 2" })).toHaveTextContent("0")
    })
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })
    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    await user.click((tableWrapper as HTMLElement).querySelectorAll('input[type="checkbox"]')[1])
    await user.click(screen.getByRole("button", { name: "Move To" }))
    await user.click(screen.getByRole("combobox", { name: "Bulk move destination" }))
    await user.click(await screen.findByText("Archive", { selector: ".ant-select-item-option-content" }))
    await user.click(screen.getByRole("button", { name: "Move Entries" }))

    await waitFor(() => {
      expect(tldwClientMock.bulkWorldBookEntries).toHaveBeenCalledWith({ entry_ids: [1], operation: "delete" })
    })
    await waitFor(() => {
      expect(screen.getByRole("status", { name: "Parent entry count 2" })).toHaveTextContent("1")
    })

    queryClient.clear()
  }, 15000)
})
