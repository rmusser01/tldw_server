import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  undoNotificationMock: {
    showUndoNotification: vi.fn()
  },
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined)
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => undoNotificationMock
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: null,
  status: "success",
  isLoading: false,
  isFetching: false,
  isPending: false,
  error: null,
  refetch: vi.fn(),
  ...value
})

const makeUseMutationResult = () => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false
})

describe("WorldBooksManager stage-2 filtering and sorting", () => {
  let currentWorldBooks: any[]

  beforeEach(() => {
    vi.clearAllMocks()
    currentWorldBooks = [
      {
        id: 1,
        name: "Arcana",
        description: "Primary magic reference",
        last_modified: "2026-02-18T09:00:00Z",
        token_budget: 800,
        enabled: true,
        entry_count: 5
      },
      {
        id: 2,
        name: "Legacy Notes",
        description: "Legacy snippets",
        last_modified: "2026-02-18T08:00:00Z",
        token_budget: 200,
        enabled: false,
        entry_count: 2
      },
      {
        id: 3,
        name: "Atlas",
        description: "Geography and regions",
        last_modified: "2026-02-18T07:00:00Z",
        token_budget: 300,
        enabled: true,
        entry_count: 1
      }
    ]

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation(() => makeUseMutationResult())

    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({ data: currentWorldBooks, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [{ id: 77, name: "Ari" }] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({
          data: {
            1: [{ id: 77, name: "Ari" }],
            2: [],
            3: []
          },
          isLoading: false
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("filters by search text, enabled state, and attachment state", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.type(screen.getByLabelText("Search world books"), "legacy")
    expect(screen.getByText("Legacy Notes")).toBeInTheDocument()
    expect(screen.queryByText("Arcana")).not.toBeInTheDocument()

    await user.clear(screen.getByLabelText("Search world books"))
    await user.click(screen.getByLabelText("Filter by enabled status"))
    await user.click(await screen.findByText("Enabled", { selector: ".ant-select-item-option-content" }))
    await waitFor(() => {
      expect(screen.getByText("Arcana")).toBeInTheDocument()
      expect(screen.getByText("Atlas")).toBeInTheDocument()
      expect(screen.queryByText("Legacy Notes")).not.toBeInTheDocument()
    })

    await user.click(screen.getByLabelText("Filter by attachment state"))
    await user.click(await screen.findByText("Has attachments", { selector: ".ant-select-item-option-content" }))
    await waitFor(() => {
      expect(screen.getByText("Arcana")).toBeInTheDocument()
      expect(screen.queryByText("Atlas")).not.toBeInTheDocument()
    })
  }, 20000)

  it("sorts by entries and keeps sort state after data changes", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByRole("columnheader", { name: "Entries" }))

    await waitFor(() => {
      const firstRowText = document.querySelector("tbody tr")?.textContent || ""
      expect(firstRowText).toContain("Atlas")
    })

    currentWorldBooks = [
      ...currentWorldBooks,
      {
        id: 4,
        name: "Pocket Notes",
        description: "Newly fetched row",
        last_modified: "2026-02-18T10:00:00Z",
        token_budget: 120,
        enabled: true,
        entry_count: 0
      }
    ]

    await user.type(screen.getByLabelText("Search world books"), "a")
    await user.clear(screen.getByLabelText("Search world books"))

    await waitFor(() => {
      const firstRowText = document.querySelector("tbody tr")?.textContent || ""
      expect(firstRowText).toContain("Pocket Notes")
    })
  }, 20000)

  it("sorts by name and enabled columns", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByRole("columnheader", { name: "Name" }))
    await user.click(screen.getByRole("columnheader", { name: "Name" }))
    await waitFor(() => {
      const firstRowText = document.querySelector("tbody tr")?.textContent || ""
      expect(firstRowText).toContain("Legacy Notes")
    })

    await user.click(screen.getByRole("columnheader", { name: "Status" }))
    await waitFor(() => {
      const firstRowText = document.querySelector("tbody tr")?.textContent || ""
      expect(firstRowText).toContain("Legacy Notes")
    })
  }, 20000)
})
