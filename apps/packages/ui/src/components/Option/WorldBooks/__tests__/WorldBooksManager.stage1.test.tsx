import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, within } from "@testing-library/react"
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

describe("WorldBooksManager stage-1 list metadata", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-02-18T12:00:00Z"))
    vi.clearAllMocks()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })

    useMutationMock.mockImplementation(() => makeUseMutationResult())

    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({
          data: [
            {
              id: 1,
              name: "Active Lore",
              description: "Primary lore book",
              last_modified: "2026-02-18T09:00:00Z",
              token_budget: 750,
              enabled: true,
              entry_count: 3
            },
            {
              id: 2,
              name: "Archive Lore",
              description: "Legacy snippets",
              last_modified: null,
              token_budget: 320,
              enabled: false,
              entry_count: 1
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [{ id: 99, name: "Aria" }] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({
          data: {
            1: [{ id: 99, name: "Aria" }],
            2: []
          },
          isLoading: false
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
  })

  it("renders last-modified and budget metadata with attachment-on-demand and disabled cues", () => {
    render(<WorldBooksManager />)

    expect(screen.getByText("Last Modified")).toBeInTheDocument()
    expect(screen.getByText("Status")).toBeInTheDocument()

    const rows = screen.getAllByRole("row")
    const activeRow = rows.find((row) => within(row).queryByText("Active Lore"))
    const archiveRow = rows.find((row) => within(row).queryByText("Archive Lore"))

    expect(activeRow).toBeDefined()
    expect(archiveRow).toBeDefined()
    expect(within(activeRow as HTMLElement).getByText("Enabled")).toBeInTheDocument()
    expect(within(archiveRow as HTMLElement).getByText("Disabled")).toBeInTheDocument()
  })
})
