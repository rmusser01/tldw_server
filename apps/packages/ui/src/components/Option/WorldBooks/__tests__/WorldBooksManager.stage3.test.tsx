import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock,
  mockBreakpoints
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
  },
  mockBreakpoints: { md: true }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return {
    ...actual,
    Grid: {
      ...actual.Grid,
      useBreakpoint: () => mockBreakpoints
    }
  }
})

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

describe("WorldBooksManager stage-3 action affordances", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true
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
              name: "Arcana",
              description: "Primary lore",
              last_modified: "2026-02-18T09:00:00Z",
              token_budget: 500,
              enabled: true,
              entry_count: 2
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [{ id: 10, name: "Aria" }] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({
          data: {
            1: [{ id: 10, name: "Aria" }]
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

  it("uses explicit attachment affordance and consistent icon action labels", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    expect(screen.getByText("Open to load")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Edit world book" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Manage entries" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Duplicate world book" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Quick attach characters" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export world book" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "View world book statistics" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete world book" })).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Quick attach characters" }))
    expect(
      await screen.findByRole("button", { name: "View attached characters for Arcana (1)" })
    ).toBeInTheDocument()
  })
})
