import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Form } from "antd"
import { WorldBookEntryManager } from "../WorldBookEntryManager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
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
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({}))
  },
  mockBreakpoints: { md: true } as Record<string, boolean>
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

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
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

const makeUseMutationResult = (opts: any) => ({
  mutate: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  mutateAsync: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  isPending: false
})

const sampleEntries = [
  {
    entry_id: 1,
    keywords: ["dragon"],
    content: "Dragons are large winged creatures that breathe fire.",
    priority: 80,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  },
  {
    entry_id: 2,
    keywords: ["elf"],
    content: "Elves are immortal beings with pointed ears.",
    priority: 60,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  }
]

// Wrapper that provides a Form instance to WorldBookEntryManager
const TestWrapper: React.FC<{
  tokenBudget?: number | null
  entries?: any[]
}> = ({ tokenBudget = 100, entries = sampleEntries }) => {
  const [form] = Form.useForm()
  return (
    <WorldBookEntryManager
      worldBookId={1}
      worldBookName="Test World Book"
      tokenBudget={tokenBudget}
      worldBooks={[{ id: 1, name: "Test World Book" }]}
      form={form}
    />
  )
}

describe("WorldBookEntryManager budget feedback", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({
          data: { entries: sampleEntries, total: sampleEntries.length },
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("renders WorldBookBudgetBar at the top when tokenBudget is provided", () => {
    render(<TestWrapper tokenBudget={500} />)

    const meter = screen.getByRole("meter")
    expect(meter).toBeInTheDocument()
    expect(meter).toHaveAttribute("aria-label", "Token budget usage")
  })

  it("does not render budget bar when tokenBudget is null", () => {
    render(<TestWrapper tokenBudget={null} />)

    // The fallback message should appear instead
    expect(screen.getByText(/Configure a token budget/)).toBeInTheDocument()
    expect(screen.queryByRole("meter")).not.toBeInTheDocument()
  })

  it("shows per-entry token estimate text near entries", () => {
    render(<TestWrapper tokenBudget={500} />)

    // Each entry's content column should show a token count
    // "Dragons are large winged creatures that breathe fire." = 52 chars -> ceil(52/4) = 13 tokens
    const tokenLabels = screen.getAllByText(/~\d+ tokens/)
    expect(tokenLabels.length).toBeGreaterThanOrEqual(2)
  })

  it("shows projected budget when content is typed in the add form", async () => {
    render(<TestWrapper tokenBudget={500} />)

    const contentField = screen.getByRole("textbox", { name: "Content" })
    fireEvent.change(contentField, {
      target: { value: "Projected budget copy for the add-entry form." }
    })

    expect(await screen.findByText(/After save/)).toBeInTheDocument()
  }, 15000)

  it("shows soft warning when projected total exceeds budget", async () => {
    render(<TestWrapper tokenBudget={25} />)

    const contentField = screen.getByRole("textbox", { name: "Content" })
    fireEvent.change(contentField, {
      target: { value: "This extra content should exceed the tiny token budget." }
    })

    expect(await screen.findByText(/over budget/)).toBeInTheDocument()
  }, 15000)

  it("Save button is NOT disabled when over budget", async () => {
    render(<TestWrapper tokenBudget={25} />)

    const contentField = screen.getByRole("textbox", { name: "Content" })
    fireEvent.change(contentField, {
      target: { value: "This content pushes us way over the tiny budget limit easily." }
    })

    const addButton = screen.getByRole("button", { name: /Add Entry/i })
    expect(addButton).not.toBeDisabled()
  }, 15000)
})
