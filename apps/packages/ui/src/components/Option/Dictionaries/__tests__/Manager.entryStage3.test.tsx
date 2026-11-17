import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches:
      /min-width:\s*576px/.test(query) ||
      /min-width:\s*768px/.test(query) ||
      /min-width:\s*992px/.test(query),
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false
  })
})

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
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
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    reorderDictionaryEntries: vi.fn(async () => ({
      success: true,
      dictionary_id: 77,
      affected_count: 3,
      entry_ids: [2, 1, 3],
      message: "Reordered"
    })),
    bulkDictionaryEntries: vi.fn(async () => ({
      success: true,
      affected_count: 1,
      failed_ids: [],
      message: "ok"
    })),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    addDictionaryEntry: vi.fn(async () => ({ id: 99 })),
    updateDictionaryEntry: vi.fn(async () => ({}))
  }
}))

vi.mock("@/hooks/useChatDraftOwner", () => ({ useChatDraftOwner: () => dictionaryOwnerFixture }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => selectDictionaryChatFixture }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: (...args: Parameters<typeof dictionarySnapshotFixture>) => dictionarySnapshotFixture(...args) }))

vi.mock("@tanstack/react-query", () => ({
  useQuery: (options: { queryKey?: readonly unknown[] }) => scopedDictionaryQueryFixture(options, useQueryMock(options)),
  useMutation: useMutationMock,
  useQueryClient: () => scopedDictionaryQueryClientFixture(useQueryClientMock())
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

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: {
      hasChatDictionaries: true
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({
    title,
    description,
    examples,
    primaryActionLabel,
    onPrimaryAction,
    secondaryActionLabel,
    onSecondaryAction
  }: any) => (
    <div>
      <h2>{title}</h2>
      {description ? <p>{description}</p> : null}
      {Array.isArray(examples) && examples.length > 0 ? (
        <ul>
          {examples.map((example: any, idx: number) => (
            <li key={idx}>{example}</li>
          ))}
        </ul>
      ) : null}
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
      {secondaryActionLabel ? (
        <button type="button" onClick={onSecondaryAction}>
          {secondaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

vi.mock("@/components/Common/LabelWithHelp", () => ({
  LabelWithHelp: ({ label }: { label: React.ReactNode }) => <span>{label}</span>
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: undefined,
  status: "success",
  error: null,
  isPending: false,
  isFetching: false,
  isLoading: false,
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
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
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
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
    }
  },
  isPending: false
})

describe("DictionariesManager entry stage-3 bulk operations and order guidance", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn(),
      getQueryData: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 77,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: 3
            }
          ]
        })
      }

      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 77,
            name: "Medical Terms",
            description: "Clinical substitutions"
          }
        })
      }

      if (key === "tldw:listDictionaryEntries" || key === "tldw:listDictionaryEntriesAll") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 1,
              dictionary_id: 77,
              pattern: "BP",
              replacement: "blood pressure",
              type: "literal",
              probability: 1,
              group: "Clinical",
              enabled: true,
              case_sensitive: false
            },
            {
              id: 2,
              dictionary_id: 77,
              pattern: "HR",
              replacement: "heart rate",
              type: "literal",
              probability: 1,
              group: "Vitals",
              enabled: true,
              case_sensitive: false
            },
            {
              id: 3,
              dictionary_id: 77,
              pattern: "/dr\\./i",
              replacement: "Doctor",
              type: "regex",
              probability: 1,
              group: "Titles",
              enabled: false,
              case_sensitive: false
            }
          ]
        })
      }

      return makeUseQueryResult({})
    })
  })

  const openEntryManagerAndSelectRows = async (
    user: ReturnType<typeof userEvent.setup>,
    rowIndexes: number[]
  ) => {
    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    const patternHeader = await screen.findByRole("columnheader", { name: "Pattern" })
    const tableWrapper = patternHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    for (const rowIndex of rowIndexes) {
      await user.click(checkboxes[rowIndex])
    }
  }

  it("shows processing-order guidance and sends activate bulk payload", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await openEntryManagerAndSelectRows(user, [1])

    expect(
      screen.getByText(
        "Entries are processed in priority order (top to bottom). Use the up/down controls to reorder."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("1 selected")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Enable" }))

    await waitFor(() => {
      expect(tldwClientMock.bulkDictionaryEntries).toHaveBeenCalledWith({
        entry_ids: [1],
        operation: "activate"
      })
    })
    expect(notificationMock.success).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Bulk action complete" })
    )
  }, 45000)

  it("reorders entries through priority controls", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("columnheader", { name: "Pattern" })

    await user.click(screen.getByRole("button", { name: "Move entry BP down" }))

    await waitFor(() => {
      expect(tldwClientMock.reorderDictionaryEntries).toHaveBeenCalledWith(77, {
        entry_ids: [2, 1, 3]
      })
    })
  }, 60000)

  it("submits group bulk action and keeps failed rows selected", async () => {
    const user = userEvent.setup()
    tldwClientMock.bulkDictionaryEntries.mockResolvedValueOnce({
      success: false,
      affected_count: 1,
      failed_ids: [2],
      message: "partial"
    })
    render(<DictionariesManager />)

    await openEntryManagerAndSelectRows(user, [1, 2])
    await user.type(screen.getByLabelText("Bulk group name"), "Reviewed")
    await user.click(screen.getByRole("button", { name: "Set Group" }))

    await waitFor(() => {
      expect(tldwClientMock.bulkDictionaryEntries).toHaveBeenCalledWith({
        entry_ids: [1, 2],
        operation: "group",
        group_name: "Reviewed"
      })
    })

    expect(notificationMock.warning).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Bulk action completed with errors"
      })
    )
    expect(screen.getByText("1 selected")).toBeInTheDocument()
  }, 120000)
})
