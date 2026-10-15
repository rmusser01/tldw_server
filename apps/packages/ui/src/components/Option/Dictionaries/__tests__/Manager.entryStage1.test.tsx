import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
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
  notificationMock
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
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    addDictionaryEntry: vi.fn(async () => ({ id: 999 }))
  }
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

describe("DictionariesManager entry stage-1 information architecture", () => {
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
              usage_count: 4,
              enabled: true,
              case_sensitive: false
            },
            {
              id: 2,
              dictionary_id: 77,
              pattern: "HR",
              replacement: "heart rate",
              type: "literal",
              probability: 0.5,
              group: "clinical",
              usage_count: 0,
              enabled: true,
              case_sensitive: false
            },
            {
              id: 3,
              dictionary_id: 77,
              pattern: "/dr\\./i",
              replacement: "Doctor",
              type: "regex",
              probability: 0.75,
              group: "Titles",
              usage_count: 2,
              enabled: true,
              case_sensitive: true
            }
          ]
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("renders type, probability, and group columns with deduped group filter options", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(await screen.findByRole("columnheader", { name: "Type" })).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Probability" })).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Group" })).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Usage" })).toBeInTheDocument()
    expect(screen.getByText("Unused")).toBeInTheDocument()
    expect(screen.getByText("heart rate").closest("tr")).toHaveClass("bg-surface2/40")

    await user.click(screen.getByLabelText("Filter entries by group"))
    expect(
      await screen.findByText("Clinical", {
        selector: ".ant-select-item-option-content"
      })
    ).toBeInTheDocument()
    expect(
      screen.queryByText("clinical", {
        selector: ".ant-select-item-option-content"
      })
    ).toBeNull()
  }, 30000)

  it("composes entry search and group filtering", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    await user.type(screen.getByLabelText("Search dictionary entries"), "blood")
    await user.click(screen.getByLabelText("Filter entries by group"))
    await user.click(
      await screen.findByText("Clinical", {
        selector: ".ant-select-item-option-content"
      })
    )

    expect(screen.getByText("blood pressure")).toBeInTheDocument()
    expect(screen.queryByText("heart rate")).toBeNull()
    expect(screen.queryByText("Doctor")).toBeNull()
  }, 15000)
})
