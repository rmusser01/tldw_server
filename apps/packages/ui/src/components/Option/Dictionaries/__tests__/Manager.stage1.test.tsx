import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

if (!window.matchMedia) {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false
    })
  })
}

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
  refetchMock,
  notificationMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  refetchMock: vi.fn(async () => undefined),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
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
      fallbackOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const template = fallbackOrOptions.defaultValue ?? key
        return Object.entries(fallbackOrOptions).reduce((result, [name, value]) => {
          if (name === "defaultValue" || value == null) return result
          return result.replaceAll(`{{${name}}}`, String(value))
        }, template)
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
    initialize: vi.fn(async () => undefined)
  }
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: undefined,
  status: "success",
  error: null,
  isPending: false,
  isFetching: false,
  isLoading: false,
  refetch: refetchMock,
  ...value
})

const makeUseMutationResult = () => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false
})

describe("DictionariesManager stage-1 empty and error states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation(() => makeUseMutationResult())
  })

  it("renders list empty state guidance with create and import actions", () => {
    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: []
        })
      }
      return makeUseQueryResult({})
    })

    render(<DictionariesManager />)

    expect(screen.getByText("No dictionaries yet")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Create your first dictionary to transform text consistently across chats."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Create your first dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Import dictionary" })
    ).toBeInTheDocument()
  })

  it("renders list error state and retries loading when requested", async () => {
    const user = userEvent.setup()
    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "error",
          error: new Error("server unreachable")
        })
      }
      return makeUseQueryResult({})
    })

    render(<DictionariesManager />)

    expect(screen.getByText("Unable to load dictionaries")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Could not load dictionaries: server unreachable"
      )
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Retry" }))
    expect(refetchMock).toHaveBeenCalledTimes(1)
  })

  it("shows entry empty-state guidance when a dictionary has no entries", async () => {
    const user = userEvent.setup()
    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 101,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: 0
            }
          ]
        })
      }
      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 101,
            name: "Medical Terms",
            description: "Clinical substitutions"
          }
        })
      }
      if (key === "tldw:listDictionaryEntries") {
        return makeUseQueryResult({
          status: "success",
          data: []
        })
      }
      return makeUseQueryResult({})
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(await screen.findByText("No entries yet")).toBeInTheDocument()
    expect(
      await screen.findByText("Add a pattern/replacement pair to start transforming text.")
    ).toBeInTheDocument()
    expect(await screen.findByText("Literal: BP -> blood pressure")).toBeInTheDocument()
    expect(
      await screen.findByRole("button", { name: "Add first entry" })
    ).toBeInTheDocument()
  }, 30000)

  it("renders entry loading error state with retry action", async () => {
    const user = userEvent.setup()
    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 101,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: 0
            }
          ]
        })
      }
      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 101,
            name: "Medical Terms",
            description: "Clinical substitutions"
          }
        })
      }
      if (key === "tldw:listDictionaryEntries") {
        return makeUseQueryResult({
          status: "error",
          error: new Error("entries endpoint unavailable")
        })
      }
      return makeUseQueryResult({})
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(await screen.findByText("Unable to load entries")).toBeInTheDocument()
    expect(
      await screen.findByText("Could not load entries: entries endpoint unavailable")
    ).toBeInTheDocument()
    await user.click(await screen.findByRole("button", { name: "Retry" }))
    expect(refetchMock).toHaveBeenCalled()
  }, 15000)
})
