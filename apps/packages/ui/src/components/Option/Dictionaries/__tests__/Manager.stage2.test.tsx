import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Modal } from "antd"
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
  confirmDangerMock,
  undoNotificationMock,
  notificationMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  confirmDangerMock: vi.fn(async () => true),
  undoNotificationMock: {
    showUndoNotification: vi.fn()
  },
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
    deleteDictionaryEntry: vi.fn(async () => ({})),
    addDictionaryEntry: vi.fn(async () => ({ id: 9001 })),
    importDictionaryJSON: vi.fn(async () => ({}))
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
  useUndoNotification: () => undoNotificationMock
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
  useConfirmDanger: () => confirmDangerMock
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

describe("DictionariesManager stage-2 recovery flows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    const cache = new Map<string, any>()
    const keyFor = (queryKey: unknown) => JSON.stringify(queryKey)

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      getQueryData: vi.fn((queryKey: unknown) => cache.get(keyFor(queryKey))),
      setQueryData: vi.fn((queryKey: unknown, updater: any) => {
        const key = keyFor(queryKey)
        const current = cache.get(key)
        const next = typeof updater === "function" ? updater(current) : updater
        cache.set(key, next)
      })
    })

    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
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
              entry_count: 1
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
          data: [
            {
              id: 77,
              dictionary_id: 101,
              pattern: "BP",
              replacement: "blood pressure",
              type: "literal",
              enabled: true,
              case_sensitive: false,
              probability: 1,
              group: "abbrev",
              max_replacements: 0
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    Modal.destroyAll()
  })

  it("shows undo flow for entry deletion and restores the entry on undo", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    await user.click(screen.getByRole("button", { name: "Delete entry BP" }))

    expect(confirmDangerMock).toHaveBeenCalled()
    expect(tldwClientMock.deleteDictionaryEntry).toHaveBeenCalledWith(77)
    expect(undoNotificationMock.showUndoNotification).toHaveBeenCalledTimes(1)

    const undoOptions = undoNotificationMock.showUndoNotification.mock.calls[0][0]
    await undoOptions.onUndo()

    expect(tldwClientMock.addDictionaryEntry).toHaveBeenCalledWith(
      101,
      expect.objectContaining({
        pattern: "BP",
        replacement: "blood pressure",
        type: "literal",
        group: "abbrev"
      })
    )
  }, 15000)

  it("shows client-side structural import errors and blocks malformed payloads", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "Import" }))
    const previewImportButton = await screen.findByRole("button", {
      name: "Preview import"
    })

    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement | null
    expect(fileInput).not.toBeNull()

    const malformedImport = new File(
      ["{}"],
      "broken-dictionary.json",
      { type: "application/json" }
    )
    Object.defineProperty(malformedImport, "text", {
      value: async () =>
        JSON.stringify({
          entries: [{ replacement: "Doctor" }]
        })
    })

    fireEvent.change(fileInput as HTMLInputElement, {
      target: { files: [malformedImport] }
    })
    await waitFor(() => {
      expect(previewImportButton).toBeEnabled()
    })
    await user.click(previewImportButton)

    await waitFor(() => {
      expect(
        screen.getByText("Unable to import this file. Fix the following and retry:")
      ).toBeInTheDocument()
    })
    expect(
      screen.getByText("Missing required field: name (non-empty string).")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Entry 1: missing required field `pattern`.")
    ).toBeInTheDocument()
    expect(tldwClientMock.importDictionaryJSON).not.toHaveBeenCalled()
  }, 15000)
})
