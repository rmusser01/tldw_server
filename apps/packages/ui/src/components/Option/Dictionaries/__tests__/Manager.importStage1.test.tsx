import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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
  tldwClientMock,
  confirmDangerMock
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
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    getDictionary: vi.fn(async () => ({})),
    listDictionaryEntries: vi.fn(async () => ({ entries: [] })),
    importDictionaryJSON: vi.fn(async () => ({ dictionary_id: 321 })),
    importDictionaryMarkdown: vi.fn(async () => ({ dictionary_id: 654 })),
    createDictionary: vi.fn(async () => ({ id: 1000 })),
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({})),
    duplicateDictionary: vi.fn(async () => ({})),
    addDictionaryEntry: vi.fn(async () => ({ id: 901 })),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({ success: true })),
    reorderDictionaryEntries: vi.fn(async () => ({ success: true })),
    exportDictionaryJSON: vi.fn(async () => ({})),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "" })),
    dictionaryStatistics: vi.fn(async () => ({})),
    validateDictionary: vi.fn(async () => ({ ok: true, errors: [], warnings: [] })),
    processDictionary: vi.fn(async () => ({
      original_text: "",
      processed_text: "",
      replacements: 0,
      iterations: 1,
      entries_used: [],
      token_budget_exceeded: false
    }))
  },
  confirmDangerMock: vi.fn(async () => true)
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
  default: ({ title, description }: any) => (
    <div>
      <h2>{title}</h2>
      {description ? <p>{description}</p> : null}
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

const getSelectInputByLabel = async (labelText: string): Promise<HTMLInputElement> => {
  const labelNode = await screen.findByText(labelText)
  const wrapper = labelNode.closest(".space-y-1")
  const input = wrapper?.querySelector('input[role="combobox"]')
  if (!input) {
    throw new Error(`Unable to resolve select input for label: ${labelText}`)
  }
  return input as HTMLInputElement
}

const chooseSelectOption = async (
  user: ReturnType<typeof userEvent.setup>,
  labelText: string,
  optionText: string
) => {
  await user.click(await getSelectInputByLabel(labelText))
  await user.click(
    await screen.findByText(optionText, {
      selector: ".ant-select-item-option-content"
    })
  )
}

const previewJsonImport = async (
  user: ReturnType<typeof userEvent.setup>,
  dictionaryName: string
) => {
  await user.click(screen.getByRole("button", { name: "Import" }))
  await chooseSelectOption(user, "Source", "Paste content")

  fireEvent.change(
    screen.getByPlaceholderText("Paste JSON dictionary content..."),
    {
      target: {
        value: JSON.stringify({
          name: dictionaryName,
          entries: [
            {
              pattern: "BP",
              replacement: "blood pressure",
              type: "literal",
              group: "Clinical"
            }
          ]
        })
      }
    }
  )

  await user.click(screen.getByRole("button", { name: "Preview import" }))
  expect(await screen.findByText("Import preview")).toBeInTheDocument()
}

describe("DictionariesManager import stage-1 preview workflow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    confirmDangerMock.mockResolvedValue(true)
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
              entry_count: 1
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  it("supports paste-based JSON preview followed by explicit confirm import", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "Import" }))
    await chooseSelectOption(user, "Source", "Paste content")

    fireEvent.change(
      screen.getByPlaceholderText("Paste JSON dictionary content..."),
      {
        target: {
          value: JSON.stringify({
            name: "JSON Preview Dict",
            entries: [
              {
                pattern: "BP",
                replacement: "blood pressure",
                type: "literal",
                group: "Clinical"
              }
            ]
          })
        }
      }
    )

    await user.click(screen.getByRole("button", { name: "Preview import" }))
    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(screen.getByText("JSON Preview Dict")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importDictionaryJSON).toHaveBeenCalledTimes(1)
    })
  }, 60000)

  it("supports markdown file preview and explicit confirm import", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "Import" }))
    await chooseSelectOption(user, "Format", "Markdown")

    const markdownFile = new File(
      [
        "# Markdown Import Dict\n\n## Entry: BP\n- **Type**: literal\n- **Replacement**: blood pressure\n"
      ],
      "markdown-import.md",
      { type: "text/markdown" }
    )
    const fileInput = document.querySelector(
      'input[type="file"]'
    ) as HTMLInputElement
    await user.upload(fileInput, markdownFile)

    await user.click(screen.getByRole("button", { name: "Preview import" }))
    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(screen.getByText("Markdown Import Dict")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))
    await waitFor(() => {
      expect(tldwClientMock.importDictionaryMarkdown).toHaveBeenCalledTimes(1)
    })
  }, 60000)

  it("offers rename resolution on 409 conflict and retries with suggested name", async () => {
    const user = userEvent.setup()
    tldwClientMock.importDictionaryJSON
      .mockRejectedValueOnce(new Error("409 conflict: dictionary already exists"))
      .mockResolvedValueOnce({ dictionary_id: 777 })

    render(<DictionariesManager />)
    await previewJsonImport(user, "Medical Terms")

    await user.click(screen.getByRole("button", { name: "Confirm import" }))
    expect(
      await screen.findByText("Dictionary name conflict")
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /Rename to "Medical Terms \(2\)"/ }))

    await waitFor(() => {
      expect(tldwClientMock.importDictionaryJSON).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.importDictionaryJSON).toHaveBeenLastCalledWith(
      expect.objectContaining({
        name: "Medical Terms (2)"
      }),
      false
    )
  }, 60000)

  it("supports replace-existing conflict resolution path", async () => {
    const user = userEvent.setup()
    tldwClientMock.importDictionaryJSON
      .mockRejectedValueOnce(new Error("409 conflict: dictionary already exists"))
      .mockResolvedValueOnce({ dictionary_id: 778 })

    render(<DictionariesManager />)
    await previewJsonImport(user, "Medical Terms")

    await user.click(screen.getByRole("button", { name: "Confirm import" }))
    expect(
      await screen.findByText("Dictionary name conflict")
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Replace existing" }))

    await waitFor(() => {
      expect(tldwClientMock.deleteDictionary).toHaveBeenCalledWith(77)
      expect(tldwClientMock.importDictionaryJSON).toHaveBeenCalledTimes(2)
    })
  }, 60000)

  it("allows canceling conflict resolution without retrying import", async () => {
    const user = userEvent.setup()
    tldwClientMock.importDictionaryJSON.mockRejectedValueOnce(
      new Error("409 conflict: dictionary already exists")
    )

    render(<DictionariesManager />)
    await previewJsonImport(user, "Medical Terms")

    await user.click(screen.getByRole("button", { name: "Confirm import" }))
    expect(
      await screen.findByText("Dictionary name conflict")
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(tldwClientMock.importDictionaryJSON).toHaveBeenCalledTimes(1)
    })
  }, 60000)

  it("warns before markdown export when advanced fields are present", async () => {
    const user = userEvent.setup()
    tldwClientMock.exportDictionaryJSON.mockResolvedValueOnce({
      name: "Medical Terms",
      entries: [
        {
          pattern: "BP",
          replacement: "blood pressure",
          probability: 0.5
        }
      ]
    })
    tldwClientMock.exportDictionaryMarkdown.mockResolvedValueOnce({
      content: "# Medical Terms"
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Export Medical Terms as Markdown"
      })
    )

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalled()
      expect(tldwClientMock.exportDictionaryMarkdown).toHaveBeenCalledTimes(1)
    })
  }, 60000)
})
