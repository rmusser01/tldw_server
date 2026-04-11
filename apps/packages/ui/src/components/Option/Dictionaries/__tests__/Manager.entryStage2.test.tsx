import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    getDictionary: vi.fn(async () => ({})),
    listDictionaryEntries: vi.fn(async () => ({ entries: [] })),
    addDictionaryEntry: vi.fn(async () => ({ id: 901 })),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    validateDictionary: vi.fn(async () => ({ ok: true, errors: [], warnings: [] }))
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

const getNumberInputByLabel = (
  label: string,
  options?: { scope?: HTMLElement; occurrence?: "first" | "last" }
): HTMLInputElement => {
  const container = options?.scope ?? document.body
  const labelNodes = within(container).getAllByText(label)
  const labelNode =
    options?.occurrence === "last"
      ? labelNodes[labelNodes.length - 1]
      : labelNodes[0]
  const formItem = labelNode.closest(".ant-form-item")
  if (!formItem) {
    throw new Error(`Could not resolve form item for label: ${label}`)
  }
  const input = formItem.querySelector("input")
  if (!input) {
    throw new Error(`Could not resolve numeric input for label: ${label}`)
  }
  return input as HTMLInputElement
}

describe("DictionariesManager entry stage-2 editing and validation flows", () => {
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
              entry_count: 1
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
              case_sensitive: false,
              timed_effects: {
                sticky: 2,
                cooldown: 0,
                delay: 0
              },
              max_replacements: 0
            }
          ]
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("supports inline pattern/replacement commit via enter/blur and cancel via escape", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    const inlinePatternButton = await screen.findByRole("button", {
      name: "Inline edit pattern BP"
    })
    await user.click(inlinePatternButton)
    const inlinePatternInput = await screen.findByLabelText(
      "Inline edit pattern for BP"
    )
    await user.clear(inlinePatternInput)
    await user.type(inlinePatternInput, "BP-updated{enter}")

    await waitFor(() => {
      expect(tldwClientMock.updateDictionaryEntry).toHaveBeenCalledWith(1, {
        pattern: "BP-updated"
      })
    })

    await user.click(
      await screen.findByRole("button", { name: "Inline edit replacement BP" })
    )
    const inlineReplacementInput = await screen.findByLabelText(
      "Inline edit replacement for BP"
    )
    await user.clear(inlineReplacementInput)
    await user.type(inlineReplacementInput, "blood pressure updated")
    fireEvent.blur(inlineReplacementInput)

    await waitFor(() => {
      expect(tldwClientMock.updateDictionaryEntry).toHaveBeenCalledWith(1, {
        replacement: "blood pressure updated"
      })
    })

    await user.click(
      await screen.findByRole("button", { name: "Inline edit replacement BP" })
    )
    const cancelledInput = await screen.findByLabelText(
      "Inline edit replacement for BP"
    )
    await user.clear(cancelledInput)
    await user.type(cancelledInput, "do not save")
    fireEvent.keyDown(cancelledInput, { key: "Escape" })

    await waitFor(() => {
      expect(tldwClientMock.updateDictionaryEntry).toHaveBeenCalledTimes(2)
    })
  }, 60000)

  it("sends timed effects in add and edit payloads", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("button", { name: "Add Entry" })

    await user.click(screen.getByRole("button", { name: "Advanced options" }))
    await user.type(
      screen.getByPlaceholderText("e.g., gonna or /colour/i"),
      "Temp"
    )
    await user.type(
      screen.getByPlaceholderText("e.g., going to or color"),
      "Temperature"
    )

    const stickyInput = getNumberInputByLabel("Sticky (seconds)")
    const cooldownInput = getNumberInputByLabel("Cooldown (seconds)")
    const delayInput = getNumberInputByLabel("Delay (seconds)")
    await user.clear(stickyInput)
    await user.type(stickyInput, "30")
    await user.clear(cooldownInput)
    await user.type(cooldownInput, "12")
    await user.clear(delayInput)
    await user.type(delayInput, "5")

    await user.click(screen.getByRole("button", { name: "Add Entry" }))

    await waitFor(() => {
      expect(tldwClientMock.addDictionaryEntry).toHaveBeenCalledWith(
        77,
        expect.objectContaining({
          pattern: "Temp",
          replacement: "Temperature",
          timed_effects: {
            sticky: 30,
            cooldown: 12,
            delay: 5
          }
        })
      )
    }, { timeout: 20000 })

    await user.click(screen.getByRole("button", { name: "Edit entry BP" }))

    const editStickyInput = getNumberInputByLabel("Sticky (seconds)", {
      occurrence: "last"
    })
    const editCooldownInput = getNumberInputByLabel("Cooldown (seconds)", {
      occurrence: "last"
    })
    const editDelayInput = getNumberInputByLabel("Delay (seconds)", {
      occurrence: "last"
    })
    await user.clear(editStickyInput)
    await user.type(editStickyInput, "45")
    await user.clear(editCooldownInput)
    await user.type(editCooldownInput, "10")
    await user.clear(editDelayInput)
    await user.type(editDelayInput, "2")
    await user.click(screen.getByRole("button", { name: "Save Changes" }))

    await waitFor(() => {
      expect(tldwClientMock.updateDictionaryEntry).toHaveBeenCalledWith(
        1,
        expect.objectContaining({
          timed_effects: {
            sticky: 45,
            cooldown: 10,
            delay: 2
          }
        })
      )
    }, { timeout: 20000 })
  }, 90000)

  it("preserves advanced timed-effect values when toggling simple/advanced mode", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("button", { name: "Add Entry" })

    await user.click(screen.getByRole("button", { name: "Advanced options" }))

    const stickyInput = getNumberInputByLabel("Sticky (seconds)")
    const cooldownInput = getNumberInputByLabel("Cooldown (seconds)")
    const delayInput = getNumberInputByLabel("Delay (seconds)")

    await user.clear(stickyInput)
    await user.type(stickyInput, "17")
    await user.clear(cooldownInput)
    await user.type(cooldownInput, "8")
    await user.clear(delayInput)
    await user.type(delayInput, "3")

    await user.click(screen.getByRole("button", { name: "Simple mode" }))
    await user.click(screen.getByRole("button", { name: "Advanced options" }))

    expect(getNumberInputByLabel("Sticky (seconds)")).toHaveValue("17")
    expect(getNumberInputByLabel("Cooldown (seconds)")).toHaveValue("8")
    expect(getNumberInputByLabel("Delay (seconds)")).toHaveValue("3")
  }, 60000)

  it("defaults add-entry submissions to case-insensitive matching", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("button", { name: "Add Entry" })
    await user.type(
      screen.getByPlaceholderText("e.g., gonna or /colour/i"),
      "Temp"
    )
    await user.type(
      screen.getByPlaceholderText("e.g., going to or color"),
      "Temperature"
    )
    await user.click(screen.getByRole("button", { name: "Add Entry" }))

    await waitFor(() => {
      expect(tldwClientMock.addDictionaryEntry).toHaveBeenCalledWith(
        77,
        expect.objectContaining({
          pattern: "Temp",
          replacement: "Temperature",
          case_sensitive: false
        })
      )
    })
  }, 60000)

  it("pairs probability controls with frequency guidance", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("button", { name: "Add Entry" })
    await user.click(screen.getByRole("button", { name: "Advanced options" }))

    expect(screen.getAllByRole("slider").length).toBeGreaterThan(0)
    expect(
      screen.getByText("Fires ~10 out of 10 messages (100%).")
    ).toBeInTheDocument()

    const probabilityInput = getNumberInputByLabel("Probability", {
      occurrence: "last"
    })
    await user.clear(probabilityInput)
    await user.type(probabilityInput, "0.3")

    await waitFor(() => {
      expect(
        screen.getByText("Fires ~3 out of 10 messages (30%).")
      ).toBeInTheDocument()
    })
  }, 60000)

  it("shows server regex safety feedback before add save", async () => {
    const user = userEvent.setup()
    tldwClientMock.validateDictionary.mockResolvedValueOnce({
      ok: false,
      errors: [
        {
          code: "regex_unsafe",
          field: "entries[0].pattern",
          message: "Potentially dangerous regex pattern detected."
        }
      ],
      warnings: []
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByRole("button", { name: "Add Entry" })

    await user.click(screen.getByRole("combobox", { name: "Match type" }))
    await user.click(
      await screen.findByText("Regex (pattern match)", {
        selector: ".ant-select-item-option-content"
      })
    )
    expect(
      await screen.findByText(/Regex helper: `\.\*` = any text/i)
    ).toBeInTheDocument()
    await user.type(
      screen.getByPlaceholderText("e.g., gonna or /colour/i"),
      "(.+)+"
    )
    await user.type(
      screen.getByPlaceholderText("e.g., going to or color"),
      "x"
    )
    await user.click(screen.getByRole("button", { name: "Add Entry" }))

    await waitFor(() => {
      expect(tldwClientMock.validateDictionary).toHaveBeenCalledTimes(1)
      expect(tldwClientMock.addDictionaryEntry).not.toHaveBeenCalled()
    })

    expect(
      screen.getAllByText("Potentially dangerous regex pattern detected.")
        .length
    ).toBeGreaterThan(0)
  }, 60000)
})
