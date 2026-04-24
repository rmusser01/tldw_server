import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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
    initialize: vi.fn(async () => undefined),
    getProviders: vi.fn(async () => ({
      default_provider: "openai",
      providers: [
        {
          name: "openai",
          models: ["gpt-4o-mini"]
        }
      ]
    })),
    createChatCompletion: vi.fn(async () =>
      new Response(
        JSON.stringify({
          choices: [
            {
              message: {
                content:
                  "north, wind -> Northern winds carry frost from the mountain pass."
              }
            }
          ]
        })
      )
    ),
    createWorldBook: vi.fn(async () => ({ id: 1 })),
    updateWorldBook: vi.fn(async () => ({})),
    deleteWorldBook: vi.fn(async () => ({})),
    listCharacters: vi.fn(async () => []),
    listCharacterWorldBooks: vi.fn(async () => []),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({})),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({}))
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

describe("WorldBooksManager information gaps stage-4 AI generation", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({
          data: [
            {
              id: 1,
              name: "Arcana",
              description: "Main lore",
              enabled: true,
              entry_count: 1,
              token_budget: 200
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({
          data: [
            {
              entry_id: 11,
              keywords: ["seed"],
              content: "Seed lore entry",
              priority: 40,
              enabled: true,
              case_sensitive: false,
              regex_match: false,
              whole_word_match: true,
              appendable: false
            }
          ],
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("generates, allows edits, and saves suggestions with provider/model metadata", async () => {
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    fireEvent.click(screen.getByText("Arcana"))
    fireEvent.click(await screen.findByRole("button", { name: "Generate entries with AI" }))

    fireEvent.change(screen.getByLabelText("AI generation topic"), {
      target: { value: "Northern weather lore" }
    })
    fireEvent.change(screen.getByLabelText("AI default group"), {
      target: { value: " Weather " }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run AI generation" }))

    await waitFor(() => {
      expect(screen.getByText("Generated with openai / gpt-4o-mini.")).toBeInTheDocument()
    })

    const generatedContent = screen.getByLabelText("Generated content 1")
    fireEvent.change(generatedContent, {
      target: { value: "Edited generated lore" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add generated suggestion 1" }))

    await waitFor(() => {
      expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledWith(
        1,
        expect.objectContaining({
          keywords: ["north", "wind"],
          content: "Edited generated lore",
          group: "Weather",
          metadata: expect.objectContaining({
            generated_with_ai: true,
            generated_provider: "openai",
            generated_model: "gpt-4o-mini",
            generated_topic: "Northern weather lore"
          })
        })
      )
    })
  }, 45000)

  it("shows an inline error when AI generation returns empty output", async () => {
    tldwClientMock.createChatCompletion.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "" } }]
        })
      )
    )

    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    fireEvent.click(screen.getByText("Arcana"))
    fireEvent.click(await screen.findByRole("button", { name: "Generate entries with AI" }))
    fireEvent.change(screen.getByLabelText("AI generation topic"), {
      target: { value: "Empty output case" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Run AI generation" }))

    await waitFor(() => {
      expect(screen.getByText("The model returned an empty result.")).toBeInTheDocument()
    })
    expect(tldwClientMock.addWorldBookEntry).not.toHaveBeenCalled()
  }, 30000)
})
