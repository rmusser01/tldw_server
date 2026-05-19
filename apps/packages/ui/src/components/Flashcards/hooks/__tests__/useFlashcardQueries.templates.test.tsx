import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  useCreateFlashcardTemplateMutation,
  useDeleteFlashcardTemplateMutation,
  useFlashcardTemplateQuery,
  useFlashcardTemplatesQuery,
  useUpdateFlashcardTemplateMutation
} from "../useFlashcardQueries"

const templateListSpy = vi.hoisted(() => vi.fn())
const templateGetSpy = vi.hoisted(() => vi.fn())
const templateCreateSpy = vi.hoisted(() => vi.fn())
const templateUpdateSpy = vi.hoisted(() => vi.fn())
const templateRemoveSpy = vi.hoisted(() => vi.fn())

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasFlashcards: true },
    loading: false
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards"
  )
  const noopAsync = vi.fn()
  return {
    ...actual,
    listDecks: noopAsync,
    listFlashcards: noopAsync,
    listFlashcardTagSuggestions: noopAsync,
    listFlashcardTemplates: templateListSpy,
    getFlashcardTemplate: templateGetSpy,
    createFlashcardTemplate: templateCreateSpy,
    updateFlashcardTemplate: templateUpdateSpy,
    deleteFlashcardTemplate: templateRemoveSpy,
    createFlashcard: noopAsync,
    createFlashcardsBulk: noopAsync,
    updateFlashcardsBulk: noopAsync,
    createDeck: noopAsync,
    updateDeck: noopAsync,
    updateFlashcard: noopAsync,
    deleteFlashcard: noopAsync,
    resetFlashcardScheduling: noopAsync,
    reviewFlashcard: noopAsync,
    getNextReviewCard: noopAsync,
    getFlashcardAssistant: noopAsync,
    respondFlashcardAssistant: noopAsync,
    generateFlashcards: noopAsync,
    getFlashcard: noopAsync,
    importFlashcards: noopAsync,
    previewStructuredQaImport: noopAsync,
    importFlashcardsJson: noopAsync,
    importFlashcardsApkg: noopAsync,
    getFlashcardsAnalyticsSummary: noopAsync,
    exportFlashcards: noopAsync,
    exportFlashcardsFile: noopAsync,
    getFlashcardsImportLimits: noopAsync
  }
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useFlashcard template queries", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    templateListSpy.mockResolvedValue({ items: [], count: 0 })
    templateGetSpy.mockResolvedValue({})
    templateCreateSpy.mockResolvedValue({})
    templateUpdateSpy.mockResolvedValue({})
    templateRemoveSpy.mockResolvedValue({})
  })

  it("uses the shared flashcard templates list key", () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })

    renderHook(() => useFlashcardTemplatesQuery(), {
      wrapper: buildWrapper(queryClient)
    })

    const keys = queryClient
      .getQueryCache()
      .findAll()
      .map((query) => query.queryKey)

    expect(keys).toContainEqual(["flashcards:templates"])
  })

  it("calls the flashcard templates list endpoint", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })

    renderHook(() => useFlashcardTemplatesQuery(), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(templateListSpy).toHaveBeenCalledWith({})
    })
  })

  it("invalidates template caches after create update and delete mutations", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    const setQueryDataSpy = vi.spyOn(queryClient, "setQueryData")
    const removeQueriesSpy = vi.spyOn(queryClient, "removeQueries")

    vi.mocked(templateCreateSpy).mockResolvedValue({
      id: 9,
      name: "Vocabulary Definition",
      model_type: "basic",
      front_template: "What does {{term}} mean?",
      back_template: "{{definition}}",
      placeholder_definitions: [],
      deleted: false,
      client_id: "test",
      version: 1
    })
    vi.mocked(templateUpdateSpy).mockResolvedValue({
      id: 9,
      name: "Renamed",
      model_type: "basic",
      front_template: "What does {{term}} mean?",
      back_template: "{{definition}}",
      placeholder_definitions: [],
      deleted: false,
      client_id: "test",
      version: 2
    })
    vi.mocked(templateRemoveSpy).mockResolvedValue(true)

    const create = renderHook(() => useCreateFlashcardTemplateMutation(), {
      wrapper: buildWrapper(queryClient)
    }).result.current
    const update = renderHook(() => useUpdateFlashcardTemplateMutation(), {
      wrapper: buildWrapper(queryClient)
    }).result.current
    const remove = renderHook(() => useDeleteFlashcardTemplateMutation(), {
      wrapper: buildWrapper(queryClient)
    }).result.current

    await act(async () => {
      await create.mutateAsync({
        name: "Vocabulary Definition",
        model_type: "basic",
        front_template: "What does {{term}} mean?",
        back_template: "{{definition}}",
        placeholder_definitions: []
      })
      await update.mutateAsync({
        templateId: 9,
        update: {
          name: "Renamed",
          expected_version: 1
        }
      })
      await remove.mutateAsync({
        templateId: 9,
        expectedVersion: 2
      })
    })

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["flashcards:templates"] })
    expect(setQueryDataSpy).toHaveBeenCalledWith(
      ["flashcards:templates", 9],
      expect.objectContaining({ id: 9, name: "Vocabulary Definition" })
    )
    expect(setQueryDataSpy).toHaveBeenCalledWith(
      ["flashcards:templates", 9],
      expect.objectContaining({ id: 9, name: "Renamed" })
    )
    expect(removeQueriesSpy).toHaveBeenCalledWith({
      queryKey: ["flashcards:templates", 9]
    })
  })

  it("uses the template detail query key", () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })

    renderHook(() => useFlashcardTemplateQuery(17), {
      wrapper: buildWrapper(queryClient)
    })

    const keys = queryClient
      .getQueryCache()
      .findAll()
      .map((query) => query.queryKey)

    expect(keys).toContainEqual(["flashcards:templates", 17])
  })
})
