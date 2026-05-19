import { beforeEach, describe, expect, it, vi } from "vitest"

const mockBgRequest = vi.hoisted(() => vi.fn())
const listSpy = vi.hoisted(() => vi.fn())
const templateListSpy = vi.hoisted(() => vi.fn())
const templateGetSpy = vi.hoisted(() => vi.fn())
const templateCreateSpy = vi.hoisted(() => vi.fn())
const templateUpdateSpy = vi.hoisted(() => vi.fn())
const templateRemoveSpy = vi.hoisted(() => vi.fn())
const decksClientMock = vi.hoisted(() => ({
  list: listSpy,
  get: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  remove: vi.fn()
}))
const tagsClientMock = vi.hoisted(() => ({
  list: listSpy,
  get: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  remove: vi.fn()
}))
const flashcardsClientMock = vi.hoisted(() => ({
  list: vi.fn(),
  get: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  remove: vi.fn()
}))
const flashcardTemplatesClientMock = vi.hoisted(() => ({
  list: templateListSpy,
  get: templateGetSpy,
  create: templateCreateSpy,
  update: templateUpdateSpy,
  remove: templateRemoveSpy
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/services/resource-client", () => ({
  buildQuery: vi.fn(() => ""),
  createResourceClient: vi.fn(({ basePath }) => {
    if (String(basePath).includes("/flashcards/templates")) {
      return flashcardTemplatesClientMock
    }
    if (String(basePath).includes("/flashcards/tags")) {
      return tagsClientMock
    }
    if (String(basePath).includes("/flashcards")) {
      return String(basePath).includes("/decks") ? decksClientMock : flashcardsClientMock
    }
    return {
      list: vi.fn(),
      get: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      remove: vi.fn()
    }
  })
}))

import {
  FLASHCARD_GENERATION_TIMEOUT_MS,
  createFlashcardTemplate,
  deleteFlashcardTemplate,
  generateFlashcards,
  getFlashcardTemplate,
  listFlashcardTemplates,
  updateFlashcardTemplate,
  listFlashcardTagSuggestions
} from "@/services/flashcards"

describe("flashcards service", () => {
  beforeEach(() => {
    mockBgRequest.mockReset()
    listSpy.mockReset()
    templateListSpy.mockReset()
    templateGetSpy.mockReset()
    templateCreateSpy.mockReset()
    templateUpdateSpy.mockReset()
    templateRemoveSpy.mockReset()
    mockBgRequest.mockResolvedValue({ flashcards: [] })
    listSpy.mockResolvedValue({ items: [], count: 0 })
    templateListSpy.mockResolvedValue({ items: [], count: 0 })
    templateGetSpy.mockResolvedValue({})
    templateCreateSpy.mockResolvedValue({})
    templateUpdateSpy.mockResolvedValue({})
    templateRemoveSpy.mockResolvedValue({})
  })

  it("uses extended timeout by default for flashcard generation", async () => {
    await generateFlashcards({ text: "ATP powers the cell." })

    expect(mockBgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/flashcards/generate",
        method: "POST",
        timeoutMs: FLASHCARD_GENERATION_TIMEOUT_MS
      })
    )
  })

  it("calls the global tag suggestions endpoint with q and limit", async () => {
    const signal = new AbortController().signal

    await listFlashcardTagSuggestions({
      q: "bio",
      limit: 25,
      signal
    })

    expect(listSpy).toHaveBeenCalledWith(
      {
        q: "bio",
        limit: 25
      },
      {
        abortSignal: signal
      }
    )
  })

  it("omits blank q values when requesting global tag suggestions", async () => {
    await listFlashcardTagSuggestions({
      q: "   ",
      limit: 10
    })

    expect(listSpy).toHaveBeenCalledWith(
      {
        limit: 10
      },
      {
        abortSignal: undefined
      }
    )
  })

  it("calls the flashcard templates list endpoint", async () => {
    await listFlashcardTemplates()

    expect(templateListSpy).toHaveBeenCalledWith({}, { abortSignal: undefined })
  })

  it("calls the flashcard templates detail endpoint", async () => {
    await getFlashcardTemplate(17)

    expect(templateGetSpy).toHaveBeenCalledWith(17, undefined, { abortSignal: undefined })
  })

  it("calls the flashcard templates create endpoint", async () => {
    await createFlashcardTemplate({
      name: "Vocabulary Definition",
      model_type: "basic",
      front_template: "What does {{term}} mean?",
      back_template: "{{definition}}",
      placeholder_definitions: []
    })

    expect(templateCreateSpy).toHaveBeenCalledWith(
      expect.objectContaining({ name: "Vocabulary Definition" }),
      expect.objectContaining({ abortSignal: undefined })
    )
  })

  it("calls the flashcard templates update endpoint", async () => {
    await updateFlashcardTemplate(17, {
      name: "Renamed",
      expected_version: 2
    })

    expect(templateUpdateSpy).toHaveBeenCalledWith(
      "17",
      expect.objectContaining({ name: "Renamed" }),
      expect.objectContaining({ abortSignal: undefined })
    )
  })

  it("calls the flashcard templates delete endpoint", async () => {
    await deleteFlashcardTemplate(17, 3)

    expect(templateRemoveSpy).toHaveBeenCalledWith(
      "17",
      { expected_version: 3 },
      expect.objectContaining({ abortSignal: undefined })
    )
  })
})
