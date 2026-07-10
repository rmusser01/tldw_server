import {
  createDeck,
  createFlashcard,
  generateFlashcards,
  listDecks,
  listFlashcards
} from "@/services/flashcards"
import { tldwClient } from "@/services/tldw"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useGenerateQuizMutation } from "../../hooks"
import { GenerateTab, type SourceReviewQuizIntent } from "../GenerateTab"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue !== "string") return key
      return defaultValue.replace(
        /\{\{\s*([^\s}]+)\s*\}\}/g,
        (_, name: string) => {
          const value = defaultValueOrOptions?.[name]
          return value == null ? "" : String(value)
        }
      )
    }
  })
}))

vi.mock("../../hooks", () => ({
  useGenerateQuizMutation: vi.fn()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    listMedia: vi.fn(),
    searchMedia: vi.fn(),
    getMediaDetails: vi.fn(),
    listNotes: vi.fn(),
    searchNotes: vi.fn()
  }
}))

vi.mock("@/services/flashcards", () => ({
  generateFlashcards: vi.fn(),
  createDeck: vi.fn(),
  createFlashcard: vi.fn(),
  listDecks: vi.fn(),
  listFlashcards: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const renderGenerateTab = (
  initialSourceReviewIntent: SourceReviewQuizIntent
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <GenerateTab
        initialSourceReviewIntent={initialSourceReviewIntent}
        onNavigateToTake={() => {}}
      />
    </QueryClientProvider>
  )
}

describe("GenerateTab source-review handoff", () => {
  const mutateAsync = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 19, title: "Neuroanatomy atlas", type: "pdf" }],
      pagination: { total_items: 1 }
    } as any)
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({} as any)
    vi.mocked(tldwClient.listNotes).mockResolvedValue({
      items: [{ id: "note-8", title: "Lecture notes" }]
    } as any)
    vi.mocked(tldwClient.searchNotes).mockResolvedValue({ items: [] } as any)
    vi.mocked(listDecks).mockResolvedValue([] as any)
    vi.mocked(listFlashcards).mockResolvedValue({ items: [], count: 0 } as any)
    vi.mocked(generateFlashcards).mockResolvedValue({
      flashcards: [],
      count: 0
    } as any)
    vi.mocked(createDeck).mockResolvedValue({ id: 1, name: "Deck" } as any)
    vi.mocked(createFlashcard).mockResolvedValue({ uuid: "card-1" } as any)
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)
  })

  it("preselects media and note references while keeping snapshots read-only", async () => {
    renderGenerateTab({
      payload: {
        occurrence_id: 42,
        plan_id: 7,
        plan_title: "Neuro review",
        activity_type: "quiz",
        source_bundle: {
          items: [
            {
              source_type: "media",
              source_id: "19",
              label: "Neuroanatomy atlas"
            },
            {
              source_type: "note",
              source_id: "note-8",
              label: "Lecture notes"
            },
            {
              source_type: "message",
              source_id: "message-3",
              label: "Tutor discussion",
              excerpt_text: "The dorsal columns carry fine touch."
            },
            {
              source_type: "unsupported" as "message",
              source_id: "external-4",
              label: "External snapshot",
              excerpt_text: "Snapshot-only detail"
            }
          ]
        }
      },
      error: null
    })

    await waitFor(() => {
      expect(screen.getAllByText("2 sources selected").length).toBeGreaterThan(
        0
      )
    })
    expect(screen.getByText("Neuroanatomy atlas")).toBeInTheDocument()
    expect(screen.getAllByText("Lecture notes").length).toBeGreaterThan(0)
    expect(
      screen.getByTestId("generate-source-review-snapshots")
    ).toHaveTextContent("Tutor discussion")
    expect(
      screen.getByTestId("generate-source-review-snapshots")
    ).toHaveTextContent("The dorsal columns carry fine touch.")
    expect(
      screen.getByTestId("generate-source-review-snapshots")
    ).toHaveTextContent("External snapshot")
    expect(mutateAsync).not.toHaveBeenCalled()
  })

  it("keeps media snapshots whose source IDs cannot be selected", async () => {
    renderGenerateTab({
      payload: {
        occurrence_id: 42,
        plan_id: 7,
        activity_type: "quiz",
        source_bundle: {
          items: [
            {
              source_type: "media",
              source_id: "legacy-media-id",
              label: "Imported anatomy figure",
              excerpt_text: "Figure snapshot retained for grounding."
            }
          ]
        }
      },
      error: null
    })

    await waitFor(() =>
      expect(
        screen.getByTestId("generate-source-review-snapshots")
      ).toHaveTextContent("Imported anatomy figure")
    )
    expect(
      screen.getByTestId("generate-source-review-snapshots")
    ).toHaveTextContent("Figure snapshot retained for grounding.")
  })

  it("shows a recoverable message when the token cannot be loaded", () => {
    renderGenerateTab({ payload: null, error: "expired_or_missing" })

    expect(
      screen.getByTestId("generate-source-review-handoff-error")
    ).toHaveTextContent("expired or is unavailable")
    expect(
      screen.getByRole("button", { name: /Generate Quiz/i })
    ).toBeDisabled()
    expect(mutateAsync).not.toHaveBeenCalled()
  })
})
