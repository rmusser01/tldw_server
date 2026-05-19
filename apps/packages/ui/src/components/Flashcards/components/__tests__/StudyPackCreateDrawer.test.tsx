import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { StudyPackCreateDrawer } from "../StudyPackCreateDrawer"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"

const navigateMock = vi.hoisted(() => vi.fn())
const createStudyPackJobMock = vi.hoisted(() => vi.fn())
const useStudyPackJobQueryMock = vi.hoisted(() => vi.fn())
const messageSpies = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("../../hooks", async () => {
  const actual = await vi.importActual<typeof import("../../hooks")>("../../hooks")
  return {
    ...actual,
    useStudyPackCreateMutation: () => ({
      mutateAsync: createStudyPackJobMock,
      isPending: false
    }),
    useStudyPackJobQuery: useStudyPackJobQueryMock
  }
})

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

const buildCompletedResponse = () => ({
  job: {
    id: 91,
    status: "completed",
    domain: "study_packs",
    queue: "default",
    job_type: "study_pack_generate"
  },
  study_pack: {
    id: 31,
    workspace_id: null,
    title: "Networks",
    deck_id: 8,
    source_bundle_json: {},
    generation_options_json: { deck_mode: "new" },
    status: "active",
    superseded_by_pack_id: null,
    created_at: "2026-04-02T18:00:00Z",
    last_modified: "2026-04-02T18:00:00Z",
    deleted: false,
    client_id: "study-pack-tests",
    version: 1
  },
  error: null
})

describe("StudyPackCreateDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStudyPackJobQueryMock.mockReturnValue({
      data: null,
      isFetching: false
    })
  })

  it("requires a title and at least one source before enabling submit", () => {
    render(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    expect(screen.getByRole("dialog", { name: "Create study pack" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create study pack" })).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Title"), {
      target: { value: "Networks" }
    })

    expect(screen.getByRole("button", { name: "Create study pack" })).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Source ID"), {
      target: { value: "42" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add source" }))

    expect(screen.getByText("media · 42")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create study pack" })).toBeEnabled()
  })

  it("prefills from an intent and navigates to review after a completed job", async () => {
    const intent: StudyPackIntent = {
      title: "Networks",
      sourceItems: [
        {
          sourceType: "media",
          sourceId: "42",
          sourceTitle: "Lecture 5"
        }
      ]
    }

    createStudyPackJobMock.mockResolvedValue({
      job: {
        id: 91,
        status: "queued",
        domain: "study_packs",
        queue: "default",
        job_type: "study_pack_generate"
      }
    })
    useStudyPackJobQueryMock.mockImplementation((jobId: number | null | undefined) =>
      jobId === 91
        ? {
            data: buildCompletedResponse(),
            isFetching: false
          }
        : {
            data: null,
            isFetching: false
          }
    )

    const onClose = vi.fn()

    render(<StudyPackCreateDrawer open onClose={onClose} initialIntent={intent} />)

    expect(screen.getByDisplayValue("Networks")).toBeInTheDocument()
    expect(screen.getByText("media · 42")).toBeInTheDocument()
    expect(screen.getByText("Lecture 5")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))

    await waitFor(() => {
      expect(createStudyPackJobMock).toHaveBeenCalledWith({
        title: "Networks",
        source_items: [
          {
            source_type: "media",
            source_id: "42",
            source_title: "Lecture 5"
          }
        ],
        deck_mode: "new"
      })
    })

    await waitFor(() => {
      expect(onClose).toHaveBeenCalledTimes(1)
      expect(navigateMock).toHaveBeenCalledWith("/flashcards?tab=review&deck_id=8", {
        replace: true
      })
    })
  })

  it("shows a friendly error when job creation fails", async () => {
    createStudyPackJobMock.mockRejectedValue(new Error("request failed"))

    render(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    fireEvent.change(screen.getByLabelText("Title"), {
      target: { value: "Networks" }
    })
    fireEvent.change(screen.getByLabelText("Source ID"), {
      target: { value: "42" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add source" }))
    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))

    await waitFor(() => {
      expect(messageSpies.error).toHaveBeenCalledWith("Study pack generation failed.")
    })
    expect(navigateMock).not.toHaveBeenCalled()
  })
})
