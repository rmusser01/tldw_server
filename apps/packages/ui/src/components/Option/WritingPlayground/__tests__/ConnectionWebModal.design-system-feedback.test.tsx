// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

type QueryResult = {
  data?: unknown
  isLoading?: boolean
}

const mockState = vi.hoisted(() => ({
  activeProjectId: null as string | null,
  queryResults: [] as QueryResult[],
}))

vi.mock("@/store/writing-playground", () => ({
  useWritingPlaygroundStore: () => ({
    activeProjectId: mockState.activeProjectId,
  }),
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn(
    () =>
      mockState.queryResults.shift() ?? {
        data: undefined,
        isLoading: false,
      },
  ),
}))

import { ConnectionWebModal } from "../modals/ConnectionWebModal"

const CLOSED_HANDLER = vi.fn()

const setQueryResults = (results: QueryResult[]) => {
  mockState.queryResults = results
}

const renderModal = () => {
  render(<ConnectionWebModal open onClose={CLOSED_HANDLER} />)
}

describe("ConnectionWebModal design-system feedback states", () => {
  beforeEach(() => {
    CLOSED_HANDLER.mockReset()
    mockState.activeProjectId = null
    setQueryResults([])
  })

  it("renders the project-required state through EmptyState", () => {
    renderModal()

    expect(screen.getByText("Select a project first")).toBeInTheDocument()
    expect(
      screen
        .getByText("Select a project first")
        .closest('[data-ds-component="EmptyState"]'),
    ).toBeInTheDocument()
  })

  it("renders the loading branch through LoadingState", () => {
    mockState.activeProjectId = "project-1"
    setQueryResults([
      { data: undefined, isLoading: true },
      { data: undefined, isLoading: false },
      { data: undefined, isLoading: false },
    ])

    renderModal()

    expect(
      document.body.querySelector('[data-ds-component="LoadingState"]'),
    ).toBeInTheDocument()
  })

  it("renders the no-data state through EmptyState", () => {
    mockState.activeProjectId = "project-1"
    setQueryResults([
      { data: { characters: [] }, isLoading: false },
      { data: { relationships: [] }, isLoading: false },
      { data: { items: [] }, isLoading: false },
    ])

    renderModal()

    expect(
      screen.getByText("Add characters or world info to visualize connections"),
    ).toBeInTheDocument()
    expect(
      screen
        .getByText("Add characters or world info to visualize connections")
        .closest('[data-ds-component="EmptyState"]'),
    ).toBeInTheDocument()
  })
})
