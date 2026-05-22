import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div
      data-testid={`route-boundary-${routeId}`}
      data-route-id={routeId}
      data-route-label={routeLabel}
    >
      {children}
    </div>
  )
}))

vi.mock("@/components/Option/Evaluations/EvaluationsPlaygroundPage", () => ({
  EvaluationsPlaygroundPage: () => (
    <div data-testid="evaluations-playground-page">Evaluations</div>
  )
}))

vi.mock("@/components/Flashcards/FlashcardsWorkspace", () => ({
  FlashcardsWorkspace: () => (
    <div data-testid="flashcards-workspace">Flashcards</div>
  )
}))

vi.mock("@/components/Quiz/QuizWorkspace", () => ({
  QuizWorkspace: () => <div data-testid="quiz-workspace">Quiz</div>
}))

vi.mock("@/components/ContentReview/ContentReviewPage", () => ({
  __esModule: true,
  default: () => <div data-testid="content-review-page">Content Review</div>
}))

vi.mock("@/components/Option/DataTables", () => ({
  DataTablesPage: () => <div data-testid="data-tables-page">Data Tables</div>
}))

vi.mock("@/components/Option/ChunkingPlayground", () => ({
  ChunkingPlayground: () => (
    <div data-testid="chunking-playground-page">Chunking</div>
  )
}))

vi.mock("@/components/Option/KanbanPlayground", () => ({
  KanbanPlayground: () => <div data-testid="kanban-playground">Kanban</div>
}))

import OptionEvaluations from "../option-evaluations"
import OptionFlashcards from "../option-flashcards"
import OptionQuiz from "../option-quiz"
import OptionModerationPlayground from "../option-moderation-playground"
import OptionContentReview from "../option-content-review"
import OptionDataTables from "../option-data-tables"
import OptionChunkingPlayground from "../option-chunking-playground"
import OptionKanbanPlayground from "../option-kanban-playground"

const sharedRouteCases = [
  {
    route: "/evaluations",
    Component: OptionEvaluations,
    routeId: "evaluations",
    routeLabel: "Evaluations",
    target: "evaluations-playground-page"
  },
  {
    route: "/flashcards",
    Component: OptionFlashcards,
    routeId: "flashcards",
    routeLabel: "Flashcards",
    target: "flashcards-workspace"
  },
  {
    route: "/quiz",
    Component: OptionQuiz,
    routeId: "quiz",
    routeLabel: "Quiz",
    target: "quiz-workspace"
  },
  {
    route: "/content-review",
    Component: OptionContentReview,
    routeId: "content-review",
    routeLabel: "Content Review",
    target: "content-review-page"
  },
  {
    route: "/data-tables",
    Component: OptionDataTables,
    routeId: "data-tables",
    routeLabel: "Data Tables",
    target: "data-tables-page"
  },
  {
    route: "/chunking-playground",
    Component: OptionChunkingPlayground,
    routeId: "chunking-playground",
    routeLabel: "Chunking Playground",
    target: "chunking-playground-page"
  },
  {
    route: "/kanban",
    Component: OptionKanbanPlayground,
    routeId: "kanban",
    routeLabel: "Kanban",
    target: "kanban-playground"
  }
] as const

describe("study, safety, and specialized route boundaries", () => {
  it.each(sharedRouteCases)(
    "keeps $route wrapped by its canonical shared page boundary",
    ({ Component, routeId, routeLabel, target }) => {
      render(<Component />)

      const boundary = screen.getByTestId(`route-boundary-${routeId}`)

      expect(screen.getByTestId("option-layout")).toBeVisible()
      expect(boundary).toHaveAttribute("data-route-id", routeId)
      expect(boundary).toHaveAttribute("data-route-label", routeLabel)
      expect(screen.getByTestId(target)).toBeVisible()
    }
  )

  it("keeps the legacy moderation playground route redirected to moderation rules", async () => {
    render(
      <MemoryRouter initialEntries={["/moderation-playground"]}>
        <Routes>
          <Route
            path="/moderation-playground"
            element={<OptionModerationPlayground />}
          />
          <Route
            path="/moderation/rules"
            element={<div data-testid="moderation-rules-target" />}
          />
        </Routes>
      </MemoryRouter>
    )

    expect(await screen.findByTestId("moderation-rules-target")).toBeVisible()
  })
})
