import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { FacetedFilters } from "../FacetedFilters"
import { PromptListToolbar } from "../PromptListToolbar"
import { DEFAULT_PROMPT_QUERY_STATE } from "../prompt-workspace-types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => `localized:${key}`
  })
}))

describe("prompt recipe localization", () => {
  it("uses translations for recipe facets", () => {
    render(
      <FacetedFilters
        typeFilter="all"
        onTypeFilterChange={vi.fn()}
        typeCounts={{}}
        syncFilter="all"
        onSyncFilterChange={vi.fn()}
        syncCounts={{}}
        tagFilter={[]}
        onTagFilterChange={vi.fn()}
        tagMatchMode="any"
        onTagMatchModeChange={vi.fn()}
        tagCounts={{}}
      />
    )

    expect(
      screen.getByText("localized:managePrompts.recipe.filters.all")
    ).toBeInTheDocument()
    expect(
      screen.getByText("localized:managePrompts.recipe.filters.system")
    ).toBeInTheDocument()
    expect(
      screen.getByText("localized:managePrompts.recipe.filters.user")
    ).toBeInTheDocument()
  })

  it("uses translations for the selected recipe toolbar option", () => {
    render(
      <PromptListToolbar
        query={{
          ...DEFAULT_PROMPT_QUERY_STATE,
          typeFilter: "recipe_system"
        }}
        allTags={[]}
        onQueryChange={vi.fn()}
      />
    )

    expect(
      screen.getByText("localized:managePrompts.recipe.filters.system")
    ).toBeInTheDocument()
  })
})
