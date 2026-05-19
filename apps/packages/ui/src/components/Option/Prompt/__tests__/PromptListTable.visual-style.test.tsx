import React from "react"
import { describe, expect, it, vi } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { PromptListTable } from "../PromptListTable"
import type { PromptListQueryState, PromptRowVM } from "../prompt-workspace-types"

vi.mock("antd", () => ({
  Table: (props: any) => (
    <div
      data-testid={props["data-testid"] || "mock-prompt-table"}
      data-classname={props.className || ""}
      data-bordered={props.bordered ? "true" : "false"}
    />
  )
}))

vi.mock("../prompt-table-columns", () => ({
  buildPromptTableColumns: () => []
}))

const baseQuery: PromptListQueryState = {
  searchText: "",
  typeFilter: "all",
  syncFilter: "all",
  usageFilter: "all",
  tagFilter: [],
  tagMatchMode: "any",
  sort: {
    key: null,
    order: null
  },
  page: 1,
  pageSize: 20,
  savedView: "all"
}

const baseRows: PromptRowVM[] = [
  {
    id: "prompt-1",
    title: "Prompt 1",
    keywords: [],
    favorite: false,
    syncStatus: "local",
    sourceSystem: "workspace",
    createdAt: Date.now(),
    usageCount: 0
  }
]

describe("PromptListTable visual hooks", () => {
  it("uses prompt table styling classes with default comfortable density", () => {
    const markup = renderToStaticMarkup(
      <PromptListTable
        rows={baseRows}
        total={1}
        isOnline
        isCompactViewport={false}
        query={baseQuery}
        selectedIds={[]}
        onQueryChange={() => undefined}
        onSelectionChange={() => undefined}
        onRowOpen={() => undefined}
      />
    )

    expect(markup).toContain("prompts-table")
    expect(markup).toContain("prompts-table-density-comfortable")
  })

  it("uses the requested density class", () => {
    const markup = renderToStaticMarkup(
      <PromptListTable
        rows={baseRows}
        total={1}
        isOnline
        isCompactViewport={false}
        query={baseQuery}
        selectedIds={[]}
        onQueryChange={() => undefined}
        onSelectionChange={() => undefined}
        onRowOpen={() => undefined}
        tableDensity="dense"
      />
    )

    expect(markup).toContain("prompts-table-density-dense")
  })

  it("does not render the bordered prop on the table", () => {
    const markup = renderToStaticMarkup(
      <PromptListTable
        rows={baseRows}
        total={1}
        isOnline
        isCompactViewport={false}
        query={baseQuery}
        selectedIds={[]}
        onQueryChange={() => undefined}
        onSelectionChange={() => undefined}
        onRowOpen={() => undefined}
      />
    )

    expect(markup).toContain('data-bordered="false"')
  })

  it("announces zero selected prompts to screen readers", () => {
    const markup = renderToStaticMarkup(
      <PromptListTable
        rows={baseRows}
        total={1}
        isOnline
        isCompactViewport={false}
        query={baseQuery}
        selectedIds={[]}
        onQueryChange={() => undefined}
        onSelectionChange={() => undefined}
        onRowOpen={() => undefined}
      />
    )

    expect(markup).toContain("0 prompts selected")
  })

  it("announces singular selection counts to screen readers", () => {
    const markup = renderToStaticMarkup(
      <PromptListTable
        rows={baseRows}
        total={1}
        isOnline
        isCompactViewport={false}
        query={baseQuery}
        selectedIds={["prompt-1"]}
        onQueryChange={() => undefined}
        onSelectionChange={() => undefined}
        onRowOpen={() => undefined}
      />
    )

    expect(markup).toContain("1 prompt selected")
  })
})
