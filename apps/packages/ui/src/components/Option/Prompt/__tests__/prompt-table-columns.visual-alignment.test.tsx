import React from "react"
import { describe, expect, it } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { buildPromptTableColumns } from "../prompt-table-columns"
import type { PromptRowVM } from "../prompt-workspace-types"

const sampleRow: PromptRowVM = {
  id: "prompt-1",
  title: "Sample prompt",
  author: "Author",
  details: "Details",
  previewSystem: "System prompt preview",
  previewUser: "User prompt preview",
  keywords: ["alpha", "beta", "gamma"],
  favorite: false,
  syncStatus: "local",
  sourceSystem: "workspace",
  updatedAt: Date.now(),
  createdAt: Date.now(),
  usageCount: 7,
  lastUsedAt: Date.now()
}

describe("prompt-table-columns visual alignment", () => {
  it("uses characters-like column widths", () => {
    const columns = buildPromptTableColumns({
      isOnline: true,
      isCompactViewport: false,
      sortKey: null,
      sortOrder: null
    })

    const getColumn = (key: string) =>
      columns.find((column: any) => String(column?.key) === key) as any

    expect(getColumn("favorite")?.width).toBe(48)
    expect(getColumn("title")?.width).toBe(360)
    expect(getColumn("preview")?.width).toBe(320)
    expect(getColumn("keywords")?.width).toBe(220)
    expect(getColumn("actions")?.width).toBe(210)
  })

  it("renders usage badge with characters-like color weight", () => {
    const columns = buildPromptTableColumns({
      isOnline: true,
      isCompactViewport: false,
      sortKey: null,
      sortOrder: null
    })

    const titleColumn = columns.find(
      (column: any) => String(column?.key) === "title"
    ) as any
    const titleNode = titleColumn?.render?.(sampleRow.title, sampleRow)
    const titleMarkup = renderToStaticMarkup(
      <>{titleNode as React.ReactNode}</>
    )

    expect(titleMarkup).toContain("bg-primary/10")
    expect(titleMarkup).toContain("text-xs")
    expect(titleMarkup).not.toContain("text-[11px]")
  })

  it("renders a recipe badge and target in the title column", () => {
    const columns = buildPromptTableColumns({
      isOnline: true,
      isCompactViewport: false,
      sortKey: null,
      sortOrder: null
    })
    const titleColumn = columns.find(
      (column: any) => String(column?.key) === "title"
    ) as any
    const titleMarkup = renderToStaticMarkup(
      <>
        {titleColumn?.render?.(sampleRow.title, {
          ...sampleRow,
          kind: "recipe",
          recipeTarget: "user"
        }) as React.ReactNode}
      </>
    )

    expect(titleMarkup).toContain("Recipe")
    expect(titleMarkup).toContain("User")
  })

  it("uses supplied localized recipe badge labels", () => {
    const columns = buildPromptTableColumns({
      isOnline: true,
      isCompactViewport: false,
      sortKey: null,
      sortOrder: null,
      labels: {
        recipe: "Localized recipe",
        recipeSystem: "Localized system recipe",
        recipeUser: "Localized user recipe"
      }
    })
    const titleColumn = columns.find(
      (column: any) => String(column?.key) === "title"
    ) as any
    const titleMarkup = renderToStaticMarkup(
      <>
        {
          titleColumn?.render?.(sampleRow.title, {
            ...sampleRow,
            kind: "recipe",
            recipeTarget: "system"
          }) as React.ReactNode
        }
      </>
    )

    expect(titleMarkup).toContain("Localized recipe")
    expect(titleMarkup).toContain("Localized system recipe")
  })
})
