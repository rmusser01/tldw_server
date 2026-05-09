import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { DiffViewer, type FileDiff } from "../DiffViewer"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

const makeDiff = (overrides: Partial<FileDiff>): FileDiff => ({
  id: overrides.id ?? "file-1",
  oldPath: overrides.oldPath ?? "src/example.ts",
  newPath: overrides.newPath ?? "src/example.ts",
  hunks: [
    {
      id: `${overrides.id ?? "file-1"}-hunk-1`,
      oldStart: 1,
      oldCount: 1,
      newStart: 1,
      newCount: 1,
      lines: [
        {
          type: "header",
          content: "@@ -1 +1 @@"
        }
      ]
    }
  ],
  ...overrides
})

const badgeFor = (label: string) => {
  const badge = screen.getByText(label).closest('[data-ds-component="Badge"]')
  expect(badge).not.toBeNull()
  return badge
}

describe("DiffViewer file status badges", () => {
  it("renders file status labels through the shared Badge primitive", () => {
    render(
      <DiffViewer
        diffs={[
          makeDiff({
            id: "new-file",
            oldPath: "/dev/null",
            newPath: "src/new.ts",
            isNew: true
          }),
          makeDiff({
            id: "deleted-file",
            oldPath: "src/deleted.ts",
            newPath: "/dev/null",
            isDeleted: true
          }),
          makeDiff({
            id: "renamed-file",
            oldPath: "src/old-name.ts",
            newPath: "src/new-name.ts",
            isRenamed: true
          }),
          makeDiff({
            id: "modified-file",
            oldPath: "src/modified.ts",
            newPath: "src/modified.ts"
          })
        ]}
      />
    )

    expect(badgeFor("NEW")).toHaveAttribute("data-ds-variant", "success")
    expect(badgeFor("DEL")).toHaveAttribute("data-ds-variant", "danger")
    expect(badgeFor("RENAME")).toHaveAttribute("data-ds-variant", "secondary")
    expect(screen.queryByText("MOD")).not.toBeInTheDocument()
  })
})
