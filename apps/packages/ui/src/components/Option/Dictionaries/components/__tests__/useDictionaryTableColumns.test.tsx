// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"

import { useDictionaryTableColumns } from "../useDictionaryTableColumns"

const actionsCellSpy = vi.fn()

vi.mock("../DictionaryActionsCell", () => ({
  DictionaryActionsCell: (props: Record<string, unknown>) => {
    actionsCellSpy(props)
    return <div data-testid="dictionary-actions-cell" />
  }
}))

vi.mock("../DictionaryValidationStatusCell", () => ({
  DictionaryValidationStatusCell: () => null
}))

function Harness({
  useCompactDictionaryActions
}: {
  useCompactDictionaryActions: boolean
}) {
  const columns = useDictionaryTableColumns({
    activeUpdateMap: {},
    validationStatus: {},
    useCompactDictionaryActions,
    onToggleActive: vi.fn(),
    onValidateDictionary: vi.fn(),
    onOpenChatContext: vi.fn(),
    onOpenEdit: vi.fn(),
    onOpenEntries: vi.fn(),
    onOpenQuickAssign: vi.fn(),
    onExportJson: vi.fn(),
    onExportMarkdown: vi.fn(),
    onOpenStats: vi.fn(),
    onOpenVersions: vi.fn(),
    onDuplicate: vi.fn(),
    onDelete: vi.fn()
  })
  const actionsColumn = columns.find((column) => column.key === "actions")

  return <>{actionsColumn?.render?.(null, { id: 7, name: "Clinical Terms" })}</>
}

describe("useDictionaryTableColumns", () => {
  beforeEach(() => {
    actionsCellSpy.mockReset()
  })

  it("passes compact-mode preference through to the actions cell", () => {
    const { rerender } = render(
      <Harness useCompactDictionaryActions={false} />
    )

    expect(screen.getByTestId("dictionary-actions-cell")).toBeInTheDocument()
    expect(actionsCellSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ useCompactDictionaryActions: false })
    )

    rerender(<Harness useCompactDictionaryActions />)

    expect(actionsCellSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ useCompactDictionaryActions: true })
    )
  })
})
