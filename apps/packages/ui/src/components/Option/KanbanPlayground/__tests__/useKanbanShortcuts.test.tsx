import { renderHook } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it, vi } from "vitest"
import { useKanbanShortcuts } from "../useKanbanShortcuts"

it("runs the card and list shortcuts outside editable fields", async () => {
  const user = userEvent.setup()
  const onNewCard = vi.fn()
  const onNewList = vi.fn()
  renderHook(() => useKanbanShortcuts({ onNewCard, onNewList }))

  await user.keyboard("n")
  expect(onNewCard).toHaveBeenCalledOnce()
  expect(onNewList).not.toHaveBeenCalled()

  await user.keyboard("l")
  expect(onNewCard).toHaveBeenCalledOnce()
  expect(onNewList).toHaveBeenCalledOnce()
})
