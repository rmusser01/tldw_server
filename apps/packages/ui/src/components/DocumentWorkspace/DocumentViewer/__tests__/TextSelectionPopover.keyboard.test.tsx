import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it, vi } from "vitest"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { TextSelectionPopover } from "../TextSelectionPopover"

vi.mock("@/hooks/document-workspace/useDocumentTTS", () => ({
  useDocumentTTS: () => ({ state: { isPlaying: false, isLoading: false } })
}))
vi.mock("@/hooks/document-workspace/useTranslate", () => ({
  useTranslate: () => ({ mutateAsync: vi.fn(), isPending: false })
}))

it("opens the actual selection menu from the keyboard and activates its focused item", async () => {
  const user = userEvent.setup()
  useDocumentWorkspaceStore.setState({
    workspaceAuthority: "fixture-owner", activeDocumentId: 81, activeRightTab: "annotations",
    pendingSelectionPrompt: null
  })
  const onClose = vi.fn()
  render(<TextSelectionPopover documentId={81} text="Keyboard selection" position={{ x: 80, y: 40 }} onClose={onClose} />)
  const ask = screen.getByRole("button", { name: "Ask AI" })
  await waitFor(() => expect(ask).toBeVisible())
  ask.focus()
  await user.keyboard("{Enter}")
  const menu = screen.getByRole("menu")
  await waitFor(() => expect(menu).toContainElement(document.activeElement as HTMLElement))
  await user.keyboard("{ArrowDown}")
  await waitFor(() => expect(screen.getByRole("menuitem", { name: "Explain" })).toHaveFocus())
  fireEvent.keyDown(document.activeElement!, { key: "Enter", code: "Enter", which: 13, keyCode: 13 })
  expect(useDocumentWorkspaceStore.getState().pendingSelectionPrompt).toEqual({
    documentId: 81,
    prompt: 'Explain this passage: "Keyboard selection"'
  })
  expect(onClose).toHaveBeenCalledTimes(1)
})
