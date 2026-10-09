import React from "react"
import { StyleProvider } from "@ant-design/cssinjs"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesEditorHeader from "../NotesEditorHeader"

const responsiveState = vi.hoisted(() => ({
  isMobile: false
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => responsiveState.isMobile
}))

type HeaderOverrideProps = Partial<React.ComponentProps<typeof NotesEditorHeader>>

const renderHeader = (overrides: HeaderOverrideProps = {}) =>
  render(
    <StyleProvider hashPriority="high">
    <NotesEditorHeader
      title="Stage 2 note"
      selectedId="note-1"
      backlinkConversationId={null}
      backlinkConversationLabel={null}
      backlinkMessageId={null}
      sourceLinks={[]}
      editorDisabled={false}
      openingLinkedChat={false}
      editorMode="edit"
      hasContent
      canSave
      canGenerateFlashcards
      canExport
      isSaving={false}
      canDelete
      isDirty={false}
      onOpenLinkedConversation={() => undefined}
      onOpenSourceLink={() => undefined}
      onChangeEditorMode={() => undefined}
      onCopy={() => undefined}
      onGenerateFlashcards={() => undefined}
      onExport={() => undefined}
      onSave={() => undefined}
      onDelete={() => undefined}
      {...overrides}
    />
    </StyleProvider>
  )

const finishMenuAnimation = async (popup: Element) => {
  await waitFor(() => expect(popup.className).toMatch(/(?:appear|enter|leave)-active/))
  // jsdom does not run CSS animations; deliver the normal completion events.
  fireEvent(popup, new Event("webkitAnimationEnd", { bubbles: true }))
  fireEvent.animationEnd(popup)
}

describe("NotesEditorHeader stage 2 touch layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it.each([false, true])("keeps click-open Export available after the pointer leaves (mobile=%s)", async (isMobile) => {
    responsiveState.isMobile = isMobile
    const onExport = vi.fn()
    renderHeader({ onExport })
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    const exportMenu = (await screen.findByText("Export", { selector: ".ant-dropdown-menu-title-content" })).closest("[role=menuitem]")!
    await finishMenuAnimation(exportMenu.closest(".ant-dropdown")!)
    expect(exportMenu).toBeVisible()
    fireEvent.click(exportMenu)
    const printLabel = await screen.findByText("Print / Save as PDF")
    await finishMenuAnimation(printLabel.closest(".ant-dropdown-menu-submenu-popup")!)
    const print = await screen.findByRole("menuitem", { name: "Print / Save as PDF" })

    vi.useFakeTimers()
    fireEvent.mouseLeave(exportMenu)
    fireEvent.mouseLeave(print.closest("ul")!)
    await act(async () => { await vi.advanceTimersByTimeAsync(200) })
    expect(exportMenu).toHaveAttribute("aria-expanded", "true")
    expect(print).toBeVisible()
    vi.useRealTimers()

    fireEvent.click(print)
    expect(onExport).toHaveBeenCalledExactlyOnceWith("print")
    await finishMenuAnimation(exportMenu.closest(".ant-dropdown")!)
    const printPopup = print.closest(".ant-dropdown-menu-submenu-popup")!
    await waitFor(() => expect(printPopup.className).toContain("leave-active"))
    await finishMenuAnimation(printPopup)
    await waitFor(() => expect(print).not.toBeVisible())
  })

  it("does not open or dispatch a disabled Export submenu", async () => {
    responsiveState.isMobile = false
    const onExport = vi.fn()
    renderHeader({ canExport: false, onExport })
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    const exportMenu = (await screen.findByText("Export", { selector: ".ant-dropdown-menu-title-content" })).closest("[role=menuitem]")!
    await finishMenuAnimation(exportMenu.closest(".ant-dropdown")!)
    expect(exportMenu).toBeVisible()
    expect(exportMenu).toHaveAttribute("aria-disabled", "true")
    fireEvent.click(exportMenu)
    expect(screen.queryByRole("menuitem", { name: "Print / Save as PDF" })).not.toBeInTheDocument()
    expect(onExport).not.toHaveBeenCalled()
  })

  it.each(["Copy", "From Template"])("keeps %s submenu dispatch after click-open", async (label) => {
    responsiveState.isMobile = false
    const onCopy = vi.fn()
    const onApplyTemplate = vi.fn()
    renderHeader({ onCopy, onApplyTemplate, templateOptions: [{ id: "template-1", label: "Template one" }] })
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    const title = await screen.findByText(label, { selector: ".ant-dropdown-menu-title-content" })
    const submenu = title.closest("[role=menuitem]")!
    await finishMenuAnimation(submenu.closest(".ant-dropdown")!)
    expect(submenu).toBeVisible()
    fireEvent.click(submenu)
    const itemLabel = await screen.findByText(label === "Copy" ? "Content only" : "Template one")
    await finishMenuAnimation(itemLabel.closest(".ant-dropdown-menu-submenu-popup")!)
    const item = screen.getByRole("menuitem", { name: label === "Copy" ? "Content only" : "Template one" })
    expect(item).toBeVisible()
    fireEvent.click(item)
    if (label === "Copy") {
      expect(onCopy).toHaveBeenCalledExactlyOnceWith("content")
      expect(onApplyTemplate).not.toHaveBeenCalled()
    } else {
      expect(onApplyTemplate).toHaveBeenCalledExactlyOnceWith("template-1")
      expect(onCopy).not.toHaveBeenCalled()
    }
  })

  it("uses wrapped toolbar layout and 44px touch targets on mobile", () => {
    responsiveState.isMobile = true
    renderHeader()

    const actions = screen.getByTestId("notes-header-actions")
    expect(actions.className).toContain("w-full")
    expect(actions.className).toContain("flex-wrap")

    const saveButton = screen.getByTestId("notes-save-button")
    const overflowButton = screen.getByTestId("notes-overflow-menu-button")

    expect(saveButton.className).toContain("ant-btn-lg")
    expect(saveButton.className).toContain("min-h-[44px]")
    expect(overflowButton.className).toContain("min-h-[44px]")
    expect(overflowButton.className).toContain("min-w-[44px]")
  })

  it("keeps compact desktop controls while allowing toolbar wrapping", () => {
    responsiveState.isMobile = false
    renderHeader()

    const actions = screen.getByTestId("notes-header-actions")
    expect(actions.className).not.toContain("w-full")
    expect(actions.className).toContain("flex-wrap")

    const saveButton = screen.getByTestId("notes-save-button")
    const overflowButton = screen.getByTestId("notes-overflow-menu-button")

    expect(saveButton.className).toContain("ant-btn-sm")
    expect(saveButton.className).not.toContain("min-h-[44px]")
    expect(overflowButton.className).not.toContain("min-h-[44px]")
  })

  it("shows editor mode toggle on desktop but not on mobile", () => {
    responsiveState.isMobile = false
    renderHeader()

    // On desktop the editor mode toggle group is visible
    expect(screen.getByRole("group", { name: "Editor mode" })).toBeInTheDocument()
  })

  it("hides editor mode toggle on mobile", () => {
    responsiveState.isMobile = true
    renderHeader()

    // On mobile the editor mode toggle group is hidden
    expect(screen.queryByRole("group", { name: "Editor mode" })).not.toBeInTheDocument()
  })

  it("disables desktop Save & new while a save is in progress", () => {
    responsiveState.isMobile = false
    renderHeader({
      isSaving: true,
      onSaveAndNew: () => undefined
    })

    expect(screen.getByTestId("notes-save-and-new-button")).toBeDisabled()
  })

  it("disables mobile Save & new overflow item while a save is in progress", async () => {
    responsiveState.isMobile = true
    renderHeader({
      isSaving: true,
      onSaveAndNew: () => undefined
    })

    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))

    const menuItem = (await screen.findByText("Save & new")).closest(".ant-dropdown-menu-item")
    expect(menuItem).toHaveClass("ant-dropdown-menu-item-disabled")
  })

  it("keeps create study pack in the desktop toolbar and mobile overflow menu", async () => {
    const onCreateStudyPack = vi.fn()

    responsiveState.isMobile = false
    const { unmount } = renderHeader({
      canCreateStudyPack: true,
      onCreateStudyPack
    })

    expect(screen.getByTestId("notes-create-study-pack-button")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    expect(screen.getAllByText("Create study pack")).toHaveLength(1)

    unmount()
    responsiveState.isMobile = true
    renderHeader({
      canCreateStudyPack: true,
      onCreateStudyPack
    })

    expect(screen.queryByTestId("notes-create-study-pack-button")).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    expect(await screen.findByText("Create study pack")).toBeInTheDocument()
  })
})
