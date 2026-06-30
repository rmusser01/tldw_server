import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
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
  )

describe("NotesEditorHeader stage 2 touch layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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

  it("preserves compact desktop toolbar density", () => {
    responsiveState.isMobile = false
    renderHeader()

    const actions = screen.getByTestId("notes-header-actions")
    expect(actions.className).not.toContain("w-full")
    expect(actions.className).not.toContain("flex-wrap")

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
})
