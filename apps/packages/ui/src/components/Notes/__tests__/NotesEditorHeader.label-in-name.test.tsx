import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesEditorHeader from "../NotesEditorHeader"

// AX-17 (#3123): a control's accessible name must contain its visible label
// (WCAG 2.5.3), so speech-input users can say what they see.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

describe("NotesEditorHeader label in name", () => {
  it("names the Save & new button by its visible label", () => {
    render(
      <NotesEditorHeader
        title="Label note"
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
        isDirty
        onOpenLinkedConversation={() => undefined}
        onOpenSourceLink={() => undefined}
        onChangeEditorMode={() => undefined}
        onCopy={() => undefined}
        onGenerateFlashcards={() => undefined}
        onExport={() => undefined}
        onSave={() => undefined}
        onSaveAndNew={() => undefined}
        onDelete={() => undefined}
      />
    )

    const button = screen.getByTestId("notes-save-and-new-button")
    expect(button).toHaveAccessibleName(expect.stringContaining("Save & new"))
  })
})
