import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { PromptFullPageEditor } from "../PromptFullPageEditor"

const mockDraftState = {
  hasDraft: false,
  draftData: null,
  saveDraft: vi.fn(),
  clearDraft: vi.fn(),
  applyDraft: vi.fn(),
  dismissDraft: vi.fn(),
  lastSaved: null
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if (fallbackOrOptions.defaultValue) {
          return Object.entries(fallbackOrOptions).reduce(
            (acc, [name, value]) =>
              name === "defaultValue"
                ? acc
                : acc.replace(new RegExp(`{{${name}}}`, "g"), String(value)),
            fallbackOrOptions.defaultValue
          )
        }
        return key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useFormDraft", () => ({
  useFormDraft: () => mockDraftState,
  formatDraftAge: () => "now"
}))

vi.mock("../PromptEditorPreview", () => ({
  PromptEditorPreview: () => <div data-testid="prompt-editor-preview" />
}))

vi.mock("@/services/prompts-api", () => ({
  previewStructuredPromptServer: vi.fn(async () => ({
    prompt_format: "structured",
    prompt_schema_version: 1,
    assembled_messages: [],
    legacy_system_prompt: "You are a careful analyst.",
    legacy_user_prompt: "Summarize {{topic}}"
  }))
}))

describe("PromptFullPageEditor structured prompts", () => {
  const baseProps = {
    open: true,
    onClose: vi.fn(),
    mode: "edit" as const,
    initialValues: {
      id: "prompt-1",
      name: "Legacy prompt",
      system_prompt: "You are a careful analyst.",
      user_prompt: "Summarize {{topic}}",
      keywords: ["analysis"]
    },
    onSubmit: vi.fn(),
    isLoading: false,
    allTags: []
  }

  it("converts a legacy full-page prompt into a structured prompt and locks raw fields", async () => {
    render(<PromptFullPageEditor {...baseProps} />)

    fireEvent.click(
      screen.getByRole("button", { name: /convert to structured/i })
    )

    const structuredPromptTitle = await screen.findByText("Structured prompt")
    expect(structuredPromptTitle).toBeInTheDocument()
    const alert = structuredPromptTitle.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
    expect(screen.getByTestId("full-editor-system-prompt")).toBeDisabled()
    expect(screen.getByTestId("full-editor-user-prompt")).toBeDisabled()
    expect(screen.getByTestId("structured-block-list")).toBeInTheDocument()
  })

  it("submits structured prompt state with the derived legacy snapshot", async () => {
    const onSubmit = vi.fn()
    render(<PromptFullPageEditor {...baseProps} onSubmit={onSubmit} />)

    fireEvent.click(
      screen.getByRole("button", { name: /convert to structured/i })
    )
    fireEvent.click(screen.getByTestId("structured-block-item-legacy_user"))
    fireEvent.change(screen.getByTestId("structured-block-content"), {
      target: { value: "Summarize {{topic}} clearly" }
    })

    fireEvent.click(screen.getByTestId("full-editor-save"))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          promptFormat: "structured",
          promptSchemaVersion: 1,
          structuredPromptDefinition: expect.objectContaining({
            blocks: expect.arrayContaining([
              expect.objectContaining({
                id: "legacy_user",
                content: "Summarize {{topic}} clearly"
              })
            ])
          }),
          system_prompt: "You are a careful analyst.",
          user_prompt: "Summarize {{topic}} clearly"
        })
      )
    })
  })
})
