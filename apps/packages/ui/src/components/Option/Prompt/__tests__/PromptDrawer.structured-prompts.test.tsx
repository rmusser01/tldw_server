import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { PromptDrawer } from "../PromptDrawer"

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
  useFormDraft: () => ({
    hasDraft: false,
    draftData: null,
    saveDraft: vi.fn(),
    clearDraft: vi.fn(),
    applyDraft: vi.fn(),
    dismissDraft: vi.fn()
  }),
  formatDraftAge: () => "now"
}))

vi.mock("../Studio/Prompts/VersionHistoryDrawer", () => ({
  VersionHistoryDrawer: () => null
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

describe("PromptDrawer structured prompts", () => {
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

  it("converts a legacy prompt into a structured prompt and locks raw fields", async () => {
    render(<PromptDrawer {...baseProps} />)

    fireEvent.click(
      screen.getByRole("button", { name: /convert to structured/i })
    )

    expect(await screen.findByText("Structured prompt")).toBeInTheDocument()
    expect(screen.getByTestId("prompt-drawer-system")).toBeDisabled()
    expect(screen.getByTestId("prompt-drawer-user")).toBeDisabled()
    expect(screen.getByTestId("structured-block-list")).toBeInTheDocument()
  })

  it("extracts template variables when converting a legacy prompt", async () => {
    const onSubmit = vi.fn()
    render(<PromptDrawer {...baseProps} onSubmit={onSubmit} />)

    fireEvent.click(
      screen.getByRole("button", { name: /convert to structured/i })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "managePrompts.form.btnSave.save" })
    )

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          structuredPromptDefinition: expect.objectContaining({
            variables: expect.arrayContaining([
              expect.objectContaining({
                name: "topic",
                required: true
              })
            ])
          })
        })
      )
    })
  })

  it("submits structured prompt state with the derived legacy snapshot", async () => {
    const onSubmit = vi.fn()
    render(<PromptDrawer {...baseProps} onSubmit={onSubmit} />)

    fireEvent.click(
      screen.getByRole("button", { name: /convert to structured/i })
    )
    fireEvent.click(screen.getByTestId("structured-block-item-legacy_user"))
    fireEvent.change(screen.getByTestId("structured-block-content"), {
      target: { value: "Summarize {{topic}} clearly" }
    })

    fireEvent.click(
      screen.getByRole("button", { name: "managePrompts.form.btnSave.save" })
    )

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
