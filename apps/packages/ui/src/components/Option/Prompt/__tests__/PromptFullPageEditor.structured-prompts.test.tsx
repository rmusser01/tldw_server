import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
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
    fireEvent.click(
      screen.getByRole("button", { name: "Edit User Prompt block" })
    )
    await waitFor(() => {
      expect(screen.getByTestId("structured-block-content")).toHaveValue(
        "Summarize {{topic}}"
      )
    })
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

  it("routes an exact saved v2 recipe to the recipe editor and updates its source id", async () => {
    const onUpdateRecipe = vi.fn(async () => undefined)
    render(
      <PromptFullPageEditor
        {...baseProps}
        initialValues={{
          id: "saved-recipe-42",
          name: "Saved clear task",
          promptFormat: "structured",
          promptSchemaVersion: 2,
          structuredPromptDefinition: structuredClone(
            CLEAR_TASK_RECIPE.definition
          )
        }}
        recipePersistenceAvailable
        onUpdateRecipe={onUpdateRecipe}
      />
    )

    expect(screen.getByTestId("single-field-recipe-editor")).toBeInTheDocument()
    expect(screen.queryByTestId("full-editor-system-prompt")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Update recipe" }))

    await waitFor(() => {
      expect(onUpdateRecipe).toHaveBeenCalledWith(
        "saved-recipe-42",
        expect.objectContaining({
          schema_version: 2,
          definition_kind: "single_text_recipe"
        })
      )
    })
  })

  it("keeps local recipe editing and apply available when persistence is disabled", () => {
    const onApplyRecipe = vi.fn()
    render(
      <PromptFullPageEditor
        {...baseProps}
        initialValues={{
          id: "saved-recipe-42",
          name: "Saved clear task",
          promptFormat: "structured",
          promptSchemaVersion: 2,
          structuredPromptDefinition: structuredClone(
            CLEAR_TASK_RECIPE.definition
          )
        }}
        recipePersistenceAvailable={false}
        recipePersistenceUnavailableReason="Reconnect to save recipes."
        onApplyRecipe={onApplyRecipe}
      />
    )

    expect(screen.getByText("Reconnect to save recipes.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Update recipe" })).toBeDisabled()
    fireEvent.change(
      screen.getByRole("textbox", {
        name: "Current value for Task (not saved)"
      }),
      { target: { value: "Summarize the release" } }
    )
    const apply = screen.getByRole("button", {
      name: "Apply to system prompt"
    })
    expect(apply).toBeEnabled()
    fireEvent.click(apply)
    expect(onApplyRecipe).toHaveBeenCalledTimes(1)
  })

  it("quarantines an invalid v2 record instead of opening either editor", () => {
    render(
      <PromptFullPageEditor
        {...baseProps}
        initialValues={{
          id: "broken-recipe",
          name: "Broken recipe",
          promptFormat: "structured",
          promptSchemaVersion: 2,
          structuredPromptDefinition: {
            ...structuredClone(CLEAR_TASK_RECIPE.definition),
            definition_kind: "future_recipe"
          }
        }}
      />
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      "This saved recipe cannot be opened safely."
    )
    expect(screen.queryByTestId("single-field-recipe-editor")).not.toBeInTheDocument()
    expect(screen.queryByTestId("full-editor-system-prompt")).not.toBeInTheDocument()
  })
})
