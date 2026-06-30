import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PromptEditorDrawer } from "../PromptEditorDrawer"

const mocks = vi.hoisted(() => ({
  createPrompt: vi.fn(),
  updatePrompt: vi.fn(),
  getPrompt: vi.fn(),
  previewPromptDefinition: vi.fn()
}))

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

vi.mock("@/services/prompt-studio", () => ({
  createPrompt: (...args: unknown[]) =>
    (mocks.createPrompt as (...args: unknown[]) => unknown)(...args),
  updatePrompt: (...args: unknown[]) =>
    (mocks.updatePrompt as (...args: unknown[]) => unknown)(...args),
  getPrompt: (...args: unknown[]) =>
    (mocks.getPrompt as (...args: unknown[]) => unknown)(...args),
  previewPromptDefinition: (...args: unknown[]) =>
    (mocks.previewPromptDefinition as (...args: unknown[]) => unknown)(...args)
}))

const makeStructuredPrompt = () => ({
  id: 71,
  project_id: 9,
  signature_id: 17,
  name: "Structured summarizer",
  system_prompt: "You are precise.",
  user_prompt: "Summarize {{topic}}",
  prompt_format: "structured",
  prompt_schema_version: 1,
  prompt_definition: {
    schema_version: 1,
    format: "structured",
    variables: [
      {
        name: "topic",
        required: true,
        input_type: "text"
      }
    ],
    blocks: [
      {
        id: "identity",
        name: "Identity",
        role: "system",
        content: "You are precise.",
        enabled: true,
        order: 10,
        is_template: false
      },
      {
        id: "task",
        name: "Task",
        role: "user",
        content: "Summarize {{topic}}",
        enabled: true,
        order: 20,
        is_template: true
      }
    ]
  },
  version_number: 3,
  updated_at: "2026-03-10T00:00:00Z"
})

const renderDrawer = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      },
      mutations: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <PromptEditorDrawer
        open
        promptId={71}
        projectId={9}
        onClose={vi.fn()}
      />
    </QueryClientProvider>
  )
}

describe("PromptEditorDrawer structured prompt mode", () => {
  beforeEach(() => {
    mocks.createPrompt.mockReset()
    mocks.updatePrompt.mockReset()
    mocks.getPrompt.mockReset()
    mocks.previewPromptDefinition.mockReset()

    mocks.getPrompt.mockResolvedValue({
      data: {
        data: makeStructuredPrompt()
      }
    })
    mocks.updatePrompt.mockResolvedValue({
      data: {
        data: makeStructuredPrompt()
      }
    })
    mocks.previewPromptDefinition.mockResolvedValue({
      data: {
        data: {
          prompt_format: "structured",
          prompt_schema_version: 1,
          assembled_messages: [
            {
              role: "system",
              content: "You are precise."
            },
            {
              role: "user",
              content: "Summarize SQLite FTS"
            }
          ],
          legacy_system_prompt: "You are precise.",
          legacy_user_prompt: "Summarize SQLite FTS"
        }
      }
    })
  })

  it("loads structured prompts and submits prompt_definition updates", async () => {
    renderDrawer()

    await screen.findByTestId("structured-block-list")
    fireEvent.click(screen.getByTestId("structured-block-item-task"))
    fireEvent.change(screen.getByTestId("structured-block-content"), {
      target: { value: "Summarize {{topic}} clearly" }
    })
    fireEvent.change(screen.getByLabelText("Change Description"), {
      target: { value: "Refined task wording" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(mocks.updatePrompt).toHaveBeenCalledWith(
        71,
        expect.objectContaining({
          prompt_format: "structured",
          prompt_schema_version: 1,
          change_description: "Refined task wording",
          prompt_definition: expect.objectContaining({
            blocks: expect.arrayContaining([
              expect.objectContaining({
                id: "task",
                content: "Summarize {{topic}} clearly"
              })
            ])
          })
        })
      )
    })
  })

  it("requests and renders structured preview output", async () => {
    renderDrawer()

    await screen.findByTestId("structured-preview-button")
    fireEvent.click(screen.getByTestId("structured-preview-button"))

    await waitFor(() => {
      expect(mocks.previewPromptDefinition).toHaveBeenCalledWith(
        expect.objectContaining({
          project_id: 9,
          signature_id: 17,
          prompt_format: "structured",
          prompt_schema_version: 1,
          prompt_definition: expect.objectContaining({
            blocks: expect.arrayContaining([
              expect.objectContaining({ id: "identity" }),
              expect.objectContaining({ id: "task" })
            ])
          })
        })
      )
    })

    const previewMatches = await screen.findAllByText("Summarize SQLite FTS")
    expect(previewMatches.length).toBeGreaterThan(0)
  })

  it("renders advanced JSON guidance through the design-system Alert", async () => {
    renderDrawer()

    const advancedLabel = await screen.findByText("Advanced Options")
    fireEvent.click(advancedLabel)

    const guidance = await screen.findByText(
      "These fields accept JSON. Few-shot examples help the model learn from examples."
    )
    const alert = guidance.closest('[data-ds-component="Alert"]')
    expect(alert).not.toBeNull()
  })

  it("projects structured edits into legacy fields and preserves them across mode switches", async () => {
    renderDrawer()

    await screen.findByTestId("structured-block-list")
    fireEvent.click(screen.getByTestId("structured-block-item-task"))
    fireEvent.change(screen.getByTestId("structured-block-content"), {
      target: { value: "Summarize {{topic}} clearly" }
    })

    fireEvent.click(screen.getByRole("radio", { name: "Legacy text" }))

    expect(
      (screen.getByLabelText("System Prompt") as HTMLTextAreaElement).value
    ).toBe("You are precise.")
    expect(
      (screen.getByLabelText("User Prompt Template") as HTMLTextAreaElement).value
    ).toBe("Summarize {{topic}} clearly")

    fireEvent.click(screen.getByRole("radio", { name: "Structured builder" }))

    await screen.findByTestId("structured-block-content")
    fireEvent.click(screen.getByTestId("structured-block-item-task"))
    expect(
      (screen.getByTestId("structured-block-content") as HTMLTextAreaElement).value
    ).toBe("Summarize {{topic}} clearly")
  })

  it("extracts template variables when switching a legacy draft to structured mode", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    mocks.createPrompt.mockResolvedValue({
      data: {
        data: {
          ...makeStructuredPrompt(),
          id: 72,
          name: "Converted prompt"
        }
      }
    })

    render(
      <QueryClientProvider client={queryClient}>
        <PromptEditorDrawer
          open
          promptId={null}
          projectId={9}
          onClose={vi.fn()}
        />
      </QueryClientProvider>
    )

    fireEvent.change(screen.getByLabelText("Prompt Name"), {
      target: { value: "Converted prompt" }
    })
    fireEvent.change(screen.getByLabelText("System Prompt"), {
      target: { value: "You are precise about {{topic}}." }
    })
    fireEvent.change(screen.getByLabelText("User Prompt Template"), {
      target: { value: "Summarize {{topic}} against {{baseline}}" }
    })

    fireEvent.click(screen.getByRole("radio", { name: "Structured builder" }))
    await screen.findByTestId("structured-block-list")

    fireEvent.click(screen.getByRole("button", { name: "Create Prompt" }))

    await waitFor(() => {
      expect(mocks.createPrompt).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt_format: "structured",
          prompt_definition: expect.objectContaining({
            variables: expect.arrayContaining([
              expect.objectContaining({ name: "topic", required: true }),
              expect.objectContaining({ name: "baseline", required: true })
            ])
          })
        }),
        expect.any(String)
      )
    })
  })
})
