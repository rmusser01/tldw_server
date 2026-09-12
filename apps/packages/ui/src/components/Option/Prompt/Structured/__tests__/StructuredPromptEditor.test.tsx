import React from "react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import { createInstance } from "i18next"
import { initReactI18next } from "react-i18next"
import { fireEvent, render, screen, within } from "@testing-library/react"

import { StructuredPromptEditor } from "../StructuredPromptEditor"

beforeAll(async () => {
  await createInstance()
    .use(initReactI18next)
    .init({
      lng: "en",
      resources: {},
      interpolation: { escapeValue: false }
    })
})

describe("StructuredPromptEditor", () => {
  it("preserves assembly_config and block metadata when editing blocks", () => {
    const onChange = vi.fn()

    render(
      <StructuredPromptEditor
        value={{
          schema_version: 1,
          format: "structured",
          assembly_config: {
            legacy_system_roles: ["developer"],
            legacy_user_roles: ["assistant"],
            block_separator: "\n--\n"
          },
          variables: [
            {
              name: "topic",
              required: true,
              input_type: "textarea"
            }
          ],
          blocks: [
            {
              id: "guidance",
              name: "Guidance",
              role: "developer",
              kind: "instructions",
              content: "Rule one",
              enabled: true,
              order: 10,
              is_template: false
            }
          ]
        }}
        onChange={onChange}
        previewResult={null}
        previewLoading={false}
        onPreview={vi.fn()}
      />
    )

    fireEvent.change(screen.getByTestId("structured-block-content"), {
      target: { value: "Rule one refined" }
    })

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        assembly_config: {
          legacy_system_roles: ["developer"],
          legacy_user_roles: ["assistant"],
          block_separator: "\n--\n"
        },
        blocks: [
          expect.objectContaining({
            id: "guidance",
            kind: "instructions",
            content: "Rule one refined"
          })
        ]
      })
    )
  })

  it("preserves variable defaults and constraints when editing variables", () => {
    const onChange = vi.fn()

    render(
      <StructuredPromptEditor
        value={{
          schema_version: 1,
          format: "structured",
          variables: [
            {
              name: "topic",
              label: "Topic",
              description: "Prompt topic",
              required: false,
              input_type: "select",
              default_value: "sqlite",
              options: ["sqlite", "fts"],
              max_length: 12
            }
          ],
          blocks: [
            {
              id: "task",
              name: "Task",
              role: "user",
              kind: "task",
              content: "Summarize {{topic}}",
              enabled: true,
              order: 10,
              is_template: true
            }
          ]
        }}
        onChange={onChange}
        previewResult={null}
        previewLoading={false}
        onPreview={vi.fn()}
      />
    )

    fireEvent.change(screen.getByTestId("structured-variable-name-0"), {
      target: { value: "topic_name" }
    })

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        variables: [
          expect.objectContaining({
            name: "topic_name",
            label: "Topic",
            description: "Prompt topic",
            input_type: "select",
            default_value: "sqlite",
            options: ["sqlite", "fts"],
            max_length: 12
          })
        ]
      })
    )
  })

  it("keeps the schema-v1 role editor and accessible reorder controls unchanged", () => {
    render(
      <StructuredPromptEditor
        value={{
          schema_version: 1,
          format: "structured",
          variables: [],
          blocks: [
            {
              id: "system",
              name: "System",
              role: "system",
              content: "System content",
              enabled: true,
              order: 10,
              is_template: false
            },
            {
              id: "user",
              name: "User",
              role: "user",
              content: "User content",
              enabled: true,
              order: 20,
              is_template: false
            }
          ]
        }}
        onChange={vi.fn()}
        previewResult={null}
        previewLoading={false}
        onPreview={vi.fn()}
      />
    )

    const role = screen.getByTestId("structured-block-role")
    expect(within(role).getAllByRole("option")).toHaveLength(4)
    expect(
      screen.getByRole("button", { name: "Move System up" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Move System down" })
    ).toBeEnabled()
    expect(
      screen.queryByRole("textbox", { name: /Starter default/ })
    ).toBeNull()
  })
})
