import React from "react"
import { beforeAll, describe, expect, it, vi } from "vitest"
import { createInstance } from "i18next"
import { initReactI18next } from "react-i18next"
import { fireEvent, render, screen } from "@testing-library/react"
import { PromptStarterCards } from "../PromptStarterCards"

beforeAll(async () => {
  await createInstance().use(initReactI18next).init({
    lng: "en",
    resources: {}
  })
})

describe("PromptStarterCards structured templates", () => {
  it("emits a structured starter prompt with preconfigured blocks", () => {
    const onUse = vi.fn()
    render(<PromptStarterCards onUse={onUse} />)

    fireEvent.click(screen.getByTestId("starter-use-code-review-assistant"))

    expect(onUse).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Code Review Assistant",
        promptFormat: "structured",
        structuredPromptDefinition: expect.objectContaining({
          format: "structured",
          blocks: expect.arrayContaining([
            expect.objectContaining({ role: "system" }),
            expect.objectContaining({ role: "developer" }),
            expect.objectContaining({ role: "user" })
          ]),
          variables: expect.arrayContaining([
            expect.objectContaining({ name: "code", required: true })
          ])
        })
      })
    )
  })

  it("opens the built-in structured recipe as an editable v2 copy", () => {
    const onUse = vi.fn()
    render(<PromptStarterCards onUse={onUse} />)

    fireEvent.click(screen.getByTestId("starter-use-structured-recipe"))

    expect(onUse).toHaveBeenCalledWith(
      expect.objectContaining({
        promptFormat: "structured",
        promptSchemaVersion: 2,
        recipeSource: expect.objectContaining({
          source_kind: "built_in",
          id: "clear_task"
        }),
        structuredPromptDefinition: expect.objectContaining({
          schema_version: 2,
          assembly_config: expect.objectContaining({ target_role: "system" })
        })
      })
    )
  })
})
