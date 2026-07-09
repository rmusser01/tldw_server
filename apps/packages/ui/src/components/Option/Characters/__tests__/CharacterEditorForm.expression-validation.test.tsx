import React from "react"
import { Form } from "antd"
import type { FormInstance, InputRef } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CharacterEditorForm } from "../CharacterEditorForm"

const t = ((key: string, fallback?: string | { defaultValue?: string }) => {
  if (typeof fallback === "string") return fallback
  return fallback?.defaultValue ?? key
}) as any

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }) => {
      if (typeof fallback === "string") return fallback
      return fallback?.defaultValue ?? key
    }
  })
}))

const renderEditorForm = (overrides: {
  setShowAdvanced?: React.Dispatch<React.SetStateAction<boolean>>
  setAdvancedSections?: React.Dispatch<React.SetStateAction<any>>
} = {}) => {
  let formRef: FormInstance | null = null
  const setShowAdvanced = overrides.setShowAdvanced ?? vi.fn()
  const setAdvancedSections = overrides.setAdvancedSections ?? vi.fn()

  const Harness = () => {
    const [form] = Form.useForm()
    formRef = form
    return (
      <CharacterEditorForm
        t={t}
        form={form}
        mode="create"
        initialValues={{
          name: "Ada",
          system_prompt: "Stay in character.",
          greeting: "Hello.",
          expression_images: [
            {
              id: "smirk",
              state: "smirk",
              starter: false,
              image: { mode: "url", url: "", base64: "" }
            }
          ]
        }}
        worldBookFieldContext={{
          options: [],
          loading: false,
          editCharacterNumericId: null
        }}
        isSubmitting={false}
        submitPendingLabel="Saving"
        submitIdleLabel="Save"
        showPreview={false}
        onTogglePreview={vi.fn()}
        onValuesChange={vi.fn()}
        onFinish={vi.fn()}
        generatingField={null}
        isGenerating={false}
        handleGenerateField={vi.fn()}
        showSystemPromptExample={false}
        setShowSystemPromptExample={vi.fn()}
        markModeDirty={vi.fn()}
        popularTags={[]}
        tagOptionsWithCounts={[]}
        characterFolderOptions={[]}
        characterFolderOptionsLoading={false}
        showAdvanced={false}
        setShowAdvanced={setShowAdvanced}
        advancedSections={{
          promptControl: false,
          generationSettings: false,
          metadata: false
        }}
        setAdvancedSections={setAdvancedSections}
        createNameRef={React.createRef<InputRef>()}
        editNameRef={React.createRef<InputRef>()}
      />
    )
  }

  render(<Harness />)
  return { form: () => formRef, setAdvancedSections, setShowAdvanced }
}

describe("CharacterEditorForm expression image validation", () => {
  it("keeps expression image validation mounted while advanced fields are collapsed", async () => {
    const { form } = renderEditorForm()

    await waitFor(() => expect(form()).not.toBeNull())

    await expect(form()!.validateFields()).rejects.toMatchObject({
      errorFields: expect.arrayContaining([
        expect.objectContaining({
          name: ["_expression_images_validation"]
        })
      ])
    })
  })

  it("opens metadata fields when collapsed expression images block submit", async () => {
    const setShowAdvanced = vi.fn()
    const setAdvancedSections = vi.fn()
    renderEditorForm({ setAdvancedSections, setShowAdvanced })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(setShowAdvanced).toHaveBeenCalledWith(true)
    })
    expect(setAdvancedSections).toHaveBeenCalled()
  })
})
