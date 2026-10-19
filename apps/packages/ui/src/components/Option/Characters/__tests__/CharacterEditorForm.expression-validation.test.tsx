import React from "react"
import { Form } from "antd"
import type { FormInstance, InputRef } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CharacterEditorForm, type CharacterEditorFormProps } from "../CharacterEditorForm"

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
  mode?: CharacterEditorFormProps["mode"]
  showPreview?: boolean
  initialValues?: CharacterEditorFormProps["initialValues"]
  setShowAdvanced?: React.Dispatch<React.SetStateAction<boolean>>
  setAdvancedSections?: React.Dispatch<React.SetStateAction<any>>
  worldBookFieldContext?: CharacterEditorFormProps["worldBookFieldContext"]
  onFinish?: CharacterEditorFormProps["onFinish"]
  showMetadata?: boolean
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
        mode={overrides.mode ?? "create"}
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
          ],
          ...overrides.initialValues
        }}
        worldBookFieldContext={overrides.worldBookFieldContext ?? {
          options: [],
          loading: false,
          editCharacterNumericId: null
        }}
        isSubmitting={false}
        submitPendingLabel="Saving"
        submitIdleLabel="Save"
        showPreview={overrides.showPreview ?? false}
        onTogglePreview={vi.fn()}
        onValuesChange={vi.fn()}
        onFinish={overrides.onFinish ?? vi.fn()}
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
        showAdvanced={overrides.showMetadata ?? false}
        setShowAdvanced={setShowAdvanced}
        advancedSections={{
          promptControl: false,
          generationSettings: false,
          metadata: overrides.showMetadata ?? false
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
    const { form } = renderEditorForm({ setAdvancedSections, setShowAdvanced })
    await waitFor(() => expect(form()).not.toBeNull())
    const scrollToField = vi
      .spyOn(form()!, "scrollToField")
      .mockImplementation(() => undefined)

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(setShowAdvanced).toHaveBeenCalledWith(true)
    })
    expect(scrollToField).toHaveBeenCalledWith(
      ["expression_images", 0, "state"],
      { block: "center" }
    )
    const updateSections = setAdvancedSections.mock.calls.at(-1)?.[0]
    expect(updateSections).toEqual(expect.any(Function))
    expect(
      updateSections({
        promptControl: false,
        generationSettings: false,
        metadata: false
      })
    ).toMatchObject({ metadata: true })
  })
})

describe("CharacterEditorForm tag entry", () => {
  it.each(["create", "edit"] as const)(
    "%s keeps tag search separate from the saved tag array and preview",
    async (mode) => {
      const { form } = renderEditorForm({
        mode,
        showPreview: true,
        initialValues: { tags: ["saved-tag"], expression_images: [] }
      })
      const input = document.querySelector<HTMLInputElement>('.ant-select-multiple input[role="combobox"]')!

      fireEvent.change(input, { target: { value: "fresh-tag" } })

      expect(form()!.getFieldValue("tags")).toEqual(["saved-tag"])
      fireEvent.keyDown(input, { key: "Enter", code: "Enter" })
      await waitFor(() =>
        expect(form()!.getFieldValue("tags")).toEqual(["saved-tag", "fresh-tag"])
      )
      const values = await form()!.validateFields()
      expect(values).toMatchObject({
        name: "Ada",
        system_prompt: "Stay in character.",
        greeting: "Hello.",
        tags: ["saved-tag", "fresh-tag"]
      })
    }
  )
})


describe("CharacterEditorForm world-book availability", () => {
  it.each(["create", "edit"] as const)(
    "%s preserves main fields without submitting unknown attachments after catalog failure",
    async (mode) => {
      const onFinish = vi.fn()
      const retry = vi.fn()
      renderEditorForm({
        mode,
        showMetadata: true,
        initialValues: { expression_images: [], world_book_ids: [11] },
        worldBookFieldContext: {
          options: [{ id: 11, name: "Lore Atlas" }],
          loading: false,
          editCharacterNumericId: null,
          error: true,
          retry
        },
        onFinish
      })

      expect(screen.getByText("Failed to load world books")).toBeInTheDocument()
      expect(document.querySelector("#world_book_ids")).toBeDisabled()
      fireEvent.click(screen.getByRole("button", { name: "Retry world books" }))
      expect(retry).toHaveBeenCalledTimes(1)
      fireEvent.click(screen.getByRole("button", { name: "Save" }))
      await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1))
      expect(onFinish.mock.calls[0][0]).toMatchObject({
        name: "Ada",
        system_prompt: "Stay in character.",
        greeting: "Hello."
      })
      expect(onFinish.mock.calls[0][0]).not.toHaveProperty("world_book_ids")
    }
  )

  it("preserves an intentional empty attachment selection after successful loading", async () => {
    const onFinish = vi.fn()
    renderEditorForm({
      showMetadata: true,
      initialValues: { expression_images: [], world_book_ids: [] },
      worldBookFieldContext: {
        options: [{ id: 11, name: "Lore Atlas" }],
        loading: false,
        editCharacterNumericId: null
      },
      onFinish
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1))
    expect(onFinish.mock.calls[0][0]).toHaveProperty("world_book_ids", [])
  })

  it("does not submit an uninitialized attachment selection while loading", async () => {
    const onFinish = vi.fn()
    renderEditorForm({
      showMetadata: true,
      initialValues: { expression_images: [], world_book_ids: [] },
      worldBookFieldContext: {
        options: [],
        loading: true,
        editCharacterNumericId: null
      },
      onFinish
    })
    expect(document.querySelector("#world_book_ids")).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1))
    expect(onFinish.mock.calls[0][0]).not.toHaveProperty("world_book_ids")
  })
})
