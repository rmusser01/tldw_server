import React from "react"
import { Form, message, type FormInstance } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  CharacterExpressionImagesSection,
  CharacterExpressionImagesValidationItem
} from "../CharacterExpressionImagesSection"
import { expressionRowsFromExtensions } from "../character-expression-images"
import type { ExpressionImageRow } from "../character-expression-images"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getImageBackends: vi.fn(),
    createImageArtifact: vi.fn()
  }
}))

const ONE_PIXEL_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE/wH+J6m3XQAAAABJRU5ErkJggg=="

const getImageBackendsMock = vi.mocked(tldwClient.getImageBackends)
const createImageArtifactMock = vi.mocked(tldwClient.createImageArtifact)

const renderSection = (
  initialRows: ExpressionImageRow[],
  props: React.ComponentProps<typeof CharacterExpressionImagesSection> = {}
) => {
  let formRef: FormInstance | null = null

  const Harness = () => {
    const [form] = Form.useForm()
    formRef = form
    return (
      <Form form={form} initialValues={{ expression_images: initialRows }}>
        <CharacterExpressionImagesSection characterName="Mira" {...props} />
      </Form>
    )
  }

  const result = render(<Harness />)
  return {
    ...result,
    setRows: (rows: ExpressionImageRow[]) => {
      formRef?.setFieldsValue({ expression_images: rows })
    },
    getRows: () =>
      formRef?.getFieldValue("expression_images") as ExpressionImageRow[]
  }
}

describe("CharacterExpressionImagesSection", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getImageBackendsMock.mockResolvedValue([])
    createImageArtifactMock.mockResolvedValue({
      artifact: { export: { content_b64: ONE_PIXEL_PNG_BASE64 } }
    } as any)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("renders starter expression rows and adds a custom row", async () => {
    render(
      <Form initialValues={{ expression_images: expressionRowsFromExtensions({}) }}>
        <CharacterExpressionImagesSection
          characterName="Mira"
          characterDescription="Archivist"
        />
      </Form>
    )

    expect(screen.getByLabelText("Expression state neutral")).toBeInTheDocument()
    expect(screen.getByLabelText("Expression state thinking")).toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: /add expression/i }))
    expect(screen.getByLabelText(/custom expression state/i)).toBeInTheDocument()
  })

  it("copies the selected preview emote directive", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    const successSpy = vi
      .spyOn(message, "success")
      .mockImplementation(() => ({}) as any)
    Object.assign(navigator, { clipboard: { writeText } })

    render(
      <Form
        initialValues={{
          expression_images: [
            {
              id: "thinking",
              state: "thinking",
              starter: true,
              image: {
                mode: "url",
                url: "https://example.test/thinking.png",
                base64: ""
              }
            }
          ]
        }}
      >
        <CharacterExpressionImagesSection characterName="Mira" />
      </Form>
    )

    await userEvent.click(
      screen.getByRole("button", { name: /copy emote directive/i })
    )
    expect(writeText).toHaveBeenCalledWith("Emote: thinking")
    expect(successSpy).toHaveBeenCalledWith("Copied emote directive.")
  })

  it("shows an error when clipboard copy is unavailable", async () => {
    const errorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => ({}) as any)
    Object.assign(navigator, { clipboard: undefined })

    render(
      <Form
        initialValues={{
          expression_images: [
            {
              id: "thinking",
              state: "thinking",
              starter: true,
              image: {
                mode: "url",
                url: "https://example.test/thinking.png",
                base64: ""
              }
            }
          ]
        }}
      >
        <CharacterExpressionImagesSection characterName="Mira" />
      </Form>
    )

    await userEvent.click(
      screen.getByRole("button", { name: /copy emote directive/i })
    )

    expect(errorSpy).toHaveBeenCalledWith("Unable to copy emote directive.")
  })

  it("shows an error when clipboard copy rejects", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("denied"))
    const errorSpy = vi
      .spyOn(message, "error")
      .mockImplementation(() => ({}) as any)
    Object.assign(navigator, { clipboard: { writeText } })

    render(
      <Form
        initialValues={{
          expression_images: [
            {
              id: "thinking",
              state: "thinking",
              starter: true,
              image: {
                mode: "url",
                url: "https://example.test/thinking.png",
                base64: ""
              }
            }
          ]
        }}
      >
        <CharacterExpressionImagesSection characterName="Mira" />
      </Form>
    )

    await userEvent.click(
      screen.getByRole("button", { name: /copy emote directive/i })
    )

    expect(writeText).toHaveBeenCalledWith("Emote: thinking")
    expect(errorSpy).toHaveBeenCalledWith("Unable to copy emote directive.")
  })

  it("surfaces duplicate state and missing custom image row messages", () => {
    renderSection([
      {
        id: "happy",
        state: "happy",
        starter: true,
        image: {
          mode: "url",
          url: "https://example.test/happy.png",
          base64: ""
        }
      },
      {
        id: "duplicate",
        state: "happy",
        starter: false,
        image: {
          mode: "url",
          url: "https://example.test/other.png",
          base64: ""
        }
      },
      {
        id: "empty-image",
        state: "smirk",
        starter: false,
        image: { mode: "url", url: "", base64: "" }
      }
    ])

    expect(screen.getByText("Expression state is duplicated.")).toBeInTheDocument()
    expect(screen.getByText("Custom expressions need an image.")).toBeInTheDocument()
  })

  it("does not preview non-allowlisted image data URLs in URL mode", () => {
    renderSection([
      {
        id: "smirk",
        state: "smirk",
        starter: false,
        image: {
          mode: "url",
          url: "data:image/svg+xml;base64,PHN2Zy8+",
          base64: ""
        }
      }
    ])

    expect(
      screen.queryByRole("img", { name: "smirk thumbnail" })
    ).not.toBeInTheDocument()
    expect(screen.getAllByText("No image").length).toBeGreaterThan(0)
  })

  it("blocks expression form submit when rows are invalid", async () => {
    const onFinish = vi.fn()
    const onFinishFailed = vi.fn()

    render(
      <Form
        initialValues={{
          expression_images: [
            {
              id: "empty-image",
              state: "smirk",
              starter: false,
              image: { mode: "url", url: "", base64: "" }
            }
          ]
        }}
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
      >
        <CharacterExpressionImagesSection characterName="Mira" />
        <CharacterExpressionImagesValidationItem />
        <button type="submit">Save</button>
      </Form>
    )

    await userEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onFinishFailed).toHaveBeenCalled())
    expect(onFinish).not.toHaveBeenCalled()
  })

  it("blocks expression form submit when invalid extensions JSON must be merged", async () => {
    const onFinish = vi.fn()
    const onFinishFailed = vi.fn()

    render(
      <Form
        initialValues={{
          extensions: "{not valid json",
          expression_images: [
            {
              id: "smirk",
              state: "smirk",
              starter: false,
              image: {
                mode: "url",
                url: "https://example.test/smirk.png",
                base64: ""
              }
            }
          ]
        }}
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
      >
        <Form.Item name="extensions" hidden>
          <input />
        </Form.Item>
        <CharacterExpressionImagesSection characterName="Mira" />
        <CharacterExpressionImagesValidationItem />
        <button type="submit">Save</button>
      </Form>
    )

    await userEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onFinishFailed).toHaveBeenCalled())
    expect(onFinish).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Fix Extensions JSON before saving expression images."
    )
  })

  it("falls back to the base avatar when the selected preview has no image or fails to load", async () => {
    const baseAvatar = {
      mode: "upload" as const,
      url: "",
      base64: ONE_PIXEL_PNG_BASE64
    }

    renderSection(
      [
        {
          id: "neutral",
          state: "neutral",
          starter: true,
          image: { mode: "url", url: "", base64: "" }
        },
        {
          id: "thinking",
          state: "thinking",
          starter: true,
          image: {
            mode: "url",
            url: "https://example.test/broken.png",
            base64: ""
          }
        }
      ],
      { baseAvatar }
    )

    const preview = screen.getByRole("img", {
      name: "Mira expression preview"
    }) as HTMLImageElement
    expect(preview.src).toContain(`data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`)

    await userEvent.selectOptions(
      screen.getByLabelText("Preview expression"),
      "thinking"
    )
    expect(preview.src).toBe("https://example.test/broken.png")

    fireEvent.error(preview)
    await waitFor(() =>
      expect(preview.src).toContain(`data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`)
    )
  })

  it("updates the row image value when URL mode is edited", async () => {
    const { getRows } = renderSection([
      {
        id: "smirk",
        state: "smirk",
        starter: false,
        image: { mode: "url", url: "", base64: "" }
      }
    ])

    await userEvent.type(
      screen.getByLabelText("Expression image URL for smirk"),
      "https://example.test/smirk.png"
    )

    expect(getRows()[0].image).toEqual({
      mode: "url",
      url: "https://example.test/smirk.png",
      base64: ""
    })
  })

  it("does not refill the generation prompt after the user clears it", async () => {
    getImageBackendsMock.mockResolvedValue([
      { id: "mock-image", name: "Mock image", is_configured: true }
    ] as any)

    renderSection([
      {
        id: "smirk",
        state: "smirk",
        starter: false,
        image: { mode: "url", url: "", base64: "" }
      }
    ])

    fireEvent.click(screen.getByText("Generate"))

    const prompt = (await screen.findByLabelText(
      "Generation prompt for smirk"
    )) as HTMLTextAreaElement
    await waitFor(() =>
      expect(prompt).toHaveValue("Portrait of Mira, showing smirk")
    )

    await userEvent.clear(prompt)

    await waitFor(() => expect(prompt).toHaveValue(""))
  })

  it("applies uploaded images by stable row id after earlier rows are removed", async () => {
    let fileReader: {
      result: string | null
      onload: (() => void) | null
      onerror: (() => void) | null
      readAsDataURL: (file: File) => void
    } | null = null

    class MockFileReader {
      result: string | null = null
      error: Error | null = null
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      readAsDataURL = vi.fn(() => {
        fileReader = this
      })
    }

    vi.stubGlobal("FileReader", MockFileReader)

    const initialRows: ExpressionImageRow[] = [
      {
        id: "neutral",
        state: "neutral",
        starter: true,
        image: { mode: "url", url: "", base64: "" }
      },
      {
        id: "target",
        state: "thinking",
        starter: false,
        image: { mode: "url", url: "", base64: "" }
      }
    ]
    const { container, getRows, setRows } = renderSection(initialRows)

    act(() => {
      setRows([
        initialRows[0],
        {
          ...initialRows[1],
          image: { mode: "upload", url: "", base64: "" }
        }
      ])
    })
    await screen.findByRole("button", { name: "Upload image" })
    const input = await waitFor(() => {
      const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
      expect(fileInput).toBeTruthy()
      return fileInput
    })

    fireEvent.change(input, {
      target: { files: [new File(["x"], "thinking.png", { type: "image/png" })] }
    })
    await userEvent.click(
      screen.getByRole("button", { name: "Remove expression neutral" })
    )

    expect(fileReader).toBeTruthy()
    fileReader!.result = `data:image/png;base64,${ONE_PIXEL_PNG_BASE64}`
    fileReader!.onload?.()

    await waitFor(() =>
      expect(getRows()).toEqual([
        expect.objectContaining({
          id: "target",
          image: {
            mode: "upload",
            url: "",
            base64: ONE_PIXEL_PNG_BASE64
          }
        })
      ])
    )
  })
})
