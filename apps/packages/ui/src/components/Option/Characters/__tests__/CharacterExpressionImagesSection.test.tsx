import React from "react"
import { Form, type FormInstance } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { CharacterExpressionImagesSection } from "../CharacterExpressionImagesSection"
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

  render(<Harness />)
  return {
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

  it("renders starter expression rows and adds a custom row", async () => {
    render(
      <Form initialValues={{ expression_images: expressionRowsFromExtensions({}) }}>
        <CharacterExpressionImagesSection
          characterName="Mira"
          characterDescription="Archivist"
        />
      </Form>
    )

    expect(screen.getByDisplayValue("neutral")).toBeInTheDocument()
    expect(screen.getByDisplayValue("thinking")).toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: /add expression/i }))
    expect(screen.getByLabelText(/custom expression state/i)).toBeInTheDocument()
  })

  it("copies the selected preview emote directive", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
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
})
