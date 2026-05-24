import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AvatarField } from "../AvatarField"
import { tldwClient } from "@/services/tldw/TldwApiClient"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getImageBackends: vi.fn(),
    createImageArtifact: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

const getImageBackendsMock = vi.mocked(tldwClient.getImageBackends)
const createImageArtifactMock = vi.mocked(tldwClient.createImageArtifact)
let consoleErrorMock: ReturnType<typeof vi.spyOn> | null = null

describe("AvatarField design-system alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    consoleErrorMock?.mockRestore()
    consoleErrorMock = null
  })

  it("renders the no-backend state through the design-system Alert primitive", async () => {
    getImageBackendsMock.mockResolvedValue([])

    render(<AvatarField value={{ mode: "generate" }} onChange={vi.fn()} />)

    const title = await screen.findByText("No image backends configured")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).toBeInTheDocument()
    expect(alert).toHaveTextContent(
      "Configure stable-diffusion or SwarmUI in server settings to enable avatar generation."
    )
  })

  it("renders generation failures through the design-system Alert primitive and keeps dismiss behavior", async () => {
    const user = userEvent.setup()
    consoleErrorMock = vi.spyOn(console, "error").mockImplementation(() => undefined)

    getImageBackendsMock.mockResolvedValue([
      {
        id: "stable-diffusion",
        name: "Stable Diffusion",
        is_configured: true
      }
    ] as any)
    createImageArtifactMock.mockRejectedValue(new Error("image_generation_failed"))

    render(<AvatarField value={{ mode: "generate" }} onChange={vi.fn()} />)

    await user.type(
      await screen.findByPlaceholderText("Portrait of a wise mentor with kind eyes..."),
      "portrait of a scholar"
    )
    await user.click(screen.getByRole("button", { name: "Generate Avatar" }))

    const errorTitle = await screen.findByText("Generation failed. Try again.")
    const alert = errorTitle.closest('[data-ds-component="Alert"]')

    expect(alert).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Dismiss" }))

    expect(screen.queryByText("Generation failed. Try again.")).not.toBeInTheDocument()
  })
})
