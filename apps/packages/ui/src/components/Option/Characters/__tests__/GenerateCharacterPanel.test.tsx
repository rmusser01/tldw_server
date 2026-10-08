import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { useQuery } from "@tanstack/react-query"
import { useStorage } from "@plasmohq/storage/hook"
import { GenerateCharacterPanel } from "../GenerateCharacterPanel"

vi.mock("@tanstack/react-query", () => ({
  useQuery: vi.fn()
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

const useQueryMock = vi.mocked(useQuery)
const useStorageMock = vi.mocked(useStorage)

describe("GenerateCharacterPanel", () => {
  beforeEach(() => {
    useStorageMock.mockReturnValue([null, vi.fn(), { isLoading: false }] as any)
    useQueryMock.mockReturnValue({
      data: [],
      isLoading: false,
      isSuccess: true,
      isFetching: false
    } as any)
  })

  it("offers discovery Retry instead of claiming no generation models exist", async () => {
    const retry = vi.fn()
    useQueryMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      isSuccess: false,
      isFetching: false,
      isError: true,
      refetch: retry
    } as ReturnType<typeof useQuery>)
    render(
      <MemoryRouter>
        <GenerateCharacterPanel
          isGenerating={false}
          error={null}
          onGenerate={vi.fn()}
          onCancel={vi.fn()}
          onClearError={vi.fn()}
        />
      </MemoryRouter>
    )
    expect(
      screen.queryByText("No AI generation model available")
    ).not.toBeInTheDocument()
    expect(screen.getByText("Could not check chat models")).toBeInTheDocument()
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }))
    expect(retry).toHaveBeenCalledOnce()
  })

  it("treats missing AI generation models separately from saved character chat", () => {
    render(
      <MemoryRouter>
        <GenerateCharacterPanel
          isGenerating={false}
          error={null}
          onGenerate={vi.fn()}
          onCancel={vi.fn()}
          onClearError={vi.fn()}
        />
      </MemoryRouter>
    )

    expect(
      screen
        .getByText("No AI generation model available")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(
      screen.getByText("No AI generation model available")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        /Saved characters remain available for browsing, editing, and chat once a chat model is configured\./
      )
    ).toBeInTheDocument()

    const settingsLink = screen.getByRole("link", {
      name: "Go to model settings"
    })
    expect(settingsLink).toHaveAttribute("href", "/settings/model")
  })

  it("renders generation errors through the design-system Alert primitive with retry behavior", async () => {
    const user = userEvent.setup()
    const onClearError = vi.fn()

    useStorageMock.mockReturnValue([
      "gpt-4o",
      vi.fn(),
      { isLoading: false }
    ] as any)
    useQueryMock.mockReturnValue({
      data: [{ model: "gpt-4o", provider: "openai" }],
      isLoading: false,
      isSuccess: true,
      isFetching: false
    } as any)

    render(
      <MemoryRouter>
        <GenerateCharacterPanel
          isGenerating={false}
          error="network connection error"
          onGenerate={vi.fn()}
          onCancel={vi.fn()}
          onClearError={onClearError}
        />
      </MemoryRouter>
    )

    const errorTitle = screen.getByText("Network connection error.")
    const alert = errorTitle.closest('[data-ds-component="Alert"]')

    expect(alert).toBeInTheDocument()
    expect(alert).toHaveTextContent(
      "Check your internet connection and try again."
    )

    await user.click(
      within(alert as HTMLElement).getByRole("button", { name: "Close" })
    )
    expect(onClearError).toHaveBeenCalledTimes(1)

    await user.click(
      within(alert as HTMLElement).getByRole("button", { name: "Try again" })
    )

    expect(onClearError).toHaveBeenCalledTimes(2)
  })

  it.each([
    { initialSelection: null, expectedProvider: "openai" },
    { initialSelection: "ollama:shared", expectedProvider: "ollama" }
  ])(
    "generates with the actual model/provider after startup or stored canonical selection %j",
    async ({ initialSelection, expectedProvider }) => {
      useStorageMock.mockImplementation(function useTestStorage() {
        const [selected, setSelected] = React.useState(initialSelection)
        return [selected, setSelected, { isLoading: false }] as ReturnType<
          typeof useStorage
        >
      })
      useQueryMock.mockReturnValue({
        data: [
          { model: "tldw:shared", provider: "ollama", nickname: "Device chat" },
          {
            model: "tldw:shared",
            provider: "openai",
            nickname: "Cloud chat",
            is_server_default: true
          }
        ],
        isLoading: false,
        isSuccess: true,
        isFetching: false
      } as ReturnType<typeof useQuery>)
      const onGenerate = vi.fn()
      const user = userEvent.setup()
      render(
        <MemoryRouter>
          <GenerateCharacterPanel
            isGenerating={false}
            error={null}
            onGenerate={onGenerate}
            onCancel={vi.fn()}
            onClearError={vi.fn()}
          />
        </MemoryRouter>
      )
      await user.type(screen.getByRole("textbox"), "A patient library guide")
      await user.click(
        screen.getByRole("button", { name: "Generate", exact: true })
      )
      expect(onGenerate).toHaveBeenCalledWith(
        "A patient library guide",
        "shared",
        expectedProvider
      )
    }
  )

  it("keeps a unique legacy stored choice usable and sends its API model ID", async () => {
    useStorageMock.mockReturnValue([
      "tldw:legacy",
      vi.fn(),
      { isLoading: false }
    ] as ReturnType<typeof useStorage>)
    useQueryMock.mockReturnValue({
      data: [
        { model: "tldw:legacy", provider: "openai", nickname: "Legacy chat" }
      ],
      isLoading: false,
      isSuccess: true,
      isFetching: false
    } as ReturnType<typeof useQuery>)
    const onGenerate = vi.fn()
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <GenerateCharacterPanel
          isGenerating={false}
          error={null}
          onGenerate={onGenerate}
          onCancel={vi.fn()}
          onClearError={vi.fn()}
        />
      </MemoryRouter>
    )
    await user.type(screen.getByRole("textbox"), "A patient library guide")
    await user.click(
      screen.getByRole("button", { name: "Generate", exact: true })
    )
    expect(onGenerate).toHaveBeenCalledWith(
      "A patient library guide",
      "legacy",
      "openai"
    )
  })

  it("requires an explicit provider for an ambiguous legacy choice", async () => {
    useStorageMock.mockImplementation(function useTestStorage() {
      const [selected, setSelected] = React.useState("tldw:shared")
      return [selected, setSelected, { isLoading: false }] as ReturnType<
        typeof useStorage
      >
    })
    useQueryMock.mockReturnValue({
      data: [
        { model: "tldw:shared", provider: "ollama", nickname: "Device chat" },
        {
          model: "tldw:shared",
          provider: "openai",
          nickname: "Cloud chat",
          is_server_default: true
        }
      ],
      isLoading: false,
      isSuccess: true,
      isFetching: false
    } as ReturnType<typeof useQuery>)
    const onGenerate = vi.fn()
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <GenerateCharacterPanel
          isGenerating={false}
          error={null}
          onGenerate={onGenerate}
          onCancel={vi.fn()}
          onClearError={vi.fn()}
        />
      </MemoryRouter>
    )
    await user.type(screen.getByRole("textbox"), "A patient library guide")
    expect(
      screen.getByRole("button", { name: "Generate", exact: true })
    ).toBeDisabled()
    await user.click(
      screen.getByRole("button", { name: "Select AI model for generation" })
    )
    await user.click(screen.getByRole("menuitem", { name: "Cloud chat" }))
    await user.click(
      screen.getByRole("button", { name: "Generate", exact: true })
    )
    expect(onGenerate).toHaveBeenCalledWith(
      "A patient library guide",
      "shared",
      "openai"
    )
  })
})
