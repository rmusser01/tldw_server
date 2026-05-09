import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen } from "@testing-library/react"
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

const useQueryMock = vi.mocked(useQuery)
const useStorageMock = vi.mocked(useStorage)

describe("GenerateCharacterPanel", () => {
  beforeEach(() => {
    useStorageMock.mockReturnValue([null, vi.fn(), { isLoading: false }] as any)
    useQueryMock.mockReturnValue({
      data: [],
      isLoading: false
    } as any)
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
})
