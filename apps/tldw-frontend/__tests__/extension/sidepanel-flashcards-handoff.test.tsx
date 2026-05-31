import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { createTabMock, getURLMock, translations } = vi.hoisted(() => ({
  createTabMock: vi.fn(),
  getURLMock: vi.fn((path: string) => `chrome-extension://tldw${path}`),
  translations: {
    "sidepanel:flashcards.opening": "Abriendo tarjetas",
    "sidepanel:flashcards.openedInTab":
      "Tarjetas se abre en el espacio de trabajo completo de la extension.",
    "sidepanel:flashcards.openAgain": "Abrir tarjetas"
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (translations[key as keyof typeof translations]) {
        return translations[key as keyof typeof translations]
      }
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      return defaultValueOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: getURLMock
    },
    tabs: {
      create: createTabMock
    }
  }
}))

import SidepanelFlashcards from "../../extension/routes/sidepanel-flashcards"

describe("sidepanel flashcards handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    createTabMock.mockResolvedValue(undefined)
  })

  it("opens the localized flashcards workspace handoff route", async () => {
    render(<SidepanelFlashcards />)

    expect(
      await screen.findByRole("heading", { name: "Abriendo tarjetas" })
    ).toBeInTheDocument()
    expect(screen.getByText(translations["sidepanel:flashcards.openedInTab"])).toBeInTheDocument()

    await waitFor(() => {
      expect(getURLMock).toHaveBeenCalledWith("/options.html#/flashcards")
    })
    await waitFor(() => {
      expect(createTabMock).toHaveBeenCalledWith({
        url: "chrome-extension://tldw/options.html#/flashcards"
      })
    })

    fireEvent.click(screen.getByRole("button", { name: "Abrir tarjetas" }))

    expect(createTabMock).toHaveBeenCalledTimes(2)
    expect(createTabMock).toHaveBeenLastCalledWith({
      url: "chrome-extension://tldw/options.html#/flashcards"
    })
  })
})
