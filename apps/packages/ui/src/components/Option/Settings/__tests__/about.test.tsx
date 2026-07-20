// @vitest-environment jsdom

import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getOllamaURL: vi.fn(),
  fetcher: vi.fn()
}))

vi.mock("~/services/tldw-server", () => ({
  getOllamaURL: (...args: unknown[]) => mocks.getOllamaURL(...args)
}))

vi.mock("@/libs/fetcher", () => ({
  default: (...args: unknown[]) => mocks.fetcher(...args)
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

import { AboutApp } from "../about"

const renderWithQueryClient = (ui: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
  )
}

describe("AboutApp", () => {
  beforeEach(() => {
    mocks.getOllamaURL.mockReset()
    mocks.fetcher.mockReset()
  })

  it("reads server version from the OpenAPI document", async () => {
    mocks.getOllamaURL.mockResolvedValue("http://127.0.0.1:8000/")
    mocks.fetcher.mockResolvedValue({
      ok: true,
      json: async () => ({
        info: {
          version: "1.2.3"
        }
      })
    })

    renderWithQueryClient(<AboutApp />)

    expect(await screen.findByText("1.2.3")).toBeInTheDocument()
    await waitFor(() => {
      expect(mocks.fetcher).toHaveBeenCalledWith("http://127.0.0.1:8000/openapi.json")
    })
  })

  it("links to the current repository and frontend license terms", async () => {
    mocks.getOllamaURL.mockResolvedValue("http://127.0.0.1:8000/")
    mocks.fetcher.mockResolvedValue({
      ok: true,
      json: async () => ({ info: { version: "1.2.3" } })
    })

    renderWithQueryClient(<AboutApp />)

    const repositoryLink = await screen.findByRole("link", {
      name: "tldw_server on GitHub"
    })
    expect(repositoryLink).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server"
    )

    expect(
      screen.getByRole("link", { name: "Source-available frontend terms" })
    ).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server/blob/dev/LICENSE"
    )
  })
})
