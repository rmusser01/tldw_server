import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, expect, it, vi } from "vitest"
import { CompanionHomeShell } from "../CompanionHomeShell"

const fetchModels = vi.hoisted(() => vi.fn())
vi.mock("@/hooks/useConnectionState", () => ({ useIsConnected: () => true }))
vi.mock("@/context/demo-mode", () => ({
  useSafeDemoMode: () => ({ demoEnabled: false })
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: fetchModels }))
vi.mock("../CompanionHomePage", () => ({ CompanionHomePage: () => null }))
vi.mock("@/components/Common/StorageQuotaBanner", () => ({
  StorageQuotaBanner: () => null
}))

function renderHome() {
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <MemoryRouter>
        <CompanionHomeShell surface="options" />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  fetchModels.mockReset()
})

it("does not claim providers are missing while discovery is pending", async () => {
  let resolve!: (models: unknown[]) => void
  fetchModels.mockReturnValue(
    new Promise((done) => {
      resolve = done
    })
  )
  renderHome()
  try {
    expect(
      screen.queryByRole("link", { name: "Review model setup" })
    ).not.toBeInTheDocument()
  } finally {
    resolve([{ id: "local-model" }])
  }
})

it("does not claim providers are missing when discovery fails", async () => {
  fetchModels.mockRejectedValue(new Error("Offline"))
  renderHome()
  await waitFor(() => expect(fetchModels).toHaveBeenCalled())
  expect(
    screen.queryByRole("link", { name: "Review model setup" })
  ).not.toBeInTheDocument()
})

it("links empty discovery to readiness and setup guidance without promising a provider editor", async () => {
  fetchModels.mockResolvedValue([])
  renderHome()
  expect(
    await screen.findByRole("link", { name: "Review model setup" })
  ).toHaveAttribute("href", "/settings/model")
  expect(
    screen.getByText(/readiness and server setup guidance/i)
  ).toBeInTheDocument()
  expect(screen.queryByText(/Connect a local model server or a hosted provider in Model Settings/)).not.toBeInTheDocument()
})
