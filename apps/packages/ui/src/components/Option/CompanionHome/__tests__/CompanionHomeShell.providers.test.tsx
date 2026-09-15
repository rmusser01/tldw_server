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
      screen.queryByText(/Configure an LLM provider/)
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
    screen.queryByText(/Configure an LLM provider/)
  ).not.toBeInTheDocument()
})

it("offers local or hosted setup only after confirming no models", async () => {
  fetchModels.mockResolvedValue([])
  renderHome()
  expect(
    await screen.findByText(/Configure an LLM provider/)
  ).toBeInTheDocument()
  expect(
    screen.getByText(/local model server or a hosted provider/i)
  ).toBeInTheDocument()
})
