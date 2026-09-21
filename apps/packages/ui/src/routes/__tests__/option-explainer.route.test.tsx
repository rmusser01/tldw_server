import React, { Suspense } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { expect, it, vi } from "vitest"

import { explainerApi } from "@/components/Option/Explainer/explainerApi"
import { optionRoutes } from "../route-registry"

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("../option-index", () => ({ default: () => null }))
vi.mock("../settings-route", () => ({
  createSettingsRoute: () => () => null
}))

it("opens the Explainer workspace from the extension options registry", async () => {
  vi.spyOn(explainerApi, "listSessions").mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0
  })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const route = optionRoutes.find(
    (candidate) => candidate.path === "/explainer"
  )

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/explainer"]}>
        <Suspense fallback={null}>
          <Routes>
            {route && <Route path={route.path} element={route.element} />}
            <Route path="*" element={<p>Page not found</p>} />
          </Routes>
        </Suspense>
      </MemoryRouter>
    </QueryClientProvider>
  )

  expect(
    await screen.findByRole("heading", { name: "Explainer" })
  ).toBeVisible()
})
