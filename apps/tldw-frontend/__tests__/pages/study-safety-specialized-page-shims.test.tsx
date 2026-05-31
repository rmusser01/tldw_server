import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

const dynamicCalls = vi.hoisted(
  () => [] as Array<{ ssr?: boolean }>
)

vi.mock("next/dynamic", async () => {
  const ReactModule = await vi.importActual<typeof import("react")>("react")

  return {
    default: (_loader: () => unknown, options?: { ssr?: boolean }) => {
      dynamicCalls.push({
        ssr: options?.ssr
      })

      return function MockDynamicPage() {
        return ReactModule.createElement("div", {
          "data-testid": "dynamic-next-page"
        })
      }
    }
  }
})

vi.mock("@web/components/navigation/RouteRedirect", () => ({
  RouteRedirect: ({ to }: { to: string }) => (
    <div data-testid="route-redirect" data-to={to} />
  )
}))

import ClaimsReviewPage from "@web/pages/claims-review"
import VNAssetsPage from "@web/pages/vn-assets"
import VNPlayPage from "@web/pages/vn-play"

describe("study, safety, and specialized Next page shims", () => {
  it("keeps claims-review as a content-review redirect page", () => {
    render(<ClaimsReviewPage />)

    expect(screen.getByTestId("route-redirect")).toHaveAttribute(
      "data-to",
      "/content-review"
    )
  })

  it("keeps VN labs routes as client-only Next dynamic pages", () => {
    render(<VNAssetsPage />)
    render(<VNPlayPage />)

    expect(dynamicCalls.filter((call) => call.ssr === false)).toHaveLength(2)
    expect(screen.getAllByTestId("dynamic-next-page")).toHaveLength(2)
  })
})
