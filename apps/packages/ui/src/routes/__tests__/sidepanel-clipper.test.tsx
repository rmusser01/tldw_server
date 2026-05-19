import React from "react"
import userEvent from "@testing-library/user-event"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelClipper from "../sidepanel-clipper"
import { buildClipDraft } from "@/services/web-clipper/draft-builder"
import {
  clearPendingClipDraft,
  writePendingClipDraft
} from "@/services/web-clipper/pending-draft"

const onlineMocks = vi.hoisted(() => ({
  useServerOnline: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => onlineMocks.useServerOnline()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

vi.mock("~/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: ({ activeTitle }: { activeTitle?: string }) => (
    <div data-testid="sidepanel-header">{activeTitle || "header"}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    routeId,
    children
  }: {
    routeId: string
    children: React.ReactNode
  }) => <div data-testid={`route-boundary-${routeId}`}>{children}</div>
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  __esModule: true,
  default: ({
    title,
    description
  }: {
    title: React.ReactNode
    description?: React.ReactNode
  }) => (
    <div data-testid="feature-empty-state">
      <div>{title}</div>
      {description ? <div>{description}</div> : null}
    </div>
  )
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

const createDraft = () =>
  buildClipDraft({
    clipId: "clip-123",
    requestedType: "article",
    pageUrl: "https://example.com/story",
    pageTitle: "Example Story",
    extracted: {
      articleText: "Alpha body copy",
      fullPageText: "Alpha body copy"
    }
  })

describe("sidepanel clipper route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    clearPendingClipDraft()
    onlineMocks.useServerOnline.mockReturnValue(true)
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasWebClipper: true
      }
    })
  })

  it("hydrates the pending draft into the review sheet", () => {
    writePendingClipDraft(createDraft())

    render(<SidepanelClipper />)

    expect(screen.getByTestId("route-boundary-sidepanel-clipper")).toBeVisible()
    expect(screen.getByTestId("sidepanel-header")).toHaveTextContent("Clipper")
    expect(screen.getByLabelText("Title")).toHaveValue("Example Story")
    expect(screen.getByLabelText("Comment")).toHaveValue("")
  })

  it("shows a missing-draft guard when no clip is available", () => {
    render(<SidepanelClipper />)

    expect(screen.getByText("No clip is ready to review yet.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Title")).not.toBeInTheDocument()
  })

  it("shows an unsupported-capability guard when the connected server lacks clipper support", () => {
    writePendingClipDraft(createDraft())
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: {
        hasWebClipper: false
      }
    })

    render(<SidepanelClipper />)

    expect(
      screen.getByText("This server does not advertise web clipper support.")
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("Title")).not.toBeInTheDocument()
  })

  it("clears the pending draft when cancel is selected", async () => {
    const user = userEvent.setup()
    writePendingClipDraft(createDraft())

    render(<SidepanelClipper />)

    await user.click(screen.getByRole("button", { name: "Cancel" }))

    expect(screen.getByText("No clip is ready to review yet.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Title")).not.toBeInTheDocument()
  })
})
