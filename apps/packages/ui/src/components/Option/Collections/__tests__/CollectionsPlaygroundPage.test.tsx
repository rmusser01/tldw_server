import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { CollectionsPlaygroundPage } from "../index"

const navigateMock = vi.fn()
const setActiveTabMock = vi.fn()
const resetStoreMock = vi.fn()

let isOnline = true
let uxState:
  | "connected_ok"
  | "configuring_auth"
  | "error_auth"
  | "error_unreachable"
  | "unconfigured" = "connected_ok"
let hasCompletedFirstRun = true

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("antd", () => ({
  Tabs: ({
    activeKey,
    items,
    onChange
  }: {
    activeKey?: string
    items?: Array<{ key: string; label: React.ReactNode; children: React.ReactNode }>
    onChange?: (key: string) => void
  }) => {
    const activeItem = items?.find((item) => item.key === activeKey) ?? items?.[0]
    return (
      <div>
        <div role="tablist">
          {items?.map((item) => (
            <button
              key={item.key}
              role="tab"
              type="button"
              onClick={() => onChange?.(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div>{activeItem?.children}</div>
      </div>
    )
  },
  Empty: ({ description }: { description?: React.ReactNode }) => (
    <div>{description}</div>
  )
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: "/collections", search: "", hash: "", state: null }),
  useNavigate: () => navigateMock
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => isOnline
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState,
    hasCompletedFirstRun
  })
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
  )
}))

vi.mock("@/components/Common/DismissibleBetaAlert", () => ({
  DismissibleBetaAlert: () => <div data-testid="collections-beta-alert">beta</div>
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  __esModule: true,
  default: ({
    title,
    description,
    primaryActionLabel,
    onPrimaryAction
  }: {
    title?: React.ReactNode
    description?: React.ReactNode
    primaryActionLabel?: React.ReactNode
    onPrimaryAction?: () => void
  }) => (
    <div>
      <h2>{title}</h2>
      <p>{description}</p>
      {primaryActionLabel ? (
        <button type="button" onClick={onPrimaryAction}>
          {primaryActionLabel}
        </button>
      ) : null}
    </div>
  )
}))

vi.mock("@/store/collections", () => ({
  useCollectionsStore: (
    selector: (state: {
      activeTab: "reading"
      setActiveTab: typeof setActiveTabMock
      resetStore: typeof resetStoreMock
    }) => unknown
  ) =>
    selector({
      activeTab: "reading",
      setActiveTab: setActiveTabMock,
      resetStore: resetStoreMock
    })
}))

vi.mock("../ReadingList/ReadingItemsList", () => ({
  ReadingItemsList: () => <div>Reading Items List</div>
}))

vi.mock("../Highlights/HighlightsList", () => ({
  HighlightsList: () => <div>Highlights List</div>
}))

vi.mock("../Templates/TemplatesList", () => ({
  TemplatesList: () => <div>Templates List</div>
}))

vi.mock("../ImportExport/ImportExportPanel", () => ({
  ImportExportPanel: () => <div>Import Export Panel</div>
}))

vi.mock("../Digests/DigestSchedulesPanel", () => ({
  DigestSchedulesPanel: () => <div>Digest Schedules Panel</div>
}))

describe("CollectionsPlaygroundPage", () => {
  beforeEach(() => {
    isOnline = true
    uxState = "connected_ok"
    hasCompletedFirstRun = true
    navigateMock.mockReset()
    setActiveTabMock.mockReset()
    resetStoreMock.mockReset()
  })

  it("shows auth guidance instead of generic offline copy when server credentials are missing", () => {
    isOnline = false
    uxState = "error_auth"

    render(<CollectionsPlaygroundPage />)

    expect(
      screen.getByText("Add your credentials before Collections can load data.")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(navigateMock).toHaveBeenCalledWith("/settings/tldw")
    expect(
      screen.queryByText("Can't reach your tldw server right now.")
    ).not.toBeInTheDocument()
  })

  it("shows setup guidance when first-run onboarding has not been completed", () => {
    isOnline = false
    uxState = "unconfigured"
    hasCompletedFirstRun = false

    render(<CollectionsPlaygroundPage />)

    expect(
      screen.getByText("Finish setup before using Collections.")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(navigateMock).toHaveBeenCalledWith("/")
  })

  it("keeps the offline message for actual unreachable server states", () => {
    isOnline = false
    uxState = "error_unreachable"

    render(<CollectionsPlaygroundPage />)

    expect(
      screen.getByText("Can't reach your tldw server right now.")
    ).toBeInTheDocument()
  })

  it("renders collections tabs when the connection is ready", () => {
    render(<CollectionsPlaygroundPage />)

    expect(screen.getByText("Collections")).toBeInTheDocument()
    expect(screen.getByText("Reading Items List")).toBeInTheDocument()
  })
})
