import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { SearchModeSettings } from "../search-mode"

const queryState = vi.hoisted(() => ({
  status: "error" as "pending" | "error" | "success"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => {
      if (key === "generalSettings.webSearch.heading") return "Manage Web Search"
      if (key === "generalSettings.webSearch.provider.label") return "Web Search Provider"
      return fallback ?? key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ status: queryState.status })
}))

vi.mock("@/hooks/useSimpleForm", () => ({
  useSimpleForm: () => ({
    getInputProps: () => ({}),
    isDirty: () => false,
    onSubmit: () => (event: { preventDefault: () => void }) =>
      event.preventDefault(),
    resetDirty: vi.fn(),
    setValues: vi.fn(),
    values: {}
  })
}))

describe("SearchModeSettings", () => {
  it("keeps the section identifiable while search settings are loading", () => {
    queryState.status = "pending"

    const { container } = render(<SearchModeSettings />)

    expect(screen.getByRole("heading", { name: "Manage Web Search" })).toBeInTheDocument()
    expect(container.querySelector(".ant-skeleton")).toBeInTheDocument()
    expect(
      screen.queryByText(
        "Web search settings are unavailable until the server is reachable."
      )
    ).not.toBeInTheDocument()
  })

  it("keeps the section identifiable when search settings cannot load", () => {
    queryState.status = "error"

    render(<SearchModeSettings />)

    expect(screen.getByRole("heading", { name: "Manage Web Search" })).toBeInTheDocument()
    expect(
      screen.getByText(
        "Web search settings are unavailable until the server is reachable."
      )
    ).toBeInTheDocument()
  })

  it("renders the editable search form after settings load", () => {
    queryState.status = "success"

    render(<SearchModeSettings />)

    expect(screen.getByRole("heading", { name: "Manage Web Search" })).toBeInTheDocument()
    expect(screen.getByText("Web Search Provider")).toBeInTheDocument()
    expect(
      screen.queryByText(
        "Web search settings are unavailable until the server is reachable."
      )
    ).not.toBeInTheDocument()
  })
})
