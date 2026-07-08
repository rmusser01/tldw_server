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
})
