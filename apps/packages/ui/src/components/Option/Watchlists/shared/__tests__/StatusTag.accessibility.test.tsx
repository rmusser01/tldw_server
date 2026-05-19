import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { StatusTag } from "../StatusTag"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

describe("StatusTag accessibility labels", () => {
  it("exposes a descriptive SR label for known run statuses", () => {
    const { container } = render(<StatusTag status="running" />)
    const badge = container.querySelector('[data-ds-component="Badge"]')

    expect(screen.getByLabelText("Run status: Running")).toHaveTextContent("Running")
    expect(screen.getByLabelText("Run status: Running")).toHaveAttribute("title", "Run status: Running")
    expect(screen.getByTestId("watchlists-status-icon-running")).toBeInTheDocument()
    expect(badge).toHaveAttribute("data-ds-variant", "secondary")
  })

  it("humanizes unknown statuses and keeps descriptive SR labels", () => {
    const { container } = render(<StatusTag status="in_progress" />)

    expect(screen.getByLabelText("Run status: In Progress")).toHaveTextContent("In Progress")
    expect(screen.getByLabelText("Run status: In Progress")).toHaveAttribute("title", "Run status: In Progress")
    expect(screen.getByTestId("watchlists-status-icon-unknown")).toBeInTheDocument()
    expect(container.querySelector('[data-ds-component="Badge"]')).toBeInTheDocument()
  })

  it("passes the compact Badge size for small status tags", () => {
    const { container } = render(<StatusTag status="pending" size="small" />)
    const badge = container.querySelector('[data-ds-component="Badge"]')

    expect(badge).toHaveAttribute("data-ds-size", "sm")
  })
})
