// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: any) => (
    <a href={typeof to === "string" ? to : "#"} {...rest}>
      {children}
    </a>
  )
}))

vi.mock("antd", () => ({
  Dropdown: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: ({ provider }: { provider: string }) => (
    <span data-testid="provider-icon">{provider}</span>
  )
}))

const baseProps = {
  activeModelKey: "openai:gpt-4o",
  apiModelLabel: "OpenAI / gpt-4o",
  catalogControls: null,
  connectionStatusLabel: "Connected",
  modelDropdownMenuItems: [],
  modelDropdownOpen: false,
  modelSelectorWarning: false,
  resolvedProviderKey: "openai",
  setModelDropdownOpen: vi.fn(),
  setModelSearchQuery: vi.fn()
}

describe("ChatModelSelectorDropdown character model usability", () => {
  it("keeps provider and model visible while naming provider setup blockers", () => {
    render(
      <ChatModelSelectorDropdown
        {...baseProps}
        modelUsabilityLabel="Provider setup needed"
        modelUsabilityTitle="Configure the selected model provider before chatting as Ada"
        modelUsabilityWarning
      />
    )

    const selector = screen.getByRole("button", {
      name: /configure the selected model provider before chatting as ada/i
    })

    expect(selector).toHaveTextContent(
      "OpenAI / gpt-4o - Provider setup needed"
    )
    expect(selector).toHaveAttribute(
      "title",
      "Configure the selected model provider before chatting as Ada"
    )
    expect(selector).not.toHaveTextContent("Healthy")
    expect(selector).not.toHaveTextContent("Ready")
  })

  it("uses neutral checking copy while model readiness is loading", () => {
    render(
      <ChatModelSelectorDropdown
        {...baseProps}
        modelUsabilityLabel="Checking model readiness"
        modelUsabilityTitle="Checking chat model readiness"
        modelUsabilityWarning
      />
    )

    const selector = screen.getByRole("button", {
      name: /checking chat model readiness/i
    })

    expect(selector).toHaveTextContent(
      "OpenAI / gpt-4o - Checking model readiness"
    )
    expect(selector).not.toHaveTextContent("Healthy")
    expect(selector).not.toHaveTextContent("Ready")
  })
})
