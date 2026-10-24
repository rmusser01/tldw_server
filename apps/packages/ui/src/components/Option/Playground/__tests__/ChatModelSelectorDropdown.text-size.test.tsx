import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConfigProvider } from "antd"

import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))

// AX-07 (#3124): the model chip and its provider status badge rendered at 10px
// and 9px. Status text must be at least 11px to stay readable.
describe("Chat model selector text size", () => {
  it("renders the model chip and its status badge at 11px or larger", () => {
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <ChatModelSelectorDropdown
          apiModelLabel="Llama.cpp / Gemma"
          connectionStatusLabel="Healthy"
          modelDropdownOpen={false}
          modelDropdownMenuItems={[]}
          resolvedProviderKey="llamacpp"
          selectedModel="llamacpp:Gemma"
          setModelDropdownOpen={() => undefined}
          setModelSearchQuery={() => undefined}
        />
      </ConfigProvider>
    )

    const chip = screen.getByTestId("model-selector")
    const badge = screen.getByTitle("Status")
    for (const element of [chip, badge]) {
      expect(element.className).not.toMatch(/\btext-\[(?:[0-9]|10)px\]/)
      expect(element).toHaveClass("text-[11px]")
    }
  })

  it("renders the one-click model switch at 11px with a visible focus ring", () => {
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <ChatModelSelectorDropdown
          apiModelLabel="gpt-4o (OpenAI)"
          connectionStatusLabel="Server connected"
          modelDropdownOpen={false}
          modelDropdownMenuItems={[]}
          modelReadiness={{
            state: "unreachable",
            label: "Unreachable",
            title: "OpenAI did not answer",
            switchTarget: { key: "anthropic:claude", modelLabel: "claude (Anthropic)", label: "Switch to claude (Anthropic)" },
            switchToTarget: () => undefined
          }}
          resolvedProviderKey="openai"
          selectedModel="openai:gpt-4o"
          setModelDropdownOpen={() => undefined}
          setModelSearchQuery={() => undefined}
        />
      </ConfigProvider>
    )

    const switchButton = screen.getByTestId("model-readiness-switch")
    expect(switchButton.className).not.toMatch(/\btext-\[(?:[0-9]|10)px\]/)
    expect(switchButton).toHaveClass("text-[11px]", "focus-visible:ring-2", "focus-visible:ring-focus")
  })
})
