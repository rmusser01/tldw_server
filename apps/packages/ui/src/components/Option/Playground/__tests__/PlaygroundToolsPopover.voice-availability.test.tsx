// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundToolsPopover } from "../PlaygroundToolsPopover"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

vi.mock("react-router-dom", () => ({
  Link: ({
    children,
    ...props
  }: {
    children: React.ReactNode
    [key: string]: unknown
  }) => <a {...props}>{children}</a>
}))

vi.mock("antd", () => ({
  Popover: ({
    content,
    children
  }: {
    content: React.ReactNode
    children: React.ReactNode
  }) => (
    <div data-testid="popover">
      {children}
      {content}
    </div>
  ),
  Tooltip: ({
    title,
    children
  }: {
    title: React.ReactNode
    children: React.ReactNode
  }) => (
    <div data-testid="voice-tooltip" data-title={String(title)}>
      {children}
    </div>
  ),
  Switch: ({
    checked,
    onChange
  }: {
    checked?: boolean
    onChange?: (checked: boolean) => void
  }) => (
    <button type="button" aria-pressed={checked} onClick={() => onChange?.(!checked)}>
      switch
    </button>
  ),
  Radio: {
    Button: ({ children }: { children: React.ReactNode }) => <button type="button">{children}</button>
  }
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    ariaLabel,
    title,
    className,
    iconOnly: _iconOnly,
    ...buttonProps
  }: {
    children: React.ReactNode
    ariaLabel?: string
    title?: string
    className?: string
    iconOnly?: boolean
    [key: string]: unknown
  }) => (
    <button
      type="button"
      aria-label={ariaLabel}
      title={title}
      className={className}
      {...buttonProps}
    >
      {children}
    </button>
  )
}))

describe("PlaygroundToolsPopover voice availability", () => {
  it("keeps the voice control visible but uses the shared unavailable reason", () => {
    const reason = "This server does not advertise voice conversation streaming."
    const props = {
      toolsPopoverOpen: true,
      onToolsPopoverChange: vi.fn(),
      isProMode: false,
      onOpenImageGenerate: vi.fn(),
      useOCR: false,
      onUseOCRChange: vi.fn(),
      hasWebSearch: false,
      simpleInternetSearch: false,
      onSimpleInternetSearchChange: vi.fn(),
      defaultInternetSearchOn: false,
      onDefaultInternetSearchOnChange: vi.fn(),
      onNavigateWebSearchSettings: vi.fn(),
      advancedToolsExpanded: true,
      onAdvancedToolsExpandedChange: vi.fn(),
      allowExternalImages: false,
      onAllowExternalImagesChange: vi.fn(),
      showMoodBadge: false,
      onShowMoodBadgeChange: vi.fn(),
      showMoodConfidence: false,
      onShowMoodConfidenceChange: vi.fn(),
      onOpenRawRequest: vi.fn(),
      voiceChatAvailable: false,
      voiceChatEnabled: false,
      voiceChatState: "idle",
      voiceChatStatusLabel: "Voice chat",
      onVoiceChatToggle: vi.fn(),
      isSending: false,
      voiceChatSettingsFields: <div data-testid="voice-settings-fields" />,
      imageProviderControl: <div data-testid="image-provider-control" />,
      historyLength: 0,
      onClearContext: vi.fn(),
      t: (key: string, fallback?: string) => fallback || key,
      voiceChatUnavailableReason: reason
    } satisfies React.ComponentProps<typeof PlaygroundToolsPopover> & {
      voiceChatUnavailableReason: string
    }

    render(
      <PlaygroundToolsPopover {...props} />
    )

    const voiceSettingsButton = screen.getByRole("button", {
      name: "Voice settings"
    })

    expect(voiceSettingsButton).toBeDisabled()
    expect(screen.getByTestId("voice-tooltip")).toHaveAttribute(
      "data-title",
      reason
    )
  })
})
