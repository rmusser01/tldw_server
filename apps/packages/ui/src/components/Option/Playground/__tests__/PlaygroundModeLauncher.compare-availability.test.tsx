// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundModeLauncher } from "../PlaygroundModeLauncher"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
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
  )
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    ariaLabel,
    title,
    className,
    ...props
  }: {
    children: React.ReactNode
    ariaLabel?: string
    title?: string
    className?: string
    [key: string]: unknown
  }) => (
    <button
      type="button"
      aria-label={ariaLabel}
      title={title}
      className={className}
      {...props}
    >
      {children}
    </button>
  )
}))

const baseProps = (
  compareFeatureEnabled: boolean
): React.ComponentProps<typeof PlaygroundModeLauncher> => ({
  open: true,
  onOpenChange: vi.fn(),
  compareModeActive: false,
  compareFeatureEnabled,
  onToggleCompare: vi.fn(),
  selectedCharacterName: null,
  onOpenActorSettings: vi.fn(),
  contextToolsOpen: false,
  onToggleKnowledgePanel: vi.fn(),
  voiceChatEnabled: false,
  voiceChatAvailable: true,
  isSending: false,
  onVoiceChatToggle: vi.fn(),
  webSearch: false,
  hasWebSearch: true,
  onToggleWebSearch: vi.fn(),
  onModeAnnouncement: vi.fn(),
  t: (key: string, fallback?: string) => fallback || key
})

// CM-05 (#3101), owner decision D4: Compare stays hidden until it is verified.
describe("PlaygroundModeLauncher compare availability", () => {
  it("hides Compare responses while compare is unavailable", () => {
    render(<PlaygroundModeLauncher {...baseProps(false)} />)

    expect(screen.queryByRole("button", { name: /Compare responses/i })).toBeNull()
    expect(screen.getByRole("button", { name: /Character \/ Scene/i })).toBeEnabled()
  })

  it("offers Compare responses once compare is enabled", () => {
    render(<PlaygroundModeLauncher {...baseProps(true)} />)

    expect(screen.getByRole("button", { name: /Compare responses/i })).toBeEnabled()
  })
})
