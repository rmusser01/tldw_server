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

// CC-08 (#3101), decision [1]: Modes holds conversation modes. Voice has its
// own composer button, and More tools' Voice settings explains when voice is
// unavailable (PlaygroundToolsPopover.voice-availability.test.tsx).
describe("PlaygroundModeLauncher voice availability", () => {
  it("leaves Voice to the composer's voice controls", () => {
    const props = {
      open: true,
      onOpenChange: vi.fn(),
      compareModeActive: false,
      compareFeatureEnabled: true,
      onToggleCompare: vi.fn(),
      selectedCharacterName: null,
      onOpenActorSettings: vi.fn(),
      onModeAnnouncement: vi.fn(),
      t: (key: string, fallback?: string) => fallback || key
    } satisfies React.ComponentProps<typeof PlaygroundModeLauncher>

    render(<PlaygroundModeLauncher {...props} />)

    expect(
      screen.queryByRole("button", { name: /Voice mode/i })
    ).not.toBeInTheDocument()
  })
})
