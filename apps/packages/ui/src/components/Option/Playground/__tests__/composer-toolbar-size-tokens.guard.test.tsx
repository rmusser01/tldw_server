// @vitest-environment jsdom
// Size tokens (CC-07 follow-up, #3101), owner decision [3]: composer toolbar
// controls are 32px on desktop and 44px on touch, through one set of tokens.
// D9 and D10 were dropped, so this covers sizes only.
import React from "react"
import { render, screen } from "@testing-library/react"
import { ConfigProvider } from "antd"
import { describe, expect, it, vi } from "vitest"

import { COMPOSER_TOOLBAR_TOKENS } from "@/components/Chat/composer/composer-toolbar-tokens"
import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"
import { PlaygroundModeLauncher } from "../PlaygroundModeLauncher"
import { PlaygroundAttachmentButton } from "../PlaygroundSendControl"
import { PlaygroundToolsPopover } from "../PlaygroundToolsPopover"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
  })
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))
vi.mock("react-router-dom", () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  )
}))

const t = (key: string, fallback?: string) => fallback ?? key

const expectTokens = (element: HTMLElement, token: string) => {
  for (const className of token.split(/\s+/)) {
    expect(element, `${className} on ${element.outerHTML.slice(0, 120)}`).toHaveClass(
      className
    )
  }
  // A literal 44px floor would undo the 32px desktop size.
  expect(element.className).not.toMatch(/(^|\s)min-h-\[44px\]/)
}

describe("composer toolbar size tokens", () => {
  it("defines 44px controls by default and on touch, 32px from the desktop breakpoint", () => {
    expect(COMPOSER_TOOLBAR_TOKENS.textControl.split(" ")).toEqual(
      expect.arrayContaining([
        "min-h-11",
        "sm:min-h-8",
        "[@media(pointer:coarse)]:min-h-11"
      ])
    )
    expect(COMPOSER_TOOLBAR_TOKENS.iconButton.split(" ")).toEqual(
      expect.arrayContaining([
        "h-11",
        "w-11",
        "sm:h-8",
        "sm:w-8",
        "[@media(pointer:coarse)]:h-11",
        "[@media(pointer:coarse)]:w-11"
      ])
    )
    expect(COMPOSER_TOOLBAR_TOKENS.iconControl.split(" ")).toEqual(
      expect.arrayContaining([
        "[@media(pointer:coarse)]:h-11",
        "[@media(pointer:coarse)]:w-11"
      ])
    )
  })

  it("sizes the model chip with the text-control token", () => {
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

    expectTokens(
      screen.getByTestId("model-selector"),
      COMPOSER_TOOLBAR_TOKENS.textControl
    )
  })

  it("sizes the Modes and More tools triggers with the tokens", () => {
    render(
      <>
        <PlaygroundModeLauncher
          open={false}
          onOpenChange={vi.fn()}
          compareModeActive={false}
          compareFeatureEnabled={false}
          onToggleCompare={vi.fn()}
          selectedCharacterName={null}
          onOpenActorSettings={vi.fn()}
          onModeAnnouncement={vi.fn()}
          t={t}
        />
        <PlaygroundToolsPopover
          toolsPopoverOpen={false}
          onToolsPopoverChange={vi.fn()}
          isProMode={false}
          onOpenImageGenerate={vi.fn()}
          useOCR={false}
          onUseOCRChange={vi.fn()}
          hasWebSearch={false}
          simpleInternetSearch={false}
          onSimpleInternetSearchChange={vi.fn()}
          defaultInternetSearchOn={false}
          onDefaultInternetSearchOnChange={vi.fn()}
          onNavigateWebSearchSettings={vi.fn()}
          advancedToolsExpanded={false}
          onAdvancedToolsExpandedChange={vi.fn()}
          allowExternalImages={false}
          onAllowExternalImagesChange={vi.fn()}
          showMoodBadge={false}
          onShowMoodBadgeChange={vi.fn()}
          showMoodConfidence={false}
          onShowMoodConfidenceChange={vi.fn()}
          onOpenRawRequest={vi.fn()}
          voiceChatAvailable
          voiceChatEnabled={false}
          voiceChatState="idle"
          voiceChatStatusLabel="Voice chat"
          onVoiceChatToggle={vi.fn()}
          isSending={false}
          voiceChatSettingsFields={null}
          imageProviderControl={null}
          historyLength={0}
          onClearContext={vi.fn()}
          t={t}
        />
      </>
    )

    expectTokens(
      screen.getByRole("button", { name: "Modes" }),
      COMPOSER_TOOLBAR_TOKENS.textControl
    )
    expectTokens(
      screen.getByTestId("tools-button"),
      COMPOSER_TOOLBAR_TOKENS.iconButton
    )
  })

  it("sizes both halves of the attach control with the icon token", () => {
    render(
      <PlaygroundAttachmentButton
        isProMode={false}
        chatMode="normal"
        onImageUpload={vi.fn()}
        onDocumentUpload={vi.fn()}
        onOpenKnowledgePanel={vi.fn()}
        attachmentMenuOpen={false}
        onAttachmentMenuChange={vi.fn()}
        t={t}
      />
    )

    expectTokens(
      screen.getByTestId("attachment-button"),
      COMPOSER_TOOLBAR_TOKENS.iconButton
    )
    expectTokens(
      screen.getByRole("button", { name: "More attachments" }),
      COMPOSER_TOOLBAR_TOKENS.iconButton
    )
  })

  it("sizes the attach control with the text token in Pro mode", () => {
    render(
      <PlaygroundAttachmentButton
        isProMode
        chatMode="normal"
        onImageUpload={vi.fn()}
        onDocumentUpload={vi.fn()}
        onOpenKnowledgePanel={vi.fn()}
        attachmentMenuOpen={false}
        onAttachmentMenuChange={vi.fn()}
        t={t}
      />
    )

    expectTokens(
      screen.getByTestId("attachment-button"),
      COMPOSER_TOOLBAR_TOKENS.textControl
    )
  })
})
