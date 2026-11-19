// @vitest-environment jsdom
// CC-08 (#3101), owner decision [1]: one name and one home per feature. The
// redesign's names stay ("Search & Context", "Runtime"); web search has one
// toggle (the Tools tray) and one spelling; Modes holds conversation modes.
import React from "react"
import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundAttachmentButton } from "../PlaygroundSendControl"
import { PlaygroundContextRail } from "../PlaygroundContextRail"
import { PlaygroundModeLauncher } from "../PlaygroundModeLauncher"
import { PlaygroundToolsPopover } from "../PlaygroundToolsPopover"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key
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
    <div>
      {children}
      {content}
    </div>
  ),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Switch: ({
    checked,
    onChange,
    "aria-label": ariaLabel
  }: {
    checked?: boolean
    onChange?: (value: boolean) => void
    "aria-label"?: string
  }) => (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={ariaLabel}
      onClick={() => onChange?.(!checked)}
    />
  ),
  Radio: () => null
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  )
}))

const t = (key: string, fallback?: string) => fallback ?? key

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")
const readLocale = (file: string) =>
  JSON.parse(
    readFileSync(path.resolve(srcRoot, "assets/locale/en", file), "utf8")
  )

const modeLauncherProps = (
  compareFeatureEnabled = false
): React.ComponentProps<typeof PlaygroundModeLauncher> => ({
  open: true,
  onOpenChange: vi.fn(),
  compareModeActive: false,
  compareFeatureEnabled,
  onToggleCompare: vi.fn(),
  selectedCharacterName: null,
  onOpenActorSettings: vi.fn(),
  onModeAnnouncement: vi.fn(),
  t
})

const toolsPopoverProps = (): React.ComponentProps<
  typeof PlaygroundToolsPopover
> => ({
  toolsPopoverOpen: true,
  onToolsPopoverChange: vi.fn(),
  isProMode: false,
  onOpenImageGenerate: vi.fn(),
  useOCR: false,
  onUseOCRChange: vi.fn(),
  hasWebSearch: true,
  simpleInternetSearch: false,
  onSimpleInternetSearchChange: vi.fn(),
  defaultInternetSearchOn: false,
  onDefaultInternetSearchOnChange: vi.fn(),
  onNavigateWebSearchSettings: vi.fn(),
  advancedToolsExpanded: false,
  onAdvancedToolsExpandedChange: vi.fn(),
  allowExternalImages: false,
  onAllowExternalImagesChange: vi.fn(),
  showMoodBadge: false,
  onShowMoodBadgeChange: vi.fn(),
  showMoodConfidence: false,
  onShowMoodConfidenceChange: vi.fn(),
  onOpenRawRequest: vi.fn(),
  voiceChatAvailable: true,
  voiceChatEnabled: false,
  voiceChatState: "idle",
  voiceChatStatusLabel: "Voice chat",
  onVoiceChatToggle: vi.fn(),
  isSending: false,
  voiceChatSettingsFields: null,
  imageProviderControl: null,
  historyLength: 0,
  onClearContext: vi.fn(),
  t
})

const contextRailProps = (): React.ComponentProps<
  typeof PlaygroundContextRail
> => ({
  hasContext: false,
  contextSummary: [],
  sessionLabel: "Local chat",
  historyLinked: false,
  temporaryChat: false,
  onToggleTemporaryChat: vi.fn(),
  contextCounts: { files: 0, knowledge: 0, media: 0, research: 0 },
  onOpenSearchContext: vi.fn()
})

describe("CC-08: one name and one home per composer feature", () => {
  it("keeps Modes to conversation modes; panels and toggles live in their own homes", () => {
    render(<PlaygroundModeLauncher {...modeLauncherProps(true)} />)

    expect(
      screen.getByRole("button", { name: /Character \/ Scene/ })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Compare responses/ })
    ).toBeInTheDocument()
    expect(screen.queryByText("Knowledge panel")).not.toBeInTheDocument()
    expect(screen.queryByText("Voice mode")).not.toBeInTheDocument()
    expect(screen.queryByText(/web search/i)).not.toBeInTheDocument()
  })

  it("leaves web search's on/off switch to the Tools tray and keeps its settings in More tools", () => {
    render(<PlaygroundToolsPopover {...toolsPopoverProps()} />)

    expect(screen.queryByText(/Enable web search/i)).not.toBeInTheDocument()
    expect(screen.getByText("Web search")).toBeInTheDocument()
    expect(screen.queryByText("Web Search")).not.toBeInTheDocument()
    expect(screen.getByText("Simple search mode")).toBeInTheDocument()
    expect(screen.getByText("Default on for new chats")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Open web search settings/ })
    ).toBeInTheDocument()
    // Search & Context has its own homes (Tools tray, attach menu, rail).
    expect(screen.queryByText(/Knowledge Panel/i)).not.toBeInTheDocument()
    expect(screen.getByText("Images")).toBeInTheDocument()
    expect(screen.queryByText("Attachments")).not.toBeInTheDocument()
  })

  it("shows web search in the Context rail as a source, not as a second toggle", () => {
    const props = contextRailProps()
    render(<PlaygroundContextRail {...props} />)

    expect(
      screen.queryByRole("button", { name: /web search/i })
    ).not.toBeInTheDocument()
    const addSources = screen.getByRole("button", {
      name: "Add sources from Search & Context"
    })
    expect(addSources).toHaveTextContent("Add sources")
    fireEvent.click(addSources)
    expect(props.onOpenSearchContext).toHaveBeenCalledTimes(1)
  })

  it("names Search & Context in the attach menu", () => {
    render(
      <PlaygroundAttachmentButton
        isProMode={false}
        isMobileViewport={false}
        chatMode="rag"
        onImageUpload={vi.fn()}
        onDocumentUpload={vi.fn()}
        onOpenKnowledgePanel={vi.fn()}
        attachmentMenuOpen
        onAttachmentMenuChange={vi.fn()}
        t={t}
      />
    )

    expect(
      screen.getByRole("button", { name: "Manage in Search & Context" })
    ).toBeInTheDocument()
    expect(screen.queryByText(/Knowledge Panel/i)).not.toBeInTheDocument()
  })

  it("uses one name for the knowledge feature and one spelling for web search in the English copy", () => {
    const playground = readLocale("playground.json")
    const sidepanel = readLocale("sidepanel.json")
    const common = readLocale("common.json")

    const knowledgeCopy = [
      playground.attachments.manageContext,
      playground.attachments.imageDisabledBody,
      playground.composer.manageContextHint,
      sidepanel.knowledge.title,
      sidepanel.rag.show
    ]
    for (const copy of knowledgeCopy) {
      expect(copy).toContain("Search & Context")
      expect(copy).not.toMatch(/Knowledge (Panel|Search)/i)
    }
    expect(common.commandPalette.toggleWebSearch).toBe("Toggle web search")
    expect(common.commandPalette.toggleKnowledgeSearch).toContain(
      "Search & Context"
    )
  })

  it("keeps the composer's own fallbacks free of the old names", () => {
    const files = [
      "components/Option/Playground/PlaygroundModeLauncher.tsx",
      "components/Option/Playground/PlaygroundToolsPopover.tsx",
      "components/Option/Playground/PlaygroundSendControl.tsx",
      "components/Option/Playground/PlaygroundContextRail.tsx",
      "components/Option/Playground/AttachmentsSummary.tsx",
      "components/Knowledge/KnowledgePanel.tsx"
    ]
    for (const file of files) {
      const source = readFileSync(path.resolve(srcRoot, file), "utf8")
      expect(source, file).not.toMatch(/"Knowledge (Panel|panel|Search)"/)
      expect(source, file).not.toMatch(/Manage in Knowledge Panel/)
      expect(source, file).not.toMatch(/"Web Search"/)
    }
  })

  it("keeps Character / Scene in Modes and announces it", () => {
    const props = modeLauncherProps(false)
    render(<PlaygroundModeLauncher {...props} />)

    const entries = screen
      .getAllByRole("button")
      .filter((button) => button.getAttribute("aria-label") !== "Modes")
    expect(entries).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: /Character \/ Scene/ }))
    expect(props.onOpenActorSettings).toHaveBeenCalledTimes(1)
  })
})
