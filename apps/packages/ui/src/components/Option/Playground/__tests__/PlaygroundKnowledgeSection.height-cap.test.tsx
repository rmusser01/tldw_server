// @vitest-environment jsdom
// CC-10 (#3101), owner decision [2]: Search & Context stays in the composer
// with its own height cap and internal scroll, so the textarea and Send stay
// in view. (The layout itself is measured in e2e/smoke/composer-calm.spec.ts.)
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { PlaygroundKnowledgeSection } from "../PlaygroundKnowledgeSection"

vi.mock("@/components/Knowledge", () => ({
  KnowledgePanel: () => <div data-testid="knowledge-panel-stub" />
}))

const props = (
  contextToolsOpen: boolean
): React.ComponentProps<typeof PlaygroundKnowledgeSection> => ({
  contextToolsOpen,
  isConnectionReady: true,
  knowledgePanelTab: "qa-search",
  knowledgePanelTabRequestId: 0,
  deferredComposerInput: "",
  attachedImage: "",
  attachedTabs: [],
  availableTabs: [],
  attachedFiles: [],
  fileRetrievalEnabled: false,
  onInsert: vi.fn(),
  onAsk: vi.fn(),
  onOpenChange: vi.fn(),
  onRemoveImage: vi.fn(),
  onRemoveTab: vi.fn(),
  onAddTab: vi.fn(),
  onClearTabs: vi.fn(),
  onRefreshTabs: vi.fn(),
  onAddFile: vi.fn(),
  onRemoveFile: vi.fn(),
  onClearFiles: vi.fn(),
  onFileRetrievalChange: vi.fn(),
  wrapComposerProfile: (_id, element) => element
})

describe("PlaygroundKnowledgeSection height cap", () => {
  it("scrolls inside its own capped box so the composer keeps its input and Send", () => {
    render(<PlaygroundKnowledgeSection {...props(true)} />)

    const box = screen.getByTestId("composer-knowledge-section")
    expect(box).toContainElement(screen.getByTestId("knowledge-panel-stub"))
    expect(box.className).toMatch(/(^|\s)max-h-\[[^\]]+\]/)
    expect(box.className).toContain("overflow-y-auto")
    expect(box.className).toContain("overscroll-contain")
  })

  it("stays out of the layout while closed", () => {
    render(<PlaygroundKnowledgeSection {...props(false)} />)

    expect(
      screen.getByTestId("composer-knowledge-section").closest("[aria-hidden]")
    ).toHaveAttribute("aria-hidden", "true")
  })
})
