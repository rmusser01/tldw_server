import { RadialCommandV5 } from "@/components/Chat/composer/variants/RadialCommandV5"
import { SplitBriefV3 } from "@/components/Chat/composer/variants/SplitBriefV3"
import { TerminalStackV1 } from "@/components/Chat/composer/variants/TerminalStackV1"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { SidepanelComposerControlArea } from "../SidepanelComposerControlArea"

const commonProps = {
  message: "",
  onMessageChange: vi.fn(),
  onSend: vi.fn()
}

const casualControlArea = (
  <div>
    <button type="button">Secondary action</button>
    <SidepanelComposerControlArea
      promptAssistAction={
        <button type="button" aria-label="Improve prompt">
          Improve
        </button>
      }>
      <button type="button" aria-label="Send message">
        Send
      </button>
    </SidepanelComposerControlArea>
  </div>
)

describe("SidepanelComposerControlArea prompt-assist parity", () => {
  it.each(["legacy", "v1", "v3", "v5"] as const)(
    "renders one Improve action immediately before Send through the %s shared control slot",
    (variant) => {
      if (variant === "legacy") {
        render(casualControlArea)
      } else if (variant === "v1") {
        render(
          <TerminalStackV1 {...commonProps} bottomBarSlot={casualControlArea} />
        )
      } else if (variant === "v3") {
        render(
          <SplitBriefV3
            {...commonProps}
            briefSections={[]}
            bottomBarSlot={casualControlArea}
          />
        )
      } else {
        render(
          <RadialCommandV5 {...commonProps} facetsSlot={casualControlArea} />
        )
      }

      expect(
        screen.getAllByRole("button", { name: "Improve prompt" })
      ).toHaveLength(1)
      const sendCluster = screen.getByTestId("sidepanel-send-action-cluster")
      const clusterButtons = within(sendCluster).getAllByRole("button")
      expect(
        clusterButtons.map((button) => button.getAttribute("aria-label"))
      ).toEqual(["Improve prompt", "Send message"])
      expect(screen.getByRole("button", { name: "Secondary action" })).not.toBe(
        sendCluster
      )
    }
  )
})
