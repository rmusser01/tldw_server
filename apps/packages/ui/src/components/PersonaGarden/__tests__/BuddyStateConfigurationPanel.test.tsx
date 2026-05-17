import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  asPersonaVisualCustomStateId,
  type PersonaVisualManifest
} from "@/types/persona-visuals"

import { BuddyStateConfigurationPanel } from "../BuddyStateConfigurationPanel"
import { BUDDY_CORE_STATE_ORDER } from "../buddyBuilderState"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string; state?: string }) =>
      (options?.defaultValue ?? _key).replace(
        "{{state}}",
        String(options?.state ?? "")
      )
  })
}))

const customState = asPersonaVisualCustomStateId("search_result")
const movingLeft = asPersonaVisualCustomStateId("moving_left")
const movingRight = asPersonaVisualCustomStateId("moving_right")

const manifest: PersonaVisualManifest = {
  manifest_version: 1,
  renderer_type: "sprite_frames",
  states: {
    idle: { animation_id: "idle-loop" },
    listening: { animation_id: "listen-loop" },
    thinking: { animation_id: "think-loop" },
    speaking: { animation_id: "speak-loop" },
    error: { animation_id: "error-loop" },
    tool_running: { animation_id: "tool-loop" },
    [movingLeft]: { animation_id: "move-left-loop" },
    [movingRight]: { animation_id: "move-right-loop" },
    [customState]: { animation_id: "search-result-loop" }
  },
  animations: {
    "idle-loop": { frames: [] },
    "listen-loop": { frames: [] },
    "think-loop": { frames: [] },
    "speak-loop": { frames: [] },
    "error-loop": { frames: [] },
    "tool-loop": { frames: [] },
    "move-left-loop": { frames: [] },
    "move-right-loop": { frames: [] },
    "search-result-loop": { frames: [] }
  },
  fallbacks: {
    [customState]: ["thinking", "idle"],
    [movingLeft]: ["idle"]
  },
  state_catalog: {
    [movingLeft]: {
      label: "Moving left",
      kind: "live_variant",
      description: "Buddy is being moved left.",
      tags: ["drag", "left"]
    },
    [movingRight]: {
      label: "Moving right",
      kind: "live_variant",
      description: "Buddy is being moved right.",
      tags: ["drag", "right"]
    },
    [customState]: {
      label: "Search result found",
      kind: "tool_variant",
      description: "Shown when a search tool returns useful results.",
      tags: ["tool", "search"]
    }
  },
  authored_triggers: [
    {
      id: "trigger-tool-name",
      source: "tool_name",
      match: "web_search",
      state: customState,
      duration_ms: 1600,
      priority: 90
    },
    {
      id: "trigger-tool-category",
      source: "tool_category",
      match: "retrieval",
      state: "thinking",
      duration_ms: 900,
      priority: 40
    }
  ]
}

const stateIds = (testId: string) =>
  within(screen.getByTestId(testId))
    .getAllByTestId("buddy-state-config-state-row")
    .map((row) => row.getAttribute("data-state-id"))

describe("BuddyStateConfigurationPanel", () => {
  it("renders core states in the documented order", () => {
    render(<BuddyStateConfigurationPanel manifest={manifest} />)

    expect(stateIds("buddy-state-config-core-states")).toEqual(
      BUDDY_CORE_STATE_ORDER
    )
  })

  it("keeps movement states separate from task-running core states", () => {
    render(<BuddyStateConfigurationPanel manifest={manifest} />)

    expect(stateIds("buddy-state-config-movement-states")).toEqual([
      "moving_left",
      "moving_right"
    ])
    expect(screen.getByTestId("buddy-state-config-core-states")).toHaveTextContent(
      "Tool running"
    )
    expect(screen.getByTestId("buddy-state-config-core-states")).not.toHaveTextContent(
      "Moving left"
    )
    expect(
      screen.getByTestId("buddy-state-config-custom-states")
    ).not.toHaveTextContent("Moving right")
  })

  it("renders custom state metadata and fallbacks from the manifest", () => {
    render(<BuddyStateConfigurationPanel manifest={manifest} />)

    const customStates = screen.getByTestId("buddy-state-config-custom-states")
    expect(customStates).toHaveTextContent("Search result found")
    expect(customStates).toHaveTextContent("tool_variant")
    expect(customStates).toHaveTextContent(
      "Shown when a search tool returns useful results."
    )
    expect(customStates).toHaveTextContent("tool")
    expect(customStates).toHaveTextContent("search")
    expect(customStates).toHaveTextContent("thinking, idle")
  })

  it("renders exact tool-name triggers separately from tool-category triggers", () => {
    render(<BuddyStateConfigurationPanel manifest={manifest} />)

    const exactToolTriggers = screen.getByTestId(
      "buddy-state-config-tool-name-triggers"
    )
    const categoryTriggers = screen.getByTestId(
      "buddy-state-config-tool-category-triggers"
    )

    expect(exactToolTriggers).toHaveTextContent("web_search")
    expect(exactToolTriggers).toHaveTextContent("Search result found")
    expect(exactToolTriggers).not.toHaveTextContent("retrieval")
    expect(categoryTriggers).toHaveTextContent("retrieval")
    expect(categoryTriggers).toHaveTextContent("Thinking")
    expect(categoryTriggers).not.toHaveTextContent("web_search")
  })

  it("uses accessible controls and delegates saving to the existing callback", () => {
    const onSaveManifest = vi.fn()
    render(
      <BuddyStateConfigurationPanel
        manifest={manifest}
        canSave
        onSaveManifest={onSaveManifest}
      />
    )

    expect(screen.getByLabelText("Idle animation")).toBeDisabled()
    expect(screen.getByLabelText("Moving right animation")).toBeDisabled()

    fireEvent.click(
      screen.getByRole("button", { name: "Save visual state configuration" })
    )

    expect(onSaveManifest).toHaveBeenCalledTimes(1)
  })
})
