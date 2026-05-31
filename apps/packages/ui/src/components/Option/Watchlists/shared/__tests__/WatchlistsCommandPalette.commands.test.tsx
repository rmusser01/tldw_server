import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  WatchlistsCommandPalette,
  useWatchlistsCommands,
  type CommandPaletteCommand
} from "../WatchlistsCommandPalette"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown) =>
      typeof defaultValue === "string" ? defaultValue : _key
  })
}))

vi.mock("antd", () => {
  const Input = ({ value, onChange, placeholder }: any) => (
    <input
      aria-label={placeholder}
      value={value || ""}
      onChange={(event) => onChange?.(event)}
    />
  )
  const Modal = ({ open, children }: any) => (open ? <div>{children}</div> : null)
  return { Input, Modal }
})

const HookProbe = ({ onCommands }: { onCommands: (commands: CommandPaletteCommand[]) => void }) => {
  const commands = useWatchlistsCommands({
    setActiveTab: vi.fn(),
    openSourceForm: vi.fn(),
    openJobForm: vi.fn(),
    openSettings: vi.fn(),
    refreshCurrentView: vi.fn(),
    startGuidedTour: vi.fn(),
    createPipeline: vi.fn(),
    exportSources: vi.fn(),
    exportRuns: vi.fn()
  })
  React.useEffect(() => {
    onCommands(commands)
  }, [commands, onCommands])
  return null
}

describe("WatchlistsCommandPalette command coverage", () => {
  it("exposes create, clone, validate, run, preview, retry, and export commands", async () => {
    let commands: CommandPaletteCommand[] = []
    render(<HookProbe onCommands={(value) => { commands = value }} />)

    const commandIds = commands.map((command) => command.id)
    expect(commandIds).toEqual(
      expect.arrayContaining([
        "create-pipeline",
        "create-feed",
        "create-monitor",
        "action-clone-feed",
        "action-clone-monitor",
        "action-validate-feeds",
        "action-run-monitor",
        "action-preview-monitor",
        "action-retry-run",
        "action-export-sources",
        "action-export-runs"
      ])
    )
    expect(commands.find((command) => command.id === "action-clone-feed")?.disabledReason).toBeTruthy()
    expect(commands.find((command) => command.id === "action-export-runs")?.disabledReason).toBeUndefined()
  })

  it("renders unavailable command reasons inline instead of silently executing impossible actions", () => {
    const onExecute = vi.fn()
    render(
      <WatchlistsCommandPalette
        open
        onClose={vi.fn()}
        commands={[
          {
            id: "action-clone-feed",
            label: "Clone selected feed",
            icon: <span />,
            category: "action",
            disabledReason: "Choose a feed row to clone.",
            onExecute
          }
        ]}
      />
    )

    expect(screen.getByText("Clone selected feed")).toBeInTheDocument()
    expect(screen.getByText("Choose a feed row to clone.")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-command-action-clone-feed"))
    expect(onExecute).not.toHaveBeenCalled()
  })
})
