import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Form } from "antd"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

// Mock the heavy EntryManager so we don't pull in its dependency tree
vi.mock("../WorldBookEntryManager", () => ({
  WorldBookEntryManager: (props: any) => (
    <div data-testid="mock-entry-manager">
      EntryManager for book {props.worldBookId}
    </div>
  ),
  DEFAULT_ENTRY_FILTER_PRESET: {
    enabledFilter: "all",
    matchFilter: "all",
    searchText: ""
  }
}))

// Mock WorldBookForm to avoid its internal complexity
vi.mock("../WorldBookForm", () => ({
  WorldBookForm: (props: any) => (
    <div data-testid="mock-world-book-form">
      WorldBookForm mode={props.mode}
    </div>
  )
}))

import { WorldBookDetailPanel } from "../WorldBookDetailPanel"

const mockWorldBook = {
  id: 1,
  name: "Fantasy Lore",
  description: "Magic systems and creatures",
  enabled: true,
  entry_count: 42,
  token_budget: 700,
  last_modified: Date.now() - 3600_000,
  scan_depth: 2,
  recursive_scanning: false
}

const mockAttachedCharacters = [
  { id: 10, name: "Gandalf" },
  { id: 20, name: "Frodo" }
]

const mockAllCharacters = [
  { id: 10, name: "Gandalf" },
  { id: 20, name: "Frodo" },
  { id: 30, name: "Aragorn" }
]

// Wrapper that provides a form instance
const TestWrapper: React.FC<{
  worldBook?: any
  attachedCharacters?: any[]
  allWorldBooks?: any[]
  allCharacters?: any[]
  activeTab?: "entries" | "attachments" | "stats" | "settings"
  statsData?: any | null
}> = ({
  worldBook = mockWorldBook,
  attachedCharacters = mockAttachedCharacters,
  allWorldBooks = [mockWorldBook],
  allCharacters = mockAllCharacters,
  activeTab = "entries",
  statsData = null
}) => {
  const [entryForm] = Form.useForm()
  const [settingsForm] = Form.useForm()
  const [currentTab, setCurrentTab] = React.useState(activeTab)
  return (
    <WorldBookDetailPanel
      worldBook={worldBook}
      attachedCharacters={attachedCharacters}
      allWorldBooks={allWorldBooks}
      allCharacters={allCharacters}
      activeTab={currentTab}
      onActiveTabChange={setCurrentTab}
      onUpdateWorldBook={vi.fn()}
      onAttachCharacter={vi.fn().mockResolvedValue(undefined)}
      onDetachCharacter={vi.fn().mockResolvedValue(undefined)}
      onOpenTestMatching={vi.fn()}
      maxRecursiveDepth={10}
      updating={false}
      entryFormInstance={entryForm}
      settingsFormInstance={settingsForm}
      statsData={statsData}
      statsLoading={false}
      statsError={null}
    />
  )
}

describe("WorldBookDetailPanel", () => {
  it("renders the summary bar with key metadata (name, entries, enabled, characters)", () => {
    render(<TestWrapper />)

    expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
      "Fantasy Lore"
    )
    expect(screen.getByText(/42/)).toBeInTheDocument()
    expect(screen.getByText(/enabled/i)).toBeInTheDocument()
    expect(screen.getByText(/2 characters/i)).toBeInTheDocument()
  })

  it("renders last-modified metadata for ISO timestamps in the summary bar", () => {
    vi.useFakeTimers()
    try {
      vi.setSystemTime(new Date("2026-02-18T12:00:00Z"))

      render(
        <TestWrapper
          worldBook={{
            ...mockWorldBook,
            last_modified: "2026-02-18T09:00:00Z"
          }}
        />
      )

      expect(screen.getByText("3 hours ago")).toBeInTheDocument()
      expect(
        screen.getByTitle("2026-02-18 09:00:00 UTC")
      ).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("renders Entries tab as default active tab", () => {
    render(<TestWrapper />)

    const entriesTab = screen.getByRole("tab", { name: /entries/i })
    expect(entriesTab).toHaveAttribute("aria-selected", "true")
  })

  it("renders all four tabs (entries, attachments, stats, settings)", () => {
    render(<TestWrapper />)

    expect(screen.getByRole("tab", { name: /entries/i })).toBeInTheDocument()
    expect(
      screen.getByRole("tab", { name: /attachments/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /stats/i })).toBeInTheDocument()
    expect(
      screen.getByRole("tab", { name: /settings/i })
    ).toBeInTheDocument()
  })

  it("switches to Settings tab on click", async () => {
    const user = userEvent.setup()
    render(<TestWrapper />)

    const settingsTab = screen.getByRole("tab", { name: /settings/i })
    await user.click(settingsTab)

    await waitFor(() => {
      expect(settingsTab).toHaveAttribute("aria-selected", "true")
    })

    const entriesTab = screen.getByRole("tab", { name: /entries/i })
    expect(entriesTab).toHaveAttribute("aria-selected", "false")
  })

  it("renders attachment links in the attachments tab", () => {
    render(<TestWrapper activeTab="attachments" />)

    const characterLink = screen.getByRole("link", { name: "Open character Gandalf" })
    expect(characterLink).toHaveAttribute(
      "href",
      "/characters?from=world-books&focusCharacterId=10&focusWorldBookId=1"
    )
  })

  it("renders live stats content instead of the loading placeholder when stats data is provided", () => {
    render(
      <TestWrapper
        activeTab="stats"
        statsData={{
          total_entries: 42,
          enabled_entries: 40,
          disabled_entries: 2,
          estimated_tokens: 320,
          token_estimation_method: "cl100k_base"
        }}
      />
    )

    expect(screen.getByTestId("stats-tab-content")).toHaveTextContent("42")
    expect(screen.getByTestId("stats-tab-content")).toHaveTextContent("320")
    expect(screen.getByText("Estimated using cl100k_base.")).toBeInTheDocument()
  })

  it("has correct landmark role (main with aria-label)", () => {
    render(<TestWrapper />)

    const main = screen.getByRole("main")
    expect(main).toHaveAttribute("aria-label", "World book detail")
  })

  it("renders placeholder when worldBook is null", () => {
    render(<TestWrapper worldBook={null} />)

    expect(
      screen.getByText(/select a world book/i)
    ).toBeInTheDocument()

    const main = screen.getByRole("main")
    expect(main).toHaveAttribute("aria-label", "World book detail")
  })
})
