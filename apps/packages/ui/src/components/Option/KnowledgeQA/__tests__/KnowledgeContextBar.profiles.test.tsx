import { fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeContextBar } from "../context/KnowledgeContextBar"

const PROFILES_STORAGE_KEY = "tldw:knowledge-qa:saved-profiles"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getProviders: vi.fn().mockResolvedValue({
      default_provider: "openai",
      providers: [
        { name: "openai", display_name: "OpenAI", models: ["gpt-4o-mini"] },
      ],
    }),
    listMedia: vi.fn().mockResolvedValue({ items: [] }),
    listNotes: vi.fn().mockResolvedValue({ items: [] }),
  },
}))

function renderContextBar(overrides: Record<string, unknown> = {}) {
  const defaults = {
    preset: "balanced" as const,
    onPresetChange: vi.fn(),
    sources: ["media_db" as const],
    onSourcesChange: vi.fn(),
    includeMediaIds: [] as number[],
    onIncludeMediaIdsChange: vi.fn(),
    includeNoteIds: [] as string[],
    onIncludeNoteIdsChange: vi.fn(),
    webEnabled: true,
    onToggleWeb: vi.fn(),
    generationProvider: null,
    generationModel: null,
    onGenerationProviderChange: vi.fn(),
    onGenerationModelChange: vi.fn(),
    contextChangedSinceLastRun: false,
    onOpenSettings: vi.fn(),
  }

  const props = { ...defaults, ...overrides }
  return { ...render(<KnowledgeContextBar {...props} />), props }
}

function openProfileMenu() {
  const profilesButton = screen.getByRole("button", { name: /Profiles/i })
  fireEvent.click(profilesButton)
}

function enterSaveMode() {
  fireEvent.click(screen.getByText("Save current settings..."))
}

function typeProfileName(name: string) {
  const nameInput = screen.getByPlaceholderText("Profile name")
  fireEvent.change(nameInput, { target: { value: name } })
}

function clickSaveButton() {
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
}

describe("KnowledgeContextBar saved search profiles", () => {
  beforeEach(() => {
    localStorage.removeItem(PROFILES_STORAGE_KEY)
  })

  afterEach(() => {
    localStorage.removeItem(PROFILES_STORAGE_KEY)
  })

  // -----------------------------------------------------------------------
  // Test 1: Profile save
  // -----------------------------------------------------------------------
  describe("profile save", () => {
    it("saves current settings to localStorage when a name is provided", () => {
      renderContextBar({
        preset: "fast",
        sources: ["media_db", "notes"],
        includeMediaIds: [42],
        includeNoteIds: ["note-1"],
        webEnabled: true,
        generationProvider: "openai",
        generationModel: "gpt-4o-mini",
      })

      openProfileMenu()
      expect(screen.getByText("No saved profiles yet.")).toBeInTheDocument()

      enterSaveMode()
      typeProfileName("My Research Setup")
      clickSaveButton()

      const stored = JSON.parse(localStorage.getItem(PROFILES_STORAGE_KEY)!)
      expect(stored).toHaveLength(1)
      expect(stored[0]).toEqual({
        name: "My Research Setup",
        sources: ["media_db", "notes"],
        includeMediaIds: [42],
        includeNoteIds: ["note-1"],
        preset: "fast",
        enableWebFallback: true,
        generationProvider: "openai",
        generationModel: "gpt-4o-mini",
      })
    })

    it("replaces a profile with the same name instead of duplicating", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Existing",
            sources: ["media_db"],
            preset: "balanced",
            enableWebFallback: false,
          },
        ])
      )

      renderContextBar({
        preset: "thorough",
        sources: ["notes"],
        webEnabled: true,
      })

      openProfileMenu()
      enterSaveMode()
      typeProfileName("Existing")
      clickSaveButton()

      const stored = JSON.parse(localStorage.getItem(PROFILES_STORAGE_KEY)!)
      expect(stored).toHaveLength(1)
      expect(stored[0].preset).toBe("thorough")
      expect(stored[0].sources).toEqual(["notes"])
      expect(stored[0].enableWebFallback).toBe(true)
    })
  })

  // -----------------------------------------------------------------------
  // Test 2: Profile load
  // -----------------------------------------------------------------------
  describe("profile load", () => {
    it("applies saved profile settings when clicked", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Deep Research",
            sources: ["media_db", "notes", "chats"],
            preset: "thorough",
            enableWebFallback: false,
          },
        ])
      )

      const { props } = renderContextBar({
        preset: "fast",
        sources: ["media_db"],
        includeMediaIds: [],
        includeNoteIds: [],
        webEnabled: true,
      })

      openProfileMenu()
      fireEvent.click(screen.getByRole("menuitem", { name: /Deep Research/i }))

      expect(props.onSourcesChange).toHaveBeenCalledWith([
        "media_db",
        "notes",
        "chats",
      ])
      expect(props.onIncludeMediaIdsChange).toHaveBeenCalledWith([])
      expect(props.onIncludeNoteIdsChange).toHaveBeenCalledWith([])
      expect(props.onPresetChange).toHaveBeenCalledWith("thorough")
      // webEnabled is true, profile says false -> onToggleWeb should be called
      expect(props.onToggleWeb).toHaveBeenCalledTimes(1)
    })

    it("does not toggle web when profile matches current state", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Quick Check",
            sources: ["media_db"],
            preset: "fast",
            enableWebFallback: true,
          },
        ])
      )

      const { props } = renderContextBar({
        preset: "balanced",
        sources: ["notes"],
        webEnabled: true,
      })

      openProfileMenu()
      fireEvent.click(screen.getByRole("menuitem", { name: /Quick Check/i }))

      expect(props.onToggleWeb).not.toHaveBeenCalled()
    })

    it("restores saved specific source sets when a profile includes them", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Pinned Evidence",
            sources: ["media_db", "notes"],
            includeMediaIds: [42, 43],
            includeNoteIds: ["note-1"],
            preset: "balanced",
            enableWebFallback: true,
            generationProvider: "anthropic",
            generationModel: "claude-3-5-sonnet",
          },
        ])
      )

      const { props } = renderContextBar({
        sources: [],
        includeMediaIds: [],
        includeNoteIds: [],
        webEnabled: true,
      })

      openProfileMenu()
      fireEvent.click(screen.getByRole("menuitem", { name: /Pinned Evidence/i }))

      expect(props.onSourcesChange).toHaveBeenCalledWith(["media_db", "notes"])
      expect(props.onIncludeMediaIdsChange).toHaveBeenCalledWith([42, 43])
      expect(props.onIncludeNoteIdsChange).toHaveBeenCalledWith(["note-1"])
      expect(props.onGenerationProviderChange).toHaveBeenCalledWith("anthropic")
      expect(props.onGenerationModelChange).toHaveBeenCalledWith(
        "claude-3-5-sonnet"
      )
    })
  })

  // -----------------------------------------------------------------------
  // Test 3: Profile delete
  // -----------------------------------------------------------------------
  describe("profile delete", () => {
    it("removes the profile from localStorage when delete is clicked", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Profile A",
            sources: ["media_db"],
            preset: "fast",
            enableWebFallback: true,
          },
          {
            name: "Profile B",
            sources: ["notes"],
            preset: "balanced",
            enableWebFallback: false,
          },
        ])
      )

      renderContextBar()

      openProfileMenu()
      const deleteButton = screen.getByRole("button", {
        name: "Delete profile Profile A",
      })
      fireEvent.click(deleteButton)

      const stored = JSON.parse(localStorage.getItem(PROFILES_STORAGE_KEY)!)
      expect(stored).toHaveLength(1)
      expect(stored[0].name).toBe("Profile B")
    })

    it("keeps the delete action discoverable for keyboard users", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Profile A",
            sources: ["media_db"],
            preset: "fast",
            enableWebFallback: true,
          },
        ])
      )

      renderContextBar()

      openProfileMenu()
      const deleteButton = screen.getByRole("button", {
        name: "Delete profile Profile A",
      })

      expect(deleteButton.className).toContain("group-focus-within:visible")
    })
  })

  // -----------------------------------------------------------------------
  // Test 4: Max limit
  // -----------------------------------------------------------------------
  describe("max profiles limit", () => {
    it("disables save button and shows limit message when 12 profiles exist", () => {
      const profiles = Array.from({ length: 12 }, (_, index) => ({
        name: `Profile ${index + 1}`,
        sources: ["media_db"] as string[],
        preset: "fast",
        enableWebFallback: true,
      }))
      localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles))

      renderContextBar()

      openProfileMenu()

      const limitButton = screen.getByText("Limit reached (12)")
      expect(limitButton).toBeInTheDocument()
      expect(limitButton.closest("button")).toBeDisabled()
    })
  })

  // -----------------------------------------------------------------------
  // Test 5: Corrupt localStorage
  // -----------------------------------------------------------------------
  describe("corrupt localStorage", () => {
    it("renders without saved profiles when localStorage data is corrupt", () => {
      localStorage.setItem(PROFILES_STORAGE_KEY, "{{not valid json")

      renderContextBar()

      openProfileMenu()
      expect(screen.getByText("No saved profiles yet.")).toBeInTheDocument()
    })

    it("filters out invalid entries from localStorage", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          { name: "Valid", sources: ["media_db"], preset: "fast", enableWebFallback: true },
          { name: "", sources: ["media_db"], preset: "fast", enableWebFallback: true },
          { name: "Bad preset", sources: ["media_db"], preset: "bogus", enableWebFallback: true },
          { name: "Bad source", sources: ["bogus"], preset: "fast", enableWebFallback: true },
          {
            name: "Bad specific IDs",
            sources: ["media_db"],
            includeMediaIds: "42",
            includeNoteIds: "note-1",
            preset: "fast",
            enableWebFallback: true,
          },
          {
            name: "Bad non-positive IDs",
            sources: ["media_db"],
            includeMediaIds: [42, 0, -3, 1.5],
            includeNoteIds: ["note-1"],
            preset: "fast",
            enableWebFallback: true,
          },
          { name: 123, sources: "not-an-array", preset: "fast", enableWebFallback: true },
          null,
          "just a string",
        ])
      )

      renderContextBar()

      openProfileMenu()
      expect(screen.getByRole("menuitem", { name: /Valid/i })).toBeInTheDocument()
      expect(screen.queryByRole("menuitem", { name: /Bad preset/i })).not.toBeInTheDocument()
      expect(screen.queryByRole("menuitem", { name: /Bad source/i })).not.toBeInTheDocument()
      expect(screen.queryByRole("menuitem", { name: /Bad specific IDs/i })).not.toBeInTheDocument()
      expect(screen.queryByRole("menuitem", { name: /Bad non-positive IDs/i })).not.toBeInTheDocument()
      // The invalid entries should not appear
      expect(screen.queryByText("123")).not.toBeInTheDocument()
    })
  })

  // -----------------------------------------------------------------------
  // Test 6: Empty name
  // -----------------------------------------------------------------------
  describe("empty name prevention", () => {
    it("disables save button when profile name is empty", () => {
      renderContextBar()

      openProfileMenu()
      enterSaveMode()

      const saveButton = screen.getByRole("button", { name: "Save" })
      expect(saveButton).toBeDisabled()
    })

    it("disables save button when profile name is whitespace only", () => {
      renderContextBar()

      openProfileMenu()
      enterSaveMode()
      typeProfileName("   ")

      const saveButton = screen.getByRole("button", { name: "Save" })
      expect(saveButton).toBeDisabled()
    })

    it("does not persist anything when Enter is pressed with empty name", () => {
      renderContextBar()

      openProfileMenu()
      enterSaveMode()

      const nameInput = screen.getByPlaceholderText("Profile name")
      fireEvent.keyDown(nameInput, { key: "Enter" })

      expect(localStorage.getItem(PROFILES_STORAGE_KEY)).toBeNull()
    })
  })

  describe("profile transfer", () => {
    it("exports profiles with a privacy disclosure before data leaves browser-local storage", () => {
      localStorage.setItem(
        PROFILES_STORAGE_KEY,
        JSON.stringify([
          {
            name: "Pinned Evidence",
            sources: ["media_db"],
            includeMediaIds: [42],
            preset: "balanced",
            enableWebFallback: true,
          },
        ])
      )

      renderContextBar()

      openProfileMenu()
      fireEvent.click(screen.getByRole("button", { name: "Export profiles" }))

      expect(screen.getAllByText(/Export includes selected source IDs, answer model, and web fallback state/i)).toHaveLength(2)
      expect(screen.getAllByText(/copied exports cannot be revoked/i)).toHaveLength(2)
      const exportedJson = (screen.getByLabelText("Exported profile JSON") as HTMLTextAreaElement).value
      expect(exportedJson).toContain('"name": "Pinned Evidence"')
      expect(exportedJson).toContain('"includeMediaIds"')
    })

    it("imports valid shared profiles and rejects invalid entries", () => {
      renderContextBar()

      openProfileMenu()
      fireEvent.click(screen.getByRole("button", { name: "Import profiles" }))
      fireEvent.change(screen.getByLabelText("Profile import JSON"), {
        target: {
          value: JSON.stringify([
            {
              name: "Shared QA Scope",
              sources: ["media_db", "notes"],
              includeMediaIds: [42],
              includeNoteIds: ["note-1"],
              preset: "thorough",
              enableWebFallback: false,
            },
            {
              name: "Invalid Scope",
              sources: ["not-real"],
              preset: "fast",
              enableWebFallback: true,
            },
          ]),
        },
      })
      fireEvent.click(screen.getByRole("button", { name: "Import" }))

      expect(screen.getByRole("menuitem", { name: /Shared QA Scope/i })).toBeInTheDocument()
      expect(screen.queryByRole("menuitem", { name: /Invalid Scope/i })).not.toBeInTheDocument()
      const stored = JSON.parse(localStorage.getItem(PROFILES_STORAGE_KEY)!)
      expect(stored).toHaveLength(1)
      expect(stored[0].name).toBe("Shared QA Scope")
    })

    it("de-duplicates profiles within an imported payload by name", () => {
      renderContextBar()

      openProfileMenu()
      fireEvent.click(screen.getByRole("button", { name: "Import profiles" }))
      fireEvent.change(screen.getByLabelText("Profile import JSON"), {
        target: {
          value: JSON.stringify([
            {
              name: "Shared QA Scope",
              sources: ["media_db"],
              preset: "fast",
              enableWebFallback: true,
            },
            {
              name: "Shared QA Scope",
              sources: ["notes"],
              preset: "thorough",
              enableWebFallback: false,
            },
          ]),
        },
      })
      fireEvent.click(screen.getByRole("button", { name: "Import" }))

      expect(screen.getAllByRole("menuitem", { name: /Shared QA Scope/i })).toHaveLength(1)
      const stored = JSON.parse(localStorage.getItem(PROFILES_STORAGE_KEY)!)
      expect(stored).toHaveLength(1)
      expect(stored[0].sources).toEqual(["notes"])
      expect(stored[0].preset).toBe("thorough")
      expect(stored[0].enableWebFallback).toBe(false)
    })
  })
})
