import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { createInstance } from "i18next"
import settings from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import {
  CharacterListContent,
  type CharacterListContentProps
} from "../CharacterListContent"

const t = ((
  key: string,
  fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
) => {
  if (typeof fallbackOrOptions === "string") return fallbackOrOptions
  if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
    return fallbackOrOptions.defaultValue || key
  }
  return key
}) as CharacterListContentProps["t"]

const createProps = (
  overrides: Partial<CharacterListContentProps> = {}
): CharacterListContentProps => ({
  t,
  status: "error",
  error: new Error("server down"),
  refetch: vi.fn(),
  data: undefined,
  totalCharacters: 0,
  pagedGalleryData: [],
  conversationCounts: undefined,
  viewMode: "table",
  characterListScope: "active",
  setCharacterListScope: vi.fn(),
  galleryDensity: "rich",
  tableDensity: "comfortable",
  currentPage: 1,
  setCurrentPage: vi.fn(),
  pageSize: 20,
  setPageSize: vi.fn(),
  sortColumn: null,
  setSortColumn: vi.fn(),
  sortOrder: null,
  setSortOrder: vi.fn(),
  hasFilters: false,
  searchTerm: "",
  filterTags: [],
  setFilterTags: vi.fn(),
  matchAllTags: false,
  folderFilterId: undefined,
  selectedFolderFilterLabel: undefined,
  creatorFilter: undefined,
  createdFromDate: "",
  createdToDate: "",
  updatedFromDate: "",
  updatedToDate: "",
  hasConversationsOnly: false,
  favoritesOnly: false,
  clearFilters: vi.fn(),
  previewCharacter: null,
  setPreviewCharacter: vi.fn(),
  previewCharacterWorldBooks: [],
  previewCharacterWorldBooksLoading: false,
  crossNavigationContext: { launchedFromWorldBooks: false },
  inlineEdit: null,
  setInlineEdit: vi.fn(),
  inlineUpdating: false,
  inlineEditInputRef: React.createRef(),
  startInlineEdit: vi.fn(),
  saveInlineEdit: vi.fn(),
  cancelInlineEdit: vi.fn(),
  selectedCharacterIds: new Set(),
  setSelectedCharacterIds: vi.fn(),
  toggleCharacterSelection: vi.fn(),
  selectAllOnPage: vi.fn(),
  clearSelection: vi.fn(),
  selectedCount: 0,
  hasSelection: false,
  allOnPageSelected: false,
  someOnPageSelected: false,
  handleBulkDelete: vi.fn(),
  handleBulkExport: vi.fn(),
  handleOpenCompareModal: vi.fn(),
  bulkOperationLoading: false,
  setBulkTagModalOpen: vi.fn(),
  handleChat: vi.fn(),
  handleChatInNewTab: vi.fn().mockResolvedValue(undefined),
  preloadCharacterEditor: vi.fn().mockResolvedValue(undefined),
  handleEdit: vi.fn(),
  handleDuplicate: vi.fn(),
  handleDelete: vi.fn().mockResolvedValue(undefined),
  handleExport: vi.fn().mockResolvedValue(undefined),
  handleViewConversations: vi.fn(),
  handleRestoreFromTrash: vi.fn(),
  handleToggleFavorite: vi.fn().mockResolvedValue(undefined),
  handleSetDefaultCharacter: vi.fn().mockResolvedValue(undefined),
  handleClearDefaultCharacter: vi.fn().mockResolvedValue(undefined),
  isDefaultCharacterRecord: vi.fn().mockReturnValue(false),
  isCharacterFavoriteRecord: vi.fn().mockReturnValue(false),
  isPersonaCreatePending: vi.fn().mockReturnValue(false),
  getCreatePersonaActionLabel: vi.fn().mockReturnValue("Create persona"),
  openPersonaGardenForCharacter: vi.fn(),
  createPersonaFromCharacter: vi.fn(),
  openVersionHistory: vi.fn(),
  openQuickChat: vi.fn(),
  deleting: false,
  exporting: null,
  setConversationCharacter: vi.fn(),
  setCharacterChats: vi.fn(),
  setChatsError: vi.fn(),
  setConversationsOpen: vi.fn(),
  openCreateModal: vi.fn(),
  setShowTemplates: vi.fn(),
  markTemplateChooserSeen: vi.fn(),
  isImportBusy: false,
  triggerImportPicker: vi.fn(),
  confirmDanger: vi.fn().mockResolvedValue(true),
  ...overrides
})

describe("CharacterListContent design-system alerts", () => {
  it("renders load errors through the design-system Alert primitive and keeps retry behavior", async () => {
    const user = userEvent.setup()
    const refetch = vi.fn()

    render(<CharacterListContent {...createProps({ refetch })} />)

    const title = screen.getByText("Couldn't load characters")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).toBeInTheDocument()

    await user.click(within(alert as HTMLElement).getByRole("button", { name: "Retry" }))

    expect(refetch).toHaveBeenCalledTimes(1)
  })
})

const createEnglishTranslator = async (includeAnnouncement = true) => {
  const resource = structuredClone(settings)
  if (!includeAnnouncement) {
    delete (resource.manageCharacters.aria as Record<string, string>).searchResults
  }
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    resources: { en: { settings: resource } }
  })
  return i18n.t as CharacterListContentProps["t"]
}

const resultProps = (
  translate: CharacterListContentProps["t"],
  count: number
) => createProps({
  t: translate,
  status: "success",
  error: null,
  totalCharacters: count,
  data: Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    name: `Character ${index + 1}`,
    description: "A character result",
    tags: []
  }))
})

describe.each([true, false])("Character result announcements (English resource: %s)", includeAnnouncement => {
  it.each([
    [0, "0 characters found"],
    [1, "1 character found"],
    [2, "2 characters found"]
  ] as const)("announces %i results with the correct noun", async (count, expected) => {
    const translate = await createEnglishTranslator(includeAnnouncement)
    render(<CharacterListContent {...resultProps(translate, count)} />)

    const announcement = screen.getByRole("status")
    expect(announcement).toHaveTextContent(expected)
    expect(announcement).toHaveAttribute("aria-live", "polite")
    expect(announcement).toHaveAttribute("aria-atomic", "true")
  })
})

it("updates the announcement when the result count changes on one translator", async () => {
  const translate = await createEnglishTranslator()
  const view = render(<CharacterListContent {...resultProps(translate, 2)} />)
  expect(screen.getByRole("status")).toHaveTextContent("2 characters found")

  view.rerender(<CharacterListContent {...resultProps(translate, 1)} />)
  expect(screen.getByRole("status")).toHaveTextContent("1 character found")

  view.rerender(<CharacterListContent {...resultProps(translate, 0)} />)
  expect(screen.getByRole("status")).toHaveTextContent("0 characters found")
})

it.each(["pending", "error"] as const)("does not announce results while %s", async status => {
  const translate = await createEnglishTranslator()
  render(<CharacterListContent {...resultProps(translate, 1)} status={status} />)

  expect(screen.getByRole("status")).toBeEmptyDOMElement()
})
