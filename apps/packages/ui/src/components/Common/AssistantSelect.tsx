import { Dropdown, Input, Tooltip } from "antd"
import type { InputRef } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import React from "react"
import { Search, Star, UserCircle2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { PersonaInfo } from "@/routes/personaTypes"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { useStoreMessageOption } from "@/store/option"
import {
  OPEN_ASSISTANT_SELECT_EVENT,
  type AssistantSelectOpenDetail,
  type AssistantSelectTab
} from "@/utils/assistant-select-events"
import {
  characterToAssistantSelection,
  personaToAssistantSelection,
  type AssistantSelection
} from "@/types/assistant-selection"
import { scheduleFocusFirstVisibleElement } from "@/utils/focus-return"
import {
  buildAssistantOverlaySnapshotFromSelection,
  resolveAssistantOverlaySnapshot
} from "@/utils/assistant-overlay"

type Props = {
  className?: string
  iconClassName?: string
  showLabel?: boolean
  variant?: "inline" | "dropdown"
  labelOverride?: string
  selectionModePreference?: "tracked" | "overlay"
}

type CharacterSummary = Record<string, unknown> & {
  id?: string | number
  slug?: string
  name?: string
  title?: string
  avatar_url?: string
  system_prompt?: string
  greeting?: string
  extensions?: Record<string, unknown> | null
}

type FavoriteCharacter = {
  id?: string
  slug?: string
  name: string
}

const normalizeCharacterSelection = (
  character: CharacterSummary
): AssistantSelection | null => {
  const normalizedId =
    character.id != null
      ? String(character.id)
      : typeof character.slug === "string" && character.slug.trim().length > 0
        ? character.slug.trim()
        : null
  const normalizedName =
    typeof character.name === "string" && character.name.trim().length > 0
      ? character.name.trim()
      : typeof character.title === "string" && character.title.trim().length > 0
        ? character.title.trim()
        : normalizedId
  if (!normalizedId || !normalizedName) return null
  return characterToAssistantSelection({
    ...character,
    id: normalizedId,
    name: normalizedName
  })
}

const normalizePersonaSelection = (
  persona: PersonaInfo
): AssistantSelection | null => {
  const normalizedId =
    persona.id != null ? String(persona.id) : null
  const normalizedName =
    typeof persona.name === "string" && persona.name.trim().length > 0
      ? persona.name.trim()
      : normalizedId
  if (!normalizedId || !normalizedName) return null
  return personaToAssistantSelection({
    ...persona,
    id: normalizedId,
    name: normalizedName
  })
}

const byAssistantName = (left: AssistantSelection, right: AssistantSelection) =>
  left.name.localeCompare(right.name)

const normalizeFavoriteEntry = (
  favorite: FavoriteCharacter
): FavoriteCharacter | null => {
  const name =
    typeof favorite.name === "string" ? favorite.name.trim() : ""
  if (!name) return null
  return {
    id: typeof favorite.id === "string" ? favorite.id : undefined,
    slug: typeof favorite.slug === "string" ? favorite.slug : undefined,
    name
  }
}

export const AssistantSelect: React.FC<Props> = ({
  className = "text-text-muted",
  iconClassName = "size-5",
  showLabel = true,
  variant = "inline",
  labelOverride,
  selectionModePreference = "tracked"
}) => {
  const { t } = useTranslation(["option", "common"])
  const [selectedAssistant, setSelectedAssistant] =
    useSelectedAssistant(null)
  const historyId = useStoreMessageOption((state) => state.historyId)
  const serverChatId = useStoreMessageOption((state) => state.serverChatId)
  const serverChatAssistantKind = useStoreMessageOption(
    (state) => state.serverChatAssistantKind
  )
  const serverChatAssistantId = useStoreMessageOption(
    (state) => state.serverChatAssistantId
  )
  const serverChatCharacterId = useStoreMessageOption(
    (state) => state.serverChatCharacterId
  )
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const [open, setOpen] = React.useState(false)
  const [searchText, setSearchText] = React.useState("")
  const selectionModeIntentRef = React.useRef<"tracked" | "overlay">(
    selectionModePreference
  )
  const [activeTab, setActiveTab] = React.useState<"character" | "persona">(
    selectedAssistant?.kind ?? "character"
  )
  const [characters, setCharacters] = React.useState<CharacterSummary[]>([])
  const [personas, setPersonas] = React.useState<PersonaInfo[]>([])
  const [charactersLoading, setCharactersLoading] = React.useState(true)
  const [personasLoading, setPersonasLoading] = React.useState(true)
  const [charactersError, setCharactersError] = React.useState(false)
  const [personasError, setPersonasError] = React.useState(false)
  const [favoriteCharacters, setFavoriteCharacters] = useStorage<
    FavoriteCharacter[]
  >("favoriteCharacters", [])
  const searchInputRef = React.useRef<InputRef | null>(null)
  const triggerButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const returnFocusSelectorRef = React.useRef<string | null>(null)
  const effectiveAssistantState = React.useMemo(
    () =>
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: serverChatAssistantKind,
          assistantId: serverChatAssistantId,
          characterId: serverChatCharacterId
        },
        settings: settings ?? null,
        draftSelection: selectedAssistant
      }),
    [
      selectedAssistant,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatCharacterId,
      settings
    ]
  )

  const restoreReturnFocus = React.useCallback(() => {
    const selector = returnFocusSelectorRef.current
    returnFocusSelectorRef.current = null
    if (!selector) {
      if (variant === "dropdown" && typeof window !== "undefined") {
        window.requestAnimationFrame(() => {
          triggerButtonRef.current?.focus()
        })
      }
      return
    }

    scheduleFocusFirstVisibleElement(selector)
  }, [variant])

  React.useEffect(() => {
    if (selectedAssistant?.kind === "character" || selectedAssistant?.kind === "persona") {
      setActiveTab(selectedAssistant.kind)
    }
  }, [selectedAssistant?.kind])

  React.useEffect(() => {
    selectionModeIntentRef.current = selectionModePreference
  }, [selectionModePreference])

  React.useEffect(() => {
    if (typeof window === "undefined") return

    const handleOpen = (event: Event) => {
      const detail = (event as CustomEvent<AssistantSelectOpenDetail>).detail
      const requestedTab = detail?.tab
      if (requestedTab === "character" || requestedTab === "persona") {
        setActiveTab(requestedTab as AssistantSelectTab)
      }
      returnFocusSelectorRef.current =
        typeof detail?.returnFocusSelector === "string" &&
        detail.returnFocusSelector.trim().length > 0
          ? detail.returnFocusSelector.trim()
          : null
      selectionModeIntentRef.current =
        detail?.applyAs === "overlay" ? "overlay" : selectionModePreference
      setSearchText("")
      setOpen(true)
    }

    window.addEventListener(OPEN_ASSISTANT_SELECT_EVENT, handleOpen)
    return () => {
      window.removeEventListener(OPEN_ASSISTANT_SELECT_EVENT, handleOpen)
    }
  }, [selectionModePreference])

  React.useEffect(() => {
    if (!open || typeof window === "undefined") return

    let frameId: number | null = null
    let attempts = 0
    let cancelled = false
    const focusWhenReady = () => {
      if (cancelled) return
      if (searchInputRef.current) {
        searchInputRef.current.focus()
        return
      }
      if (attempts < 10) {
        attempts += 1
        frameId = window.requestAnimationFrame(focusWhenReady)
      }
    }

    frameId = window.requestAnimationFrame(focusWhenReady)

    return () => {
      cancelled = true
      if (frameId !== null) {
        window.cancelAnimationFrame(frameId)
      }
    }
  }, [open])

  const loadCharacters = React.useCallback(
    async (isCancelled: () => boolean = () => false) => {
      setCharactersLoading(true)
      setCharactersError(false)
      try {
        await tldwClient.initialize()
        if (typeof tldwClient.listAllCharacters !== "function") {
          if (!isCancelled()) {
            setCharacters([])
          }
          return
        }
        const result = await tldwClient.listAllCharacters()
        if (!isCancelled()) {
          setCharacters(Array.isArray(result) ? (result as CharacterSummary[]) : [])
        }
      } catch {
        if (!isCancelled()) {
          setCharacters([])
          setCharactersError(true)
        }
      } finally {
        if (!isCancelled()) {
          setCharactersLoading(false)
        }
      }
    },
    []
  )

  const loadPersonas = React.useCallback(
    async (isCancelled: () => boolean = () => false) => {
      setPersonasLoading(true)
      setPersonasError(false)
      try {
        await tldwClient.initialize()
        if (typeof tldwClient.listPersonaProfiles !== "function") {
          if (!isCancelled()) {
            setPersonas([])
          }
          return
        }
        const result = await tldwClient.listPersonaProfiles()
        if (!isCancelled()) {
          setPersonas(Array.isArray(result) ? (result as PersonaInfo[]) : [])
        }
      } catch {
        if (!isCancelled()) {
          setPersonas([])
          setPersonasError(true)
        }
      } finally {
        if (!isCancelled()) {
          setPersonasLoading(false)
        }
      }
    },
    []
  )

  React.useEffect(() => {
    let cancelled = false
    const isCancelled = () => cancelled

    void loadCharacters(isCancelled)
    void loadPersonas(isCancelled)
    return () => {
      cancelled = true
    }
  }, [loadCharacters, loadPersonas])

  const characterEntries = React.useMemo(
    () =>
      characters
        .map(normalizeCharacterSelection)
        .filter((entry): entry is AssistantSelection => Boolean(entry)),
    [characters]
  )
  const personaEntries = React.useMemo(
    () =>
      personas
        .map(normalizePersonaSelection)
        .filter((entry): entry is AssistantSelection => Boolean(entry)),
    [personas]
  )

  const favoriteIndex = React.useMemo(() => {
    const ids = new Set<string>()
    const slugs = new Set<string>()
    const names = new Set<string>()

    ;(favoriteCharacters || [])
      .map(normalizeFavoriteEntry)
      .filter((entry): entry is FavoriteCharacter => Boolean(entry))
      .forEach((entry) => {
        if (entry.id) ids.add(entry.id)
        if (entry.slug) slugs.add(entry.slug)
        names.add(entry.name)
      })

    return { ids, slugs, names }
  }, [favoriteCharacters])

  const isFavoriteCharacter = React.useCallback(
    (entry: AssistantSelection) => {
      const slug =
        typeof entry.slug === "string" && entry.slug.trim().length > 0
          ? entry.slug.trim()
          : ""
      const name = entry.name.trim()
      return (
        favoriteIndex.ids.has(entry.id) ||
        (slug.length > 0 && favoriteIndex.slugs.has(slug)) ||
        favoriteIndex.names.has(name)
      )
    },
    [favoriteIndex]
  )

  const toggleFavoriteCharacter = React.useCallback(
    (entry: AssistantSelection) => {
      const slug =
        typeof entry.slug === "string" && entry.slug.trim().length > 0
          ? entry.slug.trim()
          : undefined
      const nextFavorite: FavoriteCharacter = {
        id: entry.id,
        slug,
        name: entry.name.trim()
      }
      if (!nextFavorite.name) return

      void setFavoriteCharacters((previousFavorites) => {
        const list = Array.isArray(previousFavorites) ? previousFavorites : []
        const nextFavorites = list.filter((favorite) => {
          if (nextFavorite.id && favorite.id === nextFavorite.id) return false
          if (nextFavorite.slug && favorite.slug === nextFavorite.slug) return false
          if (favorite.name === nextFavorite.name) return false
          return true
        })

        if (nextFavorites.length === list.length) {
          nextFavorites.push(nextFavorite)
        }

        return nextFavorites
      })
    },
    [setFavoriteCharacters]
  )

  const filteredCharacterEntries = React.useMemo(() => {
    const query = searchText.trim().toLowerCase()
    if (!query) {
      return characterEntries.slice()
    }
    return characterEntries.filter((entry) =>
      entry.name.toLowerCase().includes(query)
    )
  }, [characterEntries, searchText])

  const filteredPersonaEntries = React.useMemo(() => {
    const query = searchText.trim().toLowerCase()
    if (!query) {
      return personaEntries.slice()
    }
    return personaEntries.filter((entry) =>
      entry.name.toLowerCase().includes(query)
    )
  }, [personaEntries, searchText])

  const sortedCharacterEntries = React.useMemo(() => {
    const favorites = filteredCharacterEntries
      .filter(isFavoriteCharacter)
      .sort(byAssistantName)
    const others = filteredCharacterEntries
      .filter((entry) => !isFavoriteCharacter(entry))
      .sort(byAssistantName)
    return [...favorites, ...others]
  }, [filteredCharacterEntries, isFavoriteCharacter])

  const sortedPersonaEntries = React.useMemo(
    () => filteredPersonaEntries.slice().sort(byAssistantName),
    [filteredPersonaEntries]
  )

  const handleSelect = React.useCallback(
    async (entry: AssistantSelection) => {
      const nextMode = selectionModeIntentRef.current
      const isTrackedMode =
        effectiveAssistantState.mode === "tracked_character" ||
        effectiveAssistantState.mode === "tracked_persona"
      if (nextMode === "overlay" && isTrackedMode) {
        setOpen(false)
        setSearchText("")
        selectionModeIntentRef.current = selectionModePreference
        restoreReturnFocus()
        return
      }

      setOpen(false)
      setSearchText("")
      selectionModeIntentRef.current = selectionModePreference
      restoreReturnFocus()
      const nextEntry: AssistantSelection = {
        ...entry,
        metadata: {
          ...(entry.metadata ?? {}),
          selectionMode: nextMode
        }
      }
      await setSelectedAssistant(nextEntry)
      if (nextMode === "overlay") {
        let overlaySnapshot = buildAssistantOverlaySnapshotFromSelection(nextEntry)
        try {
          overlaySnapshot = await resolveAssistantOverlaySnapshot(nextEntry)
        } catch (error) {
          console.warn(
            "[AssistantSelect] Failed to resolve overlay snapshot; using summary fallback",
            error
          )
        }

        try {
          await updateSettings({
            assistantOverlay: overlaySnapshot
          })
        } catch (error) {
          console.warn(
            "[AssistantSelect] Failed to persist assistant overlay; keeping local selection",
            error
          )
        }
      }
    },
    [
      effectiveAssistantState.mode,
      restoreReturnFocus,
      selectionModePreference,
      setSelectedAssistant,
      updateSettings
    ]
  )

  const handleOpenChange = React.useCallback((nextOpen: boolean) => {
    setOpen(nextOpen)
    if (nextOpen) {
      selectionModeIntentRef.current = selectionModePreference
    }
    if (!nextOpen) {
      setSearchText("")
      selectionModeIntentRef.current = selectionModePreference
      restoreReturnFocus()
    }
  }, [restoreReturnFocus, selectionModePreference])

  const openActorSettings = React.useCallback(() => {
    setOpen(false)
    setSearchText("")
    selectionModeIntentRef.current = selectionModePreference
    restoreReturnFocus()
    try {
      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("tldw:open-actor-settings"))
      }
    } catch {
      // no-op
    }
  }, [restoreReturnFocus, selectionModePreference])

  const buttonLabel =
    labelOverride ||
    selectedAssistant?.name ||
    t("option:assistant.selectAssistant", "Select character or persona")

  const searchLabel = t(
    "option:assistant.searchPlaceholder",
    "Search characters and personas"
  )
  const actorLabel = t(
    "playground:composer.actorTitle",
    "Optional scene context"
  )

  const tabs = [
    {
      key: "character" as const,
      label: t("option:assistant.charactersTab", "Characters"),
      emptyLabel: searchText.trim()
        ? t("option:assistant.noCharacterMatches", "No characters match your search.")
        : t("option:assistant.noCharacters", "No characters available."),
      entries: sortedCharacterEntries,
      showFavorites: true
    },
    {
      key: "persona" as const,
      label: t("option:assistant.personasTab", "Personas"),
      emptyLabel: searchText.trim()
        ? t("option:assistant.noPersonaMatches", "No personas match your search.")
        : t("option:assistant.noPersonas", "No personas available."),
      entries: sortedPersonaEntries,
      showFavorites: false
    }
  ]
  const activeTabDefinition = tabs.find((tab) => tab.key === activeTab) ?? tabs[0]
  const activeTabEntries = activeTabDefinition?.entries ?? []
  const activeTabEmptyLabel =
    activeTabDefinition?.emptyLabel ??
    t("option:assistant.noAssistants", "No assistants available.")
  const activeTabShowsFavorites = activeTabDefinition?.showFavorites ?? false
  const activeTabLoading =
    activeTab === "character" ? charactersLoading : personasLoading
  const activeTabError =
    activeTab === "character" ? charactersError : personasError
  const retryActiveTabLoad =
    activeTab === "character" ? loadCharacters : loadPersonas
  const activeTabErrorLabel =
    activeTab === "character"
      ? t(
          "option:assistant.charactersLoadError",
          "Could not load characters."
        )
      : t("option:assistant.personasLoadError", "Could not load personas.")
  const activeTabRetryLabel =
    activeTab === "character"
      ? t("option:assistant.retryCharacters", "Retry characters")
      : t("option:assistant.retryPersonas", "Retry personas")
  const catalogLoadingLabel = t(
    "option:assistant.catalogLoadingStatus",
    "Loading character and persona catalogs"
  )
  const activeTabLoadingLabel = t(
    "option:assistant.loadingCatalogs",
    "Loading characters and personas"
  )

  const activeTabContent =
    activeTabLoading ? (
      <div
        role="status"
        aria-label={catalogLoadingLabel}
        className="px-3 py-4 text-center text-sm text-text-subtle"
      >
        {activeTabLoadingLabel}
      </div>
    ) : activeTabError ? (
      <div className="space-y-2 px-3 py-4 text-center text-sm text-text-subtle">
        <p>{activeTabErrorLabel}</p>
        <button
          type="button"
          className="rounded-md border border-border bg-surface px-2 py-1 text-xs font-medium text-text hover:bg-surface2"
          onClick={() => {
            void retryActiveTabLoad()
          }}
        >
          {activeTabRetryLabel}
        </button>
      </div>
    ) : activeTabEntries.length === 0 ? (
      <div className="px-3 py-4 text-center text-sm text-text-subtle">
        {activeTabEmptyLabel}
      </div>
    ) : (
      <div
        data-testid="assistant-select-menu"
        className="max-h-80 overflow-y-auto px-2 py-2"
      >
        <div className="flex flex-col gap-1">
          {activeTabEntries.map((entry) => {
            const isActive =
              selectedAssistant?.kind === entry.kind &&
              selectedAssistant?.id === entry.id
            const isFavorite =
              activeTabShowsFavorites && isFavoriteCharacter(entry)
            const favoriteLabel = isFavorite
              ? t(
                  "option:assistant.favoriteRemove",
                  `Remove ${entry.name} from favorites`
                )
              : t(
                  "option:assistant.favoriteAdd",
                  `Add ${entry.name} to favorites`
                )

            return (
              <div
                key={`${entry.kind}:${entry.id}`}
                className="flex items-center gap-2"
              >
                <button
                  type="button"
                  aria-label={entry.name}
                  className={`flex min-w-0 flex-1 items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition ${
                    isActive
                      ? "border-primary bg-primary/10 text-text"
                      : "border-border bg-background text-text hover:bg-surface2"
                  }`}
                  onClick={() => {
                    void handleSelect(entry)
                  }}
                >
                  {entry.avatar_url ? (
                    <img
                      src={entry.avatar_url}
                      alt={entry.name}
                      className="h-5 w-5 rounded-full"
                    />
                  ) : (
                    <UserCircle2 className="h-5 w-5 flex-shrink-0 text-text-subtle" />
                  )}
                  <span className="min-w-0 flex-1 truncate font-medium">
                    {entry.name}
                  </span>
                  <span
                    aria-hidden="true"
                    className="text-xs text-text-subtle"
                  >
                    {entry.kind === "persona" ? "Persona" : "Character"}
                  </span>
                </button>
                {activeTabShowsFavorites ? (
                  <Tooltip title={favoriteLabel}>
                    <button
                      type="button"
                      className="rounded-md p-1.5 text-text-subtle transition hover:bg-surface2"
                      aria-label={favoriteLabel}
                      onMouseDown={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                      }}
                      onClick={(event) => {
                        event.preventDefault()
                        event.stopPropagation()
                        toggleFavoriteCharacter(entry)
                      }}
                    >
                      <Star
                        className={`h-4 w-4 ${
                          isFavorite ? "fill-warn text-warn" : "text-text-subtle"
                        }`}
                      />
                    </button>
                  </Tooltip>
                ) : null}
              </div>
            )
          })}
        </div>
      </div>
    )

  const content = (
    <div
      data-testid="assistant-select-panel"
      className="w-[320px] rounded-lg border border-border bg-background shadow-lg"
    >
      <div className="border-b border-border p-2">
        <Input
          ref={searchInputRef}
          aria-label={searchLabel}
          placeholder={searchLabel}
          prefix={<Search className="size-4 text-text-subtle" />}
          value={searchText}
          allowClear
          size="small"
          onChange={(event) => setSearchText(event.target.value)}
          onKeyDown={(event) => event.stopPropagation()}
        />
      </div>
      <div
        role="tablist"
        aria-label={t("option:assistant.tabList", "Character or persona")}
        className="flex items-center gap-1 border-b border-border px-2 pt-2"
      >
        {tabs.map((tab) => {
          const isActive = tab.key === activeTab
          return (
            <button
              key={tab.key}
              id={`assistant-tab-${tab.key}`}
              type="button"
              role="tab"
              aria-selected={isActive}
              aria-controls={`assistant-tabpanel-${tab.key}`}
              className={`rounded-t-md px-3 py-2 text-sm transition ${
                isActive
                  ? "bg-surface2 font-medium text-text"
                  : "text-text-subtle hover:text-text"
              }`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          )
        })}
      </div>
      <div
        id={`assistant-tabpanel-${activeTab}`}
        role="tabpanel"
        aria-labelledby={`assistant-tab-${activeTab}`}
      >
        {activeTabContent}
      </div>
      <div className="border-t border-border p-2">
        <button
          type="button"
          className="w-full rounded-md px-3 py-2 text-left text-sm font-medium text-text transition hover:bg-surface2"
          onClick={openActorSettings}
        >
          {actorLabel}
        </button>
      </div>
    </div>
  )

  if (variant === "inline") {
    return content
  }

  return (
    <Dropdown
      open={open}
      onOpenChange={handleOpenChange}
      menu={{ items: [] }}
      popupRender={() => content}
      placement="topLeft"
      trigger={["click"]}
    >
      <Tooltip title={buttonLabel}>
        <button
          ref={triggerButtonRef}
          type="button"
          data-testid="character-select"
          className={`inline-flex items-center gap-2 ${className}`.trim()}
          aria-label={buttonLabel}
          aria-expanded={open}
        >
          <UserCircle2 className={iconClassName} />
          {showLabel ? (
            <span className="max-w-[180px] truncate text-sm">{buttonLabel}</span>
          ) : null}
        </button>
      </Tooltip>
    </Dropdown>
  )
}

export default AssistantSelect
