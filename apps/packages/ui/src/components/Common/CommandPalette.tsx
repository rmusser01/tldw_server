import React, { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { createPortal } from "react-dom"
import { useLocation, useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"
import {
  Search,
  MessageSquare,
  Settings,
  CombineIcon,
  BookText,
  BookOpen,
  NotebookPen,
  StickyNote,
  Layers,
  UploadCloud,
  Globe,
  Eye,
  BrainCircuit,
  Activity,
  X,
  Command,
  ArrowRight,
} from "lucide-react"
import {
  useShortcut,
  formatShortcut,
  type ShortcutModifier
} from "@/hooks/useKeyboardShortcuts"
import { useShortcutConfig } from "@/hooks/keyboard/useShortcutConfig"
import type { KeyboardShortcut as ConfiguredKeyboardShortcut } from "@/hooks/keyboard/useKeyboardShortcuts"
import { WORKSPACE_PLAYGROUND_PATH } from "@/routes/route-paths"
import { searchSettings } from "@/data/settings-index"
import { cn } from "@/libs/utils"

type CommandShortcut = { key: string; modifiers: ShortcutModifier[] }

const buildShortcut = (
  key: string,
  ...modifiers: ShortcutModifier[]
): CommandShortcut => ({
  key,
  modifiers
})

const toCommandShortcut = (
  shortcut: ConfiguredKeyboardShortcut | null | undefined
): CommandShortcut | undefined => {
  if (!shortcut?.key) {
    return undefined
  }
  const modifiers: ShortcutModifier[] = []
  if (shortcut.metaKey) modifiers.push("meta")
  if (shortcut.ctrlKey) modifiers.push("ctrl")
  if (shortcut.altKey) modifiers.push("alt")
  if (shortcut.shiftKey) modifiers.push("shift")
  return buildShortcut(shortcut.key, ...modifiers)
}

export interface CommandItem {
  id: string
  label: string
  description?: string
  icon: React.ReactNode
  shortcut?: CommandShortcut
  action: () => void
  targetPath?: string
  category: "navigation" | "action" | "setting" | "recent" | "prompt"
  keywords?: string[]
}

export interface CommandPaletteProps {
  /** Custom commands to add to the palette */
  additionalCommands?: CommandItem[]
  /** Callbacks for actions */
  onNewChat?: () => void
  onToggleRag?: () => void
  onToggleWebSearch?: () => void
  onIngestPage?: () => void
  onSwitchModel?: () => void
  onToggleSidebar?: () => void
  onSearchHistory?: () => void
  onSwitchChat?: (chatId: string) => void
  sidepanelChats?: { id: string; label: string }[]
  scope?: "global" | "sidepanel"
  openSignal?: number
  registerGlobalOpenShortcut?: boolean
  listenForOpenEvents?: boolean
}

export function CommandPalette({
  additionalCommands = [],
  onNewChat,
  onToggleRag,
  onToggleWebSearch,
  onIngestPage,
  onSwitchModel,
  onToggleSidebar,
  onSearchHistory,
  onSwitchChat,
  sidepanelChats,
  scope = "global",
  openSignal,
  registerGlobalOpenShortcut = true,
  listenForOpenEvents = true,
}: CommandPaletteProps) {
  const [open, setOpen] = useState(() => (openSignal ?? 0) > 0)
  const [query, setQuery] = useState("")
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const lastOpenSignalRef = useRef(openSignal ?? 0)
  const location = useLocation()
  const navigate = useNavigate()
  const { t } = useTranslation(["common", "settings"])
  const isSidepanel = scope === "sidepanel"
  const shortcutEnabled = location.pathname !== WORKSPACE_PLAYGROUND_PATH
  const { shortcuts: configuredShortcuts } = useShortcutConfig()

  const openPalette = useCallback(() => {
    setOpen(true)
  }, [])

  // Register Cmd/Ctrl+K shortcut to open
  useShortcut({
    key: "k",
    modifiers: ["meta"],
    action: openPalette,
    description: "Open command palette",
    enabled: registerGlobalOpenShortcut && shortcutEnabled,
    allowInInput: true,
  })

  useShortcut({
    key: "k",
    modifiers: ["ctrl"],
    action: openPalette,
    description: "Open command palette",
    enabled: registerGlobalOpenShortcut && shortcutEnabled,
    allowInInput: true,
  })

  // Also allow Escape to close
  useShortcut({
    key: "Escape",
    modifiers: [],
    action: () => setOpen(false),
    description: "Close command palette",
    allowInInput: true,
  })

  useEffect(() => {
    if (!listenForOpenEvents) {
      return
    }
    const handleOpen = () => setOpen(true)
    window.addEventListener("tldw:open-command-palette", handleOpen)
    return () => {
      window.removeEventListener("tldw:open-command-palette", handleOpen)
    }
  }, [listenForOpenEvents])

  useEffect(() => {
    const currentOpenSignal = openSignal ?? 0
    if (currentOpenSignal > lastOpenSignalRef.current) {
      setOpen(true)
    }
    lastOpenSignalRef.current = currentOpenSignal
  }, [openSignal])

  // Build default commands
  const defaultCommands: CommandItem[] = useMemo(() => {
    const commands: CommandItem[] = [
      // Navigation
      {
        id: "nav-chat",
        label: t("common:commandPalette.goToChat", "Go to Chat"),
        icon: <MessageSquare className="size-4" />,
        action: () => { navigate("/"); setOpen(false) },
        targetPath: "/",
        category: "navigation",
        keywords: ["playground", "conversation"],
      },
      ...(!isSidepanel ? ([
        {
          id: "nav-knowledge",
          label: t("common:commandPalette.goToKnowledge", "Go to Knowledge QA"),
          icon: <CombineIcon className="size-4" />,
          action: () => { navigate("/knowledge"); setOpen(false) },
          targetPath: "/knowledge",
          category: "navigation" as const,
          keywords: ["knowledge", "qa", "rag", "search"],
        },
        {
          id: "nav-media",
          label: t("common:commandPalette.goToMedia", "Go to Media"),
          icon: <BookText className="size-4" />,
          action: () => { navigate("/media"); setOpen(false) },
          targetPath: "/media",
          category: "navigation" as const,
          keywords: ["documents", "files", "library"],
        },
        {
          id: "nav-notes",
          label: t("common:commandPalette.goToNotes", "Go to Notes"),
          icon: <StickyNote className="size-4" />,
          action: () => { navigate("/notes"); setOpen(false) },
          targetPath: "/notes",
          category: "navigation" as const,
          keywords: ["notes", "notebook"],
        },
        {
          id: "nav-prompts",
          label: t("common:commandPalette.goToPrompts", "Go to Prompts"),
          icon: <NotebookPen className="size-4" />,
          action: () => { navigate("/prompts"); setOpen(false) },
          targetPath: "/prompts",
          category: "navigation" as const,
          keywords: ["prompts", "template", "studio"],
        },
        {
          id: "nav-flashcards",
          label: t("common:commandPalette.goToFlashcards", "Go to Flashcards"),
          icon: <Layers className="size-4" />,
          action: () => { navigate("/flashcards"); setOpen(false) },
          targetPath: "/flashcards",
          category: "navigation" as const,
          keywords: ["study", "cards", "learn"],
        },
        {
          id: "nav-documentation",
          label: t(
            "common:commandPalette.goToDocumentation",
            "Go to Documentation"
          ),
          icon: <BookOpen className="size-4" />,
          action: () => { navigate("/documentation"); setOpen(false) },
          targetPath: "/documentation",
          category: "navigation" as const,
          keywords: ["docs", "documentation", "guide", "help", "reference"],
        },
      ] as CommandItem[]) : []),
      {
        id: "nav-settings",
        label: t("common:commandPalette.goToSettings", "Go to Settings"),
        icon: <Settings className="size-4" />,
        action: () => { navigate("/settings"); setOpen(false) },
        targetPath: "/settings",
        category: "navigation",
        keywords: ["preferences", "config", "options"],
      },
      {
        id: "nav-mcp-hub",
        label: t("common:commandPalette.goToMcpHub", "Go to MCP Hub"),
        icon: <Settings className="size-4" />,
        action: () => { navigate("/settings/mcp-hub"); setOpen(false) },
        targetPath: "/settings/mcp-hub",
        category: "navigation",
        keywords: ["mcp", "hub", "acp", "policy", "server"],
      },
      ...(!isSidepanel ? ([
        {
          id: "nav-health",
          label: t(
            "common:commandPalette.goToHealth",
            "Go to Health & Diagnostics"
          ),
          icon: <Activity className="size-4" />,
          action: () => { navigate("/settings/health"); setOpen(false) },
          targetPath: "/settings/health",
          category: "navigation" as const,
          keywords: ["status", "connection", "diagnostic"],
        }
      ] as CommandItem[]) : []),
      // Actions
      ...(onNewChat
        ? ([
            {
              id: "action-new-chat",
              label: t("common:commandPalette.newChat", "New Chat"),
              icon: <MessageSquare className="size-4" />,
              shortcut: toCommandShortcut(configuredShortcuts.newChat),
              action: () => {
                onNewChat()
                setOpen(false)
              },
              category: "action",
              keywords: ["create", "start", "conversation"],
            }
          ] as CommandItem[])
        : []),
      ...(onToggleRag
        ? ([
            {
              id: "action-toggle-rag",
              label: t(
                "common:commandPalette.toggleKnowledgeSearch",
                "Toggle Search & Context"
              ),
              description: t(
                "common:commandPalette.toggleKnowledgeSearchDesc",
                "Search your knowledge base and context"
              ),
              icon: <Search className="size-4" />,
              shortcut: toCommandShortcut(configuredShortcuts.toggleChatMode),
              action: () => {
                onToggleRag()
                setOpen(false)
              },
              category: "action",
              keywords: ["search", "knowledge", "retrieve", "rag"],
            }
          ] as CommandItem[])
        : []),
      ...(onToggleWebSearch
        ? ([
            {
              id: "action-toggle-web",
              label: t(
                "common:commandPalette.toggleWebSearch",
                "Toggle Web Search"
              ),
              description: t(
                "common:commandPalette.toggleWebDesc",
                "Search the internet"
              ),
              icon: <Globe className="size-4" />,
              shortcut: toCommandShortcut(configuredShortcuts.toggleWebSearch),
              action: () => {
                onToggleWebSearch()
                setOpen(false)
              },
              category: "action",
              keywords: ["internet", "online", "browse"],
            }
          ] as CommandItem[])
        : []),
      ...(onIngestPage
        ? ([
            {
              id: "action-ingest",
              label: t("common:commandPalette.ingestPage", "Ingest Current Page"),
              description: t(
                "common:commandPalette.ingestDesc",
                "Save this page to your knowledge base"
              ),
              icon: <UploadCloud className="size-4" />,
              action: () => {
                onIngestPage()
                setOpen(false)
              },
              category: "action",
              keywords: ["save", "import", "add", "upload"],
            }
          ] as CommandItem[])
        : []),
      ...(onSwitchModel
        ? ([
            {
              id: "action-switch-model",
              label: t("common:commandPalette.switchModel", "Switch Model"),
              icon: <BrainCircuit className="size-4" />,
              action: () => {
                onSwitchModel()
                setOpen(false)
              },
              category: "action",
              keywords: ["model", "ai", "llm", "change"],
            }
          ] as CommandItem[])
        : []),
      ...(onToggleSidebar
        ? ([
            {
              id: "action-toggle-sidebar",
              label: t("common:commandPalette.toggleSidebar", "Toggle Sidebar"),
              description: t(
                "common:commandPalette.toggleSidebarDesc",
                "Show or hide the chat sidebar"
              ),
              icon: <Eye className="size-4" />,
              shortcut: toCommandShortcut(configuredShortcuts.toggleSidebar),
              action: () => {
                onToggleSidebar()
                setOpen(false)
              },
              category: "action",
              keywords: ["sidebar", "layout", "panel"],
            }
          ] as CommandItem[])
        : []),
      ...(isSidepanel && onSearchHistory
        ? ([
            {
              id: "action-search-history",
              label: t(
                "common:commandPalette.searchHistory",
                "Search chat history"
              ),
              description: t(
                "common:commandPalette.searchHistoryDesc",
                "Focus the chat search input"
              ),
              icon: <Search className="size-4" />,
              action: () => {
                onSearchHistory()
                setOpen(false)
              },
              category: "action" as const,
              keywords: ["history", "search", "chats", "sidebar"]
            }
          ] as CommandItem[])
        : []),
      ...(isSidepanel && onSwitchChat && sidepanelChats?.length
        ? sidepanelChats.slice(0, 15).map((chat) => ({
            id: `switch-chat-${chat.id}`,
            label: chat.label || t("common:untitled", "Untitled"),
            icon: <MessageSquare className="size-4" />,
            action: () => {
              onSwitchChat(chat.id)
              setOpen(false)
            },
            category: "recent" as const,
            keywords: ["switch", "chat", "conversation"]
          }))
        : [])
    ]

    return commands
  }, [
    t,
    navigate,
    onNewChat,
    onToggleRag,
    onToggleWebSearch,
    onIngestPage,
    onSwitchModel,
    onToggleSidebar,
    configuredShortcuts,
    onSearchHistory,
    onSwitchChat,
    sidepanelChats,
    isSidepanel
  ])

  // Convert settings to commands
  const settingCommands: CommandItem[] = useMemo(() => {
    if (isSidepanel || !query) return []
    // Create a translation wrapper that matches searchSettings expected signature
    const translateFn = (key: string, defaultValue?: string): string => {
      return t(key, defaultValue ?? key) as string
    }
    const results = searchSettings(query, translateFn)
    return results.slice(0, 5).map((setting) => ({
      id: `setting-${setting.id}`,
      label: t(setting.labelKey, setting.defaultLabel),
      description: setting.descriptionKey
        ? t(setting.descriptionKey, setting.defaultDescription)
        : setting.defaultDescription,
      icon: <Settings className="size-4" />,
      action: () => { navigate(setting.route); setOpen(false) },
      targetPath: setting.route,
      category: "setting" as const,
      keywords: setting.keywords,
    }))
  }, [isSidepanel, query, t, navigate])

  // Combine all commands
  const allCommands = useMemo(() => {
    return [...defaultCommands, ...additionalCommands, ...settingCommands]
  }, [defaultCommands, additionalCommands, settingCommands])

  const dedupeByTargetPath = useCallback((commands: CommandItem[]) => {
    const dedupedCommands: CommandItem[] = []
    const targetPathIndex = new Map<string, number>()

    for (const command of commands) {
      if (!command.targetPath) {
        dedupedCommands.push(command)
        continue
      }

      const existingIndex = targetPathIndex.get(command.targetPath)
      if (existingIndex === undefined) {
        targetPathIndex.set(command.targetPath, dedupedCommands.length)
        dedupedCommands.push(command)
        continue
      }

      const existingCommand = dedupedCommands[existingIndex]
      if (existingCommand.category === "setting" && command.category === "setting") {
        dedupedCommands.push(command)
        continue
      }

      if (existingCommand.category === "setting" && command.category !== "setting") {
        dedupedCommands[existingIndex] = command
      }
    }

    return dedupedCommands
  }, [])

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    let commands: CommandItem[]

    if (!query) {
      // Show all non-setting commands when no query
      commands = allCommands.filter(c => c.category !== "setting")
      return dedupeByTargetPath(commands)
    }

    const q = query.toLowerCase()
    commands = allCommands.filter((cmd) => {
      const labelMatch = cmd.label.toLowerCase().includes(q)
      const descMatch = cmd.description?.toLowerCase().includes(q)
      const keywordMatch = cmd.keywords?.some((kw) => kw.toLowerCase().includes(q))
      return labelMatch || descMatch || keywordMatch
    })
    return dedupeByTargetPath(commands)
  }, [allCommands, dedupeByTargetPath, query])

  // Clamp selection when filtered commands change
  useEffect(() => {
    const clampedIndex =
      filteredCommands.length === 0
        ? 0
        : Math.max(0, Math.min(selectedIndex, filteredCommands.length - 1))
    if (clampedIndex !== selectedIndex) {
      setSelectedIndex(clampedIndex)
    }
  }, [filteredCommands, selectedIndex])

  // Group commands by category
  const groupedCommands = useMemo(() => {
    // "recent" is reserved for future MRU/history commands and may be populated
    // by additionalCommands when that feature is implemented.
    const groups: Record<string, CommandItem[]> = {
      action: [],
      prompt: [],
      navigation: [],
      setting: [],
      recent: [],
    }
    for (const cmd of filteredCommands) {
      groups[cmd.category]?.push(cmd)
    }
    return groups
  }, [filteredCommands])

  // Reset selection when filtered results change
  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery("")
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  // Scroll selected item into view
  useEffect(() => {
    if (!listRef.current) return
    const selected = listRef.current.querySelector('[data-selected="true"]')
    selected?.scrollIntoView({ block: "nearest" })
  }, [selectedIndex])

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Tab") {
      const palette = listRef.current?.parentElement
      if (!palette) return

      const focusableSelectors =
        'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
      const focusableElements = Array.from(
        palette.querySelectorAll<HTMLElement>(focusableSelectors)
      ).filter(
        (el) =>
          !el.hasAttribute("disabled") &&
          el.getAttribute("aria-hidden") !== "true"
      )

      if (focusableElements.length === 0) {
        return
      }

      const currentIndex = focusableElements.indexOf(
        document.activeElement as HTMLElement
      )
      let nextIndex = currentIndex

      if (e.shiftKey) {
        nextIndex =
          currentIndex <= 0
            ? focusableElements.length - 1
            : currentIndex - 1
      } else {
        nextIndex =
          currentIndex === -1 || currentIndex === focusableElements.length - 1
            ? 0
            : currentIndex + 1
      }

      e.preventDefault()
      focusableElements[nextIndex]?.focus()
      return
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault()
        setSelectedIndex((i) =>
          filteredCommands.length === 0
            ? 0
            : Math.min(i + 1, filteredCommands.length - 1)
        )
        break
      case "ArrowUp":
        e.preventDefault()
        setSelectedIndex((i) =>
          filteredCommands.length === 0 ? 0 : Math.max(i - 1, 0)
        )
        break
      case "Enter":
        e.preventDefault()
        filteredCommands[selectedIndex]?.action()
        break
      // Escape is handled by useShortcut hook to avoid duplicate handlers
    }
  }, [filteredCommands, selectedIndex])

  // Execute command
  const executeCommand = useCallback((cmd: CommandItem) => {
    cmd.action()
  }, [])

  if (!open) return null
  if (typeof document === "undefined") return null

  const categories = ["recent", "action", "prompt", "navigation", "setting"] as const
  const focusRingClasses =
    "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"

  const categoryLabels: Record<string, string> = {
    action: t("common:commandPalette.categoryActions", "Actions"),
    navigation: t("common:commandPalette.categoryNavigation", "Navigation"),
    prompt: t("common:commandPalette.categoryPrompts", "Prompts"),
    setting: t("common:commandPalette.categorySettings", "Settings"),
    recent: t("common:commandPalette.categoryRecent", "Recent"),
  }

  const modalContent = (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />

      {/* Palette */}
      <div
        className="fixed left-1/2 top-[15%] sm:top-[20%] z-50 w-[calc(100%-2rem)] sm:w-full max-w-lg -translate-x-1/2 overflow-hidden rounded-xl border border-border bg-surface shadow-modal"
        role="dialog"
        aria-modal="true"
        aria-label={t("common:commandPalette.title", "Command Palette")}
        onKeyDown={handleKeyDown}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          <Search className="size-5 text-text-subtle" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("common:commandPalette.placeholder", "Type a command or search...")}
            className={cn(
              "flex-1 rounded-md bg-transparent text-base text-text placeholder:text-text-subtle",
              focusRingClasses
            )}
            autoComplete="off"
            autoCorrect="off"
            spellCheck={false}
          />
          <kbd className="hidden items-center gap-1 rounded border border-border bg-surface2 px-1.5 py-0.5 text-xs text-text-subtle sm:flex">
            Esc
          </kbd>
        </div>

        {/* Results */}
        <div
          ref={listRef}
          className="max-h-[60vh] overflow-y-auto p-2"
          role="listbox"
        >
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-text-subtle">
              {t("common:commandPalette.noResults", "No results found")}
            </div>
          ) : (
            <>
              {categories.map((category) => {
                const items = groupedCommands[category]
                if (!items?.length) return null

                const categoryStartIndex = categories
                  .slice(0, categories.indexOf(category))
                  .reduce((sum, cat) => sum + (groupedCommands[cat]?.length ?? 0), 0)

                return (
                  <div key={category} className="mb-2">
                    <div className="px-2 py-1.5 text-xs font-medium text-text-subtle">
                      {categoryLabels[category]}
                    </div>
                    {items.map((cmd, idx) => {
                      const currentIndex = categoryStartIndex + idx
                      const isSelected = currentIndex === selectedIndex

                      return (
                        <button
                          key={cmd.id}
                          onClick={() => executeCommand(cmd)}
                          onMouseEnter={() => setSelectedIndex(currentIndex)}
                          data-selected={isSelected}
                          className={cn(
                            "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors",
                            focusRingClasses,
                            isSelected
                              ? "bg-primary/10 text-text"
                              : "text-text hover:bg-surface2"
                          )}
                          role="option"
                          aria-selected={isSelected}
                        >
                          <span className={`${isSelected ? "text-primary" : "text-text-subtle"}`}>
                            {cmd.icon}
                          </span>
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate">{cmd.label}</div>
                            {cmd.description && (
                              <div className="text-xs text-text-subtle truncate">
                                {cmd.description}
                              </div>
                            )}
                          </div>
                          {cmd.shortcut && (
                            <kbd className="ml-2 flex items-center gap-0.5 rounded border border-border bg-surface2 px-1.5 py-0.5 text-xs text-text-subtle">
                              {formatShortcut(cmd.shortcut)}
                            </kbd>
                          )}
                          {isSelected && (
                            <ArrowRight className="size-4 text-primary" />
                          )}
                        </button>
                      )
                    })}
                  </div>
                )
              })}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-text-subtle">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-surface2 px-1">
                ↑
              </kbd>
              <kbd className="rounded border border-border bg-surface2 px-1">
                ↓
              </kbd>
              <span className="ml-1">{t("common:commandPalette.navigate", "navigate")}</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="rounded border border-border bg-surface2 px-1">
                ↵
              </kbd>
              <span className="ml-1">{t("common:commandPalette.select", "select")}</span>
            </span>
          </div>
          <span className="flex items-center gap-1">
            <Command className="size-3" />
            <span>K</span>
            <span className="ml-1">{t("common:commandPalette.toOpen", "to open")}</span>
          </span>
        </div>
      </div>
    </>
  )

  return createPortal(modalContent, document.body)
}

export default CommandPalette
