/**
 * PageHelpModal Component
 * Unified help modal with tabs for Tutorials and Keyboard Shortcuts
 * Triggered by pressing the ? key
 */

import React, { useState, useEffect, useCallback, useRef, useMemo } from "react"
import { createPortal } from "react-dom"
import { useTranslation } from "react-i18next"
import i18n from "i18next"
import tutorialsEn from "@/assets/locale/en/tutorials.json"
import { useLocation } from "react-router-dom"
import {
  X,
  Keyboard,
  GraduationCap,
  Check,
  Play,
  RotateCcw,
  Command,
  Lock
} from "lucide-react"
import { useShortcutConfig, formatShortcut } from "@/hooks/keyboard/useShortcutConfig"
import { isMac } from "@/hooks/keyboard/useKeyboardShortcuts"
import { useStartTutorial } from "@/hooks/useStartTutorial"
import { FEATURE_FLAGS, useFeatureFlag } from "@/hooks/useFeatureFlags"
import { useHelpModal, useTutorialStore } from "@/store/tutorials"
import {
  getTutorialsForRoute,
  type TutorialDefinition
} from "@/tutorials"

// Register the English fallback only when this demand-loaded help surface arrives.
const registerTutorialTranslations = () => {
  i18n.addResourceBundle("en", "tutorials", tutorialsEn, true, false)
  i18n.off("initialized", registerTutorialTranslations)
}
if (i18n.isInitialized) registerTutorialTranslations()
else i18n.on("initialized", registerTutorialTranslations)

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface ShortcutGroup {
  title: string
  shortcuts: {
    label: string
    keys: string
  }[]
}

type TabKey = "tutorials" | "shortcuts"

// ─────────────────────────────────────────────────────────────────────────────
// Sub-Components
// ─────────────────────────────────────────────────────────────────────────────

interface TutorialListProps {
  tutorials: TutorialDefinition[]
  completedTutorials: string[]
  onStart: (tutorialId: string) => void
}

const TutorialList: React.FC<TutorialListProps> = ({
  tutorials,
  completedTutorials,
  onStart
}) => {
  const { t } = useTranslation(["tutorials", "common"])
  const completedSet = new Set(completedTutorials)

  if (tutorials.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center">
        <GraduationCap className="mb-3 size-10 text-text-subtle opacity-50" />
        <p className="text-sm text-text-muted">
          {t("tutorials:empty.noTutorials", "No tutorials available for this page.")}
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {tutorials.map((tutorial) => {
        const isCompleted = completedSet.has(tutorial.id)
        const Icon = tutorial.icon || GraduationCap

        // Check if prerequisites are met
        const unmetPrereqs = (tutorial.prerequisites ?? []).filter(
          (prereqId) => !completedSet.has(prereqId)
        )
        const isLocked = unmetPrereqs.length > 0

        return (
          <div
            key={tutorial.id}
            className={`flex items-start gap-3 rounded-lg border border-border p-3 ${
              isLocked
                ? "bg-surface2/30 opacity-70"
                : "bg-surface2/50 hover:bg-surface2"
            }`}
          >
            <div
              className={`mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg ${
                isLocked
                  ? "bg-surface2 text-text-subtle"
                  : isCompleted
                    ? "bg-success/10 text-success"
                    : "bg-primary/10 text-primary"
              }`}
            >
              {isLocked ? (
                <Lock className="size-4" />
              ) : isCompleted ? (
                <Check className="size-4" />
              ) : (
                <Icon className="size-4" />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-medium text-text">
                  {t(tutorial.labelKey, tutorial.labelFallback)}
                </h4>
                {isCompleted && (
                  <span className="rounded bg-success/10 px-1.5 py-0.5 text-xs font-medium text-success">
                    {t("tutorials:status.completed", "Completed")}
                  </span>
                )}
              </div>
              <p className="mt-0.5 text-xs text-text-muted line-clamp-2">
                {t(tutorial.descriptionKey, tutorial.descriptionFallback)}
              </p>
              <div className="mt-2 flex items-center gap-2">
                <span className="text-xs text-text-subtle">
                  {t("tutorials:steps", { count: tutorial.steps.length })}
                </span>
                {isLocked && (
                  <span className="text-xs text-text-subtle">
                    {t("tutorials:status.locked", "Complete basics first")}
                  </span>
                )}
              </div>
            </div>

            <button
              onClick={() => onStart(tutorial.id)}
              disabled={isLocked}
              className={`min-h-[44px] shrink-0 flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${
                isLocked || isCompleted
                  ? "text-text hover:bg-surface2 transition-colors"
                  : "bg-primary text-primary-foreground hover:bg-primaryStrong hover:text-primary-strong-foreground transition-none"
              }`}>
              {isLocked ? (
                <>
                  <Lock className="size-3.5" />
                  {t("tutorials:actions.locked", "Locked")}
                </>
              ) : isCompleted ? (
                <>
                  <RotateCcw className="size-3.5" />
                  {t("tutorials:actions.replay", "Replay")}
                </>
              ) : (
                <>
                  <Play className="size-3.5" />
                  {t("tutorials:actions.start", "Start")}
                </>
              )}
            </button>
          </div>
        )
      })}
    </div>
  )
}

interface ShortcutsListProps {
  groups: ShortcutGroup[]
}

const ShortcutsList: React.FC<ShortcutsListProps> = ({ groups }) => {
  return (
    <div className="space-y-4">
      {groups.map((group, groupIndex) => (
        <div key={group.title} className={groupIndex > 0 ? "mt-5" : ""}>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-subtle">
            {group.title}
          </h3>
          <div className="space-y-1">
            {group.shortcuts.map((shortcut) => (
              <div
                key={shortcut.label}
                className="flex items-center justify-between rounded-lg px-2 py-1.5 hover:bg-surface2"
              >
                <span className="text-sm text-text">{shortcut.label}</span>
                <kbd className="ml-4 flex items-center gap-1 rounded border border-border bg-surface2 px-2 py-0.5 text-xs font-medium text-text-muted">
                  {shortcut.keys}
                </kbd>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

export type PageHelpModalProps = {
  initialTab?: TabKey
  scope?: "full" | "sidepanel"
  modeNavigationEnabled?: boolean
  sidebarToggleEnabled?: boolean
  quickChatEnabled?: boolean
  onStartTutorial?: (tutorialId: string) => boolean
}

export function PageHelpModal({
  initialTab,
  scope = "full",
  modeNavigationEnabled = true,
  sidebarToggleEnabled = true,
  quickChatEnabled = true,
  onStartTutorial
}: PageHelpModalProps = {}) {
  const { isOpen, close } = useHelpModal()
  const startTutorial = useStartTutorial()
  const completedTutorials = useTutorialStore(
    (state) => state.completedTutorials
  )

  const modalRef = useRef<HTMLDivElement | null>(null)
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  const { t } = useTranslation(["common", "tutorials"])
  const location = useLocation()
  const { shortcuts } = useShortcutConfig()
  // Compare stays hidden until it is verified (owner decision D4).
  const [compareEnabled] = useFeatureFlag(FEATURE_FLAGS.COMPARE_MODE)

  // Get tutorials for current route
  const availableTutorials = useMemo(
    () =>
      getTutorialsForRoute(location.pathname, {
        completedTutorialIds: completedTutorials,
        includeLocked: true
      }),
    [location.pathname, completedTutorials]
  )

  // Determine default tab based on available tutorials
  const defaultTab: TabKey =
    availableTutorials.length > 0 ? "tutorials" : "shortcuts"
  const [activeTab, setActiveTab] = useState<TabKey>(initialTab ?? defaultTab)

  useEffect(() => {
    setActiveTab(initialTab ?? defaultTab)
  }, [initialTab, defaultTab, location.pathname])

  useEffect(() => {
    if (isOpen) closeButtonRef.current?.focus()
  }, [isOpen])

  const modKey = isMac ? "⌘" : "Ctrl"

  // Build shortcut groups
  const shortcutGroups: ShortcutGroup[] = useMemo(
    () => [
      {
        title: t("common:shortcuts.groups.general", "General"),
        shortcuts: [
          {
            label: t(
              "common:shortcuts.openCommandPalette",
              "Open command palette"
            ),
            keys: `${modKey} + K`
          },
          {
            label: t(
              "common:shortcuts.showKeyboardShortcuts",
              "Show keyboard shortcuts"
            ),
            keys: "?"
          },
          {
            label: t("common:shortcuts.focusTextarea", "Focus message input"),
            keys: formatShortcut(shortcuts.focusTextarea)
          },
          ...(sidebarToggleEnabled
            ? [
                {
                  label: t("common:shortcuts.toggleSidebar", "Toggle sidebar"),
                  keys: formatShortcut(shortcuts.toggleSidebar)
                }
              ]
            : [])
        ]
      },
      {
        title: t("common:shortcuts.groups.chat", "Chat"),
        shortcuts: [
          {
            label: t("common:shortcuts.newChat", "Start new chat"),
            keys: formatShortcut(shortcuts.newChat)
          },
          ...(scope === "sidepanel"
            ? [
                {
                  label: t(
                    "common:shortcuts.toggleChatWithPage",
                    "Toggle chat with current page"
                  ),
                  keys: formatShortcut(shortcuts.toggleChatMode)
                },
                {
                  label: t(
                    "common:shortcuts.toggleWebSearch",
                    "Toggle web search"
                  ),
                  keys: formatShortcut(shortcuts.toggleWebSearch)
                }
              ]
            : [
                ...(quickChatEnabled
                  ? [
                      {
                        label: t(
                          "common:shortcuts.toggleQuickChat",
                          "Toggle Quick Chat Helper"
                        ),
                        keys: formatShortcut(shortcuts.toggleQuickChatHelper)
                      }
                    ]
                  : []),
                {
                  label: t(
                    "common:shortcuts.searchThread",
                    "Search this conversation"
                  ),
                  keys: `${modKey} + F`
                },
                {
                  label: t(
                    "common:shortcuts.toggleArtifacts",
                    "Toggle artifacts panel (when available)"
                  ),
                  keys: "Alt + Shift + A"
                },
                ...(compareEnabled
                  ? [
                      {
                        label: t(
                          "common:shortcuts.toggleCompare",
                          "Toggle compare mode (when enabled)"
                        ),
                        keys: "Alt + Shift + C"
                      }
                    ]
                  : []),
                {
                  label: t(
                    "common:shortcuts.toggleModes",
                    "Open mode launcher"
                  ),
                  keys: "Alt + Shift + M"
                }
              ])
        ]
      },
      {
        title: t("common:shortcuts.groups.messageCards", "Message cards"),
        shortcuts: [
          {
            label: t(
              "common:shortcuts.variantSwitch",
              "Previous / next response variant (focus a reply)"
            ),
            keys: "Alt + Shift + ← / →"
          },
          {
            label: t(
              "common:shortcuts.branchMessage",
              "Fork from the focused message"
            ),
            keys: "Alt + Shift + B"
          },
          {
            label: t(
              "common:shortcuts.regenerateLast",
              "Regenerate the last reply (focus a reply)"
            ),
            keys: "Alt + Shift + R"
          }
        ]
      },
      ...(scope === "full" && modeNavigationEnabled
        ? [
            {
              title: t("common:shortcuts.groups.navigation", "Navigation"),
              shortcuts: [
                {
                  label: t("common:shortcuts.goToChat", "Go to Chat"),
                  keys: formatShortcut(shortcuts.modePlayground)
                },
                {
                  label: t("common:shortcuts.goToSources", "Go to Sources"),
                  keys: formatShortcut(shortcuts.modeSources)
                },
                {
                  label: t("common:shortcuts.goToMedia", "Go to Media"),
                  keys: formatShortcut(shortcuts.modeMedia)
                },
                {
                  label: t("common:shortcuts.goToKnowledge", "Go to Knowledge"),
                  keys: formatShortcut(shortcuts.modeKnowledge)
                },
                {
                  label: t("common:shortcuts.goToNotes", "Go to Notes"),
                  keys: formatShortcut(shortcuts.modeNotes)
                },
                {
                  label: t("common:shortcuts.goToPrompts", "Go to Prompts"),
                  keys: formatShortcut(shortcuts.modePrompts)
                },
                {
                  label: t(
                    "common:shortcuts.goToFlashcards",
                    "Go to Flashcards"
                  ),
                  keys: formatShortcut(shortcuts.modeFlashcards)
                },
                {
                  label: t(
                    "common:shortcuts.goToWorldBooks",
                    "Go to World Books"
                  ),
                  keys: formatShortcut(shortcuts.modeWorldBooks)
                },
                {
                  label: t(
                    "common:shortcuts.goToDictionaries",
                    "Go to Dictionaries"
                  ),
                  keys: formatShortcut(shortcuts.modeDictionaries)
                },
                {
                  label: t(
                    "common:shortcuts.goToCharacters",
                    "Go to Characters"
                  ),
                  keys: formatShortcut(shortcuts.modeCharacters)
                }
              ]
            }
          ]
        : [])
    ],
    [
      modKey,
      t,
      shortcuts,
      scope,
      modeNavigationEnabled,
      sidebarToggleEnabled,
      quickChatEnabled,
      compareEnabled
    ]
  )

  const handleClose = useCallback(() => {
    close()
  }, [close])

  const handleStartTutorial = useCallback(
    (tutorialId: string) => {
      const accepted = onStartTutorial
        ? onStartTutorial(tutorialId)
        : startTutorial(tutorialId)
      if (accepted) close()
    },
    [close, startTutorial, onStartTutorial]
  )

  const handleKeyDownInModal = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== "Tab" || !modalRef.current) {
        return
      }

      const focusableSelectors =
        'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
      const focusableElements = Array.from(
        modalRef.current.querySelectorAll<HTMLElement>(focusableSelectors)
      ).filter(
        (element) =>
          !element.hasAttribute("disabled") &&
          element.getAttribute("aria-hidden") !== "true" &&
          element.tabIndex !== -1
      )

      if (focusableElements.length === 0) {
        event.preventDefault()
        return
      }

      const currentIndex = focusableElements.indexOf(
        document.activeElement as HTMLElement
      )
      let nextIndex = currentIndex

      if (event.shiftKey) {
        nextIndex =
          currentIndex <= 0 ? focusableElements.length - 1 : currentIndex - 1
      } else {
        nextIndex =
          currentIndex === -1 || currentIndex === focusableElements.length - 1
            ? 0
            : currentIndex + 1
      }

      focusableElements[nextIndex].focus()
      event.preventDefault()
    },
    []
  )

  if (!isOpen) return null
  if (typeof document === "undefined") return null

  const tutorialCount = availableTutorials.length

  const modalContent = (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-bg/70 backdrop-blur-sm"
        onClick={handleClose}
        aria-hidden="true"
      />

      {/* Modal */}
      <div
        ref={modalRef}
        className="fixed left-1/2 top-[10%] sm:top-[15%] z-50 w-[calc(100%-2rem)] sm:w-full max-w-lg -translate-x-1/2 overflow-hidden rounded-xl border border-border bg-surface shadow-2xl max-h-[80vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-modal-title"
        onKeyDown={handleKeyDownInModal}>
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <GraduationCap className="size-5 text-text-subtle" />
            <h2
              id="help-modal-title"
              className="text-base font-semibold text-text">
              {t("common:help.pageHelp", "Page Help")}
            </h2>
          </div>
          <button
            ref={closeButtonRef}
            onClick={handleClose}
            className="rounded-lg p-2 min-w-[44px] min-h-[44px] flex items-center justify-center text-text-subtle hover:bg-surface2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus"
            aria-label={t("common:close", "Close")}>
            <X className="size-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          <button
            onClick={() => setActiveTab("tutorials")}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === "tutorials"
                ? "border-b-2 border-primary text-primary"
                : "text-text-muted hover:text-text"
            }`}>
            <GraduationCap className="size-4" />
            {t("common:help.tutorials", "Tutorials")}
            {tutorialCount > 0 && (
              <span
                className={`ml-1 flex size-5 items-center justify-center rounded-full text-xs font-medium ${
                  activeTab === "tutorials"
                    ? "bg-primary text-primary-foreground"
                    : "bg-surface2 text-text-muted"
                }`}>
                {tutorialCount}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("shortcuts")}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors ${
              activeTab === "shortcuts"
                ? "border-b-2 border-primary text-primary"
                : "text-text-muted hover:text-text"
            }`}>
            <Keyboard className="size-4" />
            {t("common:help.shortcuts", "Shortcuts")}
          </button>
        </div>

        {/* Content: a focusable, labelled region so keyboard users can
            scroll a list that has no focusable items of its own (AX-18). */}
        <div
          role="region"
          aria-label={
            activeTab === "tutorials"
              ? t("common:help.tutorialList", "Tutorial list")
              : t("common:help.shortcutList", "Shortcut list")
          }
          tabIndex={0}
          className="flex-1 overflow-y-auto p-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus">
          {activeTab === "tutorials" ? (
            <TutorialList
              tutorials={availableTutorials}
              completedTutorials={completedTutorials}
              onStart={handleStartTutorial}
            />
          ) : (
            <ShortcutsList groups={shortcutGroups} />
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-border px-4 py-2.5">
          <p className="text-xs text-text-subtle">
            {activeTab === "tutorials"
              ? t(
                  "common:help.tutorialsHint",
                  "Complete tutorials to learn the interface"
                )
              : t(
                  "common:shortcuts.customizeHint",
                  "Shortcuts can be customized in Settings"
                )}
          </p>
          <div className="flex items-center gap-1 text-xs text-text-subtle">
            <Command className="size-3" />
            <span>K</span>
            <span className="ml-1">
              {t("common:shortcuts.forCommands", "for commands")}
            </span>
          </div>
        </div>
      </div>
    </>
  )

  return createPortal(modalContent, document.body)
}

export default PageHelpModal
