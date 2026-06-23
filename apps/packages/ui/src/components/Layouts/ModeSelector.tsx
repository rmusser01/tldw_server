import React from "react"
import { useTranslation } from "react-i18next"
import { Dropdown } from "antd"
import {
  useShortcutConfig,
  formatShortcut,
} from "@/hooks/keyboard/useShortcutConfig"
import type { KeyboardShortcut } from "@/hooks/keyboard/useKeyboardShortcuts"

const classNames = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(" ")

export type CoreMode =
  | "playground"
  | "media"
  | "mediaMulti"
  | "knowledge"
  | "notes"
  | "prompts"
  | "quiz"
  | "evaluations"
  | "speech"
  | "flashcards"
  | "documentation"
  | "chunkingPlayground"
  | "worldBooks"
  | "dictionaries"
  | "characters"
  | "watchlists"
  | "audioStudio"
  // Note: "promptStudio" mode has been unified with "prompts"

interface ModeSelectorProps {
  currentMode: CoreMode
  onModeChange: (mode: CoreMode) => void
}

/**
 * Mode selector tabs for switching between app modes.
 * Extracted from Header.tsx for better maintainability.
 */
export function ModeSelector({ currentMode, onModeChange }: ModeSelectorProps) {
  const { t, i18n } = useTranslation(["option", "common", "settings"])
  const { shortcuts: shortcutConfig } = useShortcutConfig()

  const primaryModes: Array<{
    key: CoreMode
    label: string
    shortcut?: KeyboardShortcut
  }> = [
    {
      key: "playground",
      label: t("option:header.modePlayground", "Chat"),
      shortcut: shortcutConfig.modePlayground,
    },
    {
      key: "notes",
      label: t("option:header.modeNotes", "Notes"),
      shortcut: shortcutConfig.modeNotes,
    },
    {
      key: "media",
      label: t("option:header.modeMedia", "Media"),
      shortcut: shortcutConfig.modeMedia,
    },
    {
      key: "flashcards",
      label: t("option:header.modeFlashcards", "Flashcards"),
      shortcut: shortcutConfig.modeFlashcards,
    },
    {
      key: "quiz",
      label: t("option:header.quiz", "Quizzes"),
      shortcut: undefined,
    },
    {
      key: "prompts",
      label: t("option:header.modePromptsPlayground", "Prompts"),
      shortcut: shortcutConfig.modePrompts,
    },
    {
      key: "chunkingPlayground",
      label: t("settings:chunkingPlayground.nav", "Chunking Playground"),
      shortcut: undefined,
    },
  ]

  const secondaryModes: Array<{
    key: CoreMode
    label: string
    shortcut?: KeyboardShortcut
  }> = React.useMemo(
    () => [
      {
        key: "knowledge",
        label: t("option:header.modeKnowledge", "Knowledge QA"),
        shortcut: shortcutConfig.modeKnowledge,
      },
      {
        key: "mediaMulti",
        label: t("option:header.libraryView", "Multi-Item Review"),
        shortcut: undefined,
      },
      {
        key: "evaluations",
        label: t("settings:evaluationsSettingsNav", "Evaluations"),
        shortcut: undefined,
      },
      {
        key: "documentation",
        label: t("option:header.modeDocumentation", "Documentation"),
        shortcut: undefined,
      },
      {
        key: "speech",
        label: t("option:header.modeSpeech", "Speech"),
        shortcut: undefined,
      },
      // Note: Prompt Studio is now unified with Prompts (accessible via /prompts)
      {
        key: "worldBooks",
        label: t("option:header.modeWorldBooks", "World Books"),
        shortcut: shortcutConfig.modeWorldBooks,
      },
      {
        key: "dictionaries",
        label: t("option:header.modeDictionaries", "Chat Dictionaries"),
        shortcut: shortcutConfig.modeDictionaries,
      },
      {
        key: "characters",
        label: t("option:header.modeCharacters", "Characters"),
        shortcut: shortcutConfig.modeCharacters,
      },
      {
        key: "watchlists",
        label: t("option:header.modeWatchlists", "Watchlists"),
        shortcut: undefined,
      },
      {
        key: "audioStudio",
        label: t("option:header.audioStudio", "Audio Studio"),
        shortcut: undefined,
      },
    ],
    [i18n.language, shortcutConfig, t]
  )

  const dropdownItems = React.useMemo(
    () =>
      secondaryModes.map((mode) => ({
        key: mode.key,
        label: mode.label,
      })),
    [secondaryModes]
  )

  const handleDropdownClick = React.useCallback(
    ({ key }: { key: string }) => {
      onModeChange(key as CoreMode)
    },
    [onModeChange]
  )

  const dropdownMenu = React.useMemo(
    () => ({
      items: dropdownItems,
      onClick: handleDropdownClick,
    }),
    [dropdownItems, handleDropdownClick]
  )

  const renderModeButton = (mode: (typeof primaryModes)[0]) => {
    const isSelected = currentMode === mode.key

    return (
      <button
        key={mode.key}
        type="button"
        role="tab"
        aria-selected={isSelected}
        onClick={() => onModeChange(mode.key)}
        title={
          mode.shortcut
            ? (t("option:header.modeShortcutHint", "{{shortcut}} to switch", {
                shortcut: formatShortcut(mode.shortcut),
              }) as string) || undefined
            : mode.label
        }
        className={classNames(
          "core-mode-button rounded-full px-3 py-1 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus",
          isSelected
            ? "core-mode-button--active active bg-primary text-white shadow-sm"
            : "bg-surface2 text-text-muted hover:bg-surface"
        )}
        data-active={isSelected ? "true" : undefined}
      >
        {mode.label}
      </button>
    )
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="font-semibold uppercase tracking-wide text-text-muted">
        {t("option:header.modesLabel", "Modes")}
      </span>
      <div
        className="flex flex-wrap gap-1"
        role="tablist"
        aria-label={t("option:header.modesAriaLabel", "Application modes")}
      >
        {primaryModes.map(renderModeButton)}
        <Dropdown menu={dropdownMenu}>
          <button
            type="button"
            className={classNames(
              "core-mode-button rounded-full px-3 py-1 text-xs font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus",
              "bg-surface2 text-text-muted hover:bg-surface"
            )}
            title={t("option:header.moreTools", "More")}
          >
            {t("option:header.moreTools", "More")}
          </button>
        </Dropdown>
      </div>
    </div>
  )
}

export default ModeSelector
