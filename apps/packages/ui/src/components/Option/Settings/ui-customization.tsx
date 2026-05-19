import React from "react"
import { Checkbox, message } from "antd"
import { useTranslation } from "react-i18next"
import { cn } from "@/libs/utils"
import { useSetting } from "@/hooks/useSetting"
import {
  PERSONA_BUDDY_SHELL_ENABLED_SETTING,
  SIDEBAR_SHORTCUT_MAX_COUNT,
  DEFAULT_SIDEBAR_SHORTCUT_SELECTION,
  SIDEBAR_SHORTCUT_SELECTION_SETTING,
  DEFAULT_HEADER_SHORTCUT_SELECTION,
  HEADER_SHORTCUT_SELECTION_SETTING,
  type HeaderShortcutId,
  type SidebarShortcutId
} from "@/services/settings/ui-settings"
import { SIDEBAR_SHORTCUT_ACTIONS } from "@/components/Common/ChatSidebar/shortcut-actions"
import {
  getHeaderShortcutGroups,
  getHeaderShortcutItems
} from "@/components/Layouts/header-shortcut-items"

const arraysEqual = <T,>(a: T[], b: T[]) =>
  a.length === b.length && a.every((value, index) => value === b[index])

const orderShortcutSelection = (selection: SidebarShortcutId[]) => {
  const selected = new Set(selection)
  return SIDEBAR_SHORTCUT_ACTIONS.filter((item) => selected.has(item.id)).map(
    (item) => item.id
  )
}

const orderHeaderShortcutSelection = (selection: HeaderShortcutId[]) => {
  const selected = new Set(selection)
  return getHeaderShortcutItems().filter((item) => selected.has(item.id)).map(
    (item) => item.id
  )
}

type SidebarShortcutSelectorProps = {
  title: string
  description: string
  selection: SidebarShortcutId[]
  defaultSelection: SidebarShortcutId[]
  onChange: (next: SidebarShortcutId[]) => void
  maxCount: number
}

const SidebarShortcutSelector = ({
  title,
  description,
  selection,
  defaultSelection,
  onChange,
  maxCount
}: SidebarShortcutSelectorProps) => {
  const { t } = useTranslation(["settings", "common", "option"])
  const normalizedSelection = React.useMemo(
    () => orderShortcutSelection(selection),
    [selection]
  )
  const selectionSet = React.useMemo(
    () => new Set(normalizedSelection),
    [normalizedSelection]
  )
  const selectionCount = normalizedSelection.length
  const isModified = !arraysEqual(normalizedSelection, defaultSelection)

  const handleToggle = (id: SidebarShortcutId) => {
    if (selectionSet.has(id)) {
      const next = orderShortcutSelection(
        normalizedSelection.filter((item) => item !== id)
      )
      onChange(next)
      return
    }
    if (selectionCount >= maxCount) {
      message.warning(
        t("uiCustomization.shortcuts.maxReached", {
          defaultValue: "You can select up to {{max}} shortcuts.",
          max: maxCount
        })
      )
      return
    }
    const next = orderShortcutSelection([...normalizedSelection, id])
    onChange(next)
  }

  return (
    <div className="panel-card p-4 space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          <p className="text-xs text-text-muted">{description}</p>
        </div>
        {isModified && (
          <button
            type="button"
            onClick={() => onChange(defaultSelection)}
            className="text-xs text-primary hover:underline"
          >
            {t("uiCustomization.shortcuts.reset", {
              defaultValue: "Reset to default"
            })}
          </button>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-text-muted">
        <span>
          {t("uiCustomization.shortcuts.countLabel", {
            defaultValue: "{{count}} / {{max}} selected",
            count: selectionCount,
            max: maxCount
          })}
        </span>
        {selectionCount >= maxCount && (
          <span>
            {t("uiCustomization.shortcuts.maxReached", {
              defaultValue: "You can select up to {{max}} shortcuts.",
              max: maxCount
            })}
          </span>
        )}
      </div>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {SIDEBAR_SHORTCUT_ACTIONS.map((item) => {
          const selected = selectionSet.has(item.id)
          const isDisabled = !selected && selectionCount >= maxCount
          return (
            <div
              key={item.id}
              className={cn(
                "rounded-lg border px-3 py-2 transition-colors",
                selected
                  ? "border-primary/60 bg-surface2"
                  : "border-border bg-surface",
                isDisabled && "opacity-50"
              )}
            >
              <Checkbox
                checked={selected}
                disabled={isDisabled}
                onChange={() => handleToggle(item.id)}
                className="w-full"
              >
                <div className="flex items-center gap-2 text-sm text-text">
                  <item.icon className="size-4" />
                  <span>{t(item.labelKey, item.labelDefault)}</span>
                </div>
              </Checkbox>
            </div>
          )
        })}
      </div>
    </div>
  )
}

type HeaderShortcutSelectorProps = {
  title: string
  description: string
  selection: HeaderShortcutId[]
  defaultSelection: HeaderShortcutId[]
  onChange: (next: HeaderShortcutId[]) => void
}

const HeaderShortcutSelector = ({
  title,
  description,
  selection,
  defaultSelection,
  onChange
}: HeaderShortcutSelectorProps) => {
  const { t } = useTranslation(["settings", "option", "common"])
  const normalizedSelection = React.useMemo(
    () => orderHeaderShortcutSelection(selection),
    [selection]
  )
  const selectionSet = React.useMemo(
    () => new Set(normalizedSelection),
    [normalizedSelection]
  )
  const selectionCount = normalizedSelection.length
  const totalCount = getHeaderShortcutItems().length
  const isModified = !arraysEqual(normalizedSelection, defaultSelection)

  const handleToggle = (id: HeaderShortcutId) => {
    if (selectionSet.has(id)) {
      const next = orderHeaderShortcutSelection(
        normalizedSelection.filter((item) => item !== id)
      )
      onChange(next)
      return
    }
    const next = orderHeaderShortcutSelection([...normalizedSelection, id])
    onChange(next)
  }

  return (
    <div className="panel-card p-4 space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold text-text">{title}</h3>
          <p className="text-xs text-text-muted">{description}</p>
        </div>
        {isModified && (
          <button
            type="button"
            onClick={() => onChange(defaultSelection)}
            className="text-xs text-primary hover:underline"
          >
            {t("uiCustomization.headerShortcuts.reset", {
              defaultValue: "Reset to default"
            })}
          </button>
        )}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-text-muted">
        <span>
          {t("uiCustomization.headerShortcuts.countLabel", {
            defaultValue: "{{count}} of {{total}} selected",
            count: selectionCount,
            total: totalCount
          })}
        </span>
      </div>
      <div className="space-y-3">
        {getHeaderShortcutGroups().map((group) => (
          <div
            key={group.id}
            className="rounded-lg border border-border/60 bg-surface px-3 py-3"
          >
            <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
              {t(group.titleKey, group.titleDefault)}
            </div>
            <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {group.items.map((item) => {
                const selected = selectionSet.has(item.id)
                return (
                  <div
                    key={item.id}
                    className={cn(
                      "rounded-md border px-2 py-2 transition-colors",
                      selected
                        ? "border-primary/60 bg-surface2"
                        : "border-border bg-surface"
                    )}
                  >
                    <Checkbox
                      checked={selected}
                      onChange={() => handleToggle(item.id)}
                      className="w-full"
                    >
                      <div className="flex items-center gap-2 text-sm text-text">
                        <item.icon className="size-4" />
                        <span>{t(item.labelKey, item.labelDefault)}</span>
                      </div>
                    </Checkbox>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export const UiCustomizationSettings = () => {
  const { t } = useTranslation(["settings", "common"])
  const [personaBuddyShellEnabled, setPersonaBuddyShellEnabled] = useSetting(
    PERSONA_BUDDY_SHELL_ENABLED_SETTING
  )
  const [shortcutSelection, setShortcutSelection] = useSetting(
    SIDEBAR_SHORTCUT_SELECTION_SETTING
  )
  const [headerShortcutSelection, setHeaderShortcutSelection] = useSetting(
    HEADER_SHORTCUT_SELECTION_SETTING
  )

  const shortcutSelectionNormalized = React.useMemo(
    () => orderShortcutSelection(shortcutSelection),
    [shortcutSelection]
  )
  const headerShortcutSelectionNormalized = React.useMemo(
    () => orderHeaderShortcutSelection(headerShortcutSelection),
    [headerShortcutSelection]
  )

  return (
    <div className="flex flex-col space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("uiCustomization.title", { defaultValue: "UI customization" })}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t("uiCustomization.subtitle", {
            defaultValue:
              "Personalize the navigation shortcuts and layout defaults."
          })}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <div className="space-y-4">
        <div className="panel-card p-4">
          <Checkbox
            checked={personaBuddyShellEnabled}
            onChange={(event) =>
              void setPersonaBuddyShellEnabled(event.target.checked)
            }
            className="w-full"
          >
            <div className="space-y-1">
              <div className="text-sm font-semibold text-text">
                {t("uiCustomization.personaBuddyShell.label", {
                  defaultValue: "Enable persona buddy shell"
                })}
              </div>
              <div className="text-xs text-text-muted">
                {t("uiCustomization.personaBuddyShell.description", {
                  defaultValue:
                    "Show the floating buddy shell on supported persona-aware surfaces."
                })}
              </div>
            </div>
          </Checkbox>
        </div>
        <SidebarShortcutSelector
          title={t("uiCustomization.shortcuts.title", {
            defaultValue: "Sidebar shortcuts"
          })}
          description={t("uiCustomization.shortcuts.description", {
            defaultValue:
              "Choose up to 10 items for the sidebar icons and shortcuts list."
          })}
          selection={shortcutSelectionNormalized}
          defaultSelection={DEFAULT_SIDEBAR_SHORTCUT_SELECTION}
          onChange={(next) => void setShortcutSelection(next)}
          maxCount={SIDEBAR_SHORTCUT_MAX_COUNT}
        />
        <HeaderShortcutSelector
          title={t("uiCustomization.headerShortcuts.title", {
            defaultValue: "Playground shortcuts"
          })}
          description={t("uiCustomization.headerShortcuts.description", {
            defaultValue:
              "Choose which shortcuts appear in the header shortcuts panel."
          })}
          selection={headerShortcutSelectionNormalized}
          defaultSelection={DEFAULT_HEADER_SHORTCUT_SELECTION}
          onChange={(next) => void setHeaderShortcutSelection(next)}
        />
      </div>
    </div>
  )
}

export default UiCustomizationSettings
