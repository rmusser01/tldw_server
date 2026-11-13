import React from "react"

/** The id of a suggestion option, shared by a menu and the textarea that points at it. */
export const comboboxOptionId = (listboxId: string, index: number) =>
  `${listboxId}-option-${index}`

type ComposerComboboxOptions = {
  /** The suggestion menu is showing. */
  open: boolean
  /** How many suggestions it lists. */
  count: number
  /** The highlighted suggestion. */
  activeIndex: number
}

/**
 * A textarea with inline suggestions (slash commands, @mentions, [[wikilinks]])
 * as an ARIA 1.2 combobox (AX-12). Focus stays in the textarea; arrowing moves
 * `aria-activedescendant` through the listbox's options, so a screen reader
 * reads each one. Render the menu's listbox with `listboxId` and its options
 * with `comboboxOptionId(listboxId, index)`.
 */
export const useComposerCombobox = ({
  open,
  count,
  activeIndex
}: ComposerComboboxOptions) => {
  const listboxId = `composer-suggestions-${React.useId().replace(/[^A-Za-z0-9_-]/g, "")}`
  const expanded = open && count > 0
  const activeDescendant =
    expanded && activeIndex >= 0 && activeIndex < count
      ? comboboxOptionId(listboxId, activeIndex)
      : undefined
  return {
    listboxId,
    inputProps: {
      role: "combobox" as const,
      "aria-autocomplete": "list" as const,
      "aria-expanded": expanded,
      "aria-controls": expanded ? listboxId : undefined,
      "aria-activedescendant": activeDescendant
    }
  }
}
