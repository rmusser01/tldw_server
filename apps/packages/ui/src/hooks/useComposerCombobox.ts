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
 * that a screen reader can follow (AX-12). The textarea keeps its textbox
 * role: a textarea may not be a combobox, and the Notes editor must not read
 * as one. It points into the menu with attributes a textbox supports:
 * aria-autocomplete, aria-controls while the menu shows, and
 * aria-activedescendant on the highlighted option. Focus stays in the
 * textarea. Render the menu's listbox with `listboxId` and its options with
 * `comboboxOptionId(listboxId, index)`.
 */
export const useComposerCombobox = ({
  open,
  count,
  activeIndex
}: ComposerComboboxOptions) => {
  const listboxId = `composer-suggestions-${React.useId().replace(/[^A-Za-z0-9_-]/g, "")}`
  const showing = open && count > 0
  const activeDescendant =
    showing && activeIndex >= 0 && activeIndex < count
      ? comboboxOptionId(listboxId, activeIndex)
      : undefined
  return {
    listboxId,
    inputProps: {
      "aria-autocomplete": "list" as const,
      "aria-controls": showing ? listboxId : undefined,
      "aria-activedescendant": activeDescendant
    }
  }
}
