import React from 'react'
import { useTranslation } from 'react-i18next'
import type { WikilinkCandidate } from '@/components/Notes/wikilinks'
import { comboboxOptionId } from '@/hooks/useComposerCombobox'
import { SuggestionCountStatus } from '@/components/Common/SuggestionCountStatus'

type WikilinkSuggestionListProps = {
  open: boolean
  listboxId: string
  suggestions: WikilinkCandidate[]
  displayCounts: Map<string, number>
  selectionIndex: number
  onApply: (candidate: WikilinkCandidate) => void
}

/**
 * The [[wikilink]] suggestions under the note editor: a listbox the editor
 * textarea points into (AX-12), with the number of matches read
 * when it opens. The keyboard stays in the textarea (useNotesWikilinks).
 */
export const WikilinkSuggestionList = ({
  open,
  listboxId,
  suggestions,
  displayCounts,
  selectionIndex,
  onApply
}: WikilinkSuggestionListProps) => {
  const { t } = useTranslation(['option'])
  const showList = open && suggestions.length > 0
  const countText = String(
    t('option:notesSearch.wikilinkSuggestionCount', {
      count: suggestions.length,
      defaultValue: '{{count}} notes',
      defaultValue_one: '{{count}} note'
    })
  )
  return (
    <>
      <SuggestionCountStatus open={showList} text={countText} />
      {showList && (
        <div
          id={listboxId}
          className="mt-2 rounded-lg border border-border bg-surface p-1"
          role="listbox"
          aria-label={t('option:notesSearch.wikilinkSuggestionsLabel', {
            defaultValue: 'Wikilink suggestions'
          })}
          data-testid="notes-wikilink-suggestions"
        >
          {suggestions.map((candidate, index) => {
            const duplicateCount = displayCounts.get(candidate.title.toLowerCase()) || 0
            const label =
              duplicateCount > 1 ? `${candidate.title} (${candidate.id})` : candidate.title
            return (
              <div
                key={`${candidate.id}-${candidate.title}`}
                id={comboboxOptionId(listboxId, index)}
                role="option"
                className={`block w-full cursor-pointer rounded px-2 py-1 text-left text-xs ${
                  index === selectionIndex
                    ? 'bg-surface2 text-text'
                    : 'text-text-muted hover:bg-surface2 hover:text-text'
                }`}
                aria-selected={index === selectionIndex}
                onMouseDown={(event) => {
                  event.preventDefault()
                  onApply(candidate)
                }}
                data-testid={`notes-wikilink-suggestion-${candidate.id.replace(/[^a-z0-9_-]/gi, '_')}`}
              >
                {label}
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}

export default WikilinkSuggestionList
