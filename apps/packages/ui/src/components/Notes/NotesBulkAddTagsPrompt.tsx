import React from 'react'
import { Modal, Select, Typography } from 'antd'
import type { ModalFuncProps } from 'antd'
import { normalizeTagList } from './notes-manager-utils'

type Translate = (key: string, options?: Record<string, unknown>) => string

/**
 * The context-aware modal API from `App.useApp()` (see `useAntdModal`). Its
 * dialogs render inside the app's ConfigProvider, so they follow the theme.
 */
export type BulkAddTagsModalApi =
  | { confirm?: (config: ModalFuncProps) => unknown }
  | null
  | undefined

export type BulkAddTagsPromptOptions = {
  noteCount: number
  /** Existing tags offered as suggestions; new tags can still be typed. */
  suggestions: readonly string[]
  renderSuggestionLabel?: (tag: string) => React.ReactNode
  t: Translate
}

type BulkAddTagsFieldProps = BulkAddTagsPromptOptions & {
  onChange: (tags: string[]) => void
}

const BulkAddTagsField: React.FC<BulkAddTagsFieldProps> = ({
  noteCount,
  suggestions,
  renderSuggestionLabel,
  t,
  onChange
}) => {
  const [value, setValue] = React.useState<string[]>([])
  const effectId = React.useId()
  const options = React.useMemo(
    () =>
      suggestions.map((tag) => ({
        value: tag,
        label: renderSuggestionLabel ? renderSuggestionLabel(tag) : tag
      })),
    [renderSuggestionLabel, suggestions]
  )

  return (
    <div className="mt-2 space-y-2">
      <Typography.Paragraph
        id={effectId}
        className="!mb-0 text-xs text-text-muted"
        data-testid="notes-bulk-add-tags-effect"
      >
        {noteCount === 1
          ? t('option:notesSearch.bulkAddTagsEffectOne', {
              defaultValue: 'Adds these tags to the selected note. Its existing tags are kept.'
            })
          : t('option:notesSearch.bulkAddTagsEffectMany', {
              defaultValue:
                'Adds these tags to {{count}} selected notes. Their existing tags are kept.',
              count: noteCount
            })}
      </Typography.Paragraph>
      <Select
        mode="tags"
        autoFocus
        className="w-full"
        value={value}
        tokenSeparators={[',']}
        placeholder={t('option:notesSearch.bulkAddTagsPlaceholder', {
          defaultValue: 'Choose or type tags'
        })}
        aria-label={t('option:notesSearch.bulkAddTagsAriaLabel', {
          defaultValue: 'Tags to add'
        })}
        aria-describedby={effectId}
        data-testid="notes-bulk-add-tags-select"
        options={options}
        onChange={(next) => {
          const tags = Array.isArray(next) ? next.map(String) : []
          setValue(tags)
          onChange(tags)
        }}
      />
    </div>
  )
}

/**
 * Ask which tags to add to the selected notes. Resolves to the normalized
 * tags (possibly empty) on confirm, or null when cancelled.
 */
export function promptBulkAddTags(
  modal: BulkAddTagsModalApi,
  options: BulkAddTagsPromptOptions
): Promise<string[] | null> {
  const { t } = options
  // The static Modal.confirm cannot read the app theme; it is only a fallback
  // for trees rendered without antd's <App> (for example isolated tests).
  const confirm =
    typeof modal?.confirm === 'function' ? modal.confirm : Modal.confirm

  return new Promise((resolve) => {
    let currentTags: string[] = []
    let settled = false
    const finish = (value: string[] | null) => {
      if (settled) return
      settled = true
      resolve(value)
    }

    confirm({
      title: t('option:notesSearch.bulkAddTagsTitle', {
        defaultValue: 'Add tags to selected notes'
      }),
      icon: null,
      content: (
        <BulkAddTagsField
          noteCount={options.noteCount}
          suggestions={options.suggestions}
          renderSuggestionLabel={options.renderSuggestionLabel}
          t={t}
          onChange={(tags) => {
            currentTags = tags
          }}
        />
      ),
      okText: t('option:notesSearch.bulkAddTagsConfirm', { defaultValue: 'Add tags' }),
      cancelText: t('common:cancel', { defaultValue: 'Cancel' }),
      // Leave focus on the tag field instead of the OK button.
      focusable: { autoFocusButton: null },
      onOk: () => finish(normalizeTagList(currentTags)),
      onCancel: () => finish(null)
    })
  })
}
