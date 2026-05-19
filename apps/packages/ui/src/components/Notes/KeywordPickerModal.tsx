import React from 'react'
import type { TFunction } from 'i18next'
import { Button, Checkbox, Input, Modal, Typography } from 'antd'

type KeywordPickerModalProps = {
  open: boolean
  availableKeywords: string[]
  filteredKeywordPickerOptions: string[]
  recentKeywordPickerOptions: string[]
  keywordNoteCountByKey: Record<string, number>
  sortMode: 'frequency_desc' | 'alpha_asc' | 'alpha_desc'
  keywordPickerQuery: string
  keywordPickerSelection: string[]
  onCancel: () => void
  onApply: () => void
  onSortModeChange: (mode: 'frequency_desc' | 'alpha_asc' | 'alpha_desc') => void
  onToggleRecentKeyword: (keyword: string) => void
  onQueryChange: (value: string) => void
  onSelectionChange: (values: string[]) => void
  onSelectAll: () => void
  onClear: () => void
  onOpenManager?: () => void
  managerDisabled?: boolean
  t: TFunction
}

type KeywordFrequencyTone = 'none' | 'low' | 'medium' | 'high'

const KEYWORD_FREQUENCY_DOT_CLASS: Record<KeywordFrequencyTone, string> = {
  none: 'bg-border',
  low: 'bg-primary/35',
  medium: 'bg-primary/60',
  high: 'bg-primary'
}

const toKeywordTestIdSegment = (keyword: string) =>
  keyword.toLowerCase().replace(/[^a-z0-9_-]/g, '_')

const KeywordPickerModal: React.FC<KeywordPickerModalProps> = ({
  open,
  availableKeywords,
  filteredKeywordPickerOptions,
  recentKeywordPickerOptions,
  keywordNoteCountByKey,
  sortMode,
  keywordPickerQuery,
  keywordPickerSelection,
  onCancel,
  onApply,
  onSortModeChange,
  onToggleRecentKeyword,
  onQueryChange,
  onSelectionChange,
  onSelectAll,
  onClear,
  onOpenManager,
  managerDisabled = false,
  t
}) => {
  const maxKeywordNoteCount = React.useMemo(() => {
    let maxCount = 0
    for (const rawCount of Object.values(keywordNoteCountByKey)) {
      const count = Number(rawCount)
      if (!Number.isFinite(count)) continue
      if (count > maxCount) maxCount = count
    }
    return maxCount
  }, [keywordNoteCountByKey])

  const renderKeywordLabel = React.useCallback(
    (keyword: string, testIdPrefix: string) => {
      const count = keywordNoteCountByKey[keyword.toLowerCase()]
      let tone: KeywordFrequencyTone = 'none'
      if (typeof count === 'number' && count > 0 && maxKeywordNoteCount > 0) {
        const ratio = count / maxKeywordNoteCount
        tone = ratio >= 0.67 ? 'high' : ratio >= 0.34 ? 'medium' : 'low'
      }
      return (
        <span
          className="inline-flex items-center gap-1.5"
          data-frequency-tone={tone}
          data-testid={`${testIdPrefix}-${toKeywordTestIdSegment(keyword)}`}
        >
          <span
            className={`inline-block h-2 w-2 rounded-full ${KEYWORD_FREQUENCY_DOT_CLASS[tone]}`}
            aria-hidden="true"
          />
          <span>{typeof count === 'number' ? `${keyword} (${count})` : keyword}</span>
        </span>
      )
    },
    [keywordNoteCountByKey, maxKeywordNoteCount]
  )

  return (
    <Modal
      open={open}
      title={t('option:notesSearch.keywordPickerTitle', {
        defaultValue: 'Browse tags'
      })}
      aria-label={t('option:notesSearch.keywordPickerTitle', {
        defaultValue: 'Browse tags'
      })}
      onCancel={onCancel}
      onOk={onApply}
      okText={t('option:notesSearch.keywordPickerApply', {
        defaultValue: 'Apply filters'
      })}
      cancelText={t('common:cancel', { defaultValue: 'Cancel' })}
      keyboard
      destroyOnHidden
    >
      <div className="space-y-3" data-testid="notes-keyword-picker-modal">
      <Input
        allowClear
        placeholder={t('option:notesSearch.keywordPickerSearch', {
          defaultValue: 'Search tags'
        })}
        value={keywordPickerQuery}
        onChange={(e) => onQueryChange(e.target.value)}
      />
      <div className="space-y-1">
        <Typography.Text type="secondary" className="text-xs text-text-muted">
          {t('option:notesSearch.keywordPickerSortLabel', {
            defaultValue: 'Sort tags'
          })}
        </Typography.Text>
        <select
          value={sortMode}
          onChange={(event) =>
            onSortModeChange(event.target.value as 'frequency_desc' | 'alpha_asc' | 'alpha_desc')
          }
          className="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm text-text"
          data-testid="notes-keyword-picker-sort-select"
          aria-label={t('option:notesSearch.keywordPickerSortAriaLabel', {
            defaultValue: 'Sort tags'
          })}
        >
          <option value="frequency_desc">
            {t('option:notesSearch.keywordPickerSortFrequency', {
              defaultValue: 'Frequency (high to low)'
            })}
          </option>
          <option value="alpha_asc">
            {t('option:notesSearch.keywordPickerSortAlphaAsc', {
              defaultValue: 'Alphabetical (A-Z)'
            })}
          </option>
          <option value="alpha_desc">
            {t('option:notesSearch.keywordPickerSortAlphaDesc', {
              defaultValue: 'Alphabetical (Z-A)'
            })}
          </option>
        </select>
      </div>
      <div className="flex items-center justify-between gap-2">
        <Typography.Text type="secondary" className="text-xs text-text-muted">
          {t('option:notesSearch.keywordPickerCount', {
            defaultValue: '{{count}} tags',
            count: availableKeywords.length
          })}
        </Typography.Text>
        <div className="flex items-center gap-2">
          <Button
            size="small"
            onClick={onSelectAll}
            disabled={availableKeywords.length === 0}
          >
            {t('option:notesSearch.keywordPickerSelectAll', {
              defaultValue: 'Select all'
            })}
          </Button>
          <Button
            size="small"
            onClick={onClear}
            disabled={keywordPickerSelection.length === 0}
          >
            {t('option:notesSearch.keywordPickerClear', {
              defaultValue: 'Clear'
            })}
          </Button>
          {onOpenManager ? (
            <Button
              size="small"
              onClick={onOpenManager}
              disabled={managerDisabled}
              data-testid="notes-keyword-picker-open-manager"
            >
              {t('option:notesSearch.keywordPickerManageAction', {
                defaultValue: 'Manage tags'
              })}
            </Button>
          ) : null}
        </div>
      </div>
      {recentKeywordPickerOptions.length > 0 && (
        <div
          className="rounded-lg border border-border bg-surface px-2 py-2"
          data-testid="notes-keyword-picker-recent-section"
        >
          <Typography.Text type="secondary" className="block text-[11px] text-text-muted">
            {t('option:notesSearch.keywordPickerRecentHeading', {
              defaultValue: 'Recently used'
            })}
          </Typography.Text>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {recentKeywordPickerOptions.map((keyword) => {
              const selected = keywordPickerSelection.includes(keyword)
              return (
                <Button
                  key={`recent-${keyword}`}
                  size="small"
                  type={selected ? 'primary' : 'default'}
                  onClick={() => onToggleRecentKeyword(keyword)}
                  data-testid={`notes-keyword-picker-recent-${toKeywordTestIdSegment(keyword)}`}
                >
                  {renderKeywordLabel(keyword, 'notes-keyword-picker-recent-label')}
                </Button>
              )
            })}
          </div>
        </div>
      )}
      <div className="max-h-64 overflow-auto rounded-lg border border-border bg-surface2 p-3">
        {filteredKeywordPickerOptions.length === 0 ? (
          <Typography.Text
            type="secondary"
            className="block text-xs text-text-muted text-center"
          >
            {t('option:notesSearch.keywordPickerEmpty', {
              defaultValue: 'No tags found'
            })}
          </Typography.Text>
        ) : (
          <Checkbox.Group
            value={keywordPickerSelection}
            onChange={(vals) => onSelectionChange(vals as string[])}
            className="w-full"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {filteredKeywordPickerOptions.map((keyword) => (
                <Checkbox
                  key={keyword}
                  value={keyword}
                  data-testid={`notes-keyword-picker-option-${toKeywordTestIdSegment(keyword)}`}
                >
                  {renderKeywordLabel(keyword, 'notes-keyword-picker-option-label')}
                </Checkbox>
              ))}
            </div>
          </Checkbox.Group>
        )}
      </div>
    </div>
    </Modal>
  )
}

export default KeywordPickerModal
