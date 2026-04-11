import React, { useMemo } from 'react'
import { Button, Dropdown, Tooltip, Typography } from 'antd'
import type { MenuProps } from 'antd'
import { useTranslation } from 'react-i18next'
import { useMobile } from '@/hooks/useMediaQuery'
import type { SaveIndicatorState } from './notes-manager-types'
import NotesSaveStatus from './NotesSaveStatus'
import {
  Link2 as LinkIcon,
  Sparkles as SparklesIcon,
  Copy as CopyIcon,
  FileDown as FileDownIcon,
  FilePlus2 as FileTemplateIcon,
  Save as SaveIcon,
  Star as StarIcon,
  Trash2 as TrashIcon,
  Eye as EyeIcon,
  Edit3 as EditIcon,
  Columns2 as SplitIcon,
  MoreHorizontal,
  Printer as PrinterIcon
} from 'lucide-react'

interface NotesEditorHeaderProps {
  title: string
  selectedId: string | number | null
  backlinkConversationId: string | null
  backlinkConversationLabel: string | null
  backlinkMessageId: string | null
  sourceLinks: Array<{ id: string; label: string }>
  editorDisabled: boolean
  openingLinkedChat: boolean
  editorMode: 'edit' | 'split' | 'preview'
  hasContent: boolean
  canSave: boolean
  canGenerateFlashcards: boolean
  canCreateStudyPack?: boolean
  canOpenNotesStudio?: boolean
  canExport: boolean
  canDuplicate?: boolean
  canPin?: boolean
  isPinned?: boolean
  templateOptions?: Array<{ id: string; label: string }>
  isSaving: boolean
  canDelete: boolean
  isDirty?: boolean
  saveIndicator?: SaveIndicatorState
  lastSavedAt?: string | null
  onOpenLinkedConversation: () => void
  onOpenSourceLink: (sourceId: string, sourceLabel: string) => void
  onApplyTemplate?: (templateId: string) => void
  onDuplicate?: () => void
  onTogglePin?: () => void
  onChangeEditorMode: (mode: 'edit' | 'split' | 'preview') => void
  onCopy: (mode: 'content' | 'markdown') => void
  onGenerateFlashcards: () => void
  onCreateStudyPack?: () => void
  onOpenNotesStudio?: () => void
  onExport: (format: 'md' | 'json' | 'print') => void
  onSave: () => void
  onDelete: () => void
  studioBadgeLabel?: string | null
}

const NotesEditorHeader: React.FC<NotesEditorHeaderProps> = ({
  title,
  selectedId,
  backlinkConversationId,
  backlinkConversationLabel,
  backlinkMessageId,
  sourceLinks,
  editorDisabled,
  openingLinkedChat,
  editorMode,
  hasContent,
  canSave,
  canGenerateFlashcards,
  canCreateStudyPack = false,
  canOpenNotesStudio = false,
  canExport,
  canDuplicate = false,
  canPin = false,
  isPinned = false,
  templateOptions = [],
  isSaving,
  canDelete,
  isDirty,
  saveIndicator = 'idle',
  lastSavedAt,
  onOpenLinkedConversation,
  onOpenSourceLink,
  onApplyTemplate,
  onDuplicate,
  onTogglePin,
  onChangeEditorMode,
  onCopy,
  onGenerateFlashcards,
  onCreateStudyPack,
  onOpenNotesStudio,
  onExport,
  onSave,
  onDelete,
  studioBadgeLabel = null,
}) => {
  const { t } = useTranslation(['option', 'common'])
  const isMobileViewport = useMobile()
  const toolbarButtonSize: 'small' | 'large' = isMobileViewport ? 'large' : 'small'
  const touchTargetClass = isMobileViewport ? 'min-h-[44px]' : undefined
  const touchTargetIconOnlyClass = isMobileViewport ? 'min-h-[44px] min-w-[44px]' : undefined

  const displayTitle =
    selectedId == null
      ? t('option:notesSearch.newNoteTitle', { defaultValue: 'New note' })
      : title ||
        t('option:notesSearch.untitledNote', {
          defaultValue: `Note ${selectedId}`
        })
  const conversationDisplayText = backlinkConversationLabel || backlinkConversationId
  const conversationDebugTooltip =
    backlinkConversationId != null
      ? `${t('option:notesSearch.linkedConversationIdLabel', {
          defaultValue: 'Conversation ID'
        })}: ${backlinkConversationId}`
      : null

  const overflowMenuItems: MenuProps['items'] = useMemo(() => {
    const items: MenuProps['items'] = []

    // --- Create group ---
    if (!editorDisabled) {
      const createChildren: MenuProps['items'] = []

      if (onDuplicate) {
        createChildren.push({
          key: 'duplicate',
          label: t('option:notesSearch.duplicateNoteAction', { defaultValue: 'Duplicate' }),
          icon: (<CopyIcon className="w-4 h-4" />),
          disabled: !canDuplicate
        })
      }

      if (onApplyTemplate && templateOptions.length > 0) {
        createChildren.push({
          key: 'template-submenu',
          label: t('option:notesSearch.templateAction', { defaultValue: 'From Template' }),
          icon: (<FileTemplateIcon className="w-4 h-4" />),
          children: templateOptions.map((tpl) => ({
            key: `template-${tpl.id}`,
            label: tpl.label
          }))
        })
      }

      if (createChildren.length > 0) {
        items.push({
          type: 'group' as const,
          label: t('option:notesSearch.overflowGroupCreate', { defaultValue: 'Create' }),
          children: createChildren
        })
      }

      // --- Organize group ---
      const organizeChildren: MenuProps['items'] = []

      if (onTogglePin) {
        organizeChildren.push({
          key: 'pin',
          label: isPinned
            ? t('option:notesSearch.unpinNoteAction', { defaultValue: 'Unpin' })
            : t('option:notesSearch.pinNoteAction', { defaultValue: 'Pin' }),
          icon: (
            <StarIcon
              className={`w-4 h-4 ${isPinned ? 'fill-current text-amber-500' : ''}`}
            />
          ),
          disabled: !canPin
        })
      }

      if (backlinkConversationId) {
        organizeChildren.push({
          key: 'open-conversation',
          label: t('option:notesSearch.openConversation', {
            defaultValue: 'Open linked conversation'
          }),
          icon: (<LinkIcon className="w-4 h-4" />)
        })
      }

      if (organizeChildren.length > 0) {
        items.push({
          type: 'group' as const,
          label: t('option:notesSearch.overflowGroupOrganize', { defaultValue: 'Organize' }),
          children: organizeChildren
        })
      }

      // --- AI group ---
      items.push({
        type: 'group' as const,
        label: t('option:notesSearch.overflowGroupAI', { defaultValue: 'AI' }),
        children: [
          {
            key: 'notes-studio',
            label: t('option:notesSearch.notesStudioAction', {
              defaultValue: 'Notes Studio'
            }),
            icon: (<SparklesIcon className="w-4 h-4" />),
            disabled: !canOpenNotesStudio
          },
          {
            key: 'flashcards',
            label: t('option:notesSearch.generateFlashcardsAction', {
              defaultValue: 'Generate flashcards'
            }),
            icon: (<SparklesIcon className="w-4 h-4" />),
            disabled: !canGenerateFlashcards
          },
          {
            key: 'study-pack',
            label: t('option:notesSearch.createStudyPackAction', {
              defaultValue: 'Create study pack'
            }),
            icon: (<SparklesIcon className="w-4 h-4" />),
            disabled: !canCreateStudyPack
          }
        ]
      })
    }

    // --- Export group ---
    items.push({
      type: 'group' as const,
      label: t('option:notesSearch.overflowGroupExport', { defaultValue: 'Export' }),
      children: [
        {
          key: 'copy-submenu',
          label: t('option:notesSearch.toolbarCopyTooltip', { defaultValue: 'Copy' }),
          icon: (<CopyIcon className="w-4 h-4" />),
          disabled: !hasContent,
          children: [
            {
              key: 'copy-content',
              label: t('option:notesSearch.copyModeContentOnly', {
                defaultValue: 'Content only'
              })
            },
            {
              key: 'copy-markdown',
              label: t('option:notesSearch.copyModeMarkdown', {
                defaultValue: 'Markdown with title'
              })
            }
          ]
        },
        {
          key: 'export-submenu',
          label: t('option:notesSearch.exportSingleAction', { defaultValue: 'Export' }),
          icon: (<FileDownIcon className="w-4 h-4" />),
          disabled: !canExport,
          children: [
            {
              key: 'export-md',
              label: t('option:notesSearch.exportSingleMd', {
                defaultValue: 'Markdown (.md)'
              })
            },
            {
              key: 'export-json',
              label: t('option:notesSearch.exportSingleJson', {
                defaultValue: 'JSON (.json)'
              })
            },
            {
              key: 'export-print',
              label: t('option:notesSearch.exportSinglePrint', {
                defaultValue: 'Print / Save as PDF'
              }),
              icon: (<PrinterIcon className="w-4 h-4" />)
            }
          ]
        }
      ]
    })

    // --- Danger divider + delete ---
    items.push({ type: 'divider' as const })
    items.push({
      key: 'delete',
      label: t('common:delete', { defaultValue: 'Delete' }),
      icon: (<TrashIcon className="w-4 h-4" />),
      danger: true,
      disabled: !canDelete
    })

    return items
  }, [
    editorDisabled,
    canDuplicate,
    canPin,
    isPinned,
    canOpenNotesStudio,
    canGenerateFlashcards,
    canCreateStudyPack,
    canExport,
    canDelete,
    hasContent,
    backlinkConversationId,
    templateOptions,
    onDuplicate,
    onApplyTemplate,
    onTogglePin,
    t
  ])

  const handleOverflowMenuClick: MenuProps['onClick'] = ({ key }) => {
    switch (key) {
      case 'duplicate':
        onDuplicate?.()
        break
      case 'pin':
        onTogglePin?.()
        break
      case 'open-conversation':
        onOpenLinkedConversation()
        break
      case 'flashcards':
        onGenerateFlashcards()
        break
      case 'study-pack':
        onCreateStudyPack?.()
        break
      case 'notes-studio':
        onOpenNotesStudio?.()
        break
      case 'copy-content':
        onCopy('content')
        break
      case 'copy-markdown':
        onCopy('markdown')
        break
      case 'export-md':
        onExport('md')
        break
      case 'export-json':
        onExport('json')
        break
      case 'export-print':
        onExport('print')
        break
      case 'delete':
        onDelete()
        break
      default:
        if (key?.startsWith('template-')) {
          onApplyTemplate?.(key.replace('template-', ''))
        }
        break
    }
  }

  return (
    <div className="flex flex-col gap-3 border-b border-border bg-surface px-4 py-3 md:flex-row md:items-center md:justify-between">
      <div className="flex flex-col gap-0.5 min-w-0">
        <div className="flex items-center gap-2">
          <Typography.Title level={5} className="!mb-0 truncate !text-text">
            {displayTitle}
          </Typography.Title>
          {studioBadgeLabel ? (
            <span className="rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary">
              {studioBadgeLabel}
            </span>
          ) : null}
          <NotesSaveStatus
            state={isDirty ? 'dirty' : saveIndicator}
            lastSavedAt={lastSavedAt}
            onRetry={onSave}
          />
        </div>
        {backlinkConversationId && (
          <div className="text-xs text-primary">
            {t('option:notesSearch.linkedConversation', {
              defaultValue: 'Linked to conversation'
            })}{': '}
            <Tooltip title={conversationDebugTooltip}>
              <span className="font-medium">{conversationDisplayText}</span>
            </Tooltip>
            {backlinkMessageId ? ` · msg ${backlinkMessageId}` : ''}
          </div>
        )}
        {sourceLinks.length > 0 && (
          <div className="mt-1 flex flex-wrap items-center gap-1">
            <span className="text-[11px] text-text-muted">
              {t('option:notesSearch.linkedSourcesLabel', {
                defaultValue: 'Sources'
              })}
              {':'}
            </span>
            {sourceLinks.map((source) => (
              <Tooltip
                key={`source-link-${source.id}`}
                title={`${t('option:notesSearch.linkedSourceIdLabel', {
                  defaultValue: 'Source node'
                })}: ${source.id}`}
              >
                <Button
                  type="link"
                  size="small"
                  className="!h-auto !px-1 text-xs"
                  onClick={() => onOpenSourceLink(source.id, source.label)}
                  data-testid={`notes-source-link-${source.id.replace(/[^a-z0-9_-]/gi, '_')}`}
                >
                  {source.label}
                </Button>
              </Tooltip>
            ))}
          </div>
        )}
      </div>
      <div
        className={
          isMobileViewport
            ? 'flex w-full flex-wrap items-center justify-start gap-2'
            : 'flex items-center gap-2'
        }
        data-testid="notes-header-actions"
      >
        <Tooltip
          title={
            !canSave
              ? t('option:notesSearch.toolbarSaveDisabledTooltip', {
                  defaultValue: 'Add a title or content to save'
                })
              : t('option:notesSearch.toolbarSaveTooltip', {
                  defaultValue: 'Save note'
                })
          }
        >
          <Button
            type="primary"
            size={toolbarButtonSize}
            onClick={onSave}
            loading={isSaving}
            disabled={!canSave}
            icon={(<SaveIcon className="w-4 h-4" />)}
            className={touchTargetClass}
            aria-label={t('option:notesSearch.toolbarSaveTooltip', {
              defaultValue: 'Save note'
            })}
            data-testid="notes-save-button"
          >
            {t('common:save', { defaultValue: 'Save' })}
          </Button>
        </Tooltip>

        {/* Editor mode toggle - visible on desktop, hidden on mobile */}
        {!isMobileViewport && (
          <div
            className="inline-flex items-center gap-1 rounded-md border border-border p-0.5"
            role="group"
            aria-label={t('option:notesSearch.editorModeGroup', {
              defaultValue: 'Editor mode'
            })}
          >
            <Tooltip
              title={t('option:notesSearch.toolbarEditModeTooltip', {
                defaultValue: 'Write markdown content'
              })}
            >
              <Button
                size={toolbarButtonSize}
                type={editorMode === 'edit' ? 'primary' : 'text'}
                onClick={() => onChangeEditorMode('edit')}
                icon={(<EditIcon className="w-4 h-4" />)}
                className={touchTargetClass}
                aria-pressed={editorMode === 'edit'}
                aria-label={t('option:notesSearch.editModeLabel', {
                  defaultValue: 'Edit'
                })}
              >
                {t('option:notesSearch.editModeLabel', {
                  defaultValue: 'Edit'
                })}
              </Button>
            </Tooltip>
            <Tooltip
              title={t('option:notesSearch.toolbarSplitModeTooltip', {
                defaultValue: 'Show editor and preview together'
              })}
            >
              <Button
                size={toolbarButtonSize}
                type={editorMode === 'split' ? 'primary' : 'text'}
                onClick={() => onChangeEditorMode('split')}
                icon={(<SplitIcon className="w-4 h-4" />)}
                className={touchTargetClass}
                aria-pressed={editorMode === 'split'}
                aria-label={t('option:notesSearch.splitModeLabel', {
                  defaultValue: 'Split'
                })}
              >
                {t('option:notesSearch.splitModeLabel', {
                  defaultValue: 'Split'
                })}
              </Button>
            </Tooltip>
            <Tooltip
              title={t('option:notesSearch.toolbarPreviewTooltip', {
                defaultValue: 'Preview rendered Markdown'
              })}
            >
              <Button
                size={toolbarButtonSize}
                type={editorMode === 'preview' ? 'primary' : 'text'}
                onClick={() => onChangeEditorMode('preview')}
                icon={(<EyeIcon className="w-4 h-4" />)}
                className={touchTargetClass}
                aria-pressed={editorMode === 'preview'}
                aria-label={t('option:notesSearch.previewModeLabel', {
                  defaultValue: 'Preview'
                })}
              >
                {t('option:notesSearch.previewModeLabel', {
                  defaultValue: 'Preview'
                })}
              </Button>
            </Tooltip>
          </div>
        )}

        {/* Overflow "More actions" menu */}
        <Dropdown
          trigger={['click']}
          menu={{
            items: overflowMenuItems,
            onClick: handleOverflowMenuClick
          }}
        >
          <Tooltip
            title={t('option:notesSearch.moreActionsTooltip', {
              defaultValue: 'More actions'
            })}
          >
            <Button
              size={toolbarButtonSize}
              icon={(<MoreHorizontal className="w-4 h-4" />)}
              className={touchTargetIconOnlyClass}
              aria-label={t('option:notesSearch.moreActionsTooltip', {
                defaultValue: 'More actions'
              })}
              data-testid="notes-overflow-menu-button"
            />
          </Tooltip>
        </Dropdown>
      </div>
    </div>
  )
}

export default NotesEditorHeader
