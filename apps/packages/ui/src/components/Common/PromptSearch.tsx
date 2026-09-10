import React from 'react'
import { Input, List, Tag, Space, Tooltip, Modal, Checkbox, Button, message, Select, Dropdown } from 'antd'
import type { MenuProps } from 'antd'
import { ChevronDown } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getAllPrompts,
  savePrompt,
  updatePrompt,
  updateLastUsedPrompt as updateLastUsedPromptDB
} from '@/db/dexie/helpers'
import { useStorage } from '@plasmohq/storage/hook'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { autoSyncPrompt, shouldAutoSyncWorkspacePrompts } from '@/services/prompt-sync'
import { useMessageOption } from "@/hooks/useMessageOption"
import { Link } from 'react-router-dom'

type PromptItem = { id?: string; title: string; content: string; is_system?: boolean; source: 'local' | 'server' }

type Props = {
  onInsertMessage: (content: string) => void
  onInsertSystem: (content: string) => void
  inputId?: string
  ariaLabel?: string
  ariaLabelledby?: string
}

export const PromptSearch: React.FC<Props> = ({ onInsertMessage, onInsertSystem, inputId, ariaLabel, ariaLabelledby }) => {
  const { t } = useTranslation(['option'])
  const { historyId } = useMessageOption()
  const [remote, setRemote] = useStorage('promptSearchIncludeServer', false)
  const [q, setQ] = React.useState('')
  const [open, setOpen] = React.useState(false)
  const [results, setResults] = React.useState<PromptItem[]>([])
  const [loading, setLoading] = React.useState(false)
  const [selected, setSelected] = React.useState<PromptItem | null>(null)
  const [editorOpen, setEditorOpen] = React.useState(false)
  const [editTitle, setEditTitle] = React.useState('')
  const [editContent, setEditContent] = React.useState('')
  const [editIsSystem, setEditIsSystem] = React.useState<boolean>(false)
  const [localOverwriteId, setLocalOverwriteId] = React.useState<string | undefined>(undefined)
  const [saving, setSaving] = React.useState(false)
  const savingRef = React.useRef(false)
  const queryClient = useQueryClient()

  const { data: localPrompts } = useQuery({ queryKey: ['promptSearchAll'], queryFn: getAllPrompts })

  const refreshPromptQueries = React.useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['promptSearchAll'] }),
      queryClient.invalidateQueries({ queryKey: ['fetchAllPrompts'] })
    ])
  }, [queryClient])

  const syncPromptIfConfigured = React.useCallback(
    async (localId: string) => {
      try {
        const autoSyncEnabled = await shouldAutoSyncWorkspacePrompts()
        if (!autoSyncEnabled) return

        const syncResult = await autoSyncPrompt(localId)
        if (!syncResult.success) {
          message.warning(
            syncResult.error ||
              (t('promptSearch.saveLocalSuccess') || 'Saved') +
                ' (server sync pending)'
          )
        }
      } catch (error: unknown) {
        message.warning(
          (error instanceof Error ? error.message : null) ||
            ((t('promptSearch.saveLocalSuccess') || 'Saved') +
              ' (server sync pending)')
        )
      }
    },
    [t]
  )

  const runSearch = React.useCallback(async (query: string) => {
    if (!query.trim()) { setResults([]); return }
    setLoading(true)
    try {
      const local = (localPrompts || []).filter((p) => (p.title?.toLowerCase().includes(query.toLowerCase()) || p.content?.toLowerCase().includes(query.toLowerCase()))).map((p) => ({ id: p.id, title: p.title, content: p.content, is_system: p.is_system, source: 'local' as const }))
      let merged: PromptItem[] = local
          if (remote) {
            try {
              await tldwClient.initialize()
              const srv = await tldwClient.searchPrompts(query).catch(() => [])
              const serverList = Array.isArray(srv) ? srv : (srv?.results || srv?.prompts || [])
              const norm = (serverList as any[]).map((x) => {
                const name = x.name || x.title || 'Untitled'
                const content =
                  x.system_prompt ||
                  x.user_prompt ||
                  x.content ||
                  x.prompt ||
                  ''
                const isSystem =
                  typeof x.is_system === 'boolean'
                    ? x.is_system
                    : !!(x.system_prompt && !x.user_prompt)
                return {
                  id: x.id || x.uuid,
                  title: String(name),
                  content: String(content),
                  is_system: !!isSystem,
                  source: 'server' as const
                }
              })
              // Merge by id+title+content fingerprint
              const seen = new Set<string>()
              merged = [...local, ...norm].filter((p) => {
                const key = `${p.source}:${p.id || ''}:${p.title}:${p.content.slice(0,64)}`
                if (seen.has(key)) return false
            seen.add(key)
            return true
          })
        } catch {}
      }
      setResults(merged.slice(0, 50))
    } finally {
      setLoading(false)
    }
  }, [localPrompts, remote])

  React.useEffect(() => {
    const id = setTimeout(() => { void runSearch(q) }, 250)
    return () => clearTimeout(id)
  }, [q, runSearch])

  const openEditor = (item: PromptItem) => {
    setSelected(item)
    setEditTitle(item.title)
    setEditContent(item.content)
    setEditIsSystem(!!item.is_system)
    setLocalOverwriteId(undefined)
    setOpen(false)
    setEditorOpen(true)
  }

  const resolveErrorMessage = React.useCallback(
    (err: unknown, fallback: string) =>
      (err instanceof Error ? err.message : null) || fallback,
    []
  )

  const runWithSaving = React.useCallback(async (action: () => Promise<void>) => {
    if (savingRef.current) return
    savingRef.current = true
    setSaving(true)
    try {
      await action()
    } finally {
      savingRef.current = false
      setSaving(false)
    }
  }, [])

  const handleInsert = async () => {
    if (editIsSystem) onInsertSystem(editContent)
    else onInsertMessage(editContent)
    try {
      if (historyId && historyId !== 'temp') {
        await updateLastUsedPromptDB(historyId, {
          prompt_id: selected?.source === 'local' && selected?.id ? String(selected.id) : undefined,
          prompt_content: editContent
        })
      }
    } catch {}
    setEditorOpen(false)
  }

  const actionMenuItems = React.useMemo<MenuProps['items']>(() => {
    const items: MenuProps['items'] = []
    // Save changes to existing prompt
    if (selected?.source === 'local' && selected?.id) {
      items.push({
        key: 'save-local',
        label: t('promptSearch.saveLocal') || 'Save changes',
        onClick: () => {
          void runWithSaving(async () => {
            try {
              await updatePrompt({
                id: String(selected.id),
                title: editTitle || 'Untitled',
                content: editContent,
                is_system: !!editIsSystem
              })
              await syncPromptIfConfigured(String(selected.id))
              await refreshPromptQueries()
              message.success(t('promptSearch.saveLocalSuccess') || 'Saved')
            } catch (e: unknown) {
              message.error(
                resolveErrorMessage(
                  e,
                  t('promptSearch.saveLocalFailed') || 'Failed'
                )
              )
            }
          })
        }
      })
    }
    if (selected?.source === 'server' && selected?.id) {
      items.push({
        key: 'save-server',
        label: t('promptSearch.saveServer') || 'Save to server',
        onClick: () => {
          void runWithSaving(async () => {
            try {
              await tldwClient.updatePrompt(String(selected.id), {
                title: editTitle || 'Untitled',
                content: editContent,
                is_system: !!editIsSystem
              })
              message.success(t('promptSearch.saveServerSuccess') || 'Saved')
            } catch (e: unknown) {
              message.error(
                resolveErrorMessage(
                  e,
                  t('promptSearch.saveServerFailed') || 'Failed'
                )
              )
            }
          })
        }
      })
    }
    // Divider before "Save as" options
    if (items.length > 0) {
      items.push({ type: 'divider' })
    }
    // Save as new locally
    items.push({
      key: 'save-new-local',
      label: t('promptSearch.saveAsNew') || 'Save as new',
      onClick: () => {
          void runWithSaving(async () => {
            try {
              const savedPrompt = await savePrompt({
                title: editTitle || 'Untitled',
                content: editContent,
                is_system: !!editIsSystem,
              })
              await syncPromptIfConfigured(savedPrompt.id)
              await refreshPromptQueries()
              message.success(t('promptSearch.saveSuccess'))
            } catch (e: unknown) {
              message.error(
              resolveErrorMessage(
                e,
                t('promptSearch.saveFailed') || 'Failed'
              )
            )
          }
        })
      }
    })
    // Save local copy (for server prompts)
    if (selected?.source === 'server') {
      items.push({
        key: 'save-local-copy',
        label: localOverwriteId
          ? (t('promptSearch.overwriteLocal') || 'Overwrite selected local')
          : (t('promptSearch.saveLocalCopy') || 'Save local copy'),
        onClick: () => {
          void runWithSaving(async () => {
            try {
              if (localOverwriteId) {
                await updatePrompt({
                  id: localOverwriteId,
                  title: editTitle || 'Untitled',
                  content: editContent,
                  is_system: !!editIsSystem
                })
                await syncPromptIfConfigured(localOverwriteId)
              } else {
                const savedPrompt = await savePrompt({
                  title: editTitle || 'Untitled',
                  content: editContent,
                  is_system: !!editIsSystem,
                })
                await syncPromptIfConfigured(savedPrompt.id)
              }
              await refreshPromptQueries()
              message.success(t('promptSearch.saveLocalSuccess') || 'Saved')
            } catch (e: unknown) {
              message.error(
                resolveErrorMessage(
                  e,
                  t('promptSearch.saveLocalFailed') || 'Failed'
                )
              )
            }
          })
        }
      })
      items.push({
        key: 'save-new-server',
        label: t('promptSearch.saveServerOnly') || 'Save as new (server)',
        onClick: () => {
          void runWithSaving(async () => {
            try {
              await tldwClient.createPrompt({
                title: editTitle || 'Untitled',
                content: editContent,
                is_system: !!editIsSystem
              })
              message.success(t('promptSearch.saveServerSuccess') || 'Saved')
            } catch (e: unknown) {
              message.error(
                resolveErrorMessage(
                  e,
                  t('promptSearch.saveServerFailed') || 'Failed'
                )
              )
            }
          })
        }
      })
    }
    return items
  }, [
    editContent,
    editIsSystem,
    editTitle,
    localOverwriteId,
    message,
    refreshPromptQueries,
    resolveErrorMessage,
    runWithSaving,
    savePrompt,
    selected,
    syncPromptIfConfigured,
    t,
    tldwClient,
    updatePrompt
  ])

  const modifierKey = React.useMemo(() => {
    if (typeof navigator === 'undefined') return 'Ctrl'
    const nav = navigator as Navigator & {
      userAgentData?: { platform?: string }
    }
    const platform =
      nav.userAgentData?.platform ??
      nav.platform ??
      nav.userAgent ??
      ''
    return platform.includes('Mac') ? '⌘' : 'Ctrl'
  }, [])

  return (
    <div className="relative w-72">
      <Tooltip title={remote ? 'Search local + server prompts' : 'Search local prompts'}>
        <Input.Search
          id={inputId}
          placeholder={t('selectAPrompt') || 'Search prompts'}
          allowClear
          value={q}
          onChange={(e) => { setQ(e.target.value); setOpen(true) }}
          onFocus={() => setOpen(true)}
          onBlur={() => { setTimeout(() => setOpen(false), 200) }}
          loading={loading}
          aria-label={ariaLabel}
          aria-labelledby={ariaLabelledby}
        />
      </Tooltip>
      {open && results.length > 0 && (
        <div className="absolute mt-1 z-30 w-72 max-h-80 overflow-auto rounded-md border border-border bg-surface shadow">
          <List
            size="small"
            dataSource={results}
            renderItem={(item) => (
              <List.Item
                className="!px-2 !py-1 cursor-pointer hover:bg-surface2"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => openEditor(item)}
              >
                <div className="truncate text-sm">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">{item.title}</span>
                    <Tag color={item.source === 'server' ? 'geekblue' : 'default'}>{item.source}</Tag>
                  </div>
                  <div className="text-xs text-text-subtle line-clamp-2">{item.content}</div>
                </div>
              </List.Item>
            )}
          />
        </div>
      )}

      <Modal
        title={selected ? selected.title : 'Prompt'}
        open={editorOpen}
        onCancel={() => setEditorOpen(false)}
        footer={null}
        destroyOnHidden
        centered
      >
        <Space orientation="vertical" className="w-full" onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); handleInsert() } }}>
          <label className="text-xs text-text-subtle">{t('promptSearch.title')}</label>
          <Input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} />
          <label className="text-xs text-text-subtle">{t('promptSearch.content')}</label>
          <Input.TextArea value={editContent} onChange={(e) => setEditContent(e.target.value)} autoSize={{ minRows: 6 }} onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); handleInsert() } }} />
          <Checkbox checked={editIsSystem} onChange={(e) => setEditIsSystem(e.target.checked)}>{t('promptSearch.systemPrompt')}</Checkbox>
          {selected?.source === 'server' && (
            <div className="mt-1">
              <label className="text-xs text-text-subtle">{t('promptSearch.overwriteLocalLabel') || 'Overwrite local prompt (optional)'}</label>
              <Select
                className="w-full"
                allowClear
                placeholder={t('promptSearch.overwriteLocalPlaceholder') || 'Select a local prompt to overwrite'}
                value={localOverwriteId}
                options={(localPrompts || []).map((p) => ({ value: p.id, label: p.title }))}
                onChange={(v) => setLocalOverwriteId(v)}
              />
            </div>
          )}
          <div className="flex items-center justify-between gap-2 mt-2">
            <Link
              to="/prompts"
              className="text-xs underline text-primary hover:text-primaryStrong"
            >
              {t('promptSearch.manageLink') || 'View/Manage Prompts'}
            </Link>
            <div className="flex gap-2 items-center">
              <Button onClick={() => setEditorOpen(false)}>{t('promptSearch.cancel')}</Button>
              <Space.Compact>
                <Button
                  type="primary"
                  onClick={handleInsert}
                  loading={saving}
                  disabled={saving}
                >
                  <span className="inline-flex items-center gap-1">
                    {t('promptSearch.insert')}
                    <kbd className="hidden sm:inline text-[10px] px-1 py-0.5 rounded bg-white/20 font-mono">
                      {modifierKey}↵
                    </kbd>
                  </span>
                </Button>
                <Dropdown
                  menu={{
                    items: actionMenuItems
                  }}
                  disabled={saving}
                  trigger={['click']}
                >
                  <Button
                    type="primary"
                    icon={<ChevronDown className="size-3" />}
                    aria-label={t('promptSearch.insertOptions', 'Open insert options') as string}
                    title={t('promptSearch.insertOptions', 'Open insert options') as string}
                  />
                </Dropdown>
              </Space.Compact>
            </div>
          </div>
        </Space>
      </Modal>
    </div>
  )
}

export default PromptSearch
