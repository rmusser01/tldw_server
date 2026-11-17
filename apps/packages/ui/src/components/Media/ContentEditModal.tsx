import { useState, useEffect, useRef } from 'react'
import { Modal, message } from 'antd'
import { useTranslation } from 'react-i18next'
import { Copy, Save } from 'lucide-react'
import { bgRequest } from '@/services/background-proxy'
import { getTextStats } from '@/utils/text-stats'
import { useConfirmDanger } from '@/components/Common/confirm-danger'
import { RouteLeavePrompt } from '@/entries/shared/route-leave-prompt'

interface ContentEditModalProps {
  open: boolean
  onClose: () => void
  initialText: string
  mediaId?: string | number
  analysisContent?: string
  prompt?: string
  onSaveNewVersion?: (text: string) => void
}

export function ContentEditModal({
  open,
  onClose,
  initialText,
  mediaId,
  analysisContent,
  prompt,
  onSaveNewVersion
}: ContentEditModalProps) {
  const { t } = useTranslation(['review', 'common'])
  const [text, setText] = useState(initialText)
  const [saving, setSaving] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const triggerRef = useRef<HTMLElement | null>(null)
  const confirmDanger = useConfirmDanger()
  const closing = useRef(false)
  const editorIdentity = useRef({ open, mediaId, text })
  editorIdentity.current = { open, mediaId, text }
  const dirty = text !== initialText

  useEffect(() => {
    editorIdentity.current.open = open
    return () => { editorIdentity.current.open = false }
  }, [open])

  useEffect(() => {
    if (!open || !dirty) return
    const protectDraft = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', protectDraft)
    return () => window.removeEventListener('beforeunload', protectDraft)
  }, [open, dirty])

  const handleClose = async () => {
    if (saving || closing.current) return
    closing.current = true
    try {
      const discard = !dirty || await confirmDanger({
        title: t('mediaPage.unsavedContentTitle', 'Discard unsaved content changes?'),
        content: t('mediaPage.unsavedContentWarning', 'Your edits have not been saved.'),
        okText: t('common:discardChanges', 'Discard changes'),
        cancelText: t('common:keepEditing', 'Keep editing')
      })
      const current = editorIdentity.current
      if (discard && current.open && current.mediaId === mediaId && current.text === text) onClose()
    } finally {
      closing.current = false
    }
  }


  useEffect(() => {
    if (open) {
      setText(initialText)
    }
  }, [open, initialText])

  useEffect(() => {
    if (open) {
      triggerRef.current = document.activeElement as HTMLElement
    }
  }, [open])

  const handleAfterOpenChange = (visible: boolean) => {
    if (visible) {
      setTimeout(() => textareaRef.current?.focus(), 0)
    } else {
      triggerRef.current?.focus()
    }
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      message.success(t('mediaPage.contentCopied', 'Content copied'))
    } catch {
      message.error(t('mediaPage.copyFailed', 'Failed to copy'))
    }
  }

  const handleSaveAsNewVersion = async () => {
    if (!mediaId) {
      message.error(t('mediaPage.noMediaId', 'No media ID available'))
      return
    }
    if (!text.trim()) {
      message.warning(
        t('mediaPage.emptyContent', 'Content cannot be empty')
      )
      return
    }

    setSaving(true)
    try {
      const getVersionNumber = (v: any): number | null => {
        const raw = v?.version_number ?? v?.version
        if (typeof raw === 'number' && Number.isFinite(raw)) return raw
        if (typeof raw === 'string' && raw.trim().length > 0) {
          const parsed = Number(raw)
          if (Number.isFinite(parsed)) return parsed
        }
        return null
      }

      const pickLatestVersion = (versions: any[]): any | null => {
        if (!Array.isArray(versions) || versions.length === 0) return null
        let best: any | null = null
        let bestNum = -Infinity
        for (const v of versions) {
          const num = getVersionNumber(v)
          if (num != null && num > bestNum) {
            best = v
            bestNum = num
          }
        }
        return best || versions[0]
      }

      let resolvedPrompt = String(prompt || '')
      let resolvedAnalysis = String(analysisContent || '')
      if (!resolvedPrompt || !resolvedAnalysis) {
        try {
          const versions = await bgRequest<any>({
            path: `/api/v1/media/${mediaId}/versions?include_content=false&limit=50&page=1`,
            method: 'GET'
          })
          const arr = Array.isArray(versions) ? versions : (versions?.items || [])
          const latest = pickLatestVersion(arr)
          if (!resolvedPrompt) {
            resolvedPrompt = String(latest?.prompt || '')
          }
          if (!resolvedAnalysis) {
            resolvedAnalysis = String(
              latest?.analysis_content || latest?.analysis || ''
            )
          }
        } catch {}
      }

      await bgRequest({
        path: `/api/v1/media/${mediaId}/versions`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: {
          content: text,
          prompt: resolvedPrompt,
          analysis_content: resolvedAnalysis
        }
      })
      message.success(t('mediaPage.versionSaved', 'Saved as new version'))
      if (onSaveNewVersion) {
        onSaveNewVersion(text)
      }
      onClose()
    } catch (err) {
      console.error('Failed to save content version:', err)
      const errMsg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message?: unknown }).message)
          : String(err || 'Unknown error')
      message.error(
        errMsg.includes('401') || errMsg.includes('403')
          ? t('mediaPage.saveUnauthorized', 'Not authorized to save')
          : t('mediaPage.saveFailed', 'Failed to save version')
      )
    } finally {
      setSaving(false)
    }
  }

  const { wordCount, charCount } = getTextStats(text)

  return (
    <Modal
      title={t('mediaPage.editContent', 'Edit Content')}
      open={open}
      onCancel={() => void handleClose()}
      closable={!saving}
      maskClosable={!saving}
      keyboard={!saving}
      afterOpenChange={handleAfterOpenChange}
      width={800}
      footer={
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 text-sm text-text hover:bg-surface2 rounded flex items-center gap-1.5 transition-colors"
              title={t('common:copy', 'Copy')}
            >
              <Copy className="w-4 h-4" />
              {t('common:copy', 'Copy')}
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void handleClose()}
              disabled={saving}
              className="px-3 py-1.5 text-sm text-text-muted hover:bg-surface2 rounded transition-colors"
              title={t('common:cancel', 'Cancel')}
            >
              {t('common:cancel', 'Cancel')}
            </button>
            {mediaId && (
              <button
                onClick={handleSaveAsNewVersion}
                disabled={saving}
                className="px-3 py-1.5 text-sm bg-primary hover:bg-primaryStrong text-white rounded flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                title={t('mediaPage.saveAsVersion', 'Save as New Version')}
              >
                <Save className="w-4 h-4" />
                {t('mediaPage.saveAsVersion', 'Save as New Version')}
              </button>
            )}
          </div>
        </div>
      }
    >
      <div className="space-y-3">
        {open && dirty && <RouteLeavePrompt when message={t('mediaPage.unsavedContentLeave', 'Leave and discard unsaved content changes?')} />}
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full min-h-[300px] p-3 text-sm font-mono rounded-lg border bg-surface text-text focus:outline-none focus:ring-2 border-border focus:ring-focus resize-y leading-relaxed"
          placeholder={t(
            'mediaPage.contentPlaceholder',
            'Edit content text...'
          )}
        />
        <div className="flex items-center justify-between text-xs text-text-muted">
          <div>
            {t('mediaPage.wordCount', '{{count}} words', { count: wordCount })}
            {' • '}
            {t('mediaPage.charCount', '{{count}} characters', {
              count: charCount
            })}
          </div>
        </div>
      </div>
    </Modal>
  )
}
