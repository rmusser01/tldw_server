import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { message } from 'antd'
import { bgRequest } from '@/services/background-proxy'
import { useConfirmDanger } from '@/components/Common/confirm-danger'
import type { MediaResultItem } from '../types'

const firstNonEmptyString = (...vals: any[]): string => {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim().length > 0) return v
  }
  return ''
}

export interface UseContentEditStateDeps {
  selectedMedia: MediaResultItem | null
  content: string
  mediaDetail: any
  selectedMediaId: string | null
  isNote: boolean
  onKeywordsUpdated?: (mediaId: string | number, keywords: string[]) => void
  onRefreshMedia?: () => void
  onDeleteItem?: (item: MediaResultItem, detail: any | null) => Promise<void>
  t: (key: string, opts?: Record<string, any>) => string
}

export function useContentEditState(deps: UseContentEditStateDeps) {
  const {
    selectedMedia,
    content,
    mediaDetail,
    selectedMediaId,
    isNote,
    onKeywordsUpdated,
    onRefreshMedia,
    onDeleteItem,
    t
  } = deps

  const confirmDanger = useConfirmDanger()

  const [editingKeywords, setEditingKeywords] = useState<string[]>([])
  const [savingKeywords, setSavingKeywords] = useState(false)
  const saveKeywordsTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [contentEditModalOpen, setContentEditModalOpen] = useState(false)
  const [editingContentText, setEditingContentText] = useState('')

  const [analysisEditModalOpen, setAnalysisEditModalOpen] = useState(false)
  const [editingAnalysisText, setEditingAnalysisText] = useState('')
  const [optimisticAnalysis, setOptimisticAnalysis] = useState('')
  const [activeAnalysisIndex, setActiveAnalysisIndex] = useState(0)
  const [analysisExpanded, setAnalysisExpanded] = useState(false)

  const [deletingItem, setDeletingItem] = useState(false)
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null)

  const isAwaitingSelectionUpdate =
    !!pendingDeleteId && !!selectedMediaId && pendingDeleteId === selectedMediaId

  const resolveNoteVersion = useCallback((detail: any, raw: any): number | null => {
    const candidates = [
      detail?.version,
      detail?.metadata?.version,
      raw?.version,
      raw?.metadata?.version
    ]
    for (const candidate of candidates) {
      if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate
      if (typeof candidate === 'string' && candidate.trim().length > 0) {
        const parsed = Number(candidate)
        if (Number.isFinite(parsed)) return parsed
      }
    }
    return null
  }, [])

  const getVersionNumber = useCallback((v: any): number | null => {
    const raw = v?.version_number ?? v?.version
    if (typeof raw === 'number' && Number.isFinite(raw)) return raw
    if (typeof raw === 'string' && raw.trim().length > 0) {
      const parsed = Number(raw)
      if (Number.isFinite(parsed)) return parsed
    }
    return null
  }, [])

  const pickLatestVersion = useCallback((versions: any[]): any | null => {
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
  }, [getVersionNumber])

  const latestVersion = useMemo(() => {
    if (!mediaDetail || typeof mediaDetail !== 'object') return null
    const direct = mediaDetail.latest_version || mediaDetail.latestVersion
    if (direct && typeof direct === 'object') return direct
    const versions = Array.isArray(mediaDetail.versions) ? mediaDetail.versions : []
    return pickLatestVersion(versions)
  }, [mediaDetail, pickLatestVersion])

  const derivedPrompt = useMemo(() => {
    if (!mediaDetail) return ''
    const fromRoot = firstNonEmptyString(mediaDetail.prompt)
    if (fromRoot) return fromRoot
    const fromProcessing = firstNonEmptyString(mediaDetail?.processing?.prompt)
    if (fromProcessing) return fromProcessing
    return firstNonEmptyString(latestVersion?.prompt)
  }, [mediaDetail, latestVersion])

  const persistedAnalysisContent = useMemo(() => {
    if (!mediaDetail) return ''
    const fromProcessing = firstNonEmptyString(mediaDetail?.processing?.analysis)
    if (fromProcessing) return fromProcessing
    const fromAnalysis = firstNonEmptyString(mediaDetail?.analysis)
    if (fromAnalysis) return fromAnalysis
    const fromAnalysisContent = firstNonEmptyString(
      mediaDetail?.analysis_content,
      mediaDetail?.analysisContent
    )
    if (fromAnalysisContent) return fromAnalysisContent
    if (Array.isArray(mediaDetail?.analyses)) {
      for (const entry of mediaDetail.analyses) {
        const text = typeof entry === 'string'
          ? entry
          : (entry?.content || entry?.text || entry?.summary || entry?.analysis_content || '')
        const resolved = firstNonEmptyString(text)
        if (resolved) return resolved
      }
    }
    const fromVersion = firstNonEmptyString(
      latestVersion?.analysis_content,
      latestVersion?.analysis
    )
    if (fromVersion) return fromVersion
    return firstNonEmptyString(mediaDetail?.summary)
  }, [mediaDetail, latestVersion])

  const derivedAnalysisContent = useMemo(() => {
    if (optimisticAnalysis) return optimisticAnalysis
    return persistedAnalysisContent
  }, [optimisticAnalysis, persistedAnalysisContent])

  const existingAnalyses = useMemo(() => {
    if (!mediaDetail) return []
    const analyses: Array<{ type: string; text: string }> = []

    if (mediaDetail.processing?.analysis && typeof mediaDetail.processing.analysis === 'string' && mediaDetail.processing.analysis.trim()) {
      analyses.push({ type: 'Analysis', text: mediaDetail.processing.analysis })
    }

    if (mediaDetail.summary && typeof mediaDetail.summary === 'string' && mediaDetail.summary.trim()) {
      analyses.push({ type: 'Summary', text: mediaDetail.summary })
    }

    const rootAnalysis = typeof mediaDetail.analysis === 'string' ? mediaDetail.analysis.trim() : ''
    if (rootAnalysis) {
      analyses.push({ type: 'Analysis', text: rootAnalysis })
    }

    const rootAnalysisContent = firstNonEmptyString(
      mediaDetail.analysis_content,
      mediaDetail.analysisContent
    )
    if (rootAnalysisContent && rootAnalysisContent !== rootAnalysis) {
      analyses.push({ type: 'Analysis', text: rootAnalysisContent })
    }

    if (Array.isArray(mediaDetail.analyses)) {
      mediaDetail.analyses.forEach((a: any, idx: number) => {
        const text = typeof a === 'string' ? a : (a?.content || a?.text || a?.summary || a?.analysis_content || '')
        const type = typeof a === 'object' && a?.type ? a.type : `Analysis ${idx + 1}`
        if (text && text.trim()) {
          analyses.push({ type, text })
        }
      })
    }

    if (Array.isArray(mediaDetail.versions)) {
      mediaDetail.versions.forEach((v: any, idx: number) => {
        if (v?.analysis_content && typeof v.analysis_content === 'string' && v.analysis_content.trim()) {
          const versionNum = v?.version_number || idx + 1
          analyses.push({ type: `Analysis (Version ${versionNum})`, text: v.analysis_content })
        }
      })
    }

    if (optimisticAnalysis) {
      const trimmed = optimisticAnalysis.trim()
      if (trimmed && !analyses.some((entry) => entry.text.trim() === trimmed)) {
        analyses.unshift({ type: 'Analysis', text: optimisticAnalysis })
      }
    }

    return analyses
  }, [mediaDetail, optimisticAnalysis])

  const activeAnalysis =
    existingAnalyses.length > 0
      ? existingAnalyses[Math.min(activeAnalysisIndex, existingAnalyses.length - 1)]
      : null

  const selectedAnalysis = activeAnalysis
  const analysisText = activeAnalysis?.text || ''
  const ANALYSIS_COLLAPSE_THRESHOLD = 2000
  const analysisIsLong = analysisText.length > ANALYSIS_COLLAPSE_THRESHOLD
  const analysisShown =
    !analysisIsLong || analysisExpanded
      ? analysisText
      : `${analysisText.slice(0, ANALYSIS_COLLAPSE_THRESHOLD)}\u2026`

  // Reset optimistic analysis on media change
  useEffect(() => {
    setOptimisticAnalysis('')
  }, [selectedMedia?.id])

  useEffect(() => {
    if (persistedAnalysisContent) {
      setOptimisticAnalysis('')
    }
  }, [persistedAnalysisContent])

  useEffect(() => {
    setActiveAnalysisIndex(0)
    setAnalysisExpanded(false)
  }, [selectedMedia?.id])

  useEffect(() => {
    setAnalysisExpanded(false)
  }, [activeAnalysisIndex])

  // Sync editing keywords with selected media
  useEffect(() => {
    if (saveKeywordsTimeout.current) {
      clearTimeout(saveKeywordsTimeout.current)
      saveKeywordsTimeout.current = null
    }
    setEditingKeywords(selectedMedia?.keywords || [])
  }, [selectedMedia?.id, selectedMedia?.keywords])

  // Clamp analysis index
  useEffect(() => {
    if (existingAnalyses.length === 0) {
      setActiveAnalysisIndex(0)
      return
    }
    if (activeAnalysisIndex >= existingAnalyses.length) {
      setActiveAnalysisIndex(existingAnalyses.length - 1)
    }
  }, [activeAnalysisIndex, existingAnalyses.length])

  // Cleanup keyword timeout
  useEffect(() => {
    return () => {
      if (saveKeywordsTimeout.current) {
        clearTimeout(saveKeywordsTimeout.current)
      }
    }
  }, [])

  // Clear pending delete when selection changes
  useEffect(() => {
    if (!pendingDeleteId) return
    if (!selectedMediaId || pendingDeleteId !== selectedMediaId) {
      setPendingDeleteId(null)
    }
  }, [pendingDeleteId, selectedMediaId])

  const persistKeywords = useCallback(
    async (newKeywords: string[]) => {
      if (!selectedMedia) return
      setSavingKeywords(true)
      try {
        const endpoint =
          selectedMedia.kind === 'note'
            ? `/api/v1/notes/${selectedMedia.id}`
            : `/api/v1/media/${selectedMedia.id}`
        const headers: Record<string, string> = { 'Content-Type': 'application/json' }
        if (selectedMedia.kind === 'note') {
          let expectedVersion = resolveNoteVersion(mediaDetail, selectedMedia.raw)
          if (expectedVersion == null) {
            try {
              const latest = await bgRequest<any>({
                path: `/api/v1/notes/${selectedMedia.id}` as any,
                method: 'GET' as any
              })
              expectedVersion = resolveNoteVersion(latest, null)
            } catch {
              expectedVersion = null
            }
          }
          if (expectedVersion == null) {
            throw new Error(
              t('review:mediaPage.noteUpdateNeedsReload', {
                defaultValue: 'Unable to update note. Reload and try again.'
              })
            )
          }
          headers['expected-version'] = String(expectedVersion)
        }

        await bgRequest({
          path: endpoint as any,
          method: 'PUT' as any,
          headers,
          body: { keywords: newKeywords }
        })
        setEditingKeywords(newKeywords)
        if (onKeywordsUpdated) {
          onKeywordsUpdated(selectedMedia.id, newKeywords)
        }
        message.success(
          t('review:mediaPage.keywordsSaved', {
            defaultValue: 'Keywords saved'
          })
        )
      } catch (err) {
        console.error('Failed to save keywords:', err)
        message.error(
          t('review:mediaPage.keywordsSaveFailed', {
            defaultValue: 'Failed to save keywords'
          })
        )
      } finally {
        setSavingKeywords(false)
      }
    },
    [mediaDetail, onKeywordsUpdated, resolveNoteVersion, selectedMedia, t]
  )

  const handleSaveKeywords = useCallback((newKeywords: string[]) => {
    setEditingKeywords(newKeywords)
    if (saveKeywordsTimeout.current) {
      clearTimeout(saveKeywordsTimeout.current)
    }
    saveKeywordsTimeout.current = setTimeout(() => {
      persistKeywords(newKeywords)
    }, 500)
  }, [persistKeywords])

  const handleDeleteItem = useCallback(async () => {
    if (!selectedMedia || !onDeleteItem || deletingItem) return
    const ok = await confirmDanger({
      title: t('common:confirmTitle', { defaultValue: 'Please confirm' }),
      content: t('review:mediaPage.deleteItemConfirm', {
        defaultValue: 'Move this item to Trash? You can restore it from Trash.'
      }),
      okText: t('common:delete', { defaultValue: 'Delete' }),
      cancelText: t('common:cancel', { defaultValue: 'Cancel' })
    })
    if (!ok) return
    setDeletingItem(true)
    try {
      await onDeleteItem(selectedMedia, mediaDetail ?? null)
      setPendingDeleteId(String(selectedMedia.id))
      message.success(t('common:deleted', { defaultValue: 'Deleted' }))
    } catch (err) {
      const msg =
        err && typeof err === 'object' && 'message' in err
          ? String((err as { message?: unknown }).message)
          : ''
      message.error(msg || t('common:deleteFailed', { defaultValue: 'Delete failed' }))
    } finally {
      setDeletingItem(false)
    }
  }, [confirmDanger, deletingItem, mediaDetail, onDeleteItem, selectedMedia, t])

  const openContentEditModal = useCallback(() => {
    setEditingContentText(content)
    setContentEditModalOpen(true)
  }, [content])

  const openAnalysisEditModal = useCallback(() => {
    if (!activeAnalysis) return
    setEditingAnalysisText(activeAnalysis.text)
    setAnalysisEditModalOpen(true)
  }, [activeAnalysis])

  return {
    // Keywords
    editingKeywords,
    savingKeywords,
    handleSaveKeywords,
    // Content edit
    contentEditModalOpen,
    setContentEditModalOpen,
    editingContentText,
    openContentEditModal,
    // Analysis edit
    analysisEditModalOpen,
    setAnalysisEditModalOpen,
    editingAnalysisText,
    setEditingAnalysisText,
    optimisticAnalysis,
    setOptimisticAnalysis,
    activeAnalysisIndex,
    setActiveAnalysisIndex,
    analysisExpanded,
    setAnalysisExpanded,
    // Delete
    deletingItem,
    pendingDeleteId,
    isAwaitingSelectionUpdate,
    handleDeleteItem,
    // Derived
    latestVersion,
    derivedPrompt,
    derivedAnalysisContent,
    persistedAnalysisContent,
    existingAnalyses,
    activeAnalysis,
    selectedAnalysis,
    analysisText,
    analysisIsLong,
    analysisShown,
    openAnalysisEditModal
  }
}
