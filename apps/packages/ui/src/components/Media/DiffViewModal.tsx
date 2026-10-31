import { useState, useMemo, useRef, useEffect, type RefObject } from 'react'
import { Modal, Radio } from 'antd'
import { useTranslation } from 'react-i18next'
import {
  DIFF_HARD_CHAR_THRESHOLD,
  computeDiffSync,
  computeDiffWithWorker,
  sampleTextForDiff,
  shouldRequireSampling,
  shouldUseWorkerDiff,
  type DiffLine
} from './diff-worker-client'

interface DiffViewModalProps {
  open: boolean
  onClose: () => void
  leftText: string
  rightText: string
  leftLabel?: string
  rightLabel?: string
  fallbackFocusRef?: RefObject<HTMLElement | null>
  metadataDiff?: {
    left?: string[]
    right?: string[]
    changed?: string[]
  }
}

export function DiffViewModal({
  open,
  onClose,
  leftText,
  rightText,
  leftLabel = 'Left',
  rightLabel = 'Right',
  fallbackFocusRef,
  metadataDiff
}: DiffViewModalProps) {
  const { t } = useTranslation(['review'])
  const [viewMode, setViewMode] = useState<'unified' | 'sideBySide'>('unified')
  const [workerDiffLines, setWorkerDiffLines] = useState<DiffLine[] | null>(null)
  const [diffError, setDiffError] = useState<string | null>(null)
  const [diffLoading, setDiffLoading] = useState(false)
  const [samplingAccepted, setSamplingAccepted] = useState(false)
  const triggerRef = useRef<HTMLElement | null>(null)
  const contentRef = useRef<HTMLDivElement>(null)

  // Capture trigger element for focus restoration
  useEffect(() => {
    if (open) {
      triggerRef.current = document.activeElement as HTMLElement
    }
  }, [open])

  const handleAfterOpenChange = (visible: boolean) => {
    if (visible) {
      // Focus the diff content for keyboard scrolling
      setTimeout(() => contentRef.current?.focus(), 0)
    } else {
      const canFocus = (candidate: HTMLElement | null): candidate is HTMLElement =>
        Boolean(
          candidate &&
            candidate.isConnected &&
            typeof candidate.focus === 'function' &&
            !(candidate as HTMLButtonElement).disabled
        )

      // Restore focus to trigger element when modal closes, then fallback to compare trigger.
      if (canFocus(triggerRef.current)) {
        triggerRef.current.focus()
        return
      }
      const explicitFallback = fallbackFocusRef?.current ?? null
      if (canFocus(explicitFallback)) {
        explicitFallback.focus()
        return
      }
      const compareFallback = document.querySelector<HTMLElement>(
        "[data-compare-content-trigger='true']"
      )
      if (canFocus(compareFallback)) {
        compareFallback.focus()
      }
    }
  }

  const requiresSampling = useMemo(
    () => shouldRequireSampling(leftText, rightText, DIFF_HARD_CHAR_THRESHOLD),
    [leftText, rightText]
  )
  const canComputeDiff = !requiresSampling || samplingAccepted

  useEffect(() => {
    if (!open) return
    setSamplingAccepted(false)
  }, [open, leftText, rightText])

  const effectiveLeftText = useMemo(
    () => (requiresSampling && samplingAccepted ? sampleTextForDiff(leftText) : leftText),
    [leftText, requiresSampling, samplingAccepted]
  )
  const effectiveRightText = useMemo(
    () => (requiresSampling && samplingAccepted ? sampleTextForDiff(rightText) : rightText),
    [rightText, requiresSampling, samplingAccepted]
  )

  const useWorker = useMemo(
    () => canComputeDiff && shouldUseWorkerDiff(effectiveLeftText, effectiveRightText),
    [canComputeDiff, effectiveLeftText, effectiveRightText]
  )

  const syncDiffLines = useMemo(
    () => (open && canComputeDiff && !useWorker ? computeDiffSync(effectiveLeftText, effectiveRightText) : []),
    [open, canComputeDiff, useWorker, effectiveLeftText, effectiveRightText]
  )

  useEffect(() => {
    if (!open || !canComputeDiff) {
      setWorkerDiffLines(null)
      setDiffError(null)
      setDiffLoading(false)
      return
    }
    if (!useWorker) {
      setWorkerDiffLines(null)
      setDiffError(null)
      setDiffLoading(false)
      return
    }

    const controller = new AbortController()
    setDiffLoading(true)
    setDiffError(null)
    setWorkerDiffLines(null)

    void computeDiffWithWorker(effectiveLeftText, effectiveRightText, controller.signal)
      .then((lines) => {
        if (controller.signal.aborted) return
        setWorkerDiffLines(lines)
        setDiffLoading(false)
      })
      .catch((error) => {
        if (controller.signal.aborted) return
        setDiffError(error instanceof Error ? error.message : 'Diff computation failed')
        setWorkerDiffLines(computeDiffSync(effectiveLeftText, effectiveRightText))
        setDiffLoading(false)
      })

    return () => {
      controller.abort()
    }
  }, [open, canComputeDiff, useWorker, effectiveLeftText, effectiveRightText])

  const diffLines = useMemo(() => {
    if (!open || !canComputeDiff) return []
    if (!useWorker) return syncDiffLines
    return workerDiffLines || []
  }, [open, canComputeDiff, useWorker, syncDiffLines, workerDiffLines])

  // Build side-by-side view data
  const sideBySideData = useMemo(() => {
    const left: Array<{ num: number; text: string; type: 'same' | 'del' | 'empty' }> = []
    const right: Array<{ num: number; text: string; type: 'same' | 'add' | 'empty' }> = []

    let leftNum = 1
    let rightNum = 1

    for (const line of diffLines) {
      if (line.type === 'same') {
        left.push({ num: leftNum++, text: line.text, type: 'same' })
        right.push({ num: rightNum++, text: line.text, type: 'same' })
      } else if (line.type === 'del') {
        left.push({ num: leftNum++, text: line.text, type: 'del' })
        right.push({ num: 0, text: '', type: 'empty' })
      } else {
        left.push({ num: 0, text: '', type: 'empty' })
        right.push({ num: rightNum++, text: line.text, type: 'add' })
      }
    }

    return { left, right }
  }, [diffLines])

  const hasMetadataDiff = useMemo(() => {
    const leftCount = metadataDiff?.left?.length || 0
    const rightCount = metadataDiff?.right?.length || 0
    const changedCount = metadataDiff?.changed?.length || 0
    return leftCount > 0 || rightCount > 0 || changedCount > 0
  }, [metadataDiff])

  const getLineClass = (type: string) => {
    switch (type) {
      case 'add':
        return 'bg-success/10 text-success'
      case 'del':
        return 'bg-danger/10 text-danger'
      case 'empty':
        return 'bg-surface2'
      default:
        return 'text-text'
    }
  }

  const getLinePrefix = (type: string) => {
    switch (type) {
      case 'add':
        return '+'
      case 'del':
        return '-'
      default:
        return ' '
    }
  }

  return (
    <Modal
      title={t('mediaPage.diffView', 'Diff View')}
      open={open}
      onCancel={onClose}
      afterOpenChange={handleAfterOpenChange}
      footer={null}
      width={900}
      className="diff-modal"
    >
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm text-text-muted">
            <span className="font-medium text-danger">{leftLabel}</span>
            {' → '}
            <span className="font-medium text-success">{rightLabel}</span>
          </span>
        </div>
        <Radio.Group
          value={viewMode}
          onChange={e => setViewMode(e.target.value)}
          size="small"
        >
          <Radio.Button value="unified">{t('mediaPage.unified', 'Unified')}</Radio.Button>
          <Radio.Button value="sideBySide">{t('mediaPage.sideBySide', 'Side by Side')}</Radio.Button>
        </Radio.Group>
      </div>

      {hasMetadataDiff && (
        <div className="mb-3 rounded-lg border border-border bg-surface2 p-2 text-xs">
          {(metadataDiff?.changed?.length || 0) > 0 && (
            <div className="mb-2">
              <div className="font-medium text-text-muted">
                {t('mediaPage.metadataChanges', 'Metadata changes')}
              </div>
              <ul className="mt-1 list-disc pl-4 text-text">
                {(metadataDiff?.changed || []).map((entry, idx) => (
                  <li key={`metadata-change-${idx}`}>{entry}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div>
              <div className="font-medium text-text-muted mb-1">{leftLabel}</div>
              <div className="space-y-0.5 text-text">
                {(metadataDiff?.left || []).length > 0 ? (
                  (metadataDiff?.left || []).map((entry, idx) => (
                    <div key={`metadata-left-${idx}`}>{entry}</div>
                  ))
                ) : (
                  <div className="text-text-subtle">—</div>
                )}
              </div>
            </div>
            <div>
              <div className="font-medium text-text-muted mb-1">{rightLabel}</div>
              <div className="space-y-0.5 text-text">
                {(metadataDiff?.right || []).length > 0 ? (
                  (metadataDiff?.right || []).map((entry, idx) => (
                    <div key={`metadata-right-${idx}`}>{entry}</div>
                  ))
                ) : (
                  <div className="text-text-subtle">—</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {requiresSampling && !samplingAccepted && (
        <div className="mb-3 rounded-lg border border-warn/50 bg-warn/10 p-3 text-sm text-text">
          <div className="font-medium">
            {t(
              'mediaPage.largeComparisonTitle',
              'Large comparison detected'
            )}
          </div>
          <div className="mt-1 text-xs text-text-muted">
            {t(
              'mediaPage.largeComparisonBody',
              'These documents are very large. Generate a sampled diff to keep this responsive.'
            )}
          </div>
          <button
            type="button"
            className="mt-2 rounded border border-border bg-surface px-2 py-1 text-xs hover:bg-surface2"
            onClick={() => setSamplingAccepted(true)}
          >
            {t('mediaPage.generateSampledDiff', 'Generate sampled diff')}
          </button>
        </div>
      )}

      {useWorker && canComputeDiff && (
        <div
          className="mb-3 rounded border border-primary/40 bg-primary/10 px-2 py-1 text-xs text-primary"
          role="status"
          aria-live="polite"
        >
          {diffLoading
            ? t('mediaPage.largeComparisonComputing', 'Large comparison: computing diff in background…')
            : t('mediaPage.largeComparisonReady', 'Large comparison finished.')}
        </div>
      )}

      {diffError && (
        <div className="mb-3 rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs text-danger">
          {t('mediaPage.largeComparisonFallback', 'Diff worker failed; fell back to direct diff.')}
        </div>
      )}

      <div
        ref={contentRef}
        tabIndex={0}
        className="border border-border rounded-lg overflow-hidden max-h-[60vh] overflow-y-auto focus:outline-none focus:ring-2 focus:ring-focus"
        onKeyDown={(e) => {
          // Keyboard navigation for diff scrolling
          const scrollAmount = 100
          const pageScrollAmount = 400
          if (e.key === 'ArrowDown' || e.key === 'j') {
            e.preventDefault()
            contentRef.current?.scrollBy({ top: scrollAmount, behavior: 'smooth' })
          } else if (e.key === 'ArrowUp' || e.key === 'k') {
            e.preventDefault()
            contentRef.current?.scrollBy({ top: -scrollAmount, behavior: 'smooth' })
          } else if (e.key === 'PageDown' || e.key === ' ') {
            e.preventDefault()
            contentRef.current?.scrollBy({ top: pageScrollAmount, behavior: 'smooth' })
          } else if (e.key === 'PageUp') {
            e.preventDefault()
            contentRef.current?.scrollBy({ top: -pageScrollAmount, behavior: 'smooth' })
          } else if (e.key === 'Home') {
            e.preventDefault()
            contentRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
          } else if (e.key === 'End') {
            e.preventDefault()
            contentRef.current?.scrollTo({ top: contentRef.current.scrollHeight, behavior: 'smooth' })
          }
        }}
        role="region"
        aria-label={t('mediaPage.diffContentRegion', 'Diff content - use arrow keys to scroll')}
      >
        {!canComputeDiff ? (
          <div className="p-4 text-center text-text-muted">
            {t('mediaPage.largeComparisonPending', 'Generate sampled diff to continue.')}
          </div>
        ) : viewMode === 'unified' ? (
          <div className="font-mono text-xs">
            {diffLines.length === 0 ? (
              <div className="p-4 text-center text-text-muted">
                {t('mediaPage.noDifferences', 'No differences found')}
              </div>
            ) : (
              diffLines.map((line, idx) => (
                <div
                  key={idx}
                  className={`px-3 py-0.5 ${getLineClass(line.type)}`}
                >
                  <span className="select-none text-text-subtle mr-2">
                    {getLinePrefix(line.type)}
                  </span>
                  <span className="whitespace-pre-wrap break-all">{line.text || ' '}</span>
                </div>
              ))
            )}
          </div>
        ) : (
          <div className="flex font-mono text-xs">
            {/* Left side */}
            <div className="flex-1 border-r border-border">
              <div className="px-3 py-1 bg-surface2 text-text-muted font-medium border-b border-border">
                {leftLabel}
              </div>
              {sideBySideData.left.map((line, idx) => (
                <div
                  key={idx}
                  className={`px-3 py-0.5 flex ${getLineClass(line.type)}`}
                >
                  <span className="w-8 text-right text-text-subtle mr-2 select-none flex-shrink-0">
                    {line.num > 0 ? line.num : ''}
                  </span>
                  <span className="whitespace-pre-wrap break-all flex-1">{line.text || ' '}</span>
                </div>
              ))}
            </div>
            {/* Right side */}
            <div className="flex-1">
              <div className="px-3 py-1 bg-surface2 text-text-muted font-medium border-b border-border">
                {rightLabel}
              </div>
              {sideBySideData.right.map((line, idx) => (
                <div
                  key={idx}
                  className={`px-3 py-0.5 flex ${getLineClass(line.type)}`}
                >
                  <span className="w-8 text-right text-text-subtle mr-2 select-none flex-shrink-0">
                    {line.num > 0 ? line.num : ''}
                  </span>
                  <span className="whitespace-pre-wrap break-all flex-1">{line.text || ' '}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Legend & keyboard hints */}
      <div className="mt-3 flex items-center justify-between text-xs text-text-muted">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-danger/20"></span>
            {t('mediaPage.removed', 'Removed')}
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-success/20"></span>
            {t('mediaPage.added', 'Added')}
          </span>
        </div>
        <span className="hidden sm:block">
          {t('mediaPage.keyboardNavHint', '↑↓ or j/k to scroll, PgUp/PgDn for pages')}
        </span>
      </div>
    </Modal>
  )
}
