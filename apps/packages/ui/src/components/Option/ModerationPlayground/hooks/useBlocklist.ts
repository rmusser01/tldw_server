import React from "react"

import {
  appendManagedBlocklist,
  deleteManagedBlocklistItem,
  getBlocklist,
  getManagedBlocklist,
  lintBlocklist,
  updateBlocklist,
  type BlocklistLintResponse,
  type BlocklistManagedItem
} from "@/services/moderation"

export interface RawReplacePreview {
  previousText: string
  nextText: string
  addedCount: number
  removedCount: number
  changedCount: number
  lint: BlocklistLintResponse
}

export interface RawReplaceUndo {
  previousText: string
  replacedText: string
}

export interface BlocklistState {
  rawText: string
  setRawText: React.Dispatch<React.SetStateAction<string>>
  rawLint: BlocklistLintResponse | null
  pendingRawPreview: RawReplacePreview | null
  rawReplaceUndo: RawReplaceUndo | null
  isDirtyRaw: boolean
  managedItems: BlocklistManagedItem[]
  managedVersion: string
  managedLine: string
  setManagedLine: React.Dispatch<React.SetStateAction<string>>
  managedLint: BlocklistLintResponse | null
  loading: boolean
  loadRaw: () => Promise<void>
  saveRaw: () => Promise<void>
  saveRawText: (text: string) => Promise<void>
  previewRawReplace: (text?: string) => Promise<RawReplacePreview>
  confirmRawReplace: () => Promise<void>
  cancelRawReplace: () => void
  undoRawReplace: () => Promise<void>
  lintRaw: () => Promise<void>
  loadManaged: () => Promise<void>
  appendManaged: () => Promise<void>
  appendLine: (line: string) => Promise<void>
  deleteManaged: (itemId: number) => Promise<void>
  lintManagedLine: () => Promise<void>
  lintLine: (line: string) => Promise<BlocklistLintResponse>
}

const normalizeRawLines = (text: string): string[] =>
  text.split(/\r?\n/).map((line) => line.trimEnd())

const normalizeRawText = (text: string): string => normalizeRawLines(text).join("\n")

const summarizeLineChanges = (previousText: string, nextText: string) => {
  const previousLines = previousText ? previousText.split("\n") : []
  const nextLines = nextText ? nextText.split("\n") : []
  const sharedLength = Math.min(previousLines.length, nextLines.length)
  let changedCount = Math.abs(nextLines.length - previousLines.length)
  for (let index = 0; index < sharedLength; index += 1) {
    if (previousLines[index] !== nextLines[index]) changedCount += 1
  }
  return {
    addedCount: Math.max(0, nextLines.length - previousLines.length),
    removedCount: Math.max(0, previousLines.length - nextLines.length),
    changedCount
  }
}

const hasManagedLintMetadata = (item: BlocklistManagedItem): boolean =>
  item.pattern_type !== undefined

export function useBlocklist(): BlocklistState {
  const [rawText, setRawText] = React.useState("")
  const [rawLint, setRawLint] = React.useState<BlocklistLintResponse | null>(null)
  const [pendingRawPreview, setPendingRawPreview] = React.useState<RawReplacePreview | null>(null)
  const pendingRawPreviewRef = React.useRef<RawReplacePreview | null>(null)
  const [rawReplaceUndo, setRawReplaceUndo] = React.useState<RawReplaceUndo | null>(null)
  const [rawBaseline, setRawBaseline] = React.useState("")
  const rawBaselineRef = React.useRef("")
  const [managedItems, setManagedItems] = React.useState<BlocklistManagedItem[]>([])
  const [managedVersion, setManagedVersion] = React.useState("")
  const [managedLine, setManagedLine] = React.useState("")
  const [managedLint, setManagedLint] = React.useState<BlocklistLintResponse | null>(null)
  const [loading, setLoading] = React.useState(false)
  const isDirtyRaw = rawText !== rawBaseline

  const loadRaw = React.useCallback(async () => {
    setLoading(true)
    try {
      const lines = await getBlocklist()
      const nextText = (lines || []).join("\n")
      setRawText(nextText)
      setRawBaseline(nextText)
      rawBaselineRef.current = nextText
      setRawLint(null)
      setPendingRawPreview(null)
      pendingRawPreviewRef.current = null
      setRawReplaceUndo(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const previewRawReplace = React.useCallback(async (text = rawText): Promise<RawReplacePreview> => {
    setLoading(true)
    try {
      const lines = normalizeRawLines(text)
      const nextText = lines.join("\n")
      const lint = await lintBlocklist({ lines })
      const previousText = rawBaselineRef.current
      const summary = summarizeLineChanges(previousText, nextText)
      const preview: RawReplacePreview = {
        previousText,
        nextText,
        ...summary,
        lint
      }
      setRawLint(lint)
      setPendingRawPreview(preview)
      pendingRawPreviewRef.current = preview
      return preview
    } finally {
      setLoading(false)
    }
  }, [rawText])

  const confirmRawReplace = React.useCallback(async () => {
    const preview = pendingRawPreviewRef.current ?? pendingRawPreview
    if (!preview) throw new Error("Preview blocklist replacement first")
    if (preview.lint.invalid_count > 0) {
      throw new Error("Fix invalid blocklist rows before replacing")
    }
    setLoading(true)
    try {
      const lines = normalizeRawLines(preview.nextText)
      await updateBlocklist(lines)
      const nextText = lines.join("\n")
      setRawText(nextText)
      setRawBaseline(nextText)
      rawBaselineRef.current = nextText
      setRawLint(null)
      setPendingRawPreview(null)
      pendingRawPreviewRef.current = null
      setRawReplaceUndo({
        previousText: preview.previousText,
        replacedText: nextText
      })
    } finally {
      setLoading(false)
    }
  }, [pendingRawPreview])

  const cancelRawReplace = React.useCallback(() => {
    setPendingRawPreview(null)
    pendingRawPreviewRef.current = null
  }, [])

  const undoRawReplace = React.useCallback(async () => {
    if (!rawReplaceUndo) return
    setLoading(true)
    try {
      const previousText = normalizeRawText(rawReplaceUndo.previousText)
      const lines = normalizeRawLines(previousText)
      await updateBlocklist(lines)
      setRawText(previousText)
      setRawBaseline(previousText)
      rawBaselineRef.current = previousText
      setRawLint(null)
      setPendingRawPreview(null)
      pendingRawPreviewRef.current = null
      setRawReplaceUndo(null)
    } finally {
      setLoading(false)
    }
  }, [rawReplaceUndo])

  const saveRaw = React.useCallback(async () => {
    await previewRawReplace(rawText)
  }, [previewRawReplace, rawText])

  const saveRawText = React.useCallback(async (text: string) => {
    await previewRawReplace(text)
  }, [previewRawReplace])

  const lintRaw = React.useCallback(async () => {
    setLoading(true)
    try {
      const lines = rawText.split(/\r?\n/)
      const lint = await lintBlocklist({ lines })
      setRawLint(lint)
    } finally {
      setLoading(false)
    }
  }, [rawText])

  const mergeManagedLintMetadata = React.useCallback(async (items: BlocklistManagedItem[]) => {
    if (!items.length || items.every(hasManagedLintMetadata)) return items
    const lint = await lintBlocklist({ lines: items.map((item) => item.line) })
    return items.map((item, index) => {
      const lintItem = lint.items.find((entry) => entry.index === index)
      if (!lintItem) return item
      return {
        ...lintItem,
        ...item,
        line: item.line
      }
    })
  }, [])

  const loadManagedState = React.useCallback(async () => {
    const { data, etag } = await getManagedBlocklist()
    const enrichedItems = await mergeManagedLintMetadata(data.items || [])
    setManagedItems(enrichedItems)
    setManagedVersion(data.version || etag || "")
  }, [mergeManagedLintMetadata])

  const loadManaged = React.useCallback(async () => {
    setLoading(true)
    try {
      await loadManagedState()
    } finally {
      setLoading(false)
    }
  }, [loadManagedState])

  const appendManaged = React.useCallback(async () => {
    if (!managedVersion) throw new Error("Load the managed blocklist first")
    const line = managedLine.trim()
    if (!line) throw new Error("Enter a line to append")
    setLoading(true)
    try {
      await appendManagedBlocklist(managedVersion, line)
      setManagedLine("")
      await loadManagedState()
    } finally {
      setLoading(false)
    }
  }, [managedVersion, managedLine, loadManagedState])

  const appendLine = React.useCallback(async (line: string) => {
    if (!managedVersion) throw new Error("Load the managed blocklist first")
    const trimmed = line.trim()
    if (!trimmed) throw new Error("Enter a line to append")
    setLoading(true)
    try {
      await appendManagedBlocklist(managedVersion, trimmed)
      setManagedLine("")
      await loadManagedState()
    } finally {
      setLoading(false)
    }
  }, [managedVersion, loadManagedState])

  const deleteManaged = React.useCallback(async (itemId: number) => {
    if (!managedVersion) return
    setLoading(true)
    try {
      await deleteManagedBlocklistItem(managedVersion, itemId)
      await loadManagedState()
    } finally {
      setLoading(false)
    }
  }, [managedVersion, loadManagedState])

  const lintManagedLine = React.useCallback(async () => {
    if (!managedLine.trim()) throw new Error("Enter a line to lint")
    setLoading(true)
    try {
      const lint = await lintBlocklist({ line: managedLine.trim() })
      setManagedLint(lint)
    } finally {
      setLoading(false)
    }
  }, [managedLine])

  const lintLine = React.useCallback(async (line: string): Promise<BlocklistLintResponse> => {
    if (!line.trim()) throw new Error("Enter a line to lint")
    setLoading(true)
    try {
      const lint = await lintBlocklist({ line: line.trim() })
      setManagedLint(lint)
      return lint
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    rawText,
    setRawText,
    rawLint,
    pendingRawPreview,
    rawReplaceUndo,
    isDirtyRaw,
    managedItems,
    managedVersion,
    managedLine,
    setManagedLine,
    managedLint,
    loading,
    loadRaw,
    saveRaw,
    saveRawText,
    previewRawReplace,
    confirmRawReplace,
    cancelRawReplace,
    undoRawReplace,
    lintRaw,
    loadManaged,
    appendManaged,
    appendLine,
    deleteManaged,
    lintManagedLine,
    lintLine
  }
}
