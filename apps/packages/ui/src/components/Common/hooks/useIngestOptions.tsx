import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import { useStorage } from '@plasmohq/storage/hook'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { tldwModels } from '@/services/tldw'
import {
  defaultEmbeddingModelForRag,
} from '@/services/tldw-server'
import {
  QUICK_INGEST_SCHEMA_FALLBACK,
  QUICK_INGEST_SCHEMA_FALLBACK_VERSION
} from '@/services/tldw/fallback-schemas'
import { useConfirmDanger } from '@/components/Common/confirm-danger'
import type { TypeDefaults, Entry } from './useIngestQueue'
import { DEFAULT_TYPE_DEFAULTS } from './useIngestQueue'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AdvSchemaEntry = {
  name: string
  type: string
  enum?: any[]
  description?: string
  title?: string
}

type QuickIngestSpecCache = {
  entries: AdvSchemaEntry[]
  source: 'server' | 'fallback'
  cachedAt: number
  version?: string
}

type IngestConnectionStatus = "online" | "offline" | "unconfigured" | "unknown"

// ---------------------------------------------------------------------------
// Module-level spec cache
// ---------------------------------------------------------------------------

const SPEC_CACHE_TTL_MS = 60 * 60 * 1000
const SPEC_FALLBACK_TTL_MS = 5 * 60 * 1000
let quickIngestSpecCache: QuickIngestSpecCache | null = null

const readSpecCache = (preferServer: boolean) => {
  const cache = quickIngestSpecCache
  if (!cache) return null
  if (!preferServer && cache.source !== 'fallback') return null
  const maxAge =
    cache.source === 'server' ? SPEC_CACHE_TTL_MS : SPEC_FALLBACK_TTL_MS
  if (Date.now() - cache.cachedAt > maxAge) return null
  return cache
}

const writeSpecCache = (next: Omit<QuickIngestSpecCache, 'cachedAt'>) => {
  quickIngestSpecCache = { ...next, cachedAt: Date.now() }
}

const SAVE_DEBOUNCE_MS = 2000

const normalizeCommonOptions = (common?: {
  perform_analysis?: boolean
  perform_chunking?: boolean
  overwrite_existing?: boolean
  chunking_mode?: string
  auto_chunking_goal?: string
  auto_chunking_use_llm?: boolean
}) => {
  const chunkingGoal: "balanced" | "qa_search" | "navigation_summary" =
    common?.auto_chunking_goal === "qa_search" ||
    common?.auto_chunking_goal === "navigation_summary"
      ? common.auto_chunking_goal
      : "balanced"
  const chunkingMode: "auto" | "manual" =
    common?.chunking_mode === "manual" ? "manual" : "auto"
  return {
    ...(common ?? {}),
    perform_analysis: common?.perform_analysis ?? true,
    perform_chunking: common?.perform_chunking ?? true,
    overwrite_existing: common?.overwrite_existing ?? false,
    chunking_mode: chunkingMode,
    auto_chunking_goal: chunkingGoal,
    auto_chunking_use_llm: common?.auto_chunking_use_llm === true
  }
}

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UseIngestOptionsDeps {
  open: boolean
  running: boolean
  ingestConnectionStatus: IngestConnectionStatus
  messageApi: MessageInstance
  qi: (key: string, defaultValue: string, options?: Record<string, any>) => string
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useIngestOptions(deps: UseIngestOptionsDeps) {
  const {
    open,
    running,
    ingestConnectionStatus,
    messageApi,
    qi,
  } = deps

  const confirmDanger = useConfirmDanger()
  const fallbackSchemaVersion = QUICK_INGEST_SCHEMA_FALLBACK_VERSION

  // ---- persisted options state ----
  const [storeRemote, setStoreRemote] = useStorage<boolean>(
    "quickIngestStoreRemote",
    true
  )
  const [reviewBeforeStorage, setReviewBeforeStorage] = useStorage<boolean>(
    "quickIngestReviewBeforeStorage",
    false
  )
  const [typeDefaults, setTypeDefaults] = useStorage<TypeDefaults>(
    "quickIngestTypeDefaults",
    DEFAULT_TYPE_DEFAULTS
  )
  const [common, setCommon] = useStorage<{
    perform_analysis: boolean
    perform_chunking: boolean
    overwrite_existing: boolean
    chunking_mode?: "auto" | "manual"
    auto_chunking_goal?: "balanced" | "qa_search" | "navigation_summary"
    auto_chunking_use_llm?: boolean
  }>("quickIngestCommon", {
    perform_analysis: true,
    perform_chunking: true,
    overwrite_existing: false,
    chunking_mode: "auto",
    auto_chunking_goal: "balanced",
    auto_chunking_use_llm: false
  })
  const [savedAdvValues, setSavedAdvValues] = useStorage<Record<string, any>>('quickIngestAdvancedValues', {})
  const [uiPrefs, setUiPrefs] = useStorage<{ advancedOpen?: boolean; fieldDetailsOpen?: Record<string, boolean> }>('quickIngestAdvancedUI', {})
  const [specPrefs, setSpecPrefs] = useStorage<{ preferServer?: boolean; lastRemote?: { version?: string; cachedAt?: number } }>('quickIngestSpecPrefs', { preferServer: true })
  const [storageHintSeen, setStorageHintSeen] = useStorage<boolean>('quickIngestStorageHintSeen', false)
  const [reviewDraftWarning, setReviewDraftWarning] = useStorage<boolean>(
    "quickIngestReviewWarningSeen",
    false
  )

  // ---- session state ----
  const [chunkingTemplateName, setChunkingTemplateName] = React.useState<string | undefined>(undefined)
  const [autoApplyTemplate, setAutoApplyTemplate] = React.useState<boolean>(false)
  const [advancedOpen, setAdvancedOpen] = React.useState<boolean>(false)
  const [advancedValues, setAdvancedValues] = React.useState<Record<string, any>>({})
  const [advSchema, setAdvSchema] = React.useState<AdvSchemaEntry[]>([])
  const [specSource, setSpecSource] = React.useState<'server' | 'fallback' | 'none'>('none')
  const [fieldDetailsOpen, setFieldDetailsOpen] = React.useState<Record<string, boolean>>({})
  const [advSearch, setAdvSearch] = React.useState<string>('')
  const [transcriptionModelOptions, setTranscriptionModelOptions] = React.useState<string[]>([])
  const [transcriptionModelsLoading, setTranscriptionModelsLoading] = React.useState(false)
  const [ragEmbeddingLabel, setRagEmbeddingLabel] = React.useState<string | null>(null)

  // ---- refs ----
  const lastSavedAdvValuesRef = React.useRef<string | null>(null)
  const lastSavedUiPrefsRef = React.useRef<string | null>(null)
  const specPrefsCacheRef = React.useRef<string | null>(null)
  const advSchemaRef = React.useRef(advSchema)
  const advancedHydratedRef = React.useRef(false)
  const uiPrefsHydratedRef = React.useRef(false)

  // ---- derived ----
  const normalizedTypeDefaults = React.useMemo(
    () => ({
      audio: typeDefaults?.audio,
      document: {
        ocr: typeDefaults?.document?.ocr ?? true
      },
      video: typeDefaults?.video
    }),
    [typeDefaults]
  )

  const processOnly = reviewBeforeStorage || !storeRemote
  const shouldStoreRemote = storeRemote && !processOnly

  const modifiedAdvancedCount = React.useMemo(
    () => Object.keys(advancedValues || {}).length,
    [advancedValues]
  )

  const advancedDefaultsDirty = React.useMemo(() => {
    const current = JSON.stringify(advancedValues || {})
    const saved = JSON.stringify(savedAdvValues || {})
    return current !== saved
  }, [advancedValues, savedAdvValues])

  const hasTypeDefaultChanges = React.useMemo(() => {
    const baselineAudio = DEFAULT_TYPE_DEFAULTS.audio
    const baselineDocument = DEFAULT_TYPE_DEFAULTS.document
    const baselineVideo = DEFAULT_TYPE_DEFAULTS.video
    const currentAudio = normalizedTypeDefaults.audio
    const currentDocument = normalizedTypeDefaults.document
    const currentVideo = normalizedTypeDefaults.video

    const audioChanged =
      (currentAudio?.language ?? null) !== (baselineAudio?.language ?? null) ||
      (currentAudio?.diarize ?? null) !== (baselineAudio?.diarize ?? null)
    const documentChanged =
      (currentDocument?.ocr ?? true) !== (baselineDocument?.ocr ?? true)
    const videoChanged =
      (currentVideo?.captions ?? null) !== (baselineVideo?.captions ?? null)

    return audioChanged || documentChanged || videoChanged
  }, [normalizedTypeDefaults])

  const lastRefreshedLabel = React.useMemo(() => {
    const ts = specPrefs?.lastRemote?.cachedAt
    if (!ts) return null
    const d = new Date(ts)
    return d.toLocaleString()
  }, [specPrefs])

  const preferServerSpec =
    typeof specPrefs?.preferServer === 'boolean' ? specPrefs.preferServer : true

  const specSourceLabel = React.useMemo(() => {
    if (specSource === 'server') {
      return qi('specSourceLive', 'Live server spec')
    }
    if (specSource === 'fallback') {
      return qi('specSourceFallback', 'Fallback spec')
    }
    return null
  }, [qi, specSource])

  // ---- advanced value setter ----
  const setAdvancedValue = React.useCallback((name: string, value: any) => {
    setAdvancedValues((prev) => {
      const next = { ...(prev || {}) }
      if (value === undefined || value === null || value === '') {
        delete next[name]
      } else {
        next[name] = value
      }
      return next
    })
  }, [])

  // ---- transcription model ----
  const resolvedAdvSchema = React.useMemo(() => {
    if (transcriptionModelOptions.length === 0) return advSchema
    return advSchema.map((field) =>
      field.name === "transcription_model"
        ? { ...field, enum: transcriptionModelOptions }
        : field
    )
  }, [advSchema, transcriptionModelOptions])

  const transcriptionModelChoices = React.useMemo(() => {
    const fallbackField = advSchema.find(
      (field) => field.name === "transcription_model"
    )
    const fallbackEnum = Array.isArray(fallbackField?.enum)
      ? fallbackField.enum.map((value) => String(value))
      : []
    const source =
      transcriptionModelOptions.length > 0
        ? transcriptionModelOptions
        : fallbackEnum
    const seen = new Set<string>()
    return source.reduce<string[]>((acc, value) => {
      const normalized = String(value)
      if (!normalized || seen.has(normalized)) return acc
      seen.add(normalized)
      acc.push(normalized)
      return acc
    }, [])
  }, [advSchema, transcriptionModelOptions])

  const transcriptionModelValue = React.useMemo(() => {
    const value = advancedValues?.transcription_model
    if (value === undefined || value === null || value === "") {
      return undefined
    }
    return String(value)
  }, [advancedValues])

  const handleTranscriptionModelChange = React.useCallback(
    (value?: string) => {
      setAdvancedValue("transcription_model", value)
    },
    [setAdvancedValue]
  )

  // ---- storage label ----
  const storageLabel = React.useMemo(() => {
    if (!storeRemote) {
      return qi('process', 'Process locally')
    }
    if (reviewBeforeStorage) {
      return qi("storeAfterReview", "Store after review")
    }
    return qi('storeRemote', 'Store to remote DB')
  }, [qi, reviewBeforeStorage, storeRemote])

  // ---- review toggle ----
  const handleReviewToggle = React.useCallback(
    async (checked: boolean) => {
      if (checked && !reviewDraftWarning) {
        const ok = await confirmDanger({
          title: qi(
            "reviewWarningTitle",
            "Store drafts locally?"
          ),
          content: (
            <div className="space-y-2 text-sm text-text-muted">
              <p>
                {qi(
                  "reviewWarningBody",
                  "Drafts are saved in this browser and may include sensitive data. You can clear drafts from the Content Review page at any time."
                )}
              </p>
              <p>
                {qi(
                  "reviewWarningBodySecondary",
                  "Review mode keeps content client-side until you commit it to your server."
                )}
              </p>
            </div>
          ),
          okText: qi("reviewWarningConfirm", "Enable review"),
          cancelText: qi("reviewWarningCancel", "Cancel"),
          danger: false,
          autoFocusButton: "ok"
        })
        if (!ok) return
        setReviewDraftWarning(true)
      }
      setReviewBeforeStorage(checked)
    },
    [
      confirmDanger,
      qi,
      reviewDraftWarning,
      setReviewBeforeStorage,
      setReviewDraftWarning
    ]
  )

  // ---- spec persist helper ----
  const persistSpecPrefs = React.useCallback(
    (next: { preferServer?: boolean; lastRemote?: { version?: string; cachedAt?: number } }) => {
      const serialized = JSON.stringify(next || {})
      if (specPrefsCacheRef.current === serialized) return
      specPrefsCacheRef.current = serialized
      setSpecPrefs(next)
    },
    [setSpecPrefs]
  )

  // ---- parseSpec ----
  const parseSpec = React.useCallback((spec: any) => {
    const getByRef = (ref: string): any => {
      if (!ref || typeof ref !== 'string' || !ref.startsWith('#/')) return null
      const parts = ref.slice(2).split('/')
      let cur: any = spec
      for (const p of parts) {
        if (cur && typeof cur === 'object' && p in cur) cur = cur[p]
        else return null
      }
      return cur
    }

    const resolveRef = (schema: any, seen = new Set<string>()): any => {
      if (!schema) return {}
      if (schema.$ref) {
        const ref = String(schema.$ref)
        if (seen.has(ref)) {
          return { type: 'string', description: 'Unresolvable schema cycle' }
        }
        seen.add(ref)
        const target = getByRef(ref)
        return target ? resolveRef(target, seen) : {}
      }
      return schema
    }

    const mergeProps = (schema: any, stack: WeakSet<object>, allowVisited = false): Record<string, any> => {
      const s = resolveRef(schema)
      let props: Record<string, any> = {}
      if (!s || typeof s !== 'object' || Array.isArray(s)) return props
      const already = stack.has(s as object)
      if (already && !allowVisited) return props
      if (!already) stack.add(s as object)
      try {
        for (const key of ['allOf', 'oneOf', 'anyOf'] as const) {
          if (Array.isArray((s as any)[key])) {
            for (const sub of (s as any)[key]) {
              props = { ...props, ...mergeProps(sub, stack) }
            }
          }
        }
        if ((s as any).properties && typeof (s as any).properties === 'object') {
          for (const [k, v] of Object.entries<any>((s as any).properties)) {
            props[k] = resolveRef(v)
          }
        }
        return props
      } finally {
        if (!already) stack.delete(s as object)
      }
    }

    const flattenProps = (obj: Record<string, any>, parent = '', stack: WeakSet<object>): Array<[string, any]> => {
      const out: Array<[string, any]> = []
      for (const [k, v0] of Object.entries<any>(obj || {})) {
        const v = resolveRef(v0)
        const name = parent ? `${parent}.${k}` : k
        const isObj = (v?.type === 'object' && v?.properties && typeof v.properties === 'object')
        if (isObj) {
          const node = v as unknown
          if (!node || typeof node !== 'object' || Array.isArray(node) || stack.has(node as object)) {
            out.push([name, v])
            continue
          }
          stack.add(node as object)
          try {
            const child = mergeProps(v, stack, true)
            out.push(...flattenProps(child, name, stack))
          } finally {
            stack.delete(node as object)
          }
        } else {
          out.push([name, v])
        }
      }
      return out
    }

    const extractSchemaFromPath = (paths: Record<string, any>, candidates: string[]) => {
      for (const candidate of candidates) {
        const entry = paths?.[candidate]
        const content = entry?.post?.requestBody?.content
        if (!content) continue
        const schemaSource =
          content['multipart/form-data'] ||
          content['application/x-www-form-urlencoded'] ||
          content['application/json'] ||
          {}
        const schema = schemaSource?.schema
        if (schema) return schema
      }
      return null
    }

    const extractEntriesFromSchema = (rootSchema: any) => {
      if (!rootSchema) return []
      const stack = new WeakSet<object>()
      const rootResolved = resolveRef(rootSchema)
      if (rootResolved && typeof rootResolved === 'object' && !Array.isArray(rootResolved)) {
        stack.add(rootResolved)
      }
      let props: Record<string, any> = {}
      let flat: Array<[string, any]> = []
      try {
        props = mergeProps(rootSchema, stack, true)
        flat = flattenProps(props, '', stack)
      } finally {
        if (rootResolved && typeof rootResolved === 'object' && !Array.isArray(rootResolved)) {
          stack.delete(rootResolved)
        }
      }
      const entries: AdvSchemaEntry[] = []
      const exclude = new Set(['urls', 'media_type'])
      for (const [name, def0] of flat) {
        if (exclude.has(name)) continue
        const def = resolveRef(def0)
        let type: string = 'string'
        if (def.type) type = Array.isArray(def.type) ? String(def.type[0]) : String(def.type)
        else if (def.enum) type = 'string'
        else if (def.anyOf || def.oneOf) type = 'string'
        const en = Array.isArray(def?.enum) ? def.enum : undefined
        const description = def?.description || def?.title || undefined
        entries.push({ name, type, enum: en, description, title: def?.title })
      }
      return entries
    }

    const mergeSchemaEntries = (...lists: AdvSchemaEntry[][]) => {
      const byName = new Map<string, AdvSchemaEntry>()
      for (const list of lists) {
        for (const entry of list) {
          const existing = byName.get(entry.name)
          if (!existing) {
            byName.set(entry.name, { ...entry })
            continue
          }
          byName.set(entry.name, {
            ...existing,
            ...entry,
            type: entry.type || existing.type,
            enum: entry.enum ?? existing.enum,
            description: entry.description ?? existing.description,
            title: entry.title ?? existing.title
          })
        }
      }
      return Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name))
    }

    const paths = spec?.paths || {}
    const mediaAddSchema = extractSchemaFromPath(paths, [
      '/api/v1/media/ingest/jobs',
      '/api/v1/media/ingest/jobs/'
    ])
    const webScrapeSchema = extractSchemaFromPath(paths, [
      '/api/v1/media/process-web-scraping',
      '/api/v1/media/process-web-scraping/',
      '/process-web-scraping',
      '/process-web-scraping/'
    ])
    const mediaAddEntries = extractEntriesFromSchema(mediaAddSchema)
    const webScrapeEntries = extractEntriesFromSchema(webScrapeSchema)
    const merged = mergeSchemaEntries(mediaAddEntries, webScrapeEntries)
    setAdvSchema(merged)
    return merged
  }, [])

  // ---- loadSpec ----
  const loadSpec = React.useCallback(
    async (
      preferServer = true,
      options: { reportDiff?: boolean; persist?: boolean; forceFetch?: boolean } = {}
    ) => {
      const { reportDiff = false, persist = false, forceFetch = false } = options
      let used: 'server' | 'fallback' | 'none' = 'none'
      let remote: any | null = null
      const prevSchema = reportDiff ? [...(advSchemaRef.current || [])] : null

      const cached = !forceFetch ? readSpecCache(preferServer) : null
      if (cached) {
        setAdvSchema(cached.entries)
        setSpecSource(cached.source)
        return cached.source
      }

      if (!preferServer) {
        setAdvSchema(QUICK_INGEST_SCHEMA_FALLBACK)
        used = 'fallback'
        writeSpecCache({
          entries: QUICK_INGEST_SCHEMA_FALLBACK,
          source: 'fallback',
          version: fallbackSchemaVersion
        })
        setSpecSource(used)
        return used
      }

      try {
        remote = await tldwClient.getOpenAPISpec()
      } catch (e) {
        console.debug(
          "[QuickIngest] Failed to load OpenAPI spec from server; using bundled fallback.",
          (e as any)?.message || e
        )
      }

      if (remote) {
        const nextSchema = parseSpec(remote)
        used = 'server'
        writeSpecCache({
          entries: nextSchema,
          source: 'server',
          version: remote?.info?.version
        })

        try {
          const rVer = remote?.info?.version
          const prevVersion = specPrefs?.lastRemote?.version
          const prevCachedAt = specPrefs?.lastRemote?.cachedAt
          const now = Date.now()
          const shouldReuseCachedAt =
            prevVersion && prevVersion === rVer && typeof prevCachedAt === 'number'

          if (persist) {
            const payload = {
              ...(specPrefs || {}),
              preferServer: true,
              lastRemote: {
                version: rVer,
                cachedAt: shouldReuseCachedAt ? prevCachedAt : now
              }
            }
            try {
              const approxSize = JSON.stringify(payload).length
              console.info(
                "[QuickIngest] Persisting quickIngestSpecPrefs (~%d bytes)",
                approxSize
              )
            } catch (e) {
              console.debug(
                "[QuickIngest] Failed to estimate quickIngestSpecPrefs size:",
                (e as any)?.message || e
              )
            }
            persistSpecPrefs(payload)
          }
        } catch (e) {
          console.debug(
            "[QuickIngest] Failed to persist OpenAPI spec metadata:",
            (e as any)?.message || e
          )
        }

        if (reportDiff) {
          let added = 0
          let removed = 0
          try {
            const beforeNames = new Set((prevSchema || []).map((f) => f.name))
            const afterNames = new Set((nextSchema || []).map((f) => f.name))
            for (const name of afterNames) {
              if (!beforeNames.has(name)) added += 1
            }
            for (const name of beforeNames) {
              if (!afterNames.has(name)) removed += 1
            }
          } catch (e) {
            console.debug(
              "[QuickIngest] Failed to compute OpenAPI field diff:",
              (e as any)?.message || e
            )
          }
          messageApi.success(
            added || removed
              ? qi(
                  'specReloadedToastDiff',
                  'Advanced spec reloaded from server (fields added: {{added}}, removed: {{removed}})',
                  { added, removed }
                )
              : qi('specReloadedToast', 'Advanced spec reloaded from server')
          )
        }
      } else {
        setAdvSchema(QUICK_INGEST_SCHEMA_FALLBACK)
        used = 'fallback'
        writeSpecCache({
          entries: QUICK_INGEST_SCHEMA_FALLBACK,
          source: 'fallback',
          version: fallbackSchemaVersion
        })
        if (reportDiff) {
          messageApi.warning(
            qi(
              'specFallbackWarning',
              'Using bundled media.add schema fallback (v{{version}}); please verify against your tldw_server /openapi.json if fields look outdated.',
              { version: fallbackSchemaVersion }
            )
          )
        }
      }

      setSpecSource(used)
      return used
    },
    [persistSpecPrefs, specPrefs, messageApi, fallbackSchemaVersion, parseSpec, qi]
  )

  // ---- side effects ----

  // Sync review + storeRemote
  React.useEffect(() => {
    if (reviewBeforeStorage && !storeRemote) {
      setStoreRemote(true)
    }
  }, [reviewBeforeStorage, storeRemote])

  // Backfill Auto chunking defaults for older persisted Quick Ingest options.
  React.useEffect(() => {
    const normalized = normalizeCommonOptions(common)
    if (JSON.stringify(normalized) !== JSON.stringify(common || {})) {
      setCommon(normalized)
    }
  }, [common, setCommon])

  // Sync refs
  React.useEffect(() => {
    specPrefsCacheRef.current = JSON.stringify(specPrefs || {})
  }, [specPrefs])

  React.useEffect(() => {
    advSchemaRef.current = advSchema
  }, [advSchema])

  React.useEffect(() => {
    lastSavedAdvValuesRef.current = JSON.stringify(savedAdvValues || {})
  }, [savedAdvValues])

  React.useEffect(() => {
    lastSavedUiPrefsRef.current = JSON.stringify(uiPrefs || {})
  }, [uiPrefs])

  // Load spec on open
  React.useEffect(() => {
    if (!open) return
    void (async () => {
      await loadSpec(preferServerSpec)
    })()
  }, [loadSpec, open, preferServerSpec])

  // Load transcription models
  React.useEffect(() => {
    if (!open || ingestConnectionStatus !== "online") {
      setTranscriptionModelsLoading(false)
      if (ingestConnectionStatus !== "online") {
        setTranscriptionModelOptions([])
      }
      return
    }
    let cancelled = false
    const fetchModels = async () => {
      setTranscriptionModelsLoading(true)
      try {
        const res = await tldwClient.getTranscriptionModels()
        const all = Array.isArray(res?.all_models) ? res.all_models : []
        const seen = new Set<string>()
        const unique: string[] = []
        for (const model of all) {
          const value = String(model)
          if (!value || seen.has(value)) continue
          seen.add(value)
          unique.push(value)
        }
        if (!cancelled) setTranscriptionModelOptions(unique)
      } catch (e) {
        if ((import.meta as any)?.env?.DEV) {
          console.warn("Failed to load transcription models for Quick Ingest", e)
        }
      } finally {
        if (!cancelled) setTranscriptionModelsLoading(false)
      }
    }
    fetchModels()
    return () => {
      cancelled = true
      setTranscriptionModelsLoading(false)
    }
  }, [ingestConnectionStatus, open])

  // Hydrate advanced values from storage (once)
  React.useEffect(() => {
    if (advancedHydratedRef.current) return
    advancedHydratedRef.current = true
    if (savedAdvValues && typeof savedAdvValues === 'object') {
      setAdvancedValues((prev) => ({ ...prev, ...savedAdvValues }))
    }
  }, [savedAdvValues])

  // Hydrate UI prefs (once)
  React.useEffect(() => {
    if (uiPrefsHydratedRef.current) return
    uiPrefsHydratedRef.current = true
    if (uiPrefs?.advancedOpen !== undefined) setAdvancedOpen(Boolean(uiPrefs.advancedOpen))
    if (uiPrefs?.fieldDetailsOpen && typeof uiPrefs.fieldDetailsOpen === 'object') setFieldDetailsOpen(uiPrefs.fieldDetailsOpen)
  }, [uiPrefs])

  // Persist UI prefs (debounced)
  React.useEffect(() => {
    const id = setTimeout(() => {
      const nextPrefs = { advancedOpen, fieldDetailsOpen }
      const serialized = JSON.stringify(nextPrefs)
      if (lastSavedUiPrefsRef.current === serialized) return
      lastSavedUiPrefsRef.current = serialized
      try { setUiPrefs(nextPrefs) } catch {}
    }, SAVE_DEBOUNCE_MS)
    return () => clearTimeout(id)
  }, [advancedOpen, fieldDetailsOpen, setUiPrefs])

  // Resolve RAG embedding model
  React.useEffect(() => {
    ;(async () => {
      try {
        const id = await defaultEmbeddingModelForRag()
        if (!id) {
          setRagEmbeddingLabel(null)
          return
        }
        const parts = String(id).split('/')
        const provider = parts.length > 1 ? parts[0] : 'unknown'
        const modelName = parts.length > 1 ? parts.slice(1).join('/') : id
        const models = await tldwModels.getEmbeddingModels().catch(() => [])
        const match = models.find((m) => m.id === id || m.id === modelName)
        const providerLabel = tldwModels.getProviderDisplayName(
          match?.provider || provider
        )
        const label = `${providerLabel} / ${modelName}`
        setRagEmbeddingLabel(label)
      } catch {
        setRagEmbeddingLabel(null)
      }
    })()
  }, [])

  return {
    // persisted state
    storeRemote, setStoreRemote,
    reviewBeforeStorage, setReviewBeforeStorage,
    typeDefaults, setTypeDefaults,
    common, setCommon,
    savedAdvValues, setSavedAdvValues,
    uiPrefs, setUiPrefs,
    specPrefs, setSpecPrefs,
    storageHintSeen, setStorageHintSeen,
    reviewDraftWarning, setReviewDraftWarning,
    // session state
    chunkingTemplateName, setChunkingTemplateName,
    autoApplyTemplate, setAutoApplyTemplate,
    advancedOpen, setAdvancedOpen,
    advancedValues, setAdvancedValues,
    advSchema, setAdvSchema,
    specSource,
    fieldDetailsOpen, setFieldDetailsOpen,
    advSearch, setAdvSearch,
    transcriptionModelOptions,
    transcriptionModelsLoading,
    ragEmbeddingLabel,
    // derived
    normalizedTypeDefaults,
    processOnly,
    shouldStoreRemote,
    modifiedAdvancedCount,
    advancedDefaultsDirty,
    hasTypeDefaultChanges,
    lastRefreshedLabel,
    preferServerSpec,
    specSourceLabel,
    resolvedAdvSchema,
    transcriptionModelChoices,
    transcriptionModelValue,
    storageLabel,
    // callbacks
    setAdvancedValue,
    handleTranscriptionModelChange,
    handleReviewToggle,
    persistSpecPrefs,
    parseSpec,
    loadSpec,
    confirmDanger,
  }
}
