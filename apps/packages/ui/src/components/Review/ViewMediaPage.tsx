import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CheckSquare,
  Square,
  Trash2,
} from 'lucide-react'
import { useServerOnline } from '@/hooks/useServerOnline'
import { useServerCapabilities } from '@/hooks/useServerCapabilities'
import {
  useConnectionActions,
  useConnectionUxState
} from '@/hooks/useConnectionState'
import { useDemoMode } from '@/context/demo-mode'
import { useMessageOption } from '@/hooks/useMessageOption'
import { useAntdMessage } from '@/hooks/useAntdMessage'
import FeatureEmptyState from '@/components/Common/FeatureEmptyState'
import { SearchBar } from '@/components/Media/SearchBar'
import { FilterPanel } from '@/components/Media/FilterPanel'
import { ResultsList } from '@/components/Media/ResultsList'
import { ContentViewer } from '@/components/Media/ContentViewer'
import { Pagination } from '@/components/Media/Pagination'
import { FilterChips } from '@/components/Media/FilterChips'
import type { MediaResultItem } from '@/components/Media/types'
import {
  useMediaNavigation
} from '@/hooks/useMediaNavigation'
import { bgRequest } from '@/services/background-proxy'
import { requestQuickIngestOpen } from '@/utils/quick-ingest-open'
import { setSetting } from '@/services/settings/registry'
import {
  DISCUSS_MEDIA_PROMPT_SETTING,
  LAST_MEDIA_ID_SETTING,
} from '@/services/settings/ui-settings'
import {
  hasDefaultMediaSearchFields,
} from '@/components/Review/mediaSearchRequest'
import {
  isMediaOnly,
  isNotesOnly,
} from '@/components/Review/mediaKinds'
import {
  parseMediaFilterParams,
  buildMediaFilterSearch,
  hasMediaFilterParams
} from '@/components/Review/mediaFilterParams'
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"
import { buildStudyPackRoute } from "@/services/tldw/study-pack-handoff"
import {
  getMediaNavigationResumeEntry,
  resolveMediaNavigationResumeSelection,
  saveMediaNavigationResumeSelection
} from '@/utils/media-navigation-resume'
import {
  trackMediaNavigationTelemetry,
  type MediaNavigationFallbackKind,
} from '@/utils/media-navigation-telemetry'
import { normalizeRequestedMediaRenderMode } from '@/utils/media-render-mode'

import { useMediaSearch } from './hooks/useMediaSearch'
import { useMediaNavigationState } from './hooks/useMediaNavigationState'
import { useMediaViewPreferences } from './hooks/useMediaViewPreferences'
import { useMediaSelection } from './hooks/useMediaSelection'
import { useMediaKeyboardShortcuts } from './hooks/useMediaKeyboardShortcuts'

export const MEDIA_STALE_CHECK_INTERVAL_MS = 30_000

const LazyJumpToNavigator = React.lazy(() =>
  import('@/components/Media/JumpToNavigator').then((module) => ({
    default: module.JumpToNavigator
  }))
)

const LazyKeyboardShortcutsOverlay = React.lazy(() =>
  import('@/components/Media/KeyboardShortcutsOverlay').then((module) => ({
    default: module.KeyboardShortcutsOverlay
  }))
)

const LazyMediaIngestJobsPanel = React.lazy(() =>
  import('@/components/Media/MediaIngestJobsPanel').then((module) => ({
    default: module.MediaIngestJobsPanel
  }))
)

const LazyMediaLibraryStatsPanel = React.lazy(() =>
  import('@/components/Media/MediaLibraryStatsPanel').then((module) => ({
    default: module.MediaLibraryStatsPanel
  }))
)

const LazyMediaBulkToolbar = React.lazy(() =>
  import('./MediaBulkToolbar').then((module) => ({
    default: module.MediaBulkToolbar
  }))
)

const LazyMediaSectionNavigator = React.lazy(() =>
  import('@/components/Media/MediaSectionNavigator').then((module) => ({
    default: module.MediaSectionNavigator
  }))
)

const ViewMediaPage: React.FC = () => {
  const { t } = useTranslation(['review', 'common', 'settings'])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const { demoEnabled } = useDemoMode()
  const { uxState } = useConnectionUxState()
  const { checkOnce } = useConnectionActions()

  // Check media support
  const mediaUnsupported = !capsLoading && capabilities && !capabilities.hasMedia

  if (!isOnline && uxState !== 'testing') {
    if (uxState === 'error_auth' || uxState === 'configuring_auth') {
      return (
        <div className="flex h-full items-center justify-center">
          <FeatureEmptyState
            title={t('review:mediaEmpty.authTitle', {
              defaultValue: 'Add your credentials to use Media'
            })}
            description={t('review:mediaEmpty.authDescription', {
              defaultValue:
                'Your server is reachable, but Media needs valid credentials before it can load.'
            })}
            examples={[]}
            primaryActionLabel={t('settings:tldw.openSettings', {
              defaultValue: 'Open Settings'
            })}
            onPrimaryAction={() => navigate('/settings/tldw')}
          />
        </div>
      )
    }

    if (uxState === 'unconfigured' || uxState === 'configuring_url') {
      return (
        <div className="flex h-full items-center justify-center">
          <FeatureEmptyState
            title={t('review:mediaEmpty.setupTitle', {
              defaultValue: 'Finish setup to use Media'
            })}
            description={t('review:mediaEmpty.setupDescription', {
              defaultValue:
                'Finish the tldw server setup flow, then return here to browse your media.'
            })}
            examples={[]}
            primaryActionLabel={t('settings:tldw.finishSetup', {
              defaultValue: 'Finish Setup'
            })}
            onPrimaryAction={() => navigate('/')}
          />
        </div>
      )
    }

    if (uxState === 'error_unreachable') {
      return (
        <div className="flex h-full items-center justify-center">
          <FeatureEmptyState
            title={t('review:mediaEmpty.unreachableTitle', {
              defaultValue: "Can't reach your tldw server right now"
            })}
            description={t('review:mediaEmpty.unreachableDescription', {
              defaultValue:
                'Your server settings are saved, but Media cannot reach the tldw server right now.'
            })}
            examples={[]}
            primaryActionLabel={t('option:buttonRetry', {
              defaultValue: 'Retry connection'
            })}
            onPrimaryAction={() => {
              void checkOnce()
            }}
            secondaryActionLabel={t('settings:healthSummary.diagnostics', {
              defaultValue: 'Health & diagnostics'
            })}
            onSecondaryAction={() => navigate('/settings/health')}
          />
        </div>
      )
    }

    if (demoEnabled) {
      return (
        <div className="flex h-full items-center justify-center">
          <FeatureEmptyState
            title={t('review:mediaEmpty.offlineTitle', {
              defaultValue: 'Media API not available offline'
            })}
            description={t('review:mediaEmpty.offlineDescription', {
              defaultValue:
                'This feature requires connection to the tldw server. Please check your server connection.'
            })}
            examples={[]}
            primaryActionLabel={t('option:buttonRetry', {
              defaultValue: 'Retry connection'
            })}
            onPrimaryAction={() => {
              void checkOnce()
            }}
            secondaryActionLabel={t('settings:tldw.openSettings', {
              defaultValue: 'Open Settings'
            })}
            onSecondaryAction={() => navigate('/settings/tldw')}
          />
        </div>
      )
    }

    return (
      <div className="flex h-full items-center justify-center">
        <FeatureEmptyState
          title={t('review:mediaEmpty.offlineTitle', {
            defaultValue: 'Server offline'
          })}
          description={t('review:mediaEmpty.offlineDescription', {
            defaultValue: 'Please check your server connection.'
          })}
          examples={[]}
          primaryActionLabel={t('option:buttonRetry', {
            defaultValue: 'Retry connection'
          })}
          onPrimaryAction={() => {
            void checkOnce()
          }}
          secondaryActionLabel={t('settings:tldw.openSettings', {
            defaultValue: 'Open Settings'
          })}
          onSecondaryAction={() => navigate('/settings/tldw')}
        />
      </div>
    )
  }

  if (isOnline && mediaUnsupported) {
    return (
      <FeatureEmptyState
        title={
          <span className="inline-flex items-center gap-2">
            <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] font-medium text-warn">
              {t('review:mediaEmpty.featureUnavailableBadge', {
                defaultValue: 'Feature unavailable'
              })}
            </span>
            <span>
              {t('review:mediaEmpty.offlineTitle', {
                defaultValue: 'Media API not available on this server'
              })}
            </span>
          </span>
        }
        description={t('review:mediaEmpty.offlineDescription', {
          defaultValue:
            'This workspace depends on Media Review support in your tldw server. You can continue using chat, notes, and other tools while you upgrade to a version that includes Media.'
        })}
        examples={[
          t('review:mediaEmpty.offlineExample1', {
            defaultValue:
              'Open Diagnostics to confirm your server version and available APIs.'
          }),
          t('review:mediaEmpty.offlineExample2', {
            defaultValue: 'After upgrading, reload the extension and return to Media.'
          }),
          t('review:mediaEmpty.offlineTechnicalDetails', {
            defaultValue:
              'Technical details: this tldw server does not advertise the Media endpoints (for example, /api/v1/media and /api/v1/media/search).'
          })
        ]}
        primaryActionLabel={t('option:buttonRetry', {
          defaultValue: 'Retry connection'
        })}
        onPrimaryAction={() => {
          void checkOnce()
        }}
        secondaryActionLabel={t('settings:healthSummary.diagnostics', {
          defaultValue: 'Open Diagnostics'
        })}
        onSecondaryAction={() => navigate('/settings/health')}
      />
    )
  }

  return <MediaPageContent />
}

const MediaPageContent: React.FC = () => {
  const { t } = useTranslation(['review', 'common'])
  const navigate = useNavigate()
  const location = useLocation()
  const message = useAntdMessage()
  const {
    setChatMode,
    setSelectedKnowledge,
    setRagMediaIds
  } = useMessageOption()

  // --- Hooks ---
  const search = useMediaSearch({ t, message })
  const { refetch: searchRefetch } = search

  const viewPrefs = useMediaViewPreferences()

  // Auto-refresh media results when Quick Ingest completes
  const searchRefetchRef = useRef(searchRefetch)
  const ingestRefreshTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => { searchRefetchRef.current = searchRefetch }, [searchRefetch])
  useEffect(() => {
    const handleIngestComplete = () => {
      if (ingestRefreshTimeoutRef.current !== null) {
        clearTimeout(ingestRefreshTimeoutRef.current)
      }
      ingestRefreshTimeoutRef.current = setTimeout(() => { searchRefetchRef.current() }, 1500)
    }
    window.addEventListener("tldw:quick-ingest-complete", handleIngestComplete)
    return () => {
      if (ingestRefreshTimeoutRef.current !== null) {
        clearTimeout(ingestRefreshTimeoutRef.current)
      }
      window.removeEventListener("tldw:quick-ingest-complete", handleIngestComplete)
    }
  }, [])

  // Compute display results (filtered by favorites/collections)
  // Need selection hook first for favorites/collections, but selection needs displayResults.
  // We break the cycle by computing displayResults here using search results + selection state.
  const nav = useMediaNavigationState({
    t, message,
    displayResults: search.results, // Will be refined below
    refetch: search.refetch
  })

  const selection = useMediaSelection({
    t, message,
    displayResults: search.results,
    selected: nav.selected,
    setSelected: nav.setSelected,
    setSelectedContent: nav.setSelectedContent,
    setSelectedDetail: nav.setSelectedDetail,
    setLastFetchedId: nav.setLastFetchedId,
    refetch: search.refetch
  })

  // Compute display results with favorites and collection filtering
  const displayResults = useMemo(() => {
    let nextResults = search.results
    if (selection.showFavoritesOnly) {
      nextResults = nextResults.filter((item) => selection.favoritesSet.has(String(item.id)))
    }
    if (selection.activeCollectionId) {
      const collection = selection.mediaCollections.find(
        (entry) => entry.id === selection.activeCollectionId
      )
      if (!collection) {
        return []
      }
      const allowedIdSet = new Set(collection.itemIds.map((id) => String(id)))
      nextResults = nextResults.filter((item) => allowedIdSet.has(String(item.id)))
    }
    return nextResults
  }, [
    selection.activeCollectionId,
    selection.favoritesSet,
    selection.mediaCollections,
    search.results,
    selection.showFavoritesOnly
  ])

  // Extended active filters (includes favorites/collection from selection)
  const hasActiveFilters = search.hasActiveFilters ||
    selection.showFavoritesOnly ||
    Boolean(selection.activeCollectionId)

  const activeFilterCount = useMemo(() => {
    return search.activeFilterCount +
      Number(selection.showFavoritesOnly) +
      Number(Boolean(selection.activeCollectionId))
  }, [search.activeFilterCount, selection.showFavoritesOnly, selection.activeCollectionId])

  const resetAllFilters = useCallback(() => {
    search.resetAllFilters()
    selection.setShowFavoritesOnly(false)
    selection.setActiveCollectionId(null)
  }, [search, selection])

  const handleSelectAllVisibleItems = useCallback(() => {
    selection.setBulkSelectedIds(displayResults.map((item) => String(item.id)))
  }, [displayResults, selection])

  const hasJumpTo = displayResults.length > 5

  // Sidebar dimensions
  const sidebarDimensions = viewPrefs.computeSidebarDimensions(
    search.pageSize,
    nav.selectedContent
  )

  // Schedule content measurement on selection/content change
  useEffect(() => {
    viewPrefs.scheduleContentMeasure()
  }, [viewPrefs.scheduleContentMeasure, nav.selected?.id, nav.selectedContent.length])

  // Navigation panel logic
  const selectedMediaIdForNavigation =
    nav.selected?.kind === 'media' && nav.selected?.id != null ? nav.selected.id : null
  const navigationControlsEnabled =
    viewPrefs.mediaNavigationPanelEnabled && selectedMediaIdForNavigation != null
  const navigationEnabled = navigationControlsEnabled

  const {
    data: navigationData,
    isLoading: isNavigationLoading,
    error: navigationError,
    refetch: refetchNavigation
  } = useMediaNavigation(selectedMediaIdForNavigation, {
    enabled: navigationEnabled,
    includeGeneratedFallback: viewPrefs.includeGeneratedFallbackValue
  })
  const navigationNodes = navigationData?.nodes || []

  const [selectedNavigationNodeId, setSelectedNavigationNodeId] =
    useState<string | null>(null)
  const [navigationSelectionNonce, setNavigationSelectionNonce] = useState(0)
  const pendingSectionSelectionTelemetryRef = useRef<{
    nodeId: string
    startedAt: number
    source: 'user' | 'restore'
  } | null>(null)
  const lastTruncatedTelemetryKeyRef = useRef<string>('')
  const lastFallbackTelemetryKeyRef = useRef<string>('')

  const selectedNavigationNode = useMemo(
    () =>
      selectedNavigationNodeId
        ? navigationNodes.find((node) => node.id === selectedNavigationNodeId) || null
        : null,
    [navigationNodes, selectedNavigationNodeId]
  )
  const showNavigationPanel =
    navigationEnabled &&
    viewPrefs.navigationPanelVisibleValue &&
    (isNavigationLoading || Boolean(navigationError) || navigationNodes.length > 0)

  const effectiveContentFormat: null = null
  const selectedNavigationTarget = useMemo(() => {
    if (!showNavigationPanel || !selectedNavigationNodeId) return null
    if (!selectedNavigationNode) return null
    return {
      target_type: selectedNavigationNode.target_type,
      target_start: selectedNavigationNode.target_start,
      target_end: selectedNavigationNode.target_end,
      target_href: selectedNavigationNode.target_href
    }
  }, [selectedNavigationNode, selectedNavigationNodeId, showNavigationPanel])

  const navigationPageCountHint = useMemo(() => {
    const toPositiveInt = (value: unknown): number | null => {
      if (typeof value === 'number' && Number.isFinite(value)) {
        const parsed = Math.trunc(value)
        return parsed > 0 ? parsed : null
      }
      if (typeof value === 'string' && value.trim().length > 0) {
        const parsed = Number.parseInt(value, 10)
        return Number.isFinite(parsed) && parsed > 0 ? parsed : null
      }
      return null
    }

    const explicitCandidates = [
      nav.selectedDetail?.page_count,
      nav.selectedDetail?.pageCount,
      nav.selectedDetail?.num_pages,
      nav.selectedDetail?.numPages,
      nav.selectedDetail?.total_pages,
      nav.selectedDetail?.totalPages,
      nav.selectedDetail?.metadata?.page_count,
      nav.selectedDetail?.metadata?.num_pages,
      nav.selectedDetail?.metadata?.total_pages,
      nav.selected?.raw?.page_count,
      nav.selected?.raw?.num_pages,
      nav.selected?.raw?.total_pages
    ]
    for (const candidate of explicitCandidates) {
      const parsed = toPositiveInt(candidate)
      if (parsed != null) return parsed
    }

    let maxPage = 0
    for (const node of navigationNodes) {
      if (node.target_type !== 'page') continue
      if (typeof node.target_start !== 'number' || !Number.isFinite(node.target_start)) {
        continue
      }
      const parsed = Math.trunc(node.target_start)
      if (parsed > maxPage) maxPage = parsed
    }
    return maxPage > 0 ? maxPage : null
  }, [navigationNodes, nav.selected?.raw, nav.selectedDetail])

  const navigationStatusLabel = useMemo(() => {
    if (!navigationEnabled) return ''
    if (isNavigationLoading) {
      return t('review:mediaNavigation.loading', {
        defaultValue: 'Loading sections...'
      })
    }
    if (navigationError) {
      return t('review:mediaNavigation.error', {
        defaultValue: 'Section navigation unavailable'
      })
    }
    if (navigationNodes.length === 0) {
      return t('review:mediaNavigation.noStructure', {
        defaultValue: 'No section structure'
      })
    }
    const generatedNodeCount = navigationNodes.filter(
      (node) => node.source === 'generated'
    ).length
    if (generatedNodeCount === navigationNodes.length) {
      return t('review:mediaNavigation.generatedCount', {
        defaultValue: 'Generated sections: {{count}}',
        count: navigationNodes.length
      })
    }
    return t('review:mediaNavigation.sectionCount', {
      defaultValue: 'Sections: {{count}}',
      count: navigationNodes.length
    })
  }, [isNavigationLoading, navigationEnabled, navigationError, navigationNodes, t])

  const handleNavigationPanelVisibilityChange = useCallback(
    (nextVisible: boolean) => {
      void viewPrefs.setNavigationPanelVisible(nextVisible)
      if (!nextVisible) {
        pendingSectionSelectionTelemetryRef.current = null
      }
      void trackMediaNavigationTelemetry({
        type: 'media_navigation_rollout_control_changed',
        scope_key_hash: viewPrefs.navigationScopeKeyHash,
        media_id: selectedMediaIdForNavigation,
        control: 'panel_visible',
        enabled: nextVisible
      })
    },
    [viewPrefs, selectedMediaIdForNavigation]
  )

  const handleGeneratedFallbackToggle = useCallback(
    (nextEnabled: boolean) => {
      void viewPrefs.setIncludeGeneratedFallback(nextEnabled)
      lastFallbackTelemetryKeyRef.current = ''
      setSelectedNavigationNodeId(null)
      pendingSectionSelectionTelemetryRef.current = null
      void trackMediaNavigationTelemetry({
        type: 'media_navigation_rollout_control_changed',
        scope_key_hash: viewPrefs.navigationScopeKeyHash,
        media_id: selectedMediaIdForNavigation,
        control: 'include_generated_fallback',
        enabled: nextEnabled
      })
    },
    [viewPrefs, selectedMediaIdForNavigation]
  )

  const persistNavigationSelection = useCallback(
    async (node: {
      id: string
      path_label: string | null
      title: string
      level: number
    }) => {
      if (selectedMediaIdForNavigation == null) return
      const evictionStats = await saveMediaNavigationResumeSelection({
        scopeKey: viewPrefs.navigationScopeKey,
        mediaId: selectedMediaIdForNavigation,
        node,
        navigationVersion: navigationData?.navigation_version
      }).catch(() => null)
      if (!evictionStats) return

      if (evictionStats.evicted_lru_count > 0) {
        void trackMediaNavigationTelemetry({
          type: 'media_navigation_resume_state_evicted',
          scope_key_hash: viewPrefs.navigationScopeKeyHash,
          evicted_entry_count: evictionStats.evicted_lru_count,
          reason: 'lru'
        })
      }
      if (evictionStats.evicted_stale_count > 0) {
        void trackMediaNavigationTelemetry({
          type: 'media_navigation_resume_state_evicted',
          scope_key_hash: viewPrefs.navigationScopeKeyHash,
          evicted_entry_count: evictionStats.evicted_stale_count,
          reason: 'stale'
        })
      }
    },
    [
      navigationData?.navigation_version,
      viewPrefs.navigationScopeKey,
      viewPrefs.navigationScopeKeyHash,
      selectedMediaIdForNavigation
    ]
  )

  // Navigation telemetry effects
  useEffect(() => {
    if (!navigationData?.stats?.truncated) return
    if (selectedMediaIdForNavigation == null) return
    const telemetryKey = [
      selectedMediaIdForNavigation,
      navigationData.navigation_version || 'unknown',
      navigationData.stats.returned_node_count,
      navigationData.stats.node_count
    ].join(':')
    if (lastTruncatedTelemetryKeyRef.current === telemetryKey) return
    lastTruncatedTelemetryKeyRef.current = telemetryKey

    void trackMediaNavigationTelemetry({
      type: 'media_navigation_payload_truncated',
      scope_key_hash: viewPrefs.navigationScopeKeyHash,
      media_id: selectedMediaIdForNavigation,
      requested_max_nodes: null,
      returned_node_count: navigationData.stats.returned_node_count,
      node_count: navigationData.stats.node_count
    })
  }, [navigationData, viewPrefs.navigationScopeKeyHash, selectedMediaIdForNavigation])

  useEffect(() => {
    if (!navigationEnabled || selectedMediaIdForNavigation == null) return
    if (isNavigationLoading || navigationError || !navigationData) return

    let fallbackKind: MediaNavigationFallbackKind | null = null
    let source: string | null = null

    if (!navigationData.available || navigationNodes.length === 0) {
      fallbackKind = 'no_structure'
    } else {
      const hasGeneratedNodes = navigationNodes.some(
        (node) => node.source === 'generated'
      )
      if (hasGeneratedNodes) {
        fallbackKind = 'generated'
        source = 'generated'
      }
    }

    if (!fallbackKind) return

    const telemetryKey = [
      selectedMediaIdForNavigation,
      navigationData.navigation_version || 'unknown',
      fallbackKind,
      source || 'none',
      navigationNodes.length
    ].join(':')
    if (lastFallbackTelemetryKeyRef.current === telemetryKey) return
    lastFallbackTelemetryKeyRef.current = telemetryKey

    void trackMediaNavigationTelemetry({
      type: 'media_navigation_fallback_used',
      scope_key_hash: viewPrefs.navigationScopeKeyHash,
      media_id: selectedMediaIdForNavigation,
      fallback_kind: fallbackKind,
      source
    })
  }, [
    isNavigationLoading,
    navigationData,
    navigationEnabled,
    navigationError,
    navigationNodes,
    viewPrefs.navigationScopeKeyHash,
    selectedMediaIdForNavigation
  ])

  useEffect(() => {
    const pendingSelection = pendingSectionSelectionTelemetryRef.current
    if (!pendingSelection) return
    if (!showNavigationPanel || !selectedNavigationNodeId) return
    if (selectedMediaIdForNavigation == null) return
    if (pendingSelection.nodeId !== selectedNavigationNodeId) return

    const selectedNode = navigationNodes.find(
      (node) => node.id === selectedNavigationNodeId
    )
    if (!selectedNode) return

    pendingSectionSelectionTelemetryRef.current = null
    const latencyMs = Math.max(0, Date.now() - pendingSelection.startedAt)

    void trackMediaNavigationTelemetry({
      type: 'media_navigation_section_selected',
      media_id: selectedMediaIdForNavigation,
      node_id: selectedNode.id,
      depth: selectedNode.level || 0,
      latency_ms: latencyMs,
      source: pendingSelection.source
    })
  }, [
    navigationNodes,
    selectedMediaIdForNavigation,
    selectedNavigationNodeId,
    showNavigationPanel
  ])

  // Reset navigation selection on item change
  useEffect(() => {
    setSelectedNavigationNodeId(null)
    setNavigationSelectionNonce(0)
    pendingSectionSelectionTelemetryRef.current = null
  }, [nav.selected?.id, nav.selected?.kind])

  // Restore/auto-select navigation node
  useEffect(() => {
    if (!showNavigationPanel) return
    if (navigationNodes.length === 0) {
      setSelectedNavigationNodeId(null)
      pendingSectionSelectionTelemetryRef.current = null
      return
    }
    const hasCurrentSelection =
      !!selectedNavigationNodeId &&
      navigationNodes.some((node) => node.id === selectedNavigationNodeId)
    if (hasCurrentSelection || selectedMediaIdForNavigation == null) return

    let cancelled = false
    ;(async () => {
      const resumeEntry = await getMediaNavigationResumeEntry({
        scopeKey: viewPrefs.navigationScopeKey,
        mediaId: selectedMediaIdForNavigation
      }).catch(() => null)
      if (cancelled) return

      const resolvedSelection = resolveMediaNavigationResumeSelection({
        nodes: navigationNodes,
        navigationVersion: navigationData?.navigation_version,
        resumeEntry
      })

      const nextNodeId = resolvedSelection?.nodeId || navigationNodes[0]?.id || null
      if (!nextNodeId) {
        setSelectedNavigationNodeId(null)
        return
      }

      const nextNode =
        navigationNodes.find((node) => node.id === nextNodeId) || navigationNodes[0]
      if (!nextNode) return

      setSelectedNavigationNodeId(nextNode.id)
      setNavigationSelectionNonce((prev) => prev + 1)
      void persistNavigationSelection(nextNode)

      if (resumeEntry && resolvedSelection) {
        pendingSectionSelectionTelemetryRef.current = {
          nodeId: nextNode.id,
          startedAt: Date.now(),
          source: 'restore'
        }
        void trackMediaNavigationTelemetry({
          type: 'media_navigation_resume_state_restored',
          scope_key_hash: viewPrefs.navigationScopeKeyHash,
          media_id: selectedMediaIdForNavigation,
          outcome: resolvedSelection.outcome
        })
      }
    })()

    return () => {
      cancelled = true
    }
  }, [
    navigationData?.navigation_version,
    navigationNodes,
    viewPrefs.navigationScopeKey,
    viewPrefs.navigationScopeKeyHash,
    persistNavigationSelection,
    selectedMediaIdForNavigation,
    selectedNavigationNodeId,
    showNavigationPanel
  ])

  // URL filter initialization
  const hasInitializedFromUrl = useRef(false)
  const suppressUrlSync = useRef(false)

  useEffect(() => {
    if (hasInitializedFromUrl.current) return
    hasInitializedFromUrl.current = true
    if (!hasMediaFilterParams(location.search)) return
    const urlFilters = parseMediaFilterParams(location.search)
    suppressUrlSync.current = true
    if (urlFilters.q) search.setQuery(urlFilters.q)
    if (urlFilters.types?.length) search.setMediaTypes(urlFilters.types)
    if (urlFilters.keywords?.length) search.setKeywordTokens(urlFilters.keywords)
    if (urlFilters.excludeKeywords?.length) search.setExcludeKeywordTokens(urlFilters.excludeKeywords)
    if (urlFilters.sort) search.setSortBy(urlFilters.sort)
    if (urlFilters.dateStart || urlFilters.dateEnd) {
      search.setDateRange({ startDate: urlFilters.dateStart || null, endDate: urlFilters.dateEnd || null })
    }
    if (urlFilters.searchMode) search.setSearchMode(urlFilters.searchMode)
    if (urlFilters.exactPhrase) search.setExactPhrase(urlFilters.exactPhrase)
    if (urlFilters.fields?.length) search.setSearchFields(urlFilters.fields)
    if (urlFilters.page) search.setPage(urlFilters.page)
    if (urlFilters.pageSize) search.setPageSize(urlFilters.pageSize)
    requestAnimationFrame(() => { suppressUrlSync.current = false })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Sync filter state to URL
  useEffect(() => {
    if (suppressUrlSync.current) return
    if (!search.hasRunInitialSearch.current) return
    const nextSearch = buildMediaFilterSearch(location.search, {
      q: search.query || undefined,
      types: search.mediaTypes.length > 0 ? search.mediaTypes : undefined,
      keywords: search.keywordTokens.length > 0 ? search.keywordTokens : undefined,
      excludeKeywords: search.excludeKeywordTokens.length > 0 ? search.excludeKeywordTokens : undefined,
      sort: search.sortBy,
      dateStart: search.dateRange.startDate,
      dateEnd: search.dateRange.endDate,
      searchMode: search.searchMode,
      exactPhrase: search.exactPhrase || undefined,
      fields:
        search.searchFields.length > 0 && !hasDefaultMediaSearchFields(search.searchFields)
          ? search.searchFields
          : undefined,
      page: search.page,
      pageSize: search.pageSize
    })
    if (nextSearch === location.search) return
    navigate(
      { pathname: location.pathname, search: nextSearch, hash: location.hash },
      { replace: true }
    )
  }, [
    search.dateRange.endDate,
    search.dateRange.startDate,
    search.exactPhrase,
    search.excludeKeywordTokens,
    search.keywordTokens,
    location.hash,
    location.pathname,
    location.search,
    search.mediaTypes,
    navigate,
    search.page,
    search.pageSize,
    search.query,
    search.searchFields,
    search.searchMode,
    search.sortBy
  ])

  // Keyboard shortcuts
  const keyboard = useMediaKeyboardShortcuts({
    hasNext: nav.hasNext,
    hasPrevious: nav.hasPrevious,
    page: search.page,
    totalPages: search.totalPages,
    displayResults,
    selectedIndex: nav.selectedIndex,
    searchCollapsed: search.searchCollapsed,
    setSearchCollapsed: search.setSearchCollapsed,
    searchInputRef: search.searchInputRef,
    setSelected: nav.setSelected,
    setPage: search.setPage
  })

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      search.handleSearch()
    }
  }

  // Chat/action handlers
  const handleChatWithMedia = useCallback(() => {
    if (!nav.selected) return

    const title = nav.selected.title || String(nav.selected.id)
    const content = nav.selectedContent || ''

    try {
      const payload = {
        mediaId: String(nav.selected.id),
        title,
        content,
        mode: 'normal' as const
      }
      void setSetting(DISCUSS_MEDIA_PROMPT_SETTING, payload)
      try {
        window.dispatchEvent(
          new CustomEvent('tldw:discuss-media', {
            detail: payload
          })
        )
      } catch {
        // ignore event errors
      }
    } catch {
      // ignore storage errors
    }
    setChatMode('normal')
    setSelectedKnowledge(null as any)
    setRagMediaIds(null)
    navigate('/')
    message.success(
      t(
        'review:reviewPage.chatPrepared',
        'Prepared chat with this media in the composer.'
      )
    )
  }, [
    nav.selectedContent,
    message,
    navigate,
    nav.selected,
    setChatMode,
    setRagMediaIds,
    setSelectedKnowledge,
    t
  ])

  const handleChatAboutMedia = useCallback(() => {
    if (!nav.selected) return

    const idNum = Number(nav.selected.id)
    if (!Number.isFinite(idNum)) {
      message.warning(
        t(
          'review:reviewPage.chatAboutMediaInvalidId',
          'This media item does not have a numeric id yet.'
        )
      )
      return
    }
    setSelectedKnowledge(null as any)
    setRagMediaIds([idNum])
    setChatMode('rag')
    try {
      const payload = {
        mediaId: String(nav.selected.id),
        mode: 'rag_media' as const
      }
      void setSetting(DISCUSS_MEDIA_PROMPT_SETTING, payload)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('tldw:discuss-media', { detail: payload }))
      }
    } catch {
      // ignore storage/event errors
    }
    navigate('/')
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('tldw:focus-composer'))
      }
    } catch {
      // ignore
    }
    message.success(
      t(
        'review:reviewPage.chatAboutMediaRagSent',
        'Opened media-scoped RAG chat.'
      )
    )
  }, [nav.selected, setSelectedKnowledge, setRagMediaIds, setChatMode, navigate, message, t])

  const handleGenerateFlashcardsFromMedia = useCallback(
    (payload: {
      text: string
      sourceId?: string
      sourceTitle?: string
    }) => {
      const sourceText = String(payload.text || "").trim()
      if (!sourceText) {
        message.warning(
          t("review:mediaPage.generateFlashcardsEmpty", {
            defaultValue: "No content available to generate flashcards."
          })
        )
        return
      }

      navigate(
        buildFlashcardsGenerateRoute({
          text: sourceText,
          sourceType: "media",
          sourceId:
            payload.sourceId ||
            (nav.selected?.id != null ? String(nav.selected.id) : undefined),
          sourceTitle: payload.sourceTitle || nav.selected?.title || undefined
        })
      )
    },
    [message, navigate, nav.selected, t]
  )

  const handleCreateStudyPackFromMedia = useCallback(() => {
    const selectedMedia = nav.selected?.kind === "media" ? nav.selected : null
    const mediaTitle = selectedMedia?.title?.trim() || ""
    const mediaId = selectedMedia?.id

    if (!mediaTitle || mediaId == null) {
      message.warning(
        t("review:mediaPage.studyPackMissingSelection", {
          defaultValue: "Select media before creating a study pack."
        })
      )
      return
    }

    navigate(
      buildStudyPackRoute({
        title: mediaTitle,
        sourceItems: [
          {
            sourceType: "media",
            sourceId: String(mediaId),
            sourceTitle: mediaTitle
          }
        ]
      })
    )
  }, [message, nav.selected, navigate, t])

  const handleCreateNoteWithContent = useCallback(async (noteContent: string, title: string) => {
    try {
      await bgRequest({
        path: '/api/v1/notes/' as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          title: title,
          content: noteContent,
          keywords: nav.selected?.keywords || []
        }
      })
      message.success('Note created successfully')
      navigate('/notes')
    } catch (err) {
      console.error('Failed to create note:', err)
      message.error('Failed to create note')
    }
  }, [nav.selected, message, navigate])

  const handleOpenInMultiReview = useCallback(() => {
    if (!nav.selected) return
    void setSetting(LAST_MEDIA_ID_SETTING, String(nav.selected.id))
    navigate('/media-multi')
  }, [nav.selected, navigate])

  const handleSendAnalysisToChat = useCallback((text: string) => {
    if (!text.trim()) {
      message.warning(t('review:reviewPage.nothingToSend', 'Nothing to send'))
      return
    }
    try {
      const payload = {
        mediaId: nav.selected ? String(nav.selected.id) : undefined,
        title: nav.selected?.title || 'Analysis',
        content: `Please review this analysis and continue the discussion:\n\n${text}`,
        mode: 'normal' as const
      }
      void setSetting(DISCUSS_MEDIA_PROMPT_SETTING, payload)
      window.dispatchEvent(new CustomEvent('tldw:discuss-media', { detail: payload }))
    } catch {
      // ignore storage errors
    }
    setChatMode('normal')
    setSelectedKnowledge(null as any)
    setRagMediaIds(null)
    navigate('/')
    message.success(t('review:reviewPage.sentToChat', 'Sent to chat'))
  }, [nav.selected, setChatMode, setSelectedKnowledge, setRagMediaIds, navigate, message, t])

  const handleRefreshMedia = useCallback(async () => {
    await nav.handleRefreshMedia(showNavigationPanel, () => {
      void refetchNavigation()
    })
  }, [nav, showNavigationPanel, refetchNavigation])

  // When the library is truly empty (no results, no search/filters active, not loading),
  // render a single-column centered onboarding view instead of the two-column split.
  const isEmptyLibrary =
    search.activeTotalCount === 0 &&
    !hasActiveFilters &&
    !search.query?.trim() &&
    !search.isLoading &&
    !search.isFetching

  if (isEmptyLibrary) {
    return (
      <div
        className="relative flex min-h-full bg-bg"
        style={{ minHeight: `${sidebarDimensions.mediaPageMinHeightPx}px` }}
      >
        <div className="flex flex-1 items-center justify-center p-8">
          <div className="max-w-lg w-full">
            <ResultsList
              results={displayResults}
              selectedId={null}
              onSelect={() => {}}
              totalCount={search.activeTotalCount}
              loadedCount={displayResults.length}
              isLoading={false}
              hasActiveFilters={false}
              searchQuery=""
              onClearSearch={() => {}}
              onClearFilters={() => {}}
              onOpenQuickIngest={(detail) => {
                requestQuickIngestOpen(detail)
              }}
              favorites={selection.favoritesSet}
              onToggleFavorite={selection.toggleFavorite}
              selectionMode={false}
              selectedIds={new Set()}
              onToggleSelected={() => {}}
              readingProgress={selection.readingProgressMap}
              viewMode="standard"
              onViewModeChange={() => {}}
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="relative flex min-h-full bg-bg"
      style={{ minHeight: `${sidebarDimensions.mediaPageMinHeightPx}px` }}
    >
      {/* Left Sidebar */}
      <div
        className={`bg-surface border-r border-border flex h-full min-h-0 flex-col transition-[width] duration-300 ease-in-out ${
          viewPrefs.sidebarCollapsedValue ? 'w-0' : 'w-full md:w-[22rem] lg:w-[25rem]'
        }`}
        style={{
          overflowX: 'hidden',
          overflowY: 'auto'
        }}
      >
        <div
          className="flex min-h-full flex-col bg-surface"
          hidden={viewPrefs.sidebarCollapsedValue}
          aria-hidden={viewPrefs.sidebarCollapsedValue}
        >
          {/* Header */}
          <div className="shrink-0 border-b border-border/80 bg-surface px-4 py-3.5">
            <div className="flex items-center justify-between gap-3">
              <h1 className="text-text text-base font-semibold">
                {t('review:mediaPage.mediaInspector', { defaultValue: 'Media Inspector' })}
              </h1>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium tabular-nums text-text-muted">
                  {displayResults.length} / {search.activeTotalCount}
                </span>
                <button
                  type="button"
                  onClick={() => navigate('/media-trash')}
                  className="inline-flex h-7 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text-muted hover:bg-surface2 hover:text-text"
                  aria-label={t('review:mediaPage.openTrash', { defaultValue: 'Trash' })}
                  title={t('review:mediaPage.openTrash', { defaultValue: 'Trash' })}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  {t('review:mediaPage.openTrash', { defaultValue: 'Trash' })}
                </button>
                <button
                  type="button"
                  onClick={selection.handleToggleBulkSelectionMode}
                  className={`inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] ${
                    selection.bulkSelectionMode
                      ? 'border-primary bg-primary/10 text-primaryStrong'
                      : 'border-border text-text-muted hover:bg-surface2 hover:text-text'
                  }`}
                  data-testid="media-bulk-mode-toggle"
                  aria-pressed={selection.bulkSelectionMode}
                >
                  {selection.bulkSelectionMode ? (
                    <CheckSquare className="h-3.5 w-3.5" />
                  ) : (
                    <Square className="h-3.5 w-3.5" />
                  )}
                  {t('review:mediaPage.bulkMode', { defaultValue: 'Bulk' })}
                </button>
              </div>
            </div>

            <div className="mt-3 inline-flex items-center gap-1 rounded-lg border border-border bg-surface2/70 p-1">
              <button
                type="button"
                onClick={() => search.handleKindChange('media')}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  isMediaOnly(search.kinds)
                    ? 'bg-primary text-white'
                    : 'text-text hover:bg-surface'
                }`}
                aria-pressed={isMediaOnly(search.kinds)}
                aria-label={t('review:mediaPage.showMediaOnly', { defaultValue: 'Show media only' })}
              >
                <span>{t('review:mediaPage.media', { defaultValue: 'Media' })}</span>
                <span className="rounded-full bg-black/10 px-1.5 py-0.5 text-[10px] font-medium">
                  {search.mediaTotal}
                </span>
              </button>
              <button
                type="button"
                onClick={() => search.handleKindChange('notes')}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  isNotesOnly(search.kinds)
                    ? 'bg-primary text-white'
                    : 'text-text hover:bg-surface'
                } ${search.searchMode === 'metadata' ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={search.searchMode === 'metadata'}
                title={
                  search.searchMode === 'metadata'
                    ? t('review:mediaPage.notesDisabledInMetadataMode', {
                        defaultValue: 'Notes are unavailable in metadata mode'
                      })
                    : undefined
                }
                aria-pressed={isNotesOnly(search.kinds)}
                aria-label={t('review:mediaPage.showNotesOnly', { defaultValue: 'Show notes only' })}
              >
                <span>{t('review:mediaPage.notes', { defaultValue: 'Notes' })}</span>
                <span className="rounded-full bg-black/10 px-1.5 py-0.5 text-[10px] font-medium">
                  {search.notesTotal}
                </span>
              </button>
            </div>
          </div>

          {/* Controls: Find Media + Jump To + Bulk Toolbar */}
          <div
            className="min-h-0 shrink overflow-y-auto border-b border-border/80"
          >

          {/* Find Media */}
          <div className="bg-surface px-4 py-3.5 space-y-3 border-b border-border/40">
            <div className="flex items-center justify-between">
              <span className="px-1 py-1 text-sm font-medium text-text">
                {t('review:mediaPage.findMedia', { defaultValue: 'Find media' })}
              </span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => search.setSearchCollapsed((prev) => !prev)}
                  className="inline-flex h-7 w-7 items-center justify-center rounded-md text-text-muted hover:bg-surface2 hover:text-text"
                  aria-expanded={!search.searchCollapsed}
                  aria-controls="media-search-panel"
                  aria-label={
                    search.searchCollapsed
                      ? t('review:mediaPage.expandFindMediaPanel', { defaultValue: 'Expand find media panel' })
                      : t('review:mediaPage.collapseFindMediaPanel', { defaultValue: 'Collapse find media panel' })
                  }
                >
                <ChevronDown
                  className={`w-4 h-4 transition-transform ${search.searchCollapsed ? '' : 'rotate-180'}`}
                />
                </button>
              </div>
            </div>
            {!search.searchCollapsed && (
              <div
                id="media-search-panel"
                className="space-y-2.5 pb-1 pr-1"
                onKeyDown={handleKeyPress}
              >
                <SearchBar
                  value={search.query}
                  onChange={search.setQuery}
                  inputRef={search.searchInputRef}
                  hasActiveFilters={hasActiveFilters}
                  onClearAll={resetAllFilters}
                />
                <FilterChips
                  mediaTypes={search.mediaTypes}
                  keywords={search.keywordTokens}
                  excludedKeywords={search.excludeKeywordTokens}
                  showFavoritesOnly={selection.showFavoritesOnly}
                  activeFilterCount={activeFilterCount}
                  onRemoveMediaType={(type) => {
                    search.setMediaTypes((prev) => prev.filter((t) => t !== type))
                    search.setPage(1)
                  }}
                  onRemoveKeyword={(keyword) => {
                    search.setKeywordTokens((prev) => prev.filter((k) => k !== keyword))
                    search.setPage(1)
                  }}
                  onRemoveExcludedKeyword={(keyword) => {
                    search.setExcludeKeywordTokens((prev) => prev.filter((k) => k !== keyword))
                    search.setPage(1)
                  }}
                  onToggleFavorites={() => {
                    selection.setShowFavoritesOnly(false)
                    search.setPage(1)
                  }}
                  onClearAll={resetAllFilters}
                />
                <button
                  onClick={search.handleSearch}
                  data-testid="media-search-submit"
                  className="w-full rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-primaryStrong"
                >
                  {t('review:mediaPage.search', { defaultValue: 'Search' })}
                </button>
                <FilterPanel
                  searchMode={search.searchMode}
                  onSearchModeChange={(nextMode) => {
                    if (nextMode === search.searchMode) return
                    search.setSearchMode(nextMode)
                    if (nextMode === 'metadata') {
                      search.setKinds({ media: true, notes: false })
                    }
                    search.setPage(1)
                  }}
                  mediaTypes={search.availableMediaTypes}
                  selectedMediaTypes={search.mediaTypes}
                  onMediaTypesChange={search.setMediaTypes}
                  sortBy={search.sortBy}
                  onSortByChange={(nextSort) => {
                    search.setSortBy(nextSort)
                    search.setPage(1)
                  }}
                  dateRange={search.dateRange}
                  onDateRangeChange={(nextDateRange) => {
                    search.setDateRange(nextDateRange)
                    search.setPage(1)
                  }}
                  exactPhrase={search.exactPhrase}
                  onExactPhraseChange={(nextExactPhrase) => {
                    search.setExactPhrase(nextExactPhrase)
                    search.setPage(1)
                  }}
                  searchFields={search.searchFields}
                  onSearchFieldsChange={(nextFields) => {
                    search.setSearchFields(nextFields)
                    search.setPage(1)
                  }}
                  enableBoostFields={search.enableBoostFields}
                  onEnableBoostFieldsChange={(enabled) => {
                    search.setEnableBoostFields(enabled)
                    search.setPage(1)
                  }}
                  boostFields={search.boostFields}
                  onBoostFieldsChange={(nextBoostFields) => {
                    search.setBoostFields(nextBoostFields)
                    search.setPage(1)
                  }}
                  metadataFilters={search.metadataFilters}
                  onMetadataFiltersChange={(nextFilters) => {
                    search.setMetadataFilters(nextFilters)
                    search.setPage(1)
                  }}
                  metadataMatchMode={search.metadataMatchMode}
                  onMetadataMatchModeChange={(mode) => {
                    search.setMetadataMatchMode(mode)
                    search.setPage(1)
                  }}
                  metadataValidationError={search.metadataValidationError}
                  selectedKeywords={search.keywordTokens}
                  onKeywordsChange={(kws) => {
                    search.setKeywordTokens(kws)
                    search.setPage(1)
                  }}
                  selectedExcludedKeywords={search.excludeKeywordTokens}
                  onExcludedKeywordsChange={(kws) => {
                    search.setExcludeKeywordTokens(kws)
                    search.setPage(1)
                  }}
                  keywordOptions={search.keywordOptions}
                  keywordSourceMode={search.keywordSourceMode}
                  onKeywordSearch={(txt) => {
                    search.loadKeywordSuggestions(txt)
                  }}
                  showFavoritesOnly={selection.showFavoritesOnly}
                  onShowFavoritesOnlyChange={(show) => {
                    selection.setShowFavoritesOnly(show)
                    search.setPage(1)
                  }}
                  favoritesCount={selection.favoritesSet.size}
                  activeFilterCount={activeFilterCount}
                  onClearAll={resetAllFilters}
                />
                <div className="space-y-2 rounded-md border border-border/80 bg-surface2/60 px-2.5 py-2">
                  <div className="flex items-center gap-2">
                    <label
                      htmlFor="media-collection-filter"
                      className="text-[11px] font-medium text-text-muted"
                    >
                      {t('review:mediaPage.collectionFilterLabel', {
                        defaultValue: 'Collection'
                      })}
                    </label>
                    <select
                      id="media-collection-filter"
                      value={selection.activeCollectionId || ''}
                      onChange={(event) => {
                        const nextValue = event.target.value.trim()
                        selection.setActiveCollectionId(nextValue || null)
                        search.setPage(1)
                      }}
                      className="h-7 flex-1 rounded border border-border bg-surface px-2 text-[11px] text-text"
                      data-testid="media-collection-filter"
                    >
                      <option value="">
                        {t('review:mediaPage.collectionAll', {
                          defaultValue: 'All items'
                        })}
                      </option>
                      {selection.mediaCollections.map((collection) => (
                        <option key={collection.id} value={collection.id}>
                          {collection.name} ({collection.itemIds.length})
                        </option>
                      ))}
                    </select>
                  </div>
                  {selection.activeCollection ? (
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] text-text-muted">
                        {t('review:mediaPage.collectionItemsVisible', {
                          defaultValue: '{{count}} item(s) in this collection.',
                          count: selection.activeCollection.itemIds.length
                        })}
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          void selection.handleOpenCollectionInMultiReview()
                        }}
                        className="rounded border border-border px-2 py-1 text-[11px] font-medium text-text hover:bg-surface"
                        data-testid="media-collection-open-multi"
                      >
                        {t('review:mediaPage.collectionOpenMultiReview', {
                          defaultValue: 'Open in multi-review'
                        })}
                      </button>
                    </div>
                  ) : (
                    <p className="text-[11px] text-text-muted">
                      {t('review:mediaPage.collectionHint', {
                        defaultValue:
                          'Create collections from bulk selection to organize related items.'
                      })}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {selection.bulkSelectionMode ? (
            <React.Suspense fallback={null}>
              <LazyMediaBulkToolbar
                selection={{
                  ...selection,
                  handleSelectAllVisibleItems
                }}
                t={t}
              />
            </React.Suspense>
          ) : null}

          </div>
          {/* end Controls wrapper */}

          {/* Results + pagination flow */}
          <div
            className="flex min-h-0 flex-1 flex-col bg-surface"
            data-sidebar-target-min-height={sidebarDimensions.sidebarResultsPanelMinHeightPx}
            style={{
              minHeight: `${sidebarDimensions.sidebarResultsPanelMinHeightPx}px`
            }}
          >
            <div
              className="min-h-0 flex-1 overflow-y-auto"
              data-sidebar-target-list-height={sidebarDimensions.sidebarResultsListMinHeightPx}
              style={{
                minHeight: `${sidebarDimensions.sidebarResultsListMinHeightPx}px`
              }}
            >
              <ResultsList
                results={displayResults}
                selectedId={nav.selected?.id || null}
                onSelect={(id) => {
                  if (selection.bulkSelectionMode) {
                    selection.toggleBulkItemSelection(id)
                    return
                  }
                  const item = displayResults.find((r) => r.id === id)
                  if (item) nav.setSelected(item)
                }}
                totalCount={search.activeTotalCount}
                loadedCount={displayResults.length}
                isLoading={search.isLoading || search.isFetching}
                hasActiveFilters={hasActiveFilters}
                searchQuery={search.query}
                onClearSearch={() => {
                  search.setQuery('')
                }}
                onClearFilters={resetAllFilters}
                onOpenQuickIngest={(detail) => {
                  requestQuickIngestOpen(detail)
                }}
                favorites={selection.favoritesSet}
                onToggleFavorite={selection.toggleFavorite}
                selectionMode={selection.bulkSelectionMode}
                selectedIds={selection.bulkSelectedIdSet}
                onToggleSelected={selection.toggleBulkItemSelection}
                readingProgress={selection.readingProgressMap}
                viewMode={viewPrefs.resultsViewMode === 'compact' ? 'compact' : 'standard'}
                onViewModeChange={(mode) => void viewPrefs.setResultsViewMode(mode)}
              />
            </div>
            <div className="shrink-0 border-t border-border bg-surface">
              <Pagination
                currentPage={search.page}
                totalPages={search.totalPages}
                onPageChange={search.setPage}
                totalItems={search.activeTotalCount}
                itemsPerPage={search.pageSize}
                currentItemsCount={search.results.length}
                pageSizeOptions={[20, 50, 100]}
                onItemsPerPageChange={(nextPageSize) => {
                  if (nextPageSize === search.pageSize) return
                  search.setPageSize(nextPageSize)
                  search.setPage(1)
                }}
              />
              {/* Keyboard shortcuts hint */}
              <div className="border-t border-border px-4 py-1.5 flex items-center justify-center">
                <button
                  type="button"
                  onClick={() => keyboard.setShortcutsOverlayOpen(true)}
                  className="inline-flex items-center gap-1.5 text-xs text-text-muted hover:text-text transition-colors"
                  title={t('review:shortcuts.pressForHelp', { defaultValue: 'Press ? for keyboard shortcuts' })}
                >
                  <kbd className="inline-flex items-center justify-center min-w-[18px] h-5 px-1 text-[10px] font-mono bg-surface2 border border-border rounded text-text-muted">
                    ?
                  </kbd>
                  <span>{t('review:shortcuts.forKeyboardShortcuts', { defaultValue: 'for shortcuts' })}</span>
                </button>
              </div>
            </div>
          </div>

          {hasJumpTo && (
            <div className="shrink-0 border-t border-border bg-surface px-4 py-2.5">
              <button
                type="button"
                onClick={() => viewPrefs.setJumpToCollapsed((prev) => !prev)}
                className="flex w-full items-center justify-between rounded-md px-1 py-1 text-sm font-medium text-text hover:bg-surface2/60"
                aria-expanded={!viewPrefs.jumpToCollapsed}
                aria-controls="media-jump-bottom-panel"
              >
                <span>{t('review:mediaPage.jumpTo', { defaultValue: 'Jump to' })}</span>
                <ChevronDown
                  className={`w-4 h-4 transition-transform ${viewPrefs.jumpToCollapsed ? '' : 'rotate-180'}`}
                />
              </button>
              {!viewPrefs.jumpToCollapsed && (
                <div
                  id="media-jump-bottom-panel"
                  className="mt-2 overflow-y-auto pr-1"
                >
                  <React.Suspense fallback={null}>
                    <LazyJumpToNavigator
                      results={displayResults.map((r) => ({ id: r.id, title: r.title }))}
                      selectedId={nav.selected?.id || null}
                      onSelect={(id) => {
                        const item = displayResults.find((r) => r.id === id)
                        if (item) nav.setSelected(item)
                      }}
                      maxButtons={12}
                      showLabel={false}
                    />
                  </React.Suspense>
                </div>
              )}
            </div>
          )}

          <div
            className="shrink-0 border-t border-border bg-surface"
            data-testid="media-sidebar-bottom-utilities"
          >
            <button
              type="button"
              onClick={() => viewPrefs.setLibraryToolsCollapsed(!viewPrefs.libraryToolsCollapsedValue)}
              className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-text hover:bg-surface2/40 hover:text-text"
              aria-expanded={!viewPrefs.libraryToolsCollapsedValue}
              aria-controls="media-library-tools-panel"
              data-testid="media-library-tools-toggle"
            >
              <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                {t('review:mediaPage.libraryTools', { defaultValue: 'Library tools' })}
              </span>
              <ChevronDown
                className={`h-4 w-4 text-text-muted transition-transform ${
                  viewPrefs.libraryToolsCollapsedValue ? '' : 'rotate-180'
                }`}
              />
            </button>

            {!viewPrefs.libraryToolsCollapsedValue && (
              <div id="media-library-tools-panel">
                <React.Suspense fallback={null}>
                  <LazyMediaIngestJobsPanel />
                  <LazyMediaLibraryStatsPanel
                    results={displayResults}
                    totalCount={search.activeTotalCount}
                    storageUsage={selection.libraryStorageUsage}
                  />
                </React.Suspense>
              </div>
            )}
          </div>
          <div
            className="shrink-0 border-t border-border/50 bg-surface2/40"
            style={{ height: `${viewPrefs.MEDIA_SIDEBAR_END_BUFFER_PX}px` }}
            aria-hidden="true"
            data-testid="media-sidebar-end-buffer"
          />
        </div>
      </div>

      {/* Collapse Button */}
      <button
        onClick={() => viewPrefs.setSidebarCollapsed(!viewPrefs.sidebarCollapsedValue)}
        className="relative w-6 self-stretch bg-surface border-r border-border hover:bg-surface2 flex items-center justify-center group transition-colors"
        aria-label={viewPrefs.sidebarCollapsedValue ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <div className="flex items-center justify-center w-full h-full">
          {viewPrefs.sidebarCollapsedValue ? (
            <ChevronRight className="w-4 h-4 text-text-subtle group-hover:text-text" />
          ) : (
            <ChevronLeft className="w-4 h-4 text-text-subtle group-hover:text-text" />
          )}
        </div>
      </button>

      {/* Main Content Area */}
      <div className="flex-1 flex min-h-0 flex-col">
        {navigationEnabled ? (
          <div className="border-b border-border bg-surface px-3 py-2">
            <div className="flex flex-wrap items-center gap-3 text-xs text-text-muted">
              <label className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={viewPrefs.navigationPanelVisibleValue}
                  onChange={(event) =>
                    handleNavigationPanelVisibilityChange(event.target.checked)
                  }
                  className="h-3.5 w-3.5 rounded border-border bg-surface"
                />
                <span>
                  {t('review:mediaNavigation.showPanel', {
                    defaultValue: 'Show chapters/sections panel'
                  })}
                </span>
              </label>

              <label className="inline-flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={viewPrefs.includeGeneratedFallbackValue}
                  onChange={(event) =>
                    handleGeneratedFallbackToggle(event.target.checked)
                  }
                  className="h-3.5 w-3.5 rounded border-border bg-surface"
                />
                <span>
                  {t('review:mediaNavigation.generatedFallback', {
                    defaultValue: 'Allow generated fallback structure'
                  })}
                </span>
              </label>

              <span className="ml-auto rounded bg-surface2 px-2 py-0.5 text-[11px] text-text-subtle">
                {navigationStatusLabel}
              </span>
            </div>
          </div>
        ) : null}

        <div className="flex-1 flex min-h-0 flex-col md:flex-row">
          {showNavigationPanel ? (
            <React.Suspense fallback={null}>
              <LazyMediaSectionNavigator
                nodes={navigationNodes}
                selectedNodeId={selectedNavigationNodeId}
                loading={isNavigationLoading}
                error={navigationError}
                onRetry={() => {
                  void refetchNavigation()
                }}
                onSelectNode={(node) => {
                  setSelectedNavigationNodeId(node.id)
                  setNavigationSelectionNonce((prev) => prev + 1)
                  pendingSectionSelectionTelemetryRef.current = {
                    nodeId: node.id,
                    startedAt: Date.now(),
                    source: 'user'
                  }
                  void persistNavigationSelection(node)
                }}
              />
            </React.Suspense>
          ) : null}

          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-end gap-2 border-b border-border bg-surface px-3 py-2">
              <button
                type="button"
                onClick={handleCreateStudyPackFromMedia}
                className="inline-flex items-center rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!nav.selected || !nav.selected?.title?.trim()}
              >
                {t("review:mediaPage.createStudyPack", {
                  defaultValue: "Create study pack"
                })}
              </button>
            </div>
            {nav.staleSelectionNotice ? (
              <div
                className="mx-3 mt-3 rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
                data-testid="media-stale-selection-notice"
              >
                {nav.staleSelectionNotice}
              </div>
            ) : null}
            {nav.selected &&
            nav.detailFetchError &&
            String(nav.detailFetchError.mediaId) === String(nav.selected.id) ? (
              <div
                className="mx-3 mt-3 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn"
                data-testid="media-detail-fetch-error"
              >
                <p className="m-0">{nav.detailFetchError.message}</p>
                <button
                  type="button"
                  onClick={nav.handleRetryDetailFetch}
                  className="mt-2 inline-flex items-center rounded border border-warn/40 bg-surface px-2 py-1 text-xs text-warn hover:bg-surface2"
                  data-testid="media-detail-fetch-retry"
                >
                  {t('common:retry', { defaultValue: 'Retry' })}
                </button>
              </div>
            ) : null}
            <ContentViewer
              selectedMedia={nav.selected}
              content={nav.selectedContent}
              mediaDetail={nav.selectedDetail}
              contentDisplayMode={normalizeRequestedMediaRenderMode(
                viewPrefs.navigationDisplayMode,
                viewPrefs.mediaRichRenderingEnabled
              )}
              resolvedContentFormat={effectiveContentFormat}
              showContentDisplayModeSelector={viewPrefs.mediaDisplayModeSelectorEnabled}
              allowRichRendering={viewPrefs.mediaRichRenderingEnabled}
              onContentDisplayModeChange={(mode) => {
                viewPrefs.setNavigationDisplayMode(
                  normalizeRequestedMediaRenderMode(mode, viewPrefs.mediaRichRenderingEnabled)
                )
              }}
              isDetailLoading={nav.detailLoading}
              onPrevious={nav.handlePrevious}
              onNext={nav.handleNext}
              hasPrevious={nav.hasPrevious}
              hasNext={nav.hasNext}
              currentIndex={nav.selectedIndex >= 0 ? nav.selectedIndex : 0}
              totalResults={displayResults.length}
              isLibraryEmpty={isEmptyLibrary}
              onChatWithMedia={handleChatWithMedia}
              onChatAboutMedia={handleChatAboutMedia}
              onGenerateFlashcardsFromContent={handleGenerateFlashcardsFromMedia}
              onRefreshMedia={handleRefreshMedia}
              onKeywordsUpdated={(mediaId, keywords) => {
                if (nav.selected && nav.selected.id === mediaId) {
                  nav.setSelected({ ...nav.selected, keywords })
                }
                search.refetch()
              }}
              onDeleteItem={selection.handleDeleteItem}
              onCreateNoteWithContent={handleCreateNoteWithContent}
              onOpenInMultiReview={handleOpenInMultiReview}
              onSendAnalysisToChat={handleSendAnalysisToChat}
              contentRef={viewPrefs.contentRef}
              navigationTarget={selectedNavigationTarget}
              navigationNodeTitle={selectedNavigationNode?.title || null}
              navigationPageCountHint={navigationPageCountHint}
              navigationSelectionNonce={navigationSelectionNonce}
            />
          </div>
        </div>
      </div>

      {/* Keyboard Shortcuts Overlay */}
      <React.Suspense fallback={null}>
        <LazyKeyboardShortcutsOverlay
          open={keyboard.shortcutsOverlayOpen}
          onClose={() => keyboard.setShortcutsOverlayOpen(false)}
        />
      </React.Suspense>
    </div>
  )
}

export default ViewMediaPage
