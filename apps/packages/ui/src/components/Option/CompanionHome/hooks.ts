import React from "react"

import {
  fetchPersonalizationProfile,
  type PersonalizationProfile,
  updatePersonalizationOptIn
} from "@/services/companion"
import {
  fetchCompanionHomeSnapshot,
  type CompanionHomeSnapshot,
  type CompanionHomeSurface
} from "@/services/companion-home"
import { listNotifications } from "@/services/notifications"
import { listScheduledTasks } from "@/services/scheduled-tasks-control-plane"
import {
  DEFAULT_COMPANION_HOME_LAYOUT,
  loadCompanionHomeLayout,
  saveCompanionHomeLayout,
  type CompanionHomeLayoutCard
} from "@/store/companion-home-layout"
import {
  buildScheduledTaskAutomationHomeItems,
  buildScheduledTaskAutomationHomeItemsFromNotifications,
  mergeScheduledTaskAutomationHomeItems,
  projectScheduledTaskResults,
  type ScheduledTaskAutomationHomeItem
} from "../ScheduledTasks/scheduled-task-results"

export const createEmptySnapshot = (
  surface: CompanionHomeSurface
): CompanionHomeSnapshot => ({
  surface,
  inbox: [],
  needsAttention: [],
  resumeWork: [],
  goalsFocus: [],
  recentActivity: [],
  readingQueue: [],
  degradedSources: ["workspace", "reading", "notes"],
  summary: {
    activityCount: 0,
    inboxCount: 0,
    needsAttentionCount: 0,
    resumeWorkCount: 0
  }
})

type UseCompanionHomeDataArgs = {
  surface: CompanionHomeSurface
  capsLoading: boolean
  hasPersonalization: boolean
  onPersonalizationEnabled?: () => void
}

type UseCompanionHomeDataResult = {
  snapshot: CompanionHomeSnapshot | null
  profile: PersonalizationProfile | null
  profileLoaded: boolean
  loading: boolean
  error: string | null
  enablingCompanion: boolean
  refresh: () => void
  enableCompanion: () => Promise<void>
}

export const useCompanionHomeData = ({
  surface,
  capsLoading,
  hasPersonalization,
  onPersonalizationEnabled
}: UseCompanionHomeDataArgs): UseCompanionHomeDataResult => {
  const [snapshot, setSnapshot] = React.useState<CompanionHomeSnapshot | null>(null)
  const [profile, setProfile] = React.useState<PersonalizationProfile | null>(null)
  const [profileLoaded, setProfileLoaded] = React.useState(false)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [enablingCompanion, setEnablingCompanion] = React.useState(false)
  const [refreshToken, setRefreshToken] = React.useState(0)

  React.useEffect(() => {
    if (capsLoading) return

    let cancelled = false
    setLoading(true)
    setError(null)
    setProfileLoaded(false)

    if (!hasPersonalization) {
      setSnapshot(createEmptySnapshot(surface))
      setProfile(null)
      setProfileLoaded(true)
      setLoading(false)
      return () => {
        cancelled = true
      }
    }

    const load = async () => {
      const [snapshotResult, profileResult] = await Promise.allSettled([
        fetchCompanionHomeSnapshot(surface),
        hasPersonalization ? fetchPersonalizationProfile() : Promise.resolve(null)
      ])

      if (cancelled) {
        return
      }

      if (snapshotResult.status === "fulfilled") {
        setSnapshot(snapshotResult.value)
      } else {
        setSnapshot(createEmptySnapshot(surface))
        setError("Companion Home is partially unavailable right now.")
      }

      if (profileResult.status === "fulfilled") {
        setProfile(profileResult.value)
        setProfileLoaded(true)
      } else {
        setProfile(null)
        setError((current) => current || "Companion setup status could not be loaded.")
      }

      setLoading(false)
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [capsLoading, hasPersonalization, refreshToken, surface])

  const refresh = React.useCallback(() => {
    setRefreshToken((value) => value + 1)
  }, [])

  const enableCompanion = React.useCallback(async () => {
    setEnablingCompanion(true)
    setError(null)
    try {
      const nextProfile = await updatePersonalizationOptIn(true)
      setProfile(nextProfile)
      onPersonalizationEnabled?.()
      refresh()
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Failed to enable companion personalization."
      )
    } finally {
      setEnablingCompanion(false)
    }
  }, [onPersonalizationEnabled, refresh])

  return {
    snapshot,
    profile,
    profileLoaded,
    loading,
    error,
    enablingCompanion,
    refresh,
    enableCompanion
  }
}

type UseScheduledTaskHomeSignalsArgs = {
  enabled: boolean
}

type UseScheduledTaskHomeSignalsResult = {
  items: ScheduledTaskAutomationHomeItem[]
  loading: boolean
  partial: boolean
  error: string | null
  refresh: () => void
}

export const useScheduledTaskHomeSignals = ({
  enabled
}: UseScheduledTaskHomeSignalsArgs): UseScheduledTaskHomeSignalsResult => {
  const [items, setItems] = React.useState<ScheduledTaskAutomationHomeItem[]>([])
  const [loading, setLoading] = React.useState(true)
  const [partial, setPartial] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [refreshToken, setRefreshToken] = React.useState(0)

  React.useEffect(() => {
    if (!enabled) return

    let cancelled = false
    setLoading(true)
    setError(null)

    const load = async () => {
      const [tasksResult, notificationsResult] = await Promise.allSettled([
        listScheduledTasks(),
        listNotifications({ limit: 50 })
      ])

      if (cancelled) {
        return
      }

      const projectedItems =
        tasksResult.status === "fulfilled"
          ? buildScheduledTaskAutomationHomeItems(
              projectScheduledTaskResults(tasksResult.value?.items ?? [])
            )
          : []
      const notificationItems =
        notificationsResult.status === "fulfilled"
          ? buildScheduledTaskAutomationHomeItemsFromNotifications(
              notificationsResult.value.items
            )
          : []
      const nextPartial =
        tasksResult.status === "rejected" ||
        notificationsResult.status === "rejected" ||
        (tasksResult.status === "fulfilled" && Boolean(tasksResult.value?.partial))

      let nextError: string | null = null
      if (tasksResult.status === "rejected") {
        nextError =
          notificationItems.length > 0
            ? "Some scheduled-task signals could not be loaded."
            : "Automation signals unavailable"
      } else if (notificationsResult.status === "rejected") {
        nextError = "Recent automation notifications could not be loaded."
      } else if (tasksResult.value?.partial) {
        nextError = "Some scheduled-task sources are temporarily unavailable."
      }

      setItems(
        mergeScheduledTaskAutomationHomeItems([
          projectedItems,
          notificationItems
        ])
      )
      setPartial(nextPartial)
      setError(nextError)
      setLoading(false)
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [enabled, refreshToken])

  const refresh = React.useCallback(() => {
    setRefreshToken((value) => value + 1)
  }, [])

  return {
    items,
    loading,
    partial,
    error,
    refresh
  }
}

type UseCompanionHomeLayoutResult = {
  layout: CompanionHomeLayoutCard[] | null
  updateLayout: (nextLayout: CompanionHomeLayoutCard[]) => void
}

export const useCompanionHomeLayout = (
  surface: CompanionHomeSurface
): UseCompanionHomeLayoutResult => {
  const [layout, setLayout] = React.useState<CompanionHomeLayoutCard[] | null>(null)
  const layoutLoadRequestRef = React.useRef(0)
  const layoutMutationVersionRef = React.useRef(0)

  React.useEffect(() => {
    let cancelled = false
    const requestId = layoutLoadRequestRef.current + 1
    const mutationVersionAtRequest = layoutMutationVersionRef.current

    layoutLoadRequestRef.current = requestId
    setLayout(null)

    const loadLayout = async () => {
      try {
        const nextLayout = await loadCompanionHomeLayout(surface)
        if (
          !cancelled &&
          layoutLoadRequestRef.current === requestId &&
          layoutMutationVersionRef.current === mutationVersionAtRequest
        ) {
          setLayout(nextLayout)
        }
      } catch {
        if (
          !cancelled &&
          layoutLoadRequestRef.current === requestId &&
          layoutMutationVersionRef.current === mutationVersionAtRequest
        ) {
          setLayout(DEFAULT_COMPANION_HOME_LAYOUT)
        }
      }
    }

    void loadLayout()

    return () => {
      cancelled = true
    }
  }, [surface])

  const updateLayout = React.useCallback(
    (nextLayout: CompanionHomeLayoutCard[]) => {
      layoutMutationVersionRef.current += 1
      setLayout(nextLayout)
      void saveCompanionHomeLayout(surface, nextLayout)
    },
    [surface]
  )

  return {
    layout,
    updateLayout
  }
}
