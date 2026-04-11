import {
  Segmented,
  Badge,
  Select,
  Tooltip,
  Popover,
  Switch,
  notification
} from "antd"
import {
  FolderKanban,
  FileText,
  TestTube,
  BarChart3,
  Sparkles,
  Settings
} from "lucide-react"
import React, { Suspense, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { useSearchParams } from "react-router-dom"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  usePromptStudioStore,
  type StudioSubTab
} from "@/store/prompt-studio"
import {
  hasPromptStudio,
  getPromptStudioStatus,
  listProjects
} from "@/services/prompt-studio"
import {
  getPromptStudioDefaults,
  setPromptStudioDefaults
} from "@/services/prompt-studio-settings"
import {
  buildPromptStudioWebSocketUrl,
  isPromptStudioStatusEvent
} from "@/services/prompt-studio-stream"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useMobile } from "@/hooks/useMediaQuery"
import { useServerOnline } from "@/hooks/useServerOnline"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { QueueHealthWidget } from "./QueueHealthWidget"
import { ProjectsTab } from "./Projects/ProjectsTab"

const StudioPromptsTab = React.lazy(() =>
  import("./Prompts/StudioPromptsTab").then((module) => ({
    default: module.StudioPromptsTab
  }))
)

const TestCasesTab = React.lazy(() =>
  import("./TestCases/TestCasesTab").then((module) => ({
    default: module.TestCasesTab
  }))
)

const EvaluationsTab = React.lazy(() =>
  import("./Evaluations/EvaluationsTab").then((module) => ({
    default: module.EvaluationsTab
  }))
)

const OptimizationsTab = React.lazy(() =>
  import("./Optimizations/OptimizationsTab").then((module) => ({
    default: module.OptimizationsTab
  }))
)

const SUB_TAB_OPTIONS: StudioSubTab[] = [
  "projects",
  "prompts",
  "testCases",
  "evaluations",
  "optimizations"
]

const isValidSubTab = (tab: string | null): tab is StudioSubTab =>
  tab !== null && SUB_TAB_OPTIONS.includes(tab as StudioSubTab)

export const getStudioStatusRefetchInterval = (
  status: { processing?: number } | null | undefined
): number => (Number(status?.processing || 0) > 0 ? 5000 : 30000)

const normalizeSettingsProjects = (
  payload: unknown
): Array<{ id: number; name: string }> => {
  const raw =
    (payload as any)?.data?.data ??
    (payload as any)?.data ??
    []

  if (!Array.isArray(raw)) {
    return []
  }

  return raw
    .map((entry) => {
      const id = Number((entry as any)?.id)
      const name = (entry as any)?.name
      if (!Number.isFinite(id) || typeof name !== "string" || name.trim().length === 0) {
        return null
      }
      return { id, name: name.trim() }
    })
    .filter((entry): entry is { id: number; name: string } => entry !== null)
}

export const StudioTabContainer: React.FC = () => {
  const { t } = useTranslation(["settings", "common", "option"])
  const [searchParams, setSearchParams] = useSearchParams()
  const isOnline = useServerOnline()
  const isMobile = useMobile()
  const queryClient = useQueryClient()

  const activeSubTab = usePromptStudioStore((s) => s.activeSubTab)
  const setActiveSubTab = usePromptStudioStore((s) => s.setActiveSubTab)
  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const setSelectedProjectId = usePromptStudioStore((s) => s.setSelectedProjectId)

  // Capability check
  const { data: hasStudio, isLoading: isCheckingCapability } = useQuery({
    queryKey: ["prompt-studio", "capability"],
    queryFn: hasPromptStudio,
    enabled: isOnline
  })

  // Queue health status
  const { data: statusResponse } = useQuery({
    queryKey: ["prompt-studio", "status"],
    queryFn: () => getPromptStudioStatus(),
    enabled: isOnline && hasStudio === true,
    refetchInterval: (query) =>
      getStudioStatusRefetchInterval((query.state.data as any)?.data?.data)
  })

  const status = (statusResponse as any)?.data?.data

  const { data: settingsDefaultsResponse } = useQuery({
    queryKey: ["prompt-studio", "settings-defaults"],
    queryFn: getPromptStudioDefaults,
    enabled: isOnline && hasStudio === true
  })

  const { data: settingsProjectsResponse } = useQuery({
    queryKey: ["prompt-studio", "settings-projects"],
    queryFn: () => listProjects({ page: 1, per_page: 100 }),
    enabled: isOnline && hasStudio === true
  })

  const settingsDefaults = {
    defaultProjectId:
      typeof settingsDefaultsResponse?.defaultProjectId === "number"
        ? settingsDefaultsResponse.defaultProjectId
        : null,
    autoSyncWorkspacePrompts:
      settingsDefaultsResponse?.autoSyncWorkspacePrompts !== false
  }

  const settingsProjects = normalizeSettingsProjects(settingsProjectsResponse)

  const updateSettingsMutation = useMutation({
    mutationFn: (updates: {
      defaultProjectId?: number | null
      autoSyncWorkspacePrompts?: boolean
    }) => setPromptStudioDefaults(updates),
    onSuccess: (nextDefaults, updates) => {
      queryClient.setQueryData(["prompt-studio", "settings-defaults"], nextDefaults)
      if (
        typeof updates.defaultProjectId === "number" &&
        updates.defaultProjectId > 0 &&
        selectedProjectId === null
      ) {
        setSelectedProjectId(updates.defaultProjectId)
      }
      notification.success({
        message: t("managePrompts.studio.settings.saved", {
          defaultValue: "Studio settings updated"
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("managePrompts.studio.settings.saveFailed", {
          defaultValue: "Could not save Studio settings"
        }),
        description: error?.message
      })
    }
  })

  const defaultProjectValue =
    typeof settingsDefaults.defaultProjectId === "number"
      ? settingsDefaults.defaultProjectId
      : "none"

  const settingsProjectOptions = [
    {
      value: "none",
      label: t("managePrompts.studio.settings.noDefaultProject", {
        defaultValue: "No default project"
      })
    },
    ...settingsProjects.map((project) => ({
      value: project.id,
      label: project.name
    }))
  ]

  // Sync URL with active sub-tab
  useEffect(() => {
    const subtabParam = searchParams.get("subtab")
    if (isValidSubTab(subtabParam) && subtabParam !== activeSubTab) {
      setActiveSubTab(subtabParam)
    }
  }, [searchParams, activeSubTab, setActiveSubTab])

  // Update URL when sub-tab changes
  useEffect(() => {
    const currentSubtab = searchParams.get("subtab")
    if (currentSubtab !== activeSubTab) {
      const newParams = new URLSearchParams(searchParams)
      newParams.set("subtab", activeSubTab)
      setSearchParams(newParams, { replace: true })
    }
  }, [activeSubTab, searchParams, setSearchParams])

  useEffect(() => {
    if (!selectedProjectId && activeSubTab !== "projects") {
      setActiveSubTab("projects")
    }
  }, [selectedProjectId, activeSubTab, setActiveSubTab])

  useEffect(() => {
    if (!isOnline || hasStudio !== true || typeof window === "undefined") {
      return
    }

    let ws: WebSocket | null = null
    let disposed = false

    const openStatusStream = async () => {
      try {
        const config = await tldwClient.getConfig()
        if (disposed || !config) return

        const wsUrl = buildPromptStudioWebSocketUrl(config, selectedProjectId)
        ws = new WebSocket(wsUrl)

        ws.onopen = () => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return
          const subscribePayload = selectedProjectId
            ? { type: "subscribe", project_id: selectedProjectId }
            : { type: "subscribe" }
          ws.send(JSON.stringify(subscribePayload))
        }

        ws.onmessage = (event) => {
          if (typeof event.data !== "string") return
          try {
            const payload = JSON.parse(event.data)
            if (isPromptStudioStatusEvent(payload)) {
              void queryClient.invalidateQueries({
                queryKey: ["prompt-studio", "status"]
              })
            }
          } catch {
            // Ignore non-JSON websocket frames.
          }
        }

        ws.onerror = () => {
          // Polling remains active as fallback when websocket errors.
        }
      } catch {
        // Polling remains active as fallback when websocket setup fails.
      }
    }

    void openStatusStream()

    return () => {
      disposed = true
      if (ws && ws.readyState < WebSocket.CLOSING) {
        ws.close()
      }
    }
  }, [isOnline, hasStudio, selectedProjectId, queryClient])

  const handleSubTabChange = (value: string | number) => {
    if (isValidSubTab(value as string)) {
      setActiveSubTab(value as StudioSubTab)
    }
  }

  const handleDefaultProjectChange = (value: string | number) => {
    const nextDefaultProjectId =
      value === "none" ? null : Number.isFinite(Number(value)) ? Number(value) : null
    updateSettingsMutation.mutate({
      defaultProjectId: nextDefaultProjectId
    })
  }

  const handleAutoSyncChange = (checked: boolean) => {
    updateSettingsMutation.mutate({
      autoSyncWorkspacePrompts: checked
    })
  }

  // Offline state
  if (!isOnline) {
    return (
      <WorkspaceConnectionGate
        featureName={t("settings:managePrompts.studio.title", {
          defaultValue: "Prompt Studio"
        })}
        setupDescription={t("settings:managePrompts.studio.connectDescription", {
          defaultValue:
            "To access full Prompt Studio features, connect to your tldw server first."
        })}
      >
        <div />
      </WorkspaceConnectionGate>
    )
  }

  // Loading capability check
  if (isCheckingCapability) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-pulse text-text-muted">
          {t("common:loading.title", { defaultValue: "Loading..." })}
        </div>
      </div>
    )
  }

  // Prompt Studio not available
  if (hasStudio === false) {
    return (
      <FeatureEmptyState
        title={t("managePrompts.studio.unavailableTitle", {
          defaultValue: "Prompt Studio not available"
        })}
        description={t("managePrompts.studio.unavailableDescription", {
          defaultValue:
            "Your server doesn't have Prompt Studio enabled, or you don't have permission to access it."
        })}
        examples={[
          t("managePrompts.studio.unavailableExample1", {
            defaultValue:
              "Contact your server administrator to enable Prompt Studio."
          }),
          t("managePrompts.studio.unavailableExample2", {
            defaultValue:
              "You can still create and manage prompts locally in the Custom tab."
          })
        ]}
      />
    )
  }

  const selectProjectFirstLabel = t("managePrompts.studio.selectProjectFirstShort", {
    defaultValue: "Select a project first"
  })

  const projectsLabel = t("managePrompts.studio.tabs.projects", {
    defaultValue: "Projects"
  })
  const promptsLabel = t("managePrompts.studio.tabs.prompts", {
    defaultValue: "Prompts"
  })
  const testCasesLabel = t("managePrompts.studio.tabs.testCases", {
    defaultValue: "Test Cases"
  })
  const evaluationsLabel = t("managePrompts.studio.tabs.evaluations", {
    defaultValue: "Evaluations"
  })
  const optimizationsLabel = t("managePrompts.studio.tabs.optimizations", {
    defaultValue: "Optimizations"
  })

  const segmentedOptions = [
    {
      value: "projects",
      label: (
        <span className="flex items-center gap-1.5" aria-label={projectsLabel}>
          <FolderKanban className="size-4" aria-hidden="true" />
          <span>{projectsLabel}</span>
        </span>
      )
    },
    {
      value: "prompts",
      label: (
        <Tooltip title={!selectedProjectId ? selectProjectFirstLabel : undefined}>
          <span
            className="flex items-center gap-1.5"
            aria-label={
              !selectedProjectId
                ? `${promptsLabel} (${selectProjectFirstLabel})`
                : promptsLabel
            }
            title={!selectedProjectId ? selectProjectFirstLabel : undefined}
          >
            <FileText className="size-4" aria-hidden="true" />
            <span>{promptsLabel}</span>
          </span>
        </Tooltip>
      ),
      disabled: !selectedProjectId
    },
    {
      value: "testCases",
      label: (
        <Tooltip title={!selectedProjectId ? selectProjectFirstLabel : undefined}>
          <span
            className="flex items-center gap-1.5"
            aria-label={
              !selectedProjectId
                ? `${testCasesLabel} (${selectProjectFirstLabel})`
                : testCasesLabel
            }
            title={!selectedProjectId ? selectProjectFirstLabel : undefined}
          >
            <TestTube className="size-4" aria-hidden="true" />
            <span>{testCasesLabel}</span>
          </span>
        </Tooltip>
      ),
      disabled: !selectedProjectId
    },
    {
      value: "evaluations",
      label: (
        <Tooltip title={!selectedProjectId ? selectProjectFirstLabel : undefined}>
          <span
            className="flex items-center gap-1.5"
            aria-label={
              !selectedProjectId
                ? `${evaluationsLabel} (${selectProjectFirstLabel})`
                : evaluationsLabel
            }
            title={!selectedProjectId ? selectProjectFirstLabel : undefined}
          >
            <BarChart3 className="size-4" aria-hidden="true" />
            <span>{evaluationsLabel}</span>
            {status?.processing > 0 && (
              <Badge
                count={status.processing}
                size="small"
                style={{ backgroundColor: "rgb(var(--color-success))" }}
              />
            )}
          </span>
        </Tooltip>
      ),
      disabled: !selectedProjectId
    },
    {
      value: "optimizations",
      label: (
        <Tooltip title={!selectedProjectId ? selectProjectFirstLabel : undefined}>
          <span
            className="flex items-center gap-1.5"
            aria-label={
              !selectedProjectId
                ? `${optimizationsLabel} (${selectProjectFirstLabel})`
                : optimizationsLabel
            }
            title={!selectedProjectId ? selectProjectFirstLabel : undefined}
          >
            <Sparkles className="size-4" aria-hidden="true" />
            <span>{optimizationsLabel}</span>
          </span>
        </Tooltip>
      ),
      disabled: !selectedProjectId
    }
  ]

  const mobileOptions = segmentedOptions.map((option) => {
    const labelText =
      option.value === "projects"
        ? projectsLabel
        : option.value === "prompts"
          ? promptsLabel
          : option.value === "testCases"
            ? testCasesLabel
            : option.value === "evaluations"
              ? evaluationsLabel
              : optimizationsLabel

    return {
      value: option.value,
      disabled: option.disabled,
      label: option.disabled
        ? `${labelText} (${selectProjectFirstLabel})`
        : labelText
    }
  })

  const renderStudioSubTab = () => {
    switch (activeSubTab) {
      case "projects":
        return <ProjectsTab />
      case "prompts":
        return (
          <Suspense fallback={<div className="py-8 text-sm text-text-muted">Loading prompts...</div>}>
            <StudioPromptsTab />
          </Suspense>
        )
      case "testCases":
        return (
          <Suspense fallback={<div className="py-8 text-sm text-text-muted">Loading test cases...</div>}>
            <TestCasesTab />
          </Suspense>
        )
      case "evaluations":
        return (
          <Suspense fallback={<div className="py-8 text-sm text-text-muted">Loading evaluations...</div>}>
            <EvaluationsTab />
          </Suspense>
        )
      case "optimizations":
        return (
          <Suspense fallback={<div className="py-8 text-sm text-text-muted">Loading optimizations...</div>}>
            <OptimizationsTab />
          </Suspense>
        )
      default:
        return <ProjectsTab />
    }
  }

  return (
    <div className="space-y-4" data-testid="studio-tab-container">
      {/* Header with sub-tabs and status */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {isMobile ? (
          <Select
            value={activeSubTab}
            onChange={handleSubTabChange}
            options={mobileOptions}
            className="w-full"
            data-testid="studio-subtab-select-mobile"
          />
        ) : (
          <Segmented
            value={activeSubTab}
            onChange={handleSubTabChange}
            options={segmentedOptions}
            data-testid="studio-subtab-selector"
          />
        )}

        <div className="flex items-center gap-2">
          <Popover
            trigger="click"
            placement="bottomRight"
            content={
              <div className="w-72 space-y-4" data-testid="studio-settings-popover">
                <div className="space-y-1">
                  <p className="text-xs font-medium text-text-muted">
                    {t("managePrompts.studio.settings.defaultProjectLabel", {
                      defaultValue: "Default project"
                    })}
                  </p>
                  <Select
                    value={defaultProjectValue}
                    options={settingsProjectOptions}
                    onChange={handleDefaultProjectChange}
                    loading={updateSettingsMutation.isPending}
                    data-testid="studio-settings-default-project"
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-text">
                      {t("managePrompts.studio.settings.autoSyncLabel", {
                        defaultValue: "Auto-sync workspace prompts"
                      })}
                    </p>
                    <p className="text-xs text-text-muted">
                      {t("managePrompts.studio.settings.autoSyncHint", {
                        defaultValue:
                          "When enabled, local prompt saves will auto-push to Studio when possible."
                      })}
                    </p>
                  </div>
                  <Switch
                    checked={settingsDefaults.autoSyncWorkspacePrompts}
                    loading={updateSettingsMutation.isPending}
                    onChange={handleAutoSyncChange}
                    data-testid="studio-settings-auto-sync"
                  />
                </div>
              </div>
            }
          >
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded border border-border bg-bg px-2 py-1 text-xs text-text-muted hover:text-text hover:bg-surface2"
              data-testid="studio-settings-button"
            >
              <Settings className="size-4" />
              {!isMobile && (
                <span>
                  {t("managePrompts.studio.settings.button", {
                    defaultValue: "Settings"
                  })}
                </span>
              )}
            </button>
          </Popover>
          <QueueHealthWidget status={status} />
        </div>
      </div>

      {/* Project context reminder */}
      {!selectedProjectId && (
        <div className="p-4 bg-warn/10 border border-warn/30 rounded-md">
          <p className="text-sm text-warn">
            {t("managePrompts.studio.selectProjectFirstDetails", {
              defaultValue:
                "Select a project in the Projects tab to unlock Prompts, Test Cases, Evaluations, and Optimizations."
            })}
          </p>
        </div>
      )}

      {/* Content area */}
      <div className="min-h-[400px]">{renderStudioSubTab()}</div>
    </div>
  )
}
