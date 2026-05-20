import { Input, InputNumber, Popover, Radio, Select, Switch, Tooltip, Upload } from "antd"
import { useQuery } from "@tanstack/react-query"
import {
  Search,
  MoreHorizontal,
  Eye,
  Globe,
  Image as ImageIcon,
  UploadCloud,
  ExternalLink,
  ChevronRight,
  MessageSquare,
  X
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { ModelSelect } from "@/components/Common/ModelSelect"
import { PromptSelect } from "@/components/Common/PromptSelect"
import { FeatureHint, useFeatureHintSeen } from "@/components/Common/FeatureHint"
import { McpToolSelector } from "@/components/Common/McpToolSelector"
import { ConversationContextPopover } from "./ConversationContextPopover"
import { useChatMoodBadgePreference } from "@/hooks/useChatMoodBadgePreference"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useMcpTools } from "@/hooks/useMcpTools"
import { browser } from "wxt/browser"
import { useStorage } from "@plasmohq/storage/hook"
import { fetchChatModels } from "@/services/tldw-server"
import {
  buildQuickIngestOpenDetailFromUrl,
  requestQuickIngestOpen,
  type QuickIngestOpenDetail
} from "@/utils/quick-ingest-open"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { buildSidepanelFullAppChatPath } from "@/utils/sidepanel-full-app-route"
import type { ToolChoice } from "@/store/option"
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings"
import type { ConversationContextComposition } from "@/types/conversation-context"
import type { ConversationContextCompositionStatus } from "@/hooks/chat/useConversationContextComposition"

interface ControlRowProps {
  // Prompt selection
  selectedSystemPrompt: string | undefined
  setSelectedSystemPrompt: (promptId: string | undefined) => void
  setSelectedQuickPrompt: (prompt: string | undefined) => void
  // Character selection
  selectedCharacterId: string | null
  setSelectedCharacterId: (id: string | null) => void
  // Conversation context
  serverChatId?: string | null
  conversationContextComposition?: ConversationContextComposition | null
  conversationContextStatus?: ConversationContextCompositionStatus
  conversationContextSaveSelection?: (
    selection: {
      worldBookIds: number[]
      dictionaryIds: number[]
    }
  ) => Promise<unknown> | unknown
  // Toggles
  webSearch: boolean
  setWebSearch: (value: boolean) => void
  chatMode: "normal" | "rag" | "vision"
  setChatMode: (mode: "normal" | "rag" | "vision") => void
  toolChoice: ToolChoice
  setToolChoice: (value: ToolChoice) => void
  chatLoopStatus?: "idle" | "running" | "complete" | "error" | "cancelled"
  pendingApprovalsCount?: number
  runningToolCount?: number
  // Image upload
  onImageUpload: (file: File) => void
  // RAG toggle
  onToggleRag: () => void
  // Connection state
  isConnected: boolean
}

const ControlRowBase: React.FC<ControlRowProps> = ({
  selectedSystemPrompt,
  setSelectedSystemPrompt,
  setSelectedQuickPrompt,
  selectedCharacterId,
  setSelectedCharacterId,
  serverChatId,
  conversationContextComposition,
  conversationContextStatus = "idle",
  conversationContextSaveSelection,
  webSearch,
  setWebSearch,
  chatMode,
  setChatMode,
  toolChoice,
  setToolChoice,
  chatLoopStatus = "idle",
  pendingApprovalsCount = 0,
  runningToolCount = 0,
  onImageUpload,
  onToggleRag,
  isConnected
}) => {
  const { t } = useTranslation(["sidepanel", "playground", "common"])
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null)
  const [moreOpen, setMoreOpen] = React.useState(false)
  const [systemPromptOverride, setSystemPromptOverride] = React.useState<
    string | undefined
  >(undefined)
  const moreBtnRef = React.useRef<HTMLButtonElement>(null)
  const { capabilities } = useServerCapabilities()
  const [activePlaylistDetail, setActivePlaylistDetail] =
    React.useState<QuickIngestOpenDetail | null>(null)
  const {
    hasMcp,
    healthState: mcpHealthState,
    discoveredTools: discoveredMcpTools,
    chatTools: chatMcpTools,
    toolCounts: mcpToolCounts,
    toolsLoading: mcpToolsLoading,
    catalogs: mcpCatalogs,
    catalogsLoading: mcpCatalogsLoading,
    toolCatalog,
    toolCatalogId,
    toolModules,
    moduleOptions,
    moduleOptionsLoading,
    toolCatalogStrict,
    setToolCatalog,
    setToolCatalogId,
    setToolModules,
    setToolCatalogStrict,
    setToolEnabled: setMcpToolEnabled,
    resetToolFilter: resetMcpToolFilter
  } = useMcpTools()
  const chatMcpToolCount = chatMcpTools.length
  const translateMcpToolSelector = React.useCallback(
    (key: string, fallback?: string, options?: Record<string, unknown>) =>
      t(key, fallback ?? key, options),
    [t]
  )

  const [catalogDraft, setCatalogDraft] = React.useState(toolCatalog)
  const [advancedToolsExpanded, setAdvancedToolsExpanded] = React.useState(false)
  const [allowExternalImages, setAllowExternalImages] = useStorage(
    "allowExternalImages",
    DEFAULT_CHAT_SETTINGS.allowExternalImages
  )
  const [showMoodBadge, setShowMoodBadge] = useChatMoodBadgePreference()
  const [showMoodConfidence, setShowMoodConfidence] = useStorage(
    "chatShowMoodConfidence",
    Boolean(selectedCharacterId)
  )

  React.useEffect(() => {
    setCatalogDraft(toolCatalog)
  }, [toolCatalog])

  const commitCatalog = React.useCallback(() => {
    const next = catalogDraft.trim()
    if (next !== toolCatalog) {
      setToolCatalog(next)
    }
    if (toolCatalogId !== null && next !== toolCatalog) {
      setToolCatalogId(null)
    }
  }, [catalogDraft, setToolCatalog, toolCatalog, toolCatalogId, setToolCatalogId])

  const catalogGroups = React.useMemo(() => {
    const global: typeof mcpCatalogs = []
    const org: typeof mcpCatalogs = []
    const team: typeof mcpCatalogs = []
    for (const catalog of mcpCatalogs) {
      if (!catalog) continue
      if (catalog.team_id != null) {
        team.push(catalog)
      } else if (catalog.org_id != null) {
        org.push(catalog)
      } else {
        global.push(catalog)
      }
    }
    return { global, org, team }
  }, [mcpCatalogs])

  const catalogById = React.useMemo(() => {
    const map = new Map<number, (typeof mcpCatalogs)[number]>()
    for (const catalog of mcpCatalogs) {
      if (catalog?.id == null) continue
      map.set(catalog.id, catalog)
    }
    return map
  }, [mcpCatalogs])

  const handleCatalogSelect = React.useCallback(
    (value?: number) => {
      if (value === null || value === undefined) {
        setToolCatalogId(null)
        setToolCatalog("")
        return
      }
      const catalog = catalogById.get(value)
      setToolCatalogId(value)
      if (catalog?.name) {
        setToolCatalog(catalog.name)
      }
    },
    [catalogById, setToolCatalog, setToolCatalogId]
  )

  const handleModuleSelect = React.useCallback(
    (value?: string[]) => {
      setToolModules(Array.isArray(value) ? value : [])
    },
    [setToolModules]
  )

  const [selectedModel] = useStorage<string | null>("selectedModel", null)
  const { data: chatModels } = useQuery({
    queryKey: ["mcp-small-models"],
    queryFn: () => fetchChatModels({ returnEmpty: true })
  })
  const selectedModelMeta = React.useMemo(() => {
    if (!selectedModel || !Array.isArray(chatModels)) return null
    return chatModels.find((model) => model.model === selectedModel) || null
  }, [chatModels, selectedModel])
  const modelCapabilities = React.useMemo(() => {
    const caps = selectedModelMeta?.details?.capabilities
    return Array.isArray(caps) ? caps.map((cap) => String(cap).toLowerCase()) : []
  }, [selectedModelMeta])
  const modelContextLength = React.useMemo(() => {
    const value =
      selectedModelMeta?.context_length ??
      selectedModelMeta?.contextLength ??
      selectedModelMeta?.details?.context_length
    return typeof value === "number" && Number.isFinite(value) ? value : null
  }, [selectedModelMeta])
  const isSmallModel =
    modelCapabilities.includes("fast") ||
    (typeof modelContextLength === "number" && modelContextLength <= 8192)

  const toolRunStatusLabel = React.useMemo(() => {
    if (pendingApprovalsCount > 0) {
      return t("sidepanel:controlRow.toolRunPending", "Pending approval")
    }
    if (runningToolCount > 0 || chatLoopStatus === "running") {
      return t("sidepanel:controlRow.toolRunRunning", "Running")
    }
    if (chatLoopStatus === "error") {
      return t("sidepanel:controlRow.toolRunFailed", "Failed")
    }
    if (chatLoopStatus === "complete") {
      return t("sidepanel:controlRow.toolRunDone", "Done")
    }
    return t("sidepanel:controlRow.toolRunIdle", "Idle")
  }, [chatLoopStatus, pendingApprovalsCount, runningToolCount, t])

  // Track if hints have been seen
  const knowledgeHintSeen = useFeatureHintSeen("knowledge-search")
  const moreToolsHintSeen = useFeatureHintSeen("more-tools")

  React.useEffect(() => {
    setSystemPromptOverride(undefined)
  }, [selectedSystemPrompt])

  const playlistImportEnabled =
    Boolean(isConnected) && Boolean(capabilities?.hasMediaPlaylistPreflight)

  const resolveActivePlaylistDetail =
    React.useCallback(async (): Promise<QuickIngestOpenDetail | null> => {
      if (!playlistImportEnabled) return null
      try {
        const tabs = await browser.tabs?.query({
          active: true,
          currentWindow: true
        })
        const activeTab = tabs?.find(
          (tab) => typeof tab.url === "string" && tab.url.trim().length > 0
        )
        return activeTab?.url ? buildQuickIngestOpenDetailFromUrl(activeTab.url) : null
      } catch {
        return null
      }
    }, [playlistImportEnabled])

  React.useEffect(() => {
    let cancelled = false

    if (!moreOpen || !playlistImportEnabled) {
      setActivePlaylistDetail(null)
      return
    }

    void resolveActivePlaylistDetail().then((detail) => {
      if (!cancelled) {
        setActivePlaylistDetail(detail)
      }
    })

    return () => {
      cancelled = true
    }
  }, [moreOpen, playlistImportEnabled, resolveActivePlaylistDetail])

  const quickIngestLabel =
    playlistImportEnabled && activePlaylistDetail
      ? t("sidepanel:controlRow.importPlaylist", "Import playlist to tldw")
      : t("sidepanel:controlRow.quickIngest", "Quick Ingest")
  const rolePlayKind =
    selectedAssistant?.kind === "persona"
      ? "persona"
      : selectedAssistant?.kind === "character" || selectedCharacterId
        ? "character"
        : null
  const rolePlayActive = Boolean(rolePlayKind)
  const rolePlayKindLabel =
    rolePlayKind === "persona"
      ? t("sidepanel:controlRow.persona", "Persona")
      : t("sidepanel:controlRow.character", "Character")
  const rolePlayName =
    selectedAssistant?.name?.trim() ||
    (selectedCharacterId
      ? t("sidepanel:controlRow.characterById", "Character {{id}}", {
          id: selectedCharacterId
        })
      : "")
  const rolePlayChipTitle = rolePlayActive
    ? t(
        "sidepanel:controlRow.characterChatChip",
        "Character Chat: {{kind}} {{name}}",
        {
          kind: rolePlayKindLabel,
          name: rolePlayName
        }
      )
    : ""
  const fullAppChatPath = React.useMemo(
    () =>
      buildSidepanelFullAppChatPath({
        selectedAssistant,
        selectedCharacterId
      }),
    [selectedAssistant, selectedCharacterId]
  )
  const fullAppButtonLabel = rolePlayActive
    ? t(
        "sidepanel:controlRow.openCharacterChatInFullUI",
        "Open Character Chat in full app"
      )
    : t("sidepanel:controlRow.openInFullUI", "Open full app")
  const openRolePlayPicker = React.useCallback(() => {
    if (typeof window === "undefined") return
    window.dispatchEvent(
      new CustomEvent("tldw:open-sidepanel-assistant-select", {
        detail: {
          tab: selectedAssistant?.kind === "persona" ? "persona" : "character"
        }
      })
    )
  }, [selectedAssistant?.kind])
  const clearRolePlaySelection = React.useCallback(() => {
    setSelectedCharacterId(null)
    void setSelectedAssistant(null)
  }, [setSelectedAssistant, setSelectedCharacterId])

  const openQuickIngest = async () => {
    const detail =
      playlistImportEnabled
        ? (await resolveActivePlaylistDetail()) ?? activePlaylistDetail
        : null
    requestQuickIngestOpen(detail ?? undefined)
    setMoreOpen(false)
    requestAnimationFrame(() => moreBtnRef.current?.focus())
  }

  const openFullApp = () => {
    const path = `/options.html#${fullAppChatPath}`
    try {
      const runtime = browser.runtime
      const url = runtime?.id && runtime.getURL ? runtime.getURL(path) : null
      if (url) {
        if (browser.tabs?.create) {
          void browser.tabs.create({ url })
        } else {
          window.open(url, "_blank")
        }
        setMoreOpen(false)
        requestAnimationFrame(() => moreBtnRef.current?.focus())
        return
      }
    } catch {}
    window.open(fullAppChatPath, "_blank")
    setMoreOpen(false)
    requestAnimationFrame(() => moreBtnRef.current?.focus())
  }

  const moreMenuContent = (
    <div
      className="flex flex-col gap-2 p-2 min-w-[200px]"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault()
          setMoreOpen(false)
          requestAnimationFrame(() => moreBtnRef.current?.focus())
        }
      }}
    >
      {/* Search & Vision Section */}
      <div className="text-caption text-text-muted font-medium">
        {t("sidepanel:controlRow.searchSection", "Search & Vision")}
      </div>

      {/* Web Search - only show if server supports it */}
      {capabilities?.hasWebSearch && (
        <div className="flex items-center justify-between gap-2">
          <span className="text-sm flex items-center gap-2">
            <Globe className="size-3.5" />
            {t("sidepanel:controlRow.webSearch", "Web Search")}
          </span>
          <Switch
            size="small"
            checked={webSearch}
            onChange={(checked) => setWebSearch(checked)}
          />
        </div>
      )}

      {/* Vision */}
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm flex items-center gap-2">
          <Eye className="size-3.5" />
          {t("sidepanel:controlRow.vision", "Vision")}
        </span>
        {/* L12: Always show tooltip for disabled controls for better accessibility */}
        <Tooltip
          title={
            chatMode === "rag"
              ? t("sidepanel:controlRow.visionDisabledKnowledge", "Disable Knowledge Search to use Vision")
              : t("sidepanel:controlRow.visionTooltip", "Enable Vision to analyze images")
          }
          open={chatMode === "rag" ? undefined : false}
        >
          <Switch
            size="small"
            checked={chatMode === "vision"}
            disabled={chatMode === "rag"}
            onChange={(checked) => setChatMode(checked ? "vision" : "normal")}
          />
        </Tooltip>
      </div>

      <div className="panel-divider my-1" />

      <div className="text-caption text-text-muted font-medium">
        {t("sidepanel:controlRow.toolChoiceLabel", "Tool choice")}
      </div>
      <Tooltip
        title={
          !hasMcp
            ? t("sidepanel:controlRow.mcpToolsUnavailable", "MCP tools unavailable")
            : mcpHealthState === "unhealthy"
              ? t("sidepanel:controlRow.mcpToolsUnhealthy", "MCP tools are offline")
              : mcpToolsLoading
                ? t("sidepanel:controlRow.mcpToolsLoading", "Loading tools...")
                : chatMcpToolCount === 0
                  ? t("sidepanel:controlRow.mcpToolsEmpty", "No MCP tools available")
                  : ""
        }
        open={
          !hasMcp ||
          mcpHealthState === "unhealthy" ||
          mcpToolsLoading ||
          chatMcpToolCount === 0
            ? undefined
            : false
        }
      >
        <Radio.Group
          size="small"
          value={toolChoice}
          onChange={(e) => setToolChoice(e.target.value as ToolChoice)}
          className="flex flex-wrap gap-2"
          aria-label={t("sidepanel:controlRow.toolChoiceLabel", "Tool choice")}
          disabled={
            !hasMcp ||
            mcpHealthState === "unhealthy" ||
            mcpToolsLoading ||
            chatMcpToolCount === 0
          }
        >
          <Radio.Button value="auto">
            {t("sidepanel:controlRow.toolChoiceAuto", "Auto")}
          </Radio.Button>
          <Radio.Button value="required">
            {t("sidepanel:controlRow.toolChoiceRequired", "Required")}
          </Radio.Button>
          <Radio.Button value="none">
            {t("sidepanel:controlRow.toolChoiceNone", "None")}
          </Radio.Button>
        </Radio.Group>
      </Tooltip>
      <div className="text-[11px] text-text-muted">
        {t("sidepanel:controlRow.toolRunStatus", "Tool run")}: {toolRunStatusLabel}
      </div>
      <div className="text-caption text-text-muted font-medium">
        {t("sidepanel:controlRow.mcpToolsLabel", "MCP tools")}
      </div>
      <McpToolSelector
        discoveredTools={discoveredMcpTools}
        toolCounts={mcpToolCounts}
        toolsLoading={mcpToolsLoading}
        hasMcp={hasMcp}
        healthState={mcpHealthState}
        onToolEnabledChange={setMcpToolEnabled}
        onReset={resetMcpToolFilter}
        compact
        t={translateMcpToolSelector}
      />

      <div className="panel-divider my-1" />
      <div className="text-caption text-text-muted font-medium">
        {t("sidepanel:controlRow.mcpToolsFiltersLabel", "Tool filters")}
      </div>
      {isSmallModel && hasMcp && (
        <div className="rounded-md border border-border bg-surface2/60 px-2 py-1 text-[11px] text-text-muted">
          {t(
            "sidepanel:controlRow.mcpSmallModelHint",
            "Small/fast model: use catalog/module filters or the discovery tools (mcp.catalogs.list → mcp.modules.list → mcp.tools.list) to keep tool context light."
          )}
        </div>
      )}
      <div className="flex flex-col gap-2">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">
            {t("sidepanel:controlRow.mcpCatalogLabel", "Catalog")}
          </label>
          <Select
            size="small"
            allowClear
            showSearch
            loading={mcpCatalogsLoading}
            value={toolCatalogId ?? undefined}
            placeholder={t("sidepanel:controlRow.mcpCatalogSelectPlaceholder", "Select a catalog")}
            onChange={(value) => handleCatalogSelect(value as number | undefined)}
            optionFilterProp="label"
            className="w-full"
          >
            {catalogGroups.team.length > 0 && (
              <Select.OptGroup label={t("sidepanel:controlRow.mcpCatalogTeam", "Team catalogs")}>
                {catalogGroups.team.map((catalog) => (
                  <Select.Option
                    key={`team-${catalog.id}`}
                    value={catalog.id}
                    label={catalog.name}
                  >
                    <div className="flex flex-col">
                      <span className="text-sm">{catalog.name}</span>
                      <span className="text-[11px] text-text-muted">ID {catalog.id}</span>
                    </div>
                  </Select.Option>
                ))}
              </Select.OptGroup>
            )}
            {catalogGroups.org.length > 0 && (
              <Select.OptGroup label={t("sidepanel:controlRow.mcpCatalogOrg", "Org catalogs")}>
                {catalogGroups.org.map((catalog) => (
                  <Select.Option
                    key={`org-${catalog.id}`}
                    value={catalog.id}
                    label={catalog.name}
                  >
                    <div className="flex flex-col">
                      <span className="text-sm">{catalog.name}</span>
                      <span className="text-[11px] text-text-muted">ID {catalog.id}</span>
                    </div>
                  </Select.Option>
                ))}
              </Select.OptGroup>
            )}
            {catalogGroups.global.length > 0 && (
              <Select.OptGroup label={t("sidepanel:controlRow.mcpCatalogGlobal", "Global catalogs")}>
                {catalogGroups.global.map((catalog) => (
                  <Select.Option
                    key={`global-${catalog.id}`}
                    value={catalog.id}
                    label={catalog.name}
                  >
                    <div className="flex flex-col">
                      <span className="text-sm">{catalog.name}</span>
                      <span className="text-[11px] text-text-muted">ID {catalog.id}</span>
                    </div>
                  </Select.Option>
                ))}
              </Select.OptGroup>
            )}
          </Select>
          <Input
            size="small"
            placeholder={t("sidepanel:controlRow.mcpCatalogPlaceholder", "catalog name")}
            value={catalogDraft}
            onChange={(e) => setCatalogDraft(e.target.value)}
            onBlur={commitCatalog}
            onPressEnter={commitCatalog}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">
            {t("sidepanel:controlRow.mcpCatalogIdLabel", "Catalog ID")}
          </label>
          <InputNumber
            size="small"
            min={0}
            value={toolCatalogId ?? undefined}
            onChange={(value) =>
              setToolCatalogId(typeof value === "number" && Number.isFinite(value) ? value : null)
            }
            placeholder={t("sidepanel:controlRow.mcpCatalogIdPlaceholder", "optional")}
            className="w-full"
          />
        </div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-text-muted">
            {t("sidepanel:controlRow.mcpCatalogStrictLabel", "Strict catalog filter")}
          </span>
          <Switch
            size="small"
            checked={toolCatalogStrict}
            onChange={(checked) => setToolCatalogStrict(checked)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs text-text-muted">
            {t("sidepanel:controlRow.mcpModuleLabel", "Module")}
          </label>
          <Select
            size="small"
            allowClear
            showSearch
            mode="multiple"
            loading={moduleOptionsLoading}
            disabled={moduleOptionsLoading || moduleOptions.length === 0}
            value={toolModules.length > 0 ? toolModules : undefined}
            placeholder={t("sidepanel:controlRow.mcpModuleSelectPlaceholder", "Select modules")}
            onChange={(value) => handleModuleSelect(value as string[] | undefined)}
            optionFilterProp="label"
            className="w-full"
          >
            {moduleOptions.map((moduleId) => (
              <Select.Option key={moduleId} value={moduleId} label={moduleId}>
                <span className="text-sm">{moduleId}</span>
              </Select.Option>
            ))}
          </Select>
        </div>
      </div>

      <div className="panel-divider my-1" />

      {/* Upload Image */}
      <Upload
        accept="image/*"
        showUploadList={false}
        beforeUpload={(file) => {
          onImageUpload(file)
          setMoreOpen(false)
          return false
        }}
      >
        <button
          data-testid="chat-upload-image"
          className="w-full text-left text-sm px-3 py-2 rounded flex items-center gap-2 hover:bg-surface2"
          title={t("sidepanel:controlRow.uploadImage", "Upload Image")}
        >
          <ImageIcon className="size-4 text-text-subtle" />
          {t("sidepanel:controlRow.uploadImage", "Upload Image")}
        </button>
      </Upload>

      <div className="panel-divider my-1" />

      <button
        type="button"
        onClick={() => setAdvancedToolsExpanded((open) => !open)}
        className="flex items-center justify-between px-2 py-1 text-[10px] font-semibold uppercase text-text-muted tracking-wider hover:text-text transition"
      >
        <span>{t("sidepanel:controlRow.advanced", "Advanced")}</span>
        <ChevronRight
          className={`h-3 w-3 transition-transform ${
            advancedToolsExpanded ? "rotate-90" : ""
          }`}
        />
      </button>
      {advancedToolsExpanded && (
        <div className="flex flex-col gap-1.5 px-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm text-text">
              {t(
                "sidepanel:controlRow.allowExternalImages",
                "Load external images in chat"
              )}
            </span>
            <Switch
              size="small"
              checked={allowExternalImages}
              onChange={(checked) => setAllowExternalImages(checked)}
            />
          </div>
          <p className="text-[11px] text-text-muted">
            {t(
              "sidepanel:controlRow.allowExternalImagesHelp",
              "When off, external image URLs are blocked and shown as links."
            )}
          </p>

          <div className="panel-divider my-1" />

          <div className="flex items-center justify-between gap-2">
            <span className="text-sm text-text">
              {t(
                "sidepanel:controlRow.showMoodBadge",
                "Show mood badge in chat"
              )}
            </span>
            <Switch
              size="small"
              checked={showMoodBadge}
              onChange={(checked) => setShowMoodBadge(checked)}
            />
          </div>
          <p className="text-[11px] text-text-muted">
            {t(
              "sidepanel:controlRow.showMoodBadgeHelp",
              "Shows labels like \"Mood: neutral\" on assistant messages."
            )}
          </p>

          <div className="flex items-center justify-between gap-2">
            <span className="text-sm text-text">
              {t(
                "sidepanel:controlRow.showMoodConfidence",
                "Show mood confidence (%)"
              )}
            </span>
            <Switch
              size="small"
              checked={showMoodConfidence}
              disabled={!showMoodBadge}
              onChange={(checked) => setShowMoodConfidence(checked)}
            />
          </div>
          <p className="text-[11px] text-text-muted">
            {t(
              "sidepanel:controlRow.showMoodConfidenceHelp",
              "Adds confidence percentage when available."
            )}
          </p>
        </div>
      )}

      <div className="panel-divider my-1" />

      <div className="text-caption text-text-muted font-medium">
        {t("sidepanel:controlRow.quickActions", "Quick actions")}
      </div>
      <button
        type="button"
        onClick={() => {
          void openQuickIngest()
        }}
        data-testid="chat-quick-ingest"
        className="w-full text-left text-sm px-3 py-2 rounded flex items-center gap-2 hover:bg-surface2"
        title={quickIngestLabel}
      >
        <UploadCloud className="size-4 text-text-subtle" />
        {quickIngestLabel}
      </button>
      <button
        type="button"
        onClick={openFullApp}
        data-testid="chat-open-full-app"
        className="w-full text-left text-sm px-3 py-2 rounded flex items-center gap-2 hover:bg-surface2"
        title={fullAppButtonLabel}
      >
        <ExternalLink className="size-4 text-text-subtle" />
        {fullAppButtonLabel}
      </button>

    </div>
  )

  return (
    <div data-testid="control-row" className="flex items-center gap-2 flex-wrap">
        {rolePlayActive && (
          <div
            data-testid="sidepanel-character-chat-chip"
            className="flex min-h-[32px] max-w-full items-center overflow-hidden rounded-md border border-primary/30 bg-primary/10 text-xs text-text"
            title={rolePlayChipTitle}
          >
            <button
              type="button"
              data-testid="sidepanel-character-chat-switch"
              onClick={openRolePlayPicker}
              className="flex min-w-0 items-center gap-1.5 px-2 py-1.5 text-left hover:bg-primary/15 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              aria-label={t(
                "sidepanel:controlRow.switchCharacterChat",
                "Switch Character Chat assistant"
              )}
            >
              <MessageSquare className="size-3.5 shrink-0 text-primary" />
              <span className="shrink-0 font-semibold">
                {t("sidepanel:controlRow.characterChat", "Character Chat")}
              </span>
              <span className="min-w-0 truncate text-text-muted">
                {rolePlayKindLabel}: {rolePlayName}
              </span>
            </button>
            <button
              type="button"
              data-testid="sidepanel-character-chat-clear"
              onClick={clearRolePlaySelection}
              className="flex h-8 w-8 shrink-0 items-center justify-center border-l border-primary/20 text-text-muted hover:bg-primary/15 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              aria-label={t(
                "sidepanel:controlRow.clearCharacterChat",
                "Clear Character Chat selection"
              )}
              title={t(
                "sidepanel:controlRow.clearCharacterChat",
                "Clear Character Chat selection"
              )}
            >
              <X className="size-3.5" />
            </button>
          </div>
        )}
        {/* Prompt, Model & Character selectors */}
        <PromptSelect
          selectedSystemPrompt={selectedSystemPrompt}
          systemPrompt={systemPromptOverride}
          setSelectedSystemPrompt={setSelectedSystemPrompt}
          setSystemPrompt={setSystemPromptOverride}
          setSelectedQuickPrompt={setSelectedQuickPrompt}
          iconClassName="size-4"
          className="px-2 text-text-muted hover:text-text"
        />
        <ModelSelect iconClassName="size-4" showSelectedName />
        <ConversationContextPopover
          chatId={serverChatId}
          selectedCharacterId={selectedCharacterId}
          setSelectedCharacterId={setSelectedCharacterId}
          composition={conversationContextComposition}
          compositionStatus={conversationContextStatus}
          saveSelection={conversationContextSaveSelection}
          iconClassName="size-4"
          className="px-2 text-text-muted hover:text-text"
        />

        {/* Divider */}
        <div className="h-4 w-px bg-border mx-1" />

        {/* Knowledge Search - opens panel */}
        <div className="relative">
          <button
            type="button"
            data-testid="control-rag-toggle"
            onClick={onToggleRag}
            className="flex items-center gap-2 px-3 py-2 sm:px-2 sm:py-1 rounded text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-focus transition-colors min-h-[44px] sm:min-h-0 text-text-muted hover:bg-surface2 hover:text-text"
            aria-label={t("sidepanel:controlRow.knowledgeSearch", "Open knowledge search")}
            title={t("sidepanel:controlRow.knowledgeSearch", "Open knowledge search")}
          >
            <Search className="size-3.5" />
            <span className="hidden sm:inline">{t("sidepanel:controlRow.knowledge", "Knowledge search")}</span>
          </button>
          {!knowledgeHintSeen && isConnected && (
            <FeatureHint
              featureKey="knowledge-search"
              title={t("common:featureHints.knowledge.title", "Search your knowledge")}
              description={t("common:featureHints.knowledge.description", "Open the knowledge search panel to find snippets and insert them into your chat.")}
              position="top"
            />
          )}
        </div>

        {/* Web Search Toggle - with label and shortcut (if available) */}
        {capabilities?.hasWebSearch && (
          <button
            type="button"
            data-testid="control-web-toggle"
            onClick={() => setWebSearch(!webSearch)}
            className={`flex items-center gap-2 px-3 py-2 sm:px-2 sm:py-1 rounded text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-focus transition-colors min-h-[44px] sm:min-h-0 ${
              webSearch
                ? "bg-primary/10 text-primary hover:bg-primary/20"
                : "text-text-muted hover:bg-surface2 hover:text-text"
            }`}
            aria-label={t("sidepanel:controlRow.webSearch", "Web Search")}
            aria-pressed={webSearch}
            title={t("sidepanel:controlRow.webSearch", "Web Search")}
          >
            <Globe className="size-3.5" />
            <span className="hidden sm:inline">{t("sidepanel:controlRow.web", "Web")}</span>
          </button>
        )}

        <Upload
          accept="image/*"
          showUploadList={false}
          beforeUpload={(file) => {
            onImageUpload(file)
            return false
          }}
        >
          <button
            type="button"
            data-testid="chat-attach-image"
            className="flex items-center gap-2 px-3 py-2 sm:px-2 sm:py-1 rounded text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-focus transition-colors min-h-[44px] sm:min-h-0 text-text-muted hover:bg-surface2 hover:text-text"
            aria-label={t("sidepanel:controlRow.attachImage", "Attach image")}
            title={t("sidepanel:controlRow.attachImage", "Attach image")}
          >
            <ImageIcon className="size-3.5" />
            <span className="hidden sm:inline">
              {t("sidepanel:controlRow.attach", "Attach")}
            </span>
          </button>
        </Upload>

        {/* More Tools Menu */}
        <div className="relative">
          <Popover
            trigger="click"
            open={moreOpen}
            onOpenChange={(visible) => {
              setMoreOpen(visible)
              if (!visible) {
                requestAnimationFrame(() => moreBtnRef.current?.focus())
              }
            }}
            content={moreMenuContent}
            placement="topRight"
          >
            <button
              ref={moreBtnRef}
              type="button"
              data-testid="control-more-menu"
              className="p-2 min-h-[44px] sm:min-h-0 rounded hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              aria-label={t("sidepanel:controlRow.moreTools", "More tools")}
              aria-haspopup="menu"
              aria-expanded={moreOpen}
              title={t("sidepanel:controlRow.moreTools", "More tools")}
            >
              <MoreHorizontal className="size-4 text-text-subtle" />
            </button>
          </Popover>
          {!moreToolsHintSeen && (
            <FeatureHint
              featureKey="more-tools"
              title={t("common:featureHints.moreTools.title", "More tools available")}
              description={t("common:featureHints.moreTools.description", "Access vision mode, image upload, quick ingest, and the full app.")}
              position="top"
            />
          )}
        </div>
      </div>
    )
}

export const ControlRow = React.memo(ControlRowBase)
ControlRow.displayName = "ControlRow"
