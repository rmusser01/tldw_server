import { useQuery } from "@tanstack/react-query"
import { Avatar, Dropdown, Input, Select, Tooltip, type MenuProps } from "antd"
import { LucideBrain, Star } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { fetchChatModels } from "@/services/tldw-server"
import { useMessage } from "@/hooks/useMessage"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { ProviderIcons } from "./ProviderIcon"
import { IconButton } from "./IconButton"

export type Props = {
  iconClassName?: string
  showSelectedName?: boolean
}

export type ModelSelectHandle = {
  openAndFocus: () => void
}

type ChatModel = {
  model: string
  nickname?: string
  provider?: string
  details?: {
    capabilities?: string[]
  }
  avatar?: string
  name?: string
}

type ModelSortMode = "favorites" | "az" | "provider" | "localFirst"

const LOCAL_PROVIDERS = new Set([
  "lmstudio",
  "llamafile",
  "ollama",
  "ollama2",
  "llamacpp",
  "vllm",
  "custom",
  "local",
  "tldw",
  "chrome"
])

const ModelSelectBase = (
  { iconClassName = "size-5", showSelectedName = false }: Props,
  ref: React.ForwardedRef<ModelSelectHandle>
) => {
  const { t } = useTranslation("common")
  const { setSelectedModel, selectedModel } = useMessage()
  const selectedModelValue =
    typeof selectedModel === "string" ? selectedModel : null
  const [menuDensity] = useStorage("menuDensity", "comfortable")
  const [favoriteModels, setFavoriteModels] = useStorage<string[]>(
    "favoriteChatModels",
    []
  )
  const [sortMode, setSortMode] = useStorage<ModelSortMode>(
    "modelSelectSortMode",
    "provider"
  )
  const [searchQuery, setSearchQuery] = React.useState("")
  const [dropdownOpen, setDropdownOpen] = React.useState(false)
  const triggerRef = React.useRef<HTMLButtonElement | null>(null)
  React.useImperativeHandle(
    ref,
    () => ({
      openAndFocus: () => {
        setDropdownOpen(true)
        window.requestAnimationFrame(() => triggerRef.current?.focus())
      }
    }),
    []
  )
  const { data, isLoading } = useQuery<ChatModel[]>({
    queryKey: ["getAllModelsForSelect"],
    queryFn: () => fetchChatModels({ returnEmpty: true })
  })

  const favoriteSet = React.useMemo(
    () => new Set((favoriteModels || []).map((value) => String(value))),
    [favoriteModels]
  )

  const toggleFavorite = React.useCallback(
    (modelId: string) => {
      void setFavoriteModels((prev) => {
        const list = Array.isArray(prev) ? prev.map(String) : []
        const next = new Set(list)
        if (next.has(modelId)) {
          next.delete(modelId)
        } else {
          next.add(modelId)
        }
        return Array.from(next)
      })
    },
    [setFavoriteModels]
  )

  const filteredData = React.useMemo(() => {
    const list = Array.isArray(data) ? data : []
    const q = searchQuery.trim().toLowerCase()
    if (!q) return list
    return list.filter((model) => {
      const provider = String(model.provider || "").toLowerCase()
      const name = String(model.nickname || model.model || "").toLowerCase()
      const modelId = String(model.model || "").toLowerCase()
      const providerLabel = getProviderDisplayName(provider).toLowerCase()
      return (
        provider.includes(q) ||
        providerLabel.includes(q) ||
        name.includes(q) ||
        modelId.includes(q)
      )
    })
  }, [data, searchQuery])

  const menuItems = React.useMemo<NonNullable<MenuProps["items"]>>(() => {
    type MenuItem = NonNullable<MenuProps["items"]>[number]
    const normalizedData = filteredData || []

    const toProviderKey = (provider?: string) => {
      const normalized =
        typeof provider === "string" && provider.trim()
          ? provider.trim()
          : "other"
      return normalized.toLowerCase()
    }

    const toGroupKey = (providerRaw: string) =>
      providerRaw === "chrome"
        ? "default"
        : LOCAL_PROVIDERS.has(providerRaw)
          ? "custom"
          : providerRaw

    const byDisplayLabel = (a: ChatModel, b: ChatModel) => {
      const aProvider = getProviderDisplayName(toProviderKey(a.provider))
      const bProvider = getProviderDisplayName(toProviderKey(b.provider))
      const aLabel = `${aProvider} ${a.nickname || a.model}`.toLowerCase()
      const bLabel = `${bProvider} ${b.nickname || b.model}`.toLowerCase()
      return aLabel.localeCompare(bLabel)
    }

    const buildItem = (model: ChatModel): MenuItem => {
      const providerRaw = toProviderKey(model.provider)
      const providerLabel = getProviderDisplayName(providerRaw)
      const modelLabel = model.nickname || model.model
      const caps = Array.isArray(model.details?.capabilities)
        ? model.details.capabilities
        : []
      const hasVision = caps.includes("vision")
      const hasTools = caps.includes("tools")
      const hasFast = caps.includes("fast")
      const isFavorite = favoriteSet.has(model.model)
      const favoriteTitle = isFavorite
        ? t("modelSelect.favoriteRemove", "Remove from favorites")
        : t("modelSelect.favoriteAdd", "Add to favorites")

      return {
        key: model.model,
        label: (
          <div className="w-56 gap-2 text-sm inline-flex items-start leading-5">
            <div>
              {model.avatar ? (
                <Avatar src={model.avatar} alt={model.name} size="small" />
              ) : (
                <ProviderIcons
                  provider={providerRaw}
                  className="h-4 w-4 text-text-subtle"
                />
              )}
            </div>
            <div className="flex flex-col min-w-0 flex-1">
              <span className="truncate">
                {providerLabel} - {modelLabel}
              </span>
              {(hasVision || hasTools || hasFast) && (
                <div className="mt-0.5 flex flex-wrap gap-1 text-[10px]">
                  {hasVision && (
                    <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-primary">
                      {t("modelSelect.capability.vision", "Vision")}
                    </span>
                  )}
                  {hasTools && (
                    <span className="rounded-full bg-accent/10 px-1.5 py-0.5 text-accent">
                      {t("modelSelect.capability.tools", "Tools")}
                    </span>
                  )}
                  {hasFast && (
                    <span className="rounded-full bg-success/10 px-1.5 py-0.5 text-success">
                      {t("modelSelect.capability.fast", "Fast")}
                    </span>
                  )}
                </div>
              )}
            </div>
            <button
              type="button"
              className="mt-0.5 rounded p-0.5 text-text-subtle transition hover:bg-surface2"
              onMouseDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
              }}
              onClick={(event) => {
                event.preventDefault()
                event.stopPropagation()
                toggleFavorite(model.model)
              }}
              aria-label={favoriteTitle}
              title={favoriteTitle}
            >
              <Star
                className={`h-3.5 w-3.5 ${
                  isFavorite ? "fill-warn text-warn" : "text-text-subtle"
                }`}
              />
            </button>
          </div>
        ),
        onClick: () => {
          if (selectedModelValue === model.model) {
            setSelectedModel(null)
          } else {
            setSelectedModel(model.model)
          }
        }
      }
    }

    if (normalizedData.length === 0) {
      return [
        {
          key: "__empty__",
          disabled: true,
          label: (
            <div className="w-56 px-2 py-2 text-xs text-text-muted">
              {t("modelSelect.noMatches", "No models match your search.")}
            </div>
          )
        }
      ]
    }

    if (sortMode === "az") {
      return normalizedData.slice().sort(byDisplayLabel).map(buildItem)
    }

    if (sortMode === "favorites") {
      const favorites = normalizedData.filter((model) =>
        favoriteSet.has(model.model)
      )
      const others = normalizedData.filter(
        (model) => !favoriteSet.has(model.model)
      )
      const items: MenuItem[] = []
      if (favorites.length > 0) {
        items.push({
          type: "group",
          key: "favorites",
          label: t("modelSelect.sort.favorites", "Favorites"),
          children: favorites.slice().sort(byDisplayLabel).map(buildItem)
        })
      }
      items.push(...others.slice().sort(byDisplayLabel).map(buildItem))
      return items
    }

    const groups = new Map<string, MenuItem[]>()
    for (const model of normalizedData) {
      const providerRaw = toProviderKey(model.provider)
      const groupKey = toGroupKey(providerRaw)
      if (!groups.has(groupKey)) groups.set(groupKey, [])
      groups.get(groupKey)!.push(buildItem(model))
    }

    const groupEntries = Array.from(groups.entries())
    if (sortMode === "localFirst") {
      groupEntries.sort(([aKey], [bKey]) => {
        const aLocal = LOCAL_PROVIDERS.has(aKey) || aKey === "default"
        const bLocal = LOCAL_PROVIDERS.has(bKey) || bKey === "default"
        if (aLocal !== bLocal) return aLocal ? -1 : 1
        return aKey.localeCompare(bKey)
      })
    }

    const items: MenuItem[] = []
    for (const [groupKey, children] of groupEntries) {
      const labelText =
        groupKey === "default"
          ? t("modelSelect.group.default", "Default")
          : groupKey === "custom"
            ? t("modelSelect.group.custom", "Custom")
            : getProviderDisplayName(groupKey)
      const iconKey = groupKey === "default" ? "chrome" : groupKey
      items.push({
        type: "group",
        key: `group-${groupKey}`,
        label: (
          <div className="flex items-center gap-1.5 text-xs leading-4 font-medium uppercase tracking-wider text-text-subtle">
            <ProviderIcons provider={iconKey} className="h-3 w-3" />
            <span>{labelText}</span>
          </div>
        ),
        children
      })
    }
    return items
  }, [
    favoriteSet,
    filteredData,
    selectedModelValue,
    setSelectedModel,
    sortMode,
    t,
    toggleFavorite
  ])

  // Get display name for selected model
  const selectedModelDisplay = React.useMemo(() => {
    if (!selectedModelValue || !data) return null
    const model = data.find(m => m.model === selectedModelValue)
    if (!model) return selectedModelValue.split('/').pop() || selectedModelValue
    // Use nickname if available, otherwise extract short name from model ID
    const shortName = model.nickname || model.model.split('/').pop() || model.model
    // Truncate if too long
    return shortName.length > 20 ? shortName.substring(0, 18) + '…' : shortName
  }, [selectedModelValue, data])

  if (isLoading && !data) {
    return (
      <Tooltip title={t("modelSelect.loading", "Loading models...")}>
        <IconButton
          ariaLabel={t("selectAModel") as string}
          disabled
          dataTestId="chat-model-select"
          className="px-2 text-text-muted animate-pulse"
        >
          <LucideBrain className={iconClassName} />
          <span className="ml-1 hidden sm:inline text-xs">
            {t("modelSelect.loading", "Loading models...")}
          </span>
        </IconButton>
      </Tooltip>
    )
  }

  return (
    <>
      {data && data.length > 0 && (
        <Dropdown
          open={dropdownOpen}
          onOpenChange={(open) => {
            setDropdownOpen(open)
            if (!open) {
              setSearchQuery("")
            }
          }}
          menu={{
            items: menuItems,
            className: `no-scrollbar ${menuDensity === 'compact' ? 'menu-density-compact' : 'menu-density-comfortable'}`,
            selectedKeys: selectedModelValue ? [selectedModelValue] : [],
            selectable: true
          }}
          popupRender={(menu) => (
            <div className="w-72">
              <div className="flex items-center gap-2 px-2 py-2 border-b border-border">
                <Input
                  size="small"
                  placeholder={t("modelSelect.searchPlaceholder", "Search models")}
                  value={searchQuery}
                  allowClear
                  className="flex-1"
                  autoFocus
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => event.stopPropagation()}
                />
                <Select
                  size="small"
                  value={sortMode}
                  onChange={(value) => setSortMode(value as ModelSortMode)}
                  options={[
                    {
                      value: "favorites",
                      label: t("modelSelect.sort.favorites", "Favorites")
                    },
                    { value: "az", label: t("modelSelect.sort.az", "A-Z") },
                    {
                      value: "provider",
                      label: t("modelSelect.sort.provider", "Provider")
                    },
                    {
                      value: "localFirst",
                      label: t("modelSelect.sort.localFirst", "Local-first")
                    }
                  ]}
                  className="min-w-[120px]"
                  onKeyDown={(event) => event.stopPropagation()}
                />
              </div>
              <div className="max-h-[420px] overflow-y-auto no-scrollbar">{menu}</div>
            </div>
          )}
          placement={"topLeft"}
          trigger={["click"]}>
          <Tooltip
            title={
              selectedModelValue
                ? `${t("modelSelect.tooltip", "Changes model for next message")}: ${selectedModelValue}`
                : t("modelSelect.tooltip", "Changes model for next message")
            }>
            <IconButton
              ref={triggerRef}
              ariaLabel={t("selectAModel") as string}
              hasPopup="menu"
              dataTestId="chat-model-select"
              className="px-2 text-text-muted">
              <LucideBrain className={iconClassName} />
              {showSelectedName && selectedModelDisplay ? (
                <span className="ml-1.5 max-w-[120px] truncate text-xs font-medium text-text">
                  {selectedModelDisplay}
                </span>
              ) : (
                <span className="ml-1 hidden sm:inline text-xs">
                  {t("modelSelect.label", "Model")}
                </span>
              )}
            </IconButton>
          </Tooltip>
        </Dropdown>
      )}
    </>
  )
}

export const ModelSelect = React.forwardRef<ModelSelectHandle, Props>(
  ModelSelectBase
)
ModelSelect.displayName = "ModelSelect"
