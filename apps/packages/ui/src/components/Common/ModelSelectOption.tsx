import { useQuery } from "@tanstack/react-query"
import { Avatar, Dropdown, Tooltip } from "antd"
import { LucideBrain } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { fetchChatModels } from "@/services/tldw-server"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { ProviderIcons } from "./ProviderIcon"
import { IconButton } from "./IconButton"

type Props = {
  iconClassName?: string
}

export const ModelSelectOption: React.FC<Props> = ({ iconClassName = "size-5" }) => {
  const { t } = useTranslation("common")
  const { selectedModel, setSelectedModel } = useSelectedModel()
  const [menuDensity] = useStorage("menuDensity", "comfortable")
  const { data } = useQuery({
    queryKey: ["getAllModelsForSelectOption"],
    queryFn: async () => {
      const models = await fetchChatModels({ returnEmpty: true })
      return models
    }
  })

  const groupedItems = React.useMemo(() => {
    const groups = new Map<string, any[]>()
    const localProviders = new Set(["lmstudio", "llamafile", "ollama", "ollama2", "llamacpp", "vllm", "custom"]) // group as "custom"
    for (const d of data || []) {
      const providerRaw = (d.provider || "other").toLowerCase()
      const groupKey = providerRaw === 'chrome' ? 'default' : (localProviders.has(providerRaw) ? 'custom' : providerRaw)
      const labelNode = (
        <div className="w-52 gap-2 text-sm truncate inline-flex items-center leading-5">
          <div>
            {d.avatar ? (
              <Avatar src={d.avatar} alt={d.name} size="small" />
            ) : (
              <ProviderIcons provider={d?.provider} className="h-4 w-4 text-text-subtle" />
            )}
          </div>
          {d?.nickname || d.model}
        </div>
      )
      const item = {
        key: d.model,
        label: labelNode,
        onClick: () => {
          if (selectedModel === d.model) {
            setSelectedModel(null)
          } else {
            setSelectedModel(d.model)
          }
        }
      }
      if (!groups.has(groupKey)) groups.set(groupKey, [])
      groups.get(groupKey)!.push(item)
    }
    const items: any[] = []
    for (const [groupKey, children] of groups) {
      const labelText =
        groupKey === "default"
          ? t("modelSelect.group.default", "Default")
          : groupKey === "custom"
            ? t("modelSelect.group.custom", "Custom")
            : getProviderDisplayName(groupKey)
      const iconKey = groupKey === "default" ? "chrome" : groupKey
      items.push({
        type: 'group',
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
  }, [data, selectedModel, setSelectedModel])

  return (
    <>
      {data && data.length > 0 && (
        <Dropdown
          menu={{
            items: groupedItems,
            style: { maxHeight: 500, overflowY: "scroll" },
            className: `no-scrollbar ${menuDensity === 'compact' ? 'menu-density-compact' : 'menu-density-comfortable'}`,
            selectedKeys: [selectedModel]
          }}
          placement={"topLeft"}
          trigger={["click"]}>
          <Tooltip title={t("selectAModel")}>
            <IconButton
              ariaLabel={t("selectAModel") as string}
              hasPopup="menu"
              className="text-text-muted h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0"
            >
              <LucideBrain className={iconClassName} />
            </IconButton>
          </Tooltip>
        </Dropdown>
      )}
    </>
  )
}

export default ModelSelectOption
