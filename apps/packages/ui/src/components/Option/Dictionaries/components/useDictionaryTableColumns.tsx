import React from "react"
import { Switch, Tooltip } from "antd"
import { Book } from "lucide-react"
import { DictionaryActionsCell } from "./DictionaryActionsCell"
import { DictionaryValidationStatusCell } from "./DictionaryValidationStatusCell"
import {
  compareDictionaryActive,
  compareDictionaryEntryCount,
  compareDictionaryName,
  formatDictionaryChatReferenceTitle,
  formatDictionaryUsageLabel,
  formatRelativeTimestamp,
  normalizeDictionaryChatState,
  resolveDictionaryChatReferenceId,
} from "../listUtils"

type DictionaryValidationStatus = {
  status: "valid" | "warning" | "error" | "loading" | "unknown"
  message?: string
}

type UseDictionaryTableColumnsParams = {
  activeUpdateMap: Record<number, boolean>
  validationStatus: Record<number, DictionaryValidationStatus>
  useCompactDictionaryActions: boolean
  onToggleActive: (record: any, checked: boolean) => Promise<void> | void
  onValidateDictionary: (dictionaryId: number) => void
  onOpenChatContext: (chatRef: any) => void
  onOpenEdit: (record: any) => void
  onOpenEntries: (dictionaryId: number) => void
  onOpenQuickAssign: (record: any) => void
  onExportJson: (record: any) => void
  onExportMarkdown: (record: any) => void
  onOpenStats: (record: any) => void
  onOpenVersions: (record: any) => void
  onDuplicate: (record: any) => void
  onDelete: (record: any) => void
}

export function useDictionaryTableColumns({
  activeUpdateMap,
  validationStatus,
  useCompactDictionaryActions,
  onToggleActive,
  onValidateDictionary,
  onOpenChatContext,
  onOpenEdit,
  onOpenEntries,
  onOpenQuickAssign,
  onExportJson,
  onExportMarkdown,
  onOpenStats,
  onOpenVersions,
  onDuplicate,
  onDelete
}: UseDictionaryTableColumnsParams): any[] {
  return React.useMemo(
    () => [
      {
        title: "",
        key: "icon",
        width: 48,
        render: () => <Book className="w-5 h-5 text-text-muted" aria-hidden="true" />
      },
      {
        title: "Name",
        dataIndex: "name",
        key: "name",
        sorter: (a: any, b: any) => compareDictionaryName(a, b)
      },
      {
        title: "Description",
        dataIndex: "description",
        key: "description",
        render: (value: string) => <span className="line-clamp-1">{value}</span>
      },
      {
        title: "Active",
        dataIndex: "is_active",
        key: "is_active",
        sorter: (a: any, b: any) => compareDictionaryActive(a, b),
        filters: [
          { text: "Active", value: true },
          { text: "Inactive", value: false }
        ],
        onFilter: (value: any, record: any) => {
          const activeFilter = value === true || value === "true"
          return Boolean(record.is_active) === activeFilter
        },
        render: (value: boolean, record: any) => (
          <Switch
            checked={Boolean(value)}
            loading={Boolean(activeUpdateMap[record.id])}
            checkedChildren="On"
            unCheckedChildren="Off"
            onChange={(checked) => {
              void onToggleActive(record, checked)
            }}
            aria-label={`Set dictionary ${record.name} ${value ? "inactive" : "active"}`}
          />
        )
      },
      {
        title: "Priority",
        dataIndex: "processing_priority",
        key: "processing_priority",
        sorter: (a: any, b: any) => {
          const toPriority = (record: any) => {
            const isActive = Boolean(record?.is_active)
            if (!isActive) return Number.POSITIVE_INFINITY
            const priority = Number(record?.processing_priority)
            return Number.isFinite(priority) && priority > 0
              ? priority
              : Number.POSITIVE_INFINITY - 1
          }
          return toPriority(a) - toPriority(b)
        },
        render: (_value: number | null | undefined, record: any) => {
          if (!record?.is_active) {
            return <span className="text-xs text-text-muted">inactive</span>
          }
          const priority = Number(record?.processing_priority)
          if (!Number.isFinite(priority) || priority <= 0) {
            return <span className="text-xs text-text-muted">pending</span>
          }
          return (
            <Tooltip title="Processing order when multiple dictionaries are active">
              <span className="text-xs font-mono">{`P${priority}`}</span>
            </Tooltip>
          )
        }
      },
      {
        title: "Entries",
        dataIndex: "entry_count",
        key: "entry_count",
        sorter: (a: any, b: any) => compareDictionaryEntryCount(a, b),
        render: (_value: number, record: any) => {
          const entryCount = Number(record?.entry_count || 0)
          const regexCount = Number(record?.regex_entry_count ?? record?.regex_entries ?? 0)
          if (regexCount > 0) {
            return `${entryCount} entries (${regexCount} regex)`
          }
          return `${entryCount} entries`
        }
      },
      {
        title: "Used by",
        dataIndex: "used_by_chat_count",
        key: "used_by_chat_count",
        sorter: (a: any, b: any) =>
          Number(a?.used_by_chat_count || 0) - Number(b?.used_by_chat_count || 0),
        render: (_value: number, record: any) => {
          const totalChats = Number(record?.used_by_chat_count || 0)
          const chatRefs = Array.isArray(record?.used_by_chat_refs) ? record.used_by_chat_refs : []

          if (totalChats <= 0) {
            return <span className="text-xs text-text-muted">—</span>
          }

          const label = formatDictionaryUsageLabel(record)
          if (chatRefs.length === 0) {
            return <span className="text-xs">{label}</span>
          }

          const firstChat = chatRefs[0]
          return (
            <div className="space-y-1">
              <button
                type="button"
                className="text-xs underline decoration-dotted cursor-pointer"
                onClick={() => onOpenChatContext(firstChat)}
                aria-label={`Open most recent linked chat for ${record?.name || "dictionary"}`}
              >
                {label}
              </button>
              <Tooltip
                title={
                  <div className="space-y-1">
                    {chatRefs.map((chat: any) => {
                      const chatId = resolveDictionaryChatReferenceId(chat)
                      const title = formatDictionaryChatReferenceTitle(chat)
                      const state = normalizeDictionaryChatState(chat?.state)
                      return (
                        <button
                          key={chatId || `${title}-${state}`}
                          type="button"
                          className="block text-left text-xs hover:underline"
                          onClick={(event) => {
                            event.preventDefault()
                            event.stopPropagation()
                            onOpenChatContext(chat)
                          }}
                          aria-label={`Open chat ${title} from dictionary usage`}
                        >
                          {title} <span className="text-text-muted">({state})</span>
                        </button>
                      )
                    })}
                  </div>
                }
              >
                <span className="text-[11px] text-text-muted underline decoration-dotted cursor-help">
                  View linked chats
                </span>
              </Tooltip>
            </div>
          )
        }
      },
      {
        title: "Updated",
        dataIndex: "updated_at",
        key: "updated_at",
        sorter: (a: any, b: any) => {
          const valueA = new Date(a?.updated_at || 0).getTime()
          const valueB = new Date(b?.updated_at || 0).getTime()
          return valueA - valueB
        },
        render: (value: string | null | undefined) => {
          const relative = formatRelativeTimestamp(value)
          const absolute = value ? new Date(value).toLocaleString() : "No updates yet"
          return (
            <Tooltip title={absolute}>
              <span className="text-xs text-text-muted">{relative}</span>
            </Tooltip>
          )
        }
      },
      {
        title: "Status",
        key: "validation_status",
        width: 132,
        render: (_: any, record: any) => (
          <DictionaryValidationStatusCell
            record={record}
            status={validationStatus[record.id]}
            onValidate={onValidateDictionary}
          />
        )
      },
      {
        title: "Actions",
        key: "actions",
        render: (_: any, record: any) => (
          <DictionaryActionsCell
            record={record}
            useCompactDictionaryActions
            onOpenEdit={onOpenEdit}
            onOpenEntries={onOpenEntries}
            onOpenQuickAssign={onOpenQuickAssign}
            onExportJson={onExportJson}
            onExportMarkdown={onExportMarkdown}
            onOpenStats={onOpenStats}
            onOpenVersions={onOpenVersions}
            onDuplicate={onDuplicate}
            onDelete={onDelete}
          />
        )
      }
    ],
    [
      activeUpdateMap,
      onDelete,
      onDuplicate,
      onExportJson,
      onExportMarkdown,
      onOpenChatContext,
      onOpenEdit,
      onOpenEntries,
      onOpenQuickAssign,
      onOpenStats,
      onOpenVersions,
      onToggleActive,
      onValidateDictionary,
      useCompactDictionaryActions,
      validationStatus
    ]
  )
}
