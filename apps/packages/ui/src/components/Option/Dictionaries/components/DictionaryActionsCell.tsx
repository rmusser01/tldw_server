import React from "react"
import { Button, Dropdown, Tooltip } from "antd"
import {
  BarChart3,
  Copy,
  Download,
  FileText,
  History,
  Link2,
  List,
  MoreHorizontal,
  Pen,
  Trash2
} from "lucide-react"

type DictionaryActionsCellProps = {
  record: any
  useCompactDictionaryActions: boolean
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

export const DictionaryActionsCell: React.FC<DictionaryActionsCellProps> = ({
  record,
  useCompactDictionaryActions,
  onOpenEdit,
  onOpenEntries,
  onOpenQuickAssign,
  onExportJson,
  onExportMarkdown,
  onOpenStats,
  onOpenVersions,
  onDuplicate,
  onDelete
}) => {
  const overflowButtonRef = React.useRef<HTMLButtonElement | null>(null)

  const focusOverflowTrigger = React.useCallback(() => {
    overflowButtonRef.current?.focus()
  }, [])

  return (
    <div className="flex gap-2 items-center">
      <Tooltip title="Edit dictionary">
        <Button
          type="text"
          size="small"
          icon={<Pen className="w-4 h-4" />}
          onClick={() => onOpenEdit(record)}
          aria-label={`Edit dictionary ${record.name}`}
        />
      </Tooltip>
      <Tooltip title="Manage entries">
        <Button
          type="text"
          size="small"
          icon={<List className="w-4 h-4" />}
          onClick={() => onOpenEntries(record.id)}
          aria-label={`Manage entries for ${record.name}`}
        />
      </Tooltip>
      {useCompactDictionaryActions ? (
        <Dropdown
          trigger={["click"]}
          menu={{
            items: [
              {
                key: "assign",
                label: "Quick assign to chats",
                icon: <Link2 className="w-4 h-4" />
              },
              { key: "json", label: "Export JSON" },
              { key: "markdown", label: "Export Markdown" },
              { key: "stats", label: "View statistics" },
              {
                key: "versions",
                label: "Version history",
                icon: <History className="w-4 h-4" />
              },
              {
                key: "duplicate",
                label: "Duplicate dictionary",
                icon: <Copy className="w-4 h-4" />
              },
              {
                key: "delete",
                danger: true,
                label: "Delete dictionary",
                icon: <Trash2 className="w-4 h-4" />
              }
            ],
            onClick: ({ key }) => {
              switch (String(key)) {
                case "assign":
                  focusOverflowTrigger()
                  onOpenQuickAssign(record)
                  return
                case "json":
                  onExportJson(record)
                  return
                case "markdown":
                  onExportMarkdown(record)
                  return
                case "stats":
                  focusOverflowTrigger()
                  onOpenStats(record)
                  return
                case "versions":
                  onOpenVersions(record)
                  return
                case "duplicate":
                  onDuplicate(record)
                  return
                case "delete":
                  onDelete(record)
                  return
                default:
                  return
              }
            }
          }}
          placement="bottomRight"
        >
          <Button
            ref={overflowButtonRef}
            type="text"
            size="small"
            icon={<MoreHorizontal className="w-4 h-4" />}
            aria-label={`More actions for ${record.name}`}
            aria-haspopup="menu"
          />
        </Dropdown>
      ) : (
        <>
          <Tooltip title="Quick assign to chat sessions">
            <Button
              type="text"
              size="small"
              icon={<Link2 className="w-4 h-4" />}
              onClick={() => onOpenQuickAssign(record)}
              aria-label={`Quick assign ${record.name} to chats`}
            />
          </Tooltip>
          <Tooltip title="Export as JSON">
            <Button
              type="text"
              size="small"
              icon={<Download className="w-4 h-4" />}
              onClick={() => onExportJson(record)}
              aria-label={`Export ${record.name} as JSON`}
            />
          </Tooltip>
          <Tooltip title="Export as Markdown">
            <Button
              type="text"
              size="small"
              icon={<FileText className="w-4 h-4" />}
              onClick={() => onExportMarkdown(record)}
              aria-label={`Export ${record.name} as Markdown`}
            />
          </Tooltip>
          <Tooltip title="View statistics">
            <Button
              type="text"
              size="small"
              icon={<BarChart3 className="w-4 h-4" />}
              onClick={() => onOpenStats(record)}
              aria-label={`View statistics for ${record.name}`}
            />
          </Tooltip>
          <Tooltip title="View version history">
            <Button
              type="text"
              size="small"
              icon={<History className="w-4 h-4" />}
              onClick={() => onOpenVersions(record)}
              aria-label={`Version history for ${record.name}`}
            />
          </Tooltip>
          <Tooltip title="Duplicate dictionary">
            <Button
              type="text"
              size="small"
              icon={<Copy className="w-4 h-4" />}
              onClick={() => onDuplicate(record)}
              aria-label={`Duplicate dictionary ${record.name}`}
            />
          </Tooltip>
          <Tooltip title="Delete dictionary">
            <Button
              type="text"
              size="small"
              danger
              icon={<Trash2 className="w-4 h-4" />}
              onClick={() => onDelete(record)}
              aria-label={`Delete dictionary ${record.name}`}
            />
          </Tooltip>
        </>
      )}
    </div>
  )
}
