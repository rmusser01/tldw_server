import React, { useRef } from "react"
import { useTranslation } from "react-i18next"
import { Plus, Trash2, ArrowUp, ArrowDown } from "lucide-react"

type StructuredPromptBlock = {
  id: string
  name: string
  role: "system" | "developer" | "user" | "assistant"
  content: string
  enabled: boolean
  order: number
  is_template: boolean
}

type BlockListPanelProps = {
  blocks: StructuredPromptBlock[]
  selectedBlockId: string | null
  onSelect: (blockId: string) => void
  onAddBlock: () => void
  onMoveBlock: (blockId: string, direction: "up" | "down") => void
  onRemoveBlock: (blockId: string) => void
  onReorderBlock?: (blockId: string, toIndex: number) => void
  showRole?: boolean
  description?: string
  addButtonRef?: React.Ref<HTMLButtonElement>
}

export const BlockListPanel: React.FC<BlockListPanelProps> = ({
  blocks,
  selectedBlockId,
  onSelect,
  onAddBlock,
  onMoveBlock,
  onRemoveBlock,
  onReorderBlock,
  showRole = true,
  description = "Ordered prompt sections assembled by the backend.",
  addButtonRef
}) => {
  const { t } = useTranslation("settings")
  const draggedBlockId = useRef<string | null>(null)

  return (
    <section className="min-w-0 rounded-xl border border-border bg-surface p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-text">Blocks</h3>
          <p className="text-xs text-text-muted">{description}</p>
        </div>
        <button
          type="button"
          ref={addButtonRef}
          onClick={onAddBlock}
          data-testid="structured-block-add"
          className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs font-medium text-text hover:bg-surface2"
        >
          <Plus className="size-3" />
          Add block
        </button>
      </div>

      <div className="space-y-2" data-testid="structured-block-list">
        {blocks.map((block, index) => {
          const isSelected = block.id === selectedBlockId
          return (
            <div
              key={block.id}
              data-testid={`structured-block-item-${block.id}`}
              draggable={Boolean(onReorderBlock)}
              onDragStart={(event) => {
                if (!onReorderBlock) return
                draggedBlockId.current = block.id
                if (event.dataTransfer) {
                  event.dataTransfer.effectAllowed = "move"
                  event.dataTransfer.setData("text/plain", block.id)
                }
              }}
              onDragOver={(event) => {
                if (onReorderBlock && draggedBlockId.current !== block.id) {
                  event.preventDefault()
                  if (event.dataTransfer) event.dataTransfer.dropEffect = "move"
                }
              }}
              onDrop={(event) => {
                const draggedId = draggedBlockId.current
                if (!onReorderBlock || !draggedId || draggedId === block.id)
                  return
                event.preventDefault()
                onReorderBlock(draggedId, index)
                draggedBlockId.current = null
              }}
              onDragEnd={() => {
                draggedBlockId.current = null
              }}
              className={`min-w-0 rounded-lg border p-2 ${
                isSelected
                  ? "border-primary bg-primary/5"
                  : "border-border bg-bg"
              } ${onReorderBlock ? "cursor-grab active:cursor-grabbing" : ""}`}
            >
              <button
                type="button"
                onClick={() => onSelect(block.id)}
                aria-pressed={isSelected}
                aria-label={t(
                  "managePrompts.structured.blockList.edit",
                  "Edit {{name}} block",
                  { name: block.name }
                )}
                className="flex min-h-11 w-full min-w-0 items-start justify-between gap-3 rounded text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <div className="min-w-0">
                  <div className="break-words text-sm font-medium text-text">
                    {block.name}
                  </div>
                  <div className="text-xs uppercase tracking-wide text-text-muted">
                    {showRole
                      ? block.role
                      : t(
                          "managePrompts.structured.blockList.number",
                          "Block {{index}}",
                          { index: index + 1 }
                        )}
                    {!block.enabled
                      ? t(
                          "managePrompts.structured.blockList.disabled",
                          " • disabled"
                        )
                      : ""}
                  </div>
                </div>
                <div className="line-clamp-2 min-w-0 max-w-[10rem] break-words text-xs text-text-muted">
                  {block.content || "No content"}
                </div>
              </button>

              <div className="mt-2 flex flex-wrap items-center gap-1">
                <button
                  type="button"
                  onClick={() => onMoveBlock(block.id, "up")}
                  disabled={index === 0}
                  aria-label={t(
                    "managePrompts.structured.blockList.moveUp",
                    "Move {{name}} up",
                    { name: block.name }
                  )}
                  data-testid={`structured-block-move-up-${block.id}`}
                  className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border text-text-muted disabled:opacity-40"
                >
                  <ArrowUp className="size-3" />
                </button>
                <button
                  type="button"
                  onClick={() => onMoveBlock(block.id, "down")}
                  disabled={index === blocks.length - 1}
                  aria-label={t(
                    "managePrompts.structured.blockList.moveDown",
                    "Move {{name}} down",
                    { name: block.name }
                  )}
                  data-testid={`structured-block-move-down-${block.id}`}
                  className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border text-text-muted disabled:opacity-40"
                >
                  <ArrowDown className="size-3" />
                </button>
                <button
                  type="button"
                  onClick={() => onRemoveBlock(block.id)}
                  aria-label={t(
                    "managePrompts.structured.blockList.remove",
                    "Remove {{name}}",
                    { name: block.name }
                  )}
                  data-testid={`structured-block-remove-${block.id}`}
                  className="inline-flex min-h-11 min-w-11 items-center justify-center rounded border border-border text-danger hover:bg-danger/5"
                >
                  <Trash2 className="size-3" />
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
