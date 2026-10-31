import { useState, useCallback, useRef } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  Dropdown,
  message,
  Spin,
  Button,
  Select,
  Modal,
  Input,
  Badge
} from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  Kanban,
  Upload,
  Download,
  RefreshCw,
  Archive,
  MoreVertical
} from "lucide-react"
import { useTranslation } from "react-i18next"

import { generateClientId } from "@/services/kanban"
import { KanbanScopeProvider, useKanbanScope } from "./KanbanScope"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { BoardView } from "./BoardView"
import { ImportPanel } from "./ImportPanel"
import { ArchivedItemsDrawer } from "./ArchivedItemsDrawer"
import { BoardGallery } from "./BoardGallery"
import { useKanbanShortcuts } from "./useKanbanShortcuts"

const BADGE_INDICATOR_STYLE = {
  backgroundColor: "rgb(var(--color-primary))",
  color: "rgb(var(--color-text))"
}

export const KanbanPlayground = () => {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  if (!config || authorityLoading) return <p role="status">Resolving Kanban connection…</p>
  return (
    <KanbanScopeProvider key={deriveConnectionAuthorityId(config)} config={config}>
      <KanbanWorkspace />
    </KanbanScopeProvider>
  )
}

const KanbanWorkspace = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const { api, key, isActive } = useKanbanScope()
  const { listBoards, getBoard, createBoard, deleteBoard, archiveBoard, exportBoard, createList } = api

  // Board selection state
  const [selectedBoardId, setSelectedBoardId] = useState<number | null>(null)

  // Create board modal state
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [newBoardName, setNewBoardName] = useState("")
  const [newBoardDescription, setNewBoardDescription] = useState("")

  // Import modal state
  const [importModalOpen, setImportModalOpen] = useState(false)

  // Archive drawer state
  const [archiveDrawerOpen, setArchiveDrawerOpen] = useState(false)

  // Ref for BoardView shortcut handlers (add-card / add-list)
  const boardShortcutRef = useRef<{
    startAddCard: () => void
    startAddList: () => void
  } | null>(null)

  // Fetch boards list
  const {
    data: boardsData,
    isLoading: boardsLoading,
    refetch: refetchBoards
  } = useQuery({
    queryKey: key("kanban-boards"),
    queryFn: () => listBoards({ limit: 100 }),
    staleTime: 60 * 1000
  })

  // Fetch selected board with lists and cards
  const {
    data: boardData,
    isLoading: boardLoading,
    refetch: refetchBoard
  } = useQuery({
    queryKey: key("kanban-board", selectedBoardId),
    queryFn: () => getBoard(selectedBoardId!),
    enabled: selectedBoardId !== null,
    staleTime: 30 * 1000
  })

  // Create board mutation
  const createBoardMutation = useMutation({
    mutationFn: ({ name, description }: { name: string; description?: string }) =>
      createBoard({ name, description, client_id: generateClientId() }),
    onSuccess: (newBoard) => {
      if (!isActive()) return
      message.success("Board created")
      queryClient.invalidateQueries({ queryKey: key("kanban-boards") })
      setSelectedBoardId(newBoard.id)
      setCreateModalOpen(false)
      setNewBoardName("")
      setNewBoardDescription("")
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to create board. Please try again.")
    }
  })

  // Delete board mutation
  const deleteBoardMutation = useMutation({
    mutationFn: (boardId: number) => deleteBoard(boardId),
    onSuccess: () => {
      if (!isActive()) return
      message.success("Board deleted")
      queryClient.invalidateQueries({ queryKey: key("kanban-boards") })
      setSelectedBoardId(null)
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to delete board. Please try again.")
    }
  })

  // Archive board mutation
  const archiveBoardMutation = useMutation({
    mutationFn: (boardId: number) => archiveBoard(boardId),
    onSuccess: () => {
      if (!isActive()) return
      message.success("Board archived")
      queryClient.invalidateQueries({ queryKey: key("kanban-boards") })
      setSelectedBoardId(null)
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to archive board. Please try again.")
    }
  })

  const handleCreateBoard = useCallback(() => {
    if (!newBoardName.trim()) {
      message.warning("Please enter a board name")
      return
    }
    createBoardMutation.mutate({
      name: newBoardName.trim(),
      description: newBoardDescription.trim() || undefined
    })
  }, [newBoardName, newBoardDescription, createBoardMutation])

  const handleDeleteBoard = useCallback(() => {
    if (!selectedBoardId) return
    deleteBoardMutation.mutate(selectedBoardId)
  }, [selectedBoardId, deleteBoardMutation])

  const handleArchiveBoard = useCallback(() => {
    if (!selectedBoardId) return
    archiveBoardMutation.mutate(selectedBoardId)
  }, [selectedBoardId, archiveBoardMutation])

  const handleExportBoard = useCallback(async () => {
    if (!selectedBoardId || !boardData || !isActive()) return
    try {
      const data = await exportBoard(selectedBoardId)
      if (!isActive()) return
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json"
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${boardData.name.replace(/[^a-z0-9]/gi, "_")}_export.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      message.success("Board exported")
    } catch (err) {
      if (!isActive()) return
      message.error("Failed to export board. Please try again.")
    }
  }, [selectedBoardId, boardData, isActive, exportBoard])

  const handleBoardImported = useCallback(
    (boardId: number) => {
      if (!isActive()) return
      queryClient.invalidateQueries({ queryKey: key("kanban-boards") })
      setSelectedBoardId(boardId)
      setImportModalOpen(false)
    },
    [isActive, key, queryClient]
  )

  // Quick setup: create To Do / In Progress / Done lists
  const handleQuickSetup = useCallback(async () => {
    if (!selectedBoardId) return
    const names = ["To Do", "In Progress", "Done"]
    try {
      for (const name of names) {
        if (!isActive()) return
        await createList(selectedBoardId, {
          name,
          client_id: generateClientId()
        })
      }
      if (!isActive()) return
      message.success("Lists created")
      queryClient.invalidateQueries({
        queryKey: key("kanban-board", selectedBoardId)
      })
    } catch (err) {
      if (!isActive()) return
      message.error("Failed to create lists. Please try again.")
    }
  }, [selectedBoardId, isActive, queryClient, key, createList])

  // Keyboard shortcuts
  const { helpOpen, setHelpOpen } = useKanbanShortcuts({
    onNewBoard: () => setCreateModalOpen(true),
    onNewCard: () => boardShortcutRef.current?.startAddCard(),
    onNewList: () => boardShortcutRef.current?.startAddList()
  })

  const boards = boardsData?.boards ?? []

  const boardSelectorOptions = boards.map((b) => ({
    value: b.id,
    label: b.name
  }))

  // Actions dropdown menu items
  const actionsMenuItems: MenuProps["items"] = [
    {
      key: "import",
      label: "Import Board...",
      icon: <Upload className="w-4 h-4" />,
      onClick: () => setImportModalOpen(true)
    },
    ...(selectedBoardId
      ? [
          {
            key: "export",
            label: "Export Board...",
            icon: <Download className="w-4 h-4" />,
            onClick: handleExportBoard
          }
        ]
      : []),
    { type: "divider" as const },
    {
      key: "archive",
      label: "Archived Items",
      icon: <Archive className="w-4 h-4" />,
      onClick: () => setArchiveDrawerOpen(true)
    }
  ]

  const renderHeader = () => (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-3">
        <Kanban className="w-6 h-6 text-primary" />
        <h1 className="text-xl font-semibold">Kanban Playground</h1>
      </div>
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <span>Boards</span>
          <Badge
            count={boards.length}
            showZero
            styles={{ indicator: BADGE_INDICATOR_STYLE }}
          />
        </div>
        <Select
          placeholder="Select a board"
          style={{ width: 200 }}
          value={selectedBoardId}
          onChange={setSelectedBoardId}
          options={boardSelectorOptions}
          loading={boardsLoading}
          allowClear
          onClear={() => setSelectedBoardId(null)}
        />
        <Button
          icon={<Plus className="w-4 h-4" />}
          onClick={() => setCreateModalOpen(true)}
        >
          New Board
        </Button>
        <Dropdown menu={{ items: actionsMenuItems }} trigger={["click"]}>
          <Button icon={<MoreVertical className="w-4 h-4" />} />
        </Dropdown>
        <Button
          icon={<RefreshCw className="w-4 h-4" />}
          onClick={() => {
            refetchBoards()
            if (selectedBoardId) refetchBoard()
          }}
        />
      </div>
    </div>
  )

  const renderContent = () => {
    const isLoading = selectedBoardId ? boardLoading : boardsLoading

    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-96">
          <Spin size="large" />
        </div>
      )
    }

    if (!selectedBoardId) {
      return (
        <BoardGallery
          boards={boards}
          onSelectBoard={setSelectedBoardId}
          onCreateBoard={() => setCreateModalOpen(true)}
        />
      )
    }

    if (boardData) {
      return (
        <>
          <BoardView
            board={boardData}
            onRefresh={() => refetchBoard()}
            onDelete={handleDeleteBoard}
            onArchive={handleArchiveBoard}
            onQuickSetup={
              boardData.lists.length === 0 ? handleQuickSetup : undefined
            }
            shortcutHandlersRef={boardShortcutRef}
          />
        </>
      )
    }

    return null
  }

  return (
    <div className="kanban-playground">
      {renderHeader()}

      <div className="min-h-[500px]">{renderContent()}</div>

      {/* Archive drawer */}
      <ArchivedItemsDrawer
        open={archiveDrawerOpen}
        onClose={() => setArchiveDrawerOpen(false)}
        boardId={selectedBoardId}
      />

      {/* Import modal */}
      <Modal
        title="Import Board"
        open={importModalOpen}
        onCancel={() => setImportModalOpen(false)}
        footer={null}
        width={600}
      >
        <ImportPanel onImported={handleBoardImported} />
      </Modal>

      {/* Create Board Modal */}
      <Modal
        title="Create New Board"
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false)
          setNewBoardName("")
          setNewBoardDescription("")
        }}
        onOk={handleCreateBoard}
        okText="Create"
        confirmLoading={createBoardMutation.isPending}
      >
        <div className="py-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Board Name
            </label>
            <Input
              placeholder="Enter board name"
              value={newBoardName}
              onChange={(e) => setNewBoardName(e.target.value)}
              onPressEnter={handleCreateBoard}
              autoFocus
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              Description{" "}
              <span className="text-text-muted font-normal">(optional)</span>
            </label>
            <Input.TextArea
              placeholder="What is this board for?"
              value={newBoardDescription}
              onChange={(e) => setNewBoardDescription(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 4 }}
            />
          </div>
        </div>
      </Modal>

      {/* Keyboard shortcuts help */}
      <Modal
        title="Keyboard Shortcuts"
        open={helpOpen}
        onCancel={() => setHelpOpen(false)}
        footer={null}
        width={360}
      >
        <div className="space-y-2 py-2">
          {[
            { key: "N", desc: "New card" },
            { key: "B", desc: "New board" },
            { key: "L", desc: "New list" },
            { key: "Esc", desc: "Close panel" },
            { key: "?", desc: "Show this help" }
          ].map(({ key, desc }) => (
            <div key={key} className="flex items-center justify-between">
              <span className="text-sm">{desc}</span>
              <kbd className="px-2 py-0.5 rounded bg-surface text-xs font-mono border">
                {key}
              </kbd>
            </div>
          ))}
        </div>
      </Modal>
    </div>
  )
}

export default KanbanPlayground
