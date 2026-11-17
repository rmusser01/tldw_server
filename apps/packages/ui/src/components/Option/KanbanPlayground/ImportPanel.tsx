import { useState, useCallback } from "react"
import { useMutation } from "@tanstack/react-query"
import { Upload, Button, message, Card, Alert, Descriptions } from "antd"
import type { UploadProps } from "antd"
import { Upload as UploadIcon, FileJson, CheckCircle } from "lucide-react"

import { useKanbanScope } from "./KanbanScope"
import type { BoardImportResponse } from "@/types/kanban"

interface ImportPanelProps {
  onImported: (boardId: number) => void
}

interface TrelloPreview {
  name: string
  desc?: string
  lists: number
  cards: number
  labels: number
  checklists: number
  isTrello: boolean
}

export const ImportPanel = ({ onImported }: ImportPanelProps) => {
  const { api, isActive } = useKanbanScope()
  const { importBoard } = api
  const [fileData, setFileData] = useState<Record<string, any> | null>(null)
  const [preview, setPreview] = useState<TrelloPreview | null>(null)
  const [importResult, setImportResult] = useState<BoardImportResponse | null>(null)

  // Import mutation
  const importMutation = useMutation({
    mutationFn: (data: Record<string, any>) => importBoard(data),
    onSuccess: (result) => {
      if (!isActive()) return
      message.success("Board imported successfully!")
      setImportResult(result)
      onImported(result.board.id)
    },
    onError: () => {
      if (!isActive()) return
      message.error("Failed to import board. Please try again.")
    }
  })

  // Parse and preview the uploaded file
  const parseFile = useCallback((content: string): TrelloPreview | null => {
    try {
      const data = JSON.parse(content)

      // Check if it's a Trello export
      const isTrello = !!(data.lists && data.cards && !data.format)

      // Check if it's our own format
      const isOwnFormat = data.format === "tldw_kanban_v1"

      if (isTrello) {
        // Trello format
        return {
          name: data.name || "Imported Board",
          desc: data.desc,
          lists: Array.isArray(data.lists) ? data.lists.length : 0,
          cards: Array.isArray(data.cards) ? data.cards.length : 0,
          labels: Array.isArray(data.labels) ? data.labels.length : 0,
          checklists: Array.isArray(data.checklists)
            ? data.checklists.length
            : 0,
          isTrello: true
        }
      } else if (isOwnFormat) {
        // Our format
        const lists = data.lists || []
        let cardCount = 0
        for (const list of lists) {
          cardCount += (list.cards || []).length
        }
        return {
          name: data.board?.name || "Imported Board",
          desc: data.board?.description,
          lists: lists.length,
          cards: cardCount,
          labels: (data.labels || []).length,
          checklists: 0, // Count from cards
          isTrello: false
        }
      }

      // Unknown format - try to make sense of it
      return {
        name: data.name || data.board?.name || "Imported Board",
        desc: data.desc || data.description,
        lists: Array.isArray(data.lists) ? data.lists.length : 0,
        cards: Array.isArray(data.cards) ? data.cards.length : 0,
        labels: Array.isArray(data.labels) ? data.labels.length : 0,
        checklists: 0,
        isTrello: false
      }
    } catch (e) {
      return null
    }
  }, [])

  const handleFileRead = useCallback(
    (file: File) => {
      if (!isActive()) return false
      const reader = new FileReader()
      reader.onerror = () => {
        if (!isActive()) return
        const errorMessage = reader.error?.message
        message.error(
          errorMessage ? `Failed to read file: ${errorMessage}` : "Failed to read file"
        )
        setFileData(null)
        setPreview(null)
      }
      reader.onload = (e) => {
        if (!isActive()) return
        const content = e.target?.result as string
        try {
          const data = JSON.parse(content)
          setFileData(data)
          const preview = parseFile(content)
          if (preview) {
            setPreview(preview)
          } else {
            message.error("Could not parse file. Please ensure it's a valid Trello or tldw export.")
            setFileData(null)
            setPreview(null)
          }
        } catch {
          message.error("Invalid JSON file")
          setFileData(null)
          setPreview(null)
        }
      }
      reader.readAsText(file)
      return false // Prevent upload
    },
    [isActive, parseFile]
  )

  const uploadProps: UploadProps = {
    name: "file",
    accept: ".json",
    showUploadList: false,
    beforeUpload: handleFileRead
  }

  const handleImport = useCallback(() => {
    if (!fileData) return
    importMutation.mutate(fileData)
  }, [fileData, importMutation])

  const handleReset = useCallback(() => {
    setFileData(null)
    setPreview(null)
    setImportResult(null)
  }, [])

  return (
    <div className="import-panel max-w-2xl">
      <h3 className="text-lg font-medium mb-4">Import Board</h3>

      <Alert
        type="info"
        className="mb-4"
        title="Supported formats"
        description={
          <ul className="list-disc ml-4 mt-2">
            <li>
              <strong>Trello JSON export</strong> - Export your board from Trello
              (Menu → More → Print, export, and share → Export as JSON)
            </li>
            <li>
              <strong>tldw Kanban export</strong> - Our native format
            </li>
          </ul>
        }
      />

      {/* Import result */}
      {importResult && (
        <Card className="mb-4 border-success/30 bg-success/10">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle className="w-5 h-5 text-success" />
            <span className="font-medium text-success">
              Import Successful!
            </span>
          </div>
          <Descriptions size="small" column={2}>
            <Descriptions.Item label="Board">
              {importResult.board.name}
            </Descriptions.Item>
            <Descriptions.Item label="Lists">
              {importResult.import_stats.lists_imported}
            </Descriptions.Item>
            <Descriptions.Item label="Cards">
              {importResult.import_stats.cards_imported}
            </Descriptions.Item>
            <Descriptions.Item label="Labels">
              {importResult.import_stats.labels_imported}
            </Descriptions.Item>
            <Descriptions.Item label="Checklists">
              {importResult.import_stats.checklists_imported}
            </Descriptions.Item>
            <Descriptions.Item label="Comments">
              {importResult.import_stats.comments_imported}
            </Descriptions.Item>
          </Descriptions>
          <Button className="mt-3" onClick={handleReset}>
            Import Another
          </Button>
        </Card>
      )}

      {/* File preview */}
      {preview && !importResult && (
        <Card className="mb-4">
          <div className="flex items-center gap-2 mb-3">
            <FileJson className="w-5 h-5 text-primary" />
            <span className="font-medium">File Preview</span>
            {preview.isTrello && (
              <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                Trello Format
              </span>
            )}
          </div>

          <Descriptions size="small" column={2}>
            <Descriptions.Item label="Board Name">
              {preview.name}
            </Descriptions.Item>
            {preview.desc && (
              <Descriptions.Item label="Description">
                {preview.desc.slice(0, 100)}
                {preview.desc.length > 100 ? "..." : ""}
              </Descriptions.Item>
            )}
            <Descriptions.Item label="Lists">{preview.lists}</Descriptions.Item>
            <Descriptions.Item label="Cards">{preview.cards}</Descriptions.Item>
            <Descriptions.Item label="Labels">
              {preview.labels}
            </Descriptions.Item>
            <Descriptions.Item label="Checklists">
              {preview.checklists}
            </Descriptions.Item>
          </Descriptions>

          <div className="flex gap-2 mt-4">
            <Button
              type="primary"
              onClick={handleImport}
              loading={importMutation.isPending}
            >
              Import Board
            </Button>
            <Button onClick={handleReset}>Cancel</Button>
          </div>
        </Card>
      )}

      {/* Upload area */}
      {!preview && !importResult && (
        <Upload.Dragger {...uploadProps} className="mb-4">
          <p className="ant-upload-drag-icon">
            <UploadIcon className="w-12 h-12 mx-auto text-text-subtle" />
          </p>
          <p className="ant-upload-text">
            Click or drag JSON file to this area
          </p>
          <p className="ant-upload-hint">
            Supports Trello exports and tldw Kanban exports
          </p>
        </Upload.Dragger>
      )}
    </div>
  )
}
