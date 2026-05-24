import React, { useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button, Input, Table as AntTable, message } from "antd"
import {
  Plus,
  Save,
  Search,
  ZoomIn,
  ZoomOut,
} from "lucide-react"

import Mermaid from "@/components/Common/Mermaid"

import {
  extractMermaidCode,
  isLikelyMermaidDiagram,
  markdownTableToCsv,
  parseMarkdownTable,
} from "./hooks/useArtifactGeneration"
import { downloadBlobFile } from "./hooks/useArtifactExport"
import type {
  FlashcardDraft,
  QuizQuestionDraft,
} from "./hooks/useQuizParsing"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction,
} from "../undo-manager"

export const MindMapArtifactViewer: React.FC<{
  title: string
  content: string
}> = ({ title, content }) => {
  const [zoom, setZoom] = useState(1)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mermaidCode = React.useMemo(() => extractMermaidCode(content), [content])
  const canRenderMermaid = React.useMemo(
    () => isLikelyMermaidDiagram(mermaidCode),
    [mermaidCode]
  )

  const handleExportSvg = () => {
    const svg = containerRef.current?.querySelector("svg")
    if (!svg) return
    const svgBlob = new Blob([svg.outerHTML], {
      type: "image/svg+xml;charset=utf-8"
    })
    downloadBlobFile(svgBlob, `${title || "mind-map"}.svg`)
  }

  const handleExportPng = async () => {
    if (!containerRef.current) return
    const html2canvas = (await import("html2canvas")).default
    const canvas = await html2canvas(containerRef.current, {
      backgroundColor: "#ffffff",
      scale: 2
    })
    canvas.toBlob((blob) => {
      if (!blob) return
      downloadBlobFile(blob, `${title || "mind-map"}.png`)
    }, "image/png")
  }

  if (!canRenderMermaid) {
    return (
      <div className="flex max-h-[70vh] flex-col gap-3">
        <div className="rounded border border-warning/40 bg-warning/10 p-3 text-sm text-text">
          Unable to render this mind map as a diagram. Showing raw output instead.
        </div>
        <div className="max-h-[56vh] overflow-auto whitespace-pre-wrap rounded border border-border bg-surface p-4 text-sm">
          {content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex max-h-[70vh] flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="small"
          icon={<ZoomOut className="h-3.5 w-3.5" />}
          onClick={() => setZoom((prev) => Math.max(0.5, Number((prev - 0.1).toFixed(2))))}
        >
          Zoom out
        </Button>
        <span className="text-xs text-text-muted">{Math.round(zoom * 100)}%</span>
        <Button
          size="small"
          icon={<ZoomIn className="h-3.5 w-3.5" />}
          onClick={() => setZoom((prev) => Math.min(2.5, Number((prev + 0.1).toFixed(2))))}
        >
          Zoom in
        </Button>
        <Button size="small" onClick={() => setZoom(1)}>
          Reset
        </Button>
        <Button size="small" onClick={handleExportSvg}>
          Export SVG
        </Button>
        <Button size="small" onClick={() => void handleExportPng()}>
          Export PNG
        </Button>
      </div>

      <div className="rounded border border-border bg-surface2/40 p-2 text-xs text-text-muted">
        Scroll to pan the diagram when zoomed in.
      </div>

      <div className="max-h-[56vh] overflow-auto rounded border border-border bg-surface p-4">
        <div
          ref={containerRef}
          style={{ transform: `scale(${zoom})`, transformOrigin: "top left" }}
          className="inline-block min-w-full"
        >
          <Mermaid code={mermaidCode} />
        </div>
      </div>
    </div>
  )
}

export const DataTableArtifactViewer: React.FC<{
  title: string
  content: string
}> = ({ title, content }) => {
  const [query, setQuery] = useState("")
  const tableData = React.useMemo(() => parseMarkdownTable(content), [content])

  const filteredRows = React.useMemo(() => {
    if (!tableData) return []
    const normalized = query.trim().toLowerCase()
    if (!normalized) return tableData.rows
    return tableData.rows.filter((row) =>
      row.some((cell) => cell.toLowerCase().includes(normalized))
    )
  }, [query, tableData])

  const columns = React.useMemo(() => {
    if (!tableData) return []
    return tableData.headers.map((header, index) => ({
      title: header || `Column ${index + 1}`,
      dataIndex: `col_${index}`,
      key: `col_${index}`,
      sorter: (a: Record<string, string>, b: Record<string, string>) =>
        String(a[`col_${index}`] || "").localeCompare(
          String(b[`col_${index}`] || ""),
          undefined,
          { sensitivity: "base", numeric: true }
        )
    }))
  }, [tableData])

  const dataSource = React.useMemo(() => {
    return filteredRows.map((row, rowIndex) => {
      const record: Record<string, string> = { key: String(rowIndex) }
      row.forEach((cell, cellIndex) => {
        record[`col_${cellIndex}`] = cell
      })
      return record
    })
  }, [filteredRows])

  const handleDownloadCsv = () => {
    if (!tableData) return
    const csv = markdownTableToCsv(tableData)
    const csvBlob = new Blob([csv], { type: "text/csv;charset=utf-8" })
    downloadBlobFile(csvBlob, `${title || "data-table"}.csv`)
  }

  if (!tableData) {
    return (
      <div className="max-h-[70vh] overflow-y-auto whitespace-pre-wrap rounded border border-border bg-surface p-3 text-sm">
        {content}
      </div>
    )
  }

  return (
    <div className="flex max-h-[70vh] flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter table rows"
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          className="w-72"
        />
        <Button size="small" onClick={handleDownloadCsv}>
          Export CSV
        </Button>
      </div>
      <AntTable
        columns={columns}
        dataSource={dataSource}
        pagination={{ pageSize: 10, size: "small" }}
        size="small"
        scroll={{ x: true, y: 420 }}
      />
    </div>
  )
}

export const FlashcardArtifactEditor: React.FC<{
  cards: FlashcardDraft[]
  onSave: (cards: FlashcardDraft[]) => void
}> = ({ cards, onSave }) => {
  const { t } = useTranslation(["playground", "common"])
  const [messageApi, messageContextHolder] = message.useMessage()
  const [draftCards, setDraftCards] = useState<FlashcardDraft[]>(cards)

  const updateCard = (
    index: number,
    patch: Partial<FlashcardDraft>
  ) => {
    setDraftCards((previous) =>
      previous.map((card, cardIndex) =>
        cardIndex === index ? { ...card, ...patch } : card
      )
    )
  }

  const removeCard = React.useCallback(
    (index: number) => {
      const removedCard = draftCards[index]
      if (!removedCard) return
      const nextCards = draftCards.filter(
        (_card, cardIndex) => cardIndex !== index
      )
      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          setDraftCards(nextCards)
        },
        undo: () => {
          setDraftCards((previous) => {
            const restored = [...previous]
            const insertionIndex = Math.max(0, Math.min(index, restored.length))
            restored.splice(insertionIndex, 0, removedCard)
            return restored
          })
        }
      })

      const undoMessageKey = `workspace-flashcard-remove-undo-${undoHandle.id}`
      const maybeOpen = (
        messageApi as { open?: (config: unknown) => void }
      ).open
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t(
          "playground:studio.flashcardRemoved",
          "Flashcard removed."
        ),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                messageApi.success(
                  t(
                    "playground:studio.flashcardRestored",
                    "Flashcard restored"
                  )
                )
              }
              messageApi.destroy(undoMessageKey)
            }}
          >
            {t("common:undo", "Undo")}
          </Button>
        )
      }

      if (typeof maybeOpen === "function") {
        maybeOpen(messageConfig)
      } else {
        const maybeWarning = (
          messageApi as { warning?: (content: string) => void }
        ).warning
        if (typeof maybeWarning === "function") {
          maybeWarning(t("playground:studio.flashcardRemoved", "Flashcard removed."))
        }
      }
    },
    [draftCards, messageApi, t]
  )

  return (
    <div className="flex max-h-[70vh] flex-col gap-3">
      {messageContextHolder}
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-muted">
          Edit generated flashcards before reusing them.
        </p>
        <Button
          size="small"
          icon={<Plus className="h-3.5 w-3.5" />}
          onClick={() =>
            setDraftCards((previous) => [...previous, { front: "", back: "" }])
          }
        >
          Add card
        </Button>
      </div>

      <div className="max-h-[54vh] space-y-3 overflow-y-auto pr-1">
        {draftCards.map((card, index) => (
          <div key={`flashcard-${index}`} className="rounded border border-border bg-surface2/30 p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-text-muted">Card {index + 1}</span>
              <Button danger size="small" onClick={() => removeCard(index)}>
                Remove
              </Button>
            </div>
            <Input.TextArea
              value={card.front}
              onChange={(event) => updateCard(index, { front: event.target.value })}
              rows={2}
              placeholder="Front (question or term)"
              className="mb-2"
            />
            <Input.TextArea
              value={card.back}
              onChange={(event) => updateCard(index, { back: event.target.value })}
              rows={3}
              placeholder="Back (answer or definition)"
            />
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <Button
          type="primary"
          icon={<Save className="h-3.5 w-3.5" />}
          onClick={() =>
            onSave(
              draftCards.filter(
                (card) => card.front.trim().length > 0 && card.back.trim().length > 0
              )
            )
          }
        >
          Save changes
        </Button>
      </div>
    </div>
  )
}

export const QuizArtifactEditor: React.FC<{
  questions: QuizQuestionDraft[]
  onSave: (questions: QuizQuestionDraft[]) => void
}> = ({ questions, onSave }) => {
  const { t } = useTranslation(["playground", "common"])
  const [messageApi, messageContextHolder] = message.useMessage()
  const [draftQuestions, setDraftQuestions] = useState<QuizQuestionDraft[]>(questions)

  const updateQuestion = (
    index: number,
    patch: Partial<QuizQuestionDraft>
  ) => {
    setDraftQuestions((previous) =>
      previous.map((question, questionIndex) =>
        questionIndex === index ? { ...question, ...patch } : question
      )
    )
  }

  const removeQuestion = React.useCallback(
    (index: number) => {
      const removedQuestion = draftQuestions[index]
      if (!removedQuestion) return
      const nextQuestions = draftQuestions.filter(
        (_question, questionIndex) => questionIndex !== index
      )
      const undoHandle = scheduleWorkspaceUndoAction({
        apply: () => {
          setDraftQuestions(nextQuestions)
        },
        undo: () => {
          setDraftQuestions((previous) => {
            const restored = [...previous]
            const insertionIndex = Math.max(0, Math.min(index, restored.length))
            restored.splice(insertionIndex, 0, removedQuestion)
            return restored
          })
        }
      })

      const undoMessageKey = `workspace-quiz-remove-undo-${undoHandle.id}`
      const maybeOpen = (
        messageApi as { open?: (config: unknown) => void }
      ).open
      const messageConfig = {
        key: undoMessageKey,
        type: "warning",
        duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
        content: t(
          "playground:studio.quizQuestionRemoved",
          "Question removed."
        ),
        btn: (
          <Button
            size="small"
            type="link"
            onClick={() => {
              if (undoWorkspaceAction(undoHandle.id)) {
                messageApi.success(
                  t(
                    "playground:studio.quizQuestionRestored",
                    "Question restored"
                  )
                )
              }
              messageApi.destroy(undoMessageKey)
            }}
          >
            {t("common:undo", "Undo")}
          </Button>
        )
      }

      if (typeof maybeOpen === "function") {
        maybeOpen(messageConfig)
      } else {
        const maybeWarning = (
          messageApi as { warning?: (content: string) => void }
        ).warning
        if (typeof maybeWarning === "function") {
          maybeWarning(
            t("playground:studio.quizQuestionRemoved", "Question removed.")
          )
        }
      }
    },
    [draftQuestions, messageApi, t]
  )

  return (
    <div className="flex max-h-[70vh] flex-col gap-3">
      {messageContextHolder}
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-muted">
          Edit generated quiz questions and answers.
        </p>
        <Button
          size="small"
          icon={<Plus className="h-3.5 w-3.5" />}
          onClick={() =>
            setDraftQuestions((previous) => [
              ...previous,
              { question: "", options: [], answer: "", explanation: "" }
            ])
          }
        >
          Add question
        </Button>
      </div>

      <div className="max-h-[54vh] space-y-3 overflow-y-auto pr-1">
        {draftQuestions.map((question, index) => (
          <div key={`quiz-${index}`} className="rounded border border-border bg-surface2/30 p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-text-muted">
                Question {index + 1}
              </span>
              <Button danger size="small" onClick={() => removeQuestion(index)}>
                Remove
              </Button>
            </div>
            <Input.TextArea
              value={question.question}
              onChange={(event) =>
                updateQuestion(index, { question: event.target.value })
              }
              rows={2}
              placeholder="Question prompt"
              className="mb-2"
            />
            <Input.TextArea
              value={question.options.join("\n")}
              onChange={(event) =>
                updateQuestion(index, {
                  options: event.target.value
                    .split("\n")
                    .map((option) => option.trim())
                    .filter(Boolean)
                })
              }
              rows={3}
              placeholder="Options (one per line)"
              className="mb-2"
            />
            <Input
              value={question.answer}
              onChange={(event) =>
                updateQuestion(index, { answer: event.target.value })
              }
              placeholder="Correct answer"
              className="mb-2"
            />
            <Input.TextArea
              value={question.explanation || ""}
              onChange={(event) =>
                updateQuestion(index, { explanation: event.target.value })
              }
              rows={2}
              placeholder="Explanation (optional)"
            />
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <Button
          type="primary"
          icon={<Save className="h-3.5 w-3.5" />}
          onClick={() =>
            onSave(
              draftQuestions.filter(
                (question) =>
                  question.question.trim().length > 0 &&
                  question.answer.trim().length > 0
              )
            )
          }
        >
          Save changes
        </Button>
      </div>
    </div>
  )
}
