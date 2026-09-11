import React from "react"
import {
  Alert,
  Button,
  Input,
  InputNumber,
  Select,
  Tooltip,
  Typography,
  message
} from "antd"
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  DeleteOutlined,
  PlusOutlined,
  SaveOutlined
} from "@ant-design/icons"
import { useTranslation } from "react-i18next"
import type { SourceCitation } from "@/services/quizzes"

import { QuizMarkdown } from "../components/QuizMarkdown"
import { SourceCitations } from "../components/SourceCitations"
import {
  getOsceStation,
  type OsceCitation,
  type OsceChecklistItemUpdate,
  type OsceKeyPointUpdate,
  type OsceRubricDomainUpdate,
  type OsceStationAuthoringResponse,
  type OsceStationCreateContent,
  type OsceStationUpdateContent
} from "@/services/osce"
import {
  useCreateOsceStationMutation,
  useUpdateOsceStationMutation
} from "../hooks/useOsceQueries"

export type OsceStationDraft = Omit<
  OsceStationCreateContent,
  "checklist_items" | "rubric_domains" | "expected_key_points"
> & {
  checklist_items: OsceChecklistItemUpdate[]
  rubric_domains: OsceRubricDomainUpdate[]
  expected_key_points: OsceKeyPointUpdate[]
}

export interface OsceStationEditorProps {
  quizId?: number
  station?: OsceStationAuthoringResponse | null
  initialContent?: OsceStationDraft
  orderIndex?: number
  onCreate?: (
    content: OsceStationCreateContent,
    orderIndex: number
  ) => Promise<OsceStationAuthoringResponse>
  onSaved?: (station: OsceStationAuthoringResponse) => void
  onDirtyStateChange?: (dirty: boolean) => void
}

const newCitation = (): OsceCitation => ({
  source_type: "media",
  source_id: "",
  label: "",
  quote: "",
  media_id: null,
  chunk_id: null,
  timestamp_seconds: null,
  page_number: null,
  source_url: null
})

const toQuizSourceCitations = (citations: OsceCitation[]): SourceCitation[] =>
  citations.map((citation) => ({
    source_type:
      citation.source_type === "media" || citation.source_type === "note"
        ? citation.source_type
        : undefined,
    source_id: citation.source_id,
    label: citation.label,
    quote: citation.quote,
    media_id: citation.media_id,
    chunk_id: citation.chunk_id,
    timestamp_seconds: citation.timestamp_seconds,
    source_url: citation.source_url
  }))

export const createEmptyOsceStationDraft = (): OsceStationDraft => ({
  schema_version: "osce.station.v1",
  title: "",
  candidate_instructions: "",
  candidate_task: "",
  patient_context: { text: "", citations: [] },
  recommended_duration_seconds: 480,
  checklist_items: [{ label: "", rationale: "", citations: [] }],
  rubric_domains: [{
    label: "",
    levels: [
      { label: "Needs development", description: "" },
      { label: "Effective", description: "" }
    ]
  }],
  expected_key_points: [{ text: "", citations: [] }]
})

const cloneDraft = (content: OsceStationDraft): OsceStationDraft =>
  JSON.parse(JSON.stringify(content)) as OsceStationDraft

const serializeDraft = (content: OsceStationDraft): string => JSON.stringify(content)

const errorStatus = (error: unknown): number | null => {
  if (!error || typeof error !== "object") return null
  const status = Number((error as { status?: unknown }).status)
  return Number.isFinite(status) ? Math.trunc(status) : null
}

const errorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.trim()) return error.message
  return "Failed to save station."
}

const move = <T,>(items: T[], from: number, to: number): T[] => {
  if (to < 0 || to >= items.length || from === to) return items
  const next = [...items]
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

const validateDraft = (draft: OsceStationDraft): string[] => {
  const errors: string[] = []
  if (!draft.title.trim()) errors.push("Station title is required.")
  if (!draft.candidate_instructions.trim()) errors.push("Candidate instructions are required.")
  if (!draft.candidate_task.trim()) errors.push("Candidate task is required.")
  if (!draft.patient_context.text.trim()) errors.push("Patient context is required.")
  if (draft.recommended_duration_seconds < 60 || draft.recommended_duration_seconds > 7200) {
    errors.push("Recommended duration must be between 60 and 7200 seconds.")
  }
  if (draft.checklist_items.length < 1 || draft.checklist_items.some((item) => !item.label.trim())) {
    errors.push("At least one labeled checklist item is required.")
  }
  if (
    draft.rubric_domains.length < 1 ||
    draft.rubric_domains.some((domain) =>
      !domain.label.trim() ||
      domain.levels.length < 2 ||
      domain.levels.length > 6 ||
      domain.levels.some((level) => !level.label.trim() || !level.description.trim())
    )
  ) {
    errors.push("Each rubric domain needs a label and two to six complete ordered levels.")
  }
  if (draft.expected_key_points.length < 1 || draft.expected_key_points.some((point) => !point.text.trim())) {
    errors.push("At least one expected key point is required.")
  }
  const citations = [
    ...draft.patient_context.citations,
    ...draft.checklist_items.flatMap((item) => item.citations),
    ...draft.expected_key_points.flatMap((point) => point.citations)
  ]
  if (citations.some((citation) => !citation.source_id.trim())) {
    errors.push("Every citation needs a source ID.")
  }
  return errors
}

type CitationEditorProps = {
  citations: OsceCitation[]
  label: string
  onChange: (citations: OsceCitation[]) => void
}

const CitationEditor: React.FC<CitationEditorProps> = ({ citations, label, onChange }) => {
  const updateCitation = (index: number, patch: Partial<OsceCitation>) => {
    onChange(citations.map((citation, itemIndex) => itemIndex === index ? { ...citation, ...patch } : citation))
  }

  return (
    <div className="space-y-2">
      <div className="flex min-h-8 flex-wrap items-center justify-between gap-2">
        <Typography.Text className="text-xs font-medium text-text-muted">{label}</Typography.Text>
        <Button
          size="small"
          type="text"
          icon={<PlusOutlined aria-hidden />}
          onClick={() => onChange([...citations, newCitation()])}
        >
          Add citation
        </Button>
      </div>
      {citations.map((citation, index) => (
        <div
          key={`${citation.source_type}-${citation.source_id}-${index}`}
          className="grid min-w-0 grid-cols-1 gap-2 border-t border-border-subtle pt-2 md:grid-cols-[10rem_minmax(0,1fr)_minmax(0,1fr)_2.75rem]"
        >
          <Select
            aria-label={`${label} ${index + 1} source type`}
            value={citation.source_type}
            options={[
              { value: "media", label: "Media" },
              { value: "document", label: "Document" },
              { value: "url", label: "URL" },
              { value: "note", label: "Note" }
            ]}
            onChange={(source_type) => updateCitation(index, {
              source_type,
              media_id: null,
              chunk_id: null,
              timestamp_seconds: null,
              page_number: null,
              source_url: null
            })}
          />
          <Input
            aria-label={`${label} ${index + 1} source ID`}
            value={citation.source_id}
            maxLength={512}
            placeholder="Source ID"
            onChange={(event) => updateCitation(index, { source_id: event.target.value })}
          />
          <Input
            aria-label={`${label} ${index + 1} label`}
            value={citation.label ?? ""}
            maxLength={200}
            placeholder="Source label"
            onChange={(event) => updateCitation(index, { label: event.target.value })}
          />
          <Tooltip title={`Remove ${label.toLowerCase()} ${index + 1}`}>
            <Button
              type="text"
              danger
              icon={<DeleteOutlined aria-hidden />}
              aria-label={`Remove ${label.toLowerCase()} ${index + 1}`}
              onClick={() => onChange(citations.filter((_, itemIndex) => itemIndex !== index))}
              className="h-10 w-11"
            />
          </Tooltip>
          <Input.TextArea
            aria-label={`${label} ${index + 1} quote`}
            value={citation.quote ?? ""}
            maxLength={1000}
            rows={2}
            placeholder="Evidence quote"
            onChange={(event) => updateCitation(index, { quote: event.target.value })}
            className="md:col-span-2"
          />
          {citation.source_type === "url" ? (
            <Input
              aria-label={`${label} ${index + 1} source URL`}
              value={citation.source_url ?? ""}
              maxLength={2048}
              placeholder="https://example.org/source"
              onChange={(event) => updateCitation(index, { source_url: event.target.value })}
              className="md:col-span-2"
            />
          ) : (
            <div className="grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-2 md:col-span-2">
              <Input
                aria-label={`${label} ${index + 1} chunk ID`}
                value={citation.chunk_id ?? ""}
                maxLength={512}
                placeholder="Chunk ID"
                onChange={(event) => updateCitation(index, { chunk_id: event.target.value || null })}
              />
              {citation.source_type === "media" ? (
                <InputNumber
                  aria-label={`${label} ${index + 1} timestamp seconds`}
                  min={0}
                  className="w-full"
                  value={citation.timestamp_seconds}
                  placeholder="Timestamp seconds"
                  onChange={(value) => updateCitation(index, { timestamp_seconds: value == null ? null : Number(value) })}
                />
              ) : citation.source_type === "document" ? (
                <InputNumber
                  aria-label={`${label} ${index + 1} page number`}
                  min={1}
                  className="w-full"
                  value={citation.page_number}
                  placeholder="Page number"
                  onChange={(value) => updateCitation(index, { page_number: value == null ? null : Number(value) })}
                />
              ) : null}
            </div>
          )}
        </div>
      ))}
      <SourceCitations citations={toQuizSourceCitations(citations)} />
    </div>
  )
}

type OrderButtonsProps = {
  label: string
  index: number
  count: number
  onMove: (to: number) => void
  onRemove: () => void
  removeDisabled?: boolean
}

const OrderButtons: React.FC<OrderButtonsProps> = ({
  label,
  index,
  count,
  onMove,
  onRemove,
  removeDisabled
}) => (
  <div className="flex h-11 shrink-0 items-center gap-1">
    <Tooltip title={`Move ${label} up`}>
      <Button
        type="text"
        icon={<ArrowUpOutlined aria-hidden />}
        aria-label={`Move ${label} up`}
        disabled={index === 0}
        onClick={() => onMove(index - 1)}
        className="h-11 w-11"
      />
    </Tooltip>
    <Tooltip title={`Move ${label} down`}>
      <Button
        type="text"
        icon={<ArrowDownOutlined aria-hidden />}
        aria-label={`Move ${label} down`}
        disabled={index === count - 1}
        onClick={() => onMove(index + 1)}
        className="h-11 w-11"
      />
    </Tooltip>
    <Tooltip title={`Remove ${label}`}>
      <Button
        type="text"
        danger
        icon={<DeleteOutlined aria-hidden />}
        aria-label={`Remove ${label}`}
        disabled={removeDisabled}
        onClick={onRemove}
        className="h-11 w-11"
      />
    </Tooltip>
  </div>
)

export const OsceStationEditor: React.FC<OsceStationEditorProps> = ({
  quizId,
  station,
  initialContent,
  orderIndex = 0,
  onCreate,
  onSaved,
  onDirtyStateChange
}) => {
  const { t } = useTranslation(["option", "common"])
  const [messageApi, contextHolder] = message.useMessage()
  const initialDraft = React.useMemo(
    () => cloneDraft(station?.content ?? initialContent ?? createEmptyOsceStationDraft()),
    [initialContent, station]
  )
  const [draft, setDraft] = React.useState<OsceStationDraft>(initialDraft)
  const [acknowledged, setAcknowledged] = React.useState(initialDraft)
  const [currentVersion, setCurrentVersion] = React.useState(station?.version ?? null)
  const [conflictServer, setConflictServer] = React.useState<OsceStationAuthoringResponse | null>(null)
  const [overwriteVersion, setOverwriteVersion] = React.useState<number | null>(null)
  const [confirmOverwrite, setConfirmOverwrite] = React.useState(false)
  const createMutation = useCreateOsceStationMutation()
  const updateMutation = useUpdateOsceStationMutation()
  const dirty = serializeDraft(draft) !== serializeDraft(acknowledged)
  const validationErrors = React.useMemo(() => validateDraft(draft), [draft])
  const saving = createMutation.isPending || updateMutation.isPending

  React.useEffect(() => {
    const next = cloneDraft(station?.content ?? initialContent ?? createEmptyOsceStationDraft())
    setDraft(next)
    setAcknowledged(next)
    setCurrentVersion(station?.version ?? null)
    setConflictServer(null)
    setOverwriteVersion(null)
    setConfirmOverwrite(false)
  }, [initialContent, station?.content, station?.id, station?.version])

  React.useEffect(() => {
    onDirtyStateChange?.(dirty)
  }, [dirty, onDirtyStateChange])

  React.useEffect(() => {
    if (!dirty) return
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = ""
    }
    window.addEventListener("beforeunload", warn)
    return () => window.removeEventListener("beforeunload", warn)
  }, [dirty])

  const acknowledge = (saved: OsceStationAuthoringResponse) => {
    const next = cloneDraft(saved.content)
    setDraft(next)
    setAcknowledged(next)
    setCurrentVersion(saved.version)
    setConflictServer(null)
    setOverwriteVersion(null)
    setConfirmOverwrite(false)
    onSaved?.(saved)
  }

  const performSave = async (expectedVersion?: number) => {
    if (validationErrors.length > 0) return
    try {
      let saved: OsceStationAuthoringResponse
      if (station && quizId != null) {
        saved = await updateMutation.mutateAsync({
          quizId,
          stationId: station.id,
          request: {
            expected_version: expectedVersion ?? currentVersion ?? station.version,
            content: draft as OsceStationUpdateContent,
            order_index: station.order_index
          }
        })
      } else if (onCreate) {
        saved = await onCreate(draft as OsceStationCreateContent, orderIndex)
      } else if (quizId != null) {
        saved = await createMutation.mutateAsync({
          quizId,
          request: { content: draft as OsceStationCreateContent, order_index: orderIndex }
        })
      } else {
        throw new Error("Quiz details must be saved with the station.")
      }
      acknowledge(saved)
      messageApi.success(t("option:quiz.osceStationSaved", { defaultValue: "Station saved." }))
    } catch (error) {
      if (errorStatus(error) === 409 && station && quizId != null) {
        try {
          const latest = await getOsceStation(quizId, station.id)
          setConflictServer(latest)
          setCurrentVersion(latest.version)
          setConfirmOverwrite(false)
          return
        } catch (latestError) {
          messageApi.error(errorMessage(latestError))
          return
        }
      }
      messageApi.error(errorMessage(error))
    }
  }

  const handleSave = () => {
    if (overwriteVersion != null) {
      setConfirmOverwrite(true)
      return
    }
    void performSave()
  }

  const updateChecklist = (index: number, patch: Partial<OsceChecklistItemUpdate>) => {
    setDraft((current) => ({
      ...current,
      checklist_items: current.checklist_items.map((item, itemIndex) =>
        itemIndex === index ? { ...item, ...patch } : item
      )
    }))
  }

  const updateDomain = (index: number, patch: Partial<OsceRubricDomainUpdate>) => {
    setDraft((current) => ({
      ...current,
      rubric_domains: current.rubric_domains.map((domain, itemIndex) =>
        itemIndex === index ? { ...domain, ...patch } : domain
      )
    }))
  }

  const updateKeyPoint = (index: number, patch: Partial<OsceKeyPointUpdate>) => {
    setDraft((current) => ({
      ...current,
      expected_key_points: current.expected_key_points.map((point, itemIndex) =>
        itemIndex === index ? { ...point, ...patch } : point
      )
    }))
  }

  return (
    <section className="w-full min-w-0 space-y-6" aria-label="Station authoring">
      {contextHolder}
      <Alert
        type="warning"
        showIcon
        title="Use fictional or deidentified scenarios"
        description="Do not enter real patient information. Keep expected answers out of candidate-facing text."
      />

      {conflictServer ? (
        <Alert
          type="warning"
          showIcon
          title="This station changed on the server."
          description={`Server version ${conflictServer.version} was loaded for conflict recovery. Your local draft is unchanged.`}
          action={(
            <div className="flex flex-wrap gap-2">
              <Button
                onClick={() => {
                  const next = cloneDraft(conflictServer.content)
                  setDraft(next)
                  setAcknowledged(next)
                  setCurrentVersion(conflictServer.version)
                  setConflictServer(null)
                  setOverwriteVersion(null)
                }}
              >
                Reload server version
              </Button>
              <Button
                type="primary"
                onClick={() => {
                  setAcknowledged(cloneDraft(conflictServer.content))
                  setCurrentVersion(conflictServer.version)
                  setOverwriteVersion(conflictServer.version)
                  setConflictServer(null)
                }}
              >
                Keep local draft
              </Button>
            </div>
          )}
        />
      ) : null}

      {confirmOverwrite ? (
        <Alert
          type="warning"
          showIcon
          title="Confirm overwrite"
          description={`Save the local draft against server version ${overwriteVersion}?`}
          action={(
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => setConfirmOverwrite(false)}>Cancel</Button>
              <Button
                type="primary"
                danger
                onClick={() => {
                  setConfirmOverwrite(false)
                  void performSave(overwriteVersion ?? undefined)
                }}
              >
                Confirm overwrite
              </Button>
            </div>
          )}
        />
      ) : null}

      <div className="grid min-w-0 grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_14rem]">
        <label className="min-w-0 space-y-1 text-sm font-medium text-text">
          <span>Station title</span>
          <Input
            aria-label="Station title"
            value={draft.title}
            maxLength={200}
            onChange={(event) => setDraft((current) => ({ ...current, title: event.target.value }))}
          />
        </label>
        <label className="space-y-1 text-sm font-medium text-text">
          <span>Recommended duration</span>
          <div className="flex min-w-0">
            <InputNumber
              aria-label="Recommended duration in seconds"
              min={60}
              max={7200}
              step={30}
              className="min-w-0 flex-1"
              value={draft.recommended_duration_seconds}
              onChange={(value) => setDraft((current) => ({
                ...current,
                recommended_duration_seconds: Number(value) || 60
              }))}
            />
            <span className="inline-flex min-h-8 items-center border border-l-0 border-border bg-surface2 px-3 text-sm font-normal text-text-muted">
              seconds
            </span>
          </div>
        </label>
      </div>

      <div className="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
        <label className="min-w-0 space-y-1 text-sm font-medium text-text">
          <span>Candidate instructions</span>
          <Input.TextArea
            aria-label="Candidate instructions"
            value={draft.candidate_instructions}
            maxLength={4000}
            rows={5}
            onChange={(event) => setDraft((current) => ({ ...current, candidate_instructions: event.target.value }))}
          />
        </label>
        <label className="min-w-0 space-y-1 text-sm font-medium text-text">
          <span>Candidate task</span>
          <Input.TextArea
            aria-label="Candidate task"
            value={draft.candidate_task}
            maxLength={4000}
            rows={5}
            onChange={(event) => setDraft((current) => ({ ...current, candidate_task: event.target.value }))}
          />
        </label>
      </div>

      <div className="min-w-0 space-y-3 border-t border-border-subtle pt-5">
        <label className="block space-y-1 text-sm font-medium text-text">
          <span>Patient context</span>
          <Input.TextArea
            aria-label="Patient context"
            value={draft.patient_context.text}
            maxLength={10000}
            rows={6}
            onChange={(event) => setDraft((current) => ({
              ...current,
              patient_context: { ...current.patient_context, text: event.target.value }
            }))}
          />
        </label>
        <CitationEditor
          label="Patient context citation"
          citations={draft.patient_context.citations}
          onChange={(citations) => setDraft((current) => ({
            ...current,
            patient_context: { ...current.patient_context, citations }
          }))}
        />
      </div>

      <div className="min-w-0 space-y-4 border-t border-border-subtle pt-5">
        <div className="flex min-h-10 flex-wrap items-center justify-between gap-2">
          <Typography.Title level={4} className="!mb-0 !text-base">Checklist items</Typography.Title>
          <Button
            type="dashed"
            icon={<PlusOutlined aria-hidden />}
            disabled={draft.checklist_items.length >= 50}
            onClick={() => setDraft((current) => ({
              ...current,
              checklist_items: [...current.checklist_items, { label: "", rationale: "", citations: [] }]
            }))}
          >
            Add checklist item
          </Button>
        </div>
        {draft.checklist_items.map((item, index) => (
          <div key={item.id ?? `checklist-${index}`} className="min-w-0 space-y-3 border-t border-border-subtle pt-4">
            <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start">
              <div className="min-w-0 flex-1 space-y-2">
                <Input
                  aria-label={`Checklist item ${index + 1}`}
                  value={item.label}
                  maxLength={1000}
                  placeholder={`Checklist item ${index + 1}`}
                  onChange={(event) => updateChecklist(index, { label: event.target.value })}
                />
                <Input.TextArea
                  aria-label={`Checklist item ${index + 1} rationale`}
                  value={item.rationale ?? ""}
                  maxLength={2000}
                  rows={3}
                  placeholder="Rationale"
                  onChange={(event) => updateChecklist(index, { rationale: event.target.value })}
                />
              </div>
              <OrderButtons
                label={`checklist item ${index + 1}`}
                index={index}
                count={draft.checklist_items.length}
                removeDisabled={draft.checklist_items.length === 1}
                onMove={(to) => setDraft((current) => ({
                  ...current,
                  checklist_items: move(current.checklist_items, index, to)
                }))}
                onRemove={() => setDraft((current) => ({
                  ...current,
                  checklist_items: current.checklist_items.filter((_, itemIndex) => itemIndex !== index)
                }))}
              />
            </div>
            <CitationEditor
              label={`Checklist item ${index + 1} citation`}
              citations={item.citations}
              onChange={(citations) => updateChecklist(index, { citations })}
            />
          </div>
        ))}
      </div>

      <div className="min-w-0 space-y-4 border-t border-border-subtle pt-5">
        <div className="flex min-h-10 flex-wrap items-center justify-between gap-2">
          <Typography.Title level={4} className="!mb-0 !text-base">Rubric domains</Typography.Title>
          <Button
            type="dashed"
            icon={<PlusOutlined aria-hidden />}
            disabled={draft.rubric_domains.length >= 12}
            onClick={() => setDraft((current) => ({
              ...current,
              rubric_domains: [...current.rubric_domains, {
                label: "",
                levels: [
                  { label: "Needs development", description: "" },
                  { label: "Effective", description: "" }
                ]
              }]
            }))}
          >
            Add rubric domain
          </Button>
        </div>
        {draft.rubric_domains.map((domain, domainIndex) => (
          <div key={domain.id ?? `domain-${domainIndex}`} className="min-w-0 space-y-3 border-t border-border-subtle pt-4">
            <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start">
              <Input
                aria-label={`Rubric domain ${domainIndex + 1}`}
                value={domain.label}
                maxLength={200}
                placeholder={`Rubric domain ${domainIndex + 1}`}
                onChange={(event) => updateDomain(domainIndex, { label: event.target.value })}
                className="min-w-0 flex-1"
              />
              <OrderButtons
                label={`rubric domain ${domainIndex + 1}`}
                index={domainIndex}
                count={draft.rubric_domains.length}
                removeDisabled={draft.rubric_domains.length === 1}
                onMove={(to) => setDraft((current) => ({
                  ...current,
                  rubric_domains: move(current.rubric_domains, domainIndex, to)
                }))}
                onRemove={() => setDraft((current) => ({
                  ...current,
                  rubric_domains: current.rubric_domains.filter((_, itemIndex) => itemIndex !== domainIndex)
                }))}
              />
            </div>
            <div className="space-y-3 pl-0 sm:pl-4">
              {domain.levels.map((level, levelIndex) => (
                <div
                  key={level.id ?? `level-${domainIndex}-${levelIndex}`}
                  className="grid min-w-0 grid-cols-1 gap-2 sm:grid-cols-[minmax(9rem,0.35fr)_minmax(0,1fr)_8.75rem] sm:items-start"
                >
                  <Input
                    aria-label={`Rubric level ${levelIndex + 1} label in domain ${domainIndex + 1}`}
                    value={level.label}
                    maxLength={200}
                    placeholder="Level label"
                    onChange={(event) => updateDomain(domainIndex, {
                      levels: domain.levels.map((item, itemIndex) => itemIndex === levelIndex
                        ? { ...item, label: event.target.value }
                        : item)
                    })}
                  />
                  <Input.TextArea
                    aria-label={`Rubric level ${levelIndex + 1} description in domain ${domainIndex + 1}`}
                    value={level.description}
                    maxLength={2000}
                    autoSize={{ minRows: 1, maxRows: 4 }}
                    placeholder="Observable performance description"
                    onChange={(event) => updateDomain(domainIndex, {
                      levels: domain.levels.map((item, itemIndex) => itemIndex === levelIndex
                        ? { ...item, description: event.target.value }
                        : item)
                    })}
                  />
                  <OrderButtons
                    label={`rubric level ${levelIndex + 1} from domain ${domainIndex + 1}`}
                    index={levelIndex}
                    count={domain.levels.length}
                    removeDisabled={domain.levels.length <= 2}
                    onMove={(to) => updateDomain(domainIndex, { levels: move(domain.levels, levelIndex, to) })}
                    onRemove={() => updateDomain(domainIndex, {
                      levels: domain.levels.filter((_, itemIndex) => itemIndex !== levelIndex)
                    })}
                  />
                </div>
              ))}
              <Button
                size="small"
                type="dashed"
                icon={<PlusOutlined aria-hidden />}
                disabled={domain.levels.length >= 6}
                onClick={() => updateDomain(domainIndex, {
                  levels: [...domain.levels, { label: "", description: "" }]
                })}
              >
                Add rubric level
              </Button>
            </div>
          </div>
        ))}
      </div>

      <div className="min-w-0 space-y-4 border-t border-border-subtle pt-5">
        <div className="flex min-h-10 flex-wrap items-center justify-between gap-2">
          <Typography.Title level={4} className="!mb-0 !text-base">Expected key points</Typography.Title>
          <Button
            type="dashed"
            icon={<PlusOutlined aria-hidden />}
            disabled={draft.expected_key_points.length >= 50}
            onClick={() => setDraft((current) => ({
              ...current,
              expected_key_points: [...current.expected_key_points, { text: "", citations: [] }]
            }))}
          >
            Add key point
          </Button>
        </div>
        {draft.expected_key_points.map((point, index) => (
          <div key={point.id ?? `point-${index}`} className="min-w-0 space-y-3 border-t border-border-subtle pt-4">
            <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start">
              <Input.TextArea
                aria-label={`Expected key point ${index + 1}`}
                value={point.text}
                maxLength={2000}
                rows={3}
                placeholder={`Expected key point ${index + 1}`}
                onChange={(event) => updateKeyPoint(index, { text: event.target.value })}
                className="min-w-0 flex-1"
              />
              <OrderButtons
                label={`expected key point ${index + 1}`}
                index={index}
                count={draft.expected_key_points.length}
                removeDisabled={draft.expected_key_points.length === 1}
                onMove={(to) => setDraft((current) => ({
                  ...current,
                  expected_key_points: move(current.expected_key_points, index, to)
                }))}
                onRemove={() => setDraft((current) => ({
                  ...current,
                  expected_key_points: current.expected_key_points.filter((_, itemIndex) => itemIndex !== index)
                }))}
              />
            </div>
            <CitationEditor
              label={`Expected key point ${index + 1} citation`}
              citations={point.citations}
              onChange={(citations) => updateKeyPoint(index, { citations })}
            />
          </div>
        ))}
      </div>

      <div className="min-w-0 space-y-3 border-t border-border-subtle pt-5">
        <Typography.Title level={4} className="!mb-0 !text-base">Candidate preview</Typography.Title>
        <div className="grid min-w-0 grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="min-w-0">
            <Typography.Text strong>Instructions</Typography.Text>
            <QuizMarkdown content={draft.candidate_instructions || "No instructions yet."} />
          </div>
          <div className="min-w-0">
            <Typography.Text strong>Task</Typography.Text>
            <QuizMarkdown content={draft.candidate_task || "No task yet."} />
          </div>
        </div>
        <div className="min-w-0">
          <Typography.Text strong>Patient context</Typography.Text>
          <QuizMarkdown content={draft.patient_context.text || "No patient context yet."} />
          <SourceCitations citations={toQuizSourceCitations(draft.patient_context.citations)} />
        </div>
      </div>

      {validationErrors.length > 0 && dirty ? (
        <Alert
          type="error"
          showIcon
          title="Complete required station fields"
          description={<ul className="list-disc pl-5">{validationErrors.map((error) => <li key={error}>{error}</li>)}</ul>}
        />
      ) : null}

      <div className="sticky bottom-0 z-10 flex min-h-16 flex-wrap items-center justify-between gap-3 border-t border-border bg-surface py-3">
        <Typography.Text type="secondary">
          {dirty ? "Unsaved changes" : "All changes saved"}
        </Typography.Text>
        <Button
          type="primary"
          size="large"
          icon={<SaveOutlined aria-hidden />}
          aria-label="Save station"
          loading={saving}
          disabled={!dirty || validationErrors.length > 0 || saving || conflictServer != null}
          onClick={handleSave}
        >
          Save station
        </Button>
      </div>
    </section>
  )
}

export default OsceStationEditor
