import React from "react"
import {
  Button,
  Card,
  Input,
  Modal,
  Radio,
  Skeleton,
  Space,
  Tag,
  Typography
} from "antd"
import { ClockCircleOutlined } from "@ant-design/icons"
import { useTranslation } from "react-i18next"

import { Alert as DesignSystemAlert, Badge as DesignSystemBadge } from "@/components/ui/primitives"
import { useServerOnline } from "@/hooks/useServerOnline"
import type {
  OsceAttempt,
  OsceCitation
} from "@/services/osce"
import type { SourceCitation } from "@/services/quizzes"
import { QuizMarkdown } from "../components/QuizMarkdown"
import { SourceCitations } from "../components/SourceCitations"
import {
  useBeginOsceSelfAssessmentMutation,
  useCompleteOsceAttemptMutation,
  useOsceAttemptQuery,
  usePatchOsceAttemptMutation
} from "../hooks/useOsceQueries"
import {
  clearOsceDraft,
  createOsceSaveQueue,
  readOsceDraft,
  type OsceSaveQueue,
  type OsceWritablePatch
} from "./osceDraftStore"

type ElapsedInput = {
  startedAt: string
  serverNow: string
  selfAssessmentStartedAt?: string | null
  frozenElapsedSeconds?: number | null
}

type LiveElapsedInput = {
  serverElapsedSeconds: number
  baselineMonotonicMs: number
  monotonicNowMs: number
  frozenElapsedSeconds?: number | null
}

const parseTime = (value: string): number | null => {
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : null
}

export const computeElapsedSeconds = ({
  startedAt,
  serverNow,
  selfAssessmentStartedAt,
  frozenElapsedSeconds
}: ElapsedInput): number => {
  if (typeof frozenElapsedSeconds === "number" && Number.isFinite(frozenElapsedSeconds)) {
    return Math.max(0, Math.floor(frozenElapsedSeconds))
  }
  const started = parseTime(startedAt)
  const ended = parseTime(selfAssessmentStartedAt || serverNow)
  if (started == null || ended == null) return 0
  return Math.max(0, Math.floor((ended - started) / 1_000))
}

export const computeLiveElapsedSeconds = ({
  serverElapsedSeconds,
  baselineMonotonicMs,
  monotonicNowMs,
  frozenElapsedSeconds
}: LiveElapsedInput): number => {
  if (typeof frozenElapsedSeconds === "number" && Number.isFinite(frozenElapsedSeconds)) {
    return Math.max(0, Math.floor(frozenElapsedSeconds))
  }
  const monotonicDelta = Math.max(0, monotonicNowMs - baselineMonotonicMs)
  return Math.max(0, Math.floor(serverElapsedSeconds + monotonicDelta / 1_000))
}

const monotonicNow = (): number => {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now()
  }
  return 0
}

const formatElapsed = (seconds: number): string => {
  const hours = Math.floor(seconds / 3_600)
  const minutes = Math.floor((seconds % 3_600) / 60)
  const remainder = seconds % 60
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`
}

const prepareErrorAlert = (node: HTMLDivElement | null): void => {
  if (node) node.setAttribute("tabindex", "-1")
}

const toSourceCitations = (citations: OsceCitation[]): SourceCitation[] => citations.map((citation) => ({
  source_type: citation.source_type === "media" || citation.source_type === "note"
    ? citation.source_type
    : null,
  source_id: citation.source_id,
  label: citation.label,
  quote: citation.quote,
  media_id: citation.media_id,
  chunk_id: citation.chunk_id,
  timestamp_seconds: citation.timestamp_seconds,
  source_url: citation.source_url
}))

const mergeDraftIntoAttempt = (
  attempt: OsceAttempt,
  draft: ReturnType<typeof readOsceDraft>
): OsceAttempt => {
  if (!draft) return attempt
  if (attempt.state === "in_progress") {
    return { ...attempt, notes: draft.notes }
  }
  return {
    ...attempt,
    notes: draft.notes,
    checklist_selections: draft.checklistSelections,
    rubric_selections: draft.rubricSelections
  }
}

type SaveStatus = "idle" | "saving" | "saved" | "offline" | "conflict" | "error"

export interface OscePracticePanelProps {
  attemptId: number
  userScope: string | number
  saveDebounceMs?: number
  onClose?: () => void
  onCompleted?: (attempt: OsceAttempt) => void
}

export const OscePracticePanel: React.FC<OscePracticePanelProps> = ({
  attemptId,
  userScope,
  saveDebounceMs = 500,
  onClose,
  onCompleted
}) => {
  const { t } = useTranslation(["option", "common"])
  const isOnline = useServerOnline()
  const attemptQuery = useOsceAttemptQuery(attemptId)
  const patchMutation = usePatchOsceAttemptMutation()
  const beginMutation = useBeginOsceSelfAssessmentMutation()
  const completeMutation = useCompleteOsceAttemptMutation()
  const [attempt, setAttempt] = React.useState<OsceAttempt | null>(null)
  const [saveStatus, setSaveStatus] = React.useState<SaveStatus>("idle")
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [revealConfirmationOpen, setRevealConfirmationOpen] = React.useState(false)
  const [, setVisualTick] = React.useState(0)
  const queueRef = React.useRef<OsceSaveQueue | null>(null)
  const queueKeyRef = React.useRef<string | null>(null)
  const patchMutationRef = React.useRef(patchMutation)
  const pendingNotesRef = React.useRef<string | null>(null)
  const notesTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const outerErrorRef = React.useRef<HTMLDivElement | null>(null)
  const modalErrorRef = React.useRef<HTMLDivElement | null>(null)
  const confirmButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const wasOnlineRef = React.useRef(isOnline)
  const reportSaveErrorRef = React.useRef<(error: unknown, focus?: boolean) => void>(() => undefined)

  patchMutationRef.current = patchMutation

  const describeSaveError = React.useCallback((error: unknown): string => {
    const status = Number((error as { status?: unknown } | null)?.status)
    if (status === 409) {
      return t("option:quiz.osceSaveConflict", {
        defaultValue: "This practice changed on the server. Your local draft was kept."
      })
    }
    if (status === 0 || !isOnline) {
      return t("option:quiz.osceSaveOffline", {
        defaultValue: "The practice could not be saved while offline. Your local draft was kept."
      })
    }
    return t("option:quiz.osceSaveError", {
      defaultValue: "The practice could not be saved. Your local draft was kept."
    })
  }, [isOnline, t])

  const reportSaveError = React.useCallback((error: unknown, focus = false) => {
    const status = Number((error as { status?: unknown } | null)?.status)
    setSaveStatus(status === 409 ? "conflict" : status === 0 || !isOnline ? "offline" : "error")
    setSaveError(describeSaveError(error))
    if (focus) {
      window.setTimeout(() => (modalErrorRef.current ?? outerErrorRef.current)?.focus(), 0)
    }
  }, [describeSaveError, isOnline])

  reportSaveErrorRef.current = reportSaveError

  const setOuterErrorAlertRef = React.useCallback((node: HTMLDivElement | null) => {
    outerErrorRef.current = node
    prepareErrorAlert(node)
  }, [])

  const setModalErrorAlertRef = React.useCallback((node: HTMLDivElement | null) => {
    modalErrorRef.current = node
    prepareErrorAlert(node)
  }, [])

  React.useEffect(() => {
    if (!revealConfirmationOpen) return
    const focusTimer = window.setTimeout(() => {
      if (modalErrorRef.current) {
        modalErrorRef.current.focus()
        return
      }
      confirmButtonRef.current?.focus()
    }, 0)
    return () => window.clearTimeout(focusTimer)
  }, [revealConfirmationOpen])

  React.useEffect(() => {
    const serverAttempt = attemptQuery.data
    if (!serverAttempt) return
    const queueKey = `${userScope}:${serverAttempt.id}`
    if (queueKeyRef.current !== queueKey) {
      const draft = readOsceDraft(userScope, serverAttempt.id)
      setAttempt(mergeDraftIntoAttempt(serverAttempt, draft))
      queueRef.current = createOsceSaveQueue({
        attempt: serverAttempt,
        initialExpectedVersion: draft?.version,
        userScope,
        update: (id, patch) => patchMutationRef.current.mutateAsync({ attemptId: id, patch }),
        onAcknowledged: (updated, isCurrentRevision) => {
          if (!isCurrentRevision) return
          setAttempt(updated)
          setSaveStatus("saved")
          setSaveError(null)
        }
      })
      queueKeyRef.current = queueKey
      if (draft && isOnline) {
        setSaveStatus("saving")
        void queueRef.current.enqueue({
          notes: draft.notes,
          checklist_selections: draft.checklistSelections,
          rubric_selections: draft.rubricSelections
        }).catch((error) => reportSaveErrorRef.current(error))
      } else if (draft) {
        setSaveStatus("offline")
      }
      return
    }

    queueRef.current?.replaceAcknowledgedAttempt(serverAttempt)
    if (!queueRef.current?.hasPending()) setAttempt(serverAttempt)
  }, [attemptQuery.data, isOnline, userScope])

  React.useEffect(() => {
    const becameOnline = isOnline && !wasOnlineRef.current
    wasOnlineRef.current = isOnline
    if (!becameOnline || !attempt || !queueRef.current) return
    const draft = readOsceDraft(userScope, attempt.id)
    if (!draft) return
    setSaveStatus("saving")
    void queueRef.current.enqueue({
      notes: draft.notes,
      checklist_selections: draft.checklistSelections,
      rubric_selections: draft.rubricSelections
    }).catch((error) => reportSaveErrorRef.current(error))
  }, [attempt, isOnline, userScope])

  React.useEffect(() => () => {
    if (notesTimerRef.current) clearTimeout(notesTimerRef.current)
  }, [])

  const enqueuePatch = React.useCallback(async (patch: OsceWritablePatch) => {
    const queue = queueRef.current
    if (!queue) throw new Error("Practice save queue is not ready.")
    setSaveStatus(isOnline ? "saving" : "offline")
    setSaveError(null)
    try {
      return await queue.enqueue(patch)
    } catch (error) {
      reportSaveError(error)
      throw error
    }
  }, [isOnline, reportSaveError])

  const queueNotes = React.useCallback((notes: string) => {
    const queue = queueRef.current
    if (!queue) return
    queue.stage({ notes })
    pendingNotesRef.current = notes
    if (notesTimerRef.current) clearTimeout(notesTimerRef.current)
    notesTimerRef.current = setTimeout(() => {
      notesTimerRef.current = null
      const pendingNotes = pendingNotesRef.current
      pendingNotesRef.current = null
      if (pendingNotes == null) return
      setSaveStatus(isOnline ? "saving" : "offline")
      void queue.enqueueStaged().catch(reportSaveError)
    }, Math.max(0, saveDebounceMs))
  }, [isOnline, reportSaveError, saveDebounceMs])

  const flushPendingWrites = React.useCallback(async () => {
    if (notesTimerRef.current) {
      clearTimeout(notesTimerRef.current)
      notesTimerRef.current = null
    }
    if (pendingNotesRef.current != null) {
      pendingNotesRef.current = null
      setSaveStatus(isOnline ? "saving" : "offline")
      await queueRef.current?.enqueueStaged()
    }
    const queue = queueRef.current
    if (!queue) throw new Error("Practice save queue is not ready.")
    await queue.flush()
  }, [enqueuePatch])

  const handleBeginSelfAssessment = async () => {
    if (!attempt || !isOnline) return
    try {
      await flushPendingWrites()
      const queue = queueRef.current
      if (!queue) return
      setRevealConfirmationOpen(false)
      const updated = await beginMutation.mutateAsync({
        attemptId: attempt.id,
        expectedVersion: queue.getAcknowledgedVersion()
      })
      queue.replaceAcknowledgedAttempt(updated)
      clearOsceDraft(userScope, attempt.id)
      setAttempt(updated)
      setSaveStatus("saved")
      setSaveError(null)
    } catch (error) {
      reportSaveError(error, true)
    }
  }

  const handleComplete = async () => {
    if (!attempt || attempt.state === "in_progress" || !isOnline) return
    try {
      await flushPendingWrites()
      const queue = queueRef.current
      if (!queue) return
      const updated = await completeMutation.mutateAsync({
        attemptId: attempt.id,
        expectedVersion: queue.getAcknowledgedVersion()
      })
      queue.replaceAcknowledgedAttempt(updated)
      clearOsceDraft(userScope, attempt.id)
      setAttempt(updated)
      setSaveStatus("saved")
      setSaveError(null)
      onCompleted?.(updated)
    } catch (error) {
      reportSaveError(error, true)
    }
  }

  const timerBaseline = React.useMemo(() => {
    if (!attempt) return null
    return {
      serverElapsedSeconds: computeElapsedSeconds({
        startedAt: attempt.started_at,
        serverNow: attempt.server_time,
        selfAssessmentStartedAt: attempt.state === "in_progress"
          ? null
          : attempt.self_assessment_started_at,
        frozenElapsedSeconds: attempt.state === "in_progress" ? null : attempt.elapsed_seconds
      }),
      baselineMonotonicMs: monotonicNow(),
      frozenElapsedSeconds: attempt.state === "in_progress" ? null : attempt.elapsed_seconds
    }
  }, [attempt?.id, attempt?.server_time, attempt?.state, attempt?.version])

  React.useEffect(() => {
    if (!attempt || attempt.state !== "in_progress") return
    const interval = window.setInterval(() => setVisualTick((value) => value + 1), 1_000)
    const handleVisibility = () => {
      if (document.visibilityState === "visible") void attemptQuery.refetch()
      setVisualTick((value) => value + 1)
    }
    document.addEventListener("visibilitychange", handleVisibility)
    return () => {
      window.clearInterval(interval)
      document.removeEventListener("visibilitychange", handleVisibility)
    }
  }, [attempt?.id, attempt?.state, attemptQuery.refetch])

  if (attemptQuery.isError) {
    return (
      <DesignSystemAlert variant="error" title={t("option:quiz.osceLoadError", {
        defaultValue: "This practice attempt could not be loaded."
      })} />
    )
  }

  if (attemptQuery.isLoading || !attempt) {
    return <Card><Skeleton active paragraph={{ rows: 6 }} /></Card>
  }

  const elapsedSeconds = timerBaseline
    ? computeLiveElapsedSeconds({ ...timerBaseline, monotonicNowMs: monotonicNow() })
    : 0
  const isRevealed = attempt.state !== "in_progress"
  const checklistItems = isRevealed ? attempt.station.checklist_items : []
  const rubricDomains = isRevealed ? attempt.station.rubric_domains : []
  const canComplete = isRevealed &&
    checklistItems.every((item) => attempt.checklist_selections[item.id] !== undefined) &&
    rubricDomains.every((domain) => attempt.rubric_selections[domain.id] !== undefined)
  const statusLabel = !isOnline
    ? t("option:quiz.osceOffline", { defaultValue: "Offline, draft kept locally" })
    : saveStatus === "saving"
      ? t("option:quiz.osceSaving", { defaultValue: "Saving" })
      : saveStatus === "saved"
        ? t("option:quiz.osceSaved", { defaultValue: "Saved" })
        : t("option:quiz.osceReady", { defaultValue: "Ready" })

  return (
    <div className="space-y-4" data-testid="osce-practice-panel">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <Typography.Title level={4} className="!mb-1 break-words">
            {attempt.station.title}
          </Typography.Title>
          <div className="flex flex-wrap gap-2">
            <DesignSystemBadge variant={isRevealed ? "info" : "primary"}>
              {isRevealed
                ? t("option:quiz.osceSelfAssessment", { defaultValue: "Self-assessment" })
                : t("option:quiz.osceCandidatePhase", { defaultValue: "Candidate phase" })}
            </DesignSystemBadge>
            <Tag icon={<ClockCircleOutlined />} aria-label={`Elapsed time ${formatElapsed(elapsedSeconds)}`}>
              {formatElapsed(elapsedSeconds)}
            </Tag>
            <Tag>{Math.round(attempt.station.recommended_duration_seconds / 60)} min recommended</Tag>
          </div>
        </div>
        {onClose && (
          <Button onClick={onClose} className="min-h-11 shrink-0">
            {t("option:quiz.backToStations", { defaultValue: "Back to stations" })}
          </Button>
        )}
      </div>

      <section className="space-y-4" aria-label="Candidate context">
        <div>
          <Typography.Text strong>{t("option:quiz.osceTask", { defaultValue: "Task" })}</Typography.Text>
          <QuizMarkdown content={attempt.station.candidate_task} className="max-w-[72ch]" />
        </div>
        <div>
          <Typography.Text strong>{t("option:quiz.osceInstructions", { defaultValue: "Instructions" })}</Typography.Text>
          <QuizMarkdown content={attempt.station.candidate_instructions} className="max-w-[72ch]" />
        </div>
        <div>
          <Typography.Text strong>{t("option:quiz.oscePatientContext", { defaultValue: "Patient context" })}</Typography.Text>
          <QuizMarkdown content={attempt.station.patient_context.text} className="max-w-[72ch]" />
          <SourceCitations citations={toSourceCitations(attempt.station.patient_context.citations as OsceCitation[])} />
        </div>
      </section>

      <section className="space-y-2" aria-label="Private notes">
        <label htmlFor={`osce-notes-${attempt.id}`} className="block font-medium text-text">
          {t("option:quiz.oscePrivateNotes", { defaultValue: "Private practice notes" })}
        </label>
        <Input.TextArea
          id={`osce-notes-${attempt.id}`}
          aria-label="Private practice notes"
          value={attempt.notes}
          maxLength={10_000}
          autoSize={{ minRows: 4, maxRows: 10 }}
          onChange={(event) => {
            const notes = event.target.value
            setAttempt((current) => current ? { ...current, notes } : current)
            queueNotes(notes)
          }}
        />
        <Typography.Text className="block text-xs text-text-muted">
          {t("option:quiz.oscePatientPrivacy", {
            defaultValue: "Do not enter real patient information. Notes are private to this practice attempt."
          })}
        </Typography.Text>
      </section>

      <div role="status" aria-live="polite" className="text-xs text-text-muted">
        {statusLabel}
      </div>

      {saveError && !revealConfirmationOpen && (
        <DesignSystemAlert
          ref={setOuterErrorAlertRef}
          variant={saveStatus === "conflict" ? "warning" : "error"}
          title={saveError}
        />
      )}

      {isRevealed && (
        <section className="space-y-5" aria-label="Self-assessment marking guide">
          <div className="space-y-3">
            <Typography.Title level={5} className="!mb-0">
              {t("option:quiz.osceExpectedPoints", { defaultValue: "Expected points" })}
            </Typography.Title>
            {attempt.station.expected_key_points.map((point) => (
              <div key={point.id} className="border-b border-border pb-3 last:border-b-0">
                <QuizMarkdown content={point.text} className="max-w-[72ch]" />
                <SourceCitations citations={toSourceCitations(point.citations)} />
              </div>
            ))}
          </div>

          <div className="space-y-3">
            <Typography.Title level={5} className="!mb-0">
              {t("option:quiz.osceChecklist", { defaultValue: "Checklist" })}
            </Typography.Title>
            {checklistItems.map((item) => (
              <fieldset key={item.id} className="space-y-2 border-b border-border pb-3 last:border-b-0">
                <legend className="font-medium text-text">{item.label}</legend>
                {item.rationale && <QuizMarkdown content={item.rationale} className="max-w-[72ch] text-sm" />}
                <Radio.Group
                  aria-label={item.label}
                  value={attempt.checklist_selections[item.id]}
                  onChange={(event) => {
                    const checklistSelections = {
                      ...attempt.checklist_selections,
                      [item.id]: event.target.value
                    }
                    setAttempt({ ...attempt, checklist_selections: checklistSelections })
                    void enqueuePatch({ checklist_selections: checklistSelections }).catch(() => undefined)
                  }}
                >
                  <Space wrap>
                    <Radio value="met">{t("option:quiz.osceMet", { defaultValue: "Met" })}</Radio>
                    <Radio value="not_met">{t("option:quiz.osceNotMet", { defaultValue: "Not met" })}</Radio>
                  </Space>
                </Radio.Group>
                <SourceCitations citations={toSourceCitations(item.citations)} />
              </fieldset>
            ))}
          </div>

          <div className="space-y-3">
            <Typography.Title level={5} className="!mb-0">
              {t("option:quiz.osceRubric", { defaultValue: "Rubric" })}
            </Typography.Title>
            {rubricDomains.map((domain) => (
              <fieldset key={domain.id} className="space-y-2 border-b border-border pb-3 last:border-b-0">
                <legend className="font-medium text-text">{domain.label}</legend>
                <Radio.Group
                  aria-label={domain.label}
                  value={attempt.rubric_selections[domain.id]}
                  onChange={(event) => {
                    const rubricSelections = {
                      ...attempt.rubric_selections,
                      [domain.id]: event.target.value
                    }
                    setAttempt({ ...attempt, rubric_selections: rubricSelections })
                    void enqueuePatch({ rubric_selections: rubricSelections }).catch(() => undefined)
                  }}
                  className="w-full"
                >
                  <div className="grid gap-2 sm:grid-cols-2">
                    {domain.levels.map((level) => (
                      <Radio key={level.id} value={level.id} className="!m-0 min-h-11 py-2">
                        <span className="font-medium">{level.label}</span>
                        <span className="block text-sm text-text-muted">{level.description}</span>
                      </Radio>
                    ))}
                  </div>
                </Radio.Group>
              </fieldset>
            ))}
          </div>
        </section>
      )}

      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        {!isRevealed ? (
          <Button
            type="primary"
            className="min-h-11 whitespace-normal"
            disabled={!isOnline || beginMutation.isPending}
            loading={beginMutation.isPending}
            onClick={() => {
              setSaveError(null)
              setRevealConfirmationOpen(true)
            }}
          >
            {t("option:quiz.osceBeginAssessment", { defaultValue: "Begin self-assessment" })}
          </Button>
        ) : attempt.state !== "completed" ? (
          <Button
            type="primary"
            className="min-h-11 whitespace-normal"
            disabled={!canComplete || !isOnline || completeMutation.isPending}
            loading={completeMutation.isPending}
            onClick={() => void handleComplete()}
          >
            {t("option:quiz.osceComplete", { defaultValue: "Complete practice" })}
          </Button>
        ) : (
          <DesignSystemBadge variant="success">
            {t("option:quiz.osceCompleted", { defaultValue: "Practice completed" })}
          </DesignSystemBadge>
        )}
      </div>

      <Modal
        title={t("option:quiz.osceRevealTitle", { defaultValue: "Reveal marking guide?" })}
        open={revealConfirmationOpen}
        onCancel={() => setRevealConfirmationOpen(false)}
        footer={(
          <Space wrap>
            <Button onClick={() => setRevealConfirmationOpen(false)}>
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
            <Button
              ref={confirmButtonRef}
              type="primary"
              onClick={() => void handleBeginSelfAssessment()}
            >
              {t("option:quiz.osceReveal", { defaultValue: "Reveal marking guide" })}
            </Button>
          </Space>
        )}
        afterOpenChange={(open) => {
          window.setTimeout(() => {
            if (open) {
              if (modalErrorRef.current) modalErrorRef.current.focus()
              else confirmButtonRef.current?.focus()
              return
            }
            outerErrorRef.current?.focus()
          }, 0)
        }}
        focusable={{ focusTriggerAfterClose: false, trap: revealConfirmationOpen }}
        destroyOnHidden
      >
        <div className="space-y-3">
          <Typography.Text>
            {t("option:quiz.osceRevealWarning", {
              defaultValue: "The timer will freeze and the marking guide will become visible. This cannot be undone."
            })}
          </Typography.Text>
          {saveError && (
            <DesignSystemAlert
              ref={setModalErrorAlertRef}
              variant={saveStatus === "conflict" ? "warning" : "error"}
              title={saveError}
            />
          )}
        </div>
      </Modal>
    </div>
  )
}

export default OscePracticePanel
