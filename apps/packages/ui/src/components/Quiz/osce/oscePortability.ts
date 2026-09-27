import type {
  Question,
  Quiz,
  QuizImportQuestion,
  QuizImportV2OsceStation
} from "@/services/quizzes"
import type { OsceStationAuthoringResponse } from "@/services/osce"

export type QuestionQuizExportEntry = {
  activity_type: "questions"
  quiz: Partial<Quiz> & Pick<Quiz, "name">
  questions: Array<Partial<Question> & QuizImportQuestion>
}

export type OsceQuizExportEntry = {
  activity_type: "osce"
  quiz: Partial<Quiz> & Pick<Quiz, "name">
  stations: Array<OsceStationAuthoringResponse & Record<string, unknown>>
}

export type QuizExportEntry = QuestionQuizExportEntry | OsceQuizExportEntry

const OMIT_PROVENANCE_KEY = /(api.?key|secret|token|password|credential|candidate.?notes?|attempts?)/i

const sanitizeInformationalValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sanitizeInformationalValue)
  if (!value || typeof value !== "object") return value

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !OMIT_PROVENANCE_KEY.test(key))
      .map(([key, nested]) => [key, sanitizeInformationalValue(nested)])
  )
}

const exportQuizMetadata = (quiz: QuestionQuizExportEntry["quiz"] | OsceQuizExportEntry["quiz"]) => ({
  id: quiz.id,
  name: quiz.name,
  description: quiz.description ?? null,
  workspace_id: quiz.workspace_id ?? null,
  workspace_tag: quiz.workspace_tag ?? null,
  media_id: quiz.media_id ?? null,
  source_bundle_json: quiz.source_bundle_json ?? null,
  activity_type: quiz.activity_type,
  generation_profile: quiz.generation_profile ?? null,
  total_questions: quiz.total_questions,
  total_stations: quiz.total_stations,
  time_limit_seconds: quiz.time_limit_seconds ?? null,
  passing_score: quiz.passing_score ?? null,
  version: quiz.version,
  created_at: quiz.created_at ?? null,
  last_modified: quiz.last_modified ?? null
})

const exportQuestion = (question: QuestionQuizExportEntry["questions"][number]) => ({
  id: question.id,
  question_type: question.question_type,
  question_text: question.question_text,
  options: question.options ?? null,
  group_id: question.group_id ?? null,
  group_prompt: question.group_prompt ?? null,
  correct_answer: question.correct_answer,
  explanation: question.explanation ?? null,
  hint: question.hint ?? null,
  hint_penalty_points: question.hint_penalty_points ?? 0,
  source_citations: question.source_citations ?? null,
  points: question.points ?? 1,
  order_index: question.order_index ?? 0,
  tags: question.tags ?? null
})

const exportStation = (
  station: OsceQuizExportEntry["stations"][number]
): QuizImportV2OsceStation => ({
  id: station.id,
  quiz_id: station.quiz_id,
  content: station.content,
  order_index: station.order_index,
  version: station.version,
  origin: station.origin,
  provenance: sanitizeInformationalValue(station.provenance) as Record<string, unknown> | null,
  source_bundle: sanitizeInformationalValue(station.source_bundle) as QuizImportV2OsceStation["source_bundle"],
  verification_state: station.verification_state,
  verification_timestamp: station.verification_timestamp,
  verification_summary: station.verification_summary,
  created_at: station.created_at,
  updated_at: station.updated_at
})

export const buildQuizExport = (
  entries: QuizExportEntry[],
  exportedAt = new Date().toISOString()
) => {
  const containsOsce = entries.some((entry) => entry.activity_type === "osce")

  if (!containsOsce) {
    return {
      export_format: "tldw.quiz.export.v1" as const,
      exported_at: exportedAt,
      source: "quiz-manage-tab",
      quiz_count: entries.length,
      quizzes: (entries as QuestionQuizExportEntry[]).map((entry) => ({
        quiz: exportQuizMetadata(entry.quiz),
        questions: entry.questions.map(exportQuestion)
      }))
    }
  }

  return {
    export_format: "tldw.quiz.export.v2" as const,
    exported_at: exportedAt,
    quizzes: entries.map((entry) => entry.activity_type === "osce"
      ? {
          activity_type: "osce" as const,
          quiz: { ...exportQuizMetadata(entry.quiz), activity_type: "osce" as const },
          stations: entry.stations.map(exportStation)
        }
      : {
          activity_type: "questions" as const,
          quiz: { ...exportQuizMetadata(entry.quiz), activity_type: "questions" as const },
          questions: entry.questions.map(exportQuestion)
        })
  }
}
