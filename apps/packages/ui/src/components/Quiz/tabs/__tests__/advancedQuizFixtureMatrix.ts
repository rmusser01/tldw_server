import { readFileSync } from "node:fs"
import { resolve } from "node:path"

type FixtureQuestion = Record<string, unknown> & {
  question_type: string
  question_text?: string
  options?: string[]
  correct_answer?: number | string
}

type FixtureCatalogProfile = {
  id: string
  label: string
  description: string
  status: "available"
  output_kind: "questions" | "osce_stations"
  default_num_stations: number | null
  default_num_questions: number
  default_difficulty: "easy" | "medium" | "hard" | "mixed"
  default_question_types: string[]
}

type FixtureProfile = {
  catalog: FixtureCatalogProfile
  request: Record<string, unknown> & { generation_profile: string }
  output: {
    output_kind: "questions" | "osce_stations"
    questions?: FixtureQuestion[]
    osce_stations?: Array<Record<string, unknown>>
  }
}

type MalformedOutputCase = {
  id: string
  profile: string
  mode: string
  output: Record<string, unknown>
  error?: string
  expected?: Record<string, unknown>
  question_plan?: Array<Record<string, unknown>>
  selected_sources?: Array<Record<string, unknown>>
}

type AdvancedQuizFixtureMatrix = {
  schema_version: number
  profiles: Record<string, FixtureProfile>
  malformed_output_cases: MalformedOutputCase[]
}

const fixturePath = resolve(
  __dirname,
  "../../../../../../../../tldw_Server_API/tests/Quizzes/fixtures/advanced_quiz_generation_profiles.json",
)

export const advancedQuizFixtureMatrix = JSON.parse(
  readFileSync(fixturePath, "utf8"),
) as AdvancedQuizFixtureMatrix
