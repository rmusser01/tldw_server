import React from "react"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  useCreateQuestionMutation,
  useDeleteQuestionMutation,
  useQuestionsQuery,
  useQuizQuery,
  useQuizzesQuery
} from "../useQuizQueries"
import {
  createQuestion,
  deleteQuestion,
  getQuiz,
  listQuestions,
  listQuizzes,
  type Question,
  type QuestionCreate,
  type Quiz,
  type QuizListParams
} from "@/services/quizzes"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasQuizzes: true }, loading: false })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/services/quizzes", async () => {
  const actual = await vi.importActual<typeof import("@/services/quizzes")>("@/services/quizzes")
  return { ...actual, listQuizzes: vi.fn(), getQuiz: vi.fn(), listQuestions: vi.fn(), createQuestion: vi.fn(), deleteQuestion: vi.fn() }
})

const quizId = 55
const questionId = 71
const questionInput: QuestionCreate = {
  question_type: "multiple_choice",
  question_text: "Which organ pumps blood?",
  options: ["Heart", "Liver"],
  correct_answer: 0,
  points: 1,
  order_index: 0
}
const question: Question = {
  ...questionInput,
  id: questionId,
  quiz_id: quizId,
  points: 1,
  order_index: 0,
  deleted: false,
  client_id: "test-client",
  version: 3
}
const quiz: Quiz = {
  id: quizId,
  name: "Biology",
  total_questions: 0,
  activity_type: "questions",
  total_stations: 0,
  deleted: false,
  client_id: "test-client",
  version: 1
}
const listParams: QuizListParams[] = [
  { limit: 20, offset: 0 },
  { q: "Biology", limit: 10, offset: 0 }
]
const clients: QueryClient[] = []
let serverQuestions: Question[]

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  clients.push(client)
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  return { client, wrapper }
}

const useLists = () => ({
  general: useQuizzesQuery(listParams[0]),
  search: useQuizzesQuery(listParams[1])
})
const useQuestionMutations = () => ({
  create: useCreateQuestionMutation(),
  remove: useDeleteQuestionMutation()
})

beforeEach(() => {
  vi.resetAllMocks()
  serverQuestions = []
  vi.mocked(listQuizzes).mockImplementation(async () => ({
    items: [{ ...quiz, total_questions: serverQuestions.length }], count: 1
  }))
  vi.mocked(getQuiz).mockImplementation(async () => ({ ...quiz, total_questions: serverQuestions.length }))
  vi.mocked(listQuestions).mockImplementation(async () => ({ items: [...serverQuestions], count: serverQuestions.length }))
  vi.mocked(createQuestion).mockImplementation(async () => {
    serverQuestions = [question]
    return question
  })
  vi.mocked(deleteQuestion).mockImplementation(async () => { serverQuestions = [] })
})

afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
})

describe("question mutations refresh authoritative quiz list counts", () => {
  it.each([
    ["create", "active", 0, 1],
    ["create", "inactive", 0, 1],
    ["delete", "active", 1, 0],
    ["delete", "inactive", 1, 0]
  ] as const)("%s updates %s fresh list variants from %i to %i", async (operation, visibility, initial, expected) => {
    serverQuestions = initial ? [question] : []
    const { client, wrapper } = createWrapper()
    let lists = renderHook(useLists, { wrapper })
    const details = renderHook(() => ({ detail: useQuizQuery(quizId), questions: useQuestionsQuery(quizId) }), { wrapper })
    const mutations = renderHook(useQuestionMutations, { wrapper })
    await waitFor(() => {
      expect(lists.result.current.general.data?.items[0]?.total_questions).toBe(initial)
      expect(lists.result.current.search.data?.items[0]?.total_questions).toBe(initial)
      expect(details.result.current.questions.data?.count).toBe(initial)
      expect(details.result.current.detail.data?.total_questions).toBe(initial)
    })
    expect(listQuizzes).toHaveBeenCalledTimes(2)
    listParams.forEach((params) => {
      expect(client.getQueryState(["quizzes:list", params])?.isInvalidated).toBe(false)
    })
    if (visibility === "inactive") lists.unmount()

    await act(async () => {
      if (operation === "create") {
        await mutations.result.current.create.mutateAsync({ quizId, question: questionInput })
      } else {
        await mutations.result.current.remove.mutateAsync({ quizId, questionId, version: question.version })
      }
    })
    if (operation === "create") expect(createQuestion).toHaveBeenCalledWith(quizId, questionInput)
    else expect(deleteQuestion).toHaveBeenCalledWith(quizId, questionId, question.version)
    await waitFor(() => {
      expect(details.result.current.questions.data?.count).toBe(expected)
      expect(details.result.current.detail.data?.total_questions).toBe(expected)
    })
    if (visibility === "inactive") {
      expect(listQuizzes).toHaveBeenCalledTimes(2)
      lists = renderHook(useLists, { wrapper })
    }
    await waitFor(() => {
      expect(lists.result.current.general.data?.items[0]?.total_questions).toBe(expected)
      expect(lists.result.current.search.data?.items[0]?.total_questions).toBe(expected)
    })
    expect(listQuizzes).toHaveBeenCalledTimes(4)
    expect(lists.result.current.general.data?.count).toBe(1)
    expect(lists.result.current.search.data?.count).toBe(1)
  })

  it.each(["create", "delete"] as const)("failed %s preserves a fresh list and exposes the API refusal", async (operation) => {
    serverQuestions = operation === "delete" ? [question] : []
    const initial = serverQuestions.length
    const refusal = new Error("Question mutation refused")
    if (operation === "create") vi.mocked(createQuestion).mockRejectedValueOnce(refusal)
    else vi.mocked(deleteQuestion).mockRejectedValueOnce(refusal)
    const { client, wrapper } = createWrapper()
    const lists = renderHook(useLists, { wrapper })
    const mutations = renderHook(useQuestionMutations, { wrapper })
    await waitFor(() => expect(lists.result.current.general.data?.items[0]?.total_questions).toBe(initial))
    await waitFor(() => expect(lists.result.current.search.data?.items[0]?.total_questions).toBe(initial))

    await act(async () => {
      const mutation = operation === "create"
        ? mutations.result.current.create.mutateAsync({ quizId, question: questionInput })
        : mutations.result.current.remove.mutateAsync({ quizId, questionId, version: question.version })
      await expect(mutation).rejects.toBe(refusal)
    })
    expect(listQuizzes).toHaveBeenCalledTimes(2)
    expect(lists.result.current.general.data?.items[0]?.total_questions).toBe(initial)
    expect(lists.result.current.search.data?.items[0]?.total_questions).toBe(initial)
    listParams.forEach((params) => expect(client.getQueryState(["quizzes:list", params])?.isInvalidated).toBe(false))
    await waitFor(() => {
      expect((operation === "create" ? mutations.result.current.create : mutations.result.current.remove).error).toBe(refusal)
    })
  })
})
