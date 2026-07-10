import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab, buildQuizManageQueryParams } from "../ManageTab"
import {
  useCreateQuizMutation,
  useCreateQuestionMutation,
  useDeleteQuestionMutation,
  useDeleteQuizMutation,
  useQuestionsQuery,
  useQuizzesQuery,
  useUpdateQuestionMutation,
  useUpdateQuizMutation
} from "../../hooks"
import { importQuizzesJson, listQuestions } from "@/services/quizzes"
import { tldwAuth, tldwClient } from "@/services/tldw"

const interpolate = (template: string, values: Record<string, unknown> | undefined) => {
  return template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("../../hooks", () => ({
  useQuizzesQuery: vi.fn(),
  useQuestionsQuery: vi.fn(),
  useCreateQuizMutation: vi.fn(),
  useDeleteQuizMutation: vi.fn(),
  useUpdateQuizMutation: vi.fn(),
  useCreateQuestionMutation: vi.fn(),
  useUpdateQuestionMutation: vi.fn(),
  useDeleteQuestionMutation: vi.fn()
}))

vi.mock("@/services/quizzes", () => ({
  listQuestions: vi.fn(),
  importQuizzesJson: vi.fn()
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    getMediaDetails: vi.fn(),
    getConfig: vi.fn()
  },
  tldwAuth: {
    getCurrentUser: vi.fn()
  }
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ManageTab bulk and duplicate actions", () => {
  const createQuizMutateAsync = vi.fn(async () => ({ id: 77 }))
  const createQuestionMutateAsync = vi.fn(async () => undefined)
  const deleteQuizMutateAsync = vi.fn(async () => undefined)
  const updateQuizMutateAsync = vi.fn(async () => undefined)
  const importQuizzesJsonMock = vi.fn(async () => ({
    imported_quizzes: 1,
    failed_quizzes: 0,
    imported_questions: 2,
    failed_questions: 0,
    items: [
      {
        source_index: 0,
        quiz_id: 901,
        imported_questions: 2,
        failed_questions: 0
      }
    ],
    errors: []
  }))
  let createObjectURLSpy: ReturnType<typeof vi.fn>
  let revokeObjectURLSpy: ReturnType<typeof vi.fn>
  let anchorClickSpy: ReturnType<typeof vi.spyOn>
  let clipboardWriteTextSpy: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.clearAllMocks()

    createObjectURLSpy = vi.fn(() => "blob:quiz-export")
    revokeObjectURLSpy = vi.fn()
    clipboardWriteTextSpy = vi.fn(async () => undefined)
    Object.defineProperty(URL, "createObjectURL", {
      value: createObjectURLSpy,
      configurable: true,
      writable: true
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      value: revokeObjectURLSpy,
      configurable: true,
      writable: true
    })
    Object.defineProperty(window.navigator, "clipboard", {
      value: { writeText: clipboardWriteTextSpy },
      configurable: true
    })
    anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})

    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 1,
            name: "Quiz A",
            description: "Alpha",
            workspace_tag: null,
            total_questions: 2,
            passing_score: 70,
            media_id: null,
            time_limit_seconds: 600,
            deleted: false,
            client_id: "test",
            version: 1
          },
          {
            id: 2,
            name: "Quiz B",
            description: "Beta",
            workspace_tag: null,
            total_questions: 1,
            passing_score: 80,
            media_id: null,
            time_limit_seconds: 300,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ],
        count: 2
      },
      isLoading: false,
      refetch: vi.fn()
    } as any)

    vi.mocked(useQuestionsQuery).mockReturnValue({
      data: { items: [], count: 0 },
      isLoading: false,
      refetch: vi.fn()
    } as any)

    vi.mocked(useCreateQuizMutation).mockReturnValue({
      mutateAsync: createQuizMutateAsync
    } as any)
    vi.mocked(useCreateQuestionMutation).mockReturnValue({
      mutateAsync: createQuestionMutateAsync
    } as any)
    vi.mocked(useDeleteQuizMutation).mockReturnValue({
      mutateAsync: deleteQuizMutateAsync
    } as any)
    vi.mocked(useUpdateQuizMutation).mockReturnValue({
      mutateAsync: updateQuizMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useUpdateQuestionMutation).mockReturnValue({ mutateAsync: vi.fn(async () => undefined), isPending: false } as any)
    vi.mocked(useDeleteQuestionMutation).mockReturnValue({ mutateAsync: vi.fn(async () => undefined) } as any)
    vi.mocked(importQuizzesJson).mockImplementation(importQuizzesJsonMock)

    vi.mocked(listQuestions).mockResolvedValue({
      items: [
        {
          id: 301,
          quiz_id: 1,
          question_type: "multiple_choice",
          question_text: "Question one",
          options: ["A", "B", "C", "D"],
          correct_answer: 0,
          explanation: "",
          source_citations: [
            {
              label: "Source chunk",
              quote: "Evidence quote",
              media_id: 55,
              chunk_id: "chunk-7"
            }
          ],
          points: 1,
          order_index: 0,
          deleted: false,
          client_id: "test",
          version: 1
        },
        {
          id: 302,
          quiz_id: 1,
          question_type: "true_false",
          question_text: "Question two",
          options: null,
          correct_answer: "true",
          explanation: "",
          points: 1,
          order_index: 1,
          deleted: false,
          client_id: "test",
          version: 1
        },
        {
          id: 303,
          quiz_id: 1,
          question_type: "matching",
          question_text: "Question three",
          options: ["CPU", "RAM"],
          correct_answer: {
            CPU: "Processor",
            RAM: "Memory"
          },
          explanation: "Basic system components.",
          points: 2,
          order_index: 2,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      count: 3
    } as any)
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      title: "Source Title"
    } as any)
    vi.mocked(tldwClient.getConfig).mockResolvedValue({
      authMode: "single-user"
    } as any)
    vi.mocked(tldwAuth.getCurrentUser).mockResolvedValue({
      id: 1,
      username: "quiz-admin",
      role: "admin",
      is_active: true
    } as any)
  })

  afterEach(() => {
    anchorClickSpy.mockRestore()
  })

  it("duplicates a quiz and copies question payloads", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-duplicate-1"))

    await waitFor(() => {
      expect(createQuizMutateAsync).toHaveBeenCalledTimes(1)
    })

    expect(createQuizMutateAsync).toHaveBeenCalledWith({
      name: "Quiz A (Copy)",
      description: "Alpha",
      workspace_tag: undefined,
      media_id: undefined,
      time_limit_seconds: 600,
      passing_score: 70
    })

    expect(createQuestionMutateAsync).toHaveBeenCalledTimes(3)
    expect(createQuestionMutateAsync).toHaveBeenNthCalledWith(1, {
      quizId: 77,
      question: {
        question_type: "multiple_choice",
        question_text: "Question one",
        options: ["A", "B", "C", "D"],
        correct_answer: 0,
        explanation: undefined,
        source_citations: [
          {
            label: "Source chunk",
            quote: "Evidence quote",
            media_id: 55,
            chunk_id: "chunk-7"
          }
        ],
        points: 1,
        order_index: 0
      }
    })
    expect(createQuestionMutateAsync).toHaveBeenNthCalledWith(2, {
      quizId: 77,
      question: {
        question_type: "true_false",
        question_text: "Question two",
        options: undefined,
        correct_answer: "true",
        explanation: undefined,
        points: 1,
        order_index: 1
      }
    })
    expect(createQuestionMutateAsync).toHaveBeenNthCalledWith(3, {
      quizId: 77,
      question: {
        question_type: "matching",
        question_text: "Question three",
        options: ["CPU", "RAM"],
        correct_answer: {
          CPU: "Processor",
          RAM: "Memory"
        },
        explanation: "Basic system components.",
        points: 2,
        order_index: 2
      }
    })
  })

  it("keeps only failed selections after bulk delete partial failure", async () => {
    deleteQuizMutateAsync
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error("delete failed"))

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    const quizACheckbox = screen.getByRole("checkbox", { name: /Select quiz Quiz A/i }) as HTMLInputElement
    const quizBCheckbox = screen.getByRole("checkbox", { name: /Select quiz Quiz B/i }) as HTMLInputElement

    fireEvent.click(quizACheckbox)
    fireEvent.click(quizBCheckbox)

    fireEvent.click(screen.getByTestId("manage-bulk-delete"))

    const popconfirmTitle = await screen.findByText("Delete selected quizzes?")
    const popconfirm = popconfirmTitle.closest(".ant-popover") ?? document.querySelector(".ant-popover")
    expect(popconfirm).not.toBeNull()
    if (!popconfirm) return

    fireEvent.click(within(popconfirm as HTMLElement).getByRole("button", { name: /^Delete$/i }))

    await waitFor(() => {
      expect(deleteQuizMutateAsync).toHaveBeenCalledTimes(2)
    })

    expect(quizACheckbox.checked).toBe(false)
    expect(quizBCheckbox.checked).toBe(true)
  }, 20000)

  it("exports selected quizzes as JSON payload", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    const quizACheckbox = screen.getByRole("checkbox", { name: /Select quiz Quiz A/i })
    fireEvent.click(quizACheckbox)
    fireEvent.click(screen.getByTestId("manage-bulk-export"))

    await waitFor(() => {
      expect(listQuestions).toHaveBeenCalledWith(1, {
        include_answers: true,
        limit: 200,
        offset: 0
      })
    })

    expect(createObjectURLSpy).toHaveBeenCalledTimes(1)
    expect(anchorClickSpy).toHaveBeenCalledTimes(1)
    expect(revokeObjectURLSpy).toHaveBeenCalledTimes(1)
  }, 20000)

  it("exports a single quiz with v1 export metadata", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-export-1"))

    await waitFor(() => {
      expect(listQuestions).toHaveBeenCalledWith(1, {
        include_answers: true,
        limit: 200,
        offset: 0
      })
    })
    expect(createObjectURLSpy).toHaveBeenCalledTimes(1)

    const blob = createObjectURLSpy.mock.calls[0][0] as Blob
    const payload = JSON.parse(await blob.text()) as {
      export_format: string
      quiz_count: number
      quizzes: Array<{
        quiz: { name: string; id: number }
        questions: Array<{ question_text: string; order_index: number; source_citations?: unknown }>
      }>
    }

    expect(payload.export_format).toBe("tldw.quiz.export.v1")
    expect(payload.quiz_count).toBe(1)
    expect(payload.quizzes[0].quiz.id).toBe(1)
    expect(payload.quizzes[0].quiz.name).toBe("Quiz A")
    expect(payload.quizzes[0].questions[0].question_text).toBe("Question one")
    expect(payload.quizzes[0].questions[0].order_index).toBe(0)
    expect(payload.quizzes[0].questions[0].source_citations).toEqual([
      {
        label: "Source chunk",
        quote: "Evidence quote",
        media_id: 55,
        chunk_id: "chunk-7"
      }
    ])
  }, 20000)

  it("imports quizzes from a v1 export payload via the server import endpoint", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    const payload = {
      export_format: "tldw.quiz.export.v1",
      quizzes: [
        {
          quiz: {
            name: "Imported Quiz",
            description: "Imported description",
            workspace_tag: "workspace:bio",
            media_id: 77,
            time_limit_seconds: 900,
            passing_score: 80
          },
          questions: [
            {
              question_type: "matching",
              question_text: "Match terms",
              options: ["CPU", "RAM"],
              correct_answer: {
                CPU: "Processor",
                RAM: "Memory"
              },
              points: 2,
              order_index: 1
            },
            {
              question_type: "multiple_choice",
              question_text: "2 + 2 = ?",
              options: ["3", "4"],
              correct_answer: 1,
              points: 1,
              order_index: 0
            }
          ]
        }
      ]
    }

    const file = new File([JSON.stringify(payload)], "quiz-import.json", {
      type: "application/json"
    })
    fireEvent.change(screen.getByTestId("manage-import-input"), {
      target: {
        files: [file]
      }
    })

    await waitFor(() => {
      expect(importQuizzesJsonMock).toHaveBeenCalledTimes(1)
    })

    expect(importQuizzesJsonMock).toHaveBeenCalledWith({
      export_format: "tldw.quiz.export.v1",
      quizzes: [
        {
          quiz: {
            name: "Imported Quiz",
            description: "Imported description",
            workspace_tag: "workspace:bio",
            media_id: 77,
            time_limit_seconds: 900,
            passing_score: 80
          },
          questions: [
            {
              question_type: "multiple_choice",
              question_text: "2 + 2 = ?",
              options: ["3", "4"],
              correct_answer: 1,
              explanation: undefined,
              hint: undefined,
              hint_penalty_points: 0,
              points: 1,
              order_index: 0,
              source_citations: undefined,
              tags: undefined
            },
            {
              question_type: "matching",
              question_text: "Match terms",
              options: ["CPU", "RAM"],
              correct_answer: {
                CPU: "Processor",
                RAM: "Memory"
              },
              explanation: undefined,
              hint: undefined,
              hint_penalty_points: 0,
              points: 2,
              order_index: 1,
              source_citations: undefined,
              tags: undefined
            }
          ]
        }
      ]
    })

    expect(createQuizMutateAsync).not.toHaveBeenCalled()
    expect(createQuestionMutateAsync).not.toHaveBeenCalled()
  }, 20000)

  it("copies a shareable quiz assignment link from list actions", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("quiz-share-1")).not.toBeDisabled()
    })
    fireEvent.click(screen.getByTestId("quiz-share-1"))
    fireEvent.click(await screen.findByRole("button", { name: "Copy Link" }))

    await waitFor(() => {
      expect(clipboardWriteTextSpy).toHaveBeenCalledTimes(1)
    })
    const copiedUrl = clipboardWriteTextSpy.mock.calls[0][0] as string
    expect(copiedUrl).toContain("/quiz?")
    const params = new URLSearchParams(copiedUrl.slice(copiedUrl.indexOf("?") + 1))
    expect(params.get("tab")).toBe("take")
    expect(params.get("start_quiz_id")).toBe("1")
    expect(params.get("highlight_quiz_id")).toBe("1")
    expect(params.get("assignment_mode")).toBe("shared")
  }, 20000)

  it("includes assignment due date metadata in copied share links", async () => {
    vi.mocked(tldwClient.getConfig).mockResolvedValueOnce({
      authMode: "multi-user"
    } as any)
    vi.mocked(tldwAuth.getCurrentUser).mockResolvedValueOnce({
      id: 7,
      username: "lead-user",
      role: "lead",
      is_active: true
    } as any)

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("quiz-share-1")).not.toBeDisabled()
    })
    fireEvent.click(screen.getByTestId("quiz-share-1"))
    fireEvent.change(screen.getByTestId("quiz-share-due-at-input"), {
      target: { value: "2026-03-01T14:30" }
    })
    fireEvent.change(screen.getByTestId("quiz-share-note-input"), {
      target: { value: "Complete before lab session." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Copy Link" }))

    await waitFor(() => {
      expect(clipboardWriteTextSpy).toHaveBeenCalled()
    })
    const copiedUrl = clipboardWriteTextSpy.mock.calls[clipboardWriteTextSpy.mock.calls.length - 1][0] as string
    const params = new URLSearchParams(copiedUrl.slice(copiedUrl.indexOf("?") + 1))
    const dueAt = params.get("assignment_due_at")
    expect(dueAt).toBeTruthy()
    expect(Number.isNaN(new Date(String(dueAt)).getTime())).toBe(false)
    expect(params.get("assignment_note")).toBe("Complete before lab session.")
    expect(params.get("assigned_by_role")).toBe("lead")
  }, 20000)

  it("disables share action for non-privileged multi-user roles", async () => {
    vi.mocked(tldwClient.getConfig).mockResolvedValueOnce({
      authMode: "multi-user"
    } as any)
    vi.mocked(tldwAuth.getCurrentUser).mockResolvedValueOnce({
      id: 9,
      username: "member-user",
      role: "member",
      is_active: true
    } as any)

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    await waitFor(() => {
      expect(tldwAuth.getCurrentUser).toHaveBeenCalledTimes(1)
    })
    expect(screen.getByTestId("quiz-share-1")).toBeDisabled()
  }, 20000)

  it("opens a printable quiz view from list actions", async () => {
    vi.mocked(listQuestions).mockResolvedValueOnce({
      items: [
        {
          id: 301,
          quiz_id: 1,
          question_type: "true_false",
          question_text: "Question one",
          options: null,
          correct_answer: "true",
          explanation: "",
          points: '</p><img data-quiz-marker="unsafe-image" src="https://attacker.example/track" onerror="globalThis.__quizXss=1"><style data-quiz-marker="unsafe-style">body{background-image:url(https://attacker.example/background)}</style><form data-quiz-marker="unsafe-form" action="https://attacker.example/submit"><input name="secret"></form><p>' as any,
          order_index: 0,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      count: 1
    } as any)

    const printSpy = vi.fn()
    const focusSpy = vi.fn()
    const documentWriteSpy = vi.fn()
    const openSpy = vi.spyOn(window, "open").mockImplementation(
      () => ({
        document: {
          open: vi.fn(),
          write: documentWriteSpy,
          close: vi.fn()
        },
        focus: focusSpy,
        print: printSpy
      } as any)
    )

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-print-1"))

    await waitFor(() => {
      expect(documentWriteSpy).toHaveBeenCalledTimes(1)
    })
    expect(printSpy).toHaveBeenCalledTimes(1)

    const writtenHtml = String(documentWriteSpy.mock.calls[0]?.[0] ?? "")
    expect(writtenHtml.match(/<!doctype html>/gi) ?? []).toHaveLength(1)
    expect(writtenHtml).toContain("<html")
    expect(writtenHtml).toContain("<head>")
    expect(writtenHtml).toContain("<title>Quiz A - Printable Quiz</title>")
    expect(writtenHtml).toContain("<style>")
    expect(writtenHtml).toContain("<body>")
    expect(writtenHtml).toContain("Points: 0")
    expect(writtenHtml).not.toContain("onerror")
    expect(writtenHtml).not.toContain("__quizXss")
    expect(writtenHtml).not.toContain("data-quiz-marker")
    expect(writtenHtml).not.toContain("attacker.example")
    expect(writtenHtml).not.toContain("<img")
    expect(writtenHtml).not.toContain("<form")

    openSpy.mockRestore()
  }, 20000)

  it("shows source media title as a link when media_id is present", async () => {
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: {
        items: [
          {
            id: 9,
            name: "Media-backed Quiz",
            description: "Has source",
            workspace_tag: null,
            total_questions: 1,
            passing_score: 75,
            media_id: 42,
            time_limit_seconds: 300,
            deleted: false,
            client_id: "test",
            version: 1
          }
        ],
        count: 1
      },
      isLoading: false,
      refetch: vi.fn()
    } as any)
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      title: "Cell Biology Source"
    } as any)

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    const sourceLink = await screen.findByRole("link", { name: "Cell Biology Source" })
    expect(sourceLink).toHaveAttribute("href", "/media?id=42")
  }, 20000)

  it("hides workspace quizzes by default and reveals them when the toggle is enabled", async () => {
    vi.mocked(useQuizzesQuery).mockImplementation((params: any) => ({
      data: {
        items: params?.workspace_id === "workspace-1"
          ? [
              {
                id: 3,
                name: "Workspace Quiz",
                description: "Scoped",
                workspace_tag: "workspace:workspace-1",
                workspace_id: "workspace-1",
                total_questions: 4,
                passing_score: 85,
                media_id: null,
                time_limit_seconds: 450,
                deleted: false,
                client_id: "test",
                version: 1
              }
            ]
          : params?.include_workspace_items
          ? [
              {
                id: 1,
                name: "Quiz A",
                description: "Alpha",
                workspace_tag: null,
                workspace_id: null,
                total_questions: 2,
                passing_score: 70,
                media_id: null,
                time_limit_seconds: 600,
                deleted: false,
                client_id: "test",
                version: 1
              },
              {
                id: 3,
                name: "Workspace Quiz",
                description: "Scoped",
                workspace_tag: "workspace:ws-1",
                workspace_id: "ws-1",
                total_questions: 4,
                passing_score: 85,
                media_id: null,
                time_limit_seconds: 450,
                deleted: false,
                client_id: "test",
                version: 1
              }
            ]
          : [
              {
                id: 1,
                name: "Quiz A",
                description: "Alpha",
                workspace_tag: null,
                workspace_id: null,
                total_questions: 2,
                passing_score: 70,
                media_id: null,
                time_limit_seconds: 600,
                deleted: false,
                client_id: "test",
                version: 1
              }
            ],
        count: 2
      },
      isLoading: false,
      refetch: vi.fn(),
      _params: params
    } as any))

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    expect(screen.getByText("Quiz A")).toBeInTheDocument()
    expect(screen.queryByText("Workspace Quiz")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: /Show workspace quizzes/i }))

    await waitFor(() => {
      expect(screen.getByText("Workspace Quiz")).toBeInTheDocument()
    })
    expect(vi.mocked(useQuizzesQuery)).toHaveBeenLastCalledWith(
      expect.objectContaining({
        include_workspace_items: true
      })
    )
  })

  it("filters quizzes to a selected workspace without widening the general list", async () => {
    const user = userEvent.setup()
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    await user.click(screen.getByRole("checkbox", { name: /Show workspace quizzes/i }))
    expect(screen.getByTestId("quiz-manage-workspace-filter")).toBeInTheDocument()
    expect(
      buildQuizManageQueryParams({
        searchQuery: "",
        page: 1,
        pageSize: 10,
        showWorkspaceQuizzes: true,
        selectedWorkspaceId: "workspace-1"
      })
    ).toEqual(
      expect.objectContaining({
        workspace_id: "workspace-1",
        include_workspace_items: false
      })
    )
  })

  it("patches quiz workspace scope in place when moving scope", async () => {
    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-edit-1"))

    const workspaceInput = await screen.findByLabelText(/Workspace ID/i)
    fireEvent.change(workspaceInput, { target: { value: "workspace-1" } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(updateQuizMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          quizId: 1,
          update: expect.objectContaining({
            workspace_id: "workspace-1",
            workspace_tag: "workspace:workspace-1"
          })
        })
      )
    })
  })

  it("clears quiz workspace scope in place when moving back to general", async () => {
    vi.mocked(useQuizzesQuery).mockReturnValueOnce({
      data: {
        items: [
          {
            id: 1,
            name: "Quiz A",
            description: "Alpha",
            workspace_id: "workspace-1",
            workspace_tag: "workspace:workspace-1",
            total_questions: 2,
            passing_score: 70,
            media_id: null,
            time_limit_seconds: 600,
            deleted: false,
            client_id: "test",
            version: 3
          }
        ],
        count: 1
      },
      isLoading: false,
      refetch: vi.fn()
    } as any)

    render(
      <ManageTab
        onNavigateToCreate={() => {}}
        onNavigateToGenerate={() => {}}
        onStartQuiz={() => {}}
      />
    )

    fireEvent.click(screen.getByTestId("quiz-edit-1"))

    const workspaceInput = await screen.findByLabelText(/Workspace ID/i)
    fireEvent.change(workspaceInput, { target: { value: "" } })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(updateQuizMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          quizId: 1,
          update: expect.objectContaining({
            workspace_id: null,
            workspace_tag: null,
            expected_version: 3
          })
        })
      )
    })
  })
})
