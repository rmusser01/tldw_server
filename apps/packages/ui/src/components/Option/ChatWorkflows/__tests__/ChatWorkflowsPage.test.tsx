import React from "react"
import { fireEvent, render as renderUI, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const state = vi.hoisted(() => {
  const templates = [
    {
      id: 1,
      title: "Discovery",
      description: "Collect launch context",
      version: 1,
      status: "active" as const,
      steps: [
        {
          id: "goal",
          step_index: 0,
          label: "Goal",
          base_question: "What outcome are we aiming for?",
          question_mode: "stock" as const,
          context_refs: []
        }
      ]
    }
  ]

  return {
    templates,
    createMutateAsync: vi.fn(),
    updateMutateAsync: vi.fn(),
    generateMutateAsync: vi.fn(),
    startRunMutateAsync: vi.fn(),
    submitAnswerMutateAsync: vi.fn(),
    submitRoundMutateAsync: vi.fn(),
    cancelRunMutateAsync: vi.fn(),
    continueRunMutateAsync: vi.fn(),
    activeRun: null as any,
    transcript: null as any,
    navigate: vi.fn(),
    notification: {
      success: vi.fn(),
      error: vi.fn()
    }
  }
})

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

vi.mock("@/hooks/useChatWorkflows", () => ({
  useChatWorkflowTemplates: () => ({
    data: state.templates,
    isLoading: false,
    isError: false,
    error: null
  }),
  useCreateChatWorkflowTemplate: () => ({
    mutateAsync: state.createMutateAsync,
    isPending: false
  }),
  useUpdateChatWorkflowTemplate: () => ({
    mutateAsync: state.updateMutateAsync,
    isPending: false
  }),
  useGenerateChatWorkflowDraft: () => ({
    mutateAsync: state.generateMutateAsync,
    isPending: false
  }),
  useStartChatWorkflowRun: () => ({
    mutateAsync: state.startRunMutateAsync,
    isPending: false
  }),
  useChatWorkflowRun: () => ({
    data: state.activeRun,
    isLoading: false,
    isError: false,
    error: null
  }),
  useChatWorkflowTranscript: () => ({
    data: state.transcript,
    isLoading: false,
    isError: false,
    error: null
  }),
  useSubmitChatWorkflowAnswer: () => ({
    mutateAsync: state.submitAnswerMutateAsync,
    isPending: false
  }),
  useSubmitChatWorkflowRound: () => ({
    mutateAsync: state.submitRoundMutateAsync,
    isPending: false
  }),
  useCancelChatWorkflowRun: () => ({
    mutateAsync: state.cancelRunMutateAsync,
    isPending: false
  }),
  useContinueChatWorkflowRun: () => ({
    mutateAsync: state.continueRunMutateAsync,
    isPending: false
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => state.notification
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => state.navigate
  }
})

import { ChatWorkflowsPage } from "../ChatWorkflowsPage"

const render: typeof renderUI = (ui, options) =>
  renderUI(ui, {
    wrapper: ({ children }) => <MemoryRouter>{children}</MemoryRouter>,
    ...options
  })

describe("ChatWorkflowsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    state.createMutateAsync.mockResolvedValue({
      id: 2,
      title: "New workflow",
      version: 1,
      status: "active",
      steps: []
    })
    state.updateMutateAsync.mockResolvedValue({
      id: 1,
      title: "Discovery",
      version: 2,
      status: "active",
      steps: []
    })
    state.generateMutateAsync.mockResolvedValue({
      template_draft: {
        title: "Generated workflow",
        description: "Generated from a goal",
        version: 1,
        steps: [
          {
            id: "generated-step",
            step_index: 0,
            label: "Step 1",
            base_question: "What should happen first?",
            question_mode: "stock",
            context_refs: []
          }
        ]
      }
    })
    state.startRunMutateAsync.mockImplementation(async () => {
      state.activeRun = {
        run_id: "run-123",
        template_id: 1,
        template_version: 1,
        status: "active",
        current_step_index: 0,
        current_question: "What outcome are we aiming for?",
        selected_context_refs: [],
        answers: []
      }
      state.transcript = {
        run_id: "run-123",
        messages: []
      }
      return state.activeRun
    })
    state.submitAnswerMutateAsync.mockImplementation(async (payload) => {
      state.activeRun = {
        ...state.activeRun,
        status: "completed",
        current_step_index: 1,
        current_question: null,
        completed_at: "2026-03-07T00:00:00Z",
        answers: [
          {
            step_id: "goal",
            step_index: 0,
            displayed_question: "What outcome are we aiming for?",
            answer_text: payload.answer_text,
            question_generation_meta: {}
          }
        ]
      }
      state.transcript = {
        run_id: "run-123",
        messages: [
          {
            role: "assistant",
            content: "What outcome are we aiming for?",
            step_index: 0
          },
          {
            role: "user",
            content: payload.answer_text,
            step_index: 0
          }
        ]
      }
      return state.activeRun
    })
    state.submitRoundMutateAsync.mockImplementation(async (payload) => {
      state.activeRun = {
        ...state.activeRun,
        current_step_kind: "dialogue_round_step",
        current_round_index: 1,
        current_prompt: "Defend your evidence.",
        current_question: "Defend your evidence.",
        rounds: [
          {
            round_index: 0,
            user_message: payload.user_message,
            debate_llm_message: "Counterargument",
            moderator_decision: "continue",
            moderator_summary: "Push harder on the weakest premise.",
            next_user_prompt: "Defend your evidence.",
            status: "completed"
          }
        ]
      }
      state.transcript = {
        run_id: state.activeRun.run_id,
        messages: [
          {
            role: "user",
            content: payload.user_message,
            step_index: 0
          },
          {
            role: "debate_llm",
            content: "Counterargument",
            step_index: 0
          },
          {
            role: "moderator",
            content: "Push harder on the weakest premise.\n\nDefend your evidence.",
            step_index: 0
          }
        ]
      }
      return state.activeRun
    })
    state.cancelRunMutateAsync.mockImplementation(async () => {
      state.activeRun = {
        ...state.activeRun,
        status: "canceled",
        current_question: null,
        canceled_at: "2026-03-07T00:00:00Z"
      }
      return state.activeRun
    })
    state.continueRunMutateAsync.mockResolvedValue({
      conversation_id: "server-chat-456"
    })
    state.activeRun = null
    state.transcript = null
  })

  it("loads a library template into the builder", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "Edit in Builder" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Discovery")).toBeInTheDocument()
    })
    expect(
      screen.getByDisplayValue("What outcome are we aiming for?")
    ).toBeInTheDocument()
  })

  it("seeds the builder with a generated draft", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Generate" }))
    fireEvent.change(screen.getByLabelText("Workflow goal"), {
      target: { value: "Launch a customer onboarding flow" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Generate Draft" }))

    await waitFor(() => {
      expect(state.generateMutateAsync).toHaveBeenCalledWith({
        goal: "Launch a customer onboarding flow",
        base_question: "",
        desired_step_count: 4,
        context_refs: []
      })
    })
    expect(screen.getByDisplayValue("Generated workflow")).toBeInTheDocument()
    expect(screen.getByDisplayValue("What should happen first?")).toBeInTheDocument()
  })

  it("creates a new template from the builder draft", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "New Template" }))
    fireEvent.change(screen.getByLabelText("Template title"), {
      target: { value: "Intake" }
    })
    fireEvent.change(screen.getByLabelText("Question for step 1"), {
      target: { value: "What are we trying to learn?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save Template" }))

    await waitFor(() => {
      expect(state.createMutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Intake",
          steps: [
            expect.objectContaining({
              step_index: 0,
              base_question: "What are we trying to learn?"
            })
          ]
        })
      )
    })
    expect(state.notification.success).toHaveBeenCalled()
  })

  it("starts a guided run from the library and shows the first question", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }))

    await waitFor(() => {
      expect(state.startRunMutateAsync).toHaveBeenCalledWith({
        template_id: 1,
        selected_context_refs: []
      })
    })
    expect(screen.getByText("Run in progress")).toBeInTheDocument()
    expect(screen.getByText("What outcome are we aiming for?")).toBeInTheDocument()
  })

  it("starts a guided run from the current builder draft", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "New Template" }))
    fireEvent.change(screen.getByLabelText("Template title"), {
      target: { value: "Intake draft" }
    })
    fireEvent.change(screen.getByLabelText("Question for step 1"), {
      target: { value: "What do we need to learn first?" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Start Run" }))

    await waitFor(() => {
      expect(state.startRunMutateAsync).toHaveBeenCalledWith({
        template_draft: expect.objectContaining({
          title: "Intake draft",
          steps: [
            expect.objectContaining({
              step_index: 0,
              base_question: "What do we need to learn first?"
            })
          ]
        }),
        selected_context_refs: []
      })
    })
  })

  it("loads the built-in Socratic dialogue template into the builder", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "Use Socratic Dialogue" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Socratic Dialogue")).toBeInTheDocument()
    })
    expect(
      screen.getByDisplayValue("State your current thesis or position.")
    ).toBeInTheDocument()
  })

  it("submits a dialogue response when the active step is a moderated round", async () => {
    state.startRunMutateAsync.mockImplementationOnce(async () => {
      state.activeRun = {
        run_id: "run-dialogue",
        template_version: 1,
        status: "active",
        current_step_index: 0,
        current_question: "State your current thesis or position.",
        current_step_kind: "dialogue_round_step",
        current_prompt: "State your current thesis or position.",
        current_round_index: 0,
        selected_context_refs: [],
        answers: [],
        rounds: []
      }
      state.transcript = {
        run_id: "run-dialogue",
        messages: []
      }
      return state.activeRun
    })

    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "Use Socratic Dialogue" }))
    await waitFor(() => {
      expect(screen.getByDisplayValue("Socratic Dialogue")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }))

    await waitFor(() => {
      expect(screen.getByText("State your current thesis or position.")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Response for current round"), {
      target: { value: "My thesis is sound." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Submit Response" }))

    await waitFor(() => {
      expect(state.submitRoundMutateAsync).toHaveBeenCalledWith({
        user_message: "My thesis is sound."
      })
    })
    expect(screen.getByText("Defend your evidence.")).toBeInTheDocument()
  })

  it("hands a completed workflow off to free chat", async () => {
    render(<ChatWorkflowsPage />)

    fireEvent.click(screen.getByRole("button", { name: "Start Run" }))

    await waitFor(() => {
      expect(screen.getByText("What outcome are we aiming for?")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("Answer for current step"), {
      target: { value: "Ship a feature" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Submit Answer" }))

    await waitFor(() => {
      expect(state.submitAnswerMutateAsync).toHaveBeenCalledWith({
        step_index: 0,
        answer_text: "Ship a feature"
      })
    })
    expect(screen.getByText("Workflow complete")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Continue to Chat" }))

    await waitFor(() => {
      expect(state.continueRunMutateAsync).toHaveBeenCalled()
    })
    expect(state.navigate).toHaveBeenCalledWith(
      "/chat?settingsServerChatId=server-chat-456"
    )
  })
})
