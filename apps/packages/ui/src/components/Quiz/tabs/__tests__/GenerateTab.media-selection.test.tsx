import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GenerateTab } from "../GenerateTab";
import { useGenerateQuizMutation } from "../../hooks";
import { tldwClient } from "@/services/tldw";
import {
  createDeck,
  createFlashcard,
  generateFlashcards,
  listDecks,
  listFlashcards,
} from "@/services/flashcards";
import {
  QUIZ_GENERATION_PROFILES,
  type QuestionBase,
  type QuestionCreate,
  type QuestionUpdate,
  type QuizImportQuestion,
} from "@/services/quizzes";

const navigationMocks = {
  navigate: vi.fn(),
};

const interpolate = (
  template: string,
  values: Record<string, unknown> | undefined,
) => {
  return template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key];
    return value == null ? "" : String(value);
  });
};

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string;
            [key: string]: unknown;
          },
    ) => {
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions;
      const defaultValue = defaultValueOrOptions?.defaultValue;
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions);
      }
      return key;
    },
  }),
}));

vi.mock("../../hooks", () => ({
  useGenerateQuizMutation: vi.fn(),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigationMocks.navigate,
}));

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    listMedia: vi.fn(),
    searchMedia: vi.fn(),
    getMediaDetails: vi.fn(),
    listNotes: vi.fn(),
    searchNotes: vi.fn(),
  },
}));

vi.mock("@/services/flashcards", () => ({
  generateFlashcards: vi.fn(),
  createDeck: vi.fn(),
  createFlashcard: vi.fn(),
  listDecks: vi.fn(),
  listFlashcards: vi.fn(),
}));

if (!(globalThis as any).ResizeObserver) {
  (globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

describe("GenerateTab scalable media selection and generation flow", () => {
  const renderWithQueryClient = ({
    onNavigateToTake,
    onNavigateToManage,
  }: {
    onNavigateToTake?: (intent?: any) => void;
    onNavigateToManage?: () => void;
  } = {}) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    return render(
      <QueryClientProvider client={queryClient}>
        <GenerateTab
          onNavigateToTake={onNavigateToTake ?? (() => {})}
          onNavigateToManage={onNavigateToManage}
        />
      </QueryClientProvider>,
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
    navigationMocks.navigate.mockReset();

    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({} as any);
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ items: [] } as any);
    vi.mocked(tldwClient.searchNotes).mockResolvedValue({ items: [] } as any);
    vi.mocked(generateFlashcards).mockResolvedValue({
      flashcards: [],
      count: 0,
    } as any);
    vi.mocked(listDecks).mockResolvedValue([] as any);
    vi.mocked(listFlashcards).mockResolvedValue({ items: [], count: 0 } as any);
    vi.mocked(createDeck).mockResolvedValue({
      id: 100,
      name: "Generated Deck",
    } as any);
    vi.mocked(createFlashcard).mockResolvedValue({
      uuid: "card-1",
    } as any);

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        quiz: { id: 1, name: "Generated Quiz" },
        questions: [{ id: 1 }],
      })),
      isPending: false,
    } as any);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("loads media in pages and keeps selection stable while loading more", async () => {
    vi.mocked(tldwClient.listMedia)
      .mockResolvedValueOnce({
        items: [
          { id: 1, title: "Doc 1", type: "pdf" },
          { id: 2, title: "Doc 2", type: "pdf" },
        ],
        pagination: { total_items: 3 },
      } as any)
      .mockResolvedValueOnce({
        items: [{ id: 3, title: "Doc 3", type: "pdf" }],
        pagination: { total_items: 3 },
      } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(
        screen.getByText("Showing 2 of 3 media items"),
      ).toBeInTheDocument();
    });

    const combobox = screen.getAllByRole("combobox")[0];
    fireEvent.mouseDown(combobox);
    fireEvent.click(await screen.findByText("Doc 1 (pdf)"));

    const generateButton = screen.getByRole("button", {
      name: /Generate Quiz/i,
    });
    expect(generateButton).not.toBeDisabled();

    fireEvent.click(screen.getByTestId("generate-media-load-more"));

    await waitFor(() => {
      expect(screen.getByText("3 media items available")).toBeInTheDocument();
    });

    expect(generateButton).not.toBeDisabled();
    expect(tldwClient.listMedia).toHaveBeenNthCalledWith(1, {
      page: 1,
      results_per_page: 50,
    });
    expect(tldwClient.listMedia).toHaveBeenNthCalledWith(2, {
      page: 2,
      results_per_page: 50,
    });
  }, 20000);

  it("uses server-side media search when search term is entered", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 1, title: "Doc 1", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);

    vi.mocked(tldwClient.searchMedia).mockResolvedValue({
      items: [{ id: 9, title: "Cell Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(tldwClient.listMedia).toHaveBeenCalledWith({
        page: 1,
        results_per_page: 50,
      });
    });

    const combobox = screen.getAllByRole("combobox")[0];
    fireEvent.mouseDown(combobox);
    fireEvent.change(combobox, { target: { value: "cell" } });

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 400));
    });

    await waitFor(() => {
      expect(tldwClient.searchMedia).toHaveBeenCalledWith(
        { query: "cell" },
        { page: 1, results_per_page: 50 },
      );
    });
  }, 20000);

  it("submits mixed sources in generation payload", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.listNotes).mockResolvedValue({
      items: [{ id: "note-1", title: "Lecture Notes" }],
    } as any);
    vi.mocked(listDecks).mockResolvedValue([
      { id: 20, name: "Deck One" },
    ] as any);
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [
        { uuid: "card-1", deck_id: 20, front: "ATP", back: "Energy currency" },
      ],
      count: 1,
    } as any);

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Cell Biology Checkpoint" },
      questions: [{ id: 1 }, { id: 2 }, { id: 3 }],
    }));
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Notes (pdf)"));

    const noteCombobox = screen.getAllByRole("combobox")[1];
    fireEvent.mouseDown(noteCombobox);
    fireEvent.click(await screen.findByText("Lecture Notes"));

    const deckCombobox = screen.getAllByRole("combobox")[2];
    fireEvent.mouseDown(deckCombobox);
    fireEvent.click(await screen.findByText("Deck One"));

    await waitFor(() => {
      expect(listFlashcards).toHaveBeenCalledWith(
        expect.objectContaining({ deck_id: 20, due_status: "all" }),
      );
    });

    const cardCombobox = screen.getAllByRole("combobox")[3];
    fireEvent.mouseDown(cardCombobox);
    fireEvent.click(await screen.findByText("Deck One: ATP - Energy currency"));

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          sources: expect.arrayContaining([
            { source_type: "media", source_id: "10" },
            { source_type: "note", source_id: "note-1" },
            { source_type: "flashcard_deck", source_id: "20" },
            { source_type: "flashcard_card", source_id: "card-1" },
          ]),
        }),
      }),
    );
  }, 20000);

  it("keeps generate button disabled when no sources are selected", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Cell Biology Checkpoint" },
      questions: [{ id: 1 }],
    }));
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    const generateButton = screen.getByRole("button", {
      name: /Generate Quiz/i,
    });
    expect(generateButton).toBeDisabled();
    expect(screen.getByTestId("generate-blocking-reason")).toHaveTextContent(
      "Select at least one source to generate a quiz.",
    );
    fireEvent.click(generateButton);
    expect(mutateAsync).not.toHaveBeenCalled();
  }, 20000);

  it("summarizes selected sources and lets users remove them before generation", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.listNotes).mockResolvedValue({
      items: [{ id: "note-1", title: "Lecture Notes" }],
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Notes (pdf)"));

    const noteCombobox = screen.getAllByRole("combobox")[1];
    fireEvent.mouseDown(noteCombobox);
    fireEvent.click(await screen.findByText("Lecture Notes"));

    expect(screen.getByTestId("generate-selected-sources")).toHaveTextContent(
      "Media",
    );
    expect(screen.getByTestId("generate-selected-sources")).toHaveTextContent(
      "Biology Notes",
    );
    expect(screen.getByTestId("generate-selected-sources")).toHaveTextContent(
      "Notes",
    );
    expect(screen.getByTestId("generate-selected-sources")).toHaveTextContent(
      "Lecture Notes",
    );
    expect(screen.getByTestId("generate-generation-brief")).toHaveTextContent(
      "2 sources selected",
    );

    fireEvent.click(
      screen.getByRole("button", { name: /Remove Biology Notes/i }),
    );

    expect(
      screen.getByTestId("generate-selected-sources"),
    ).not.toHaveTextContent("Biology Notes");
    expect(screen.getByTestId("generate-generation-brief")).toHaveTextContent(
      "1 source selected",
    );
  }, 20000);

  it("clears the study materials toggle when the selected media is removed", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Cell Biology Checkpoint" },
      questions: [{ id: 1 }],
    }));
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Notes (pdf)"));

    const studyMaterialsToggle = screen.getByRole("checkbox", {
      name: /Also generate a flashcard deck/i,
    });
    fireEvent.click(studyMaterialsToggle);
    expect(studyMaterialsToggle).toBeChecked();

    fireEvent.click(
      screen.getByRole("button", { name: /Remove Biology Notes/i }),
    );

    expect(
      screen.getByRole("checkbox", {
        name: /Also generate a flashcard deck/i,
      }),
    ).not.toBeChecked();

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Notes (pdf)"));
    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });
    expect(generateFlashcards).not.toHaveBeenCalled();
  }, 20000);

  it("shows preview-first flow and passes focus topics in generation payload", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);

    const onNavigateToTake = vi.fn();
    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Cell Biology Checkpoint" },
      questions: [{ id: 1 }, { id: 2 }, { id: 3 }],
    }));

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient({ onNavigateToTake });

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Notes (pdf)"));

    const allComboboxes = screen.getAllByRole("combobox");
    const focusInput = allComboboxes[allComboboxes.length - 1];
    fireEvent.mouseDown(focusInput);
    fireEvent.change(focusInput, { target: { value: "cell membrane" } });
    fireEvent.keyDown(focusInput, { key: "Enter", code: "Enter" });
    fireEvent.change(focusInput, { target: { value: "mitosis" } });
    fireEvent.keyDown(focusInput, { key: "Enter", code: "Enter" });

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(screen.getByTestId("generate-preview-card")).toBeInTheDocument();
    });

    expect(onNavigateToTake).not.toHaveBeenCalled();
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          sources: [{ source_type: "media", source_id: "10" }],
          focus_topics: ["cell membrane", "mitosis"],
        }),
        signal: expect.any(AbortSignal),
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: /Take Quiz/i }));
    expect(onNavigateToTake).toHaveBeenCalledWith({
      startQuizId: 42,
      highlightQuizId: 42,
      sourceTab: "generate",
    });
  }, 20000);

  it("applies Best of Five profile defaults in generation payload", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Clinical Notes", type: "pdf" }],
      pagination: { total_items: 1 }
    } as any)

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Clinical BOF" },
      questions: [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }]
    }))

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    renderWithQueryClient()

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument()
    })

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0])
    fireEvent.click(await screen.findByText("Clinical Notes (pdf)"))

    const profileSelect = screen.getByTestId("generate-profile-select")
    fireEvent.mouseDown(profileSelect.querySelector(".ant-select-selector") ?? profileSelect)
    fireEvent.click(await screen.findByText("Best of Five"))

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          sources: [{ source_type: "media", source_id: "10" }],
          generation_profile: "best_of_five",
          num_questions: 5,
          question_types: ["multiple_choice"]
        }),
        signal: expect.any(AbortSignal)
      })
    )
  }, 20000)

  it("exposes the available EMQ profile and grouped question contract", () => {
    const questionBase: QuestionBase = {
      id: 1,
      quiz_id: 7,
      question_type: "multiple_choice",
      question_text: "Choose the most likely diagnosis.",
      options: ["Diagnosis A", "Diagnosis B"],
      group_id: null,
      group_prompt: null,
      points: 1,
      order_index: 0,
      deleted: false,
      client_id: "test",
      version: 1
    }
    const questionCreate: QuestionCreate = {
      question_type: "multiple_choice",
      question_text: "Choose the most likely diagnosis.",
      options: ["Diagnosis A", "Diagnosis B"],
      correct_answer: 0,
      group_id: "clinical-emq",
      group_prompt: "For each patient, choose one diagnosis from the shared bank."
    }
    const questionUpdate: QuestionUpdate = {
      group_id: null,
      group_prompt: null
    }
    const importQuestion: QuizImportQuestion = {
      question_type: "multiple_choice",
      question_text: "Choose the most likely diagnosis.",
      options: ["Diagnosis A", "Diagnosis B"],
      correct_answer: 0,
      group_id: "clinical-emq",
      group_prompt: "For each patient, choose one diagnosis from the shared bank."
    }

    expect(questionBase.group_id).toBeNull()
    expect(questionCreate.group_id).toBe("clinical-emq")
    expect(questionUpdate.group_prompt).toBeNull()
    expect(importQuestion.group_prompt).toContain("shared bank")
    expect(QUIZ_GENERATION_PROFILES.find((profile) => profile.id === "emq")).toMatchObject({
      label: "EMQ",
      status: "available",
      default_num_questions: 5,
      default_question_types: ["multiple_choice"]
    })
  })

  it("applies EMQ profile defaults in generation payload", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Clinical Notes", type: "pdf" }],
      pagination: { total_items: 1 }
    } as any)

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 42, name: "Clinical EMQ" },
      questions: [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }]
    }))

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    renderWithQueryClient()

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument()
    })

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0])
    fireEvent.click(await screen.findByText("Clinical Notes (pdf)"))

    const profileSelect = screen.getByTestId("generate-profile-select")
    fireEvent.mouseDown(profileSelect.querySelector(".ant-select-selector") ?? profileSelect)
    fireEvent.click(await screen.findByText("EMQ"))

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          sources: [{ source_type: "media", source_id: "10" }],
          generation_profile: "emq",
          num_questions: 5,
          question_types: ["multiple_choice"]
        }),
        signal: expect.any(AbortSignal)
      })
    )
  }, 20000)

  it("supports canceling in-flight generation and does not navigate", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 5, title: "Long Transcript", type: "audio" }],
      pagination: { total_items: 1 },
    } as any);

    const onNavigateToTake = vi.fn();
    const mutateAsync = vi.fn(
      ({ signal }: { signal?: AbortSignal }) =>
        new Promise((resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => {
              const abortError = new Error("aborted");
              abortError.name = "AbortError";
              reject(abortError);
            },
            { once: true },
          );
        }),
    );

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient({ onNavigateToTake });

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Long Transcript (audio)"));

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    const cancelButton = await screen.findByTestId("generate-cancel-button");
    fireEvent.click(cancelButton);

    await waitFor(() => {
      expect(
        screen.queryByTestId("generate-cancel-button"),
      ).not.toBeInTheDocument();
    });

    expect(onNavigateToTake).not.toHaveBeenCalled();
    expect(mutateAsync).toHaveBeenCalledTimes(1);
    const mutationCall = mutateAsync.mock.calls[0]?.[0];
    expect(mutationCall?.signal).toBeInstanceOf(AbortSignal);
    expect(mutationCall?.signal?.aborted).toBe(true);
  }, 20000);

  it("supports canceling while combined flashcard generation is in flight", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 77, title: "Biology Source", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.getMediaDetails).mockImplementation(
      async (_mediaId, options) => {
        if (options?.include_content) {
          return {
            content: {
              text: "Mitosis and meiosis are core processes in cell division.",
            },
          } as any;
        }
        return { content: { word_count: 1200 } } as any;
      },
    );

    let flashcardsSignal: AbortSignal | undefined;
    vi.mocked(generateFlashcards).mockImplementation(
      (_input: any, options?: { signal?: AbortSignal }) => {
        flashcardsSignal = options?.signal;
        return new Promise((_, reject) => {
          options?.signal?.addEventListener(
            "abort",
            () => {
              const abortError = new Error("aborted");
              abortError.name = "AbortError";
              reject(abortError);
            },
            { once: true },
          );
        }) as any;
      },
    );

    const onNavigateToTake = vi.fn();
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        quiz: { id: 99, name: "Biology Mastery" },
        questions: [{ id: 1 }, { id: 2 }],
      })),
      isPending: false,
    } as any);

    renderWithQueryClient({ onNavigateToTake });

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Source (pdf)"));
    fireEvent.click(screen.getByTestId("generate-study-materials-toggle"));
    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(generateFlashcards).toHaveBeenCalledTimes(1);
    });
    expect(flashcardsSignal).toBeInstanceOf(AbortSignal);

    fireEvent.click(await screen.findByTestId("generate-cancel-button"));

    await waitFor(() => {
      expect(
        screen.queryByTestId("generate-cancel-button"),
      ).not.toBeInTheDocument();
    });

    expect(flashcardsSignal?.aborted).toBe(true);
    expect(createDeck).not.toHaveBeenCalled();
    expect(createFlashcard).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("generate-preview-card"),
    ).not.toBeInTheDocument();
    expect(onNavigateToTake).not.toHaveBeenCalled();
  }, 20000);

  it("shows question-count recommendation guidance tied to selected media length", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 21, title: "Lengthy Source", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { word_count: 2400 },
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Lengthy Source (pdf)"));

    await waitFor(() => {
      expect(
        screen.getByTestId("generate-question-count-guidance"),
      ).toHaveTextContent(
        "Estimated source length: ~2,400 words. Recommended: 10-20 questions.",
      );
    });

    expect(
      screen.getByTestId("generate-difficulty-guidance"),
    ).toHaveTextContent("Easy: Basic recall and straightforward definitions.");
  }, 20000);

  it("can generate quiz and flashcards together and expose both destination links", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 77, title: "Biology Source", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.getMediaDetails).mockImplementation(
      async (_mediaId, options) => {
        if (options?.include_content) {
          return {
            content: {
              text: "Mitosis and meiosis are core processes in cell division.",
            },
          } as any;
        }
        return { content: { word_count: 1200 } } as any;
      },
    );
    vi.mocked(generateFlashcards).mockResolvedValue({
      flashcards: [
        { front: "Mitosis", back: "Cell division for growth and repair" },
        { front: "Meiosis", back: "Cell division creating gametes" },
      ],
      count: 2,
    } as any);
    vi.mocked(createDeck).mockResolvedValue({
      id: 55,
      name: "Biology Source - Flashcards",
    } as any);
    vi.mocked(createFlashcard).mockResolvedValue({
      uuid: "card-1",
    } as any);

    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 99, name: "Biology Mastery" },
      questions: [{ id: 1 }, { id: 2 }],
    }));
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false,
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Biology Source (pdf)"));
    fireEvent.click(screen.getByTestId("generate-study-materials-toggle"));
    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(screen.getByTestId("generate-preview-card")).toBeInTheDocument();
      expect(
        screen.getByTestId("generate-study-materials-summary"),
      ).toBeInTheDocument();
    });

    expect(generateFlashcards).toHaveBeenCalledWith(
      expect.objectContaining({
        num_cards: 10,
        difficulty: "mixed",
      }),
      { signal: expect.any(AbortSignal) },
    );
    expect(createDeck).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Biology Mastery - Flashcards",
      }),
      { signal: expect.any(AbortSignal) },
    );
    expect(createFlashcard).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByTestId("generate-open-flashcards-button"));
    expect(navigationMocks.navigate).toHaveBeenCalledWith(
      expect.stringContaining(
        "/flashcards?tab=review&study_source=quiz&quiz_id=99&deck_id=55",
      ),
    );
  }, 20000);

  it("surfaces fallback handoff when combined flashcard generation cannot extract source content", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 88, title: "Sparse Source", type: "pdf" }],
      pagination: { total_items: 1 },
    } as any);
    vi.mocked(tldwClient.getMediaDetails).mockImplementation(
      async (_mediaId, options) => {
        if (options?.include_content) {
          return {} as any;
        }
        return { content: { word_count: 400 } } as any;
      },
    );
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        quiz: { id: 33, name: "Sparse Quiz" },
        questions: [{ id: 1 }],
      })),
      isPending: false,
    } as any);

    renderWithQueryClient();

    await waitFor(() => {
      expect(screen.getByText("1 media items available")).toBeInTheDocument();
    });

    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    fireEvent.click(await screen.findByText("Sparse Source (pdf)"));
    fireEvent.click(screen.getByTestId("generate-study-materials-toggle"));
    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }));

    await waitFor(() => {
      expect(screen.getByTestId("generate-preview-card")).toBeInTheDocument();
      expect(
        screen.getByTestId("generate-study-materials-summary"),
      ).toBeInTheDocument();
    });

    expect(generateFlashcards).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("generate-continue-flashcards-button"));
    expect(navigationMocks.navigate).toHaveBeenCalledWith(
      "/flashcards?tab=importExport",
    );
  }, 20000);
});
