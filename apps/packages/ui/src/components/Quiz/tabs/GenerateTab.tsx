import React from "react";
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Form,
  InputNumber,
  Select,
  Space,
  Spin,
  Tooltip,
  message,
} from "antd";
import { useTranslation } from "react-i18next";
import {
  CloseOutlined,
  InfoCircleOutlined,
  ReloadOutlined,
  RocketOutlined,
  StopOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useGenerateQuizMutation } from "../hooks";
import { useDebounce } from "@/hooks/useDebounce";
import { tldwClient } from "@/services/tldw";
import type { QuestionType, QuizGenerateSource } from "@/services/quizzes";
import {
  createDeck,
  createFlashcard,
  generateFlashcards,
  listDecks,
  listFlashcards,
  type FlashcardGeneratedDraft,
} from "@/services/flashcards";
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff";
import { buildFlashcardsStudyRouteFromQuiz } from "@/services/tldw/quiz-flashcards-handoff";
import type { TakeTabNavigationIntent } from "../navigation";
import type { QuizQuestionPlanItem } from "@/services/quizzes";

interface GenerateTabProps {
  onNavigateToTake: (intent?: TakeTabNavigationIntent) => void;
  onNavigateToManage?: () => void;
}

interface MediaItem {
  id: number;
  title: string;
  type: string;
}

interface MediaListResponse {
  items: MediaItem[];
  total: number | null;
}

interface NoteItem {
  id: string;
  title: string;
}

interface DeckItem {
  id: number;
  name: string;
}

interface CardItem {
  id: string;
  label: string;
  deckId: number;
}

type GeneratedPreview = {
  quizId: number;
  quizName: string;
  questionCount: number;
  flashcardsSummary: FlashcardsSummary | null;
};

type FlashcardsSummary = {
  status: "success" | "partial" | "failed";
  deckId?: number;
  deckName?: string;
  generatedCount: number;
  savedCount: number;
  failedCount: number;
  errorDetail?: string | null;
  handoffRoute?: string;
};

const MEDIA_PAGE_SIZE = 50;
const MAX_FLASHCARDS_IN_STUDY_FLOW = 30;
const MAX_FLASHCARD_SOURCE_TEXT_CHARS = 20_000;

type QuestionPlanRowState = QuizQuestionPlanItem & {
  enabled: boolean;
  labelKey: string;
  labelDefault: string;
};

type SelectedSourceSummary = {
  key: string;
  typeLabel: string;
  label: string;
  onRemove: () => void;
};

const DEFAULT_QUESTION_PLAN_ROWS: QuestionPlanRowState[] = [
  {
    question_type: "multiple_choice",
    labelKey: "option:quiz.questionTypeMultipleChoice",
    labelDefault: "Multiple Choice",
    enabled: true,
    count: 5,
    option_count: 4,
  },
  {
    question_type: "true_false",
    labelKey: "option:quiz.questionTypeTrueFalse",
    labelDefault: "True/False",
    enabled: true,
    count: 3,
  },
  {
    question_type: "fill_blank",
    labelKey: "option:quiz.questionTypeFillBlank",
    labelDefault: "Fill in the Blank",
    enabled: true,
    count: 2,
  },
  {
    question_type: "multi_select",
    labelKey: "option:quiz.questionTypeMultiSelect",
    labelDefault: "Multi-select",
    enabled: false,
    count: 1,
    option_count: 4,
  },
  {
    question_type: "matching",
    labelKey: "option:quiz.questionTypeMatching",
    labelDefault: "Matching",
    enabled: false,
    count: 1,
    pair_count: 4,
  },
];

const sanitizeInputNumber = (
  value: number | string | null,
  min: number,
  max: number,
): number | null => {
  const next =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value)
        : null;
  if (next == null || !Number.isFinite(next)) return null;
  return Math.min(max, Math.max(min, Math.round(next)));
};

const DIFFICULTY_OPTIONS: Array<{
  label: string;
  value: "easy" | "medium" | "hard" | "mixed";
  description: string;
}> = [
  {
    label: "Easy",
    value: "easy",
    description: "Basic recall and straightforward definitions.",
  },
  {
    label: "Medium",
    value: "medium",
    description: "Concept application and moderate reasoning.",
  },
  {
    label: "Hard",
    value: "hard",
    description: "Multi-step reasoning and subtle distinctions.",
  },
  {
    label: "Mixed",
    value: "mixed",
    description: "Balanced blend of easy, medium, and hard.",
  },
];

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") return null;
  return value as Record<string, unknown>;
};

const asString = (value: unknown): string | null => {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
};

const asNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const isAbortError = (error: unknown): boolean => {
  if (error instanceof Error && error.name === "AbortError") return true;
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "";
  return message.toLowerCase().includes("abort");
};

const isFormValidationError = (error: unknown): boolean => {
  const record = asRecord(error);
  return Array.isArray(record?.errorFields);
};

const normalizeFocusTopics = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  const unique = new Set<string>();
  value.forEach((topic) => {
    if (typeof topic !== "string") return;
    const trimmed = topic.trim();
    if (trimmed) unique.add(trimmed);
  });
  return Array.from(unique);
};

const extractErrorDetail = (error: unknown): string | null => {
  const extract = (value: unknown): string | null => {
    if (!value) return null;
    if (typeof value === "string") return value.trim() || null;
    if (Array.isArray(value)) {
      for (const entry of value) {
        const detail = extract(entry);
        if (detail) return detail;
      }
      return null;
    }
    const record = asRecord(value);
    if (!record) return null;
    return (
      extract(record.detail) ??
      extract(record.message) ??
      extract(record.msg) ??
      null
    );
  };

  if (error instanceof Error) {
    const message = error.message.trim();
    if (message && !/failed to generate quiz/i.test(message)) {
      return message;
    }
  }

  const record = asRecord(error);
  if (!record) return null;

  return (
    extract(record.detail) ??
    extract(record.error) ??
    extract(asRecord(record.response)?.data) ??
    extract(record.message) ??
    null
  );
};

const extractWordCount = (details: unknown): number | null => {
  const record = asRecord(details);
  if (!record) return null;
  return (
    asNumber(asRecord(record.content)?.word_count) ??
    asNumber(
      asRecord(asRecord(record.processing)?.safe_metadata)?.word_count,
    ) ??
    asNumber(asRecord(record.metadata)?.word_count) ??
    null
  );
};

const normalizeMediaListResponse = (raw: unknown): MediaListResponse => {
  const record = asRecord(raw);
  const rawItems =
    record?.items ?? record?.media ?? record?.results ?? record?.data ?? [];
  const array = Array.isArray(rawItems) ? rawItems : [];

  const items = array
    .map((entry) => {
      const item = asRecord(entry);
      if (!item) return null;
      const id = asNumber(item.id ?? item.media_id);
      if (id == null) return null;
      return {
        id,
        title: asString(item.title) ?? asString(item.name) ?? `Media #${id}`,
        type: asString(item.type) ?? asString(item.media_type) ?? "unknown",
      } satisfies MediaItem;
    })
    .filter((item): item is MediaItem => item != null);

  const pagination = asRecord(record?.pagination);
  const total =
    asNumber(pagination?.total_items) ??
    asNumber(record?.total_items) ??
    asNumber(record?.count) ??
    null;

  return { items, total };
};

const normalizeNoteListResponse = (raw: unknown): NoteItem[] => {
  const record = asRecord(raw);
  const rawItems =
    record?.items ?? record?.notes ?? record?.results ?? record?.data ?? raw;
  const array = Array.isArray(rawItems) ? rawItems : [];
  const seen = new Set<string>();

  return array
    .map((entry) => {
      const item = asRecord(entry);
      if (!item) return null;
      const id = asString(item.id ?? item.note_id);
      if (!id || seen.has(id)) return null;
      seen.add(id);
      return {
        id,
        title: asString(item.title) ?? asString(item.name) ?? `Note ${id}`,
      } satisfies NoteItem;
    })
    .filter((item): item is NoteItem => item != null);
};

const normalizeDeckListResponse = (raw: unknown): DeckItem[] => {
  const array = Array.isArray(raw) ? raw : [];
  const seen = new Set<number>();

  return array
    .map((entry) => {
      const item = asRecord(entry);
      if (!item) return null;
      const id = asNumber(item.id);
      if (id == null || id <= 0 || seen.has(id)) return null;
      seen.add(id);
      return {
        id,
        name: asString(item.name) ?? `Deck ${id}`,
      } satisfies DeckItem;
    })
    .filter((item): item is DeckItem => item != null);
};

const normalizeFlashcardListResponse = (
  raw: unknown,
  deckNames: Map<number, string>,
): CardItem[] => {
  const record = asRecord(raw);
  const rawItems = record?.items ?? record?.results ?? record?.data ?? [];
  const array = Array.isArray(rawItems) ? rawItems : [];
  const seen = new Set<string>();

  return array
    .map((entry) => {
      const item = asRecord(entry);
      if (!item) return null;
      const id = asString(item.uuid ?? item.id);
      if (!id || seen.has(id)) return null;
      const deckId = asNumber(item.deck_id);
      if (deckId == null || deckId <= 0) return null;
      seen.add(id);
      const front = asString(item.front) ?? "";
      const back = asString(item.back) ?? "";
      const preview = [front, back].filter(Boolean).join(" - ");
      const deckName = deckNames.get(deckId) ?? `Deck ${deckId}`;
      return {
        id,
        deckId,
        label: preview ? `${deckName}: ${preview}` : `${deckName}: ${id}`,
      } satisfies CardItem;
    })
    .filter((item): item is CardItem => item != null);
};

const getFirstNonEmptyString = (...values: unknown[]): string => {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
  }
  return "";
};

const extractMediaText = (details: unknown): string => {
  if (typeof details === "string") return details.trim();
  const record = asRecord(details);
  if (!record) return "";

  const content = record.content;
  if (typeof content === "string" && content.trim().length > 0) {
    return content.trim();
  }
  const contentRecord = asRecord(content);
  if (contentRecord) {
    const nested = getFirstNonEmptyString(
      contentRecord.text,
      contentRecord.content,
      contentRecord.raw_text,
      contentRecord.rawText,
      contentRecord.transcript,
      contentRecord.summary,
    );
    if (nested) return nested;
  }

  const fromRoot = getFirstNonEmptyString(
    record.text,
    record.transcript,
    record.raw_text,
    record.rawText,
    record.raw_content,
    record.rawContent,
    record.summary,
  );
  if (fromRoot) return fromRoot;

  const latestVersion =
    asRecord(record.latest_version) ?? asRecord(record.latestVersion);
  if (latestVersion) {
    const fromLatest = getFirstNonEmptyString(
      latestVersion.content,
      latestVersion.text,
      latestVersion.transcript,
      latestVersion.raw_text,
      latestVersion.rawText,
      latestVersion.summary,
    );
    if (fromLatest) return fromLatest;
  }

  const data = asRecord(record.data);
  if (data) {
    const fromData = getFirstNonEmptyString(
      data.content,
      data.text,
      data.transcript,
      data.raw_text,
      data.rawText,
      data.summary,
    );
    if (fromData) return fromData;
  }

  return "";
};

const clampFlashcardsCount = (questionCount: number): number =>
  Math.max(
    3,
    Math.min(MAX_FLASHCARDS_IN_STUDY_FLOW, Math.round(questionCount)),
  );

const normalizeGeneratedDrafts = (
  drafts: FlashcardGeneratedDraft[] | null | undefined,
): FlashcardGeneratedDraft[] => {
  if (!Array.isArray(drafts)) return [];
  return drafts.filter((draft) => {
    const front = typeof draft.front === "string" ? draft.front.trim() : "";
    const back = typeof draft.back === "string" ? draft.back.trim() : "";
    return front.length > 0 && back.length > 0;
  });
};

const buildGeneratedDeckName = (quizName: string): string => {
  const trimmed = quizName.trim();
  if (!trimmed) return "Generated Study Deck";
  return `${trimmed} - Flashcards`;
};

export const GenerateTab: React.FC<GenerateTabProps> = ({
  onNavigateToTake,
  onNavigateToManage,
}) => {
  const { t } = useTranslation(["option", "common", "settings"]);
  const navigate = useNavigate();
  const [form] = Form.useForm();
  const [selectedMediaId, setSelectedMediaId] = React.useState<number | null>(
    null,
  );
  const [selectedNoteIds, setSelectedNoteIds] = React.useState<string[]>([]);
  const [selectedDeckIds, setSelectedDeckIds] = React.useState<number[]>([]);
  const [selectedCardIds, setSelectedCardIds] = React.useState<string[]>([]);
  const [messageApi, contextHolder] = message.useMessage();
  const [mediaSearchInput, setMediaSearchInput] = React.useState("");
  const [notesSearchInput, setNotesSearchInput] = React.useState("");
  const [mediaPage, setMediaPage] = React.useState(1);
  const [loadedMediaItems, setLoadedMediaItems] = React.useState<MediaItem[]>(
    [],
  );
  const [mediaTotal, setMediaTotal] = React.useState<number | null>(null);
  const [selectedMediaWordCount, setSelectedMediaWordCount] = React.useState<
    number | null
  >(null);
  const [generationInFlight, setGenerationInFlight] = React.useState(false);
  const [generatedPreview, setGeneratedPreview] =
    React.useState<GeneratedPreview | null>(null);
  const [questionPlanRows, setQuestionPlanRows] = React.useState<
    QuestionPlanRowState[]
  >(() => DEFAULT_QUESTION_PLAN_ROWS.map((row) => ({ ...row })));
  const debouncedMediaSearch = useDebounce(mediaSearchInput, 300);
  const debouncedNotesSearch = useDebounce(notesSearchInput, 300);
  const generateAbortRef = React.useRef<AbortController | null>(null);
  const selectedDifficulty = Form.useWatch("difficulty", form) ?? "mixed";
  const shouldGenerateStudyMaterials = Boolean(
    Form.useWatch("generateStudyMaterials", form),
  );

  const generateMutation = useGenerateQuizMutation();

  const {
    data: mediaPageData,
    isLoading: isLoadingList,
    isFetching: isFetchingMedia,
    error: listError,
    refetch: refetchMedia,
  } = useQuery<MediaListResponse>({
    queryKey: ["quiz-generate-media-list", debouncedMediaSearch, mediaPage],
    queryFn: async () => {
      const searchTerm = debouncedMediaSearch.trim();
      if (searchTerm) {
        const response = await tldwClient.searchMedia(
          { query: searchTerm },
          { page: mediaPage, results_per_page: MEDIA_PAGE_SIZE },
        );
        return normalizeMediaListResponse(response);
      }
      const response = await tldwClient.listMedia({
        page: mediaPage,
        results_per_page: MEDIA_PAGE_SIZE,
      });
      return normalizeMediaListResponse(response);
    },
    placeholderData: (previousData) => previousData,
    staleTime: 60 * 1000,
  });

  const {
    data: notesData = [],
    isLoading: isLoadingNotes,
    error: notesError,
    refetch: refetchNotes,
  } = useQuery<NoteItem[]>({
    queryKey: ["quiz-generate-note-list", debouncedNotesSearch],
    queryFn: async () => {
      const searchTerm = debouncedNotesSearch.trim();
      if (searchTerm) {
        const response = await tldwClient.searchNotes(searchTerm);
        return normalizeNoteListResponse(response);
      }
      const response = await tldwClient.listNotes({
        page: 1,
        results_per_page: 200,
        include_keywords: false,
      });
      return normalizeNoteListResponse(response);
    },
    staleTime: 60 * 1000,
  });

  const {
    data: decksData = [],
    isLoading: isLoadingDecks,
    error: decksError,
    refetch: refetchDecks,
  } = useQuery<DeckItem[]>({
    queryKey: ["quiz-generate-decks"],
    queryFn: async () => {
      const decks = await listDecks();
      return normalizeDeckListResponse(decks);
    },
    staleTime: 60 * 1000,
  });

  const {
    data: cardsData = [],
    isFetching: isFetchingCards,
    error: cardsError,
    refetch: refetchCards,
  } = useQuery<CardItem[]>({
    queryKey: ["quiz-generate-cards-by-deck", selectedDeckIds],
    queryFn: async () => {
      const deckNames = new Map<number, string>();
      decksData.forEach((deck) => {
        deckNames.set(deck.id, deck.name);
      });

      const responses = await Promise.all(
        selectedDeckIds.map((deckId) =>
          listFlashcards({
            deck_id: deckId,
            due_status: "all",
            limit: 200,
            offset: 0,
            order_by: "created_at",
          }),
        ),
      );

      const merged: CardItem[] = [];
      const seen = new Set<string>();
      responses.forEach((response) => {
        normalizeFlashcardListResponse(response, deckNames).forEach((card) => {
          if (seen.has(card.id)) return;
          seen.add(card.id);
          merged.push(card);
        });
      });
      return merged;
    },
    enabled: selectedDeckIds.length > 0,
    staleTime: 60 * 1000,
  });

  React.useEffect(() => {
    setMediaPage(1);
    setLoadedMediaItems([]);
    setMediaTotal(null);
  }, [debouncedMediaSearch]);

  React.useEffect(() => {
    if (!mediaPageData) return;
    setMediaTotal(mediaPageData.total);
    setLoadedMediaItems((prev) => {
      if (mediaPage === 1) {
        return mediaPageData.items;
      }
      const map = new Map<number, MediaItem>();
      for (const item of prev) {
        map.set(item.id, item);
      }
      for (const item of mediaPageData.items) {
        map.set(item.id, item);
      }
      return Array.from(map.values());
    });
  }, [mediaPage, mediaPageData]);

  React.useEffect(() => {
    if (selectedMediaId == null) {
      setSelectedMediaWordCount(null);
      return;
    }

    const controller = new AbortController();
    void (async () => {
      try {
        const details = await tldwClient.getMediaDetails(selectedMediaId, {
          include_content: false,
          include_versions: false,
          include_version_content: false,
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setSelectedMediaWordCount(extractWordCount(details));
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) return;
        setSelectedMediaWordCount(null);
      }
    })();

    return () => {
      controller.abort();
    };
  }, [selectedMediaId]);

  React.useEffect(() => {
    if (selectedMediaId != null) return;
    if (!form.getFieldValue("generateStudyMaterials")) return;
    form.setFieldsValue({ generateStudyMaterials: false });
  }, [form, selectedMediaId]);

  React.useEffect(() => {
    if (selectedDeckIds.length === 0) {
      setSelectedCardIds((prev) => (prev.length === 0 ? prev : []));
      return;
    }
    const availableCardIds = new Set(cardsData.map((card) => card.id));
    setSelectedCardIds((prev) => {
      const next = prev.filter((cardId) => availableCardIds.has(cardId));
      if (
        next.length === prev.length &&
        next.every((id, index) => id === prev[index])
      ) {
        return prev;
      }
      return next;
    });
  }, [cardsData, selectedDeckIds]);

  React.useEffect(() => {
    return () => {
      generateAbortRef.current?.abort();
    };
  }, []);

  const hasMoreMedia =
    mediaTotal != null
      ? loadedMediaItems.length < mediaTotal
      : (mediaPageData?.items.length ?? 0) >= MEDIA_PAGE_SIZE;

  const isLoadingMoreMedia = isFetchingMedia && mediaPage > 1;

  const mediaOptions = React.useMemo(() => {
    const options = loadedMediaItems.map((item) => ({
      value: item.id,
      label: `${item.title || `Media #${item.id}`} (${item.type})`,
    }));
    if (
      selectedMediaId != null &&
      !options.some((option) => option.value === selectedMediaId)
    ) {
      options.unshift({
        value: selectedMediaId,
        label: t("option:quiz.sourceMedia", {
          defaultValue: "Source media #{{id}}",
          id: selectedMediaId,
        }),
      });
    }
    return options;
  }, [loadedMediaItems, selectedMediaId, t]);

  const noteOptions = React.useMemo(
    () =>
      notesData.map((item) => ({
        value: item.id,
        label: item.title,
      })),
    [notesData],
  );

  const deckOptions = React.useMemo(
    () =>
      decksData.map((deck) => ({
        value: deck.id,
        label: deck.name,
      })),
    [decksData],
  );

  const cardOptions = React.useMemo(
    () =>
      cardsData.map((card) => ({
        value: card.id,
        label: card.label,
      })),
    [cardsData],
  );

  const difficultyOptions = React.useMemo(
    () =>
      DIFFICULTY_OPTIONS.map((option) => ({
        value: option.value,
        label: t(`option:quiz.difficulty.${option.value}`, {
          defaultValue: option.label,
        }),
        description: t(`option:quiz.difficulty.${option.value}.description`, {
          defaultValue: option.description,
        }),
      })),
    [t],
  );

  const selectedSources = React.useMemo<QuizGenerateSource[]>(() => {
    const sourceMap = new Map<string, QuizGenerateSource>();
    const put = (source: QuizGenerateSource) => {
      sourceMap.set(`${source.source_type}:${source.source_id}`, source);
    };

    if (selectedMediaId != null) {
      put({ source_type: "media", source_id: String(selectedMediaId) });
    }
    selectedNoteIds.forEach((noteId) => {
      put({ source_type: "note", source_id: noteId });
    });
    selectedDeckIds.forEach((deckId) => {
      put({ source_type: "flashcard_deck", source_id: String(deckId) });
    });
    selectedCardIds.forEach((cardId) => {
      put({ source_type: "flashcard_card", source_id: cardId });
    });
    return Array.from(sourceMap.values());
  }, [selectedCardIds, selectedDeckIds, selectedMediaId, selectedNoteIds]);

  const hasSelectedSources = selectedSources.length > 0;

  const selectedSourcesLabel = React.useMemo(() => {
    if (selectedSources.length === 0) {
      return t("option:quiz.noSourcesSelected", {
        defaultValue: "No sources selected",
      });
    }
    if (selectedSources.length === 1) {
      return t("option:quiz.oneSourceSelected", {
        defaultValue: "1 source selected",
      });
    }
    return t("option:quiz.sourcesSelected", {
      defaultValue: "{{count}} sources selected",
      count: selectedSources.length,
    });
  }, [selectedSources.length, t]);

  const selectedMedia = React.useMemo(
    () =>
      selectedMediaId == null
        ? null
        : (loadedMediaItems.find((item) => item.id === selectedMediaId) ??
          null),
    [loadedMediaItems, selectedMediaId],
  );

  const selectedSourceSummaries = React.useMemo<SelectedSourceSummary[]>(() => {
    const noteLabelById = new Map(
      noteOptions.map((option) => [option.value, String(option.label)]),
    );
    const deckLabelById = new Map(
      deckOptions.map((option) => [option.value, String(option.label)]),
    );
    const cardLabelById = new Map(
      cardOptions.map((option) => [option.value, String(option.label)]),
    );
    const items: SelectedSourceSummary[] = [];

    if (selectedMediaId != null) {
      const label = selectedMedia?.title || `Media #${selectedMediaId}`;
      items.push({
        key: `media:${selectedMediaId}`,
        typeLabel: t("option:quiz.mediaSources", { defaultValue: "Media" }),
        label,
        onRemove: () => setSelectedMediaId(null),
      });
    }

    selectedNoteIds.forEach((noteId) => {
      const label = noteLabelById.get(noteId) ?? `Note ${noteId}`;
      items.push({
        key: `note:${noteId}`,
        typeLabel: t("option:quiz.noteSources", { defaultValue: "Notes" }),
        label,
        onRemove: () =>
          setSelectedNoteIds((current) =>
            current.filter((id) => id !== noteId),
          ),
      });
    });

    selectedDeckIds.forEach((deckId) => {
      const label = deckLabelById.get(deckId) ?? `Deck ${deckId}`;
      items.push({
        key: `flashcard_deck:${deckId}`,
        typeLabel: t("option:quiz.deckSources", {
          defaultValue: "Flashcard Decks",
        }),
        label,
        onRemove: () =>
          setSelectedDeckIds((current) =>
            current.filter((id) => id !== deckId),
          ),
      });
    });

    selectedCardIds.forEach((cardId) => {
      const label = cardLabelById.get(cardId) ?? `Flashcard ${cardId}`;
      items.push({
        key: `flashcard_card:${cardId}`,
        typeLabel: t("option:quiz.cardSources", { defaultValue: "Flashcards" }),
        label,
        onRemove: () =>
          setSelectedCardIds((current) =>
            current.filter((id) => id !== cardId),
          ),
      });
    });

    return items;
  }, [
    cardOptions,
    deckOptions,
    noteOptions,
    selectedCardIds,
    selectedDeckIds,
    selectedMedia,
    selectedMediaId,
    selectedNoteIds,
    t,
  ]);

  const enabledPlanRows = React.useMemo<QuizQuestionPlanItem[]>(
    () =>
      questionPlanRows
        .filter((row) => row.enabled)
        .map((row) => {
          const item: QuizQuestionPlanItem = {
            question_type: row.question_type,
            count: row.count,
          };
          if (row.option_count != null) item.option_count = row.option_count;
          if (row.pair_count != null) item.pair_count = row.pair_count;
          return item;
        }),
    [questionPlanRows],
  );

  const totalQuestions = React.useMemo(
    () => enabledPlanRows.reduce((sum, row) => sum + row.count, 0),
    [enabledPlanRows],
  );

  const generateBlockReason = React.useMemo(() => {
    if (!hasSelectedSources) {
      return t("option:quiz.generateBlockedNoSources", {
        defaultValue: "Select at least one source to generate a quiz.",
      });
    }
    if (totalQuestions === 0) {
      return t("option:quiz.generateBlockedNoQuestions", {
        defaultValue: "Enable at least one question type.",
      });
    }
    if (totalQuestions > 100) {
      return t("option:quiz.generateBlockedTooManyQuestions", {
        defaultValue: "Reduce the mix to 100 questions or fewer.",
      });
    }
    return null;
  }, [hasSelectedSources, totalQuestions, t]);

  const canGenerate = !generateBlockReason && !generationInFlight;

  const updateQuestionPlanRow = React.useCallback(
    (
      questionType: QuestionType,
      patch: Partial<
        Omit<QuestionPlanRowState, "question_type" | "labelKey" | "labelDefault">
      >,
    ) => {
      setQuestionPlanRows((rows) =>
        rows.map((row) =>
          row.question_type === questionType ? { ...row, ...patch } : row,
        ),
      );
    },
    [],
  );

  const questionCountRecommendation = React.useMemo(() => {
    if (!selectedMediaWordCount || selectedMediaWordCount <= 0) {
      return t("option:quiz.questionCountRecommendation", {
        defaultValue: "Recommended: 5-10 questions per 1,000 words of source.",
      });
    }

    const units = Math.max(1, Math.round(selectedMediaWordCount / 1000));
    const minQuestions = Math.min(50, Math.max(5, units * 5));
    const maxQuestions = Math.min(50, Math.max(minQuestions, units * 10));

    return t("option:quiz.questionCountRecommendationSized", {
      defaultValue:
        "Estimated source length: ~{{wordCount}} words. Recommended: {{minQuestions}}-{{maxQuestions}} questions.",
      wordCount: selectedMediaWordCount.toLocaleString(),
      minQuestions,
      maxQuestions,
    });
  }, [selectedMediaWordCount, t]);

  const handleCancelGeneration = React.useCallback(() => {
    if (!generationInFlight) return;
    generateAbortRef.current?.abort();
  }, [generationInFlight]);

  const generateStudyMaterialsFlashcards = React.useCallback(
    async (params: {
      mediaId: number;
      mediaTitle: string;
      quizName: string;
      numQuestions: number;
      difficulty?: "easy" | "medium" | "hard" | "mixed";
      focusTopics: string[];
      signal?: AbortSignal;
    }): Promise<FlashcardsSummary> => {
      const fallbackRoute = "/flashcards?tab=importExport";
      const throwIfAborted = () => {
        if (params.signal?.aborted) {
          const abortError = new Error("aborted");
          abortError.name = "AbortError";
          throw abortError;
        }
      };

      throwIfAborted();

      try {
        const details = await tldwClient.getMediaDetails(params.mediaId, {
          include_content: true,
          include_versions: false,
          include_version_content: false,
          signal: params.signal,
        });
        throwIfAborted();

        const sourceText = extractMediaText(details).slice(
          0,
          MAX_FLASHCARD_SOURCE_TEXT_CHARS,
        );
        if (!sourceText) {
          return {
            status: "failed",
            generatedCount: 0,
            savedCount: 0,
            failedCount: 0,
            errorDetail: t("option:quiz.studyMaterialsMissingSourceText", {
              defaultValue:
                "Could not extract enough source text to generate flashcards.",
            }),
            handoffRoute: fallbackRoute,
          };
        }

        const handoffRoute = buildFlashcardsGenerateRoute({
          text: sourceText,
          sourceType: "media",
          sourceId: String(params.mediaId),
          sourceTitle: params.mediaTitle,
        });

        const generated = await generateFlashcards(
          {
            text: sourceText,
            num_cards: clampFlashcardsCount(params.numQuestions),
            difficulty: params.difficulty,
            focus_topics:
              params.focusTopics.length > 0 ? params.focusTopics : undefined,
          },
          {
            signal: params.signal,
          },
        );
        throwIfAborted();

        const drafts = normalizeGeneratedDrafts(generated.flashcards);
        if (drafts.length === 0) {
          return {
            status: "failed",
            generatedCount: 0,
            savedCount: 0,
            failedCount: 0,
            errorDetail: t("option:quiz.studyMaterialsEmptyFlashcards", {
              defaultValue: "Flashcard generation returned no usable cards.",
            }),
            handoffRoute,
          };
        }

        const deck = await createDeck(
          {
            name: buildGeneratedDeckName(params.quizName),
            description: t("option:quiz.studyMaterialsDeckDescription", {
              defaultValue: "Generated from {{title}}",
              title: params.mediaTitle,
            }),
          },
          {
            signal: params.signal,
          },
        );
        throwIfAborted();

        const createResults = await Promise.allSettled(
          drafts.map((draft) =>
            createFlashcard(
              {
                deck_id: deck.id,
                front: draft.front,
                back: draft.back,
                tags: draft.tags,
                notes: draft.notes ?? undefined,
                extra: draft.extra ?? undefined,
                model_type: draft.model_type ?? "basic",
                reverse: draft.model_type === "basic_reverse",
                is_cloze: draft.model_type === "cloze",
                source_ref_type: "media",
                source_ref_id: String(params.mediaId),
              },
              {
                signal: params.signal,
              },
            ),
          ),
        );
        throwIfAborted();

        const savedCount = createResults.filter(
          (result) => result.status === "fulfilled",
        ).length;
        const failedCount = drafts.length - savedCount;
        const status: FlashcardsSummary["status"] =
          savedCount === 0 ? "failed" : failedCount > 0 ? "partial" : "success";

        return {
          status,
          deckId: deck.id,
          deckName: deck.name,
          generatedCount: drafts.length,
          savedCount,
          failedCount,
          errorDetail:
            status === "failed"
              ? t("option:quiz.studyMaterialsSaveFailed", {
                  defaultValue: "Unable to save generated flashcards.",
                })
              : null,
          handoffRoute,
        };
      } catch (error) {
        if (isAbortError(error)) throw error;
        return {
          status: "failed",
          generatedCount: 0,
          savedCount: 0,
          failedCount: 0,
          errorDetail:
            extractErrorDetail(error) ??
            t("option:quiz.studyMaterialsFailed", {
              defaultValue:
                "Failed to generate flashcards from the selected source.",
            }),
          handoffRoute: fallbackRoute,
        };
      }
    },
    [t],
  );

  const handleGenerate = async () => {
    if (generationInFlight) {
      return;
    }

    if (!hasSelectedSources) {
      messageApi.warning(
        t("option:quiz.selectAtLeastOneSource", {
          defaultValue: "Select at least one source before generating.",
        }),
      );
      return;
    }

    if (totalQuestions === 0 || totalQuestions > 100) {
      return;
    }

    let requestAbortController: AbortController | null = null;

    try {
      const values = await form.validateFields();
      setGeneratedPreview(null);

      const focusTopics = normalizeFocusTopics(values.focusTopics);
      const shouldGenerateStudyMaterials = Boolean(
        values.generateStudyMaterials,
      );
      requestAbortController = new AbortController();
      generateAbortRef.current = requestAbortController;
      setGenerationInFlight(true);

      const generated = await generateMutation.mutateAsync({
        request: {
          sources: selectedSources,
          num_questions: totalQuestions,
          question_plan: enabledPlanRows,
          difficulty: values.difficulty,
          focus_topics: focusTopics.length > 0 ? focusTopics : undefined,
        },
        signal: requestAbortController.signal,
      });
      if (requestAbortController.signal.aborted) return;

      const generatedQuizName =
        generated.quiz.name || `Quiz #${generated.quiz.id}`;
      let flashcardsSummary: FlashcardsSummary | null = null;

      if (shouldGenerateStudyMaterials) {
        if (selectedMediaId == null) {
          flashcardsSummary = {
            status: "failed",
            generatedCount: 0,
            savedCount: 0,
            failedCount: 0,
            errorDetail: t("option:quiz.studyMaterialsMediaRequired", {
              defaultValue:
                "Flashcard deck generation currently requires a selected media source.",
            }),
            handoffRoute: "/flashcards?tab=importExport",
          };
        } else {
          flashcardsSummary = await generateStudyMaterialsFlashcards({
            mediaId: selectedMediaId,
            mediaTitle: selectedMedia?.title || `Media #${selectedMediaId}`,
            quizName: generatedQuizName,
            numQuestions: totalQuestions,
            difficulty: values.difficulty,
            focusTopics,
            signal: requestAbortController.signal,
          });
        }
      }

      if (requestAbortController.signal.aborted) return;

      setGeneratedPreview({
        quizId: generated.quiz.id,
        quizName: generatedQuizName,
        questionCount: generated.questions.length,
        flashcardsSummary,
      });

      if (!flashcardsSummary) {
        messageApi.success(
          t("option:quiz.generateSuccessReview", {
            defaultValue: "Quiz generated. Review it before starting.",
          }),
        );
      } else if (flashcardsSummary.status === "success") {
        messageApi.success(
          t("option:quiz.generateStudyMaterialsSuccess", {
            defaultValue: "Quiz and flashcards generated successfully.",
          }),
        );
      } else if (flashcardsSummary.status === "partial") {
        messageApi.warning(
          t("option:quiz.generateStudyMaterialsPartial", {
            defaultValue: "Quiz generated. Some flashcards could not be saved.",
          }),
        );
      } else {
        messageApi.warning(
          t("option:quiz.generateStudyMaterialsFailedNotice", {
            defaultValue: "Quiz generated. Flashcard generation needs review.",
          }),
        );
      }
    } catch (error) {
      if (isFormValidationError(error)) return;
      if (requestAbortController?.signal.aborted || isAbortError(error)) {
        messageApi.info(
          t("option:quiz.generateCancelled", {
            defaultValue: "Quiz generation canceled.",
          }),
        );
        return;
      }

      const detail = extractErrorDetail(error);
      messageApi.error(
        detail
          ? t("option:quiz.generateErrorDetailed", {
              defaultValue: "Failed to generate quiz: {{detail}}",
              detail,
            })
          : t("option:quiz.generateError", {
              defaultValue: "Failed to generate quiz",
            }),
      );
    } finally {
      if (generateAbortRef.current === requestAbortController) {
        generateAbortRef.current = null;
      }
      setGenerationInFlight(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl">
      {contextHolder}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(22rem,0.85fr)] lg:items-start">
        <div className="space-y-6">
          <Card
            title={t("option:quiz.selectSources", {
              defaultValue: "Select Sources",
            })}
            size="small"
          >
            <div className="space-y-4">
              {!isLoadingList && loadedMediaItems.length === 0 && (
                <Alert
                  type="info"
                  showIcon
                  data-testid="quiz-generate-no-media"
                  title={t("option:quiz.generate.noMedia", {
                    defaultValue: "No media content found",
                  })}
                  description={
                    <>
                      {t("option:quiz.generate.noMediaHint", {
                        defaultValue:
                          "Import videos, articles, or documents in your ",
                      })}
                      <a href="/media">
                        {t("option:quiz.generate.mediaLibrary", {
                          defaultValue: "Media Library",
                        })}
                      </a>
                      {t("option:quiz.generate.noMediaSuffix", {
                        defaultValue: ", then return here to generate quizzes.",
                      })}
                    </>
                  }
                  className="mb-4"
                />
              )}

              <div
                className="rounded-md border border-border-subtle bg-surface2/50 p-3"
                data-testid="generate-selected-sources"
              >
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
                    {t("option:quiz.selectedSources", {
                      defaultValue: "Selected sources",
                    })}
                  </div>
                  <div className="text-xs text-text-muted">
                    {selectedSourcesLabel}
                  </div>
                </div>
                {selectedSourceSummaries.length > 0 ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {selectedSourceSummaries.map((source) => (
                      <span
                        key={source.key}
                        className="inline-flex max-w-full items-center gap-2 rounded-full border border-border bg-surface px-2 py-1 text-xs text-text"
                      >
                        <span className="shrink-0 rounded-full bg-surface2 px-1.5 py-0.5 font-medium text-text-muted">
                          {source.typeLabel}
                        </span>
                        <span className="min-w-0 truncate">{source.label}</span>
                        <Button
                          type="text"
                          size="small"
                          icon={<CloseOutlined />}
                          aria-label={t("option:quiz.removeSelectedSource", {
                            defaultValue: "Remove {{label}}",
                            label: source.label,
                          })}
                          disabled={generationInFlight}
                          onClick={source.onRemove}
                          className="!h-5 !w-5 !min-w-0 !p-0"
                        />
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-text-subtle">
                    {t("option:quiz.noSelectedSourcesHint", {
                      defaultValue:
                        "Choose media, notes, decks, or individual flashcards to ground this quiz.",
                    })}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium text-text-subtle">
                  {t("option:quiz.mediaSources", { defaultValue: "Media" })}
                </div>
                {listError ? (
                  <Alert
                    type="error"
                    title={t(
                      "settings:chunkingPlayground.loadMediaListError",
                      "Failed to load media library",
                    )}
                    action={
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={() => void refetchMedia()}
                      >
                        {t("common:retry", { defaultValue: "Retry" })}
                      </Button>
                    }
                  />
                ) : (
                  <>
                    <Select
                      showSearch
                      placeholder={t("option:quiz.selectMediaPlaceholder", {
                        defaultValue: "Select media item...",
                      })}
                      loading={isLoadingList && mediaPage === 1}
                      value={selectedMediaId}
                      onChange={(value) => setSelectedMediaId(value)}
                      onSearch={(value) => setMediaSearchInput(value)}
                      options={mediaOptions}
                      filterOption={false}
                      className="w-full"
                      disabled={generationInFlight}
                      notFoundContent={
                        isLoadingList && mediaPage === 1 ? (
                          <Spin size="small" />
                        ) : (
                          t("option:quiz.noMediaFound", {
                            defaultValue: "No media found",
                          })
                        )
                      }
                    />
                    {mediaTotal != null ? (
                      <div className="text-xs text-text-subtle">
                        {t("option:quiz.mediaCount", {
                          defaultValue:
                            loadedMediaItems.length < mediaTotal
                              ? "Showing {{loaded}} of {{count}} media items"
                              : "{{count}} media items available",
                          loaded: loadedMediaItems.length,
                          count: mediaTotal,
                        })}
                      </div>
                    ) : (
                      <div className="text-xs text-text-subtle">
                        {t("option:quiz.loadedMediaCount", {
                          defaultValue: "Showing {{count}} media items",
                          count: loadedMediaItems.length,
                        })}
                      </div>
                    )}
                    {hasMoreMedia && (
                      <Button
                        type="default"
                        size="small"
                        onClick={() => setMediaPage((prev) => prev + 1)}
                        loading={isLoadingMoreMedia}
                        disabled={isLoadingMoreMedia || generationInFlight}
                        data-testid="generate-media-load-more"
                      >
                        {t("common:loadMore", { defaultValue: "Load More" })}
                      </Button>
                    )}
                  </>
                )}
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium text-text-subtle">
                  {t("option:quiz.noteSources", { defaultValue: "Notes" })}
                </div>
                {notesError ? (
                  <Alert
                    type="error"
                    title={t("option:quiz.loadNotesError", {
                      defaultValue: "Failed to load notes",
                    })}
                    action={
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={() => void refetchNotes()}
                      >
                        {t("common:retry", { defaultValue: "Retry" })}
                      </Button>
                    }
                  />
                ) : (
                  <Select
                    mode="multiple"
                    showSearch
                    value={selectedNoteIds}
                    onChange={(values) => setSelectedNoteIds(values)}
                    onSearch={(value) => setNotesSearchInput(value)}
                    placeholder={t("option:quiz.selectNotesPlaceholder", {
                      defaultValue: "Select notes...",
                    })}
                    options={noteOptions}
                    loading={isLoadingNotes}
                    disabled={generationInFlight}
                    filterOption={false}
                    className="w-full"
                    data-testid="generate-note-select"
                    notFoundContent={
                      isLoadingNotes ? (
                        <Spin size="small" />
                      ) : (
                        t("option:quiz.noNotesFound", {
                          defaultValue: "No notes found",
                        })
                      )
                    }
                  />
                )}
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium text-text-subtle">
                  {t("option:quiz.deckSources", {
                    defaultValue: "Flashcard Decks",
                  })}
                </div>
                {decksError ? (
                  <Alert
                    type="error"
                    title={t("option:quiz.loadDecksError", {
                      defaultValue: "Failed to load flashcard decks",
                    })}
                    action={
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={() => void refetchDecks()}
                      >
                        {t("common:retry", { defaultValue: "Retry" })}
                      </Button>
                    }
                  />
                ) : (
                  <Select
                    mode="multiple"
                    value={selectedDeckIds}
                    onChange={(values) => setSelectedDeckIds(values)}
                    placeholder={t("option:quiz.selectDecksPlaceholder", {
                      defaultValue: "Select flashcard decks...",
                    })}
                    options={deckOptions}
                    loading={isLoadingDecks}
                    disabled={generationInFlight}
                    className="w-full"
                    data-testid="generate-deck-select"
                  />
                )}
              </div>

              <div className="space-y-2">
                <div className="text-xs font-medium text-text-subtle">
                  {t("option:quiz.cardSources", { defaultValue: "Flashcards" })}
                </div>
                {cardsError ? (
                  <Alert
                    type="warning"
                    title={t("option:quiz.loadCardsError", {
                      defaultValue:
                        "Failed to load flashcards for selected decks",
                    })}
                    action={
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={() => void refetchCards()}
                      >
                        {t("common:retry", { defaultValue: "Retry" })}
                      </Button>
                    }
                  />
                ) : (
                  <Select
                    mode="multiple"
                    value={selectedCardIds}
                    onChange={(values) => setSelectedCardIds(values)}
                    placeholder={t("option:quiz.selectCardsPlaceholder", {
                      defaultValue:
                        selectedDeckIds.length > 0
                          ? "Select flashcards from selected decks..."
                          : "Select one or more decks first",
                    })}
                    options={cardOptions}
                    loading={isFetchingCards}
                    disabled={
                      generationInFlight || selectedDeckIds.length === 0
                    }
                    className="w-full"
                    data-testid="generate-card-select"
                  />
                )}
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-4 lg:sticky lg:top-4">
          <Card
            title={t("option:quiz.quizSettings", {
              defaultValue: "Quiz Settings",
            })}
            size="small"
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                difficulty: "mixed",
                focusTopics: [],
                generateStudyMaterials: false,
              }}
            >
              <div className="mb-6 space-y-3">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="text-sm font-medium text-text">
                      {t("option:quiz.questionPlan", {
                        defaultValue: "Question Mix",
                      })}
                    </div>
                    <div
                      className="text-xs text-text-subtle"
                      data-testid="generate-question-count-guidance"
                    >
                      {questionCountRecommendation}
                    </div>
                  </div>
                  <div
                    className="text-sm font-medium text-text"
                    data-testid="generate-question-plan-total"
                  >
                    {t("option:quiz.questionPlanTotal", {
                      defaultValue: "Total: {{count}}",
                      count: totalQuestions,
                    })}
                  </div>
                </div>

                <div className="space-y-2" data-testid="generate-question-plan">
                  {questionPlanRows.map((row) => {
                    const rowLabel = String(
                      t(row.labelKey, { defaultValue: row.labelDefault }),
                    );
                    return (
                    <div
                      key={row.question_type}
                      className="grid gap-3 rounded border border-border-subtle p-3 sm:grid-cols-[minmax(11rem,1fr)_minmax(8rem,10rem)_minmax(8rem,10rem)] sm:items-end"
                      data-testid={`generate-question-plan-row-${row.question_type}`}
                    >
                      <Checkbox
                        checked={row.enabled}
                        disabled={generationInFlight}
                        onChange={(event) =>
                          updateQuestionPlanRow(row.question_type, {
                            enabled: event.target.checked,
                          })
                        }
                      >
                        {rowLabel}
                      </Checkbox>

                      <label className="block text-xs font-medium text-text-subtle">
                        <span className="mb-1 block">
                          {t("option:quiz.questionPlanCount", {
                            defaultValue: "Count",
                          })}
                        </span>
                        <span
                          data-testid={`generate-question-plan-count-${row.question_type}`}
                        >
                          <InputNumber
                            min={1}
                            max={100}
                            precision={0}
                            step={1}
                            value={row.count}
                            aria-label={`${rowLabel} count`}
                            className="w-full"
                            disabled={generationInFlight || !row.enabled}
                            onChange={(value) => {
                              const next = sanitizeInputNumber(value, 1, 100);
                              if (next != null) {
                                updateQuestionPlanRow(row.question_type, {
                                  count: next,
                                });
                              }
                            }}
                          />
                        </span>
                      </label>

                      {row.option_count != null ? (
                        <div className="space-y-1">
                          <label className="block text-xs font-medium text-text-subtle">
                            <span className="mb-1 block">
                              {t("option:quiz.questionPlanOptions", {
                                defaultValue: "Options",
                              })}
                            </span>
                            <span
                              data-testid={`generate-question-plan-option-count-${row.question_type}`}
                            >
                              <InputNumber
                                min={2}
                                max={6}
                                precision={0}
                                step={1}
                                value={row.option_count}
                                aria-label={`${rowLabel} options`}
                                className="w-full"
                                disabled={generationInFlight || !row.enabled}
                                onChange={(value) => {
                                  const next = sanitizeInputNumber(value, 2, 6);
                                  if (next != null) {
                                    updateQuestionPlanRow(row.question_type, {
                                      option_count: next,
                                    });
                                  }
                                }}
                              />
                            </span>
                          </label>
                          <div
                            className="flex gap-1"
                            aria-label={`${rowLabel} option presets`}
                          >
                            {[4, 5].map((count) => (
                              <Button
                                key={count}
                                size="small"
                                type={
                                  row.option_count === count
                                    ? "primary"
                                    : "default"
                                }
                                aria-label={t("option:quiz.useOptionPreset", {
                                  defaultValue:
                                    "Use {{count}} options for {{label}}",
                                  count,
                                  label: rowLabel,
                                })}
                                disabled={generationInFlight || !row.enabled}
                                onClick={() =>
                                  updateQuestionPlanRow(row.question_type, {
                                    option_count: count,
                                  })
                                }
                              >
                                {count}
                              </Button>
                            ))}
                          </div>
                        </div>
                      ) : row.pair_count != null ? (
                        <label className="block text-xs font-medium text-text-subtle">
                          <span className="mb-1 block">
                            {t("option:quiz.questionPlanPairs", {
                              defaultValue: "Pairs",
                            })}
                          </span>
                          <span
                            data-testid={`generate-question-plan-pair-count-${row.question_type}`}
                          >
                            <InputNumber
                              min={2}
                              max={6}
                              precision={0}
                              step={1}
                              value={row.pair_count}
                              aria-label={`${rowLabel} pairs`}
                              className="w-full"
                              disabled={generationInFlight || !row.enabled}
                              onChange={(value) => {
                                const next = sanitizeInputNumber(value, 2, 6);
                                if (next != null) {
                                  updateQuestionPlanRow(row.question_type, {
                                    pair_count: next,
                                  });
                                }
                              }}
                            />
                          </span>
                        </label>
                      ) : null}
                    </div>
                    );
                  })}
                </div>
              </div>

              <Form.Item
                name="difficulty"
                label={
                  <span className="inline-flex items-center gap-1">
                    <span>
                      {t("option:quiz.difficulty", {
                        defaultValue: "Difficulty",
                      })}
                    </span>
                    <Tooltip
                      title={t("option:quiz.difficultyTooltip", {
                        defaultValue:
                          "Choose difficulty based on learner skill and source complexity.",
                      })}
                    >
                      <InfoCircleOutlined
                        aria-label={t("option:quiz.difficultyHelp", {
                          defaultValue: "Difficulty help",
                        })}
                      />
                    </Tooltip>
                  </span>
                }
                extra={
                  <div
                    className="space-y-1 text-xs text-text-subtle"
                    data-testid="generate-difficulty-guidance"
                  >
                    {difficultyOptions.map((option) => (
                      <div key={option.value}>
                        <strong>{option.label}:</strong> {option.description}
                      </div>
                    ))}
                  </div>
                }
              >
                <Select
                  options={difficultyOptions.map((option) => ({
                    value: option.value,
                    label: option.label,
                  }))}
                  disabled={generationInFlight}
                />
              </Form.Item>

              <Form.Item
                name="focusTopics"
                label={t("option:quiz.focusTopics", {
                  defaultValue: "Focus Topics (optional)",
                })}
                extra={t("option:quiz.focusTopicsHelp", {
                  defaultValue:
                    "Add keywords or topics to prioritize during generation.",
                })}
              >
                <Select
                  mode="tags"
                  tokenSeparators={[","]}
                  aria-label={t("option:quiz.focusTopics", {
                    defaultValue: "Focus Topics (optional)",
                  })}
                  placeholder={t("option:quiz.focusTopicsPlaceholder", {
                    defaultValue:
                      "Examples: key formulas, chapter 4, terminology",
                  })}
                  disabled={generationInFlight}
                  suffixIcon={null}
                  notFoundContent={null}
                  open={false}
                />
              </Form.Item>

              <div className="space-y-1">
                <Form.Item
                  name="generateStudyMaterials"
                  valuePropName="checked"
                  noStyle
                >
                  <Checkbox
                    disabled={generationInFlight || selectedMediaId == null}
                    data-testid="generate-study-materials-toggle"
                  >
                    {t("option:quiz.generateStudyMaterialsToggle", {
                      defaultValue:
                        "Also generate a flashcard deck from this source",
                    })}
                  </Checkbox>
                </Form.Item>
                <p
                  className="text-xs text-text-subtle"
                  data-testid="generate-study-materials-help"
                >
                  {selectedMediaId == null
                    ? t("option:quiz.studyMaterialsRequiresMedia", {
                        defaultValue:
                          "Flashcard deck generation currently requires one selected media source.",
                      })
                    : t("option:quiz.studyMaterialsReadyHint", {
                        defaultValue:
                          "Uses the selected media content to create a companion deck.",
                      })}
                </p>
              </div>
            </Form>
          </Card>

          <Card
            size="small"
            title={t("option:quiz.generationBrief", {
              defaultValue: "Generation Brief",
            })}
            data-testid="generate-generation-brief"
          >
            <div className="space-y-4">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <dt className="text-text-subtle">
                  {t("option:quiz.sources", { defaultValue: "Sources" })}
                </dt>
                <dd className="text-right font-medium text-text">
                  {selectedSourcesLabel}
                </dd>
                <dt className="text-text-subtle">
                  {t("option:quiz.questions", { defaultValue: "Questions" })}
                </dt>
                <dd className="text-right font-medium text-text">
                  {totalQuestions}
                </dd>
                <dt className="text-text-subtle">
                  {t("option:quiz.difficulty", { defaultValue: "Difficulty" })}
                </dt>
                <dd className="text-right font-medium text-text">
                  {difficultyOptions.find(
                    (option) => option.value === selectedDifficulty,
                  )?.label ?? selectedDifficulty}
                </dd>
                <dt className="text-text-subtle">
                  {t("option:quiz.studyMaterials", {
                    defaultValue: "Study materials",
                  })}
                </dt>
                <dd className="text-right font-medium text-text">
                  {shouldGenerateStudyMaterials
                    ? t("common:enabled", { defaultValue: "Enabled" })
                    : t("common:off", { defaultValue: "Off" })}
                </dd>
              </dl>

              {generateBlockReason ? (
                <Alert
                  type="warning"
                  showIcon
                  data-testid="generate-blocking-reason"
                  title={generateBlockReason}
                />
              ) : generationInFlight ? (
                <div className="rounded-md border border-border-subtle bg-surface2/50 p-3 text-center">
                  <Spin />
                  <p className="mt-2 text-sm text-text-muted">
                    {t("option:quiz.generating", {
                      defaultValue: "Generating quiz...",
                    })}
                  </p>
                  <p className="mt-1 text-xs text-text-subtle">
                    {t("option:quiz.generatingHint", {
                      defaultValue:
                        "This usually takes 15-60 seconds, depending on source size.",
                    })}
                  </p>
                </div>
              ) : (
                <div
                  role="status"
                  className="rounded-md border border-border-subtle bg-surface2/50 px-3 py-2 text-xs text-text-muted"
                >
                  {t("option:quiz.readyToGenerate", {
                    defaultValue:
                      "Ready to generate from the selected sources.",
                  })}
                </div>
              )}

              {generationInFlight ? (
                <Button
                  icon={<StopOutlined />}
                  onClick={handleCancelGeneration}
                  danger
                  block
                  data-testid="generate-cancel-button"
                >
                  {t("common:cancel", { defaultValue: "Cancel" })}
                </Button>
              ) : (
                <Button
                  type="primary"
                  icon={<RocketOutlined />}
                  size="large"
                  onClick={handleGenerate}
                  loading={generationInFlight}
                  disabled={!canGenerate}
                  block
                >
                  {t("option:quiz.generateQuiz", {
                    defaultValue: "Generate Quiz",
                  })}
                </Button>
              )}
            </div>
          </Card>

          {generatedPreview && (
            <Card
              size="small"
              title={t("option:quiz.generatedPreviewTitle", {
                defaultValue: "Generated Quiz Ready",
              })}
              data-testid="generate-preview-card"
            >
              <div className="space-y-3">
                <p className="text-sm text-text">
                  {t("option:quiz.generatedPreviewSummary", {
                    defaultValue:
                      '"{{name}}" is ready with {{count}} questions.',
                    name: generatedPreview.quizName,
                    count: generatedPreview.questionCount,
                  })}
                </p>
                <p className="text-xs text-text-subtle">
                  {t("option:quiz.generatedPreviewHint", {
                    defaultValue:
                      "Review it first, then choose whether to take it now or manage it.",
                  })}
                </p>
                {generatedPreview.flashcardsSummary ? (
                  <Alert
                    data-testid="generate-study-materials-summary"
                    type={
                      generatedPreview.flashcardsSummary.status === "success"
                        ? "success"
                        : generatedPreview.flashcardsSummary.status ===
                            "partial"
                          ? "warning"
                          : "error"
                    }
                    showIcon
                    title={t("option:quiz.generatedFlashcardsSummary", {
                      defaultValue:
                        generatedPreview.flashcardsSummary.status === "success"
                          ? "Flashcards ready: {{saved}} cards saved to {{deckName}}."
                          : generatedPreview.flashcardsSummary.status ===
                              "partial"
                            ? "Flashcards partially saved: {{saved}} saved, {{failed}} failed."
                            : "Flashcard generation needs attention.",
                      saved: generatedPreview.flashcardsSummary.savedCount,
                      failed: generatedPreview.flashcardsSummary.failedCount,
                      deckName:
                        generatedPreview.flashcardsSummary.deckName ||
                        t("option:quiz.generatedFlashcardsDeckFallback", {
                          defaultValue: "generated deck",
                        }),
                    })}
                    description={
                      generatedPreview.flashcardsSummary.errorDetail
                        ? generatedPreview.flashcardsSummary.errorDetail
                        : undefined
                    }
                  />
                ) : null}
                <Space wrap>
                  {onNavigateToManage ? (
                    <Button type="primary" onClick={onNavigateToManage}>
                      {t("option:quiz.reviewInManage", {
                        defaultValue: "Review in Manage",
                      })}
                    </Button>
                  ) : null}
                  <Button
                    type={onNavigateToManage ? "default" : "primary"}
                    onClick={() =>
                      onNavigateToTake({
                        startQuizId: generatedPreview.quizId,
                        highlightQuizId: generatedPreview.quizId,
                        sourceTab: "generate",
                      })
                    }
                  >
                    {t("option:quiz.takeGeneratedQuiz", {
                      defaultValue: "Take Quiz",
                    })}
                  </Button>
                  {generatedPreview.flashcardsSummary?.deckId ? (
                    <Button
                      data-testid="generate-open-flashcards-button"
                      onClick={() =>
                        navigate(
                          buildFlashcardsStudyRouteFromQuiz({
                            quizId: generatedPreview.quizId,
                            deckId: generatedPreview.flashcardsSummary?.deckId,
                          }),
                        )
                      }
                    >
                      {t("option:quiz.openGeneratedFlashcards", {
                        defaultValue: "Open Flashcards Deck",
                      })}
                    </Button>
                  ) : null}
                  {generatedPreview.flashcardsSummary?.handoffRoute &&
                  generatedPreview.flashcardsSummary.status !== "success" ? (
                    <Button
                      data-testid="generate-continue-flashcards-button"
                      onClick={() =>
                        navigate(
                          generatedPreview.flashcardsSummary
                            ?.handoffRoute as string,
                        )
                      }
                    >
                      {t("option:quiz.continueFlashcardsGeneration", {
                        defaultValue: "Continue in Flashcards",
                      })}
                    </Button>
                  ) : null}
                  <Button onClick={() => setGeneratedPreview(null)}>
                    {t("option:quiz.generateAnother", {
                      defaultValue: "Generate Another",
                    })}
                  </Button>
                </Space>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default GenerateTab;
