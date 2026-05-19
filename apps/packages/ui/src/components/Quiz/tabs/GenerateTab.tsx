import React from "react"
import { Alert, Button, Card, Checkbox, Form, InputNumber, Select, Space, Spin, Tooltip, message } from "antd"
import { useTranslation } from "react-i18next"
import { InfoCircleOutlined, RocketOutlined, StopOutlined } from "@ant-design/icons"
import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { useGenerateQuizMutation } from "../hooks"
import { useDebounce } from "@/hooks/useDebounce"
import { tldwClient } from "@/services/tldw"
import type { QuestionType, QuizGenerateSource } from "@/services/quizzes"
import {
  createDeck,
  createFlashcard,
  generateFlashcards,
  listDecks,
  listFlashcards,
  type FlashcardGeneratedDraft
} from "@/services/flashcards"
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"
import { buildFlashcardsStudyRouteFromQuiz } from "@/services/tldw/quiz-flashcards-handoff"
import type { TakeTabNavigationIntent } from "../navigation"

interface GenerateTabProps {
  onNavigateToTake: (intent?: TakeTabNavigationIntent) => void
  onNavigateToManage?: () => void
}

interface MediaItem {
  id: number
  title: string
  type: string
}

interface MediaListResponse {
  items: MediaItem[]
  total: number | null
}

interface NoteItem {
  id: string
  title: string
}

interface DeckItem {
  id: number
  name: string
}

interface CardItem {
  id: string
  label: string
  deckId: number
}

type GeneratedPreview = {
  quizId: number
  quizName: string
  questionCount: number
  flashcardsSummary: FlashcardsSummary | null
}

type FlashcardsSummary = {
  status: "success" | "partial" | "failed"
  deckId?: number
  deckName?: string
  generatedCount: number
  savedCount: number
  failedCount: number
  errorDetail?: string | null
  handoffRoute?: string
}

const MEDIA_PAGE_SIZE = 50
const MAX_FLASHCARDS_IN_STUDY_FLOW = 30
const MAX_FLASHCARD_SOURCE_TEXT_CHARS = 20_000

const QUESTION_TYPE_OPTIONS: { label: string; value: QuestionType }[] = [
  { label: "Multiple Choice", value: "multiple_choice" },
  { label: "True/False", value: "true_false" },
  { label: "Fill in the Blank", value: "fill_blank" }
]

const DIFFICULTY_OPTIONS: Array<{
  label: string
  value: "easy" | "medium" | "hard" | "mixed"
  description: string
}> = [
  { label: "Easy", value: "easy", description: "Basic recall and straightforward definitions." },
  { label: "Medium", value: "medium", description: "Concept application and moderate reasoning." },
  { label: "Hard", value: "hard", description: "Multi-step reasoning and subtle distinctions." },
  { label: "Mixed", value: "mixed", description: "Balanced blend of easy, medium, and hard." }
]

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") return null
  return value as Record<string, unknown>
}

const asString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

const asNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

const isAbortError = (error: unknown): boolean => {
  if (error instanceof Error && error.name === "AbortError") return true
  const message = error instanceof Error ? error.message : typeof error === "string" ? error : ""
  return message.toLowerCase().includes("abort")
}

const isFormValidationError = (error: unknown): boolean => {
  const record = asRecord(error)
  return Array.isArray(record?.errorFields)
}

const normalizeFocusTopics = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const unique = new Set<string>()
  value.forEach((topic) => {
    if (typeof topic !== "string") return
    const trimmed = topic.trim()
    if (trimmed) unique.add(trimmed)
  })
  return Array.from(unique)
}

const extractErrorDetail = (error: unknown): string | null => {
  const extract = (value: unknown): string | null => {
    if (!value) return null
    if (typeof value === "string") return value.trim() || null
    if (Array.isArray(value)) {
      for (const entry of value) {
        const detail = extract(entry)
        if (detail) return detail
      }
      return null
    }
    const record = asRecord(value)
    if (!record) return null
    return (
      extract(record.detail) ??
      extract(record.message) ??
      extract(record.msg) ??
      null
    )
  }

  if (error instanceof Error) {
    const message = error.message.trim()
    if (message && !/failed to generate quiz/i.test(message)) {
      return message
    }
  }

  const record = asRecord(error)
  if (!record) return null

  return (
    extract(record.detail) ??
    extract(record.error) ??
    extract(asRecord(record.response)?.data) ??
    extract(record.message) ??
    null
  )
}

const extractWordCount = (details: unknown): number | null => {
  const record = asRecord(details)
  if (!record) return null
  return (
    asNumber(asRecord(record.content)?.word_count) ??
    asNumber(asRecord(asRecord(record.processing)?.safe_metadata)?.word_count) ??
    asNumber(asRecord(record.metadata)?.word_count) ??
    null
  )
}

const normalizeMediaListResponse = (raw: unknown): MediaListResponse => {
  const record = asRecord(raw)
  const rawItems = record?.items ?? record?.media ?? record?.results ?? record?.data ?? []
  const array = Array.isArray(rawItems) ? rawItems : []

  const items = array
    .map((entry) => {
      const item = asRecord(entry)
      if (!item) return null
      const id = asNumber(item.id ?? item.media_id)
      if (id == null) return null
      return {
        id,
        title: asString(item.title) ?? asString(item.name) ?? `Media #${id}`,
        type: asString(item.type) ?? asString(item.media_type) ?? "unknown"
      } satisfies MediaItem
    })
    .filter((item): item is MediaItem => item != null)

  const pagination = asRecord(record?.pagination)
  const total = (
    asNumber(pagination?.total_items) ??
    asNumber(record?.total_items) ??
    asNumber(record?.count) ??
    null
  )

  return { items, total }
}

const normalizeNoteListResponse = (raw: unknown): NoteItem[] => {
  const record = asRecord(raw)
  const rawItems = record?.items ?? record?.notes ?? record?.results ?? record?.data ?? raw
  const array = Array.isArray(rawItems) ? rawItems : []
  const seen = new Set<string>()

  return array
    .map((entry) => {
      const item = asRecord(entry)
      if (!item) return null
      const id = asString(item.id ?? item.note_id)
      if (!id || seen.has(id)) return null
      seen.add(id)
      return {
        id,
        title: asString(item.title) ?? asString(item.name) ?? `Note ${id}`
      } satisfies NoteItem
    })
    .filter((item): item is NoteItem => item != null)
}

const normalizeDeckListResponse = (raw: unknown): DeckItem[] => {
  const array = Array.isArray(raw) ? raw : []
  const seen = new Set<number>()

  return array
    .map((entry) => {
      const item = asRecord(entry)
      if (!item) return null
      const id = asNumber(item.id)
      if (id == null || id <= 0 || seen.has(id)) return null
      seen.add(id)
      return {
        id,
        name: asString(item.name) ?? `Deck ${id}`
      } satisfies DeckItem
    })
    .filter((item): item is DeckItem => item != null)
}

const normalizeFlashcardListResponse = (
  raw: unknown,
  deckNames: Map<number, string>
): CardItem[] => {
  const record = asRecord(raw)
  const rawItems = record?.items ?? record?.results ?? record?.data ?? []
  const array = Array.isArray(rawItems) ? rawItems : []
  const seen = new Set<string>()

  return array
    .map((entry) => {
      const item = asRecord(entry)
      if (!item) return null
      const id = asString(item.uuid ?? item.id)
      if (!id || seen.has(id)) return null
      const deckId = asNumber(item.deck_id)
      if (deckId == null || deckId <= 0) return null
      seen.add(id)
      const front = asString(item.front) ?? ""
      const back = asString(item.back) ?? ""
      const preview = [front, back].filter(Boolean).join(" - ")
      const deckName = deckNames.get(deckId) ?? `Deck ${deckId}`
      return {
        id,
        deckId,
        label: preview ? `${deckName}: ${preview}` : `${deckName}: ${id}`
      } satisfies CardItem
    })
    .filter((item): item is CardItem => item != null)
}

const getFirstNonEmptyString = (...values: unknown[]): string => {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim()
    }
  }
  return ""
}

const extractMediaText = (details: unknown): string => {
  if (typeof details === "string") return details.trim()
  const record = asRecord(details)
  if (!record) return ""

  const content = record.content
  if (typeof content === "string" && content.trim().length > 0) {
    return content.trim()
  }
  const contentRecord = asRecord(content)
  if (contentRecord) {
    const nested = getFirstNonEmptyString(
      contentRecord.text,
      contentRecord.content,
      contentRecord.raw_text,
      contentRecord.rawText,
      contentRecord.transcript,
      contentRecord.summary
    )
    if (nested) return nested
  }

  const fromRoot = getFirstNonEmptyString(
    record.text,
    record.transcript,
    record.raw_text,
    record.rawText,
    record.raw_content,
    record.rawContent,
    record.summary
  )
  if (fromRoot) return fromRoot

  const latestVersion = asRecord(record.latest_version) ?? asRecord(record.latestVersion)
  if (latestVersion) {
    const fromLatest = getFirstNonEmptyString(
      latestVersion.content,
      latestVersion.text,
      latestVersion.transcript,
      latestVersion.raw_text,
      latestVersion.rawText,
      latestVersion.summary
    )
    if (fromLatest) return fromLatest
  }

  const data = asRecord(record.data)
  if (data) {
    const fromData = getFirstNonEmptyString(
      data.content,
      data.text,
      data.transcript,
      data.raw_text,
      data.rawText,
      data.summary
    )
    if (fromData) return fromData
  }

  return ""
}

const clampFlashcardsCount = (questionCount: number): number =>
  Math.max(3, Math.min(MAX_FLASHCARDS_IN_STUDY_FLOW, Math.round(questionCount)))

const normalizeGeneratedDrafts = (
  drafts: FlashcardGeneratedDraft[] | null | undefined
): FlashcardGeneratedDraft[] => {
  if (!Array.isArray(drafts)) return []
  return drafts.filter((draft) => {
    const front = typeof draft.front === "string" ? draft.front.trim() : ""
    const back = typeof draft.back === "string" ? draft.back.trim() : ""
    return front.length > 0 && back.length > 0
  })
}

const buildGeneratedDeckName = (quizName: string): string => {
  const trimmed = quizName.trim()
  if (!trimmed) return "Generated Study Deck"
  return `${trimmed} - Flashcards`
}

export const GenerateTab: React.FC<GenerateTabProps> = ({ onNavigateToTake, onNavigateToManage }) => {
  const { t } = useTranslation(["option", "common", "settings"])
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [selectedMediaId, setSelectedMediaId] = React.useState<number | null>(null)
  const [selectedNoteIds, setSelectedNoteIds] = React.useState<string[]>([])
  const [selectedDeckIds, setSelectedDeckIds] = React.useState<number[]>([])
  const [selectedCardIds, setSelectedCardIds] = React.useState<string[]>([])
  const [messageApi, contextHolder] = message.useMessage()
  const [mediaSearchInput, setMediaSearchInput] = React.useState("")
  const [notesSearchInput, setNotesSearchInput] = React.useState("")
  const [mediaPage, setMediaPage] = React.useState(1)
  const [loadedMediaItems, setLoadedMediaItems] = React.useState<MediaItem[]>([])
  const [mediaTotal, setMediaTotal] = React.useState<number | null>(null)
  const [selectedMediaWordCount, setSelectedMediaWordCount] = React.useState<number | null>(null)
  const [generationInFlight, setGenerationInFlight] = React.useState(false)
  const [generatedPreview, setGeneratedPreview] = React.useState<GeneratedPreview | null>(null)
  const debouncedMediaSearch = useDebounce(mediaSearchInput, 300)
  const debouncedNotesSearch = useDebounce(notesSearchInput, 300)
  const generateAbortRef = React.useRef<AbortController | null>(null)

  const generateMutation = useGenerateQuizMutation()

  const {
    data: mediaPageData,
    isLoading: isLoadingList,
    isFetching: isFetchingMedia,
    error: listError
  } = useQuery<MediaListResponse>({
    queryKey: ["quiz-generate-media-list", debouncedMediaSearch, mediaPage],
    queryFn: async () => {
      const searchTerm = debouncedMediaSearch.trim()
      if (searchTerm) {
        const response = await tldwClient.searchMedia(
          { query: searchTerm },
          { page: mediaPage, results_per_page: MEDIA_PAGE_SIZE }
        )
        return normalizeMediaListResponse(response)
      }
      const response = await tldwClient.listMedia({
        page: mediaPage,
        results_per_page: MEDIA_PAGE_SIZE
      })
      return normalizeMediaListResponse(response)
    },
    placeholderData: (previousData) => previousData,
    staleTime: 60 * 1000
  })

  const {
    data: notesData = [],
    isLoading: isLoadingNotes,
    error: notesError
  } = useQuery<NoteItem[]>({
    queryKey: ["quiz-generate-note-list", debouncedNotesSearch],
    queryFn: async () => {
      const searchTerm = debouncedNotesSearch.trim()
      if (searchTerm) {
        const response = await tldwClient.searchNotes(searchTerm)
        return normalizeNoteListResponse(response)
      }
      const response = await tldwClient.listNotes({
        page: 1,
        results_per_page: 200,
        include_keywords: false
      })
      return normalizeNoteListResponse(response)
    },
    staleTime: 60 * 1000
  })

  const {
    data: decksData = [],
    isLoading: isLoadingDecks,
    error: decksError
  } = useQuery<DeckItem[]>({
    queryKey: ["quiz-generate-decks"],
    queryFn: async () => {
      const decks = await listDecks()
      return normalizeDeckListResponse(decks)
    },
    staleTime: 60 * 1000
  })

  const {
    data: cardsData = [],
    isFetching: isFetchingCards,
    error: cardsError
  } = useQuery<CardItem[]>({
    queryKey: ["quiz-generate-cards-by-deck", selectedDeckIds],
    queryFn: async () => {
      const deckNames = new Map<number, string>()
      decksData.forEach((deck) => {
        deckNames.set(deck.id, deck.name)
      })

      const responses = await Promise.all(
        selectedDeckIds.map((deckId) =>
          listFlashcards({
            deck_id: deckId,
            due_status: "all",
            limit: 200,
            offset: 0,
            order_by: "created_at"
          })
        )
      )

      const merged: CardItem[] = []
      const seen = new Set<string>()
      responses.forEach((response) => {
        normalizeFlashcardListResponse(response, deckNames).forEach((card) => {
          if (seen.has(card.id)) return
          seen.add(card.id)
          merged.push(card)
        })
      })
      return merged
    },
    enabled: selectedDeckIds.length > 0,
    staleTime: 60 * 1000
  })

  React.useEffect(() => {
    setMediaPage(1)
    setLoadedMediaItems([])
    setMediaTotal(null)
  }, [debouncedMediaSearch])

  React.useEffect(() => {
    if (!mediaPageData) return
    setMediaTotal(mediaPageData.total)
    setLoadedMediaItems((prev) => {
      if (mediaPage === 1) {
        return mediaPageData.items
      }
      const map = new Map<number, MediaItem>()
      for (const item of prev) {
        map.set(item.id, item)
      }
      for (const item of mediaPageData.items) {
        map.set(item.id, item)
      }
      return Array.from(map.values())
    })
  }, [mediaPage, mediaPageData])

  React.useEffect(() => {
    if (selectedMediaId == null) {
      setSelectedMediaWordCount(null)
      return
    }

    const controller = new AbortController()
    void (async () => {
      try {
        const details = await tldwClient.getMediaDetails(selectedMediaId, {
          include_content: false,
          include_versions: false,
          include_version_content: false,
          signal: controller.signal
        })
        if (controller.signal.aborted) return
        setSelectedMediaWordCount(extractWordCount(details))
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) return
        setSelectedMediaWordCount(null)
      }
    })()

    return () => {
      controller.abort()
    }
  }, [selectedMediaId])

  React.useEffect(() => {
    if (selectedMediaId != null) return
    if (!form.getFieldValue("generateStudyMaterials")) return
    form.setFieldsValue({ generateStudyMaterials: false })
  }, [form, selectedMediaId])

  React.useEffect(() => {
    if (selectedDeckIds.length === 0) {
      setSelectedCardIds((prev) => (prev.length === 0 ? prev : []))
      return
    }
    const availableCardIds = new Set(cardsData.map((card) => card.id))
    setSelectedCardIds((prev) => {
      const next = prev.filter((cardId) => availableCardIds.has(cardId))
      if (next.length === prev.length && next.every((id, index) => id === prev[index])) {
        return prev
      }
      return next
    })
  }, [cardsData, selectedDeckIds])

  React.useEffect(() => {
    return () => {
      generateAbortRef.current?.abort()
    }
  }, [])

  const hasMoreMedia = mediaTotal != null
    ? loadedMediaItems.length < mediaTotal
    : (mediaPageData?.items.length ?? 0) >= MEDIA_PAGE_SIZE

  const isLoadingMoreMedia = isFetchingMedia && mediaPage > 1

  const mediaOptions = React.useMemo(() => {
    const options = loadedMediaItems.map((item) => ({
      value: item.id,
      label: `${item.title || `Media #${item.id}`} (${item.type})`
    }))
    if (selectedMediaId != null && !options.some((option) => option.value === selectedMediaId)) {
      options.unshift({
        value: selectedMediaId,
        label: t("option:quiz.sourceMedia", {
          defaultValue: "Source media #{{id}}",
          id: selectedMediaId
        })
      })
    }
    return options
  }, [loadedMediaItems, selectedMediaId, t])

  const noteOptions = React.useMemo(
    () =>
      notesData.map((item) => ({
        value: item.id,
        label: item.title
      })),
    [notesData]
  )

  const deckOptions = React.useMemo(
    () =>
      decksData.map((deck) => ({
        value: deck.id,
        label: deck.name
      })),
    [decksData]
  )

  const cardOptions = React.useMemo(
    () =>
      cardsData.map((card) => ({
        value: card.id,
        label: card.label
      })),
    [cardsData]
  )

  const selectedSources = React.useMemo<QuizGenerateSource[]>(() => {
    const sourceMap = new Map<string, QuizGenerateSource>()
    const put = (source: QuizGenerateSource) => {
      sourceMap.set(`${source.source_type}:${source.source_id}`, source)
    }

    if (selectedMediaId != null) {
      put({ source_type: "media", source_id: String(selectedMediaId) })
    }
    selectedNoteIds.forEach((noteId) => {
      put({ source_type: "note", source_id: noteId })
    })
    selectedDeckIds.forEach((deckId) => {
      put({ source_type: "flashcard_deck", source_id: String(deckId) })
    })
    selectedCardIds.forEach((cardId) => {
      put({ source_type: "flashcard_card", source_id: cardId })
    })
    return Array.from(sourceMap.values())
  }, [selectedCardIds, selectedDeckIds, selectedMediaId, selectedNoteIds])

  const hasSelectedSources = selectedSources.length > 0

  const selectedMedia = React.useMemo(
    () =>
      selectedMediaId == null
        ? null
        : loadedMediaItems.find((item) => item.id === selectedMediaId) ?? null,
    [loadedMediaItems, selectedMediaId]
  )

  const questionCountRecommendation = React.useMemo(() => {
    if (!selectedMediaWordCount || selectedMediaWordCount <= 0) {
      return t("option:quiz.questionCountRecommendation", {
        defaultValue: "Recommended: 5-10 questions per 1,000 words of source."
      })
    }

    const units = Math.max(1, Math.round(selectedMediaWordCount / 1000))
    const minQuestions = Math.min(50, Math.max(5, units * 5))
    const maxQuestions = Math.min(50, Math.max(minQuestions, units * 10))

    return t("option:quiz.questionCountRecommendationSized", {
      defaultValue:
        "Estimated source length: ~{{wordCount}} words. Recommended: {{minQuestions}}-{{maxQuestions}} questions.",
      wordCount: selectedMediaWordCount.toLocaleString(),
      minQuestions,
      maxQuestions
    })
  }, [selectedMediaWordCount, t])

  const handleCancelGeneration = React.useCallback(() => {
    if (!generationInFlight) return
    generateAbortRef.current?.abort()
  }, [generationInFlight])

  const generateStudyMaterialsFlashcards = React.useCallback(async (params: {
    mediaId: number
    mediaTitle: string
    quizName: string
    numQuestions: number
    difficulty?: "easy" | "medium" | "hard" | "mixed"
    focusTopics: string[]
    signal?: AbortSignal
  }): Promise<FlashcardsSummary> => {
    const fallbackRoute = "/flashcards?tab=importExport"
    const throwIfAborted = () => {
      if (params.signal?.aborted) {
        const abortError = new Error("aborted")
        abortError.name = "AbortError"
        throw abortError
      }
    }

    throwIfAborted()

    try {
      const details = await tldwClient.getMediaDetails(params.mediaId, {
        include_content: true,
        include_versions: false,
        include_version_content: false,
        signal: params.signal
      })
      throwIfAborted()

      const sourceText = extractMediaText(details).slice(0, MAX_FLASHCARD_SOURCE_TEXT_CHARS)
      if (!sourceText) {
        return {
          status: "failed",
          generatedCount: 0,
          savedCount: 0,
          failedCount: 0,
          errorDetail: t("option:quiz.studyMaterialsMissingSourceText", {
            defaultValue: "Could not extract enough source text to generate flashcards."
          }),
          handoffRoute: fallbackRoute
        }
      }

      const handoffRoute = buildFlashcardsGenerateRoute({
        text: sourceText,
        sourceType: "media",
        sourceId: String(params.mediaId),
        sourceTitle: params.mediaTitle
      })

      const generated = await generateFlashcards({
        text: sourceText,
        num_cards: clampFlashcardsCount(params.numQuestions),
        difficulty: params.difficulty,
        focus_topics: params.focusTopics.length > 0 ? params.focusTopics : undefined
      })
      throwIfAborted()

      const drafts = normalizeGeneratedDrafts(generated.flashcards)
      if (drafts.length === 0) {
        return {
          status: "failed",
          generatedCount: 0,
          savedCount: 0,
          failedCount: 0,
          errorDetail: t("option:quiz.studyMaterialsEmptyFlashcards", {
            defaultValue: "Flashcard generation returned no usable cards."
          }),
          handoffRoute
        }
      }

      const deck = await createDeck({
        name: buildGeneratedDeckName(params.quizName),
        description: t("option:quiz.studyMaterialsDeckDescription", {
          defaultValue: "Generated from {{title}}",
          title: params.mediaTitle
        })
      })
      throwIfAborted()

      const createResults = await Promise.allSettled(
        drafts.map((draft) =>
          createFlashcard({
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
            source_ref_id: String(params.mediaId)
          })
        )
      )

      const savedCount = createResults.filter((result) => result.status === "fulfilled").length
      const failedCount = drafts.length - savedCount
      const status: FlashcardsSummary["status"] =
        savedCount === 0 ? "failed" : failedCount > 0 ? "partial" : "success"

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
                defaultValue: "Unable to save generated flashcards."
              })
            : null,
        handoffRoute
      }
    } catch (error) {
      if (isAbortError(error)) throw error
      return {
        status: "failed",
        generatedCount: 0,
        savedCount: 0,
        failedCount: 0,
        errorDetail: extractErrorDetail(error) ?? t("option:quiz.studyMaterialsFailed", {
          defaultValue: "Failed to generate flashcards from the selected source."
        }),
        handoffRoute: fallbackRoute
      }
    }
  }, [t])

  const handleGenerate = async () => {
    if (generationInFlight) {
      return
    }

    if (!hasSelectedSources) {
      messageApi.warning(
        t("option:quiz.selectAtLeastOneSource", {
          defaultValue: "Select at least one source before generating."
        })
      )
      return
    }

    let requestAbortController: AbortController | null = null

    try {
      const values = await form.validateFields()
      setGeneratedPreview(null)

      const focusTopics = normalizeFocusTopics(values.focusTopics)
      const shouldGenerateStudyMaterials = Boolean(values.generateStudyMaterials)
      requestAbortController = new AbortController()
      generateAbortRef.current = requestAbortController
      setGenerationInFlight(true)

      const generated = await generateMutation.mutateAsync({
        request: {
          sources: selectedSources,
          num_questions: values.numQuestions,
          question_types: values.questionTypes,
          difficulty: values.difficulty,
          focus_topics: focusTopics.length > 0 ? focusTopics : undefined
        },
        signal: requestAbortController.signal
      })
      if (requestAbortController.signal.aborted) return

      const generatedQuizName = generated.quiz.name || `Quiz #${generated.quiz.id}`
      let flashcardsSummary: FlashcardsSummary | null = null

      if (shouldGenerateStudyMaterials) {
        if (selectedMediaId == null) {
          flashcardsSummary = {
            status: "failed",
            generatedCount: 0,
            savedCount: 0,
            failedCount: 0,
            errorDetail: t("option:quiz.studyMaterialsMediaRequired", {
              defaultValue: "Flashcard deck generation currently requires a selected media source."
            }),
            handoffRoute: "/flashcards?tab=importExport"
          }
        } else {
          flashcardsSummary = await generateStudyMaterialsFlashcards({
            mediaId: selectedMediaId,
            mediaTitle: selectedMedia?.title || `Media #${selectedMediaId}`,
            quizName: generatedQuizName,
            numQuestions: values.numQuestions ?? (generated.questions.length || 10),
            difficulty: values.difficulty,
            focusTopics,
            signal: requestAbortController.signal
          })
        }
      }

      setGeneratedPreview({
        quizId: generated.quiz.id,
        quizName: generatedQuizName,
        questionCount: generated.questions.length,
        flashcardsSummary
      })

      if (!flashcardsSummary) {
        messageApi.success(
          t("option:quiz.generateSuccessReview", {
            defaultValue: "Quiz generated. Review it before starting."
          })
        )
      } else if (flashcardsSummary.status === "success") {
        messageApi.success(
          t("option:quiz.generateStudyMaterialsSuccess", {
            defaultValue: "Quiz and flashcards generated successfully."
          })
        )
      } else if (flashcardsSummary.status === "partial") {
        messageApi.warning(
          t("option:quiz.generateStudyMaterialsPartial", {
            defaultValue: "Quiz generated. Some flashcards could not be saved."
          })
        )
      } else {
        messageApi.warning(
          t("option:quiz.generateStudyMaterialsFailedNotice", {
            defaultValue: "Quiz generated. Flashcard generation needs review."
          })
        )
      }
    } catch (error) {
      if (isFormValidationError(error)) return
      if (requestAbortController?.signal.aborted || isAbortError(error)) {
        messageApi.info(
          t("option:quiz.generateCancelled", { defaultValue: "Quiz generation canceled." })
        )
        return
      }

      const detail = extractErrorDetail(error)
      messageApi.error(
        detail
          ? t("option:quiz.generateErrorDetailed", {
            defaultValue: "Failed to generate quiz: {{detail}}",
            detail
          })
          : t("option:quiz.generateError", { defaultValue: "Failed to generate quiz" })
      )
    } finally {
      if (generateAbortRef.current === requestAbortController) {
        generateAbortRef.current = null
      }
      setGenerationInFlight(false)
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      {contextHolder}

      <Card
        title={t("option:quiz.selectSources", { defaultValue: "Select Sources" })}
        size="small"
      >
        <div className="space-y-4">
          {!isLoadingList && loadedMediaItems.length === 0 && (
            <Alert
              type="info"
              showIcon
              data-testid="quiz-generate-no-media"
              message={t("option:quiz.generate.noMedia", { defaultValue: "No media content found" })}
              description={
                <>
                  {t("option:quiz.generate.noMediaHint", {
                    defaultValue: "Import videos, articles, or documents in your "
                  })}
                  <a href="/media">
                    {t("option:quiz.generate.mediaLibrary", { defaultValue: "Media Library" })}
                  </a>
                  {t("option:quiz.generate.noMediaSuffix", {
                    defaultValue: ", then return here to generate quizzes."
                  })}
                </>
              }
              className="mb-4"
            />
          )}
          <div className="space-y-2">
            <div className="text-xs font-medium text-text-subtle">
              {t("option:quiz.mediaSources", { defaultValue: "Media" })}
            </div>
            {listError ? (
              <Alert
                type="error"
                title={t("settings:chunkingPlayground.loadMediaListError", "Failed to load media library")}
              />
            ) : (
              <>
                <Select
                  showSearch
                  placeholder={t("option:quiz.selectMediaPlaceholder", { defaultValue: "Select media item..." })}
                  loading={isLoadingList && mediaPage === 1}
                  value={selectedMediaId}
                  onChange={(value) => setSelectedMediaId(value)}
                  onSearch={(value) => setMediaSearchInput(value)}
                  options={mediaOptions}
                  filterOption={false}
                  className="w-full"
                  disabled={generationInFlight}
                  notFoundContent={
                    isLoadingList && mediaPage === 1
                      ? <Spin size="small" />
                      : t("option:quiz.noMediaFound", { defaultValue: "No media found" })
                  }
                />
                {mediaTotal != null ? (
                  <div className="text-xs text-text-subtle">
                    {t("option:quiz.mediaCount", {
                      defaultValue: loadedMediaItems.length < mediaTotal
                        ? "Showing {{loaded}} of {{count}} media items"
                        : "{{count}} media items available",
                      loaded: loadedMediaItems.length,
                      count: mediaTotal
                    })}
                  </div>
                ) : (
                  <div className="text-xs text-text-subtle">
                    {t("option:quiz.loadedMediaCount", {
                      defaultValue: "Showing {{count}} media items",
                      count: loadedMediaItems.length
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
                title={t("option:quiz.loadNotesError", { defaultValue: "Failed to load notes" })}
              />
            ) : (
              <Select
                mode="multiple"
                showSearch
                value={selectedNoteIds}
                onChange={(values) => setSelectedNoteIds(values)}
                onSearch={(value) => setNotesSearchInput(value)}
                placeholder={t("option:quiz.selectNotesPlaceholder", { defaultValue: "Select notes..." })}
                options={noteOptions}
                loading={isLoadingNotes}
                disabled={generationInFlight}
                filterOption={false}
                className="w-full"
                data-testid="generate-note-select"
                notFoundContent={
                  isLoadingNotes
                    ? <Spin size="small" />
                    : t("option:quiz.noNotesFound", { defaultValue: "No notes found" })
                }
              />
            )}
          </div>

          <div className="space-y-2">
            <div className="text-xs font-medium text-text-subtle">
              {t("option:quiz.deckSources", { defaultValue: "Flashcard Decks" })}
            </div>
            {decksError ? (
              <Alert
                type="error"
                title={t("option:quiz.loadDecksError", { defaultValue: "Failed to load flashcard decks" })}
              />
            ) : (
              <Select
                mode="multiple"
                value={selectedDeckIds}
                onChange={(values) => setSelectedDeckIds(values)}
                placeholder={t("option:quiz.selectDecksPlaceholder", { defaultValue: "Select flashcard decks..." })}
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
                title={t("option:quiz.loadCardsError", { defaultValue: "Failed to load flashcards for selected decks" })}
              />
            ) : (
              <Select
                mode="multiple"
                value={selectedCardIds}
                onChange={(values) => setSelectedCardIds(values)}
                placeholder={t("option:quiz.selectCardsPlaceholder", {
                  defaultValue: selectedDeckIds.length > 0
                    ? "Select flashcards from selected decks..."
                    : "Select one or more decks first"
                })}
                options={cardOptions}
                loading={isFetchingCards}
                disabled={generationInFlight || selectedDeckIds.length === 0}
                className="w-full"
                data-testid="generate-card-select"
              />
            )}
          </div>
        </div>
      </Card>

      <Card
        title={t("option:quiz.quizSettings", { defaultValue: "Quiz Settings" })}
        size="small"
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            numQuestions: 10,
            questionTypes: ["multiple_choice", "true_false"],
            difficulty: "mixed",
            focusTopics: [],
            generateStudyMaterials: false
          }}
        >
          <Form.Item
            name="numQuestions"
            label={t("option:quiz.numQuestions", { defaultValue: "Number of Questions" })}
            extra={
              <span className="text-xs text-text-subtle" data-testid="generate-question-count-guidance">
                {questionCountRecommendation}
              </span>
            }
          >
            <InputNumber min={5} max={50} className="w-full" disabled={generationInFlight} />
          </Form.Item>

          <Form.Item
            name="questionTypes"
            label={t("option:quiz.questionTypes", { defaultValue: "Question Types" })}
          >
            <Checkbox.Group options={QUESTION_TYPE_OPTIONS} disabled={generationInFlight} />
          </Form.Item>

          <Form.Item
            name="difficulty"
            label={(
              <span className="inline-flex items-center gap-1">
                <span>{t("option:quiz.difficulty", { defaultValue: "Difficulty" })}</span>
                <Tooltip
                  title={t("option:quiz.difficultyTooltip", {
                    defaultValue: "Choose difficulty based on learner skill and source complexity."
                  })}
                >
                  <InfoCircleOutlined aria-label={t("option:quiz.difficultyHelp", { defaultValue: "Difficulty help" })} />
                </Tooltip>
              </span>
            )}
            extra={(
              <div className="space-y-1 text-xs text-text-subtle" data-testid="generate-difficulty-guidance">
                {DIFFICULTY_OPTIONS.map((option) => (
                  <div key={option.value}>
                    <strong>{option.label}:</strong> {option.description}
                  </div>
                ))}
              </div>
            )}
          >
            <Select
              options={DIFFICULTY_OPTIONS.map((option) => ({
                value: option.value,
                label: option.label
              }))}
              disabled={generationInFlight}
            />
          </Form.Item>

          <Form.Item
            name="focusTopics"
            label={t("option:quiz.focusTopics", { defaultValue: "Focus Topics (optional)" })}
            extra={t("option:quiz.focusTopicsHelp", {
              defaultValue: "Add keywords or topics to prioritize during generation."
            })}
          >
            <Select
              mode="tags"
              tokenSeparators={[","]}
              placeholder={t("option:quiz.focusTopicsPlaceholder", {
                defaultValue: "Examples: key formulas, chapter 4, terminology"
              })}
              disabled={generationInFlight}
              open={false}
            />
          </Form.Item>

          <Form.Item name="generateStudyMaterials" valuePropName="checked" className="!mb-0">
            <Checkbox
              disabled={generationInFlight || selectedMediaId == null}
              data-testid="generate-study-materials-toggle"
            >
              {t("option:quiz.generateStudyMaterialsToggle", {
                defaultValue: "Also generate a flashcard deck from this source"
              })}
            </Checkbox>
          </Form.Item>
        </Form>
      </Card>

      {generationInFlight && (
        <Card size="small">
          <div className="text-center space-y-4">
            <Spin size="large" />
            <p className="text-text-muted">
              {t("option:quiz.generating", { defaultValue: "Generating quiz..." })}
            </p>
            <p className="text-xs text-text-subtle">
              {t("option:quiz.generatingHint", {
                defaultValue: "This usually takes 15-60 seconds, depending on source size."
              })}
            </p>
            <Button
              icon={<StopOutlined />}
              onClick={handleCancelGeneration}
              danger
              data-testid="generate-cancel-button"
            >
              {t("common:cancel", { defaultValue: "Cancel" })}
            </Button>
          </div>
        </Card>
      )}

      {generatedPreview && (
        <Card
          size="small"
          title={t("option:quiz.generatedPreviewTitle", { defaultValue: "Generated Quiz Ready" })}
          data-testid="generate-preview-card"
        >
          <div className="space-y-3">
            <p className="text-sm text-text">
              {t("option:quiz.generatedPreviewSummary", {
                defaultValue: "\"{{name}}\" is ready with {{count}} questions.",
                name: generatedPreview.quizName,
                count: generatedPreview.questionCount
              })}
            </p>
            <p className="text-xs text-text-subtle">
              {t("option:quiz.generatedPreviewHint", {
                defaultValue: "Review it first, then choose whether to take it now or manage it."
              })}
            </p>
            {generatedPreview.flashcardsSummary ? (
              <Alert
                data-testid="generate-study-materials-summary"
                type={
                  generatedPreview.flashcardsSummary.status === "success"
                    ? "success"
                    : generatedPreview.flashcardsSummary.status === "partial"
                      ? "warning"
                      : "error"
                }
                showIcon
                title={t("option:quiz.generatedFlashcardsSummary", {
                  defaultValue:
                    generatedPreview.flashcardsSummary.status === "success"
                      ? "Flashcards ready: {{saved}} cards saved to {{deckName}}."
                      : generatedPreview.flashcardsSummary.status === "partial"
                        ? "Flashcards partially saved: {{saved}} saved, {{failed}} failed."
                        : "Flashcard generation needs attention.",
                  saved: generatedPreview.flashcardsSummary.savedCount,
                  failed: generatedPreview.flashcardsSummary.failedCount,
                  deckName:
                    generatedPreview.flashcardsSummary.deckName ||
                    t("option:quiz.generatedFlashcardsDeckFallback", {
                      defaultValue: "generated deck"
                    })
                })}
                description={
                  generatedPreview.flashcardsSummary.errorDetail
                    ? generatedPreview.flashcardsSummary.errorDetail
                    : undefined
                }
              />
            ) : null}
            <Space wrap>
              <Button
                type="primary"
                onClick={() =>
                  onNavigateToTake({
                    startQuizId: generatedPreview.quizId,
                    highlightQuizId: generatedPreview.quizId,
                    sourceTab: "generate"
                  })
                }
              >
                {t("option:quiz.takeGeneratedQuiz", { defaultValue: "Take Quiz" })}
              </Button>
              {onNavigateToManage ? (
                <Button onClick={onNavigateToManage}>
                  {t("option:quiz.reviewInManage", { defaultValue: "Review in Manage" })}
                </Button>
              ) : null}
              {generatedPreview.flashcardsSummary?.deckId ? (
                <Button
                  data-testid="generate-open-flashcards-button"
                  onClick={() =>
                    navigate(
                      buildFlashcardsStudyRouteFromQuiz({
                        quizId: generatedPreview.quizId,
                        deckId: generatedPreview.flashcardsSummary?.deckId
                      })
                    )
                  }
                >
                  {t("option:quiz.openGeneratedFlashcards", { defaultValue: "Open Flashcards Deck" })}
                </Button>
              ) : null}
              {generatedPreview.flashcardsSummary?.handoffRoute &&
              generatedPreview.flashcardsSummary.status !== "success" ? (
                <Button
                  data-testid="generate-continue-flashcards-button"
                  onClick={() => navigate(generatedPreview.flashcardsSummary?.handoffRoute as string)}
                >
                  {t("option:quiz.continueFlashcardsGeneration", {
                    defaultValue: "Continue in Flashcards"
                  })}
                </Button>
              ) : null}
              <Button onClick={() => setGeneratedPreview(null)}>
                {t("option:quiz.generateAnother", { defaultValue: "Generate Another" })}
              </Button>
            </Space>
          </div>
        </Card>
      )}

      <Button
        type="primary"
        icon={<RocketOutlined />}
        size="large"
        onClick={handleGenerate}
        loading={generationInFlight}
        disabled={!hasSelectedSources || generationInFlight}
        block
      >
        {t("option:quiz.generateQuiz", { defaultValue: "Generate Quiz" })}
      </Button>
    </div>
  )
}

export default GenerateTab
