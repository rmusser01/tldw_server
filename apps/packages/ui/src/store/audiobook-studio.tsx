import { createWithEqualityFn } from "zustand/traditional"
import type { TtsProviderOverrides } from "@/services/tts-provider"

export type ChapterStatus = "pending" | "generating" | "completed" | "error"

export type AudioChapter = {
  id: string
  title: string
  content: string
  order: number
  voiceConfig: TtsProviderOverrides & { speed?: number }
  status: ChapterStatus
  audioBlob?: Blob
  audioUrl?: string
  audioDuration?: number
  errorMessage?: string
  requestedBackend?: string
  actualBackend?: string
  fallbackUsed?: boolean
}

export type SplitMode = "paragraphs" | "headings" | "custom"

type AudiobookStudioStore = {
  // Current project ID (for persistence)
  currentProjectId: string | null
  setCurrentProjectId: (id: string | null) => void

  // Raw content input
  rawContent: string
  setRawContent: (content: string) => void

  // Chapter management
  chapters: AudioChapter[]
  setChapters: (chapters: AudioChapter[]) => void
  addChapter: (chapter: Omit<AudioChapter, "id" | "order">) => void
  updateChapter: (id: string, updates: Partial<AudioChapter>) => void
  removeChapter: (id: string) => void
  reorderChapters: (fromIndex: number, toIndex: number) => void
  clearChapters: () => void

  // Split content into chapters
  splitIntoChapters: (mode: SplitMode, customDelimiter?: string) => void

  // Generation state
  isGenerating: boolean
  setIsGenerating: (generating: boolean) => void
  generationQueue: string[]
  setGenerationQueue: (queue: string[]) => void
  currentGeneratingId: string | null
  setCurrentGeneratingId: (id: string | null) => void

  // Default voice config for new chapters
  defaultVoiceConfig: TtsProviderOverrides & { speed?: number }
  setDefaultVoiceConfig: (config: TtsProviderOverrides & { speed?: number }) => void

  // Project metadata
  projectTitle: string
  setProjectTitle: (title: string) => void
  projectAuthor: string
  setProjectAuthor: (author: string) => void
  projectDescription: string
  setProjectDescription: (description: string) => void
  projectCoverImageUrl: string | null
  setProjectCoverImageUrl: (url: string | null) => void

  // Utility
  getTotalDuration: () => number
  getCompletedChapters: () => AudioChapter[]
  revokeAllAudioUrls: () => void
}

const splitByParagraphs = (text: string): string[] => {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
}

const splitByHeadings = (text: string): { title: string; content: string }[] => {
  const lines = text.split("\n")
  const chapters: { title: string; content: string }[] = []
  let currentTitle = "Introduction"
  let currentContent: string[] = []

  for (const line of lines) {
    const headingMatch = line.match(/^#{1,3}\s+(.+)$/)
    if (headingMatch) {
      if (currentContent.length > 0) {
        chapters.push({
          title: currentTitle,
          content: currentContent.join("\n").trim()
        })
      }
      currentTitle = headingMatch[1].trim()
      currentContent = []
    } else {
      currentContent.push(line)
    }
  }

  if (currentContent.length > 0) {
    chapters.push({
      title: currentTitle,
      content: currentContent.join("\n").trim()
    })
  }

  return chapters.filter((c) => c.content.length > 0)
}

const splitByCustom = (text: string, delimiter: string): string[] => {
  if (!delimiter) return [text]
  return text
    .split(delimiter)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
}

export const useAudiobookStudioStore = createWithEqualityFn<AudiobookStudioStore>(
  (set, get) => ({
    currentProjectId: null,
    rawContent: "",
    chapters: [],
    isGenerating: false,
    generationQueue: [],
    currentGeneratingId: null,
    defaultVoiceConfig: {},
    projectTitle: "Untitled Audiobook",
    projectAuthor: "",
    projectDescription: "",
    projectCoverImageUrl: null,

    setCurrentProjectId: (id) => set({ currentProjectId: id }),

    setRawContent: (content) => set({ rawContent: content }),

    setChapters: (chapters) => set({ chapters }),

    addChapter: (chapter) => {
      const state = get()
      const newChapter: AudioChapter = {
        ...chapter,
        id: crypto.randomUUID(),
        order: state.chapters.length
      }
      set({ chapters: [...state.chapters, newChapter] })
    },

    updateChapter: (id, updates) => {
      set((state) => ({
        chapters: state.chapters.map((ch) =>
          ch.id === id ? { ...ch, ...updates } : ch
        )
      }))
    },

    removeChapter: (id) => {
      const state = get()
      const chapter = state.chapters.find((ch) => ch.id === id)
      if (chapter?.audioUrl) {
        try {
          URL.revokeObjectURL(chapter.audioUrl)
        } catch {}
      }
      set((state) => ({
        chapters: state.chapters
          .filter((ch) => ch.id !== id)
          .map((ch, idx) => ({ ...ch, order: idx }))
      }))
    },

    reorderChapters: (fromIndex, toIndex) => {
      set((state) => {
        const chapters = [...state.chapters]
        const [removed] = chapters.splice(fromIndex, 1)
        chapters.splice(toIndex, 0, removed)
        return {
          chapters: chapters.map((ch, idx) => ({ ...ch, order: idx }))
        }
      })
    },

    clearChapters: () => {
      const state = get()
      state.chapters.forEach((ch) => {
        if (ch.audioUrl) {
          try {
            URL.revokeObjectURL(ch.audioUrl)
          } catch {}
        }
      })
      set({ chapters: [], generationQueue: [], currentGeneratingId: null })
    },

    splitIntoChapters: (mode, customDelimiter) => {
      const state = get()
      const text = state.rawContent

      if (!text.trim()) return

      // Clear existing chapters first
      state.clearChapters()

      let newChapters: AudioChapter[] = []

      if (mode === "headings") {
        const sections = splitByHeadings(text)
        newChapters = sections.map((section, idx) => ({
          id: crypto.randomUUID(),
          title: section.title,
          content: section.content,
          order: idx,
          voiceConfig: { ...state.defaultVoiceConfig },
          status: "pending" as const
        }))
      } else if (mode === "custom" && customDelimiter) {
        const parts = splitByCustom(text, customDelimiter)
        newChapters = parts.map((content, idx) => ({
          id: crypto.randomUUID(),
          title: `Chapter ${idx + 1}`,
          content,
          order: idx,
          voiceConfig: { ...state.defaultVoiceConfig },
          status: "pending" as const
        }))
      } else {
        // paragraphs mode (default)
        const paragraphs = splitByParagraphs(text)
        newChapters = paragraphs.map((content, idx) => ({
          id: crypto.randomUUID(),
          title: `Section ${idx + 1}`,
          content,
          order: idx,
          voiceConfig: { ...state.defaultVoiceConfig },
          status: "pending" as const
        }))
      }

      set({ chapters: newChapters })
    },

    setIsGenerating: (generating) => set({ isGenerating: generating }),

    setGenerationQueue: (queue) => set({ generationQueue: queue }),

    setCurrentGeneratingId: (id) => set({ currentGeneratingId: id }),

    setDefaultVoiceConfig: (config) => set({ defaultVoiceConfig: config }),

    setProjectTitle: (title) => set({ projectTitle: title }),

    setProjectAuthor: (author) => set({ projectAuthor: author }),

    setProjectDescription: (description) =>
      set({ projectDescription: description }),

    setProjectCoverImageUrl: (url) => set({ projectCoverImageUrl: url }),

    getTotalDuration: () => {
      const state = get()
      return state.chapters.reduce(
        (sum, ch) => sum + (ch.audioDuration || 0),
        0
      )
    },

    getCompletedChapters: () => {
      const state = get()
      return state.chapters.filter((ch) => ch.status === "completed")
    },

    revokeAllAudioUrls: () => {
      const state = get()
      state.chapters.forEach((ch) => {
        if (ch.audioUrl) {
          try {
            URL.revokeObjectURL(ch.audioUrl)
          } catch {}
        }
      })
    }
  })
)

// Expose for debugging in non-production builds
if (typeof window !== "undefined" && process.env.NODE_ENV !== "production") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useAudiobookStudioStore = useAudiobookStudioStore
}
