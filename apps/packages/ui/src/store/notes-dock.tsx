import { createWithEqualityFn } from "zustand/traditional"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"

export type NotesDockPosition = { x: number; y: number }
export type NotesDockSize = { width: number; height: number }

type NoteSnapshot = {
  title: string
  content: string
  keywords: string[]
  version?: number | null
}

export type NotesDockNote = {
  localId: string
  id?: number
  title: string
  content: string
  keywords: string[]
  version?: number | null
  snapshot: NoteSnapshot | null
  isDirty: boolean
}

type NotesDockState = {
  isOpen: boolean
  position: NotesDockPosition
  size: NotesDockSize
  notes: NotesDockNote[]
  activeNoteId: string | null
  setOpen: (open: boolean) => void
  toggleOpen: () => void
  setPosition: (position: NotesDockPosition) => void
  setSize: (size: NotesDockSize) => void
  createDraft: () => NotesDockNote
  openNote: (note: Omit<NotesDockNote, "localId" | "snapshot" | "isDirty">) => void
  setActiveNote: (localId: string | null) => void
  updateNote: (localId: string, patch: Partial<Omit<NotesDockNote, "localId" | "snapshot" | "isDirty">>) => void
  markSaved: (localId: string, saved: NoteSnapshot & { id?: number; version?: number | null }) => void
  discardNoteChanges: (localId: string) => void
  removeNote: (localId: string) => void
  discardAll: () => void
}

const DEFAULT_POSITION: NotesDockPosition = { x: 24, y: 80 }
const DEFAULT_SIZE: NotesDockSize = { width: 640, height: 520 }

const createMemoryStorage = (): StateStorage => ({
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
})

const generateLocalId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `note_${Date.now()}_${Math.random().toString(36).slice(2)}`
}

const areKeywordsEqual = (a: string[], b: string[]) => {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

const computeDirty = (note: NotesDockNote) => {
  if (!note.snapshot) {
    return Boolean(
      note.title.trim() || note.content.trim() || note.keywords.length > 0
    )
  }
  return (
    note.title !== note.snapshot.title ||
    note.content !== note.snapshot.content ||
    !areKeywordsEqual(note.keywords, note.snapshot.keywords)
  )
}

export const useNotesDockStore = createWithEqualityFn<NotesDockState>()(
  persist(
    (set, get) => ({
      isOpen: false,
      position: DEFAULT_POSITION,
      size: DEFAULT_SIZE,
      notes: [],
      activeNoteId: null,
      setOpen: (open) => set({ isOpen: open }),
      toggleOpen: () => set((state) => ({ isOpen: !state.isOpen })),
      setPosition: (position) => set({ position }),
      setSize: (size) => set({ size }),
      createDraft: () => {
        const note: NotesDockNote = {
          localId: generateLocalId(),
          title: "",
          content: "",
          keywords: [],
          version: null,
          snapshot: null,
          isDirty: false
        }
        set((state) => ({
          notes: [...state.notes, note],
          activeNoteId: note.localId
        }))
        return note
      },
      openNote: (noteInput) => {
        set((state) => {
          const existingIndex = noteInput.id
            ? state.notes.findIndex((note) => note.id === noteInput.id)
            : -1
          const snapshot: NoteSnapshot = {
            title: noteInput.title,
            content: noteInput.content,
            keywords: noteInput.keywords ?? [],
            version: noteInput.version ?? null
          }
          if (existingIndex >= 0) {
            const existing = state.notes[existingIndex]
            const updated: NotesDockNote = {
              ...existing,
              ...noteInput,
              keywords: noteInput.keywords ?? [],
              snapshot,
              isDirty: false
            }
            const nextNotes = [...state.notes]
            nextNotes[existingIndex] = updated
            return { notes: nextNotes, activeNoteId: existing.localId }
          }

          const newNote: NotesDockNote = {
            localId: generateLocalId(),
            id: noteInput.id,
            title: noteInput.title ?? "",
            content: noteInput.content ?? "",
            keywords: noteInput.keywords ?? [],
            version: noteInput.version ?? null,
            snapshot,
            isDirty: false
          }
          return {
            notes: [...state.notes, newNote],
            activeNoteId: newNote.localId
          }
        })
      },
      setActiveNote: (localId) => set({ activeNoteId: localId }),
      updateNote: (localId, patch) =>
        set((state) => {
          const nextNotes = state.notes.map((note) => {
            if (note.localId !== localId) return note
            const updated: NotesDockNote = {
              ...note,
              ...patch,
              keywords: patch.keywords ?? note.keywords
            }
            return { ...updated, isDirty: computeDirty(updated) }
          })
          return { notes: nextNotes }
        }),
      markSaved: (localId, saved) =>
        set((state) => {
          const nextNotes = state.notes.map((note) => {
            if (note.localId !== localId) return note
            const snapshot: NoteSnapshot = {
              title: saved.title,
              content: saved.content,
              keywords: saved.keywords ?? [],
              version: saved.version ?? null
            }
            const updated: NotesDockNote = {
              ...note,
              id: saved.id ?? note.id,
              title: saved.title,
              content: saved.content,
              keywords: saved.keywords ?? [],
              version: saved.version ?? note.version,
              snapshot,
              isDirty: false
            }
            return updated
          })
          return { notes: nextNotes }
        }),
      discardNoteChanges: (localId) =>
        set((state) => {
          const target = state.notes.find((note) => note.localId === localId)
          if (!target) return state
          if (!target.snapshot) {
            const remaining = state.notes.filter((note) => note.localId !== localId)
            const nextActive =
              state.activeNoteId === localId
                ? remaining[remaining.length - 1]?.localId ?? null
                : state.activeNoteId
            return { notes: remaining, activeNoteId: nextActive }
          }
          const restored: NotesDockNote = {
            ...target,
            title: target.snapshot.title,
            content: target.snapshot.content,
            keywords: [...target.snapshot.keywords],
            version: target.snapshot.version ?? target.version ?? null,
            isDirty: false
          }
          return {
            notes: state.notes.map((note) =>
              note.localId === localId ? restored : note
            )
          }
        }),
      removeNote: (localId) =>
        set((state) => {
          const remaining = state.notes.filter((note) => note.localId !== localId)
          const nextActive =
            state.activeNoteId === localId
              ? remaining[remaining.length - 1]?.localId ?? null
              : state.activeNoteId
          return { notes: remaining, activeNoteId: nextActive }
        }),
      discardAll: () => {
        const { notes } = get()
        notes.forEach((note) => {
          get().discardNoteChanges(note.localId)
        })
      }
    }),
    {
      name: "tldw-notes-dock",
      // Baseline version so future shape changes can migrate instead of discarding
      // persisted state (see apps/FRONTEND_AUDIT.md §6 / TASK-12102).
      version: 1,
      migrate: (persisted) => persisted as any,
      storage: createJSONStorage(() =>
        typeof window !== "undefined" ? localStorage : createMemoryStorage()
      ),
      partialize: (state) => ({
        position: state.position,
        size: state.size
      })
    }
  )
)
