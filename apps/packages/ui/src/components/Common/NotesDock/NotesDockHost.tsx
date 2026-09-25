import React, { Suspense, lazy } from "react"
import { useNotesDockStore } from "@/store/notes-dock"
import { useMobile } from "@/hooks/useMediaQuery"
import { isEditableTarget } from "@/utils/editable-target"

const NotesDockPanel = lazy(() =>
  import("./NotesDockPanel").then((m) => ({ default: m.NotesDockPanel }))
)

const NOTES_DOCK_SHORTCUT_EVENT = "tldw:notes-dock-request-close"

export const NotesDockHost: React.FC = () => {
  const isOpen = useNotesDockStore((state) => state.isOpen)
  const setOpen = useNotesDockStore((state) => state.setOpen)
  const isMobile = useMobile()

  React.useEffect(() => {
    if (isMobile) return

    const handleGlobalToggleShortcut = (event: KeyboardEvent) => {
      const normalizedKey = String(event.key || "").toLowerCase()
      if (normalizedKey !== "n") return
      if (!event.shiftKey || (!event.metaKey && !event.ctrlKey) || event.altKey) return
      if (event.repeat || event.defaultPrevented) return
      if (isEditableTarget(event.target)) return

      event.preventDefault()
      if (isOpen) {
        window.dispatchEvent(new CustomEvent(NOTES_DOCK_SHORTCUT_EVENT))
      } else {
        setOpen(true)
      }
    }

    window.addEventListener("keydown", handleGlobalToggleShortcut)
    return () => {
      window.removeEventListener("keydown", handleGlobalToggleShortcut)
    }
  }, [isMobile, isOpen, setOpen])

  if (isMobile || !isOpen) return null

  return (
    <Suspense fallback={null}>
      <NotesDockPanel />
    </Suspense>
  )
}

export default NotesDockHost
