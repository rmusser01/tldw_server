import { useEffect, useState, type FC, type ReactNode } from "react"
import { Drawer } from "antd"
import {
  resolveWritingLayoutMode,
  type WritingLayoutMode
} from "./writing-layout-utils"

const LIBRARY_SIDEBAR_WIDTH_PX = 280
const LIBRARY_DRAWER_WIDTH_PX = 320
const INSPECTOR_SIDEBAR_WIDTH_PX = 320
const INSPECTOR_DRAWER_WIDTH_PX = 360

type WritingPlaygroundShellProps = {
  children: ReactNode
  focusMode?: boolean
  libraryOpen: boolean
  inspectorOpen: boolean
  onLibraryToggle: () => void
  onInspectorToggle: () => void
  onLibraryClose: () => void
  onInspectorClose: () => void
  libraryContent: ReactNode
  inspectorContent: ReactNode
}

export const WritingPlaygroundShell: FC<WritingPlaygroundShellProps> = ({
  children,
  focusMode,
  libraryOpen,
  inspectorOpen,
  onLibraryToggle,
  onInspectorToggle,
  onLibraryClose,
  onInspectorClose,
  libraryContent,
  inspectorContent
}) => {
  const [layoutMode, setLayoutMode] = useState<WritingLayoutMode>(() => {
    if (typeof window === "undefined") return "expanded"
    return resolveWritingLayoutMode(window.innerWidth)
  })

  useEffect(() => {
    if (typeof window === "undefined") return
    const onResize = () => {
      setLayoutMode(resolveWritingLayoutMode(window.innerWidth))
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])

  const isCompact = layoutMode === "compact"

  return (
    <div
      data-testid="writing-playground-shell"
      data-layout-mode={layoutMode}
      className="flex h-full w-full overflow-hidden">
      {/* Pinned library sidebar for expanded mode */}
      {!isCompact && libraryOpen && !focusMode && (
        <div
          data-testid="writing-library-sidebar"
          style={{ width: LIBRARY_SIDEBAR_WIDTH_PX }}
          className="flex-shrink-0 border-r border-border overflow-y-auto bg-surface">
          {libraryContent}
        </div>
      )}

      {/* Compact mode library drawer */}
      {isCompact && !focusMode && (
        <Drawer
          title="Sessions"
          placement="left"
          open={libraryOpen}
          onClose={onLibraryClose}
          width={LIBRARY_DRAWER_WIDTH_PX}
          styles={{ body: { padding: 0 } }}>
          {libraryContent}
        </Drawer>
      )}

      {/* Main editor area fills remaining space */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
        {children}
      </div>

      {/* Pinned inspector sidebar for expanded mode */}
      {!isCompact && inspectorOpen && !focusMode && (
        <div
          data-testid="writing-inspector-sidebar"
          style={{ width: INSPECTOR_SIDEBAR_WIDTH_PX }}
          className="flex-shrink-0 border-l border-border overflow-y-auto bg-surface">
          {inspectorContent}
        </div>
      )}

      {/* Compact mode inspector drawer */}
      {isCompact && !focusMode && (
        <Drawer
          title="Settings"
          placement="right"
          open={inspectorOpen}
          onClose={onInspectorClose}
          width={INSPECTOR_DRAWER_WIDTH_PX}
          styles={{ body: { padding: 0 } }}>
          {inspectorContent}
        </Drawer>
      )}
    </div>
  )
}
