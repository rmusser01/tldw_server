import React, { Suspense, lazy } from "react"

import type { CommandPaletteProps } from "@/components/Common/CommandPalette"
import { CommandPaletteHost } from "@/components/Common/CommandPaletteHost"

const PageHelpModal = lazy(() =>
  import("@/components/Common/PageHelpModal").then((m) => ({
    default: m.PageHelpModal
  }))
)

type EventOnlyHostsProps = {
  commandPaletteProps?: CommandPaletteProps
  includePromptCommands?: boolean
}

export const EventOnlyHosts = ({
  commandPaletteProps,
  includePromptCommands = false
}: EventOnlyHostsProps) => (
  <>
    <CommandPaletteHost
      commandPaletteProps={commandPaletteProps}
      includePromptCommands={includePromptCommands}
    />
    <Suspense fallback={null}>
      <PageHelpModal />
    </Suspense>
  </>
)
