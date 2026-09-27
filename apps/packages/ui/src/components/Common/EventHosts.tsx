import React from "react"

import type { CommandPaletteProps } from "@/components/Common/CommandPalette"
import { CommandPaletteHost } from "@/components/Common/CommandPaletteHost"
import { PageHelpModalHost } from "@/components/Common/PageHelpModalHost"

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
    <PageHelpModalHost />
  </>
)
