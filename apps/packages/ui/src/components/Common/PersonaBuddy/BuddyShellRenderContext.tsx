import React from "react"

import type {
  PersonaBuddyRenderContext,
  PersonaBuddySummary
} from "@/types/persona-buddy"

type BuddyShellRenderContextValue = {
  renderContext: PersonaBuddyRenderContext | null
  setRenderContext: (value: PersonaBuddyRenderContext | null) => void
  clearRenderContext: () => void
}

const noop = () => {}

const BuddyShellRenderContext = React.createContext<BuddyShellRenderContextValue>(
  {
    renderContext: null,
    setRenderContext: noop,
    clearRenderContext: noop
  }
)

const areBuddySummariesEqual = (
  left: PersonaBuddySummary | null | undefined,
  right: PersonaBuddySummary | null | undefined
) =>
  left === right ||
  (left != null &&
    right != null &&
    left.has_buddy === right.has_buddy &&
    left.persona_name === right.persona_name &&
    left.role_summary === right.role_summary &&
    left.visual?.species_id === right.visual?.species_id &&
    left.visual?.silhouette_id === right.visual?.silhouette_id &&
    left.visual?.palette_id === right.visual?.palette_id)

const areRenderContextsEqual = (
  left: PersonaBuddyRenderContext | null,
  right: PersonaBuddyRenderContext | null
) =>
  left === right ||
  (left !== null &&
    right !== null &&
    left.surface_id === right.surface_id &&
    left.surface_active === right.surface_active &&
    left.active_persona_id === right.active_persona_id &&
    left.position_bucket === right.position_bucket &&
    areBuddySummariesEqual(left.buddy_summary, right.buddy_summary) &&
    left.live_session_id === right.live_session_id &&
    left.live_voice_state === right.live_voice_state &&
    left.active_tool_status === right.active_tool_status &&
    left.wake_armed === right.wake_armed &&
    left.recovery_mode === right.recovery_mode &&
    left.visual_state === right.visual_state &&
    left.persona_source === right.persona_source)

type BuddyShellRenderContextProviderProps = {
  children: React.ReactNode
  initialContext?: PersonaBuddyRenderContext | null
}

export const BuddyShellRenderContextProvider: React.FC<
  BuddyShellRenderContextProviderProps
> = ({ children, initialContext = null }) => {
  const [renderContext, setRenderContextState] =
    React.useState<PersonaBuddyRenderContext | null>(initialContext)

  const setRenderContext = React.useCallback(
    (value: PersonaBuddyRenderContext | null) => {
      setRenderContextState((current) =>
        areRenderContextsEqual(current, value) ? current : value
      )
    },
    []
  )

  const clearRenderContext = React.useCallback(() => {
    setRenderContextState(null)
  }, [])

  const value = React.useMemo(
    () => ({
      renderContext,
      setRenderContext,
      clearRenderContext
    }),
    [clearRenderContext, renderContext, setRenderContext]
  )

  return (
    <BuddyShellRenderContext.Provider value={value}>
      {children}
    </BuddyShellRenderContext.Provider>
  )
}

export const useBuddyShellRenderContext = (): PersonaBuddyRenderContext | null =>
  React.useContext(BuddyShellRenderContext).renderContext

export const useBuddyShellRenderContextController = () =>
  React.useContext(BuddyShellRenderContext)

export const useSetBuddyShellRenderContext = () =>
  React.useContext(BuddyShellRenderContext).setRenderContext
