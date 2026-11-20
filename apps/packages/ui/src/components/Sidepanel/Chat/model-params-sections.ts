/**
 * Which model-parameter sections are open in the side panel (XS-15). The
 * choice is kept per browser profile, as the Pro sidebar's collapsed state is
 * (sidepanel-sidebar-layout.ts), so reopening the panel keeps the user's
 * layout. Storage failures fall back to all sections closed.
 */
import React from "react"

export const MODEL_PARAMS_SECTIONS_STORAGE_KEY = "tldw:sidepanel-model-params-sections"

export type ModelParamsSection = "provider" | "params"
type OpenSections = Record<ModelParamsSection, boolean>

const CLOSED: OpenSections = { provider: false, params: false }

const readOpenSections = (): OpenSections => {
  try {
    const raw = window.localStorage.getItem(MODEL_PARAMS_SECTIONS_STORAGE_KEY)
    const parsed = raw ? (JSON.parse(raw) as Partial<OpenSections>) : null
    return {
      provider: parsed?.provider === true,
      params: parsed?.params === true
    }
  } catch {
    return CLOSED
  }
}

const writeOpenSections = (sections: OpenSections) => {
  try {
    window.localStorage.setItem(MODEL_PARAMS_SECTIONS_STORAGE_KEY, JSON.stringify(sections))
  } catch {
    // The layout is a convenience; losing it is harmless.
  }
}

export const useModelParamsSections = () => {
  const [open, setOpen] = React.useState<OpenSections>(readOpenSections)
  const toggle = React.useCallback((section: ModelParamsSection) => {
    setOpen((previous) => {
      const next = { ...previous, [section]: !previous[section] }
      writeOpenSections(next)
      return next
    })
  }, [])
  return [open, toggle] as const
}
