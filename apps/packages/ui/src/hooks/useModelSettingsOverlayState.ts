import React from "react"

/** Keep only an opened model editor present through its own native close motion. */
export const useModelSettingsOverlayState = () => {
  const [state, publish] = React.useState({ open: false, present: false, revision: 0 })
  const live = React.useRef(state)
  const mounted = React.useRef(true)
  React.useLayoutEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])
  const isCurrent = React.useCallback(() =>
    mounted.current && live.current.revision === state.revision,
    [state.revision])
  const setOpen = React.useCallback<React.Dispatch<React.SetStateAction<boolean>>>(requested => {
    if (!mounted.current) return
    const current = live.current
    const open = typeof requested === "function" ? requested(current.open) : requested
    if (!open && !current.open) return
    const next = { open, present: true, revision: current.revision + 1 }
    live.current = next
    publish(next)
  }, [])
  const afterClose = React.useCallback(() => {
    if (!isCurrent() || live.current.open || !live.current.present) return false
    const next = { ...live.current, present: false }
    live.current = next
    publish(next)
    return true
  }, [isCurrent])
  const retire = React.useCallback(() => {
    if (!mounted.current) return
    const next = { open: false, present: false, revision: live.current.revision + 1 }
    live.current = next
    publish(next)
  }, [])
  return { open: state.open, present: state.present, setOpen, afterClose, isCurrent, retire }
}
