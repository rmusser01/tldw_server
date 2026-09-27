import * as React from 'react'

/** Private Notes state must disappear as soon as its verified account changes. */
export function useNotesAuthorityState<T>(scope: string | null | undefined, initialValue: T) {
  const initial = React.useRef(initialValue)
  const authority = React.useRef({ scope, generation: 0 })
  if (authority.current.scope !== scope) {
    authority.current = { scope, generation: authority.current.generation + 1 }
  }
  const generation = authority.current.generation
  const [state, setState] = React.useState({ generation, value: initial.current })
  const value = scope && state.generation === generation ? state.value : initial.current
  const setValue = React.useCallback((update: React.SetStateAction<T>) => {
    if (!scope || authority.current.generation !== generation) return
    setState((current) => {
      if (authority.current.generation !== generation) return current
      const previous = current.generation === generation ? current.value : initial.current
      const next = typeof update === 'function' ? (update as (value: T) => T)(previous) : update
      return { generation, value: next }
    })
  }, [generation, scope])
  return [value, setValue] as const
}
