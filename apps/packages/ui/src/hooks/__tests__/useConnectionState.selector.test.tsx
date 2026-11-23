// @vitest-environment jsdom
import React from "react"
import { act, cleanup, render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState,
  useKnowledgeStatus
} from "@/hooks/useConnectionState"
import { useConnectionStore } from "@/store/connection"
import { ConnectionPhase, type ConnectionState } from "@/types/connection"

const initialState = useConnectionStore.getState().state

const patchState = (patch: Partial<ConnectionState>) =>
  act(() => {
    useConnectionStore.setState((prev) => ({ state: { ...prev.state, ...patch } }))
  })

// What one health poll does to the store while connected: an isChecking
// set, then the final set with a new lastCheckedAt.
const simulatePoll = () => {
  patchState({ isChecking: true })
  patchState({ isChecking: false, lastCheckedAt: Date.now() })
}

const countRenders = (useValue: () => unknown) => {
  const seen: unknown[] = []
  const Probe = () => {
    seen.push(useValue())
    return null
  }
  render(<Probe />)
  return seen
}

describe("useConnectionState", () => {
  beforeEach(() => {
    useConnectionStore.setState({
      state: {
        ...initialState,
        phase: ConnectionPhase.CONNECTED,
        isConnected: true,
        lastCheckedAt: 1
      }
    })
  })

  afterEach(() => {
    cleanup()
    useConnectionStore.setState({ state: initialState })
  })

  it("returns the whole state without a selector", () => {
    const seen = countRenders(() => useConnectionState())
    expect(seen.at(-1)).toBe(useConnectionStore.getState().state)
  })

  it("does not re-render a selecting consumer on a health poll", () => {
    const seen = countRenders(() =>
      useConnectionState((s) => ({ isConnected: s.isConnected }))
    )
    expect(seen).toHaveLength(1)

    simulatePoll()
    expect(seen).toHaveLength(1)

    patchState({ isConnected: false, phase: ConnectionPhase.ERROR })
    expect(seen).toHaveLength(2)
    expect(seen.at(-1)).toEqual({ isConnected: false })
  })

  it("keeps the helper hooks stable across a health poll", () => {
    const knowledge = countRenders(() => useKnowledgeStatus())
    const actions = countRenders(() => useConnectionActions())
    const ux = countRenders(() => useConnectionUxState())

    simulatePoll()

    expect(knowledge).toHaveLength(1)
    expect(actions).toHaveLength(1)
    expect(ux).toHaveLength(1)
  })
})
