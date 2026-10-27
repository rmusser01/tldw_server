// @vitest-environment jsdom
/**
 * CM-09 (#3101): role-play steering is offered only in character chats, and
 * every send applies whatever steering the option store holds. Leaving a
 * character chat (for a plain chat, or by turning on compare) must drop any
 * steering the user armed there, while a character chat keeps it.
 */
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"

import { useStoreMessageOption } from "@/store/option"
import { useCharacterChatSteeringScope } from "../useCharacterChatSteeringScope"

const steering = () => {
  const state = useStoreMessageOption.getState()
  return {
    mode: state.messageSteeringMode,
    forceNarrate: state.messageSteeringForceNarrate
  }
}

const armSteering = () =>
  act(() => {
    const state = useStoreMessageOption.getState()
    state.setMessageSteeringMode("continue_as_user")
    state.setMessageSteeringForceNarrate(true)
  })

describe("useCharacterChatSteeringScope", () => {
  beforeEach(() => {
    useStoreMessageOption.getState().clearMessageSteering()
  })

  it("keeps steering armed while a character chat is active", () => {
    renderHook(() => useCharacterChatSteeringScope(true))

    armSteering()

    expect(steering()).toEqual({ mode: "continue_as_user", forceNarrate: true })
  })

  it("clears steering once the chat is no longer a character chat", () => {
    const { rerender } = renderHook(
      ({ characterChat }) => useCharacterChatSteeringScope(characterChat),
      { initialProps: { characterChat: true } }
    )
    armSteering()

    rerender({ characterChat: false })

    expect(steering()).toEqual({ mode: "none", forceNarrate: false })
  })

  it("clears steering that is still armed when a plain chat opens", () => {
    armSteering()

    renderHook(() => useCharacterChatSteeringScope(false))

    expect(steering()).toEqual({ mode: "none", forceNarrate: false })
  })

  it("clears steering armed while a plain chat is showing", () => {
    renderHook(() => useCharacterChatSteeringScope(false))

    armSteering()

    expect(steering()).toEqual({ mode: "none", forceNarrate: false })
  })
})
