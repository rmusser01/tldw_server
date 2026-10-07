import React from "react"

/**
 * What the side panel decided before a new turn is sent (XP-08, #3105).
 *
 * - `proceed: false`: hold the send. The gate has already told the user why
 *   and put their message back in the composer.
 * - `refreshed: true`: the tab's transcript was reloaded from the server, so
 *   the send must read the chat store again instead of its render snapshot.
 */
export type SidepanelSendGateResult = {
  proceed: boolean
  refreshed: boolean
}

export type SidepanelSendGate = (request: {
  message: string
}) => Promise<SidepanelSendGateResult>

/**
 * The side panel registers its pre-send freshness check here. `useMessage`
 * runs it before each new turn, so a tab that fell behind its server chat is
 * refreshed (or held) instead of silently forking the conversation.
 *
 * The value is a ref so every `useMessage` caller in the panel (composer,
 * background actions) reads the route's latest check without re-rendering.
 */
export const SidepanelSendGateContext =
  React.createContext<React.MutableRefObject<SidepanelSendGate | null> | null>(
    null
  )
