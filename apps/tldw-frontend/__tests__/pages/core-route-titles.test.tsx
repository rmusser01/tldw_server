import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import Head from "next/head"
import initHeadManager from "next/dist/client/head-manager"
import { HeadManagerContext } from "next/dist/shared/lib/head-manager-context.shared-runtime"

vi.mock("next/dynamic", () => ({
  default: (_load: unknown, options: { ssr?: boolean }) => {
    if (options.ssr !== false) throw new Error("Shared routes require browser-only loading")
    return () => <main>Shared route</main>
  }
}))

import Home from "@web/pages/index"
import Setup from "@web/pages/setup"
import Media from "@web/pages/media"
import MediaTrash from "@web/pages/media-trash"
import Analysis from "@web/pages/media-multi"
import Knowledge from "@web/pages/knowledge"
import Flashcards from "@web/pages/flashcards"
import ServerSettings from "@web/pages/settings/tldw"
import ProviderKeys from "@web/pages/settings/provider-keys"
import Notes from "@web/pages/notes"

afterEach(() => {
  cleanup()
  document.head.innerHTML = ""
})

describe("core route title ownership", () => {
  it.each([
    [Home, "Home | tldw"],
    [Setup, "Setup | tldw"],
    [Media, "Media | tldw"],
    [MediaTrash, "Trash | tldw"],
    [Analysis, "Media Analysis | tldw"],
    [Knowledge, "Knowledge | tldw"],
    [Flashcards, "Flashcards | tldw"],
    [ServerSettings, "Server Settings | tldw"],
    [ProviderKeys, "Provider Keys | tldw"],
    [Notes, "Notes | tldw"]
  ] as const)("replaces previous private metadata with %s", async (Page, title) => {
    const manager = initHeadManager()
    const { rerender } = render(
      <HeadManagerContext.Provider value={manager}>
        <Head><title>Private Chat</title><meta name="description" content="Private topic" /></Head>
      </HeadManagerContext.Provider>
    )
    await waitFor(() => expect(document.title).toBe("Private Chat"))
    rerender(<HeadManagerContext.Provider value={manager}><Page /></HeadManagerContext.Provider>)
    await waitFor(() => expect(document.title).toBe(title))
    expect(document.querySelector('meta[name="description"]')).toBeNull()
    rerender(
      <HeadManagerContext.Provider value={manager}>
        <Head><title>Signed out | tldw</title></Head>
      </HeadManagerContext.Provider>
    )
    await waitFor(() => expect(document.title).toBe("Signed out | tldw"))
  })
})
