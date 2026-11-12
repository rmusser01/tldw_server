import React from "react"
import { App } from "antd"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ContentEditModal } from "@/components/Media/ContentEditModal"
import { useContentEditState } from "@/components/Media/hooks/useContentEditState"
import { bgRequest } from "@/services/background-proxy"
import { useMediaNavigationState } from "../hooks/useMediaNavigationState"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))
vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/settings/registry")>()),
  clearSetting: vi.fn(),
  getSetting: vi.fn().mockResolvedValue(null),
  setSetting: vi.fn()
}))
vi.mock("@/components/Review/ViewMediaPage", () => ({
  MEDIA_STALE_CHECK_INTERVAL_MS: 60_000
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) => typeof fallback === "string" ? fallback : key
  })
}))

const source = "Rowan Observatory\n\n" +
  "Rowan Observatory opened in 2019 in Cedar Ridge. Its director is Dr. Mira Vale.\n" +
  "The observatory's main telescope is named Selene.\n\n" +
  "Public tours begin every Friday at 18:00. Visitors must reserve a place before\n" +
  "arriving. The tour reference code is ORBIT-742.\n\n" +
  "The observatory studies variable stars and shares a monthly public report.\n" +
  "The source provides no ticket price and no current weather forecast.\n"

const navigationDeps = {
  t: (key: string) => key,
  message: { error: vi.fn(), warning: vi.fn(), success: vi.fn() },
  displayResults: [],
  refetch: vi.fn().mockResolvedValue({ data: [] })
}

function MediaEditorHarness() {
  const navigation = useMediaNavigationState(navigationDeps)
  const edit = useContentEditState({
    selectedMedia: navigation.selected,
    selectedMediaId: navigation.selected ? String(navigation.selected.id) : null,
    content: navigation.selectedContent,
    mediaDetail: navigation.selectedDetail,
    isNote: false,
    t: (key, options) => options?.defaultValue || key
  })
  return <>
    <button disabled={!navigation.selectedDetail} onClick={edit.openContentEditModal}>
      Open source editor
    </button>
    <ContentEditModal
      open={edit.contentEditModalOpen}
      onClose={() => edit.setContentEditModalOpen(false)}
      initialText={edit.editingContentText || navigation.selectedContent}
      mediaId={navigation.selected?.id}
    />
  </>
}

describe("Media detail hydration to Edit Content", () => {
  beforeEach(() => vi.clearAllMocks())

  it("opens the untouched 422-byte uploaded source with its final LF", async () => {
    vi.mocked(bgRequest).mockResolvedValue({
      media_id: 1,
      source: { title: "F-SOURCE" },
      content: { text: source }
    })
    render(<MemoryRouter initialEntries={["/media?id=1"]}>
      <App><MediaEditorHarness /></App>
    </MemoryRouter>)
    const open = screen.getByRole("button", { name: "Open source editor" })
    await waitFor(() => expect(open).not.toBeDisabled())
    fireEvent.click(open)
    const editor = await screen.findByRole<HTMLTextAreaElement>("textbox")
    expect(editor.value).toBe(source)
    expect(editor.value.length).toBe(422)
    expect(vi.mocked(bgRequest).mock.calls.every(([request]) =>
      request.method == null || request.method === "GET"
    )).toBe(true)
  })
})
