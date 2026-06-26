import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { JSONContent } from "@tiptap/react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  getManuscriptScene,
  updateManuscriptScene,
  type ManuscriptSceneResponse
} from "@/services/writing-playground"
import { useActiveManuscriptScene } from "../hooks/useActiveManuscriptScene"

vi.mock("@/services/writing-playground", () => ({
  getManuscriptScene: vi.fn(),
  updateManuscriptScene: vi.fn()
}))

type NodeType = "part" | "chapter" | "scene" | null

type HookProps = {
  activeNodeId: string | null
  activeNodeType: NodeType
  initialEditorText?: string
  initialTipTapContent?: JSONContent | null
}

const sceneContent = (text: string): JSONContent => ({
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text }]
    }
  ]
})

const makeScene = (
  overrides: Partial<ManuscriptSceneResponse> = {}
): ManuscriptSceneResponse => ({
  id: "scene-1",
  chapter_id: "chapter-1",
  project_id: "project-1",
  title: "Scene 1",
  sort_order: 1,
  content: sceneContent("Saved scene text") as Record<string, unknown>,
  content_plain: "Saved scene text",
  synopsis: null,
  word_count: 3,
  pov_character_id: null,
  status: "draft",
  created_at: "2026-06-23T12:00:00Z",
  last_modified: "2026-06-23T12:00:00Z",
  deleted: false,
  client_id: "test-client",
  version: 3,
  ...overrides
})

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity
      }
    }
  })

function renderActiveSceneHook(initialProps: HookProps) {
  const queryClient = createQueryClient()
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )

  const hook = renderHook(
    (props: HookProps) => {
      const [editorText, setEditorText] = React.useState(
        props.initialEditorText ?? ""
      )
      const [tipTapContent, setTipTapContent] =
        React.useState<JSONContent | null>(
          props.initialTipTapContent ?? null
        )
      const binding = useActiveManuscriptScene({
        activeNodeId: props.activeNodeId,
        activeNodeType: props.activeNodeType,
        editorText,
        setEditorText,
        tipTapContent,
        setTipTapContent
      })

      return {
        binding,
        editorText,
        setEditorText,
        tipTapContent,
        setTipTapContent
      }
    },
    { initialProps, wrapper }
  )

  return { ...hook, queryClient }
}

beforeEach(() => {
  vi.mocked(getManuscriptScene).mockReset()
  vi.mocked(updateManuscriptScene).mockReset()
})

afterEach(() => {
  vi.clearAllMocks()
})

describe("useActiveManuscriptScene", () => {
  it('does not query when active node type is not "scene"', () => {
    const { result } = renderActiveSceneHook({
      activeNodeId: "chapter-1",
      activeNodeType: "chapter"
    })

    expect(getManuscriptScene).not.toHaveBeenCalled()
    expect(result.current.binding.scene).toBeNull()
    expect(result.current.binding.isSceneBound).toBe(false)
  })

  it("loads saved content_plain and content into editor state when a scene becomes active", async () => {
    const scene = makeScene()
    vi.mocked(getManuscriptScene).mockResolvedValue(scene)

    const { result } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.editorText).toBe("Saved scene text")
    })

    expect(getManuscriptScene).toHaveBeenCalledTimes(1)
    expect(getManuscriptScene).toHaveBeenCalledWith("scene-1")
    expect(result.current.tipTapContent).toEqual(scene.content)
    expect(result.current.binding.scene).toEqual(scene)
  })

  it("tracks isSceneBound, sceneId, sceneVersion, and dirty state", async () => {
    vi.mocked(getManuscriptScene).mockResolvedValue(makeScene())

    const { result } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.binding.isSceneBound).toBe(true)
    })

    expect(result.current.binding.sceneId).toBe("scene-1")
    expect(result.current.binding.sceneVersion).toBe(3)
    expect(result.current.binding.isSceneDirty).toBe(false)
    expect(result.current.binding.canCreateRangeAnnotation).toBe(true)

    act(() => {
      result.current.setEditorText("Unsaved scene text")
    })

    expect(result.current.binding.sceneId).toBe("scene-1")
    expect(result.current.binding.sceneVersion).toBe(3)
    expect(result.current.binding.isSceneDirty).toBe(true)
    expect(result.current.binding.canCreateRangeAnnotation).toBe(false)
  })

  it("save calls updateManuscriptScene with the saved version", async () => {
    const scene = makeScene()
    const nextRichContent = sceneContent("Saved edited scene")
    const savedScene = makeScene({
      content: nextRichContent as Record<string, unknown>,
      content_plain: "Saved edited scene",
      version: 4
    })
    vi.mocked(getManuscriptScene).mockResolvedValue(scene)
    vi.mocked(updateManuscriptScene).mockResolvedValue(savedScene)

    const { result } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.binding.sceneId).toBe("scene-1")
    })

    act(() => {
      result.current.setEditorText("Saved edited scene")
      result.current.setTipTapContent(nextRichContent)
    })

    await act(async () => {
      await result.current.binding.saveScene()
    })

    expect(updateManuscriptScene).toHaveBeenCalledWith(
      "scene-1",
      {
        content_plain: "Saved edited scene",
        content: nextRichContent
      },
      3
    )
    expect(result.current.binding.sceneVersion).toBe(4)
    expect(result.current.binding.isSceneDirty).toBe(false)
  })

  it("plain text edits save matching rich JSON instead of stale scene content", async () => {
    const scene = makeScene()
    const expectedRichContent = sceneContent("Plain edited scene")
    const savedScene = makeScene({
      content: expectedRichContent as Record<string, unknown>,
      content_plain: "Plain edited scene",
      version: 4
    })
    vi.mocked(getManuscriptScene).mockResolvedValue(scene)
    vi.mocked(updateManuscriptScene).mockResolvedValue(savedScene)

    const { result } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.binding.sceneId).toBe("scene-1")
    })

    act(() => {
      result.current.setEditorText("Plain edited scene")
    })

    await act(async () => {
      await result.current.binding.saveScene()
    })

    expect(updateManuscriptScene).toHaveBeenCalledWith(
      "scene-1",
      {
        content_plain: "Plain edited scene",
        content: expectedRichContent
      },
      3
    )
  })

  it("annotation range actions are disabled when editor text differs from saved scene text", async () => {
    vi.mocked(getManuscriptScene).mockResolvedValue(makeScene())

    const { result } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.binding.canCreateRangeAnnotation).toBe(true)
    })

    act(() => {
      result.current.setEditorText("Saved scene text with a local edit")
    })

    expect(result.current.binding.canCreateRangeAnnotation).toBe(false)
  })

  it("preserves dirty editor content when a different active manuscript scene is selected", async () => {
    vi.mocked(getManuscriptScene).mockImplementation(async (sceneId) => {
      if (sceneId === "scene-2") {
        return makeScene({
          id: "scene-2",
          title: "Scene 2",
          content: sceneContent("Second saved scene") as Record<string, unknown>,
          content_plain: "Second saved scene",
          version: 8
        })
      }
      return makeScene()
    })

    const { result, rerender } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.editorText).toBe("Saved scene text")
    })

    act(() => {
      result.current.setEditorText("Unsaved scene one")
    })
    rerender({
      activeNodeId: "scene-2",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(getManuscriptScene).toHaveBeenCalledWith("scene-2")
    })

    expect(result.current.editorText).toBe("Unsaved scene one")
    expect(result.current.binding.isSceneBound).toBe(false)
    expect(result.current.binding.sceneId).toBeNull()
    expect(result.current.binding.sceneVersion).toBeNull()
    expect(result.current.binding.isSceneDirty).toBe(false)
    expect(result.current.binding.canCreateRangeAnnotation).toBe(false)
  })

  it("does not expose or save the previous scene while a new scene selection is loading", async () => {
    let resolveSecondScene:
      | ((scene: ManuscriptSceneResponse) => void)
      | null = null
    vi.mocked(getManuscriptScene).mockImplementation((sceneId) => {
      if (sceneId === "scene-2") {
        return new Promise<ManuscriptSceneResponse>((resolve) => {
          resolveSecondScene = resolve
        })
      }
      return Promise.resolve(makeScene())
    })
    vi.mocked(updateManuscriptScene).mockResolvedValue(makeScene({ version: 4 }))

    const { result, rerender } = renderActiveSceneHook({
      activeNodeId: "scene-1",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(result.current.binding.sceneId).toBe("scene-1")
    })

    rerender({
      activeNodeId: "scene-2",
      activeNodeType: "scene"
    })

    await waitFor(() => {
      expect(getManuscriptScene).toHaveBeenCalledWith("scene-2")
    })

    expect(result.current.binding.isSceneBound).toBe(false)
    expect(result.current.binding.sceneId).toBeNull()
    expect(result.current.binding.canCreateRangeAnnotation).toBe(false)

    act(() => {
      result.current.setEditorText("Edit during transition")
    })

    await act(async () => {
      await result.current.binding.saveScene()
    })

    expect(updateManuscriptScene).not.toHaveBeenCalled()

    await act(async () => {
      resolveSecondScene?.(
        makeScene({
          id: "scene-2",
          title: "Scene 2",
          content: sceneContent("Second saved scene") as Record<string, unknown>,
          content_plain: "Second saved scene",
          version: 8
        })
      )
    })

    await waitFor(() => {
      expect(result.current.binding.sceneId).toBe("scene-2")
    })
    expect(result.current.editorText).toBe("Second saved scene")
  })
})
