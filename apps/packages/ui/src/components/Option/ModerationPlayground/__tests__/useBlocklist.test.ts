// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { renderHook, act } from "@testing-library/react"

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const getBlocklistMock = vi.fn()
const updateBlocklistMock = vi.fn()
const lintBlocklistMock = vi.fn()
const getManagedBlocklistMock = vi.fn()
const appendManagedBlocklistMock = vi.fn()
const deleteManagedBlocklistItemMock = vi.fn()

vi.mock("@/services/moderation", () => ({
  getBlocklist: (...args: unknown[]) => getBlocklistMock(...args),
  updateBlocklist: (...args: unknown[]) => updateBlocklistMock(...args),
  lintBlocklist: (...args: unknown[]) => lintBlocklistMock(...args),
  getManagedBlocklist: (...args: unknown[]) => getManagedBlocklistMock(...args),
  appendManagedBlocklist: (...args: unknown[]) => appendManagedBlocklistMock(...args),
  deleteManagedBlocklistItem: (...args: unknown[]) => deleteManagedBlocklistItemMock(...args)
}))

import { useBlocklist } from "../hooks/useBlocklist"

describe("useBlocklist", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    lintBlocklistMock.mockImplementation((payload: { line?: string; lines?: string[] }) => {
      const lines = payload.lines ?? (payload.line ? [payload.line] : [])
      return Promise.resolve({
        items: lines.map((line, index) => ({
          index,
          line,
          ok: true,
          pattern_type: line.trim().startsWith("/") ? "regex" : "literal",
          action: "block",
          categories: ["uncategorized"],
          sample: line
        })),
        valid_count: lines.length,
        invalid_count: 0
      })
    })
  })

  it("returns initial state", () => {
    const { result } = renderHook(() => useBlocklist())
    expect(result.current.rawText).toBe("")
    expect(result.current.rawLint).toBeNull()
    expect(result.current.isDirtyRaw).toBe(false)
    expect(result.current.managedItems).toEqual([])
    expect(result.current.managedVersion).toBe("")
    expect(result.current.managedLine).toBe("")
    expect(result.current.managedLint).toBeNull()
    expect(result.current.loading).toBe(false)
  })

  it("loadRaw fetches blocklist and joins lines", async () => {
    getBlocklistMock.mockResolvedValue(["line1", "line2", "line3"])

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadRaw()
    })

    expect(result.current.rawText).toBe("line1\nline2\nline3")
    expect(result.current.rawLint).toBeNull()
  })

  it("loadRaw handles null response", async () => {
    getBlocklistMock.mockResolvedValue(null)

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadRaw()
    })

    expect(result.current.rawText).toBe("")
  })

  it("saveRaw splits text and calls updateBlocklist", async () => {
    updateBlocklistMock.mockResolvedValue({ status: "ok", count: 2 })
    lintBlocklistMock.mockResolvedValue({ items: [], valid_count: 2, invalid_count: 0 })

    const { result } = renderHook(() => useBlocklist())

    act(() => {
      result.current.setRawText("rule1\nrule2  ")
    })

    await act(async () => {
      await result.current.saveRaw()
    })

    expect(updateBlocklistMock).not.toHaveBeenCalled()
    expect(lintBlocklistMock).toHaveBeenCalledWith({ lines: ["rule1", "rule2"] })
    expect(result.current.pendingRawPreview).toMatchObject({
      nextText: "rule1\nrule2",
      lint: { valid_count: 2, invalid_count: 0 }
    })

    await act(async () => {
      await result.current.confirmRawReplace()
    })

    expect(updateBlocklistMock).toHaveBeenCalledWith(["rule1", "rule2"])
    expect(result.current.isDirtyRaw).toBe(false)
  })

  it("tracks raw editor dirtiness relative to the loaded baseline", async () => {
    getBlocklistMock.mockResolvedValue(["rule1", "rule2"])

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadRaw()
    })

    expect(result.current.isDirtyRaw).toBe(false)

    act(() => {
      result.current.setRawText("rule1\nrule2\nrule3")
    })

    expect(result.current.isDirtyRaw).toBe(true)
  })

  it("lintRaw calls lintBlocklist and stores result", async () => {
    const lintResult = { items: [], valid_count: 1, invalid_count: 0 }
    lintBlocklistMock.mockResolvedValue(lintResult)

    const { result } = renderHook(() => useBlocklist())

    act(() => {
      result.current.setRawText("test rule")
    })

    await act(async () => {
      await result.current.lintRaw()
    })

    expect(lintBlocklistMock).toHaveBeenCalledWith({ lines: ["test rule"] })
    expect(result.current.rawLint).toEqual(lintResult)
  })

  it("loadManaged fetches managed blocklist", async () => {
    getManagedBlocklistMock.mockResolvedValue({
      data: { version: "v1", items: [{ id: 1, line: "bad-word" }] },
      etag: "etag1"
    })
    lintBlocklistMock.mockResolvedValue({
      items: [
        {
          index: 0,
          line: "bad-word",
          ok: true,
          pattern_type: "literal",
          action: "block",
          categories: ["uncategorized"],
          sample: "bad-word"
        }
      ],
      valid_count: 1,
      invalid_count: 0
    })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    expect(lintBlocklistMock).toHaveBeenCalledWith({ lines: ["bad-word"] })
    expect(result.current.managedItems).toEqual([
      expect.objectContaining({
        id: 1,
        line: "bad-word",
        pattern_type: "literal",
        action: "block",
        categories: ["uncategorized"],
        ok: true
      })
    ])
    expect(result.current.managedVersion).toBe("v1")
  })

  it("does not relint managed rows when backend metadata is already present", async () => {
    getManagedBlocklistMock.mockResolvedValue({
      data: {
        version: "v1",
        items: [{ id: 1, line: "bad-word", pattern_type: "literal", action: "block", ok: true }]
      },
      etag: null
    })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    expect(lintBlocklistMock).not.toHaveBeenCalled()
    expect(result.current.managedItems[0]).toMatchObject({
      pattern_type: "literal",
      action: "block",
      ok: true
    })
  })

  it("blocks raw replace confirmation when preview lint has invalid rows", async () => {
    lintBlocklistMock.mockResolvedValue({
      items: [{ index: 0, line: "/[bad/", ok: false, pattern_type: "regex", error: "invalid regex" }],
      valid_count: 0,
      invalid_count: 1
    })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.previewRawReplace("/[bad/")
    })

    await expect(
      act(async () => {
        await result.current.confirmRawReplace()
      })
    ).rejects.toThrow("Fix invalid blocklist rows before replacing")
    expect(updateBlocklistMock).not.toHaveBeenCalled()
  })

  it("can undo a confirmed raw replace with the previous baseline", async () => {
    getBlocklistMock.mockResolvedValue(["old"])
    lintBlocklistMock.mockResolvedValue({ items: [], valid_count: 1, invalid_count: 0 })
    updateBlocklistMock.mockResolvedValue({ status: "ok", count: 1 })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadRaw()
      await result.current.previewRawReplace("new")
      await result.current.confirmRawReplace()
    })

    expect(result.current.rawText).toBe("new")
    expect(result.current.rawReplaceUndo).toMatchObject({ previousText: "old" })

    await act(async () => {
      await result.current.undoRawReplace()
    })

    expect(updateBlocklistMock).toHaveBeenLastCalledWith(["old"])
    expect(result.current.rawText).toBe("old")
    expect(result.current.rawReplaceUndo).toBeNull()
  })

  it("loadManaged falls back to etag for version", async () => {
    getManagedBlocklistMock.mockResolvedValue({
      data: { version: "", items: [] },
      etag: "etag-fallback"
    })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    expect(result.current.managedVersion).toBe("etag-fallback")
  })

  it("appendManaged throws if no version loaded", async () => {
    const { result } = renderHook(() => useBlocklist())

    act(() => {
      result.current.setManagedLine("new rule")
    })

    await expect(
      act(async () => {
        await result.current.appendManaged()
      })
    ).rejects.toThrow("Load the managed blocklist first")
  })

  it("appendManaged throws if line is empty", async () => {
    getManagedBlocklistMock.mockResolvedValue({
      data: { version: "v1", items: [] },
      etag: null
    })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    await expect(
      act(async () => {
        await result.current.appendManaged()
      })
    ).rejects.toThrow("Enter a line to append")
  })

  it("appendManaged calls API and reloads", async () => {
    appendManagedBlocklistMock.mockResolvedValue({ version: "v2", index: 1, count: 2 })
    getManagedBlocklistMock
      .mockResolvedValueOnce({
        data: { version: "v1", items: [{ id: 1, line: "old" }] },
        etag: null
      })
      .mockResolvedValueOnce({
        data: { version: "v2", items: [{ id: 1, line: "old" }, { id: 2, line: "new" }] },
        etag: null
      })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    act(() => {
      result.current.setManagedLine("new")
    })

    await act(async () => {
      await result.current.appendManaged()
    })

    expect(appendManagedBlocklistMock).toHaveBeenCalledWith("v1", "new")
    expect(result.current.managedLine).toBe("")
    expect(result.current.managedItems).toHaveLength(2)
    expect(result.current.managedVersion).toBe("v2")
  })

  it("deleteManaged calls API and reloads", async () => {
    deleteManagedBlocklistItemMock.mockResolvedValue({ version: "v2", count: 0 })
    getManagedBlocklistMock
      .mockResolvedValueOnce({
        data: { version: "v1", items: [{ id: 1, line: "bad" }] },
        etag: null
      })
      .mockResolvedValueOnce({
        data: { version: "v2", items: [] },
        etag: null
      })

    const { result } = renderHook(() => useBlocklist())

    await act(async () => {
      await result.current.loadManaged()
    })

    await act(async () => {
      await result.current.deleteManaged(1)
    })

    expect(deleteManagedBlocklistItemMock).toHaveBeenCalledWith("v1", 1)
    expect(result.current.managedItems).toEqual([])
  })

  it("lintManagedLine throws if line is empty", async () => {
    const { result } = renderHook(() => useBlocklist())

    await expect(
      act(async () => {
        await result.current.lintManagedLine()
      })
    ).rejects.toThrow("Enter a line to lint")
  })

  it("lintManagedLine calls lintBlocklist with single line", async () => {
    const lintResult = { items: [{ index: 0, line: "test", ok: true }], valid_count: 1, invalid_count: 0 }
    lintBlocklistMock.mockResolvedValue(lintResult)

    const { result } = renderHook(() => useBlocklist())

    act(() => {
      result.current.setManagedLine("test")
    })

    await act(async () => {
      await result.current.lintManagedLine()
    })

    expect(lintBlocklistMock).toHaveBeenCalledWith({ line: "test" })
    expect(result.current.managedLint).toEqual(lintResult)
  })
})
