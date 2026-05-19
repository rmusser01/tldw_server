import React from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useContentSelectionActions } from '../useContentSelectionActions'

function setSelectionRange(startNode: Node, startOffset: number, endNode: Node, endOffset: number) {
  const selection = window.getSelection()
  expect(selection).not.toBeNull()

  const range = document.createRange()
  range.setStart(startNode, startOffset)
  range.setEnd(endNode, endOffset)
  Object.defineProperty(range, 'getBoundingClientRect', {
    configurable: true,
    value: vi.fn(() => new DOMRect(10, 20, 120, 18))
  })

  selection!.removeAllRanges()
  selection!.addRange(range)
}

describe('useContentSelectionActions', () => {
  beforeEach(() => {
    document.body.innerHTML = ''
    window.getSelection()?.removeAllRanges()
    vi.restoreAllMocks()
  })

  it('ignores selections outside contentBodyRef', () => {
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Inside content'
    const outside = document.createElement('p')
    outside.textContent = 'Outside content'
    document.body.append(contentBody, outside)

    setSelectionRange(outside.firstChild as Text, 0, outside.firstChild as Text, 7)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection: vi.fn()
      })
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })

    expect(result.current.selectionActionState).toBeNull()
  })

  it('returns selected text and anchor rect for content selections', () => {
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Selected content text'
    document.body.append(contentBody)
    setSelectionRange(contentBody.firstChild as Text, 0, contentBody.firstChild as Text, 8)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection: vi.fn()
      })
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })

    expect(result.current.selectionActionState).toMatchObject({
      selectedText: 'Selected',
      mappingConfidence: 'text-only'
    })
    expect(result.current.selectionActionState?.anchorRect).toMatchObject({
      x: 10,
      y: 20,
      width: 120,
      height: 18
    })
  })

  it('maps exact data-read-along-segment-id ancestors when present', () => {
    const contentBody = document.createElement('div')
    const startSegment = document.createElement('span')
    startSegment.dataset.readAlongSegmentId = 'segment-start'
    startSegment.textContent = 'Start segment'
    const endSegment = document.createElement('span')
    endSegment.dataset.readAlongSegmentId = 'segment-end'
    endSegment.textContent = 'End segment'
    contentBody.append(startSegment, endSegment)
    document.body.append(contentBody)
    setSelectionRange(
      startSegment.firstChild as Text,
      0,
      endSegment.firstChild as Text,
      'End'.length
    )

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection: vi.fn()
      })
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })

    expect(result.current.selectionActionState).toMatchObject({
      startSegmentId: 'segment-start',
      endSegmentId: 'segment-end',
      mappingConfidence: 'exact'
    })
  })

  it('falls back to text-only when no segment wrapper exists', () => {
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Plain selected text'
    document.body.append(contentBody)
    setSelectionRange(contentBody.firstChild as Text, 0, contentBody.firstChild as Text, 5)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection: vi.fn()
      })
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })

    expect(result.current.selectionActionState).toMatchObject({
      selectedText: 'Plain',
      mappingConfidence: 'text-only'
    })
    expect(result.current.selectionActionState?.startSegmentId).toBeUndefined()
    expect(result.current.selectionActionState?.endSegmentId).toBeUndefined()
  })

  it('opens selection actions from document selectionchange before explicit mouseup', async () => {
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Selectionchange selected text'
    document.body.append(contentBody)
    setSelectionRange(contentBody.firstChild as Text, 0, contentBody.firstChild as Text, 15)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection: vi.fn()
      })
    )

    act(() => {
      document.dispatchEvent(new Event('selectionchange'))
    })

    await waitFor(() => {
      expect(result.current.selectionActionState).toMatchObject({
        selectedText: 'Selectionchange',
        mappingConfidence: 'text-only'
      })
    })
  })

  it('clears open selection actions when the document selection is cleared', async () => {
    const onApplyAnnotationSelection = vi.fn()
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Clearable selected text'
    document.body.append(contentBody)
    setSelectionRange(contentBody.firstChild as Text, 0, contentBody.firstChild as Text, 9)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result } = renderHook(() =>
      useContentSelectionActions({
        contentBodyRef: ref,
        contentIdentityKey: 'media:777:plain:a',
        onApplyAnnotationSelection
      })
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })
    expect(result.current.selectionActionState?.selectedText).toBe('Clearable')

    act(() => {
      window.getSelection()?.removeAllRanges()
      document.dispatchEvent(new Event('selectionchange'))
    })

    await waitFor(() => {
      expect(result.current.selectionActionState).toBeNull()
    })

    act(() => {
      result.current.applyAnnotationSelection()
    })
    expect(onApplyAnnotationSelection).not.toHaveBeenCalled()
  })

  it('refuses to apply a selection opened under a previous content identity key', () => {
    const onApplyAnnotationSelection = vi.fn()
    const contentBody = document.createElement('div')
    contentBody.textContent = 'Identity guarded text'
    document.body.append(contentBody)
    setSelectionRange(contentBody.firstChild as Text, 0, contentBody.firstChild as Text, 8)

    const ref = { current: contentBody } as React.RefObject<HTMLDivElement | null>
    const { result, rerender } = renderHook(
      ({ contentIdentityKey }: { contentIdentityKey: string }) =>
        useContentSelectionActions({
          contentBodyRef: ref,
          contentIdentityKey,
          onApplyAnnotationSelection
        }),
      {
        initialProps: { contentIdentityKey: 'media:777:plain:a' }
      }
    )

    act(() => {
      result.current.handleContentSelectionEvent()
    })
    expect(result.current.selectionActionState?.selectedText).toBe('Identity')

    rerender({ contentIdentityKey: 'media:778:plain:b' })

    act(() => {
      result.current.applyAnnotationSelection()
    })

    expect(onApplyAnnotationSelection).not.toHaveBeenCalled()
    expect(result.current.selectionActionState).toBeNull()
  })
})
