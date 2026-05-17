import { describe, expect, it } from 'vitest'

import {
  buildReadAlongSegments,
  resolveReadAlongScope,
  splitSegmentForTtsRequest
} from '../media-read-along-segments'

describe('media read-along segmentation', () => {
  it('segments leading transcript timings as transcript lines with timestamps', () => {
    const content = '[00:01] First line.\n[00:04] Second line.'

    const segments = buildReadAlongSegments({
      mediaId: 'm1',
      content,
      displayContent: 'First line.\nSecond line.',
      renderMode: 'plain',
      hideTranscriptTimings: true
    })

    expect(segments).toHaveLength(2)
    expect(segments[0]).toMatchObject({
      id: 'm1:0:transcript-line:8:19',
      index: 0,
      kind: 'transcript-line',
      text: 'First line.',
      sourceStart: 8,
      sourceEnd: 19,
      timestampSeconds: 1
    })
    expect(segments[1]).toMatchObject({
      id: 'm1:1:transcript-line:28:40',
      kind: 'transcript-line',
      text: 'Second line.',
      timestampSeconds: 4
    })
    expect(segments[1].sourceStart).toBeGreaterThan(segments[0].sourceEnd)
  })

  it('segments prose sentences with deterministic offsets', () => {
    const content = 'Alpha one. Beta two!\n\nGamma three? Delta four.'

    const segments = buildReadAlongSegments({
      mediaId: 'm2',
      content,
      displayContent: 'Alpha one. Beta two!',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })

    expect(segments.map((segment) => segment.text)).toEqual([
      'Alpha one.',
      'Beta two!',
      'Gamma three?',
      'Delta four.'
    ])
    expect(
      segments.map((segment) => ({
        id: segment.id,
        sourceStart: segment.sourceStart,
        sourceEnd: segment.sourceEnd
      }))
    ).toEqual([
      { id: 'm2:0:sentence:0:10', sourceStart: 0, sourceEnd: 10 },
      { id: 'm2:1:sentence:11:20', sourceStart: 11, sourceEnd: 20 },
      { id: 'm2:2:sentence:22:34', sourceStart: 22, sourceEnd: 34 },
      { id: 'm2:3:sentence:35:46', sourceStart: 35, sourceEnd: 46 }
    ])
  })

  it('resolves read-from-here against canonical full content, not a rendered window', () => {
    const content = 'Alpha one. Beta two. Gamma three. Delta four.'
    const segments = buildReadAlongSegments({
      mediaId: 'm3',
      content,
      displayContent: 'Alpha one. Beta two.',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })

    const queue = resolveReadAlongScope({
      scope: 'from-here',
      segments,
      selection: {
        selectedText: 'Beta',
        mappingConfidence: 'nearest',
        sourceStart: content.indexOf('Beta'),
        sourceEnd: content.indexOf('Beta') + 'Beta'.length,
        anchorRect: new DOMRect()
      }
    })

    expect(queue.map((segment) => segment.text)).toEqual([
      'Beta two.',
      'Gamma three.',
      'Delta four.'
    ])
  })

  it('returns every canonical segment for full-item scope', () => {
    const content = 'Alpha one. Beta two. Gamma three.'
    const segments = buildReadAlongSegments({
      mediaId: 'm4',
      content,
      displayContent: 'Alpha one.',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })

    const queue = resolveReadAlongScope({
      scope: 'full-item',
      segments,
      selection: {
        selectedText: 'Alpha',
        mappingConfidence: 'exact',
        sourceStart: 0,
        sourceEnd: 5,
        anchorRect: new DOMRect()
      }
    })

    expect(queue.map((segment) => segment.text)).toEqual([
      'Alpha one.',
      'Beta two.',
      'Gamma three.'
    ])
  })

  it('expands current-section using heading and paragraph metadata', () => {
    const content = '# Intro\nAlpha one. Beta two.\n\n# Details\nGamma three. Delta four.'
    const segments = buildReadAlongSegments({
      mediaId: 'm5',
      content,
      displayContent: content,
      renderMode: 'markdown',
      hideTranscriptTimings: false
    })

    const queue = resolveReadAlongScope({
      scope: 'current-section',
      segments,
      selection: {
        selectedText: 'Gamma',
        mappingConfidence: 'nearest',
        sourceStart: content.indexOf('Gamma'),
        sourceEnd: content.indexOf('Gamma') + 'Gamma'.length,
        anchorRect: new DOMRect()
      }
    })

    expect(queue.map((segment) => segment.text)).toEqual([
      'Gamma three.',
      'Delta four.'
    ])
    expect(new Set(queue.map((segment) => segment.sectionId))).toEqual(
      new Set(['section-1'])
    )
  })

  it('falls back to a transient selection when selection cannot map to source segments', () => {
    const segments = buildReadAlongSegments({
      mediaId: 'm6',
      content: 'Alpha one. Beta two.',
      displayContent: 'Alpha one. Beta two.',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })

    const queue = resolveReadAlongScope({
      scope: 'selection',
      segments,
      selection: {
        selectedText: 'Detached rendered text',
        mappingConfidence: 'text-only',
        anchorRect: new DOMRect()
      }
    })

    expect(queue).toEqual([
      expect.objectContaining({
        id: 'm6:0:transient-selection:text-0babd90f:0:22',
        index: 0,
        kind: 'transient-selection',
        text: 'Detached rendered text',
        sourceStart: 0,
        sourceEnd: 22
      })
    ])
  })

  it('builds deterministic non-colliding transient fallback ids', () => {
    const firstSegments = buildReadAlongSegments({
      mediaId: 'media-a',
      content: 'Alpha one.',
      displayContent: 'Alpha one.',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })
    const secondSegments = buildReadAlongSegments({
      mediaId: 'media-b',
      content: 'Alpha one.',
      displayContent: 'Alpha one.',
      renderMode: 'plain',
      hideTranscriptTimings: false
    })

    const firstOffsetQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: firstSegments,
      selection: {
        selectedText: 'Same',
        mappingConfidence: 'text-only',
        sourceStart: 50,
        sourceEnd: 54,
        anchorRect: new DOMRect()
      }
    })
    const secondOffsetQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: firstSegments,
      selection: {
        selectedText: 'Same',
        mappingConfidence: 'text-only',
        sourceStart: 60,
        sourceEnd: 64,
        anchorRect: new DOMRect()
      }
    })
    const secondMediaQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: secondSegments,
      selection: {
        selectedText: 'Same',
        mappingConfidence: 'text-only',
        sourceStart: 50,
        sourceEnd: 54,
        anchorRect: new DOMRect()
      }
    })
    const firstTextOnlyQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: firstSegments,
      selection: {
        selectedText: 'WXYZ',
        mappingConfidence: 'text-only',
        anchorRect: new DOMRect()
      }
    })
    const secondTextOnlyQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: firstSegments,
      selection: {
        selectedText: 'ABCD',
        mappingConfidence: 'text-only',
        anchorRect: new DOMRect()
      }
    })
    const repeatTextOnlyQueue = resolveReadAlongScope({
      scope: 'selection',
      segments: firstSegments,
      selection: {
        selectedText: 'WXYZ',
        mappingConfidence: 'text-only',
        anchorRect: new DOMRect()
      }
    })

    expect(firstOffsetQueue[0].id).toBe('media-a:0:transient-selection:50:54')
    expect(secondOffsetQueue[0].id).toBe('media-a:0:transient-selection:60:64')
    expect(secondMediaQueue[0].id).toBe('media-b:0:transient-selection:50:54')
    expect(firstTextOnlyQueue[0].id).toMatch(
      /^media-a:0:transient-selection:text-[a-f0-9]+:0:4$/
    )
    expect(secondTextOnlyQueue[0].id).toMatch(
      /^media-a:0:transient-selection:text-[a-f0-9]+:0:4$/
    )
    expect(firstTextOnlyQueue[0].id).toBe(repeatTextOnlyQueue[0].id)
    expect(
      new Set([
        firstOffsetQueue[0].id,
        secondOffsetQueue[0].id,
        secondMediaQueue[0].id,
        firstTextOnlyQueue[0].id,
        secondTextOnlyQueue[0].id
      ]).size
    ).toBe(5)
  })

  it('splits overlong segment requests and preserves parent segment id', () => {
    const parts = splitSegmentForTtsRequest(
      {
        id: 's1',
        index: 0,
        kind: 'sentence',
        text: 'word '.repeat(80).trim(),
        sourceStart: 0,
        sourceEnd: 399
      },
      120
    )

    expect(parts.length).toBeGreaterThan(1)
    expect(parts.every((part) => part.parentSegmentId === 's1')).toBe(true)
    expect(parts.every((part) => part.text.length <= 120)).toBe(true)
    expect(parts.map((part) => part.text).join(' ')).toBe('word '.repeat(80).trim())
  })
})
