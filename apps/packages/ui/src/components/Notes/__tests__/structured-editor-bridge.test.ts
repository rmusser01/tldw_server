import { describe, expect, it, vi } from 'vitest'
import { TextAreaFormatBridge } from './fixtures/native-textarea-format-candidate'
import { createDocumentRequest, createCheckResult, createFormatResult } from '@/services/document-editing/types'

const key = { authority_id: 'local', document_id: 'note', session_generation: 1, draft_revision: 1, language_revision: 1, request_id: 1 }
const request = createDocumentRequest({ key, text: '{"a":1}', language: 'json' })
const result = createFormatResult({ key, state: 'formatted', replacement: '{\n  "a": 1\n}', check: createCheckResult(request, { state: 'checked', complete: true, diagnostics: [], truncated: false, format_allowed: true }) })
function setup() {
  const textarea = document.createElement('textarea')
  document.body.replaceChildren(textarea)
  textarea.value = request.text
  textarea.focus()
  const edit = vi.fn()
  const bridge = new TextAreaFormatBridge(textarea, edit)
  bridge.captureRequest(request)
  // jsdom has no history implementation; live Playwright covers this boundary.
  const command = vi.fn((_name, _ui, text) => { textarea.value = text; return true })
  Object.defineProperty(document, 'execCommand', { configurable: true, value: command })
  return { textarea, bridge, command, edit }
}
describe('guarded native formatting', () => {
  it('declines browsers without a native insert command', () => {
    const {bridge} = setup()
    Object.defineProperty(document, 'execCommand', { configurable: true, value: undefined })
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    bridge.dispose()
  })
  it('rejects pending widget typing before host revision catches up', () => {
    const {textarea, bridge, command} = setup()
    textarea.value += ' '
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    expect(command).not.toHaveBeenCalled()
    bridge.dispose()
  })
  it('requires every current and captured identity field', () => {
    const {bridge, command} = setup()
    for (const field of Object.keys(key)) {
      expect(bridge.applyFormat(result, { ...key, [field]: 'other' }, true)).toBe(false)
    }
    bridge.captureRequest(createDocumentRequest({ ...request, key: {...key, request_id: 2} }))
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    expect(command).not.toHaveBeenCalled()
    bridge.dispose()
  })
  it('rejects read-only, conflict, disabled, composition, focus loss, and detached editor', () => {
    const {textarea, bridge, command} = setup()
    expect(bridge.applyFormat(result, key, false)).toBe(false)
    textarea.readOnly = true
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    textarea.readOnly = false
    textarea.disabled = true
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    textarea.disabled = false
    textarea.focus()
    textarea.dispatchEvent(new CompositionEvent('compositionstart'))
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    textarea.dispatchEvent(new CompositionEvent('compositionend'))
    textarea.blur()
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    textarea.remove()
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    expect(command).not.toHaveBeenCalled()
    bridge.dispose()
  })
  it('applies once, maps selection, restores scroll, and retires the snapshot', () => {
    const {textarea, bridge, command, edit} = setup()
    textarea.setSelectionRange(1, 4, 'backward')
    textarea.scrollTop = 17
    expect(bridge.applyFormat(result, key, true)).toBe(true)
    expect(command).toHaveBeenCalledWith('insertText', false, result.replacement)
    expect(textarea.value.slice(textarea.selectionStart, textarea.selectionEnd)).toBe('"a"')
    expect(textarea.selectionDirection).toBe('backward')
    expect(textarea.scrollTop).toBe(17)
    expect(edit).toHaveBeenCalledWith(result.replacement)
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    bridge.dispose()
  })
  it('no-op, failed native command and disposal add no user edit', () => {
    const {bridge, command, edit} = setup()
    expect(bridge.applyFormat({...result, replacement: request.text}, key, true)).toBe(false)
    command.mockImplementation(() => false)
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    bridge.dispose()
    expect(bridge.applyFormat(result, key, true)).toBe(false)
    expect(edit).not.toHaveBeenCalled()
  })
})
