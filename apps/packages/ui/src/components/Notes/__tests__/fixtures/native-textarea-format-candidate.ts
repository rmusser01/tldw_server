/** FAILED I1 qualification on WebKit 26.0. Test-only candidate; do not adopt. */
import { createDocumentRequest, type DocumentRequest, type FormatResult, type RequestKey } from '@/services/document-editing/types'

function sameKey(a: RequestKey, b: RequestKey): boolean {
  return a.authority_id === b.authority_id && a.document_id === b.document_id &&
    a.session_generation === b.session_generation && a.draft_revision === b.draft_revision &&
    a.language_revision === b.language_revision && a.request_id === b.request_id
}
function mapOffset(source: string, replacement: string, offset: number): number {
  if (offset === source.length) return replacement.length
  let rank = 0
  for (let i = 0; i < offset; i++) if (!/\s/.test(source[i])) rank++
  let target = replacement.length
  for (let i = 0; i < replacement.length; i++) {
    if (!/\s/.test(replacement[i])) {
      if (rank === 0) { target = i; break }
      rank--
    }
  }
  // Preserve distance to the next anchor within its preceding whitespace run.
  let sourceEnd = offset, runStart = target
  while (sourceEnd < source.length && /\s/.test(source[sourceEnd])) sourceEnd++
  while (runStart > 0 && /\s/.test(replacement[runStart - 1])) runStart--
  return target - Math.min(sourceEnd - offset, target - runStart)
}

export class TextAreaFormatBridge {
  private request: DocumentRequest | null = null
  private composing = false
  private disposed = false
  private readonly startComposition = () => { this.composing = true }
  private readonly endComposition = () => { this.composing = false }

  constructor(private textarea: HTMLTextAreaElement, private onUserEdit: (text: string) => void) {
    textarea.addEventListener('compositionstart', this.startComposition)
    textarea.addEventListener('compositionend', this.endComposition)
  }
  captureRequest(request: DocumentRequest): void {
    this.request = createDocumentRequest(request)
  }
  dispose(): void {
    this.disposed = true
    this.request = null
    this.textarea.removeEventListener('compositionstart', this.startComposition)
    this.textarea.removeEventListener('compositionend', this.endComposition)
  }
  /** Call synchronously from a user gesture with the editor still focused. */
  applyFormat(result: FormatResult, currentKey: RequestKey, editable: boolean): boolean {
    const element = this.textarea, request = this.request, doc = element.ownerDocument
    if (this.disposed || !request || !sameKey(result.key, currentKey) ||
      !sameKey(request.key, currentKey) || !sameKey(result.check.key, currentKey) ||
      request.is_excerpt || !editable || element.readOnly || element.disabled ||
      typeof doc.execCommand !== 'function' || !element.isConnected || doc.activeElement !== element || this.composing ||
      result.state !== 'formatted' || result.replacement === null || !result.check.format_allowed ||
      element.value !== request.text || result.replacement === element.value) return false
    const replacement = result.replacement
    const start = mapOffset(request.text, replacement, element.selectionStart)
    const end = mapOffset(request.text, replacement, element.selectionEnd)
    const direction = element.selectionDirection
    const top = element.scrollTop, left = element.scrollLeft
    const originalStart = element.selectionStart, originalEnd = element.selectionEnd
    // I1 follow-up experiment: attempt a native group boundary on each side.
    element.blur()
    element.focus({ preventScroll: true })
    if (this.disposed || this.request !== request || this.composing ||
      !element.isConnected || element.readOnly || element.disabled ||
      doc.activeElement !== element || element.value !== request.text) return false
    element.select()
    const applied = doc.execCommand('insertText', false, replacement)
    if (!applied || element.value !== replacement) {
      element.setSelectionRange(originalStart, originalEnd, direction)
      element.scrollTop = top; element.scrollLeft = left
      return false
    }
    this.request = null
    this.onUserEdit(element.value)
    element.blur()
    element.focus({ preventScroll: true })
    element.setSelectionRange(start, end, direction)
    element.scrollTop = top; element.scrollLeft = left
    return true
  }
}
