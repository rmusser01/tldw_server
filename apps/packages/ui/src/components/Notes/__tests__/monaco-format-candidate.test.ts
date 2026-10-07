import { describe, expect, it } from 'vitest';
import { sourceProfile } from './fixtures/monaco-format-candidate';

describe('Monaco candidate exact source admission', () => {
  it('admits homogeneous LF/CRLF and refuses representations the model normalizes', () => {
    for (const text of [
      '',
      '{"s":"a&b <tag> 😀  x","n":9007199254740993}  \n',
      'a\r\nb\r\n',
      'a\n',
      'a',
    ]) {
      expect(sourceProfile(text)).not.toBe(null);
    }
    for (const text of ['\ufeff{}', 'a\rb', 'a\r\nb\n']) expect(sourceProfile(text)).toBe(null);
  });
});

// Admission-only fake: native editing/history is exclusively proved by Playwright.
it('refuses same-key source drift before any native transaction', async () => {
  const { MonacoFormatBridge } = await import('./fixtures/monaco-format-candidate');
  const { createDocumentRequest, createCheckResult, createFormatResult } =
    await import('@/services/document-editing/types');
  const { vi } = await import('vitest');
  const request = createDocumentRequest({
    key: {
      authority_id: 'a',
      document_id: 'd',
      session_generation: 1,
      draft_revision: 1,
      language_revision: 1,
      request_id: 1,
    },
    text: '{}',
    language: 'json',
  });
  const check = createCheckResult(request, {
    state: 'checked',
    complete: true,
    diagnostics: [],
    truncated: false,
    format_allowed: true,
  });
  const result = createFormatResult({
    key: request.key,
    state: 'formatted',
    replacement: '{\n}',
    check,
  });
  const model = { isDisposed: () => false, getValue: () => '{ }', getEOL: () => '\n' };
  const executeEdits = vi.fn(),
    pushUndoStop = vi.fn();
  const native = {
    getModel: () => model,
    getDomNode: () => ({ isConnected: true }),
    hasTextFocus: () => true,
    inComposition: false,
    getRawOptions: () => ({ readOnly: false }),
    executeEdits,
    pushUndoStop,
  };
  const bridge = new MonacoFormatBridge(
    native as unknown as ConstructorParameters<typeof MonacoFormatBridge>[0],
    {} as ConstructorParameters<typeof MonacoFormatBridge>[1]
  );
  bridge.captureRequest(request);
  expect(bridge.applyFormat(result, request.key, true)).toBe(false);
  expect(executeEdits).not.toHaveBeenCalled();
  expect(pushUndoStop).not.toHaveBeenCalled();
  bridge.dispose();
  expect(bridge.applyFormat(result, request.key, true)).toBe(false);
});
