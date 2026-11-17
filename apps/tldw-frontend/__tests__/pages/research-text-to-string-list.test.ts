import { describe, expect, it, vi } from 'vitest';

// The page module pulls in the API client, which resolves networking config at
// import time; mock it so the pure helper can be imported in isolation (same
// pattern as research-run-console.test.tsx).
vi.mock('@web/lib/api/researchRuns', () => ({}));

import { textToStringList } from '@web/pages/research';

describe('textToStringList (Set dedupe semantics)', () => {
  it('splits on newlines and trims each line', () => {
    expect(textToStringList(' focus \nconstraints\n  open  ')).toEqual([
      'focus',
      'constraints',
      'open',
    ]);
  });

  it('removes duplicates while preserving first-occurrence order', () => {
    expect(textToStringList('b\na\nb\na\nc')).toEqual(['b', 'a', 'c']);
  });

  it('treats lines as duplicates only after trimming', () => {
    expect(textToStringList('a\n a \na')).toEqual(['a']);
  });

  it('drops empty and whitespace-only lines', () => {
    expect(textToStringList('\n\n   \n\t\na\n\n')).toEqual(['a']);
  });

  it('returns an empty list for empty input', () => {
    expect(textToStringList('')).toEqual([]);
  });

  it('handles CRLF input by trimming the carriage return', () => {
    expect(textToStringList('a\r\nb\r\na')).toEqual(['a', 'b']);
  });
});
