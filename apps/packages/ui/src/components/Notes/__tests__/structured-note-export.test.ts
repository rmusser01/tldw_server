import { describe, expect, it } from 'vitest';
import {
  buildPortableNote,
  buildStructuredSource,
  type SingleNoteExportData,
} from '../export-utils';

const source = '\ufeff  {"x":9007199254740993,"huge":1e400,"html":"<>&"}\r\n ';

describe('structured note export contracts', () => {
  it('distinguishes exact source from a portable note', () => {
    const note = { title: 'Data', content: source, keywords: ['data'], content_language: 'json' };
    expect(buildStructuredSource(note)).toEqual({ filename: 'Data.json', text: source });
    const portable = JSON.parse(buildPortableNote(note));
    expect(portable.content).toBe(source);
    expect(portable.content_language).toBe('json');
    expect(portable.keywords).toEqual(['data']);
  });

  it.each([
    ['yaml', 'yaml'],
    ['jsonl', 'jsonl'],
    ['markdown', 'md'],
    ['text', 'txt'],
    [null, 'txt'],
    [' future/😀 ', 'txt'],
  ])('preserves opaque mode %s and uses only supported source extensions', (mode, extension) => {
    const note = {
      id: 'n-1',
      title: 'Data',
      content: source,
      keywords: [],
      content_language: mode,
    };
    expect(buildStructuredSource(note)).toEqual({ filename: `Data.${extension}`, text: source });
    const portable = JSON.parse(buildPortableNote(note));
    expect(portable.content_language).toBe(mode);
    expect(portable.id).toBe('n-1');
  });

  it('does not claim portable metadata for an unknown baseline', () => {
    expect(() => buildPortableNote({ title: 'Data', content: source, keywords: [] })).toThrow();
  });

  it('refuses a preview row as a complete portable note', () => {
    const preview = {
      title: 'Data',
      content_preview: 'truncated…',
      keywords: [],
      content_language: 'json',
    };
    expect(() => buildPortableNote(preview as unknown as SingleNoteExportData)).toThrow(
      /complete source/i
    );
  });

  it('retains empty source for download', () => {
    expect(
      buildStructuredSource({
        title: 'Empty',
        content: '',
        keywords: [],
        content_language: 'jsonl',
      })
    ).toEqual({ filename: 'Empty.jsonl', text: '' });
  });
});
