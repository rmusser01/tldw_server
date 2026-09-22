import { describe, expect, it } from 'vitest';
import {
  assertCitedRowanAnswer,
  assertFullSourceHandoff,
  assertPriceAbstention,
  assertSavedTurn,
  ROWAN_SOURCE,
  readMediaRecord,
  readSavedMessages,
} from '../uat390-grounding';

const answer =
  'Dr. Mira Vale directs Rowan Observatory in Cedar Ridge. Tours begin Friday at 18:00. [1]';
const sources = [
  {
    source_id: '812',
    chunk_id: 'chunk-4',
    excerpt: 'Its director is Dr. Mira Vale. It is in Cedar Ridge. Tours begin Friday at 18:00.',
  },
];
const check = (text = answer, evidence = sources) =>
  assertCitedRowanAnswer(text, evidence, '812', '913');

describe('UAT390 acceptance discriminates actual source grounding', () => {
  it('accepts supported facts and the canonical source citation', () =>
    expect(() => check()).not.toThrow());
  it('rejects an unrelated result even with the same answer', () =>
    expect(() => check(answer, [{ ...sources[0], source_id: '913' }])).toThrow(/wrong source/));
  it('rejects a chunk ID substituted for the canonical source ID', () =>
    expect(() => check(answer, [{ ...sources[0], source_id: 'chunk-4' }])).toThrow(/wrong source/));
  it('rejects a generic answer containing only the source name', () =>
    expect(() => check('Rowan Observatory is an observatory. [1]')).toThrow(/director/));
  it('rejects correct facts without citations', () =>
    expect(() => check(answer.replace(' [1]', ''))).toThrow(/no inspectable citation/));
  it('rejects a stale out-of-range citation', () =>
    expect(() => check(answer.replace('[1]', '[2]'))).toThrow(/no source/));
  it('rejects evidence without the asserted facts', () =>
    expect(() => check(answer, [{ ...sources[0], excerpt: 'Rowan Observatory' }])).toThrow(
      /director/
    ));
  it('rejects distractor facts appended to an otherwise correct answer', () =>
    expect(() => check(`${answer} Tours also run Tuesdays at 09:00.`)).toThrow(/Distractor/));
  it('rejects an ID-only Media-to-Chat placeholder', () =>
    expect(() => assertFullSourceHandoff("Let's talk about media 812.", ROWAN_SOURCE)).toThrow(
      /complete source/
    ));
  it('accepts a full source plus a question', () =>
    expect(() =>
      assertFullSourceHandoff(
        `Chat with this media: Rowan\n\n${ROWAN_SOURCE}\nQuestion?`,
        ROWAN_SOURCE
      )
    ).not.toThrow());
  it('accepts an explicit source limitation', () =>
    expect(() =>
      assertPriceAbstention('The source does not provide a ticket price.')
    ).not.toThrow());
  it('rejects a hedged invented price', () =>
    expect(() =>
      assertPriceAbstention('The source does not provide a price, but tickets cost $12.')
    ).toThrow(/Unsupported/));
  it('rejects a generic answer to the price question', () =>
    expect(() => assertPriceAbstention('You can visit Rowan Observatory.')).toThrow(/limitation/));
});

describe('UAT390 saved turn identity', () => {
  const messages = [
    { id: 'user-1', role: 'user', content: 'Question?' },
    { id: 'answer-1', role: 'assistant', content: answer },
  ];
  it('accepts exact canonical messages', () =>
    expect(() => assertSavedTurn(messages, 'Question?', answer)).not.toThrow());
  it.each([
    ['duplicate answer', [...messages, messages[1]]],
    ['changed answer', [messages[0], { ...messages[1], content: 'Generic answer' }]],
    ['missing source prompt', [{ ...messages[0], content: 'Different question' }, messages[1]]],
    ['reversed order', [...messages].reverse()],
    ['missing ID', [messages[0], { ...messages[1], id: '' }]],
    ['colliding IDs', [messages[0], { ...messages[1], id: 'user-1' }]],
  ])('rejects %s', (_name, altered) =>
    expect(() => assertSavedTurn(altered, 'Question?', answer)).toThrow()
  );
});

describe('UAT390 canonical API response boundaries', () => {
  const media = {
    media_id: 812,
    source: { title: 'uat390-run-F-SOURCE.txt' },
    content: { text: 'Exact source text' },
  };
  it('reads MediaDetailResponse.media_id/source.title/content.text', () => {
    expect(readMediaRecord(media, '812', 'Exact source text', 'uat390-run')).toEqual({
      id: '812',
      title: 'uat390-run-F-SOURCE.txt',
      text: 'Exact source text',
    });
  });
  it('rejects guessed flat media/job response fields', () => {
    expect(() =>
      readMediaRecord(
        { id: 812, title: media.source.title, content: 'Exact source text' },
        '812',
        'Exact source text',
        'uat390-run'
      )
    ).toThrow(/media ID/);
  });
  it('rejects another canonical ID even when its content matches', () => {
    expect(() =>
      readMediaRecord({ ...media, media_id: 913 }, '812', 'Exact source text', 'uat390-run')
    ).toThrow(/media ID/);
  });
  it('rejects a same-ID record from another namespace', () => {
    expect(() =>
      readMediaRecord(
        { ...media, source: { title: 'other-run' } },
        '812',
        'Exact source text',
        'uat390-run'
      )
    ).toThrow(/another run/);
  });
  it('rejects wrong content from the same canonical ID', () => {
    expect(() =>
      readMediaRecord(
        { ...media, content: { text: 'Different source' } },
        '812',
        'Exact source text',
        'uat390-run'
      )
    ).toThrow(/Wrong saved source/);
  });
  const messages = [{ id: 'm1', conversation_id: 'c1', sender: 'assistant', content: 'Answer' }];
  it('reads canonical sender and conversation identity', () => {
    expect(readSavedMessages(messages, 'c1')).toEqual([{ ...messages[0], role: 'assistant' }]);
  });
  it('rejects messages from another conversation', () => {
    expect(() => readSavedMessages(messages, 'c2')).toThrow(/another conversation/);
  });
  it('does not accept a list-response envelope as the message array', () => {
    expect(() => readSavedMessages({ messages }, 'c1')).toThrow(/array/);
  });
  it('rejects the UI role field substituted for canonical sender', () => {
    expect(() =>
      readSavedMessages([{ ...messages[0], sender: undefined, role: 'assistant' }], 'c1')
    ).toThrow(/sender/);
  });
});

it('rejects a hedged claim of free admission', () => {
  expect(() =>
    assertPriceAbstention('The source does not provide a price, but admission is free.')
  ).toThrow(/Unsupported/);
});

it.each([
  'Tickets cost ten dollars.',
  'Admission costs twenty-five euros.',
  'Tickets cost one hundred pounds.',
  'The ticket price is ten.',
  'Tickets cost twenty.',
  'Tours cost five.',
  'Admission is free.',
  'Tickets are complimentary.',
  'Complimentary tours are available.',
])('rejects an invented price after abstention: %s', (claim) => {
  expect(() =>
    assertPriceAbstention(`The source does not provide a ticket price. ${claim}`)
  ).toThrow(/Unsupported/);
});

it.each([
  'The observatory opened in 2019; tours begin Friday at 18:00.',
  'The tour reference code is ORBIT-742. [1]',
  'One source gives two facts about the tour schedule.',
  'Tours require one reservation before arrival.',
])('keeps unrelated numbers outside the price oracle: %s', (context) => {
  expect(() =>
    assertPriceAbstention(`The source does not provide a ticket price. ${context}`)
  ).not.toThrow();
});
