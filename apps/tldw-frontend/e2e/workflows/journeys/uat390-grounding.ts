/** Frozen public fixtures and acceptance oracles for UAT390; no application mocks. */
import assert from 'node:assert/strict';

export const ROWAN_SOURCE = `Rowan Observatory

Rowan Observatory opened in 2019 in Cedar Ridge. Its director is Dr. Mira Vale.
The observatory's main telescope is named Selene.

Public tours begin every Friday at 18:00. Visitors must reserve a place before
arriving. The tour reference code is ORBIT-742.

The observatory studies variable stars and shares a monthly public report.
The source provides no ticket price and no current weather forecast.
`;
export const LARCH_SOURCE =
  'Larch Observatory opened in 2021. Its director is Dr. Tomas Reed. It is in Pine Hollow. Public tours begin on Tuesdays at 09:00.';
export const ROWAN_QUESTION = 'Who directs Rowan Observatory, where is it, and when are tours?';
export const PRICE_QUESTION =
  'What is the ticket price at Rowan Observatory? Use only the source; state when it does not provide the answer.';

export type EvidenceSource = { source_id: string; excerpt: string; chunk_id?: string };
export type SavedMessage = { id: string; role: string; content: string };

// TXT extraction may normalize line endings and surrounding whitespace only.
export const normalizeSourceText = (text: string): string => text.replace(/\r\n/g, '\n').trim();

export function assertRowanFacts(answer: string): void {
  assert.match(answer, /\bMira Vale\b/i, 'Rowan director is missing');
  assert.match(answer, /\bCedar Ridge\b/i, 'Rowan location is missing');
  assert.match(answer, /\bFridays?\b/i, 'Rowan tour day is missing');
  assert.match(answer, /\b18:00\b|\b6(?::00)?\s*p\.?m\.?\b/i, 'Rowan tour time is missing');
  assert.doesNotMatch(
    answer,
    /Tomas Reed|Pine Hollow|Tuesdays?|09:00/i,
    "Distractor facts leaked into Rowan's answer"
  );
}

export function assertCitedRowanAnswer(
  answer: string,
  sources: EvidenceSource[],
  sourceId: string,
  distractorId: string
): void {
  assertRowanFacts(answer);
  assert.notEqual(sourceId, distractorId, 'Source and distractor must be distinct');
  const citations = [...answer.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]));
  assert.ok(citations.length > 0, 'Answer has no inspectable citation');
  for (const citation of citations) {
    const source = sources[citation - 1];
    assert.ok(source, `Citation ${citation} has no source`);
    assert.equal(source.source_id, sourceId, `Citation ${citation} targets the wrong source`);
  }
  assertRowanFacts([...new Set(citations)].map((index) => sources[index - 1].excerpt).join('\n'));
}

export function assertFullSourceHandoff(draft: string, sourceText: string): void {
  assert.ok(
    normalizeSourceText(draft).includes(normalizeSourceText(sourceText)),
    'Handoff lost the complete source text'
  );
  assert.ok(!draft.includes(LARCH_SOURCE), 'Handoff contains the distractor');
}

export function assertPriceAbstention(answer: string): void {
  assert.match(
    answer,
    /not (?:provided|specified|stated|mentioned|given|available)|does(?:n't| not) (?:provide|specify|state|mention|give)|no (?:ticket )?price|cannot (?:determine|answer)/i,
    'Missing explicit source limitation'
  );
  assert.doesNotMatch(
    answer,
    /[$€£]\s*\d|\d+(?:\.\d+)?\s*(?:dollars?|euros?|pounds?)|(?:ticket price|admission|tickets?|tours?)\s+(?:is|are|costs?)\s+(?:\d|zero|free)|free\s+(?:admission|entry|tickets?|tours?)/i,
    'Unsupported ticket price'
  );
}

export function assertSavedTurn(messages: SavedMessage[], prompt: string, answer: string): void {
  const turns = messages.filter((message) => message.role !== 'system');
  assert.deepEqual(
    turns.map(({ role, content }) => ({ role, content })),
    [
      { role: 'user', content: prompt },
      { role: 'assistant', content: answer },
    ],
    'Saved turn lost, changed, reordered, or duplicated content'
  );
  assert.ok(
    turns.every((message) => Boolean(message.id)),
    'Saved message ID is missing'
  );
  assert.equal(
    new Set(turns.map((message) => message.id)).size,
    turns.length,
    'Saved message IDs collide'
  );
}

/** Decode the documented MediaDetailResponse, never a similarly shaped job. */
export function readMediaRecord(
  payload: unknown,
  expectedId: string,
  text: string,
  namespace: string
) {
  const record = payload as {
    media_id?: number;
    source?: { title?: string };
    content?: { text?: string };
  };
  assert.equal(
    String(record?.media_id),
    expectedId,
    'Canonical media ID differs from the UI target'
  );
  assert.equal(typeof record.source?.title, 'string', 'Missing canonical source title');
  assert.ok(record.source!.title!.includes(namespace), 'Source belongs to another run');
  assert.equal(typeof record.content?.text, 'string', 'Missing extracted content.text');
  assert.equal(
    normalizeSourceText(record.content!.text!),
    normalizeSourceText(text),
    'Wrong saved source text'
  );
  return { id: expectedId, title: record.source!.title!, text: record.content!.text! };
}

/** The messages endpoints persist sender, not the UI's derived role field. */
export function readSavedMessages(
  payload: unknown,
  conversationId: string
): Array<
  SavedMessage & {
    rag_context?: {
      retrieved_documents: EvidenceSource[];
      generated_answer: string;
      trust_state?: string;
    };
  }
> {
  assert.ok(Array.isArray(payload), 'Canonical messages must be an array');
  return payload.map((message) => {
    assert.equal(
      message.conversation_id,
      conversationId,
      'Message belongs to another conversation'
    );
    assert.equal(typeof message.id, 'string', 'Canonical message ID is missing');
    assert.ok(message.id.length > 0, 'Canonical message ID is empty');
    assert.equal(typeof message.content, 'string', 'Canonical message content is missing');
    const role = String(message.sender ?? '').toLowerCase();
    assert.ok(
      ['user', 'assistant', 'system'].includes(role),
      'Unexpected canonical sender for an ordinary Chat'
    );
    return { ...message, role };
  });
}
