import assert from 'node:assert/strict';

export const BIOLOGY_NOTE_CONTENT = [
  'The mitochondria is the powerhouse of the cell.',
  'DNA stands for deoxyribonucleic acid.',
  'Photosynthesis converts light energy into chemical energy.',
  'The human body has 206 bones.',
  'Water boils at 100 degrees Celsius at sea level.',
].join('\n\n');

// Deliberately bounded to F-BIOLOGY. Unknown paraphrases require inspection;
// keyword presence alone must never certify an unsupported or contradictory QA.
const supportedPairs: Array<{ fact: string; question: RegExp; answer: RegExp }> = [
  {
    fact: 'mitochondria',
    question: /^(?:what is|which organelle is(?: known as)?) the powerhouse of the cell$/,
    answer: /^(?:the )?mitochondri(?:a|on)(?: (?:is|are) the powerhouse of the cell)?$/,
  },
  {
    fact: 'mitochondria',
    question: /^what (?:is|are) (?:the )?mitochondri(?:a|on)(?: known as)?$/,
    answer: /^(?:the )?powerhouse of the cell$/,
  },
  {
    fact: 'dna',
    question:
      /^(?:what does dna stand for|what is dna short for|what is the full (?:form|name) of dna)$/,
    answer: /^(?:dna stands for )?deoxyribonucleic acid$/,
  },
  {
    fact: 'photosynthesis',
    question: /^what does photosynthesis convert$/,
    answer: /^(?:photosynthesis converts )?light energy into chemical energy$/,
  },
  {
    fact: 'photosynthesis',
    question: /^(?:what|which) process converts light energy into chemical energy$/,
    answer: /^photosynthesis$/,
  },
  {
    fact: 'bones',
    question:
      /^(?:how many bones does (?:the|an adult) human body have|how many bones are (?:there )?in (?:the|an adult) human body)$/,
    answer: /^(?:(?:the|an adult) human body has )?206(?: bones)?$/,
  },
  {
    fact: 'water',
    question:
      /^(?:at what temperature does water boil|what is the boiling (?:point|temperature) of water) at sea level$/,
    answer: /^(?:water boils at )?100 degrees celsius(?: at sea level)?$/,
  },
];

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/100\s*°\s*c\b/g, '100 degrees celsius')
    .replace(/[*"“”]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[.,?!:;]+$/, '')
    .trim();
}

/** Assert five distinct source-supported facts and return their identities. */
export function assertBiologyCardSet(cards: { front: string; back: string }[]): string[] {
  assert.equal(cards.length, 5, 'Expected exactly five F-BIOLOGY cards');
  const facts = cards.map((card, index) => {
    const pair = supportedPairs.find(
      ({ question, answer }) =>
        question.test(normalize(card.front)) && answer.test(normalize(card.back))
    );
    assert.ok(pair, `Unsupported F-BIOLOGY card ${index + 1}: ${JSON.stringify(card)}`);
    return pair.fact;
  });
  assert.equal(new Set(facts).size, 5, 'Expected five distinct source facts');
  return facts;
}
