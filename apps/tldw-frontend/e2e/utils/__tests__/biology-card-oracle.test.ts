import { describe, expect, it } from 'vitest';
import { assertBiologyCardSet } from '../biology-card-oracle';

const cards = [
  { front: 'What is the powerhouse of the cell?', back: 'The mitochondria.' },
  { front: 'What does DNA stand for?', back: 'Deoxyribonucleic acid.' },
  { front: 'What does photosynthesis convert?', back: 'Light energy into chemical energy.' },
  { front: 'How many bones does the human body have?', back: '206 bones.' },
  { front: 'At what temperature does water boil at sea level?', back: '100 degrees Celsius.' },
];

describe('F-BIOLOGY card acceptance oracle', () => {
  it.each(['What does photosynthesis convert?', 'What does photosynthesis convert light energy into?'])('accepts inspected native source sentences: %s', photosynthesisQuestion => {
    expect(assertBiologyCardSet([
      { front: 'What is the mitochondria?', back: 'The mitochondria is the powerhouse of the cell.' },
      { front: 'What does DNA stand for?', back: 'DNA stands for deoxyribonucleic acid.' },
      { front: photosynthesisQuestion, back: 'Photosynthesis converts light energy into chemical energy.' },
      { front: 'How many bones does the human body have?', back: 'The human body has 206 bones.' },
      { front: 'At what temperature does water boil at sea level?', back: 'Water boils at 100 degrees Celsius at sea level.' },
    ])).toEqual(['mitochondria', 'dna', 'photosynthesis', 'bones', 'water'])
  })

  it('accepts exactly one supported question and answer per source fact', () => {
    expect(assertBiologyCardSet(cards)).toEqual([
      'mitochondria',
      'dna',
      'photosynthesis',
      'bones',
      'water',
    ]);
  });

  it.each([0, 1, 4, 6])('rejects a %i-card partial or excess batch', (count) => {
    expect(() => assertBiologyCardSet([...cards, cards[0]].slice(0, count))).toThrow(
      /exactly five/
    );
  });

  it('rejects a paraphrased duplicate that replaces a missing fact', () => {
    expect(() =>
      assertBiologyCardSet([
        ...cards.slice(0, 4),
        { front: 'What is DNA short for?', back: 'DNA stands for deoxyribonucleic acid.' },
      ])
    ).toThrow(/distinct source facts/);
  });

  it.each([
    { front: cards[3].front, back: '207 bones.' },
    { front: cards[3].front, back: '20.6 bones.' },
    { front: cards[3].front, back: '2?06 bones.' },
    { front: cards[3].front, back: '206 bones, including 100 skull bones.' },
    { front: cards[3].front, back: 'The human body does not have 206 bones.' },
    { front: cards[3].front, back: cards[1].back },
    { front: 'How many bones does a cat have?', back: '206 bones.' },
    { front: 'Does the human body have 207 bones?', back: '206 bones.' },
    { front: '', back: '206 bones.' },
  ])('rejects unsupported or mismatched content: $front / $back', (invalid) => {
    expect(() =>
      assertBiologyCardSet(cards.map((card, index) => (index === 3 ? invalid : card)))
    ).toThrow(/Unsupported/);
  });

  it('accepts harmless punctuation, full source sentences and reversed questions', () => {
    expect(
      assertBiologyCardSet([
        {
          front: 'Which organelle is known as the powerhouse of the cell?',
          back: 'The mitochondria is the powerhouse of the cell.',
        },
        { front: 'What does DNA stand for?', back: 'DNA stands for deoxyribonucleic acid.' },
        {
          front: 'Which process converts light energy into chemical energy?',
          back: 'Photosynthesis.',
        },
        { front: 'How many bones are in the human body?', back: 'The human body has 206 bones.' },
        {
          front: 'What is the boiling point of water at sea level?',
          back: 'Water boils at 100°C at sea level.',
        },
      ])
    ).toHaveLength(5);
  });
});
