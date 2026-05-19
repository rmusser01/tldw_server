import { describe, it, expect } from 'vitest';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

describe('correlation ID generation', () => {
  it('generateRequestId returns a valid UUID v4', async () => {
    const { generateRequestId } = await import('../correlation-id');
    const id = generateRequestId();
    expect(id).toMatch(UUID_REGEX);
  });

  it('generates unique IDs', async () => {
    const { generateRequestId } = await import('../correlation-id');
    const ids = new Set(Array.from({ length: 100 }, () => generateRequestId()));
    expect(ids.size).toBe(100);
  });
});
