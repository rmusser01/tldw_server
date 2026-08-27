import { describe, expect, it } from 'vitest';

import { validateWebhookUrl } from '../webhook-url';

describe('validateWebhookUrl', () => {
  it.each([
    'https://receiver.example/hooks/events?tenant=42',
    'http://receiver.example:8080/hooks/events',
    'https://[2001:db8::1]/hooks/events',
  ])('accepts a syntactically safe server-policy candidate: %s', (value) => {
    expect(validateWebhookUrl(`  ${value}  `)).toEqual({ valid: true, value });
  });

  it.each([
    ['', /required/i],
    ['file:///tmp/callback', /absolute HTTP or HTTPS/i],
    ['https:receiver.example/hook', /absolute HTTP or HTTPS/i],
    ['https://receiver.example\\hook', /absolute HTTP or HTTPS/i],
    ['https://operator:secret@receiver.example/hook', /credentials or a fragment/i],
    ['https://receiver.example/hook#private', /credentials or a fragment/i],
    ['https://invalid_host.example/hook', /invalid hostname/i],
    ['https://receiver.example:0/hook', /invalid port/i],
  ])('rejects an unsafe destination before server policy evaluation: %s', (value, message) => {
    expect(validateWebhookUrl(value)).toEqual({
      valid: false,
      message: expect.stringMatching(message),
    });
  });

  it('enforces both character and UTF-8 byte limits', () => {
    const tooManyCharacters = `https://receiver.example/${'a'.repeat(2_048)}`;
    const tooManyUtf8Bytes = `https://receiver.example/${'é'.repeat(1_020)}`;

    expect(validateWebhookUrl(tooManyCharacters)).toEqual({
      valid: false,
      message: expect.stringMatching(/2,048 characters and UTF-8 bytes/i),
    });
    expect(tooManyUtf8Bytes.length).toBeLessThanOrEqual(2_048);
    expect(validateWebhookUrl(tooManyUtf8Bytes)).toEqual({
      valid: false,
      message: expect.stringMatching(/2,048 characters and UTF-8 bytes/i),
    });
  });
});
