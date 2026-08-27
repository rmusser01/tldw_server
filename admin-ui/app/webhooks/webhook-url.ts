const MAX_WEBHOOK_URL_LENGTH = 2_048;
const INVALID_URL_CHARACTERS = /[\u0000-\u001f\u007f\\]/u;
const DNS_LABEL = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/iu;

export type WebhookUrlValidation =
  | { valid: true; value: string }
  | { valid: false; message: string };

/** Validate destination invariants known to the browser without duplicating server egress policy. */
export const validateWebhookUrl = (rawValue: string): WebhookUrlValidation => {
  const value = rawValue.trim();
  if (!value) {
    return { valid: false, message: 'Destination URL is required.' };
  }
  if (
    value.length > MAX_WEBHOOK_URL_LENGTH
    || new TextEncoder().encode(value).byteLength > MAX_WEBHOOK_URL_LENGTH
  ) {
    return {
      valid: false,
      message: 'Destination URL must be no more than 2,048 characters and UTF-8 bytes.',
    };
  }
  if (INVALID_URL_CHARACTERS.test(value) || !/^https?:\/\//iu.test(value)) {
    return { valid: false, message: 'Destination must be an absolute HTTP or HTTPS URL.' };
  }

  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    return { valid: false, message: 'Destination must be an absolute HTTP or HTTPS URL.' };
  }
  if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname) {
    return { valid: false, message: 'Destination must be an absolute HTTP or HTTPS URL.' };
  }
  if (parsed.username || parsed.password || parsed.hash) {
    return {
      valid: false,
      message: 'Destination URL must not include credentials or a fragment.',
    };
  }
  if (parsed.port && (Number(parsed.port) < 1 || Number(parsed.port) > 65_535)) {
    return { valid: false, message: 'Destination URL has an invalid port.' };
  }

  const hostname = parsed.hostname.replace(/^\[|\]$/gu, '');
  const isIpv6 = hostname.includes(':');
  const normalizedDomain = hostname.replace(/\.$/u, '');
  if (
    hostname.includes('%')
    || (!isIpv6 && (
      normalizedDomain.length > 253
      || normalizedDomain.split('.').some((label) => !DNS_LABEL.test(label))
    ))
  ) {
    return { valid: false, message: 'Destination URL has an invalid hostname.' };
  }

  return { valid: true, value };
};
