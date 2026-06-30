const RUNTIME_OVERLAY_CONSOLE_PATTERNS = [
  /Runtime(?:\s+\w+)?\s+Error/i,
  /Runtime SyntaxError/i,
  /Invalid or unexpected token/i,
  /Objects are not valid as a React child/i,
  /message\.error is not a function/i,
];

const RUNTIME_OVERLAY_BODY_PATTERNS = [
  /Unhandled Runtime(?:\s+\w+)?\s+Error/i,
  /Runtime SyntaxError/i,
  /Build Error/i,
  /Application error/i,
  /Invalid or unexpected token/i,
  /Objects are not valid as a React child/i,
  /message\.error is not a function/i,
];

const TRANSIENT_RUNTIME_OVERLAY_PATTERNS = [
  /Runtime SyntaxError/i,
  /Invalid or unexpected token/i,
  /Unexpected end of input/i,
];

export function hasRuntimeOverlayConsoleSignal(input: string): boolean {
  return RUNTIME_OVERLAY_CONSOLE_PATTERNS.some((pattern) => pattern.test(input));
}

export function hasRuntimeOverlayBodySignal(input: string): boolean {
  return RUNTIME_OVERLAY_BODY_PATTERNS.some((pattern) => pattern.test(input));
}

export function hasTransientRuntimeOverlaySignal(input: string): boolean {
  return TRANSIENT_RUNTIME_OVERLAY_PATTERNS.some((pattern) => pattern.test(input));
}
