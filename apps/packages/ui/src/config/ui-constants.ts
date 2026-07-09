/**
 * UI constants for the sidepanel chat composer and related components.
 * Centralizes magic numbers for easier maintenance.
 */

export const COMPOSER_CONSTANTS = {
  // Textarea sizing
  TEXTAREA_MAX_HEIGHT_PRO: 160,
  TEXTAREA_MAX_HEIGHT_CASUAL: 120,
  TEXTAREA_MIN_HEIGHT_PRO: 60,
  TEXTAREA_MIN_HEIGHT_CASUAL: 44,

  // Timing
  PLACEHOLDER_DEBOUNCE_MS: 400,
  DRAFT_SAVE_DEBOUNCE_MS: 500,
  DRAFT_SAVED_DISPLAY_MS: 4_000,
  PRIVATE_MODE_WARNING_DURATION: 6,
} as const

/**
 * Default values for STT (Speech-to-Text) settings.
 * These match the default values in useStorage calls.
 */
export const STT_DEFAULTS = {
  MODEL: "",
  TEMPERATURE: 0,
  TASK: "transcribe",
  RESPONSE_FORMAT: "json",
  TIMESTAMP_GRANULARITIES: "segment",
  PROMPT: "",

  // Segmentation parameters
  SEG_K: 6,
  SEG_MIN_SEGMENT_SIZE: 5,
  SEG_LAMBDA_BALANCE: 0.01,
  SEG_UTTERANCE_EXPANSION_WIDTH: 2,
  SEG_EMBEDDINGS_PROVIDER: "",
  SEG_EMBEDDINGS_MODEL: "",
} as const

/**
 * Storage keys used across the application.
 */
export const STORAGE_KEYS = {
  SIDEPANEL_CHAT_DRAFT: "tldw:sidepanelChatDraft",
} as const

/**
 * Spacing design tokens for consistent layout.
 * Uses Tailwind CSS class names for direct application.
 */
export const SPACING = {
  // Composer layout
  COMPOSER_PADDING: "px-3",
  COMPOSER_GAP_PRO: "gap-2",
  COMPOSER_GAP_CASUAL: "gap-2.5",
  CARD_PADDING: "p-3",

  // Control elements
  CONTROL_GAP: "gap-2",
  BUTTON_GAP: "gap-1.5",

  // Borders & rounding
  CARD_BORDER_RADIUS: "rounded-3xl",
  INPUT_BORDER_RADIUS: "rounded-2xl",
} as const

/**
 * Get composer gap class based on UI mode.
 */
export const getComposerGap = (isProMode: boolean): string =>
  isProMode ? SPACING.COMPOSER_GAP_PRO : SPACING.COMPOSER_GAP_CASUAL
