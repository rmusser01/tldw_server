type FormatDateOptions = {
  fallback?: string;
  locale?: string;
  options?: Intl.DateTimeFormatOptions;
};

export const formatDateTime = (
  value?: string | null,
  { fallback = '-', locale, options }: FormatDateOptions = {}
): string => {
  if (!value) return fallback;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(locale, options);
};

export const formatDate = (
  value?: string | null,
  { fallback = '-', locale }: FormatDateOptions = {}
): string => {
  if (!value) return fallback;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(locale);
};

type FormatNumberOptions = {
  fallback?: string;
  precision?: number;
};

type FormatDurationOptions = {
  fallback?: string;
};

export const formatBytes = (
  value?: number | null,
  { fallback = '-', precision = 1 }: FormatNumberOptions = {}
): string => {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return fallback;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(precision)} ${units[idx]}`;
};

export const formatDuration = (
  value?: number | null,
  { fallback = '-' }: FormatDurationOptions = {}
): string => {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return fallback;
  const totalSeconds = Math.floor(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
};

export const formatLatency = (
  value?: number | null,
  { fallback = '-', precision = 1 }: FormatNumberOptions = {}
): string => {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  return `${value.toFixed(precision)} ms`;
};

export function formatTokens(
  value?: number | null,
  { fallback = '—' }: { fallback?: string } = {}
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}
