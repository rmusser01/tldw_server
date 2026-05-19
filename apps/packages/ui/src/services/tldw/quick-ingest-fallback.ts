type UploadLikeError = {
  status?: number;
  error?: unknown;
  details?: unknown;
};

export const extractQuickIngestErrorText = (error: unknown): string => {
  const parts: string[] = [];

  if (error instanceof Error && error.message) {
    parts.push(error.message);
  } else if (typeof error === "string" && error.trim()) {
    parts.push(error);
  }

  if (error && typeof error === "object") {
    const record = error as UploadLikeError;
    if (typeof record.error === "string" && record.error.trim()) {
      parts.push(record.error);
    }
    const details = record.details;
    if (typeof details === "string" && details.trim()) {
      parts.push(details);
    } else if (
      details &&
      typeof details === "object" &&
      !Array.isArray(details)
    ) {
      for (const key of ["detail", "message", "error"]) {
        const value = (details as Record<string, unknown>)[key];
        if (typeof value === "string" && value.trim()) {
          parts.push(value);
        }
      }
    }
  }

  return parts.join(" ").trim();
};

export const shouldFallbackToPersistentAdd = (error: unknown): boolean => {
  const status = (error as UploadLikeError | null)?.status;
  if (status !== 429) return false;
  const normalized = extractQuickIngestErrorText(error).toLowerCase();
  return /concurrent job limit|max(?:imum)? concurrent|queue is full|queue full/.test(
    normalized,
  );
};

export const normalizePersistentAddResponse = <T>(data: T): T => {
  if (!data || typeof data !== "object" || Array.isArray(data)) return data;
  const results = (data as { results?: unknown }).results;
  if (!Array.isArray(results)) return data;
  return {
    ...(data as Record<string, unknown>),
    results: results.map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return item;
      const record = item as Record<string, unknown>;
      if (record.media_id != null || record.db_id == null) return item;
      return {
        ...record,
        media_id: record.db_id,
      };
    }),
  } as T;
};
