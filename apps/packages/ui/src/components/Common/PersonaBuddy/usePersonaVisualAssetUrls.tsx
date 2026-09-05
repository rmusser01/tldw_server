import React from "react";

import { toAllowedPath } from "@/services/tldw/path-utils";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import type { PersonaVisualAsset } from "@/types/persona-visuals";

// Only server-owned relative paths use credential-bearing transport.
const isProtectedAsset = (url: string) =>
  /^\/api\/v1\/persona\/[^?#]+\/assets\/[^/]+\/content$/.test(url);

const MAX_CACHED_FRAMES = 8;
const MAX_CACHED_BYTES = 16 * 1024 * 1024;

export const usePersonaVisualAssetUrls = (
  assets: Record<string, PersonaVisualAsset>,
  activeAsset: PersonaVisualAsset | null,
) => {
  const key = JSON.stringify(
    Object.values(assets)
      .filter((asset) => isProtectedAsset(asset.url))
      .map((asset) => [asset.url, asset.mime_type] as const)
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  const cache = React.useRef(
    new Map<string, { url: string | null; size: number }>(),
  );
  const path =
    activeAsset && isProtectedAsset(activeAsset.url) ? activeAsset.url : null;
  const mimeType = activeAsset?.mime_type;
  const [loaded, setLoaded] = React.useState<{
    key: string;
    urls: Record<string, string | null>;
  }>({ key: "", urls: {} });

  // Pack lifetime owns URLs and failed-source tombstones; frame changes do not.
  React.useEffect(() => {
    const entries = cache.current;
    return () => {
      entries.forEach(({ url }) => {
        if (url) URL.revokeObjectURL(url);
      });
      entries.clear();
    };
  }, [key]);

  React.useEffect(() => {
    if (!path) return;
    const entries = cache.current;
    const publish = () =>
      setLoaded({
        key,
        urls: Object.fromEntries(
          [...entries].map(([source, entry]) => [source, entry.url]),
        ),
      });
    const cached = entries.get(path);
    if (cached) {
      // Refresh insertion order so recently displayed frames remain reusable.
      entries.delete(path);
      entries.set(path, cached);
      publish();
      return;
    }
    const controller = new AbortController();
    void (async () => {
      let url: string | null = null;
      let size = 0;
      try {
        const response = await tldwClient.fetchWithAuth(toAllowedPath(path), {
          method: "GET",
          responseType: "arrayBuffer",
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        if (!response.ok) throw new Error("Asset request failed");
        const data = response.data;
        const bytes =
          data instanceof ArrayBuffer
            ? new Uint8Array(data)
            : ArrayBuffer.isView(data)
              ? Uint8Array.from(
                  new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
                )
              : Array.isArray(data)
                ? Uint8Array.from(data)
                : null;
        if (!bytes) throw new Error("Asset response is not binary");
        size = bytes.byteLength;
        url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
      } catch {
        // A failed source remains failed until the pack's sources change.
      }
      if (controller.signal.aborted) return;
      entries.set(path, { url, size });
      // Bound retained frames by both count and bytes. A single larger current
      // frame is allowed, but all older blobs are evicted in that case.
      const blobs = [...entries].filter(([, entry]) => entry.url);
      let totalBytes = blobs.reduce(
        (total, [, entry]) => total + entry.size,
        0,
      );
      while (
        blobs.length > 1 &&
        (blobs.length > MAX_CACHED_FRAMES || totalBytes > MAX_CACHED_BYTES)
      ) {
        const [source, entry] = blobs.shift()!;
        URL.revokeObjectURL(entry.url!);
        entries.delete(source);
        totalBytes -= entry.size;
      }
      publish();
    })();
    return () => controller.abort();
  }, [key, path, mimeType]);

  return (asset: PersonaVisualAsset): string | null | undefined =>
    isProtectedAsset(asset.url)
      ? loaded.key === key
        ? loaded.urls[asset.url]
        : undefined
      : asset.url;
};

export const PersonaVisualAssetImage = ({
  asset,
  ...props
}: {
  asset: PersonaVisualAsset;
} & Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src">) => {
  const resolveUrl = usePersonaVisualAssetUrls({ [asset.id]: asset }, asset);
  const url = resolveUrl(asset);
  return url ? <img {...props} src={url} /> : null;
};
