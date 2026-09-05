import React from "react";

import { toAllowedPath } from "@/services/tldw/path-utils";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import type { PersonaVisualAsset } from "@/types/persona-visuals";

// Only server-owned relative paths use credential-bearing transport.
const isProtectedAsset = (url: string) =>
  /^\/api\/v1\/persona\/[^?#]+\/assets\/[^/]+\/content$/.test(url);

export const usePersonaVisualAssetUrls = (
  assets: Record<string, PersonaVisualAsset>,
) => {
  const key = JSON.stringify(
    Object.values(assets)
      .filter((asset) => isProtectedAsset(asset.url))
      .map((asset) => [asset.url, asset.mime_type] as const)
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  const [loaded, setLoaded] = React.useState<{
    key: string;
    urls: Record<string, string | null>;
  }>({ key: "", urls: {} });

  React.useEffect(() => {
    const controller = new AbortController();
    const objectUrls: string[] = [];
    const entries = JSON.parse(key) as [string, string][];
    for (const [path, mimeType] of new Map(entries)) {
      void (async () => {
        let url: string | null = null;
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
                    new Uint8Array(
                      data.buffer,
                      data.byteOffset,
                      data.byteLength,
                    ),
                  )
                : Array.isArray(data)
                  ? Uint8Array.from(data)
                  : null;
          if (!bytes) throw new Error("Asset response is not binary");
          url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
          objectUrls.push(url);
        } catch {
          // A failed source remains failed until the pack's sources change.
        }
        if (controller.signal.aborted) return;
        setLoaded((previous) => ({
          key,
          urls: { ...(previous.key === key ? previous.urls : {}), [path]: url },
        }));
      })();
    }
    return () => {
      controller.abort();
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [key]);

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
  const resolveUrl = usePersonaVisualAssetUrls({ [asset.id]: asset });
  const url = resolveUrl(asset);
  return url ? <img {...props} src={url} /> : null;
};
