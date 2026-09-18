import {vi} from "vitest";
import type {VisualIdentityCapabilitiesResponse} from "@/types/visual-identities";
const client={getVisualIdentityCapabilities:vi.fn(async (): Promise<VisualIdentityCapabilitiesResponse> => ({
    upload_max_bytes: 1024 * 1024,
    archive_max_bytes: 4 * 1024 * 1024,
    max_dimension: 2048,
    max_frame_count: 120,
    supported_mime_types: ["image/png", "image/webp"],
    avif_enabled: false
  }))};
client.getVisualIdentityCapabilities.mockResolvedValue({
      upload_max_bytes: 1024 * 1024,
      archive_max_bytes: 4 * 1024 * 1024,
      max_dimension: 2048,
      max_frame_count: 120,
      supported_mime_types: ["image/png", "image/webp"],
      avif_enabled: false,
      metadata_supported: false,
      metadata_unavailable_reason: "visual identity metadata is unavailable"
    });
client.getVisualIdentityCapabilities.mockResolvedValue({
      upload_max_bytes: 1024 * 1024,
      archive_max_bytes: 4 * 1024 * 1024,
      max_dimension: 2048,
      max_frame_count: 120,
      supported_mime_types: ["image/png", "image/webp"],
      avif_enabled: false,
      metadata_supported: true
    });
