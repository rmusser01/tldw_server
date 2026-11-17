import {
  tldwClient,
  type ScopedRequestOptions
} from "@/services/tldw/TldwApiClient"
import type { MediaDocumentVersion } from "@/services/tldw/domains/media"
import type { WorkspaceSourceApiResponse } from "@/services/tldw/domains/workspace-api"
import type {
  WebCaptureDescriptor,
  WebClipperSaveRequest
} from "@/services/web-clipper/types"
import { sha256Text } from "@/store/workspace-migration"
import type { WebArticleCapturePin, WorkspaceSource } from "@/types/workspace"

const uuidPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
// Python str.strip whitespace: JS trim additionally removes BOM and misses NEL/1c..1f.
const trimCaptureText = (text: string): string =>
  text.replace(
    // eslint-disable-next-line no-control-regex -- Python str.strip includes these control code points.
    /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g,
    ""
  )
export class WebCaptureNotCurrentError extends Error {
  constructor(
    message: string,
    public sourceId?: string,
    public captureEvidenceKnown = false
  ) {
    super(message)
  }
}

const fail = (): never => {
  throw new WebCaptureNotCurrentError(
    "Web capture is unconfirmed or no longer current. Retry or recapture before Ask."
  )
}

function validateUrl(url: string): void {
  if (
    !url ||
    [...url].length > 4096 ||
    // eslint-disable-next-line no-control-regex -- URL trust boundary explicitly rejects controls.
    /[\u0000-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\\]/u.test(
      url
    )
  )
    throw new Error(
      "Capture requires a public HTTP(S) URL without credentials."
    )
  const parsed = new URL(url)
  const host = parsed.hostname
    .toLowerCase()
    .replace(/^\[|\]$/g, "")
    .replace(/\.$/, "")
  if (
    !/^https?:$/.test(parsed.protocol) ||
    /@/.test(url.split(/[/?#]/).slice(2, 3).join("")) ||
    parsed.username ||
    parsed.password ||
    host === "localhost" ||
    /\.(localhost|local)$/.test(host) ||
    !/[.:]/.test(host)
  )
    throw new Error(
      "Capture requires a public HTTP(S) URL without credentials."
    )
  // Match the backend's Python 3.12 public-literal classification; acquisition checks DNS.
  if (isNonPublicLiteral(host))
    throw new Error("Capture URL must not target a private address.")
}
function isNonPublicLiteral(host: string): boolean {
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) {
    const [a, b, c, d] = host.split(".").map(Number)
    return (
      a === 0 ||
      a === 10 ||
      a === 127 ||
      a >= 240 ||
      (a === 169 && b === 254) ||
      (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 &&
        (b === 168 ||
          (b === 0 && c === 0 && d !== 9 && d !== 10) ||
          (b === 0 && c === 2))) ||
      (a === 100 && b >= 64 && b <= 127) ||
      (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100))) ||
      (a === 203 && b === 0 && c === 113)
    )
  }
  if (!host.includes(":")) return false
  if (host.startsWith("::ffff:")) {
    const words = host
      .slice(7)
      .split(":")
      .map((word) => parseInt(word, 16))
    return isNonPublicLiteral(
      `${words[0] >> 8}.${words[0] & 255}.${words[1] >> 8}.${words[1] & 255}`
    )
  }
  const [first, second] = host
    .split(":")
    .map((word) => parseInt(word || "0", 16))
  if (first === 0x2001 && second < 0x200) {
    return !(
      ["2001:1::1", "2001:1::2"].includes(host) ||
      second === 3 ||
      host.startsWith("2001:4:112:") ||
      (second >= 0x20 && second <= 0x3f)
    )
  }
  return (
    host === "::" ||
    host === "::1" ||
    host.startsWith("64:ff9b:1:") ||
    (first === 0x100 && host.startsWith("100::")) ||
    (first === 0x2001 && second === 0xdb8) ||
    first === 0x2002 ||
    (first === 0x3fff && second < 0x1000) ||
    (first & 0xfe00) === 0xfc00 ||
    (first & 0xffc0) === 0xfe80
  )
}
function validateTime(value: string): void {
  if (
    value.length > 128 ||
    value.startsWith("0000-") ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$/.test(value)
  )
    throw new Error("Capture time must be ISO UTC.")
  const date = new Date(value)
  if (
    !Number.isFinite(date.getTime()) ||
    date.toISOString().slice(0, 19) !== value.slice(0, 19)
  )
    throw new Error("Capture time must be valid ISO UTC.")
}
function descriptorOf(
  body: Pick<WebClipperSaveRequest, "source_url" | "capture_metadata">,
): WebCaptureDescriptor {
  const descriptor = body.capture_metadata
    ?.web_capture_v1 as WebCaptureDescriptor;
  if (
    !descriptor ||
    Object.keys(descriptor).sort().join(",") !==
      "captured_at,content_sha256,mode,refresh_of,requested_url" ||
    descriptor.mode !== "server_article" ||
    descriptor.requested_url !== body.source_url ||
    !/^[0-9a-f]{64}$/.test(descriptor.content_sha256) ||
    (descriptor.refresh_of !== null && !uuidPattern.test(descriptor.refresh_of))
  )
    fail();
  validateUrl(descriptor.requested_url);
  validateTime(descriptor.captured_at);
  return { ...descriptor };
}
function assertScope(options: ScopedRequestOptions): void {
  options.signal?.throwIfAborted()
  if (!options.requestScope)
    throw new Error(
      "An owned request scope is required for capture confirmation."
    )
}

export async function prepareWebCaptureAcceptance(input: {
  url: string
  title: string
  text: string
  capturedAt: string
  workspaceId: string
  refreshOf?: string | null
}): Promise<WebClipperSaveRequest> {
  input = { ...input } // Snapshot primitive preview inputs before asynchronous hashing.
  validateUrl(input.url)
  validateTime(input.capturedAt)
  if (
    !input.workspaceId ||
    [...input.workspaceId].length > 128 ||
    (input.refreshOf != null && !uuidPattern.test(input.refreshOf))
  )
    throw new Error("Invalid capture destination or refresh identity.")
  const text = trimCaptureText(input.text)
  if (
    !text ||
    [...text].length > 1_000_000 ||
    /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/u.test(
      text
    )
  )
    throw new Error("Capture text is empty, too large or invalid Unicode.")
  if (!input.title || [...input.title].length > 512)
    throw new Error("Capture title is too long.")
  const descriptor: WebCaptureDescriptor = Object.freeze({
    mode: "server_article",
    requested_url: input.url,
    captured_at: input.capturedAt,
    content_sha256: await sha256Text(text),
    refresh_of: input.refreshOf ?? null
  })
  return Object.freeze({
    clip_id: crypto.randomUUID(),
    clip_type: "article",
    source_url: input.url,
    source_title: input.title,
    destination_mode: "workspace",
    workspace: Object.freeze({
      workspace_id: input.workspaceId,
      default_review_state: "needs_review"
    }),
    content: Object.freeze({ full_extract: text }),
    enhancements: Object.freeze({ run_ocr: false, run_vlm: false }),
    capture_metadata: Object.freeze({ web_capture_v1: descriptor })
  })
}
function verifySource(
  source: WorkspaceSourceApiResponse | undefined,
  clipId: string,
  workspaceId: string,
  url: string,
  mediaId?: number
): asserts source is WorkspaceSourceApiResponse {
  if (
    !source ||
    source.id !== `web-clipper:${clipId}` ||
    source.workspace_id !== workspaceId ||
    source.url !== url ||
    !Number.isSafeInteger(source.media_id) ||
    source.media_id <= 0 ||
    (mediaId !== undefined && source.media_id !== mediaId)
  )
    fail()
}
async function verifyVersion(
  version: MediaDocumentVersion,
  mediaId: number,
  clipId: string,
  workspaceId: string,
  descriptor: WebCaptureDescriptor,
  text?: string
): Promise<void> {
  const md = version.safe_metadata
  const actual = (md?.capture_metadata as Record<string, unknown> | undefined)
    ?.web_capture_v1 as WebCaptureDescriptor | undefined
  if (
    version.media_id !== mediaId ||
    !Number.isSafeInteger(version.version_number) ||
    version.version_number <= 0 ||
    !uuidPattern.test(version.uuid ?? "") ||
    md?.source !== "web_clipper" ||
    md?.clip_type !== "article" ||
    md?.clip_id !== clipId ||
    md?.workspace_id !== workspaceId ||
    md?.source_url !== descriptor.requested_url ||
    !actual ||
    Object.keys(actual).length !== 5 ||
    Object.keys(descriptor).some(
      (key) =>
        actual[key as keyof WebCaptureDescriptor] !==
        descriptor[key as keyof WebCaptureDescriptor]
    ) ||
    typeof version.content !== "string" ||
    (text !== undefined && version.content !== text) ||
    (await sha256Text(version.content)) !== descriptor.content_sha256
  )
    fail()
}
async function verifyClipPlacement(
  clipId: string,
  workspaceId: string,
  options: ScopedRequestOptions,
  check: () => void,
): Promise<void> {
  const status = await tldwClient.getWebClipStatus(clipId, options);
  check();
  if (
    status.clip_id !== clipId ||
    !["saved", "saved_with_warnings"].includes(status.status) ||
    typeof status.note?.id !== "string" ||
    !status.note.id.trim() ||
    !status.workspace_placements.some(
      (p) =>
        p.workspace_id === workspaceId && p.source_note_id === status.note.id,
    )
  )
    fail();
}

/** Recover only unique, intact canonical history; an absent checkpoint proves nothing. */
export async function recoverWebCapturePin(
  source: WorkspaceSourceApiResponse,
  options: ScopedRequestOptions,
  assertCurrent: () => void,
  known?: WebArticleCapturePin,
): Promise<WebArticleCapturePin | undefined> {
  const check = () => {
    assertScope(options);
    assertCurrent();
  };
  check();
  let captureEvidenceKnown = Boolean(known);
  try {
    const clipId = source.id.slice("web-clipper:".length);
    verifySource(source, clipId, source.workspace_id, source.url ?? "");
    await verifyClipPlacement(clipId, source.workspace_id, options, check);
    let accepted: MediaDocumentVersion | undefined;
    let descriptor: WebCaptureDescriptor | undefined;
    if (known) {
      if (
        known.clipId !== clipId ||
        known.mediaId !== source.media_id ||
        known.requestedUrl !== source.url
      )
        fail();
      descriptor = descriptorOf({
        source_url: source.url!,
        capture_metadata: {
          web_capture_v1: {
            mode: "server_article",
            requested_url: known.requestedUrl,
            captured_at: known.capturedAt,
            content_sha256: known.contentSha256,
            refresh_of: known.refreshOf,
          },
        },
      });
      accepted = {
        media_id: source.media_id,
        version_number: known.versionNumber,
        uuid: known.versionUuid,
        created_at: known.capturedAt,
      };
    } else {
      let ordinary = false;
      let previousNumber = Infinity;
      const limit = 10;
      for (let page = 1; ; page++) {
        const versions = await tldwClient.listMediaDocumentVersions(
          source.media_id,
          options,
          { limit, page },
        );
        check();
        if (!Array.isArray(versions) || versions.length > limit) fail();
        for (const version of versions) {
          if (
            version.media_id !== source.media_id ||
            !Number.isSafeInteger(version.version_number) ||
            version.version_number <= 0 ||
            version.version_number >= previousNumber
          )
            fail();
          previousNumber = version.version_number;
          const md = version.safe_metadata;
          const capture =
            md?.capture_metadata as WebClipperSaveRequest["capture_metadata"];
          if (capture && "web_capture_v1" in capture) {
            captureEvidenceKnown = true;
            if (accepted) fail();
            descriptor = descriptorOf({
              source_url: source.url!,
              capture_metadata: capture,
            });
            await verifyVersion(
              version,
              source.media_id,
              clipId,
              source.workspace_id,
              descriptor,
            );
            check();
            accepted = version;
          } else if (
            md?.source === "web_clipper" &&
            md.clip_id === clipId &&
            md.workspace_id === source.workspace_id &&
            md.source_url === source.url
          ) {
            ordinary = true;
          }
        }
        if (versions.length < limit) break;
      }
      if (!accepted && ordinary) return undefined;
    }
    if (!accepted || !descriptor) fail();
    const exact = await tldwClient.getMediaDocumentVersion(
      source.media_id,
      accepted.version_number,
      options,
    );
    check();
    if (
      exact.version_number !== accepted.version_number ||
      exact.uuid !== accepted.uuid
    )
      fail();
    await verifyVersion(
      exact,
      source.media_id,
      clipId,
      source.workspace_id,
      descriptor,
    );
    check();
    return {
      clipId,
      requestedUrl: descriptor.requested_url,
      capturedAt: descriptor.captured_at,
      contentSha256: descriptor.content_sha256,
      refreshOf: descriptor.refresh_of,
      mediaId: source.media_id,
      versionNumber: exact.version_number,
      versionUuid: exact.uuid!,
    };
  } catch (reason) {
    check();
    throw new WebCaptureNotCurrentError(
      reason instanceof Error ? reason.message : "Capture history unavailable",
      source.id,
      captureEvidenceKnown,
    );
  }
}

export async function confirmWebCaptureAcceptance(
  body: WebClipperSaveRequest,
  options: ScopedRequestOptions,
  assertCurrent: () => void,
  pin?: WebArticleCapturePin
): Promise<{ source: WorkspaceSourceApiResponse; pin: WebArticleCapturePin }> {
  const check = () => {
    assertScope(options)
    assertCurrent()
  }
  check()
  const descriptor = descriptorOf(body)
  const clipId = body.clip_id
  const workspaceId = body.workspace?.workspace_id
  const text = body.content?.full_extract
  if (
    !uuidPattern.test(clipId) ||
    !workspaceId ||
    typeof text !== "string" ||
    !text ||
    trimCaptureText(text) !== text ||
    (await sha256Text(text)) !== descriptor.content_sha256
  )
    fail()
  check()
  await verifyClipPlacement(clipId, workspaceId, options, check);
  const sources = await tldwClient.getWorkspaceSources(workspaceId, options)
  check()
  const source = sources.find((row) => row.id === `web-clipper:${clipId}`)
  verifySource(source, clipId, workspaceId, descriptor.requested_url)
  // A durable checkpoint addresses its exact active version, independent of pages.
  if (
    pin &&
    (pin.mediaId !== source.media_id ||
      pin.clipId !== clipId ||
      pin.requestedUrl !== descriptor.requested_url ||
      pin.capturedAt !== descriptor.captured_at ||
      pin.contentSha256 !== descriptor.content_sha256 ||
      pin.refreshOf !== descriptor.refresh_of ||
      !Number.isSafeInteger(pin.versionNumber) ||
      pin.versionNumber <= 0 ||
      !uuidPattern.test(pin.versionUuid))
  )
    fail()
  let accepted:
    | Pick<MediaDocumentVersion, "version_number" | "uuid">
    | undefined = pin
    ? { version_number: pin.versionNumber, uuid: pin.versionUuid }
    : undefined
  if (!pin) {
    const limit = 10
    for (let page = 1; !accepted; page++) {
      check()
      const versions = await tldwClient.listMediaDocumentVersions(
        source.media_id,
        options,
        { limit, page }
      )
      check()
      for (const version of versions) {
        check()
        try {
          await verifyVersion(
            version,
            source.media_id,
            clipId,
            workspaceId,
            descriptor,
            text
          )
        } catch (reason) {
          check()
          if (!(reason instanceof WebCaptureNotCurrentError)) throw reason
          continue
        }
        check()
        accepted = version
        break
      }
      if (versions.length < limit) break
    }
  }
  if (!accepted) fail()
  check()
  const exact = await tldwClient.getMediaDocumentVersion(
    source.media_id,
    accepted.version_number,
    options
  )
  check()
  if (
    exact.version_number !== accepted.version_number ||
    exact.uuid !== accepted.uuid
  )
    fail()
  await verifyVersion(
    exact,
    source.media_id,
    clipId,
    workspaceId,
    descriptor,
    text
  )
  check()
  return {
    source,
    pin: {
      clipId,
      requestedUrl: descriptor.requested_url,
      capturedAt: descriptor.captured_at,
      contentSha256: descriptor.content_sha256,
      refreshOf: descriptor.refresh_of,
      mediaId: source.media_id,
      versionNumber: exact.version_number,
      versionUuid: exact.uuid!
    }
  }
}
export async function assertWebCaptureHeadCurrent(
  source: WorkspaceSource,
  workspaceId: string,
  options: ScopedRequestOptions
): Promise<void> {
  try {
    assertScope(options)
    const pin = source.webCapture
    if (
      !pin ||
      source.id !== `web-clipper:${pin.clipId}` ||
      source.mediaId !== pin.mediaId ||
      source.url !== pin.requestedUrl
    )
      fail()
    const sources = await tldwClient.getWorkspaceSources(workspaceId, options)
    assertScope(options)
    verifySource(
      sources.find((row) => row.id === source.id),
      pin.clipId,
      workspaceId,
      pin.requestedUrl,
      pin.mediaId
    )
    const versions = await tldwClient.listMediaDocumentVersions(
      pin.mediaId,
      options
    )
    assertScope(options)
    const head = versions.reduce<MediaDocumentVersion | undefined>(
      (best, v) => (!best || v.version_number > best.version_number ? v : best),
      undefined
    )
    if (
      !head ||
      head.version_number !== pin.versionNumber ||
      head.uuid !== pin.versionUuid
    )
      fail()
    await verifyVersion(head, pin.mediaId, pin.clipId, workspaceId, {
      mode: "server_article",
      requested_url: pin.requestedUrl,
      captured_at: pin.capturedAt,
      content_sha256: pin.contentSha256,
      refresh_of: pin.refreshOf
    })
    assertScope(options)
    // Deleting a version does not roll back Media.content (the current FTS body).
    const current = await tldwClient.getMediaDetails(pin.mediaId, {
      ...options,
      include_content: true,
      include_versions: false,
      include_version_content: false
    })
    assertScope(options)
    if (
      current?.media_id !== pin.mediaId ||
      typeof current.content?.text !== "string" ||
      current.content.text !== head.content ||
      (await sha256Text(current.content.text)) !== pin.contentSha256
    )
      fail()
    assertScope(options)
  } catch (reason) {
    if (reason instanceof WebCaptureNotCurrentError) reason.sourceId = source.id
    throw reason
  }
}
