# TTS API

## User Guide Map

- [Getting Started — STT and TTS](../User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md) — quickstart for first successful speech requests.
- [TTS Providers Getting Started](../User_Guides/WebUI_Extension/TTS_Getting_Started.md) — provider selection and first successful synthesis.
- [TTS Provider Setup Guide](../User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md) — runbook index for deep provider setup/tuning.

PocketTTS note:

- `pocket_tts` is the Python/ONNX runtime.
- `pocket_tts_cpp` is the compiled native runtime.
- Both use the same public request fields and the same `/api/v1/audio/speech` route.
- For either provider, voice cloning requests must supply either a direct base64 `voice_reference` or `voice: "custom:<voice_id>"` for a stored reference.
- `pocket_tts_cpp` only exposes streaming when the local CLI probe proves the install is truly incremental; otherwise the adapter fails closed.
- This page documents public request inputs only. Provider-internal `extra_params` keys are intentionally omitted.

OmniVoice note:

- `omnivoice` is a first-class local provider backed by a managed loopback sidecar runtime.
- Requests that explicitly resolve to OmniVoice and omit `voice` normalize to `voice: "auto"`.
- Explicit caller-provided voices are preserved, including `custom:<voice_id>` selections.
- Setup/install details for the dedicated runtime live in `Docs/STT-TTS/TTS-SETUP-GUIDE.md`.

1. Client pattern (Python)

  Non‑streaming usage (stream: false), handling JSON errors vs audio:
```python
  import base64
  import json
  import os
  from urllib.error import HTTPError
  from urllib.request import Request, urlopen

  BASE_URL = os.getenv("TLDW_BASE_URL", "http://127.0.0.1:8000")
  API_KEY = os.environ["SINGLE_USER_API_KEY"]
  url = f"{BASE_URL}/api/v1/audio/speech"

  payload = {
      "model": "kokoro",
      "input": "Hello from client",
      "voice": "af_heart",
      "response_format": "mp3",
      "stream": False,
  }

  req = Request(
      url,
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
      method="POST",
  )

  try:
      with urlopen(req) as resp:
          # Success: treat as audio
          # Optional alignment header (non-streaming, when available)
          alignment_b64 = resp.headers.get("X-TTS-Alignment")
          if alignment_b64:
              alignment = json.loads(base64.urlsafe_b64decode(alignment_b64).decode("utf-8"))
          with open("speech.mp3", "wb") as f:
              f.write(resp.read())
  except HTTPError as err:
      # Structured error path
      body = err.read().decode("utf-8")
      try:
          err_json = json.loads(body)
      except json.JSONDecodeError:
          err_json = {"detail": body}
      detail = err_json.get("detail", err_json)
      raise RuntimeError(f"TTS failed ({err.code}): {detail}")
```

Streaming variant (Python):
```python
  import json
  import os
  from urllib.error import HTTPError
  from urllib.request import Request, urlopen

  payload_stream = {**payload, "stream": True}
  req = Request(
      url,
      data=json.dumps(payload_stream).encode("utf-8"),
      headers={"Content-Type": "application/json", "X-API-KEY": API_KEY},
      method="POST",
  )

  try:
      with urlopen(req) as resp:
          with open("speech_streamed.mp3", "wb") as f:
              while True:
                  chunk = resp.read(8192)
                  if not chunk:
                      break
                  f.write(chunk)
  except HTTPError as err:
      body = err.read().decode("utf-8")
      try:
          err_json = json.loads(body)
      except json.JSONDecodeError:
          err_json = {"detail": body}
      detail = err_json.get("detail", err_json)
      raise RuntimeError(f"TTS failed ({err.code}): {detail}")
```
  Key points for all Python clients:

  - Use try/except HTTPError to catch non-2xx responses.
  - Only treat body as audio when the request succeeds (no exception).
  - On non‑200, parse JSON and surface detail to the caller.
  - Alignment metadata (if available) is returned via `X-TTS-Alignment` (base64url JSON) on non-streaming responses.
  - `return_download_link` requires `stream: false`. When enabled, the response includes `X-Download-Path` and `X-Generated-File-Id` headers pointing at `/api/v1/storage/files/{id}/download`.
  - For streaming alignment, call `POST /api/v1/audio/speech/metadata` with the same payload.
  - Kokoro phoneme override keys (inside `extra_params`):
    - `phoneme_overrides` or `phoneme_map`: request-level map/list that takes highest precedence
    - `disable_phoneme_overrides`: boolean kill-switch for this request
    - `phoneme_overrides_enabled`: explicit on/off flag (overrides provider default)
  - Runtime precedence for Kokoro overrides: `request > provider > global`.

  ———

  1.b Provider and Voice Catalog Shapes

  `GET /api/v1/audio/providers` returns capabilities and voice catalog:

```json
  {
    "providers": {
      "openai": {
        "provider_name": "OpenAI",
        "languages": ["en"],
        "formats": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
        "max_text_length": 4096,
        "supports_streaming": true,
        "supports_voice_cloning": false,
        "default_format": "mp3",
        "sample_rate": 24000,
        "voices": [
          { "id": "alloy", "name": "Alloy", "gender": "neutral", "language": "en" },
          { "id": "nova",  "name": "Nova",  "gender": "female",  "language": "en" }
        ]
      },
      "kokoro": {
        "provider_name": "Kokoro",
        "languages": ["en-us", "en-gb", "en"],
        "formats": ["mp3", "wav", "opus"],
        "max_text_length": 1000000,
        "supports_streaming": true,
        "default_format": "wav",
        "sample_rate": 24000,
        "voices": [
          { "id": "af_bella", "name": "Bella", "gender": "female", "language": "en-us" }
        ]
      }
    },
    "voices": {
      "openai": [
        { "id": "alloy", "name": "Alloy", "gender": "neutral", "language": "en" },
        { "id": "nova",  "name": "Nova",  "gender": "female",  "language": "en" }
      ],
      "kokoro": [
        { "id": "af_bella", "name": "Bella", "gender": "female", "language": "en-us" }
      ]
    },
    "timestamp": "2025-01-01T12:00:00Z"
  }
```

  `GET /api/v1/audio/voices/catalog` (optionally `?provider=openai`) returns a provider→voices map:

```json
  {
    "openai": [
      { "id": "alloy", "name": "Alloy", "gender": "neutral", "language": "en" },
      { "id": "echo",  "name": "Echo",  "gender": "male",    "language": "en" },
      { "id": "nova",  "name": "Nova",  "gender": "female",  "language": "en" }
    ]
  }
```

  These shapes are approximate but match the real fields produced by `TTSServiceV2.get_capabilities()` and `TTSServiceV2.list_voices()`.

  ———

  2. Client pattern (JavaScript / TypeScript)

  Non‑streaming with fetch:
```javascript
  const baseUrl = process.env.TLDW_BASE_URL ?? "http://127.0.0.1:8000";

  async function ttsOnce(input: string): Promise<Blob> {
    const resp = await fetch(`${baseUrl}/api/v1/audio/speech`, {
      method: "POST",
      headers: {
        "X-API-KEY": process.env.SINGLE_USER_API_KEY!,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "kokoro",
        input,
        voice: "af_heart",
        response_format: "mp3",
        stream: false,
      }),
    });

    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const json = await resp.json();
        detail = json.detail ?? JSON.stringify(json);
      } catch (_) {
        detail = await resp.text();
      }
      throw new Error(`TTS failed: ${detail}`);
    }

    return await resp.blob(); // audio/mpeg
  }
```
  Streaming (browser) is trickier; typical pattern is:

  - Check resp.ok first.
  - If ok, read resp.body as a stream (e.g., into a MediaSource or buffer it).
  - If not ok, read JSON/text as above and treat as an error.
  - For streaming alignment, call `POST /api/v1/audio/speech/metadata` with the same payload and parse the JSON response.



 Assumptions

  - Prometheus data source in Grafana is configured (e.g. named Prometheus).
  - You’re scraping:
      - tts_requests_total{provider,model,voice,format,status} (counter)
      - tts_request_duration_seconds_bucket (histogram)
      - tts_fallback_outcomes_total{from_provider,to_provider,outcome,category} (counter)
      - tts_ttfb_seconds_bucket (histogram)
      - voice_to_voice_seconds_bucket (histogram, when voice-to-voice timing is provided)
  - For HTTP, adjust metric names/labels to your setup (examples use http_requests_total and path="/api/v1/audio/speech"; if yours differ, just swap names/labels).

  ———

  1. Panel: TTS Success Rate

  - Panel type: Time series
  - Title: TTS Success Rate
  - Query:

  sum(rate(tts_requests_total{status="success"}[5m]))
  /
  sum(rate(tts_requests_total[5m]))

  - Legend: success_rate
  - Display: as percentage (set unit to percent (0-1)).

  ———

  2. Panel: TTS Requests by Status

  - Panel type: Time series (stacked)
  - Title: TTS Requests by Status
  - Queries:

  sum by (status) (rate(tts_requests_total[5m]))

  - Stacking: normal
  - Legend: {{status}}

  This gives you success vs failure volume over time.

  ———

  3. Panel: TTS Latency (p50, p95)

  - Panel type: Time series
  - Title: TTS Latency (p50 / p95)
  - Queries:

  p50:

  histogram_quantile(
    0.50,
    sum(rate(tts_request_duration_seconds_bucket[5m])) by (le)
  )

  p95:

  histogram_quantile(
    0.95,
    sum(rate(tts_request_duration_seconds_bucket[5m])) by (le)
  )

  - Legend: p50, p95
  - Unit: s (seconds).

  If you want per-provider, add provider to the by clause:

  histogram_quantile(
    0.95,
    sum(rate(tts_request_duration_seconds_bucket[5m])) by (provider, le)
  )

  ———

  4. Panel: HTTP Error Rate for /audio/speech

  (Assuming http_requests_total{path="/api/v1/audio/speech",status="200"} or similar.)

  - Panel type: Time series
  - Title: HTTP Error Rate /api/v1/audio/speech
  - Query:

  sum(rate(http_requests_total{path="/api/v1/audio/speech", status=~"5.."}[5m]))
  /
  sum(rate(http_requests_total{path="/api/v1/audio/speech"}[5m]))

  - Unit: percent (0-1).

  If your metric is fastapi_requests_total or something else, just substitute the name and labels.

  ———

  5. Panel: Failures by Provider

  - Panel type: Time series or Bar chart
  - Title: TTS Failures by Provider
  - Query:

  sum by (provider) (
    rate(tts_requests_total{status="failure"}[5m])
  )

  - Legend: {{provider}}.

  This quickly shows which provider is causing most failures.

  ———

  5b. Panel: Fallback Outcomes by Category

  - Panel type: Stacked bar or time series
  - Title: TTS Fallback Outcomes
  - Query:

  sum by (outcome, category) (
    rate(tts_fallback_outcomes_total[5m])
  )

  - Legend: {{outcome}} / {{category}}

  This distinguishes retryable/provider/network fallback behavior from exhausted or unavailable fallback paths.

  ———

  6. Alerts (Grafana 8+ unified alerts conceptually)

  Use the same PromQL queries in alert rules.

  - Alert: TTS success rate below 95%
      - Query expression:

        sum(rate(tts_requests_total{status="success"}[5m]))
        /
        sum(rate(tts_requests_total[5m]))
      - Condition: WHEN query(A) IS BELOW 0.95 FOR 10m
      - Labels: severity=warning
      - Annotation summary: TTS success rate below 95%
      - Annotation description: short text like Check /api/v1/audio/speech logs and provider status.
  - Alert: HTTP 5xx rate for /audio/speech > 1%
      - Query expression:

        sum(rate(http_requests_total{path="/api/v1/audio/speech", status=~"5.."}[5m]))
        /
        sum(rate(http_requests_total{path="/api/v1/audio/speech"}[5m]))
      - Condition: WHEN query(A) IS ABOVE 0.01 FOR 10m
      - Labels: severity=critical
      - Annotation summary: High 5xx rate on /api/v1/audio/speech.
  - Alert: p95 TTS latency too high
      - Query expression (p95):

        histogram_quantile(
          0.95,
          sum(rate(tts_request_duration_seconds_bucket[5m])) by (le)
        )
      - Condition: WHEN query(A) IS ABOVE 5 FOR 10m (5 seconds as an example SLO)
      - Labels: severity=warning
      - Annotation summary: TTS p95 latency > 5s.
  - Alert: fallback exhaustion spikes
      - Query expression:

        sum(rate(tts_fallback_outcomes_total{outcome=~"exhausted|unavailable"}[10m]))
      - Condition: WHEN query(A) IS ABOVE 0 FOR 10m
      - Labels: severity=warning
      - Annotation summary: TTS fallback exhausted/unavailable.
      - Annotation description: Check provider health, credentials, and adapter availability.

  ———

  7. Request Correlation

  - Always send `X-Request-ID` for client-generated correlation.
  - `POST /api/v1/audio/speech` and `POST /api/v1/audio/speech/metadata` return `X-Request-Id` and propagate request/correlation IDs into TTS metadata where available.
  - When triaging incidents, pivot by `request_id` in logs and compare with `tts_fallback_outcomes_total` spikes for the same time window.
