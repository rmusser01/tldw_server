# Video summarization Service Prompt

Approved scope: synchronous `/api/v1/media/process-videos`. Track in TASK-13194.

Expose `media.video.summarization` in the existing shared Settings editor with
an atomic literal pair: `system` and `final_summary`. The system part applies to
every analysis call. The final-summary part applies only when recursive mode
combines multiple chunk summaries; it must not become a chunk instruction.

Capture one owner-specific configuration after input validation and before
uploads or transcription. Explicit request `system_prompt` wins independently;
explicit `custom_prompt` continues to apply to initial and final passes, including
intentional empty text. When no override exists, retain the deployed analyzer
system default and the existing final-summary instruction. Initial analysis has
no new default user suffix. Skip prompt storage when analysis is inactive, or
all relevant prompt parts are explicit. Close lookup connections on their worker.

Pass final-summary text separately through the existing batch/core call chain,
without adding request schema fields or a new storage system. Existing direct
and queued callers retain their legacy defaults. Normalize the canonical
`api_provider` ahead of legacy `api_name`, matching synchronous audio, so the
canonical field does not accidentally disable analysis and prompt resolution.
Transcription, chunking,
providers, output contracts and confabulation checks are out of scope.

Verify owner isolation, independent precedence, literal braces, empty multipart
parts, frozen settings across files/passes, defaults/reset, disabled-analysis
bypass, corruption/cleanup, and shared editor pair save/reset. Run focused video
and service-prompt regressions, Bandit, and OpenAPI generation before review.
