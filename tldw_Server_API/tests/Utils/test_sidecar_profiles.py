from Helper_Scripts.common.sidecar_profiles import resolve_workers


def test_resolve_workers_tts_only_profile():
    workers = resolve_workers(
        profile="tts-only",
        explicit_workers_csv=None,
        default_workers_csv="media_ingest,audio_jobs,embeddings",
    )
    assert workers == ["audio_jobs"]


def test_resolve_workers_tts_only_profile_with_audio_key():
    workers = resolve_workers(
        profile="tts-only",
        explicit_workers_csv=None,
        default_workers_csv="media_ingest,audio,embeddings",
    )
    assert workers == ["audio"]


def test_resolve_workers_tts_only_profile_includes_the_long_form_tts_worker():
    # The tts-only profile must run the worker that processes POST /audio/speech/jobs (TASK-13849).
    workers = resolve_workers(
        profile="tts-only",
        explicit_workers_csv=None,
        default_workers_csv="media_ingest,audio,tts,embeddings",
    )
    assert workers == ["audio", "tts"]
