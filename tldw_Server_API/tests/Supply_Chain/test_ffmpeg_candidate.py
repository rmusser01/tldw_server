"""Behavioral tests for the candidate-only FFmpeg evaluator."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 ------
 V....D libx264              libx264 H.264 (codec h264)
 A....D libmp3lame           libmp3lame MP3 (codec mp3)
"""
DECODERS = """Decoders:
 V..... = Video
 A..... = Audio
 ------
 V....D h264                 H.264 / AVC
 A....D opus                 Opus
"""
FILTERS = """Filters:
  T.. = Timeline support
  A = Audio input/output
 ... aresample         A->A       Resample audio.
 ... libplacebo        V->V       Apply libplacebo filtering.
 ... pp                V->V       Filter video using libpostproc.
"""
FORMATS = """Formats:
 D.. = Demuxing supported
 .E. = Muxing supported
 ..d = Is a device
 ---
 D   matroska,webm    Matroska / WebM
  E  mp4             MP4
 D d video4linux2,v4l2 Video4Linux2 device grab
"""
PROTOCOLS = """Supported file protocols:
Input:
  file
  http
Output:
  file
  rtmp
"""


def test_verify_sha256_accepts_unchanged_content(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import verify_sha256

    source = tmp_path / "source.tar.xz"
    source.write_bytes(b"authenticated source")
    verify_sha256(source, hashlib.sha256(source.read_bytes()).hexdigest())


@pytest.mark.parametrize("state", ["tampered", "missing"])
def test_verify_sha256_rejects_untrusted_content(tmp_path: Path, state: str) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, verify_sha256

    source = tmp_path / "source.tar.xz"
    source.write_bytes(b"authenticated source")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    if state == "tampered":
        source.write_bytes(b"modified source")
    else:
        source.unlink()
    with pytest.raises(CandidateError, match="source"):
        verify_sha256(source, expected)


@pytest.mark.parametrize("expected", ["", "0" * 63, "G" * 64, "A" * 64, "sha256:" + "0" * 64])
def test_verify_sha256_rejects_malformed_expected_hash(tmp_path: Path, expected: str) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, verify_sha256

    source = tmp_path / "source.tar.xz"
    source.write_bytes(b"source")
    with pytest.raises(CandidateError, match="SHA-256"):
        verify_sha256(source, expected)


@pytest.mark.parametrize(
    ("category", "listing", "expected"),
    [
        ("encoders", ENCODERS, {"libx264", "libmp3lame"}),
        ("decoders", DECODERS, {"h264", "opus"}),
        ("filters", FILTERS, {"aresample", "libplacebo", "pp"}),
        ("demuxers", FORMATS, {"matroska", "webm", "video4linux2", "v4l2"}),
        ("muxers", FORMATS, {"mp4"}),
        ("input_protocols", PROTOCOLS, {"file", "http"}),
        ("output_protocols", PROTOCOLS, {"file", "rtmp"}),
    ],
)
def test_parse_capabilities_reads_real_listing_shapes(
    category: str,
    listing: str,
    expected: set[str],
) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import parse_capabilities

    assert parse_capabilities(listing, category) == expected


@pytest.mark.parametrize("listing", ["", "Encoders:\n ------\n malformed row\n"])
def test_parse_capabilities_rejects_empty_or_malformed_inventory(listing: str) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, parse_capabilities

    with pytest.raises(CandidateError, match="inventory"):
        parse_capabilities(listing, "encoders")


def test_compare_capabilities_allows_only_removed_pp_filter() -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import compare_capabilities

    baseline = {
        "encoders": {"libx264", "libmp3lame", "libopus"},
        "decoders": {"h264", "mp3", "opus"},
        "demuxers": {"mp3", "mov"},
        "muxers": {"mp3", "mp4"},
        "filters": {"aresample", "libplacebo", "pp"},
        "input_protocols": {"file", "http"},
        "output_protocols": {"file", "rtmp"},
    }
    candidate = {category: set(values) for category, values in baseline.items()}
    candidate["filters"].remove("pp")
    assert compare_capabilities(baseline, candidate) == {}


def test_compare_capabilities_reports_every_other_removed_capability() -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import compare_capabilities

    baseline = {
        "encoders": {"libx264", "libmp3lame", "libopus"},
        "decoders": {"h264", "mp3", "opus"},
        "demuxers": {"mp3", "mov"},
        "muxers": {"mp3", "mp4"},
        "filters": {"aresample", "libplacebo", "pp"},
        "input_protocols": {"file", "http"},
        "output_protocols": {"file", "rtmp"},
    }
    candidate = {category: set(values) for category, values in baseline.items()}
    candidate["encoders"] -= {"libmp3lame", "libopus"}
    candidate["decoders"] -= {"h264"}
    candidate["filters"] -= {"libplacebo", "pp"}
    candidate["output_protocols"] -= {"rtmp"}
    assert compare_capabilities(baseline, candidate) == {
        "encoders": {"libmp3lame", "libopus"},
        "decoders": {"h264"},
        "filters": {"libplacebo"},
        "output_protocols": {"rtmp"},
    }


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ({}, {}),
        ({"encoders": {"libx264"}}, {"encoders": {"libx264"}}),
    ],
)
def test_compare_capabilities_rejects_incomplete_inventory(
    baseline: dict[str, set[str]],
    candidate: dict[str, set[str]],
) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, compare_capabilities

    with pytest.raises(CandidateError, match="inventory"):
        compare_capabilities(baseline, candidate)


@pytest.mark.parametrize("identity", ["latest", "candidate:test", "sha256:" + "A" * 64, "sha256:" + "0" * 63])
def test_candidate_image_identity_must_be_an_immutable_digest(identity: str) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, validate_candidate_image

    with pytest.raises(CandidateError, match="image"):
        validate_candidate_image(identity)


def test_candidate_image_identity_accepts_a_sha256_digest() -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import validate_candidate_image

    validate_candidate_image("sha256:" + "a" * 64)


def test_metadata_validation_rejects_a_wrong_or_missing_value() -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, require_metadata

    with pytest.raises(CandidateError, match="metadata"):
        require_metadata(
            {"codec_name": "pcm_s16le", "sample_rate": "48000"}, codec_name="pcm_s16le", sample_rate="16000"
        )
    with pytest.raises(CandidateError, match="metadata"):
        require_metadata({"codec_name": "png"}, codec_name="png", width=160, height=120)


def _real_tools() -> tuple[Path, Path]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg and ffprobe are required for the real candidate probe")
    return Path(ffmpeg), Path(ffprobe)


def _combined_baseline(ffmpeg: Path) -> str:
    listings = []
    for option in ("-encoders", "-decoders", "-demuxers", "-muxers", "-filters", "-protocols"):
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", option],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        listings.append(result.stdout)
    return "".join(listings)


@pytest.mark.integration
def test_synthetic_probe_rejects_a_directory_with_stale_segments(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import CandidateError, run_synthetic_probes

    ffmpeg, ffprobe = _real_tools()
    media = tmp_path / "media"
    media.mkdir()
    (media / "segment-00.wav").write_bytes(b"stale")
    with pytest.raises(CandidateError, match="empty"):
        run_synthetic_probes(ffmpeg, ffprobe, media)


@pytest.mark.integration
def test_evaluate_candidate_creates_a_compatible_real_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts.Supply_Chain import ffmpeg_candidate

    ffmpeg, ffprobe = _real_tools()
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(_combined_baseline(ffmpeg))
    source = tmp_path / "ffmpeg.tar.xz"
    source.write_bytes(b"portable source fixture")
    expected_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(ffmpeg_candidate, "FFMPEG_SOURCE_SHA256", expected_source_sha256)
    output = tmp_path / "output"
    report = ffmpeg_candidate.evaluate_candidate(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        baseline_file=baseline,
        source_archive=source,
        candidate_image="sha256:" + "a" * 64,
        output_dir=output,
    )

    assert report["compatible"] is True
    assert report["source"]["sha256"] == expected_source_sha256
    assert report["missing_capabilities"] == {}
    assert report["configure_arguments"]
    assert "signature_evidence" not in report
    assert json.loads((output / "candidate-evaluation.json").read_text()) == report
    assert (output / "inventory" / "buildconf.txt").read_text().strip()


@pytest.mark.integration
def test_cli_persists_an_incompatible_report_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.Supply_Chain import ffmpeg_candidate

    ffmpeg, ffprobe = _real_tools()
    baseline_text = _combined_baseline(ffmpeg).replace(
        "Decoders:",
        " V....D removed_candidate_encoder Deliberately missing encoder\nDecoders:",
        1,
    )
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(baseline_text)
    source = tmp_path / "ffmpeg.tar.xz"
    source.write_bytes(b"portable source fixture")
    expected_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(ffmpeg_candidate, "FFMPEG_SOURCE_SHA256", expected_source_sha256)
    output = tmp_path / "output"
    argv = [
        "--ffmpeg",
        str(ffmpeg),
        "--ffprobe",
        str(ffprobe),
        "--baseline",
        str(baseline),
        "--source-archive",
        str(source),
        "--candidate-image",
        "sha256:" + "b" * 64,
        "--output-dir",
        str(output),
    ]
    with pytest.raises(SystemExit) as exit_info:
        ffmpeg_candidate.main(argv)

    assert exit_info.value.code == 1
    persisted = json.loads((output / "candidate-evaluation.json").read_text())
    assert persisted["compatible"] is False
    assert persisted["missing_capabilities"] == {"encoders": ["removed_candidate_encoder"]}
    assert json.loads(capsys.readouterr().out) == persisted


def test_cli_rejects_forged_signature_text_as_an_unknown_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import main

    forged = tmp_path / "forged-signature.txt"
    forged.write_text(
        "[GNUPG:] VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8 2026-08-12\n"
        "cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635  ffmpeg-9.0.1.tar.xz\n"
    )
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--baseline",
                str(tmp_path / "baseline.txt"),
                "--source-archive",
                str(tmp_path / "source.tar.xz"),
                "--candidate-image",
                "sha256:" + "c" * 64,
                "--output-dir",
                str(tmp_path / "output"),
                "--signature-evidence",
                str(forged),
            ]
        )
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --signature-evidence" in capsys.readouterr().err


@pytest.mark.integration
def test_real_ffmpeg_inventory_and_synthetic_media_probes(tmp_path: Path) -> None:
    from Helper_Scripts.Supply_Chain.ffmpeg_candidate import collect_inventory, run_synthetic_probes

    ffmpeg, ffprobe = _real_tools()
    inventory = collect_inventory(ffmpeg, tmp_path / "inventory")
    assert "libx264" in inventory["encoders"]
    assert "h264" in inventory["decoders"]
    assert "aresample" in inventory["filters"]
    assert (tmp_path / "inventory" / "encoders.txt").read_text().startswith("Encoders:")
    assert (tmp_path / "inventory" / "protocols.txt").read_text().startswith("Supported file protocols:")

    result = run_synthetic_probes(ffmpeg, ffprobe, tmp_path / "media")
    assert result["wav_resample"] == {"codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}
    assert result["audio_round_trips"] == {"mp3": "mp3", "flac": "flac", "opus": "opus", "aac": "aac"}
    assert result["mp4"] == {"video_codec": "h264", "audio_codec": "aac"}
    assert result["thumbnail"] == {"codec_name": "png", "width": 160, "height": 120}
    assert result["segment_concat"]["segments"] >= 2
    assert result["segment_concat"]["decoded_bytes"] > 16000
    assert result["segment_concat"]["decoded_samples"] >= 48000
