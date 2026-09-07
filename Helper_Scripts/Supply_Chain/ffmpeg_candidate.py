"""Evaluate the authenticated FFmpeg 9 candidate without promoting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess  # nosec B404
from collections.abc import Mapping
from pathlib import Path

FFMPEG_SOURCE_SHA256 = "cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635"
CAPABILITY_CATEGORIES = (
    "encoders",
    "decoders",
    "demuxers",
    "muxers",
    "filters",
    "input_protocols",
    "output_protocols",
)
APPROVED_RETIREMENTS = {
    "encoders": frozenset({"sonic", "sonicls", "v308", "v408", "v410"}),
    "decoders": frozenset({"sonic", "v308", "v408", "v410"}),
    "muxers": frozenset({"opengl", "sdl", "sdl2"}),
    "filters": frozenset({"pp"}),
    "input_protocols": frozenset({"hls"}),
}
_TOP_LEVEL_HEADERS = frozenset({"Encoders:", "Decoders:", "Formats:", "Filters:", "Supported file protocols:"})
_CODEC_ROW = re.compile(r"^\s[VAS][A-Z.]{5}\s+([A-Za-z0-9_.+,-]+)\s+")
_FILTER_ROW = re.compile(r"^\s[A-Z.|]{2,4}\s+([A-Za-z0-9_.+,-]+)\s+")
_FORMAT_ROW = re.compile(r"^ ([D ])([E ])([d ]) ([A-Za-z0-9_.+,-]+)\s+")
_PROTOCOL_ROW = re.compile(r"^\s{2}([A-Za-z0-9_.+,-]+)\s*$")


class CandidateError(ValueError):
    """Raised when candidate evidence is missing, malformed, or incompatible."""


def verify_sha256(path: Path, expected: str) -> None:
    """Require a regular file to match one lowercase SHA-256 digest."""
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise CandidateError("expected SHA-256 is malformed")
    try:
        if path.is_symlink() or not path.is_file():
            raise CandidateError("source file is missing or not regular")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise CandidateError("source file could not be read") from exc
    if digest != expected:
        raise CandidateError("source SHA-256 mismatch")


def _named_section(output: str, header: str) -> list[str]:
    lines = output.splitlines()
    try:
        start = lines.index(header) + 1
    except ValueError as exc:
        raise CandidateError(f"malformed {header[:-1].lower()} inventory") from exc
    end = next((index for index in range(start, len(lines)) if lines[index] in _TOP_LEVEL_HEADERS), len(lines))
    return lines[start:end]


def _aliases(value: str) -> set[str]:
    return {name for name in value.split(",") if name}


def parse_capabilities(output: str, category: str) -> set[str]:
    """Parse one real FFmpeg capability listing, including aliases."""
    if category in {"encoders", "decoders"}:
        header = "Encoders:" if category == "encoders" else "Decoders:"
        rows = _named_section(output, header)
        pattern = _CODEC_ROW
    elif category == "filters":
        rows = _named_section(output, "Filters:")
        pattern = _FILTER_ROW
    elif category in {"demuxers", "muxers"}:
        if "Formats:" not in output.splitlines():
            raise CandidateError(f"malformed {category} inventory")
        rows = output.splitlines()
        flag_index = 1 if category == "demuxers" else 2
        capabilities: set[str] = set()
        for row in rows:
            match = _FORMAT_ROW.match(row)
            if match is not None and match.group(flag_index).strip():
                capabilities.update(_aliases(match.group(4)))
        if not capabilities:
            raise CandidateError(f"empty or malformed {category} inventory")
        return capabilities
    elif category in {"input_protocols", "output_protocols"}:
        rows = _named_section(output, "Supported file protocols:")
        section = "Input:" if category == "input_protocols" else "Output:"
        try:
            start = rows.index(section) + 1
        except ValueError as exc:
            raise CandidateError(f"malformed {category} inventory") from exc
        end = next((index for index in range(start, len(rows)) if rows[index] in {"Input:", "Output:"}), len(rows))
        capabilities = set()
        for row in rows[start:end]:
            match = _PROTOCOL_ROW.fullmatch(row)
            if match is not None:
                capabilities.update(_aliases(match.group(1)))
        if not capabilities:
            raise CandidateError(f"empty or malformed {category} inventory")
        return capabilities
    else:
        raise CandidateError(f"unknown capability category: {category}")

    capabilities = set()
    for row in rows:
        match = pattern.match(row)
        if match is not None:
            capabilities.update(_aliases(match.group(1)))
    if not capabilities:
        raise CandidateError(f"empty or malformed {category} inventory")
    return capabilities


def compare_capabilities(
    baseline: Mapping[str, set[str]],
    candidate: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """Return removals outside the explicitly approved FFmpeg 9 retirements."""
    if any(not baseline.get(category) or not candidate.get(category) for category in CAPABILITY_CATEGORIES):
        raise CandidateError("incomplete or empty capability inventory")
    missing: dict[str, set[str]] = {}
    for category, expected in baseline.items():
        actual = candidate.get(category)
        if not expected or not actual:
            raise CandidateError(f"empty or missing {category} inventory")
        removed = expected - actual
        removed.difference_update(APPROVED_RETIREMENTS.get(category, ()))
        if removed:
            missing[category] = removed
    return missing


def observed_approved_retirements(
    baseline: Mapping[str, set[str]],
    candidate: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """Return only approved retirements observed in this capability delta."""
    observed = {}
    for category, approved in APPROVED_RETIREMENTS.items():
        retired = (baseline[category] - candidate[category]) & approved
        if retired:
            observed[category] = retired
    return observed


def validate_candidate_image(identity: str) -> None:
    """Require the immutable local image ID recorded by Docker inspect."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", identity) is None:
        raise CandidateError("candidate image identity must be a SHA-256 digest")


def require_metadata(actual: Mapping[str, object], **expected: object) -> None:
    """Fail when ffprobe metadata differs from the probe contract."""
    if any(actual.get(field) != value for field, value in expected.items()):
        raise CandidateError("candidate media metadata mismatch")


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 60) -> str:
    try:
        # Commands are fixed argument arrays; no shell parses candidate paths.
        result = subprocess.run(  # nosec B603
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CandidateError(f"candidate command failed: {command[1] if len(command) > 1 else command[0]}") from exc
    return result.stdout


def collect_inventory(ffmpeg: Path, output_dir: Path) -> dict[str, set[str]]:
    """Capture raw FFmpeg configuration/listings and return parsed inventories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "buildconf": "-buildconf",
        "encoders": "-encoders",
        "decoders": "-decoders",
        "demuxers": "-demuxers",
        "muxers": "-muxers",
        "filters": "-filters",
        "protocols": "-protocols",
    }
    raw = {}
    for name, option in commands.items():
        raw[name] = _run([str(ffmpeg), "-hide_banner", option])
        (output_dir / f"{name}.txt").write_text(raw[name], encoding="utf-8")
    return {
        "encoders": parse_capabilities(raw["encoders"], "encoders"),
        "decoders": parse_capabilities(raw["decoders"], "decoders"),
        "demuxers": parse_capabilities(raw["demuxers"], "demuxers"),
        "muxers": parse_capabilities(raw["muxers"], "muxers"),
        "filters": parse_capabilities(raw["filters"], "filters"),
        "input_protocols": parse_capabilities(raw["protocols"], "input_protocols"),
        "output_protocols": parse_capabilities(raw["protocols"], "output_protocols"),
    }


def _probe_streams(ffprobe: Path, media: Path) -> list[dict[str, object]]:
    output = _run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,sample_rate,channels,width,height",
            "-of",
            "json",
            str(media),
        ]
    )
    try:
        streams = json.loads(output)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CandidateError(f"malformed ffprobe output for {media.name}") from exc
    if not isinstance(streams, list) or not streams:
        raise CandidateError(f"ffprobe returned no streams for {media.name}")
    return streams


def _decoded_pcm(ffmpeg: Path, media: Path, pcm: Path) -> tuple[int, int]:
    _run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(media),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(pcm),
        ]
    )
    decoded = pcm.read_bytes()
    samples = len(decoded) // 2
    if samples < 48000 or not any(decoded):
        raise CandidateError(f"decoded media is short or silent: {media.name}")
    return len(decoded), samples


def run_synthetic_probes(ffmpeg: Path, ffprobe: Path, output_dir: Path) -> dict[str, object]:
    """Exercise the candidate's software-only media workflows with real data."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CandidateError("synthetic probe output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "source.wav"
    _run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1.2",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(source),
        ]
    )
    resampled = output_dir / "resampled.wav"
    _run([str(ffmpeg), "-y", "-v", "error", "-i", str(source), "-ar", "16000", str(resampled)])
    audio_stream = _probe_streams(ffprobe, resampled)[0]
    wav_result = {field: audio_stream[field] for field in ("codec_name", "sample_rate", "channels")}
    require_metadata(wav_result, codec_name="pcm_s16le", sample_rate="16000", channels=1)

    audio_round_trips = {}
    encodings = {
        "mp3": ("libmp3lame", "mp3", "mp3"),
        "flac": ("flac", "flac", "flac"),
        "opus": ("libopus", "opus", "opus"),
        "aac": ("aac", "m4a", "aac"),
    }
    for name, (encoder, extension, expected_codec) in encodings.items():
        encoded = output_dir / f"audio.{extension}"
        _run([str(ffmpeg), "-y", "-v", "error", "-i", str(source), "-c:a", encoder, str(encoded)])
        stream = _probe_streams(ffprobe, encoded)[0]
        if stream.get("codec_name") != expected_codec:
            raise CandidateError(f"unexpected {name} codec metadata")
        _decoded_pcm(ffmpeg, encoded, output_dir / f"{name}.pcm")
        audio_round_trips[name] = expected_codec

    video = output_dir / "video.mp4"
    _run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1.2",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ]
    )
    video_streams = _probe_streams(ffprobe, video)
    video_codec = next(stream.get("codec_name") for stream in video_streams if stream.get("codec_type") == "video")
    audio_codec = next(stream.get("codec_name") for stream in video_streams if stream.get("codec_type") == "audio")
    if (video_codec, audio_codec) != ("h264", "aac"):
        raise CandidateError("unexpected MP4 stream metadata")

    thumbnail = output_dir / "thumbnail.png"
    _run([str(ffmpeg), "-y", "-v", "error", "-i", str(video), "-frames:v", "1", str(thumbnail)])
    image_stream = _probe_streams(ffprobe, thumbnail)[0]
    thumbnail_result = {field: image_stream[field] for field in ("codec_name", "width", "height")}
    require_metadata(thumbnail_result, codec_name="png", width=160, height=120)

    _run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            "0.4",
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            "segment-%02d.wav",
        ],
        cwd=output_dir,
    )
    segments = sorted(output_dir.glob("segment-*.wav"))
    if len(segments) < 2:
        raise CandidateError("segment probe produced fewer than two segments")
    concat_list = output_dir / "segments.txt"
    concat_list.write_text("".join(f"file '{path.name}'\n" for path in segments), encoding="utf-8")
    joined = output_dir / "joined.wav"
    _run(
        [
            str(ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            concat_list.name,
            "-c",
            "copy",
            joined.name,
        ],
        cwd=output_dir,
    )
    joined_size, joined_samples = _decoded_pcm(ffmpeg, joined, output_dir / "joined.pcm")
    return {
        "wav_resample": wav_result,
        "audio_round_trips": audio_round_trips,
        "mp4": {"video_codec": video_codec, "audio_codec": audio_codec},
        "thumbnail": thumbnail_result,
        "segment_concat": {"segments": len(segments), "decoded_bytes": joined_size, "decoded_samples": joined_samples},
        "scope": "software-only probes; no real GPU or device capability proven",
    }


def evaluate_candidate(
    *,
    ffmpeg: Path,
    ffprobe: Path,
    baseline_file: Path,
    source_archive: Path,
    candidate_image: str,
    output_dir: Path,
) -> dict[str, object]:
    """Create candidate-bound source, inventory, delta, and probe evidence."""
    validate_candidate_image(candidate_image)
    verify_sha256(source_archive, FFMPEG_SOURCE_SHA256)
    baseline_text = baseline_file.read_text(encoding="utf-8")
    baseline = {category: parse_capabilities(baseline_text, category) for category in CAPABILITY_CATEGORIES}
    inventory_dir = output_dir / "inventory"
    candidate = collect_inventory(ffmpeg, inventory_dir)
    missing = compare_capabilities(baseline, candidate)
    approved = observed_approved_retirements(baseline, candidate)
    buildconf = (inventory_dir / "buildconf.txt").read_text(encoding="utf-8")
    configure_arguments = [line.strip() for line in buildconf.splitlines() if line.strip().startswith("--")]
    if not configure_arguments:
        raise CandidateError("candidate build configuration is missing")
    report = {
        "candidate_image": candidate_image,
        "source": {"name": source_archive.name, "sha256": FFMPEG_SOURCE_SHA256},
        "configure_arguments": configure_arguments,
        "capability_counts": {category: len(values) for category, values in candidate.items()},
        "capabilities": {category: sorted(values) for category, values in candidate.items()},
        "approved_retirements": {category: sorted(values) for category, values in approved.items()},
        "missing_capabilities": {category: sorted(values) for category, values in missing.items()},
        "probes": run_synthetic_probes(ffmpeg, ffprobe, output_dir / "media"),
        "compatible": not missing,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate-evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the fixed candidate evaluation and preserve its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/tldw-ffmpeg9/bin/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("/opt/tldw-ffmpeg9/bin/ffprobe"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--candidate-image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_candidate(
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        baseline_file=args.baseline,
        source_archive=args.source_archive,
        candidate_image=args.candidate_image,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))
    if not report["compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
