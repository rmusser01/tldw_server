from pathlib import Path

USER_GUIDE = Path("Docs/User_Guides/STT_Benchmark_User_Guide.md")
PUBLISHED_USER_GUIDE = Path(
    "Docs/Published/User_Guides/STT_Benchmark_User_Guide.md"
)
PROTOCOL = Path("Docs/Development/STT_Benchmark_Protocol.md")
BENCHMARK_README = Path("Helper_Scripts/benchmarks/README.md")
SETUP_GUIDES = (
    Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
)


def test_audio_cpp_operator_workflow_documents_security_and_timing_contract() -> None:
    guide = USER_GUIDE.read_text(encoding="utf-8")

    required = (
        "## Optional: user-managed audio.cpp server",
        "https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md",
        "audio_cpp_enabled",
        "audio_cpp_base_url",
        "audio_cpp_default_model",
        "audio_cpp_timeout_seconds",
        "STT_AUDIO_CPP_ENABLED",
        "STT_AUDIO_CPP_BASE_URL",
        "STT_AUDIO_CPP_DEFAULT_MODEL",
        "STT_AUDIO_CPP_TIMEOUT_SECONDS",
        "GET /health",
        "GET /v1/models",
        "`audio-cpp:<model>`",
        "`audio-cpp=<model>`",
        "`--allow-network-targets`",
        "uncompressed PCM RIFF/WAVE",
        "identity remains descriptive and unresolved",
        "restart `audiocpp_server` immediately before the run",
        "Warm calls reuse tldw_server's discovery cache",
    )
    for text in required:
        assert text in guide


def test_audio_cpp_protocol_and_compact_readme_match_operator_contract() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    readme = BENCHMARK_README.read_text(encoding="utf-8")

    for text in (
        "`audio-cpp=<model>`",
        "`--allow-network-targets`",
        "identity remains unresolved",
        "uncompressed PCM RIFF/WAVE",
    ):
        assert text in protocol
        assert text in readme


def test_audio_setup_guides_link_to_optional_audio_cpp_workflow() -> None:
    expected = (
        "../User_Guides/STT_Benchmark_User_Guide.md"
        "#optional-user-managed-audiocpp-server"
    )
    for path in SETUP_GUIDES:
        text = path.read_text(encoding="utf-8")
        assert expected in text
        assert "separately managed" in text


def test_benchmark_guide_keeps_supported_target_and_artifact_lifecycle_syntax() -> None:
    guide = USER_GUIDE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    assert "--target 'parakeet-mlx" not in guide
    assert "benchmark planner currently rejects it" in guide
    assert "external=external:REPLACE_WITH_CONFIGURED_PROVIDER" in guide
    assert "model portion `external:<provider>`" in guide
    assert "`run` does not create `summary.json` or `summary.md`" in protocol
    assert "The `report` command creates or refreshes both projections." in protocol


def test_published_audio_cpp_guides_are_byte_identical_to_sources() -> None:
    mirrors = (
        (USER_GUIDE, PUBLISHED_USER_GUIDE),
        (
            SETUP_GUIDES[0],
            Path(
                "Docs/Published/Getting_Started/"
                "First_Time_Audio_Setup_CPU.md"
            ),
        ),
        (
            SETUP_GUIDES[1],
            Path(
                "Docs/Published/Getting_Started/"
                "First_Time_Audio_Setup_GPU_Accelerated.md"
            ),
        ),
    )
    for source, published in mirrors:
        assert source.read_bytes() == published.read_bytes()
