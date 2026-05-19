from pathlib import Path

import pytest

REQUIRED = [
    "## Prepare",
    "## Start",
    "## Verify",
    "## First Value",
    "## Audio Path",
    "## Troubleshoot",
    "## Optional Add-ons",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_each_profile_has_required_sections() -> None:
    guides = [
        "Docs/Getting_Started/Profile_Local_Single_User.md",
        "Docs/Getting_Started/Profile_Docker_Single_User.md",
        "Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md",
    ]
    for guide in guides:
        text = Path(guide).read_text(encoding="utf-8")
        positions = []
        for heading in REQUIRED:
            _require(heading in text, f"{guide} missing required heading: {heading}")
            positions.append(text.index(heading))
        _require(
            positions == sorted(positions),
            f"{guide} should present lifecycle headings in order",
        )
        _require(
            "First-Time Audio Setup: CPU Systems" in text,
            f"{guide} should point to the CPU audio guide",
        )
        _require(
            "First-Time Audio Setup: GPU/Accelerated Systems" in text,
            f"{guide} should point to the GPU/accelerated audio guide",
        )


def test_gpu_addon_is_legacy_pointer_to_hardware_guides() -> None:
    text = Path("Docs/Getting_Started/GPU_STT_Addon.md").read_text(encoding="utf-8")
    _require("legacy pointer" in text, "GPU_STT_Addon should be marked as a legacy pointer")
    _require(
        "First-Time Audio Setup: GPU/Accelerated Systems" in text,
        "GPU_STT_Addon should point to the accelerated guide",
    )
    _require(
        "First-Time Audio Setup: CPU Systems" in text,
        "GPU_STT_Addon should point to the CPU guide",
    )


def test_first_time_audio_guides_have_core_sections() -> None:
    guide_requirements = {
        "Docs/Getting_Started/First_Time_Audio_Setup_CPU.md": [
            "## Before You Start",
            "## Step 1: Choose Your Base Setup Path",
            "## Step 2: Set the CPU STT Defaults",
            "## Step 3: Set Up the Recommended CPU TTS Path (`supertonic`)",
            "## Step 4: First Successful Verification",
            "## Troubleshooting",
            "parakeet-onnx",
            "supertonic",
        ],
        "Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md": [
            "## Before You Start",
            "## Step 1: Choose Your Base Setup Path",
            "## Step 2: Configure Accelerated STT",
            "## Step 3: Configure the Recommended TTS Path (`supertonic`)",
            "## Step 4: First Successful Verification",
            "## Troubleshooting",
            "faster-whisper",
            "parakeet-mlx",
            "supertonic",
        ],
    }
    for guide, required_content in guide_requirements.items():
        text = Path(guide).read_text(encoding="utf-8")
        for item in required_content:
            _require(item in text, f"{guide} missing expected content: {item}")
