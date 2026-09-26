#!/usr/bin/env python3
"""
Helper script to install dependencies required for the vendored Chatterbox integration.

Usage:
  python Helper_Scripts/install_chatterbox_deps.py [--with-lang]

Flags:
  --with-lang    Also install optional multilingual text-preprocessing deps

Notes:
  - This script uses pip to install from the current interpreter.
  - If you're using a virtualenv, ensure it is activated first.
"""
import argparse
import os
import subprocess
import sys

CORE = [
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "transformers>=4.51.0",
    "tokenizers>=0.13.0",
    "huggingface_hub>=0.21.0",
    "safetensors>=0.4.2",
    "librosa>=0.10.0",
    "diffusers>=0.29.0",
    "einops>=0.7.0",
    "conformer>=0.3.2",
    "s3tokenizer>=0.1.6",
]

LANG = [
    "pykakasi>=2.2.1",
    "dicta-onnx>=0.2.0; python_version >= '3.11'",
    "spacy-pkuseg>=0.0.33",
]


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-lang", action="store_true", help="install optional multilingual extras")
    args = ap.parse_args()

    # Environment-controlled flags
    if os.getenv("TLDW_SETUP_SKIP_PIP"):
        print("[chatterbox] Skipping pip installs: TLDW_SETUP_SKIP_PIP=1")
        return

    cmd = [sys.executable, "-m", "pip", "install", "-U"]
    idx = os.getenv('TLDW_SETUP_PIP_INDEX_URL')
    if idx:
        cmd += ["--index-url", idx]

    # install core deps first
    run(cmd + CORE)

    if args.with_lang:
        run(cmd + LANG)

    print("\nChatterbox dependencies installed successfully.")
    print("If you will use GPU, ensure the right torch build for your CUDA/ROCm.")


if __name__ == "__main__":
    main()
