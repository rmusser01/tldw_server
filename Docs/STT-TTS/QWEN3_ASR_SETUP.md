# Qwen3-ASR Setup Guide

## Overview

Qwen3-ASR is a state-of-the-art speech-to-text model from Alibaba's Qwen team, offering exceptional multilingual transcription capabilities with optional word-level timestamps via forced alignment.

### Key Features

| Feature | Details |
|---------|---------|
| **Languages** | 30 languages + 22 Chinese dialects |
| **Accuracy** | 1.63 WER on LibriSpeech clean (competitive with Whisper-large-v3) |
| **Chinese** | 2.71 WER on AISHELL-2 (2x better than Whisper) |
| **Timestamps** | 42.9ms average alignment accuracy (3x better than WhisperX) |
| **Singing/BGM** | Supported |

### Model Variants

| Model | Parameters | Use Case | VRAM Estimate |
|-------|------------|----------|---------------|
| **Qwen3-ASR-1.7B** (default) | 1.7B | Production quality | ~8-16GB |
| **Qwen3-ASR-0.6B** | 0.6B | Resource-constrained / high-throughput | ~2-4GB |
| **Qwen3-ForcedAligner-0.6B** | 0.9B | Word-level timestamps | ~2-4GB additional |

## Prerequisites

- Python 3.11+
- A working tldw_server installation
- FFmpeg on your system path
- For GPU: CUDA-compatible GPU with sufficient VRAM

### Required Dependencies

```bash
# Core dependencies (included in tldw_server)
pip install torch transformers soundfile
```

## Model Download

**Important**: Qwen3-ASR models must be manually downloaded before use. Auto-download is disabled by default for production stability.

### Using Hugging Face CLI

```bash
# Install HuggingFace CLI if not already installed
pip install -U "huggingface_hub[cli]"

# Download the 1.7B model (recommended for production)
hf download Qwen/Qwen3-ASR-1.7B --local-dir ./models/qwen3_asr/1.7B

# Or download the 0.6B model (for resource-constrained environments)
hf download Qwen/Qwen3-ASR-0.6B --local-dir ./models/qwen3_asr/0.6B

# Optional: Download the Forced Aligner for word-level timestamps
hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/qwen3_asr/aligner
```

### Using Python

```python
from huggingface_hub import snapshot_download

# Download 1.7B model
snapshot_download(
    repo_id="Qwen/Qwen3-ASR-1.7B",
    local_dir="./models/qwen3_asr/1.7B"
)

# Download 0.6B model
snapshot_download(
    repo_id="Qwen/Qwen3-ASR-0.6B",
    local_dir="./models/qwen3_asr/0.6B"
)

# Optional: Download Forced Aligner
snapshot_download(
    repo_id="Qwen/Qwen3-ForcedAligner-0.6B",
    local_dir="./models/qwen3_asr/aligner"
)
```

## Configuration

Edit `tldw_Server_API/Config_Files/config.txt` to enable Qwen3-ASR:

### Basic Configuration

```ini
[STT-Settings]
# Enable Qwen3-ASR
qwen3_asr_enabled = true

# Path to downloaded model (LOCAL path, required)
qwen3_asr_model_path = ./models/qwen3_asr/1.7B

# Device: cuda or cpu
qwen3_asr_device = cuda

# Data type: bfloat16, float16, or float32
qwen3_asr_dtype = bfloat16
```

### Make Qwen3-ASR the Default Provider (Optional)

```ini
[STT-Settings]
# Set as default transcriber
default_transcriber = qwen3-asr

# Enable Qwen3-ASR
qwen3_asr_enabled = true
qwen3_asr_model_path = ./models/qwen3_asr/1.7B
qwen3_asr_device = cuda
qwen3_asr_dtype = bfloat16
```

### Enable Word-Level Timestamps

```ini
[STT-Settings]
qwen3_asr_enabled = true
qwen3_asr_model_path = ./models/qwen3_asr/1.7B
qwen3_asr_device = cuda
qwen3_asr_dtype = bfloat16

# Enable forced alignment for word timestamps
qwen3_asr_aligner_enabled = true
qwen3_asr_aligner_path = ./models/qwen3_asr/aligner
```

### Full Configuration Reference

```ini
[STT-Settings]
# --- Qwen3-ASR Core Settings ---
# Enable/disable Qwen3-ASR provider
qwen3_asr_enabled = false

# LOCAL path to downloaded model (required if enabled)
qwen3_asr_model_path = ./models/qwen3_asr/1.7B

# Device: cuda or cpu
qwen3_asr_device = cuda

# Data type: bfloat16 (recommended), float16, or float32
qwen3_asr_dtype = bfloat16

# Maximum batch size for inference
qwen3_asr_max_batch_size = 32

# Maximum new tokens for generation
qwen3_asr_max_new_tokens = 4096

# Allow auto-download from HuggingFace (disabled by default)
qwen3_asr_allow_download = false

# Target sample rate for audio (default 16000 Hz)
qwen3_asr_sample_rate = 16000

# --- Qwen3-ASR Forced Aligner (for word-level timestamps) ---
# Enable forced alignment for word-level timestamps
qwen3_asr_aligner_enabled = false

# LOCAL path to forced aligner model
qwen3_asr_aligner_path = ./models/qwen3_asr/aligner

# --- Qwen3-ASR Backend (future vLLM streaming support) ---
# Backend: transformers (default) or vllm
qwen3_asr_backend = transformers

# GPU memory utilization when using vLLM backend (0.0-1.0)
qwen3_asr_vllm_gpu_memory_utilization = 0.7
```

## API Usage

### OpenAI-Compatible Endpoint

Qwen3-ASR is accessible through the standard OpenAI-compatible transcription endpoint.

#### cURL Example

```bash
# Basic transcription
curl -X POST "http://127.0.0.1:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@/path/to/audio.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "response_format=json"

# With language hint
curl -X POST "http://127.0.0.1:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@/path/to/chinese_audio.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "language=zh" \
  -F "response_format=json"

# With word-level timestamps
curl -X POST "http://127.0.0.1:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@/path/to/audio.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "timestamp_granularities=[\"word\"]" \
  -F "response_format=verbose_json"
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="YOUR_API_KEY"
)

# Basic transcription
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b",
        file=f,
        response_format="json"
    )
    print(transcript.text)

# Using the 0.6B model for faster inference
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr-0.6b",
        file=f,
        response_format="json"
    )
    print(transcript.text)

# With word timestamps
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="qwen3-asr-1.7b",
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    print(transcript.text)
    for word in transcript.words:
        print(f"  {word.word}: {word.start:.2f}s - {word.end:.2f}s")
```

### Python (requests)

```python
import requests

# Basic transcription
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/audio/transcriptions",
        headers={"X-API-KEY": "YOUR_API_KEY"},
        files={"file": f},
        data={
            "model": "qwen3-asr-1.7b",
            "response_format": "json"
        }
    )
    result = response.json()
    print(result["text"])
```

### Direct Python Usage

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR import (
    transcribe_with_qwen3_asr,
    is_qwen3_asr_available,
    get_qwen3_asr_capabilities,
)

# Check availability
if is_qwen3_asr_available():
    print("Qwen3-ASR is ready")

    # Get capabilities
    caps = get_qwen3_asr_capabilities()
    print(f"Word timestamps: {caps['word_timestamps']}")

# Transcribe audio
result = transcribe_with_qwen3_asr(
    "audio.wav",
    model_path="./models/qwen3_asr/1.7B",
    language="en",  # Optional, auto-detected if not specified
    word_timestamps=True,  # Enable word-level timestamps
)

print(f"Text: {result['text']}")
print(f"Language: {result['language']}")

if "words" in result:
    for word in result["words"]:
        print(f"  {word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

## Model Name Mapping

The following model names are accepted via the API:

| Model Name | Maps To | Notes |
|------------|---------|-------|
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B` | Production quality (default) |
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B` | High throughput / resource-constrained |
| `qwen3-asr` | Configured default | Uses `qwen3_asr_model_path` from config |
| `qwen3_asr_1.7b` | `Qwen/Qwen3-ASR-1.7B` | Underscore alias |
| `qwen3_asr_0.6b` | `Qwen/Qwen3-ASR-0.6B` | Underscore alias |

## Response Format

### Standard JSON Response

```json
{
  "text": "The transcribed text content",
  "language": "en"
}
```

### Verbose JSON Response (with timestamps)

```json
{
  "text": "Hello world",
  "language": "en",
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 2.5,
      "Text": "Hello world"
    }
  ],
  "words": [
    {"word": "Hello", "start": 0.0, "end": 0.4},
    {"word": "world", "start": 0.5, "end": 1.0}
  ]
}
```

## Supported Languages

Qwen3-ASR supports 30 languages with automatic detection:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | German | de |
| Chinese (Mandarin) | zh | French | fr |
| Japanese | ja | Spanish | es |
| Korean | ko | Italian | it |
| Arabic | ar | Portuguese | pt |
| Russian | ru | Dutch | nl |
| Vietnamese | vi | Thai | th |
| Indonesian | id | Polish | pl |
| Turkish | tr | Swedish | sv |
| Hindi | hi | Czech | cs |
| Ukrainian | uk | Greek | el |
| Hebrew | he | Hungarian | hu |
| Finnish | fi | Romanian | ro |
| Danish | da | Norwegian | no |
| Malay | ms | Filipino | fil |

Plus 22 Chinese dialects (auto-detected, not exposed in API).

## Performance Benchmarks

### Word Error Rate (WER) Comparison

| Dataset | Qwen3-ASR-1.7B | Whisper-large-v3 |
|---------|----------------|------------------|
| LibriSpeech clean | 1.63 | 1.8 |
| LibriSpeech other | 3.05 | 3.5 |
| AISHELL-2 (Chinese) | 2.71 | 5.4 |
| Common Voice 15 | 8.3 | 9.2 |

### Throughput (at concurrency 128)

- Qwen3-ASR-0.6B: 2000x real-time
- Qwen3-ASR-1.7B: 800x real-time

### Timestamp Alignment Accuracy

- Average alignment error: 42.9ms
- Comparison: WhisperX averages ~120ms

## Troubleshooting

### Common Issues

#### Model Not Found

```
BadRequestError: Qwen3-ASR model path does not exist: ./models/qwen3_asr/1.7B
```

**Solution**: Download the model first:
```bash
hf download Qwen/Qwen3-ASR-1.7B --local-dir ./models/qwen3_asr/1.7B
```

#### Qwen3-ASR Disabled

```
BadRequestError: Qwen3-ASR is disabled. Set [STT-Settings].qwen3_asr_enabled=true
```

**Solution**: Edit `config.txt` and set:
```ini
qwen3_asr_enabled = true
```

#### Word Timestamps Not Working

```
WARNING - Qwen3-ASR: word timestamps requested but aligner not enabled
```

**Solution**:
1. Download the forced aligner model
2. Enable it in config:
```ini
qwen3_asr_aligner_enabled = true
qwen3_asr_aligner_path = ./models/qwen3_asr/aligner
```

#### CUDA Out of Memory

**Solutions**:
1. Use the smaller 0.6B model:
   ```ini
   qwen3_asr_model_path = ./models/qwen3_asr/0.6B
   ```
2. Switch to CPU (slower but works):
   ```ini
   qwen3_asr_device = cpu
   ```
3. Use float16 instead of bfloat16:
   ```ini
   qwen3_asr_dtype = float16
   ```

#### Slow First Request

The first transcription request loads the model into memory, which can take 30-60 seconds depending on your hardware. Subsequent requests will be much faster.

### Health Check

```bash
curl "http://127.0.0.1:8000/api/v1/audio/transcriptions/health?model=qwen3-asr-1.7b"
```

### Debug Logging

Check server logs for detailed information:
```bash
# Look for Qwen3-ASR related messages
grep -i "qwen3" server.log
```

## Comparison with Other Providers

| Feature | Qwen3-ASR | Whisper | VibeVoice | Parakeet |
|---------|-----------|---------|-----------|----------|
| Languages | 30+ | 99 | ~50 | English |
| Chinese Quality | Excellent | Good | Good | N/A |
| Word Timestamps | Yes (aligner) | Yes | Yes | No |
| Streaming | No* | No | No | Yes |
| Diarization | No | No | Yes | No |
| VRAM (1.7B) | ~8-16GB | ~10GB | ~8GB | ~2GB |

*Streaming support planned via vLLM backend.

### When to Use Qwen3-ASR

- **Best for**: Chinese language transcription, high-accuracy multilingual content
- **Good for**: Production deployments requiring quality transcription
- **Consider alternatives for**: Real-time streaming, speaker diarization, low-resource environments

## Resources

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner on HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)

---

*Last Updated: 2026-01-29*
*Version: 1.0.0*
