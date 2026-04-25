# TTS Module Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying and configuring the Text-to-Speech (TTS) module in production environments, including support for voice cloning, multiple providers, and transcription services.

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **RAM**:
  - 4GB for API-only providers
  - 8GB for Kokoro (ONNX)
  - 4GB for KittenTTS
  - 16GB for Higgs/Chatterbox/Dia
  - 32GB for VibeVoice-7B
  - 12GB+ for IndexTTS2 (emotion + codec pipelines)
- **Storage**:
  - 1GB for code + dependencies
  - 1GB per local model (Kokoro)
  - 3-14GB per PyTorch model
- **Python**: 3.8+ (3.10 recommended)

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: CUDA 11.8+ for PyTorch models
- **VRAM**:
  - 4GB for Kokoro (optional)
  - 2GB for KittenTTS (optional; CPU works)
  - 6GB for Higgs/Chatterbox
  - 8GB for Dia
  - 16GB for VibeVoice-7B
  - 12GB for IndexTTS2 zero-shot cloning (higher for multi-speaker batches)
- **Apple Silicon**: MLX support for Parakeet (see https://github.com/senstella/parakeet-mlx)

## Prerequisites

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    git \
    curl \
    build-essential
```

#### macOS
```bash
brew install ffmpeg espeak git

# For MLX support (Apple Silicon)
pip install mlx parakeet-mlx mlx-audio
```

#### Windows
```powershell
# Install with Chocolatey
choco install ffmpeg espeak git
```

### Python Dependencies
```bash
# Core dependencies
pip install -e .

# Additional TTS dependencies
pip install \
    av \
    httpx \
    numpy \
    scipy \
    soundfile \
    librosa \
    onnxruntime \
    kokoro-onnx \
    phonemizer-fork \
    espeakng_loader \
    huggingface-hub \
    "index-tts @ git+https://github.com/index-tts/index-tts.git"

# For GPU acceleration
pip install onnxruntime-gpu  # For ONNX models
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For PyTorch

# For voice cloning support
pip install transformers accelerate

# For transcription
pip install faster-whisper openai-whisper

# For Apple Silicon (MLX)
pip install mlx parakeet-mlx mlx-audio  # Parakeet transcription + Qwen3-TTS on M1/M2/M3
```

## Configuration

### 1. Main Configuration (config.txt)

```ini
[API]
# OpenAI TTS
openai_api_key = sk-...your-key-here...
openai_model = tts-1-hd  # or tts-1 for standard quality

# ElevenLabs
elevenlabs_api_key = xi-...your-key-here...
elevenlabs_model = eleven_multilingual_v2

# Custom OpenAI-compatible endpoint
custom_openai_api_key = your-custom-key
custom_openai_api_ip = https://your-api-endpoint.com

[TTS-Settings]
# Default provider
default_tts_provider = openai  # or kokoro, higgs, chatterbox, vibevoice

# Rate limiting
tts_rate_limit = 10  # requests per minute

# Maximum input length
max_input_length = 4096

# Voice cloning
enable_voice_cloning = true
max_voice_reference_size = 10485760  # 10MB in bytes
voice_reference_min_duration = 3.0  # seconds
voice_reference_max_duration = 30.0  # seconds

[Authentication]
require_auth = true
api_token = your-secure-token-here

[STT-Settings]
# For transcription services
default_batch_transcription_model = parakeet-tdt-0.6b-v3-onnx
default_streaming_transcription_model = parakeet-tdt-0.6b-v3-onnx
nemo_model_variant = onnx  # For Parakeet ONNX
enable_mlx = true  # For Apple Silicon acceleration
```

### 2. TTS Provider Configuration (tts_providers_config.yaml)

```yaml
# Provider priority for fallback
provider_priority:
  - openai          # Primary - reliable commercial service
  - elevenlabs      # Secondary - premium quality
  - kokoro          # Tertiary - local fallback
  - higgs           # Multi-lingual specialist
  - chatterbox      # Emotion specialist
  - dia             # Dialogue specialist
  - index_tts       # Zero-shot voice cloning (local)
  - vibevoice       # Long-form specialist

# Provider configurations
providers:
  openai:
    enabled: true
    api_key: ${OPENAI_API_KEY}
    model: tts-1-hd
    timeout: 30
    max_retries: 3
    base_url: https://api.openai.com/v1

  elevenlabs:
    enabled: true
    api_key: ${ELEVENLABS_API_KEY}
    model: eleven_multilingual_v2
    stability: 0.5
    similarity_boost: 0.5
    use_speaker_boost: true

  kokoro:
    enabled: true
    use_onnx: true
    model_path: ./models/kokoro/onnx/model.onnx
    voices_json: ./models/kokoro/voices
    device: cpu  # or cuda
    phonemizer_backend: espeak

  kitten_tts:
    enabled: true
    model: KittenML/kitten-tts-nano-0.8
    model_revision: 8d6d5a1851ffd13c894c40227c888302c2a86ef7
    device: cpu
    auto_download: true
    extra_params:
      cache_dir: ./cache/kitten_tts
      clean_text: false

  higgs:
    enabled: true
    model_path: bosonai/higgs-audio-v2-generation-3B-base
    tokenizer_path: bosonai/higgs-audio-v2-tokenizer
    device: cuda
    use_fp16: true
    batch_size: 1
    # Voice cloning settings
    enable_voice_cloning: true
    min_reference_duration: 3.0
    max_reference_duration: 10.0

  chatterbox:
    enabled: true
    model_path: resemble-ai/chatterbox
    device: cuda
    use_fp16: true
    enable_watermark: true  # Perth watermarking
    target_latency_ms: 200
    # Voice cloning settings
    enable_voice_cloning: true
    min_reference_duration: 5.0
    max_reference_duration: 20.0

  dia:
    enabled: true
    model_path: nari-labs/dia
    device: cuda
    use_safetensors: true
    use_bf16: true
    auto_detect_speakers: true
    max_speakers: 5

  index_tts:
    enabled: false              # Enable after downloading checkpoints
    model_dir: checkpoints/index_tts2
    cfg_path: checkpoints/index_tts2/config.yaml
    device: cuda                # CPU works for debugging only
    use_fp16: true
    use_cuda_kernel: true
    interval_silence: 200       # Milliseconds between text segments
    max_text_tokens_per_segment: 120
    quick_streaming_tokens: 0
    # Requires voice_reference audio; see voice_mappings.clone_required

  vibevoice:
    enabled: true
    variant: 1.5B  # or 7B, 7B-Q8
    model_path: microsoft/VibeVoice-1.5B  # or vibevoice/VibeVoice-7B (official), FabioSarracino/VibeVoice-Large-Q8 (7B-Q8)
    device: cuda
    use_fp16: true
    enable_music: true
    max_speakers: 4
    # Voice cloning settings
    enable_voice_cloning: true
    min_reference_duration: 3.0
    max_reference_duration: 30.0

  qwen3_tts:
    enabled: true
    runtime: auto              # auto | upstream | mlx | remote
    model: auto
    model_path: null
    mlx_model: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
    device: mps
    tokenizer_model: Qwen/Qwen3-TTS-Tokenizer-12Hz

> Keep the `voice_mappings.generic.*.index_tts` entries set to `clone_required` so upstream services remember to include voice reference audio with every request.

# Fallback configuration
fallback:
  enabled: true
  max_attempts: 3
  retry_delay_ms: 1000
  backoff_multiplier: 2

### KittenTTS Notes

- KittenTTS requires `espeak-ng` to be installed on the host because phonemization is initialized through `espeakng_loader`.
- The provider keeps voice catalog metadata static so `/api/v1/audio/voices/catalog` does not trigger model downloads.
- If you change `model`, also update `model_revision` to the matching immutable Hugging Face commit hash.

# Circuit breaker configuration
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_calls: 3
  error_types:
    - timeout
    - connection_error
    - api_error

# Performance settings
performance:
  max_concurrent_generations: 4
  # Production default: return HTTP errors instead of embedding error bytes
  stream_errors_as_audio: false
  request_timeout: 60
  stream_chunk_size: 4096
  cache_enabled: false
  cache_ttl: 3600
```

### Qwen3-TTS Runtime Notes

- `runtime=auto` prefers `mlx` on Apple Silicon and `upstream` elsewhere.
- `runtime=mlx` supports preset-speaker synthesis in v1 and rejects uploaded `custom:<voice_id>` voices, `Base`, and `VoiceDesign`.
- `runtime=mlx` accepts `stream=true`, but streams buffered fallback chunks after local generation completes.
- `runtime=remote` reuses the standard provider `base_url` and `api_key` fields.
- `runtime=remote` advertises only conservative capabilities unless `capability_override` is set.
- Hosted Qwen backends receive Qwen-specific fields in `extra_body`, including `ref_audio_b64`, `ref_text`, `voice_clone_prompt`, `x_vector_only_mode`, and `description`.

Remote example:

```yaml
providers:
  qwen3_tts:
    enabled: true
    runtime: remote
    base_url: http://127.0.0.1:8001/v1/audio/speech
    api_key: ${QWEN_REMOTE_API_KEY}
    capability_override:
      supports_streaming: true
      supports_voice_cloning: true
      supports_emotion_control: false
      supported_modes: [custom_voice_preset, voice_clone]
      supported_voices: [Cherry, Ethan]
      supports_uploaded_custom_voices: false
```

### Apple Silicon Smoke Check

1. Install dependencies:
   ```bash
   source .venv/bin/activate
   pip install mlx mlx-audio
   ```
2. Configure the provider:
   ```yaml
   providers:
     qwen3_tts:
       enabled: true
       runtime: "mlx"
       model: "auto"
       device: "mps"
   ```
3. Start the API:
   ```bash
   source .venv/bin/activate
   python -m uvicorn tldw_Server_API.app.main:app --reload
   ```
4. Send a preset-speaker request:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen3_tts","input":"hello from Apple Silicon","voice":"Vivian","response_format":"wav","stream":false}'
   ```
5. Verify behavior:
   - preset-speaker requests return audio
   - `stream=true` returns buffered chunks instead of failing validation
   - `custom:<voice_id>` requests fail validation
   - `Base` and `VoiceDesign` requests fail validation on `runtime=mlx`
   - `/api/v1/audio/health` includes runtime metadata such as `qwen3_tts:mlx`
6. Run the gated MLX integration smoke test:
   ```bash
   source .venv/bin/activate
   TLDW_RUN_QWEN3_MLX_INTEGRATION=1 \
   QWEN3_MLX_MODEL=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
   python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_qwen3_mlx_runtime_integration.py -v
   ```

## Manual GPU Smoke Test (IndexTTS2)

1. **Environment check** - Verify CUDA is visible:
   ```bash
   python - <<'PY'
   import torch
   assert torch.cuda.is_available(), "CUDA device not detected"
   print(torch.cuda.get_device_name(0))
   PY
   ```
2. **Adapter dry run** - Use a short voice reference (5-10 s WAV) and confirm end-to-end generation (update the `voice_refs/coach.wav` path to your sample):
   ```bash
   python - <<'PY'
   import base64, asyncio
   from pathlib import Path
   from tldw_Server_API.app.core.TTS.adapters.index_tts_adapter import IndexTTS2Adapter
   from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat

   async def main():
       adapter = IndexTTS2Adapter({
           "index_tts_model_dir": "checkpoints/index_tts2",
           "index_tts_cfg_path": "checkpoints/index_tts2/config.yaml",
           "index_tts_device": "cuda",
           "index_tts_use_fp16": True,
       })
       await adapter.initialize()
       voice_bytes = Path("voice_refs/coach.wav").read_bytes()
       request = TTSRequest(
           text="IndexTTS2 smoke test on GPU.",
           voice="clone_required",
           voice_reference=voice_bytes,
           format=AudioFormat.MP3,
           stream=False,
           provider="index_tts",
       )
       response = await adapter.generate(request)
       Path("smoke-test-index-tts.mp3").write_bytes(response.audio_data)
       print("Generated:", response.sample_rate, "Hz")

   asyncio.run(main())
   PY
   ```
3. **Streaming verification** - Launch the API and request a streaming response to ensure chunk pacing:
   ```bash
   uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
   ```
   ```bash
   python - <<'PY'
   import base64, json
   from pathlib import Path
   payload = {
       "model": "index_tts",
       "input": "Streaming verification for the IndexTTS2 adapter.",
       "voice": "clone_required",
       "voice_reference": base64.b64encode(Path("voice_refs/coach.wav").read_bytes()).decode(),
       "response_format": "mp3",
       "stream": True
   }
   Path("index-tts-stream.json").write_text(json.dumps(payload))
   PY
   curl -X POST http://localhost:8000/api/v1/audio/speech \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <token>" \
     -d @index-tts-stream.json \
     --output stream-index-tts.mp3
   ```
   Monitor server logs for `IndexTTS2 adapter initialized` and confirm `stream-index-tts.mp3` plays without artifacts.

Document findings and GPU specs alongside the audio/TTS e2e suite (for example `tldw_Server_API/tests/e2e/test_audio_stt_tts_e2e.py`) to keep regressions visible.

# Logging
logging:
  level: INFO  # DEBUG for troubleshooting
  include_metrics: true
  log_file: logs/tts.log
```

## Deployment Options

### Option 1: Basic Deployment

```bash
# Clone repository
git clone https://github.com/your-repo/tldw_server.git
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Configure
cp Config_Files/Backup_Config.txt config.txt
# Edit config.txt with your settings

# Create YAML config
cat > tldw_Server_API/app/core/TTS/tts_providers_config.yaml << EOF
# Add your YAML configuration here
EOF

# Start server
python -m uvicorn tldw_Server_API.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    libespeak-ng1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install from pyproject
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Install additional TTS dependencies
RUN pip install \
    onnxruntime \
    kokoro-onnx \
    phonemizer \
    transformers \
    accelerate \
    soundfile \
    librosa

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/
COPY config.txt .

# Create Kokoro model directories (v1.0 layout)
RUN mkdir -p models/kokoro/onnx models/kokoro/voices

# Set environment variables (only needed if auto-detect fails)
# ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/audio/health || exit 1

# Run application
CMD ["uvicorn", "tldw_Server_API.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]
```

Build and run:
```bash
# Build image
docker build -t tldw-tts:latest .

# Run with volume mounts for config and models
docker run -d \
    --name tldw-tts \
    -p 8000:8000 \
    -v $(pwd)/config.txt:/app/config.txt:ro \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    tldw-tts:latest

# For GPU support
docker run -d \
    --name tldw-tts \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/config.txt:/app/config.txt:ro \
    -v $(pwd)/models:/app/models \
    --restart unless-stopped \
    tldw-tts:latest
```

### Option 3: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  tts:
    build: .
    container_name: tldw-tts
    ports:
      - "8000:8000"
    volumes:
      - ./config.txt:/app/config.txt:ro
      - ./models:/app/models
      - ./logs:/app/logs
      - ./tts_providers_config.yaml:/app/tldw_Server_API/app/core/TTS/tts_providers_config.yaml:ro
    environment:
      - PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
      - TRANSFORMERS_CACHE=/app/models
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up -d
```

### Option 4: Production with Kubernetes

```yaml
# tts-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tldw-tts
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tldw-tts
  template:
    metadata:
      labels:
        app: tldw-tts
    spec:
      containers:
      - name: tts
        image: tldw-tts:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: openai-key
        - name: ELEVENLABS_API_KEY
          valueFrom:
            secretKeyRef:
              name: tts-secrets
              key: elevenlabs-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"  # For GPU nodes
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: config
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /api/v1/audio/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/audio/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: tts-models-pvc
      - name: config
        configMap:
          name: tts-config
---
apiVersion: v1
kind: Service
metadata:
  name: tldw-tts-service
spec:
  selector:
    app: tldw-tts
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Model Setup

### Downloading Local Models

#### Kokoro (ONNX v1.0)
Recommended installer:
```bash
python Helper_Scripts/TTS_Installers/install_tts_kokoro.py
# Overwrite existing assets:
# python Helper_Scripts/TTS_Installers/install_tts_kokoro.py --force
```
Alternative helper (assets only):
```bash
python Helper_Scripts/download_kokoro_assets.py \
  --repo-id onnx-community/Kokoro-82M-v1.0-ONNX-timestamped \
  --model-path models/kokoro/onnx/model.onnx \
  --voices-dir models/kokoro/voices
```
Manual (`hf` CLI):
```bash
pip install -U "huggingface_hub"
mkdir -p models/kokoro
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped onnx/model.onnx --local-dir models/kokoro/
hf download onnx-community/Kokoro-82M-v1.0-ONNX-timestamped voices          --local-dir models/kokoro/
```
Resulting layout:
```
models/kokoro/
  onnx/model.onnx
  voices/
```

#### Higgs Audio V2
```bash
# Using the hf CLI
pip install -U "huggingface_hub"
hf download bosonai/higgs-audio-v2-generation-3B-base \
    --local-dir models/higgs
```

#### Chatterbox
```bash
# Download from Resemble AI
hf download resemble-ai/chatterbox \
    --local-dir models/chatterbox
```

#### VibeVoice
```bash
# Clone VibeVoice repository
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .

# Models auto-download on first use, or pre-download:
# 1.5B model
hf download microsoft/VibeVoice-1.5B \
    --local-dir models/vibevoice-1.5b

# 7B model (larger, better quality)
hf download vibevoice/VibeVoice-7B \
    --local-dir models/vibevoice-7b

# Optional: Community 8-bit quantized 7B variant (reduced VRAM)
hf download FabioSarracino/VibeVoice-Large-Q8 \
    --local-dir models/vibevoice-7b-q8
```

## Voice Cloning Setup

### Preparing Voice References

1. **Audio Requirements**:
   - Format: WAV or MP3
   - Sample Rate: 22.05kHz - 48kHz (will be resampled)
   - Duration: 3-30 seconds (provider-specific)
   - Quality: Clean, single speaker, minimal background noise

2. **Convert Audio** (if needed):
```bash
# Convert to WAV 24kHz mono
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav

# Trim to specific duration
ffmpeg -i input.wav -ss 0 -t 10 -ar 24000 -ac 1 reference.wav
```

3. **Test Voice Cloning**:
```python
import base64
import json
from urllib import request

# Prepare reference
with open("reference.wav", "rb") as f:
    voice_ref = base64.b64encode(f.read()).decode()

# Test with different providers
for model in ["higgs", "chatterbox", "vibevoice"]:
    payload = {
        "model": model,
        "input": "Testing voice cloning.",
        "voice": "default",
        "voice_reference": voice_ref,
        "response_format": "mp3",
    }
    req = request.Request(
        "http://localhost:8000/api/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req) as resp, open(f"clone_{model}.mp3", "wb") as f:
        f.write(resp.read())
```

## Nginx Configuration

```nginx
# /etc/nginx/sites-available/tts
upstream tts_backend {
    least_conn;
    server localhost:8000 max_fails=3 fail_timeout=30s;
    server localhost:8001 max_fails=3 fail_timeout=30s;  # If running multiple instances
    keepalive 32;
}

server {
    listen 80;
    listen [::]:80;
    server_name tts.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name tts.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/tts.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tts.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Logging
    access_log /var/log/nginx/tts_access.log;
    error_log /var/log/nginx/tts_error.log;

    # API endpoints
    location /api/v1/audio/ {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for future)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        tcp_nodelay on;
        tcp_nopush off;
        proxy_set_header X-Accel-Buffering no;

        # Timeouts for long generation
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # File upload limits (for voice references and transcription)
        client_max_body_size 25M;
        client_body_buffer_size 128k;
    }

    # Health check endpoint (no auth required)
    location /api/v1/audio/health {
        proxy_pass http://tts_backend;
        proxy_set_header Host $host;
        access_log off;  # Reduce log noise
    }

    # API documentation
    location /docs {
        proxy_pass http://tts_backend;
        proxy_set_header Host $host;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tts /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring & Logging

### Health Monitoring

```bash
# Check service health
curl http://localhost:8000/api/v1/audio/health

# Monitor with watch
watch -n 5 'curl -s http://localhost:8000/api/v1/audio/health | jq'
```

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tts'
    static_configs:
    - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Log Aggregation

```bash
# View logs
tail -f logs/tts.log | grep -E "ERROR|WARNING"

# Parse JSON logs with jq
tail -f logs/tts.log | jq 'select(.level=="ERROR")'

# Log rotation
cat > /etc/logrotate.d/tts << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 app app
}
EOF
```

### Grafana Dashboard

Create a dashboard with:
- Request rate by provider
- Response time percentiles
- Error rate by type
- Circuit breaker status
- Active requests gauge
- Voice cloning usage
- Audio format distribution

## Performance Tuning

### System Optimization

```bash
# Increase file descriptors
ulimit -n 65536

# TCP tuning for streaming
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
```

### Python Optimization

```python
# In main.py or startup
import uvloop
import asyncio

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Increase worker threads for I/O
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
```

### GPU Optimization

```bash
# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"  # Optimize for your GPU

# Monitor GPU usage
nvidia-smi dmon -s u -d 5
```

### Caching Strategy

```yaml
# In tts_providers_config.yaml
cache:
  enabled: true
  backend: redis  # or memory
  ttl: 3600
  max_size: 1000
  key_prefix: tts_
```

## Testing

### Unit Tests
```bash
# Run all TTS tests
pytest tests/TTS/ -v

# Run with coverage
pytest tests/TTS/ --cov=tldw_Server_API.app.core.TTS --cov-report=html
```

### Load Testing
```bash
# Install locust
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class TTSUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate_speech(self):
        self.client.post("/api/v1/audio/speech", json={
            "model": "kokoro",
            "input": "Load test message.",
            "voice": "af_bella"
        })

    @task
    def health_check(self):
        self.client.get("/api/v1/audio/health")
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10
```

### Integration Tests
```python
# test_integration.py
import base64
import json
from urllib import request

def test_voice_cloning_workflow():
    # Upload voice reference
    with open("test_voice.wav", "rb") as f:
        voice_ref = base64.b64encode(f.read()).decode()

    # Test with each provider
    for model in ["higgs", "chatterbox", "vibevoice"]:
        payload = {
            "model": model,
            "input": "Test message",
            "voice_reference": voice_ref,
        }
        req = request.Request(
            "http://localhost:8000/api/v1/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as resp:
            body = resp.read()
            assert resp.status == 200
            assert len(body) > 0
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Module not found" | Missing dependency | `pip install [module]` |
| "CUDA out of memory" | Model too large | Use CPU mode or smaller variant |
| "Circuit breaker open" | Provider failures | Check logs, wait for recovery |
| "Voice cloning failed" | Invalid audio format | Convert to WAV 24kHz mono |
| "Slow generation" | CPU mode | Enable GPU or use smaller model |
| "Connection refused" | Service not running | Check process, restart service |
| "Authentication failed" | Invalid token | Check config.txt API keys |

### Debug Commands

```bash
# Check service status
systemctl status tldw-tts

# Check port binding
netstat -tlnp | grep 8000

# Test model loading
python -c "from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter; print('OK')"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test audio processing
python -c "import soundfile; import librosa; print('Audio libs OK')"

# Verify espeak
espeak-ng --version

# Check disk space
df -h /app/models
```

### Enable Debug Logging

```yaml
# In tts_providers_config.yaml
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - console
    - file
  file_path: logs/tts_debug.log
```

### Performance Profiling

```python
# Add to main.py for profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# ... application runs ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(50)
```

## Security Best Practices

### API Security
1. **Always enable authentication** in production
2. **Use HTTPS** for all API endpoints
3. **Implement rate limiting** per IP and per user
4. **Validate input** length and content
5. **Sanitize** file uploads for voice references

### Voice Cloning Ethics
1. **Require consent** for voice cloning
2. **Log usage** for audit trails
3. **Enable watermarking** where available
4. **Implement usage quotas**
5. **Block malicious content** generation

### Container Security
```dockerfile
# Run as non-root user
RUN useradd -m -u 1000 ttsuser
USER ttsuser

# Use specific versions
FROM python:3.10.12-slim

# Scan for vulnerabilities
# docker scan tldw-tts:latest
```

## Backup & Recovery

### Model Backup
```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models_backup_20240831.tar.gz
```

### Configuration Backup
```bash
# Backup configs
cp config.txt config.txt.bak
cp tts_providers_config.yaml tts_providers_config.yaml.bak

# Version control configs (without secrets)
git add tts_providers_config.yaml
git commit -m "Update TTS configuration"
```

### Database Backup (if using caching)
```bash
# Redis backup
redis-cli BGSAVE

# Copy dump
cp /var/lib/redis/dump.rdb backups/redis_$(date +%Y%m%d).rdb
```

## Support & Resources

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [TTS README](./TTS-README.md)
- [Setup Guide](../../TTS-SETUP-GUIDE.md)
- [Voice Cloning Guide](./TTS-VOICE-CLONING.md)

### Model Resources
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [Higgs Audio](https://github.com/boson-ai/higgs-audio)
- [Chatterbox](https://github.com/resemble-ai/chatterbox)
- [VibeVoice](https://github.com/microsoft/VibeVoice)
- [Parakeet MLX](https://github.com/senstella/parakeet-mlx)

### Community
- Report issues on GitHub
- Join Discord for support
- Check Wiki for FAQs

---

*Last Updated: 2025-08-31*
*Version: 2.0.0*
> **IndexTTS2 models**: Clone the [index-tts](https://github.com/index-tts/index-tts) repo if you need bleeding-edge changes and run `pip install -e .`. Download checkpoints into `checkpoints/index_tts2/` (see README for file layout).
