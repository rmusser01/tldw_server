# Speech-To-Text Documentation


## To Do
Switch to mobiusLabs model: https://github.com/SYSTRAN/faster-whisper/issues/1030
https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://github.com/EvolvingLMMs-Lab/Aero-1
https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
https://github.com/cxyfer/GeminiASR
https://github.com/senstella/parakeet-mlx
Benchmark
    https://github.com/dnhkng/GLaDOS/blob/main/src/glados/ASR/asr.py
    https://github.com/SYSTRAN/faster-whisper/blob/master/tests/test_transcribe.py


API
    https://github.com/heimoshuiyu/whisper-fastapi


Review potential use of quantized STT Models:
    * https://opennmt.net/CTranslate2/quantization.html

## Overview
- **Usage**
    1. If transcribing english audio, use [Whisper-Turbo v3](https://huggingface.co/openai/whisper-large-v3-turbo)
       * Model used in this project: [Deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
    2. If transcribing non-english audio, use [Whisper-Large distil v3](https://huggingface.co/distil-whisper/distil-large-v3)
    3. If that fails, then use [Whisper-Large v3](https://huggingface.co/openai/whisper-large-v3) -> Whisper-Large v2
- **Voice-Audio-Detection(VAD)**
    - Use VAD to detect voice activity in audio files.
    - This feature is currently not properly understood, so :shrug:
- **Speaker-Diarization**
    - Use Pyannote to determine speaker boundaries in audio files.
    - This feature is currently either implemented poorly or it's not that great at diarization.
- **Transcription**
    - Use Faster Whisper to transcribe audio files. Uses Whisper models currently
    - Faster_whisper is a re-implementation of whisper using via CTranslate2(an inference engine for Transformers models)
          - Supports both CPU and GPU + Quantization
          - https://opennmt.net/CTranslate2/quantization.html

- **Backend**
  - faster_whisper
  - Model: `WhisperModel` (User-selectable)

- **Whisper Models**
  - https://huggingface.co/distil-whisper/distil-large-v3

### Speech-to-Text
- **Flow**
    1. Convert an input file (like .m4a or .mp4) to .wav via `convert_to_wav(...)`.
    2. Transcribe with `speech_to_text(...)`, which uses Faster Whisper to generate time-stamped text segments.
    3. (Optional)Diarize the same .wav with `audio_diarization(...)`, which uses pyannote to determine speaker boundaries.
    4. (Optional)Combine them in `combine_transcription_and_diarization(...)` to match the transcription segments to the speakers based on time.
- **Key Libraries:**
    - [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) (faster_whisper.WhisperModel) for transcription.
    - [Pyannote](https://github.com/pyannote/pyannote-audio) (pyannote.audio.pipelines.speaker_diarization) for speaker diarization.
    - [FFmpeg](https://www.ffmpeg.org/) (via subprocess or os.system) to convert audio to the desired WAV format.
    - [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (for optional live recording).


### Benchmarks
- https://github.com/Picovoice/speech-to-text-benchmark
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard



### Link Dump:
STT
    https://github.com/KoljaB/RealtimeSTT
    https://github.com/southbridgeai/offmute
    https://github.com/flatmax/speech-to-text
    https://github.com/collabora/WhisperLive
    https://github.com/fedirz/faster-whisper-server
    https://github.com/ufal/whisper_streaming
    MoonShine
        https://github.com/usefulsensors/moonshine
        https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web
        https://huggingface.co/onnx-community/moonshine-base-ONNX
    https://github.com/FreedomIntelligence/Soundwave
        https://arxiv.org/abs/2502.12900
https://github.com/psdwizzard/MeetingBuddy
https://github.com/mehtabmahir/easy-whisper-ui

