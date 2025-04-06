# Audio Pipelines

## Introduction
This page serves as documentation regarding the audio processing pipelines within tldw and provides context/justification for the decisions made within them.


## Audio Pipelines



### Audio Language Models


### Diarization
Improvement: https://github.com/revdotcom/reverb

```
For diarization:
  pyannote/speaker-diarization-3.1 Does a decent job. But I’ve found it creates too many speakers and doesn’t do a perfect job.

For cleaning up diarization accuracy:
  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

The approach I’ve found best to cleanup the diarization (or replace pyannote entirely) is to generate speaker embeddings for each segment whisper generates, then group by matching the speaker embeddings.

For segment in segments:
Generate speaker embedding
For known speakers:
If match, add to array of segments for that speaker.
Else create a new entry for a new speaker.

I have found that to massively reduce the number of speakers found in an audio recording. Though if someone gets emotional or changes their speech significantly it still produces a bonus extra speaker. But far less than before.
```

### Link Dump:
https://github.com/kadirnar/whisper-plus

Papers
https://arxiv.org/pdf/2212.04356

WER:
https://pubs.aip.org/asa/jel/article/4/2/025206/3267247/Evaluating-OpenAI-s-Whisper-ASR-Performance

Transcription:
https://github.com/AugmendTech/treeseg
https://www.arxiv.org/abs/2407.12028
https://github.com/Purfview/whisper-standalone-win
https://huggingface.co/spaces/aadnk/faster-whisper-webui
https://huggingface.co/spaces/zhang082799/openai-whisper-large-v3-turbo
https://petewarden.com/2024/10/21/introducing-moonshine-the-new-state-of-the-art-for-speech-to-text/
https://github.com/usefulsensors/moonshine?tab=readme-ov-file
https://github.com/revdotcom/reverb-self-hosted/tree/main/reverb-self-hosted-api
https://github.com/SpeechColab/GigaSpeech
https://huggingface.co/nvidia/canary-1b
https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://www.futurebeeai.com/blog/breaking-down-word-error-rate
https://github.com/MahmoudAshraf97/whisper-diarization/
https://github.com/transcriptionstream/transcriptionstream
https://github.com/SYSTRAN/faster-whisper
https://whisperapi.com/word-error-rate-wer
https://github.com/oliverguhr/deepmultilingualpunctuation
https://arxiv.org/abs/2311.00430
https://github.com/PyAV-Org/PyAV[
https://github.com/snakers4/silero-vad
https://github.com/m-bain/whisperX
https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription




### Benchmarking
- **Benchmark Goals:**
    1. `Performance` - Processing time per audio file.
    2. `Accuracy` - Word Error Rate (WER).
    3. `Resource Utilization` - CPU, memory, and disk usage.
    4. `Scalability` - Performance with concurrent tasks or larger datasets.
- **Benchmark Hooks:**
    1. Timing - track processing time per function, per audio file. 
    2. Resource Utilization - track CPU, memory, and disk usage.
    3. Accuracy - WER Calculation step


https://github.com/Roon311/ASR-Evaluation