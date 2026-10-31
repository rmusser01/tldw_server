# Text-To-Speech

## Overview
Use of functions for individual services.
Function for each service, streaming & non-streaming.
Non-streaming will return a file, streaming will return a stream.
Use of temporary files for storage.
Use of pydub for audio manipulation.
Use of pydub for audio merging.

```
https://github.com/Enemyx-net/VibeVoice-ComfyUI
https://www.reddit.com/r/LocalLLaMA/comments/1n8flne/how_can_i_reduce_the_first_chunk_size_in/
 To reduce it, you’ll need to look at the inference/streaming loop (usually where the decoder pushes tokens into the vocoder). Try adjusting the buffer size or flushing frequency so that the model yields audio frames every few hundred tokens instead of waiting for a long sequence. Look for parameters like chunk_size, frame_stride, or flush_interval in the streaming code. Setting those lower should let you get ~1 second of audio out first, then continue smoothly.
```

https://github.com/resemble-ai/chatterbox
https://github.com/liampetti/fulloch
https://huggingface.co/OpenMOSS-Team/models?sort=created
https://github.com/gokhaneraslan/chatterbox-finetuning

https://github.com/BovineOverlord/Derpy-Turtle-The-Kokoro-Trainer
https://www.swyx.io/podcast



https://scenema.ai/audio
https://github.com/ScenemaAI/scenema-audio
https://huggingface.co/ScenemaAI/scenema-audio


https://github.com/k2-fsa/OmniVoice
https://huggingface.co/tmdarkbr/echo-tts-gguf
https://github.com/Cirius0310/echo-tts-cpp
https://github.com/Cirius0310/echo-tts-cpp/tree/master/examples
https://github.com/pkalogiros/audiomass
https://github.com/Zyphra/ZONOS2
https://huggingface.co/Zyphra/ZONOS2
https://openmoss.ai/MOSS-TTS-Nano-Demo/
https://github.com/ekwek1/soprano
https://github.com/OpenMOSS/MOSS-TTS



https://huggingface.co/nineninesix/kani-tts-2-en
https://github.com/nineninesix-ai/kani-tts-2
https://www.stavros.io/posts/i-made-a-voice-note-taker/
https://github.com/skylord123/pebble-home-assistant-ws
https://community.home-assistant.io/t/my-journey-to-a-reliable-and-enjoyable-locally-hosted-voice-assistant/944860
https://github.com/ekwek1/soprano?tab=readme-ov-file#installation
https://huggingface.co/kyutai
https://huggingface.co/Echo9Zulu/Kokoro-82M-FP16-OpenVINO
https://github.com/OpenBMB/VoxCPM
https://github.com/jordandare/echo-tts
https://github.com/nari-labs/dia2
https://github.com/NVIDIA/personaplex?tab=readme-ov-file
https://github.com/kyutai-labs/pocket-tts
https://github.com/supertone-inc/supertonic
https://github.com/k2-fsa/ZipVoice



Gemma
    https://ai.google.dev/gemma/docs/get_started
    https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma








### Pocket TTS Voice cloning
```python
"""Run voice cloning with local weights."""
import sys
sys.path.insert(0, 'pocket-tts')

from pathlib import Path
import soundfile as sf

from pocket_tts.utils.config import load_config
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import (
    DEFAULT_TEMPERATURE,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_EOS_THRESHOLD,
)

print('Loading config...')
config = load_config(Path('local_config.yaml'))

print('Loading model with local weights...')
model = TTSModel._from_pydantic_config_with_weights(
    config,
    temp=DEFAULT_TEMPERATURE,
    lsd_decode_steps=DEFAULT_LSD_DECODE_STEPS,
    noise_clamp=DEFAULT_NOISE_CLAMP,
    eos_threshold=DEFAULT_EOS_THRESHOLD,
)

print('Loading voice from WAV file...')
voice_state = model.get_state_for_audio_prompt('drank_virtual_encounters_are_now_available_in_the_arena.wav')

print('Generating audio...')
audio = model.generate_audio(voice_state, 'Hello, this is a test of the voice cloning feature. The quick brown fox jumps over the lazy dog.')

sf.write('output_cloned.wav', audio, 24000)
print('Done! Saved to output_cloned.wav')

```

https://generativeai.pub/glm-4-voice-9b-real-time-multilingual-voice-conversation-ai-install-locally-in-minutes-ce2fcd6c8fd8

Commercial
    https://deepgram.com/
    https://docs.cartesia.ai/get-started/overview


https://github.com/ananta-sj/ReFlow-Studio

https://github.com/bytedance/MegaTTS3
    https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT/files

https://github.com/High-Logic/Genie-TTS/tree/master/src/genie_tts
https://metis-demo.github.io/#metis-tts
https://www.reddit.com/r/LocalLLaMA/comments/1n0bhd7/microsoft_vibevoice_tts_opensourced_supports_90/
https://github.com/Extraltodeus/Quick-slap-tkinter-dia/blob/main/README.md
https://github.com/RobViren/kokovoicelab
https://github.com/psdwizzard/chatterbox-Audiobook
https://sh.itjust.works/post/40062592
https://kyutai.org/next/stt
https://github.com/eel-brah/kokorodoki
https://kyutai.org/next/unmute
https://speech.fish.audio/inference/#2-generate-semantic-tokens-from-text
https://github.com/fishaudio/fish-speech
https://huggingface.co/fishaudio/openaudio-s1-mini
https://github.com/MaTeZZ/ComfyUI-HiggsAudio_2
https://www.youtube.com/watch?v=CDJ5BuGKW6s
https://voicebox.sh/
https://github.com/pipecat-ai/nemotron-january-2026
https://departmentofproduct.substack.com/p/new-ai-voice-features-and-use-cases?triedRedirect=true
https://ai.google.dev/gemini-api/docs/speech-generation
https://github.com/stepfun-ai/Step-Audio2
https://github.com/fastrepl/hyprnote
https://github.com/travisvn/chatterbox-tts-api
https://huggingface.co/maya-research/maya1
https://github.com/thxxx/harper?tab=readme-ov-file
https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/tree/main
https://github.com/marhensa/vibevoice-realtime-openai-api



Dataset Creation
    https://github.com/taresh18/TTSizer



https://huggingface.co/jordand/echo-tts-base
https://github.com/KevinAHM/echo-tts-api

https://huggingface.co/nvidia/audio-flamingo-3-chat

https://www.reddit.com/r/LocalLLaMA/comments/1l3c8is/grmrv3_a_set_of_models_for_reliable_grammar/

https://github.com/Jeremy-Harper/chatterboxPro
https://kyutai.org/next/stt
https://huggingface.co/kyutai/stt-2.6b-en
https://github.com/rsxdalv/chatterbox/tree/fast
https://arxiv.org/abs/2506.11130
https://huggingface.co/spaces/webml-community/conversational-webgpu/tree/main
https://github.com/huggingface/transformers.js-examples/tree/main/conversational-webgpu
https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi
https://github.com/devnen/Chatterbox-TTS-Server
https://github.com/PyAV-Org/PyAV
https://github.com/santinic/audiblez

https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b
https://github.com/ekwek1/soprano
https://github.com/hkjarral/Asterisk-AI-Voice-Agent
https://huggingface.co/blog/YatharthS/llm-tts-models
    https://github.com/ysharma3501/MiraTTS?tab=readme-ov-file
    https://huggingface.co/YatharthS/MiraTTS/tree/main

https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
https://huggingface.co/ResembleAI/chatterbox-turbo
https://github.com/diodiogod/TTS-Audio-Suite

https://decodingml.substack.com/p/deploying-agents-as-real-time-apis
https://github.com/Nighthawk42/mOrpheus
https://github.com/RobViren/kvoicewalk
https://github.com/w-okada/voice-changer
https://github.com/Plachtaa/seed-vc
https://github.com/resemble-ai/chatterbox
https://github.com/lamm-mit/PDF2Audio
https://huggingface.co/Zyphra/Zonos-v0.1-hybrid
https://github.com/QuantiusBenignus/blahst/
https://jordandarefsky.com/blog/2024/parakeet/
https://www.reddit.com/r/LocalLLaMA/comments/1kvknlo/speechless_speech_instruction_training_without/
https://arxiv.org/abs/2505.17417
https://pub.towardsai.net/%EF%B8%8F-building-a-local-speech-to-text-system-with-parakeet-tdt-0-6b-v2-ebd074ba8a4c
https://www.tamingllms.com/notebooks/input.html#generating-long-form-content
https://github.com/collabora/WhisperLive
https://github.com/Lex-au/Vocalis?tab=readme-ov-file
https://github.com/Anjok07/ultimatevocalremovergui
https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
https://github.com/devnen/Dia-TTS-Server
https://github.com/Lex-au/Orpheus-FastAPI
https://github.com/MoonshotAI/Kimi-Audio
https://huggingface.co/collections/stepfun-ai/step-audio-67b33accf45735bb21131b0b
https://github.com/mbzuai-oryx/LLMVoX
https://github.com/abus-aikorea/voice-pro
https://github.com/rmusser01/LiveBench/tree/llamacpp-api
https://colab.research.google.com/github/ReisCook/Voice_Extractor_Colab/blob/main/Voice_Extractor_Colab.ipynb
https://github.com/ReisCook/Voice_Extractor
https://github.com/santinic/audiblez
https://github.com/Capsize-Games/airunner
https://github.com/Lex-au/Vocalis
    https://www.reddit.com/r/LocalLLaMA/comments/1jy1x1b/comment/mmvm73e/
https://pypi.org/project/pvporcupine/
https://github.com/EvolvingLMMs-Lab/Aero-1
https://github.com/kunwar-vikrant/aivy
https://huggingface.co/lmms-lab/Aero-1-Audio
https://github.com/davidbrowne17/csm-streaming
https://github.com/senstella/csm-mlx
https://github.com/davidbrowne17/csm-streaming
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/export.html
https://github.com/yukiarimo/hanasu
https://codingwithcody.com/2025/04/02/containerized-voice-identification/
https://github.com/NullMagic2/SoftWhisper
https://github.com/thomasgauthier/csm-hf/
https://github.com/zeropointnine/tts-toy
https://github.com/edwko/OuteTTS
https://index-tts.github.io/#section2
https://arxiv.org/abs/2504.03289
https://github.com/yukiarimo/hanasu
https://github.com/taresh18/conversify-speech
https://github.com/playht/playdiffusion
https://huggingface.co/senstella/csm-expressiva-1b
https://github.com/KartDriver/mira_converse/blob/main/server/src/server_audio_core.py
https://github.com/zenforic/csm-multi
https://github.com/isaiahbjork/orpheus-tts-local
https://github.com/tarun7r/Vocal-Agent
https://huggingface.co/DavidBrowne17/Muchi
https://github.com/davidbrowne17/chatterbox-streaming
https://webrtchacks.com/the-unofficial-guide-to-openai-realtime-webrtc-api/
https://github.com/petermg/Chatterbox-TTS-Extended
https://github.com/murtaza-nasir/speakr
https://github.com/rhulha/EchoMate

https://github.com/rhulha/EchoMate_ServerSide/
https://github.com/zeropointnine/tts-audiobook-tool
https://github.com/canopyai/Orpheus-TTS
    https://github.com/canopyai/Orpheus-TTS
    https://github.com/isaiahbjork/orpheus-tts-local/
    https://github.com/Lex-au/Orpheus-FastAPI
webrtc
https://huggingface.co/spaces/fastrtc/talk-to-claude/blob/main/app.py
https://github.com/freddyaboulton/fastrtchttps://github.com/rhulha/Speech2Speech
https://github.com/boson-ai/higgs-audio/blob/main/examples/serve_engine/run_hf_example.py
https://allenai.org/blog/olmoasr
Canary
    https://huggingface.co/nvidia/canary-1b-flash
    https://huggingface.co/nvidia/canary-180m-flash

VibeVoice
    https://microsoft.github.io/VibeVoice/
    https://github.com/microsoft/VibeVoice
    https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f


### Services
- Google Cloud Text-to-Speech
    - https://cloud.google.com/text-to-speech/docs/ssml


### Benchmarks
https://huggingface.co/blog/big-bench-audio-release
    https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio
https://artificialanalysis.ai/models/speech-to-speech
https://github.com/Picovoice/speech-to-text-benchmark?tab=readme-ov-file
https://sakshi113.github.io/mmau_homepage/


### Streaming
https://github.com/KoljaB/RealtimeSTT/tree/master
https://github.com/amanvirparhar/weebo/tree/main


### Other
https://wave-pulse.io/
https://arxiv.org/abs/2501.01384
https://github.com/stepfun-ai/Step-Audio2
https://huggingface.co/stepfun-ai/Step-Audio-2-mini
    https://arxiv.org/abs/2507.16632



Voice Cleanup
    https://www.lalal.ai/voice-cleaner/
    https://github.com/kwatcharasupat/bandit-v2
    https://github.com/resemble-ai/resemble-enhance
    https://github.com/resemble-ai/resemble-enhance/issues/69

### Link Dump:
https://huggingface.co/NexaAIDev/Qwen2-Audio-7B-GGUF
https://github.com/shagunmistry/NotebookLM_Alternative/tree/main/ai_helper
https://docs.cartesia.ai/get-started/make-an-api-request
https://arxiv.org/abs/2412.18566
https://github.com/pipecat-ai/pipecat/tree/a367a038f1a3967292b5de5b43b8600a82a73fb6?tab=readme-ov-file
https://github.com/Purfview/whisper-standalone-win
https://github.com/ictnlp/LLaMA-Omni
https://github.com/thepersonalaicompany/amurex
https://huggingface.co/Vikhrmodels/salt-asr_wav-uni_1_tts_wav-uni_1-12k
https://levelup.gitconnected.com/build-a-real-time-ai-voice-and-video-chat-app-with-function-calling-by-gemini-2-0-49599a48fbe9?gi=c894f6c092be
https://huggingface.co/papers/2501.02832

Blogposts
    https://blog.duy.dev/build-your-own-voice-assistant-and-run-it-locally/
    https://www.twilio.com/en-us/blog/twilio-openai-realtime-api-launch-integration

Full Pipelines
    https://github.com/lhl/voicechat2
    https://github.com/eustlb/speech-to-speech
    https://github.com/dnhkng/GlaDOS
    https://github.com/mezbaul-h/june
    https://github.com/matatonic/openedai-speech
    https://github.com/pixelpump/Ai-Interview-Assistant-Python
    https://github.com/huggingface/speech-to-speech
    https://github.com/harvestingmoon/S2S
    https://github.com/livekit/agents
    https://github.com/ictnlp/LLaMA-Omni
    https://github.com/smellslikeml/dolla_llama/blob/main/app/app.py

Voicw2Voicw Models
    https://github.com/Standard-Intelligence/hertz-dev


Editing Suites
    https://github.com/abus-aikorea/voice-pro



ElevenLabs
    https://github.com/elevenlabs/elevenlabs-examples/blob/main/examples/text-to-speech/python/text_to_speech_file.py
    https://elevenlabs.io/docs/api-reference/text-to-speech
    https://elevenlabs.io/docs/developer-guides/how-to-use-tts-with-streaming
Alltalk
AlwaysReddy - (uses Piper)
Amphion
    https://github.com/open-mmlab/Amphion
    https://huggingface.co/amphion/Vevo
        https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md
        https://openreview.net/pdf?id=anQDiQZhDP
    https://versavoice.github.io/
Auralis
    https://github.com/astramind-ai/Auralis
    https://www.astramind.ai/post/auralis
Bark
    https://github.com/suno-ai/bark
Cartesia
    https://docs.cartesia.ai/get-started/make-an-api-request
Chat TTS
    https://huggingface.co/2Noise/ChatTTS
    https://chattts.com/#Demo
    https://github.com/2noise/ChatTTS
Coqui TTS
    https://github.com/idiap/coqui-ai-TTS
    https://huggingface.co/spaces/coqui/xtts/blob/main/app.py
    https://github.com/coqui-ai/TTS
CosyVoice2
    https://funaudiollm.github.io/cosyvoice2/
Daswers XTTS GUI
Diva
    https://huggingface.co/spaces/WillHeld/diva-audio
F5 TTS
    https://github.com/SWivid/F5-TTS
Fish-Speech
    https://github.com/fishaudio/fish-speech/tree/main
    https://github.com/fishaudio/fish-speech/blob/main/Start_Agent.md
    https://huggingface.co/fishaudio/fish-agent-v0.1-3b/tree/main
Gemini
    https://ai.google.dev/gemini-api/docs#rest
    https://ai.google.dev/gemini-api/docs/models/gemini-v2
    https://github.com/google-gemini/cookbook/blob/main/quickstarts/Audio.ipynb
    https://github.com/livekit/agents/blob/main/examples/voice-pipeline-agent/gemini_voice_agent.py
Google
    https://github.com/google-gemini/cookbook/tree/main/gemini-2
    https://discuss.ai.google.dev/t/how-does-one-get-access-to-the-api-for-tts-features-of-gemini-2-0/53925/15
    https://illuminate.google.com/home?pli=1
GLM-4-Voice
    https://github.com/THUDM/GLM-4-Voice/blob/main/README_en.md
    https://github.com/THUDM/GLM-4-Voice/tree/main
    https://huggingface.co/cydxg/glm-4-voice-9b-int4/blob/main/README_en.md
GPT-SoviTTS
    https://github.com/cpumaxx/sovits-ff-plugin
    https://github.com/JarodMica/GPT-SoVITS-Package
Kokoro
    https://www.reddit.com/r/LocalLLaMA/comments/1jcufi6/improvements_to_kokoro_tts_v10/
    https://github.com/nazdridoy/kokoro-tts
    https://github.com/nazdridoy/kokoro-tts
    https://github.com/wwang1110/kokoro_batch

lina TTS
    https://github.com/theodorblackbird/lina-speech/blob/main/InferenceLina.ipynb
    https://github.com/theodorblackbird/lina-speech
LLaSA
    https://huggingface.co/blog/srinivasbilla/llasa-tts
    https://github.com/zhenye234/LLaSA_training
    https://huggingface.co/spaces/srinivasbilla/llasa-3b-tts
    https://github.com/nivibilla/local-llasa-tts
LMNT
MaskGCT
    https://maskgct.github.io/#emotion-samples
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/README.md
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/maskgct_demo.ipynb
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/maskgct_inference.py
    https://huggingface.co/amphion/MaskGCT
MeloTTS
    https://github.com/myshell-ai/MeloTTS
    https://huggingface.co/myshell-ai
Mimic
    https://github.com/MycroftAI/mimic3
MoonShine
    https://huggingface.co/onnx-community/moonshine-base-ONNX
    https://huggingface.co/spaces/webml-community/moonshine-web
    https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web
Neuro-sama
    https://github.com/JarodMica/open-neruosama
Open-LLM-VTuber
OpenVoice
    https://github.com/myshell-ai/OpenVoice
Outte
    https://github.com/edwko/outetts
    https://github.com/edwko/OuteTTS/pull/46#issuecomment-2527817670
    https://huggingface.co/collections/OuteAI/outetts-03-6786b1ebc7aeb757bc17a2fa
    https://github.com/edwko/OuteTTS/blob/main/docs/interface_v2_usage.md
Parler
    https://github.com/huggingface/parler-tts
Paroli - Streaming mode implementation of the Piper TTS with RK3588 NPU acceleration support.
Phi-4
    https://huggingface.co/microsoft/Phi-4-multimodal-instruct
PiperTTS - A fast, local neural text to speech system that is optimized for the Raspberry Pi 4.
    https://github.com/rhasspy/piper
    https://github.com/rhasspy/piper/issues/644
    https://github.com/rhasspy/piper/discussions/326#discussioncomment-7935208
    https://noerguerra.com/how-to-read-text-aloud-with-piper-and-python/
    https://ssamjh.nz/create-custom-piper-tts-voice/
    https://www.trycatchdebug.net/news/1377664/realtime-tts-with-pipertts-and-openai
    https://huggingface.co/rhasspy/piper-voices/tree/main
    https://k2-fsa.github.io/sherpa/onnx/tts/piper.html
    https://github.com/rhasspy/piper/blob/master/TRAINING.md
    https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main

PiperUI
RVC
    https://www.youtube.com/watch?v=Q8du7n0vgfU
Seed-VC
    https://github.com/Plachtaa/seed-vc
Sherpa ONNX
    https://github.com/k2-fsa/sherpa-onnx
Silero
SpeechT5
    https://github.com/microsoft/SpeechT5
SoundStorm
    https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/
    https://github.com/lucidrains/soundstorm-pytorch
Styletts2
    https://www.youtube.com/watch?v=dCmAbcJ5v5k
Tortoise TTS
VallE-X
VITA
    https://github.com/VITA-MLLM/VITA
VoiceCraft -
xtts
xtts2
Yapper
    https://github.com/n1teshy/yapper-tts
YourTTS
    https://github.com/Edresson/YourTTS


https://docs.inferless.com/cookbook/serverless-customer-service-bot
https://github.com/dnhkng/GlaDOS



TTS
    https://github.com/KoljaB/RealtimeTTS
    https://si.inc/hertz-dev/

101
    https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-for-different-use-cases
    https://clideo.com/resources/what-is-tts
    https://pub.towardsai.net/the-ultimate-guide-to-audio-processing-principles-techniques-and-applications-7724efea00e8
    RVC 101
        https://gudgud96.github.io/2024/09/26/annotated-rvc/

Datasets(?)
    https://voice-models.com/



Podcastfy
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/tts/base.py
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/text_to_speech.py
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/content_generator.py

Models
      https://huggingface.co/NexaAIDev/Qwen2-Audio-7B-GGUF

Merging Audio
    https://github.com/jiaaro/pydub



TTS Pipeline
    https://www.astramind.ai/post/auralis


https://github.com/cpumaxx/sovits-ff-plugin
https://github.com/satvik314/opensource_notebooklm/blob/main/opensource_notebooklm.ipynb


Train using: https://github.com/Mangio621/Mangio-RVC-Fork/releases,
import the .pth into https://huggingface.co/wok000/vcclient000/tree/main to convert your voice in near real time with about a .25s delay



Train using: https://github.com/Mangio621/Mangio-RVC-Fork/releases,
import the .pth into https://huggingface.co/wok000/vcclient000/tree/main to convert your voice in near real time with about a .25s delay






https://github.com/RVC-Boss/GPT-SoVITS
https://www.bilibili.com/video/BV11iiNegEGP/
https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)
https://rentry.org/GPT-SoVITS-guide
https://rentry.org/GPT-SoVITS-guide
It's just the 3 buttons (speech-to-text, ssl, semantics) and then training.

The default training settings on the gradio UI are fine but I save epoch 12-16-24 on SoVITS for testing as that's the sweet spot range.

Next thing that matters a lot is the ref audio you pick, and you can also drop your entire dataset into the "multiple references to average tone" box, which can improve the voice

Only thing I changed was remove the space at the beginning of each lines in your list file

(Look at batch size/ list file)

And make sure you get the latest version https://github.com/RVC-Boss/GPT-SoVITS/releases



```
import asyncio
import base64
import json
import numpy as np
import os
import websockets
import wave
import contextlib
import pygame
from IPython.display import display, Markdown

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

voices = {"Puck", "Charon", "Kore", "Fenrir", "Aoede"}

# --- Configuration ---
MODEL = 'models/gemini-2.0-flash'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
HOST = 'generativelanguage.googleapis.com'
URI = f'wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'

# Audio parameters
WAVE_CHANNELS = 1  # Mono audio
WAVE_RATE = 24000
WAVE_SAMPLE_WIDTH = 2


@contextlib.contextmanager
def wave_file(filename, channels=WAVE_CHANNELS, rate=WAVE_RATE, sample_width=WAVE_SAMPLE_WIDTH):
    """Context manager for creating and managing wave files."""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            yield wf
    except wave.Error as e:
        print(f"{RED}Error opening wave file '{filename}': {e}{RESET}")
        raise


async def audio_playback_task(file_name, stop_event):
    """Plays audio using pygame until stopped."""
    print(f"{BLUE}Starting playback: {file_name}{RESET}")
    try:
        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            await asyncio.sleep(0.1)
    except pygame.error as e:
        print(f"{RED}Pygame error during playback: {e}{RESET}")
    except Exception as e:
        print(f"{RED}Unexpected error during playback: {e}{RESET}")
    finally:
        print(f"{BLUE}Playback complete: {file_name}{RESET}")


async def generate_audio(ws, text_input: str, voice_name="Kore") -> None:
    """
    Sends text input to the Gemini API, receives an audio response, saves it to a file, and plays it back.
    Relies on the server to maintain the session history.
    """
    pygame.mixer.init()  # Initialize pygame mixer

    msg = {
        "client_content": {
            "turns": [{"role": "user", "parts": [{"text": text_input}]}],
            "turn_complete": True,
        }
    }
    await ws.send(json.dumps(msg))

    responses = []
    async for raw_response in ws:
        response = json.loads(raw_response.decode())
        server_content = response.get("serverContent")
        if server_content is None:
            break

        model_turn = server_content.get("modelTurn")
        if model_turn:
            parts = model_turn.get("parts")
            if parts:
                for part in parts:
                    if "inlineData" in part and "data" in part["inlineData"]:
                        pcm_data = base64.b64decode(part["inlineData"]["data"])
                        responses.append(np.frombuffer(pcm_data, dtype=np.int16))

        turn_complete = server_content.get("turnComplete")
        if turn_complete:
            break

    if responses:
        display(Markdown(f"{YELLOW}**Response >**{RESET}"))
        audio_array = np.concatenate(responses)
        file_name = 'output.wav'
        with wave_file(file_name) as wf:
            wf.writeframes(audio_array.tobytes())
        stop_event = asyncio.Event()
        try:
            await audio_playback_task(file_name, stop_event)
        except Exception as e:
            print(f"{RED}Error during audio playback: {e}{RESET}")
    else:
        print(f"{YELLOW}No audio returned{RESET}")
    pygame.mixer.quit()  # clean up pygame mixer


async def main():
    print(f"{GREEN}Available voices: {', '.join(voices)}{RESET}")
    default_voice = "Kore"
    print(f"{GREEN}Default voice is set to: {default_voice}, you can change it in the code{RESET}")

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": default_voice  # Set voice
                }
            }
        }
    }

    async with websockets.connect(URI) as ws:

        async def setup(ws) -> None:
            await ws.send(
                json.dumps(
                    {
                        "setup": {
                            "model": MODEL,
                            "generation_config": config,
                        }
                    }
                )
            )

            raw_response = await ws.recv(decode=False)
            setup_response = json.loads(raw_response.decode("ascii"))
            print(f"{GREEN}Connected: {setup_response}{RESET}")

        await setup(ws)
        while True:
            text_prompt = input(f"{YELLOW}Enter your text (or type 'exit' to quit): {RESET}")
            if text_prompt.lower() == "exit":
                break

            try:
                await generate_audio(ws, text_prompt, default_voice)
            except Exception as e:
                print(f"{RED}An error occurred: {e}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
```

------------------------------------------------------------------------------------------------
#### GPT-SoVITS

- [GPT-SoVITS](f)
- [GPT-SoVITS-guide rentry.org](https://rentry.org/GPT-SoVITS-guide)
- Setup Guide: https://ai-hub-docs.vercel.app/tts/gpt-sovits/


GPT-SoviTTS
    https://levelup.gitconnected.com/great-api-design-comprehensive-guide-from-basics-to-best-practices-9b4e0b613a44?source=home---------56-1--------------------0fc48da7_5ce6_48ca_92d2_260680a20318-------3
    https://rentry.org/GPT-SoVITS-guide
    https://github.com/RVC-Boss/GPT-SoVITS
    https://github.com/cpumaxx/sovits-ff-plugin
    https://github.com/HanxSmile/Simplify-GPT-SoVITS
    https://github.com/lrxwisdom001/GPT-SoVITS-Novels/tree/main/voice_synthesis
    openneurosama - https://github.com/JarodMica/open-neruosama/blob/master/main.py
    https://huggingface.co/cpumaxx/SoVITS-anime-mini-tts

https://tts.x86.st/
Finetuning is very quick (about 5 minutes). Captioning of audio was automated with faster-whisper (it is required that the audio is captioned).
With the default batch size of 12, training takes 9.5~ GB.
Batch Size Scales linearly with VRAM. 1 batch size = 4 GB of VRAM.

Inference
    https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/inference_cli.py
    No WebUI Inference on Colab: https://colab.research.google.com/drive/1gC1lRxuOh4qW8Yz5TA10BEUPR28nJ3VR
    Training on Colab: https://colab.research.google.com/drive/1NQGKXYxJcJyTPnHsSyusTdD0l4IdMS37#scrollTo=nhyKqVwcPnvz
    No WebUI Training on Colab: https://colab.research.google.com/drive/1LmeM8yUyT9MTYF8OXc-NiBonvdh6hII6

Datasets
    https://ai-hub-docs.vercel.app/rvc/resources/datasets/
    https://mvsep.com/en

API
    https://github.com/cpumaxx/sovits-ff-plugin

Comfyui integration
    https://github.com/heshengtao/comfyui_LLM_party

- **101**
    - F
- **Setup**
    - F
- **Training**
    - F
- **Inference**
    - F
- **Fine-Tuning**
    - F
------------------------------------------------------------------------------------------------



------------------------------------------------------------------------------------------------
#### Kokoro
https://huggingface.co/hexgrad/Kokoro-82M

    https://www.reddit.com/r/LocalLLaMA/comments/1hyf1pf/comment/m6m86zm/?context=3


Eventually migrate to using: https://github.com/thewh1teagle/kokoro-onnx
- **101**
    - https://kokorotts.org/
    - https://huggingface.co/hexgrad/Kokoro-82M
    - F
    - Onnx Model: https://github.com/thewh1teagle/kokoro-onnx
- **Setup**
    1. f
    2. f
    3. f
    4. f
    5. f
- **Inference**
- **API**
- Finetuning
  - https://huggingface.co/hexgrad/Kokoro-82M/discussions/19
- **Examples**
    - https://github.com/thewh1teagle/kokoro-onnx/tree/main/examples
    - https://github.com/ugotworms/professor-kokoro-radio
    - https://github.com/remsky/Kokoro-FastAPI
    - https://github.com/kaminoer/KokoDOS/tree/main/glados

------------------------------------------------------------------------------------------------




------------------------------------------------------------------------------------------------
### Dataset Creation/Curation
https://voiceguide.arimil.com/


------------------------------------------------------------------------------------------------
