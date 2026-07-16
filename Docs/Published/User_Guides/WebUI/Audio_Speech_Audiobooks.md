# Audio, Speech, And Audiobooks

Use these pages when you want to transcribe audio, generate speech, check voice readiness, or produce long-form audiobook output.

## Pages And Feature Sets

| Page or feature | Surface/status | What it lets you do | Common uses |
| --- | --- | --- | --- |
| `/speech` | WebUI, extension options | Open the speech overview and readiness surface. | Choose STT or TTS, inspect setup state. |
| `/stt` | WebUI, extension options | Run speech-to-text workflows. | File transcription, dictation, audio conversion workflows. |
| `/tts` | WebUI, extension options | Generate speech from text. | Voice previews, narration, generated audio. |
| `/audio` | Legacy alias | Compatibility route for the newer speech route family. | Old bookmarks and route compatibility. |
| `/audiobook-studio` | Advanced self-hosted | Build long-form audiobook projects with chapters, voices, generation, and output review. | Chapterized narration, audiobook production. |
| `/settings/speech` | Shared UI | Configure speech defaults and provider behavior. | STT/TTS setup, voice and transcription defaults. |

## Larger Systems

Speech workflows depend on backend capability. STT may require local models, accelerated runtimes, or provider configuration. TTS may require provider keys, local runtime setup, voice catalogs, or audio conversion support. Audiobook Studio builds on TTS plus project/chapter state, so it is more sensitive to provider readiness and long-running job behavior.

## Recovery Tips

- If transcription models do not load, confirm the backend advertises the expected STT provider and check server logs.
- If voices are empty, check `/settings/speech`, provider keys, and the TTS voice catalog endpoint.
- If `/audio` appears in old links, treat it as a compatibility route and use `/speech`, `/stt`, or `/tts` directly.
- For long audio jobs, expect background processing and status checks rather than immediate completion.

## Related Docs

- [Getting started with STT and TTS](../WebUI_Extension/Getting-Started-STT_and_TTS.md)
- [TTS getting started](../WebUI_Extension/TTS_Getting_Started.md)
- [TTS setup guide](../WebUI_Extension/TTS-SETUP-GUIDE.md)
- [Dictation strategy and settings](../WebUI_Extension/Dictation_Strategy_and_Settings.md)
- [Audio transcription API](../../API-related/Audio_Transcription_API.md)
- [TTS API](../../API-related/TTS_API.md)
- [Audio chat API notes](../../API-related/Audio_Transcription_API.md)
