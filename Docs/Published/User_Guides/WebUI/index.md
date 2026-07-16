# WebUI And Extension Guide

This section explains what the WebUI and browser extension let you do. Use it to choose the right page, understand which surface a feature belongs to, and find deeper setup or workflow docs.

## Surfaces

| Surface | What it means |
| --- | --- |
| WebUI | The full browser application served by the Next.js app. |
| Extension options | The browser extension's full-page options UI, usually using the same shared route components as the WebUI. |
| Extension sidepanel | Compact browser-adjacent tools for chat, clipping, persona, companion, agent, and flashcard review workflows. |
| Shared UI | A route or feature implemented in the shared UI package and reused by multiple surfaces. |
| Admin/operator | Deployment, server, org, model runtime, monitoring, usage, billing, and governance pages. |
| Hosted-only | Account or billing pages that mainly apply to hosted or multi-user deployments. |
| Experimental/labs | Beta, specialized, or advanced workflows that may require extra server capability. |
| Legacy alias | A compatibility route that redirects or points users to a newer canonical page. |
| Internal QA/debug | Test or preview pages that normal users should ignore. |

## Start Here

| Need | Start with |
| --- | --- |
| Find a page or feature | [Page and feature index](Page_Feature_Index.md) |
| Connect to a server or configure auth | [Start, account, and settings](Start_Account_Settings.md) |
| Chat with models, characters, personas, or assistants | [Chat, characters, and assistants](Chat_Characters_Assistants.md) |
| Add sources, search knowledge, or manage media | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| Use transcription, TTS, or audiobook workflows | [Audio, speech, and audiobooks](Audio_Speech_Audiobooks.md) |
| Study, write, generate artifacts, or review content | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| Automate, integrate, moderate, or administer a server | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| Use browser-sidepanel workflows | [Extension sidepanel](Extension_Sidepanel.md) |
| Understand advanced, hosted, alias, or debug pages | [Experimental and specialized pages](Experimental_And_Specialized.md) |

## How To Read The Labels

`Default self-hosted` pages are expected in ordinary local or Docker deployments when the backend feature is enabled. `Advanced self-hosted` pages usually need extra server capability, operator setup, or a more specialized workflow. `Hosted-only` pages are account or billing surfaces that may not apply to local-only installations. `Legacy alias` pages exist for compatibility and point to a newer canonical route. `Internal QA/debug` pages are for testing and should not be treated as product workflows.

## Related Existing Guides

- [Current WebUI user guide](../WebUI_Extension/User_Guide.md)
- [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md)
- [Chat pages](../WebUI_Extension/Chat_Pages.md)
- [Getting started with STT and TTS](../WebUI_Extension/Getting-Started-STT_and_TTS.md)
- [Browser extension user docs](https://github.com/rmusser01/tldw_server/tree/main/apps/extension/docs)
