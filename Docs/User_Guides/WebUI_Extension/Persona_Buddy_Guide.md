# Persona Buddy — setup, live voice, approvals, and troubleshooting

For the independent Buddy opened from the composer or workspace's **Buddy &
Persona** control, see [Buddy and Persona management](../WebUI/Buddy_And_Persona_Management.md).
This guide covers the Persona-backed shell and its separate full Live session.

Persona Buddy is the floating companion on supported Persona-aware WebUI and
extension surfaces. It displays a Persona's active visual pack and provides a
small entry point for text sessions, feedback, and navigation to the full Live
session. The full Live view owns microphone capture, playback, and plan review.

This guide covers the **server-backed browser experience**. For the terminal
application, see the [Chatbook user guide](https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/User_Guide/index.md).
The two apps have different selection and voice controls.

## Before you start

- Run the server and WebUI, connect the WebUI to that server, and authenticate.
  Follow [Quickstart](../../Getting_Started/QUICKSTART.md) or your deployment's
  existing setup instructions.
- Make sure the server exposes Persona features and you can open **Persona
  Garden** at `/persona`.
- For conversation, configure the server's default Chat provider and model.
  Voice preparation checks that target; each submitted turn then uses the
  authenticated Chat route, which checks effective credentials (including
  applicable BYOK), access policy, moderation, and budget. Preparation alone
  does not certify that a Chat request will succeed.
- For voice, install the selected speech-to-text model and choose a TTS provider
  under **Profiles**. Use browser speech, a registered server provider, or a
  configured speech gateway. Local providers need their model assets and
  dependencies; remote providers and gateways need their endpoint and credentials.
- In **Settings → UI customization**, enable **Enable persona buddy shell**.
  Use a desktop-width WebUI window: the full WebUI hides the floating shell at
  narrow widths. The extension sidepanel has its own supported layout.

Art selection alone does not start a microphone, configure a provider, or grant
tool permission. Keep using the authenticated WebUI for protected visual assets;
a copied asset URL may require the same authentication.

## Choose Migu or another visual pack

1. Open **Persona Garden** and select or create the Persona you want to use.
2. Open the **Visuals** tab. In **Buddy builder**, select **Bundled Buddy**.
3. Find **pixel-migu** and click **Copy as draft** on that entry.
   **pixel-migu** and **Migu Marker Basic** are separate choices; availability
   depends on your server's catalog.
4. Check **Review draft readiness** and **Draft preview**. Resolve any listed
   blockers, then choose **Continue to activation**.
5. Choose **Activate** under **Validation and activation** when you want the
   copied pack to become the Persona's active visual pack.
6. Return to the supported Persona surface and check the floating image.

You can also import a portable pack through the preview/review flow. Copying,
previewing, and importing are separate from activation; do not assume the active
art has changed until activation succeeds. **Refresh catalog** reloads the
starter choices. If it fails, **Retry catalog** retries that read without copying
or activating a pack.

The server's Migu UAT used a selected Persona and activated Migu pack. It does
not establish that every existing account already contains that Persona. Your
catalog, saved names, and active pack can differ.

## Open, move, and send a text message

Click the floating Buddy to expand or collapse its controls. Use **Drag Buddy**
to move the floating panel; dragging the image is not the documented drag handle.
The shell keeps position preferences for its surface and constrains placement to
the viewport. Do not apply Chatbook's terminal resize keys to this browser shell.

| Control or feedback | What it does |
|---|---|
| **Start** | Starts a text session for the selected Persona. This is not microphone Start. |
| Session selection | Chooses the Live session shown in the shell. Check the session before sending or stopping. |
| **Message your buddy** → **Send** | Sends text to the current session, starting an eligible text session when needed. |
| Feedback area | Shows message outcomes, replies, errors, or a need for review. |
| **Stop** | Stops the selected session's work. Already-dispatched provider work may finish remotely, but stopped output must not restart local playback. |
| **Open Full Live View** | Opens the corresponding Persona/session for the transcript, voice, and review controls. |
| **Set up voice**, **Listen**, or **Stop listening** | Opens that session's Live view. The label reflects availability/activity; microphone control happens in Live. |

For a first text test, send a short request such as “Reply with: the blue notebook
is ready.” Check the actual reply in the feedback area or full transcript. An
idle or speaking image alone is not evidence that the provider returned an answer.

## Make one voice turn

Before recording, open **Profiles**, set **STT model** and **TTS provider**, then
choose **Save assistant defaults**. **TTS model** and **TTS voice** are optional
overrides: leave them blank to use the selected provider's applicable defaults,
or enter values that provider supports. A blank voice uses the server's configured
voice when this is its default provider; otherwise the selected adapter supplies
its own default. Browser-wide voice preferences do not override a blank Persona
voice. For **browser**, blank selects the browser's native default voice.

If Live is already connected, choose **Disconnect**, then **Connect** after saving.
A connected session keeps the settings captured when it connected. Returning from
Profiles or choosing Start again does not apply saved changes. Check Live's TTS
provider, model selection and voice before starting; “default model” or “default
voice” means the provider will resolve that omitted value.

Choose **browser** for the browser's speech synthesis, or choose a server TTS
provider or configured speech gateway from the provider list. A listed provider
still needs its required setup. For example, **kokoro** needs local model and
voice assets; it is one option, not a requirement. The legacy Persona value
**tldw** continues to select Kokoro. **Use browser fallback** inherits the
browser-wide TTS preference; it does not force browser speech. Check the effective
provider shown in Live. See [TTS setup](TTS-SETUP-GUIDE.md) for provider setup.

Persona Live uses the selected route and does not silently switch providers when
preparation or synthesis fails. Fix the reported setup problem or explicitly
select another provider, save defaults, and reconnect Live. Browser speech requires browser support and an available
voice; server readiness cannot prove that browser playback will work.

Start with manual control so you decide when recognized speech is submitted:

1. From Buddy, open **Open Full Live View** for the intended session. If the
   session is disconnected, choose **Connect**.
2. Check the displayed **STT** model/language and **TTS** provider/model/voice. These
   are read-only summaries; correct them under **Profiles** and save the assistant
   defaults if they are wrong.
3. Leave **Auto-commit**, **Auto-resume**, and **Barge-in** off for the initial
   test. Pause background narration or other audio.
4. Choose **Start listening**. Wait for preparation and the **listening** state,
   and allow browser microphone access when requested. Do not speak while the
   runtime is still preparing.
5. Say one short phrase. Watch **Last heard**: these are provisional words and
   may change as recognition receives more audio.
6. While the turn is still listening and recognized speech is present, choose
   **Send now**. This submits the transcript currently shown, once.
7. Read the committed transcript and reply, listen for playback, and check that
   Live returns to **idle**. With Auto-resume off, a new recording requires a
   fresh **Start listening**. Choose **Disconnect** when finished.

**Send now submits; Stop voice cancels.** Do not click **Stop voice** and then
expect **Send now** to submit the cancelled recording. Last heard can remain
visible after Stop, completion, or disconnect while Send now is disabled. If
recognition is wrong, cancel and retry, or type the intended message in the text
composer. Last heard is not an editable draft like Chatbook dictation.

Keep a Persona voice turn within **30 seconds, including silence**. Whisper and
Parakeet ONNX revise the complete bounded turn rather than appending stale chunk
guesses. This reduces repeated fragments; it does not guarantee correct speech
recognition. Automatic turn detection, when enabled, waits for recognition through
the detected speech boundary. Manual Send now uses the currently displayed text.

Stop invalidates late recognition and playback. A native decoder can still be
finishing in the background; a quick retry may report busy until cleanup completes.
Known Parakeet decoder error statuses produce speech failure feedback instead of
being presented as words.

The September 6 physical tests used normal server Parakeet ONNX configuration,
DeepSeek conversation, and local Kokoro output. Whisper was also tested for the
responsiveness repair. These are historical test choices, not required providers
or universal defaults. They do not qualify every TTS provider, gateway, browser
voice, or a separate realtime provider connection.

## Review approvals in the exact session

When Buddy shows **Needs approval** or review feedback, open **Open Full Live
View** and inspect the pending plan or tool request in that session. Check the
proposed actions and inputs, then explicitly confirm the intended steps or cancel.
Opening Live, sending a message, choosing art, or enabling voice does not approve
tools. Tool execution remains subject to the server's authentication, policy,
moderation, and budget checks.

Changing Persona through setup or Live ends the former connection and clears its
resume selection, transcript, and pending approval view. Choose **Connect** for
the newly selected Persona. Do not review a different session on the assumption
that it is the one shown by Buddy.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Buddy is absent | Check **Enable persona buddy shell**, desktop width, the selected Persona, active pack, and whether the current surface supports Buddy. |
| Starter catalog fails | Use **Retry catalog** after checking connectivity and authentication. A retry does not activate anything. |
| Image fails while text works | Check the visual diagnostic and authenticated pack loading. Confirm the pack is active. Do not make protected assets public as a workaround. |
| “Visual pack did not load — rate_limited” | Stop repeated reloads and let the limit window recover, then reconnect. Record the session and reproduction steps if it recurs. Reconnection restored the image in UAT, but the repeated-request trigger remains unresolved (TASK-13211). |
| **Start listening** is unavailable | Connect the intended session and check preparation feedback, the configured Chat target, selected STT/TTS setup, and browser microphone permission. |
| **Send now** is unavailable | It needs recognized speech in the current listening turn. If you already stopped, sent, or disconnected, start a fresh turn or use text. |
| Wrong or repeated words | Wait for the provisional text to settle before sending; pause background audio. If wrong, cancel and retry. A successful previous phrase does not guarantee the next one. |
| No reply or no sound | Read the provider/TTS error and committed transcript. Check browser audio permission, output device, volume, and the selected voice. An audio-chunk notice or animated sprite alone does not prove audible playback. |
| Selected TTS provider fails | Check that provider's model, voice, dependencies or credentials. After changing saved defaults, choose **Disconnect → Connect** in Live. For a gateway, check its configured route. Persona Live does not silently substitute Kokoro or another provider. |
| Voice prepares but Chat fails | Check the actual Chat error, authenticated access, effective credentials, and budget. Preparation checks the target; Chat admission happens when the turn is submitted. |
| Shorter-turn or busy notice | Keep the retry under 30 seconds; after Stop, allow the previous decoder's cleanup to finish. |
| Audio rate limit | The browser stops capture. Wait one minute before retrying **Start listening**, as directed by the notice. Operators should review configured limits rather than bypass them. |
| Approval appears stuck | Open the exact session in Full Live, inspect the pending request, and confirm or cancel there. |

## Validation status

The guide's initial source check used server `dev` **83af7e5dcf** on
**2026-09-07**; provider-selection instructions also reflect the TASK-13214
implementation. This guide consolidates current controls and recorded UAT; it
does not claim a new physical microphone test or all-provider playback test for
this documentation update.

- Human UAT verified real provider replies, clear Kokoro output, and stopped
  recording/playback afterward. Exact recognition varied between attempts.
- PR #2927 repaired decoding responsiveness, whole-turn revisions, and error
  handling; 202 targeted Python tests passed and Qodo's findings were resolved.
- Full request-correlated floating **listening → thinking → speaking → idle**
  validation remains open under TASK-13202. Repeated visual-pack/session requests
  reached rate limits in one observation; TASK-13211 tracks the unresolved trigger.
  A stable idle image after reconnect is not proof that this defect is fixed.

## Related documentation

- [Personas: profiles, sessions, APIs, and voice configuration](../Server/Personas_User_Guide.md)
- [Persona Live wake phrases](Persona_Live_Wake_Phrases.md)
- [Persona Visual Packs](../../Code_Documentation/Persona_Visual_Packs.md)
- [Detailed Migu voice UAT and review evidence](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md)
