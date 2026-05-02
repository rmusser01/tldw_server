# Getting Started with Home Assistant + Music Assistant + tldw_server

This guide captures a practical first-room setup for a home voice assistant that uses:

- `Home Assistant` for room/device orchestration, automations, timers, and home control
- `Music Assistant` for music playback and source management
- `tldw_server` for advanced conversational behavior, summaries, RAG, notes, and persona-style responses
- a per-room Linux voice satellite paired to an existing Bluetooth speaker

The target outcome is a single room prototype that is stable enough to copy to additional rooms.

## Recommended Architecture

Use one small Linux voice satellite per room. Do not try to centralize Bluetooth audio on one Linux server for the whole house.

Recommended split:

- Room node:
  - microphone array
  - wake word handling
  - local Bluetooth pairing to that room's speaker
  - speech output and music playback target for that room
- `Home Assistant`:
  - room identity
  - device orchestration
  - timers, alarms, announcements
  - home-control intents
  - routing to the correct room player
- `Music Assistant`:
  - `Spotify`
  - `YouTube Music`
  - internet radio
  - local files
  - queueing and playback control
- `tldw_server`:
  - advanced assistant tasks
  - RAG and summaries
  - note-aware answers
  - richer conversational responses than stock Assist

Recommended `v1` playback behavior:

- use `pause/resume` for announcements and assistant speech while music is playing
- do not assume true ducking will work reliably with generic Bluetooth player paths

## Recommended First-Room Hardware

For the first room, buy to a hardware class, not to one exact Raspberry Pi SKU.

Availability-first buying rule:

- do not budget or plan this build from historical Raspberry Pi MSRP alone
- use live reseller pricing and actual in-stock listings at the time you buy
- if the exact Pi board you want is out of stock or badly overpriced, buy an equivalent Linux node instead

Required hardware characteristics for the room node:

- `64-bit Linux` capable
- `2GB RAM` minimum
- `dual-band Wi-Fi` preferred
- `Bluetooth` for room-local speaker pairing
- at least one usable `USB` port for the mic array
- stable power supply and a case appropriate for always-on use

Preferred first-room example, if actually available at a sane price:

- `1x Raspberry Pi 4 Model B (2GB or 4GB)`
- `1x Raspberry Pi 15W USB-C Power Supply`
- `1x Raspberry Pi 4 Case`
- `1x ReSpeaker XVF3800 USB 4-Mic Array`
- `1x USB-C data cable` for the XVF3800
- `1x 32GB or 64GB microSD card`, preferably high-endurance
- `1x existing Anker Bluetooth speaker`
- `Optional: Raspberry Pi 4 Case Fan` if the room is warm or the case is enclosed

Acceptable substitutes if `Pi 4` stock or pricing is bad:

- `Raspberry Pi 5` if you already have one or can buy one easily
- a `used/refurb x86 mini PC or thin client` that can run Linux Voice Assistant cleanly
- another `ARM64 SBC` that meets the requirements above

Why this class of hardware:

- the `ReSpeaker XVF3800` matters more than raw CPU for perceived voice quality
- a stronger Linux node is safer than a `Pi Zero 2 W` for the first prototype when combining Linux audio, Wi-Fi, and Bluetooth playback
- this setup is better for validating the architecture before trying to reduce cost in later rooms

## Current References

These references were spot-checked on `2026-04-30`; still verify live reseller stock, pricing, and model-specific install notes before buying hardware.

- Raspberry Pi 4:
  - https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
  - https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/
- Raspberry Pi 15W USB-C PSU:
  - https://www.raspberrypi.com/products/type-c-power-supply/
- Raspberry Pi 4 Case:
  - https://www.raspberrypi.com/products/raspberry-pi-4-case/
- Raspberry Pi 4 Case Fan:
  - https://www.raspberrypi.com/products/raspberry-pi-4-case-fan/
- ReSpeaker XVF3800:
  - https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html
  - https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/
- Linux Voice Assistant:
  - https://github.com/OHF-Voice/linux-voice-assistant
- Music Assistant:
  - https://www.music-assistant.io/
  - https://www.music-assistant.io/installation/
  - https://www.music-assistant.io/integration/installation/
  - https://www.music-assistant.io/integration/announcements/
  - https://www.music-assistant.io/integration/voice/
- Home Assistant developer/user docs:
  - https://developers.home-assistant.io/docs/intent_conversation_api/
  - https://www.home-assistant.io/docs/automation/trigger/
  - https://www.home-assistant.io/integrations/rest_command/

Notes:

- `YouTube Music` support in Music Assistant currently depends on paid access and cookie-based authentication, so treat it as the least stable music source.
- Music Assistant currently does not provide built-in universal voice-initiated music playback in Home Assistant out of the box; that part may require custom HA voice glue or the community voice-support repo.

## First-Room Setup Order

Do these in order. Do not skip ahead to `tldw_server` integration before the room node and media path are healthy.

### 1. Build the Room Node

- Install the current supported `64-bit Linux` image for the selected node, such as `Raspberry Pi OS Lite` for a Raspberry Pi
- Enable `SSH`
- Give the node a room-specific hostname such as `voice-office-01`
- Boot the room node
- Update packages
- Put the node on `5 GHz Wi-Fi` if available

Why `5 GHz`:

- it reduces `Wi-Fi + Bluetooth` coexistence pain on the room node

Success criteria:

- the node is reachable over SSH
- the system is updated
- the room node has a stable network connection

### 2. Confirm Audio Hardware

- Plug in the `ReSpeaker XVF3800`
- Confirm Linux detects it as a USB audio input device
- Verify the mute button and indicator behavior

Success criteria:

- the mic array is visible to the OS
- the mute button behaves as expected

### 3. Pair the Room Speaker

- Pair the room's Anker Bluetooth speaker to the room node
- Set it as the default output device
- Reboot once and verify it reconnects automatically

Success criteria:

- you can play a test sound through the speaker
- the speaker reconnects after reboot without re-pairing

### 4. Install Linux Voice Assistant

- Install the current supported `Linux Voice Assistant` stack or use its Raspberry Pi image
- Set the audio input device explicitly to the `XVF3800`
- Set the audio output device explicitly to the paired Bluetooth speaker
- Enable local wake word
- Join it to your `Home Assistant` instance

Success criteria:

- wake word works in-room
- `Home Assistant` receives speech
- a simple Assist command such as turning on a light works end-to-end

### 5. Install Music Assistant

- Install the `Music Assistant` server
- Add the official `Home Assistant` integration
- Add providers in this order:
  - `local files`
  - `internet radio`
  - `Spotify`
  - `YouTube Music`
- Expose the room player in `Home Assistant`

Recommended order rationale:

- start with the easiest and most deterministic sources first
- leave `YouTube Music` for last because it is the least stable

Success criteria:

- the room player is visible in `Home Assistant`
- you can start and stop music from `Home Assistant`
- you can play from each configured source

### 6. Configure Assistant Speech Over Music

- Use `pause/resume` for assistant responses and announcements while music is playing
- Test both short and longer speech responses

Success criteria:

- music pauses
- speech plays clearly
- music resumes afterward
- the room does not end up stuck in a paused state

### 7. Add tldw_server as the Advanced-Brain Backend

Start with one narrow integration path from `Home Assistant` to `tldw_server`.

Recommended `v1` approach:

- use a `Home Assistant` automation, script, or `rest_command`
- send selected advanced requests to `tldw_server`
- get back plain text
- speak the result in the originating room through `Home Assistant`

Good first advanced commands:

- "summarize my notes about X"
- "what do I know about Y"
- "read me the latest summary for Z"

Success criteria:

- the request reaches `tldw_server`
- `tldw_server` returns useful text
- the response is spoken in the correct room

### 8. Only Then Replicate to More Rooms

- clone the known-good configuration or image
- change room identity, device name, and Bluetooth pairing per room
- roll out to one more room before doing the whole house

## Home Assistant / Music Assistant / tldw_server Responsibility Split

Keep responsibilities clean.

`Home Assistant` should own:

- room identity
- wake-word pipeline and room awareness
- timers, alarms, and announcements
- home-control intents
- choosing the correct room player
- deciding when to pause and resume media for speech

`Music Assistant` should own:

- music source integration
- search and queueing
- playback control
- room player/media actions

`tldw_server` should own:

- advanced conversational responses
- RAG and note-aware answers
- summaries and knowledge-oriented tasks

Recommended routing examples:

- "turn on the office fan" -> `Home Assistant`
- "play Miles Davis in the office" -> `Home Assistant` + `Music Assistant`
- "pause music" -> `Home Assistant` + `Music Assistant`
- "summarize yesterday's contractor notes" -> `Home Assistant` -> `tldw_server` -> `Home Assistant` speech output

## Validation Checklist

Do not buy or configure multiple room nodes until the first room passes these checks.

### Boot and Recovery

- reboot the node `3 times`
- verify the mic, Bluetooth speaker, and `Home Assistant` connection all return without manual fixes

### Bluetooth Stability

- play music for `30 minutes`
- stop, wait `10 minutes`, then resume
- repeat this cycle `3 times`

### Speaker Reconnect

- power the Anker speaker off and on `3 times`
- verify the room node recovers without re-pairing

### Voice Quality

- run `20` wake-word or command attempts from `1-2 m`
- run `20` more from `3-4 m`
- run `10` off-axis attempts

Target:

- the quiet-room case should have very few misses

### Music Interruption

- while music is playing, trigger `10` announcements or TTS replies
- verify pause -> speak -> resume works every time

### Home Control

- run `20` basic commands such as lights, timers, volume, pause, and next track

### Music Sources

- confirm playback works from:
  - `local files`
  - `internet radio`
  - `Spotify`
  - `YouTube Music`

### Advanced Assistant Path

- run `10` advanced `tldw_server` requests
- verify they are answered in the correct room

## Failure Points To Watch Before Scaling

- `Wi-Fi/Bluetooth interference`
  - if you see audio dropouts, keep the room node on `5 GHz Wi-Fi` and avoid weak-signal placement
- `Bluetooth reconnect reliability`
  - if the speaker regularly needs manual re-pairing, stop and fix that before scaling
- `Mic/speaker placement`
  - do not place the mic array directly beside or behind the speaker
- `Power issues`
  - use a vendor-recommended power supply for the selected node; cheap supplies can cause undervoltage and unstable USB/audio behavior
- `Network topology`
  - `Music Assistant` wants the server and players on the same Layer 2 network; complicated VLAN/firewall setups are a common breakage point
- `Voice-initiated music playback`
  - expect to add custom HA glue for voice-driven playback requests
- `YouTube Music fragility`
  - treat it as the least trustworthy source until the rest of the system is stable

## Go / No-Go Rule

Go forward only if the first room passes all of these:

- `3` successful reboots
- `3` successful speaker power cycles
- `30+` minutes of stable music playback
- `10` clean interruption cycles
- no repeated manual Bluetooth repair steps

Do not buy multiples if you have to:

- restart Bluetooth regularly
- restart the audio stack regularly
- re-pair the speaker more than once during normal testing

Live with the first room for at least `3-7 days` before copying the design to the rest of the house.

## Suggested Next Steps

After the first room is stable:

1. Add a second room and confirm the pattern duplicates cleanly.
2. Standardize your room-node naming, speaker pairing procedure, and recovery steps.
3. Expand the `Home Assistant` -> `tldw_server` command set gradually.
4. Only then benchmark cost-down nodes, such as a `Pi Zero 2 W`, in secondary rooms.
