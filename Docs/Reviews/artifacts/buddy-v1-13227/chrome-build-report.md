# Final Chrome production artifact: Buddy transcript and speech errors

Load unpacked: `/private/tmp/tldw-buddy-chrome-prod-zy5ty355/apps/extension/build/chrome-mv3`. This directory now contains this final build, superseding earlier unpacked output. Baseline and prior visual-only builds remain in their separately named ZIPs and reports.

Base source commit: `1fc19c7c8384f38eca2becdf53fb8bbcd205be7a` plus exactly one source overlay: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyInteraction.tsx`. This is not unchanged archive bytes. A private assistant-only presentation helper supplies summary/hint to both transcript and read-aloud. Speech retains the conversation title prefix; ordinary messages and user-quoted error envelopes remain verbatim. No error envelope or diagnostic detail is spoken for decoded assistant failures.

- Patch: `buddy-friendly-error-speech-ui.patch`; SHA-256 `5cb801894f1d0e4acedc22251d8a8888667fa7aa9e00560340166c75130968df`.
- Modified component SHA-256: `4c1fed6dae04651e0ec31aaca56822732373b4bb717562b5d0a75de1d1b9b94f`.
- Unchanged frozen lock SHA-256: `1e04471a55d77690e18508009448cb4fe492147ab994bdf1a7f45fb03b58b730`. Existing dependencies reused without reinstall.
- Build: `bun run build:chrome:prod`, exit 0, 43.0 seconds; shared-token post-build check passed.
- Final ZIP: `tldw-chrome-production-1fc19c7-buddy-friendly-error-speech.zip`; SHA-256 `1375a51f13cb851ca3e12c277b0da2b7e6da19f510038382fe3c6d6c778048e8`.
- 1378 files; manifest background/options/side-panel/icon/content-script targets and ZIP integrity verified.
- Previous visual-only ZIP `tldw-chrome-production-1fc19c7-buddy-friendly-error.zip` retains SHA-256 `fea38e5e4981877ac0cc16150bcbfeecedd9696dbfdfdcff38e865b948bd012c`. Its report/evidence are unchanged. Baseline ZIP/report/evidence also preserved.

Full source/patch/lock hashes, manifest and individual artifact hashes: `build-evidence-friendly-error-speech.json`. Full build log: `build-chrome-prod-friendly-error-speech.log`. Existing duplicate-autoimport, caniuse and dependency bundler warnings are retained.

Qualification-source tests: 9 passed (BuddyInteraction 7, decoder 2), 1.89 seconds. Activity-to-authorized-read-to-speech regression first failed with exact raw envelope/detail in the speech call (1 failed, 6 passed), then passed with the conversation title plus summary/hint. Earlier transcript and user-envelope regressions remain green. Prettier and diff checks passed; ESLint zero errors and the same six baseline warnings. Existing Node/i18next test warnings remain. Logs: `/private/tmp/buddy-error-transcript-speech-{red,green,prettier,eslint}.log`.

No browser actions, physical audio, microphone use, screenshots or paid providers performed. Speech is asserted at the mocked TTS boundary; native runtime verification is owned by root.
