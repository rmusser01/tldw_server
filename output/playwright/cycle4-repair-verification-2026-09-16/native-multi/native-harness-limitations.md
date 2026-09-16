# UAT114 native verification limitation

Six real headed Playwright pages each reported document.visibilityState=visible and maintained six notification streams. Installed Playwright source unconditionally enables focus emulation (retained source excerpt). Removing that override via described native CDP emulation command, both detached and retained sessions, changed document.hasFocus but all six visibility states stayed visible. No document.visibilityState override, synthetic visibility event, installed-tool patch or application source modification was used.

Supported fallback attempted: CUA createBrowserTab(chrome, localhost18381, named UAT session) returned Browser is not available: chrome. cua-driver serve attempted its supported CuaDriver.app relaunch; no daemon appeared within5seconds, exit1. cua-driver status confirmed daemon not running. Read-only check_permissions(prompt:false) reported Accessibility not granted, Screen Recording granted, and explicitly warned that outside-daemon results may be inaccurate. No system permission was changed or prompted.

Therefore hidden-tab cancellation and visibility catch-up were not verifiable with the available native harness. This is not a product failure or a pass. Five extra test tabs were subsequently closed to release artificial six-visible-tab saturation; original preserved browser tabs were untouched.
