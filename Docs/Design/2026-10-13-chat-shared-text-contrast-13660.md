# Shared chat text contrast — TASK-13660

Actual fresh Casual /chat at1280x720/light, with sidebar and Customize collapsed, reports three text failures: command shortcut4.43, Recover drafts4.06, Healthy2.96. The shared header uses a subtle hint token; the shared recovery's text button inherits an accent foreground; the shared model selector uses warning/success colors as tiny text foregrounds. Those colors remain useful for borders/background cues but do not meet small-text contrast.

Use existing text-text-muted for the command hint and text-text for the recovery action and model status/label. Preserve warning/success backgrounds, warning border/icon and truthful status wording, plus all command, recovery scope and selection handlers. Override only the shared recovery text owner rather than changing the app-wide text-button variant. No new abstraction, token, dependency or global palette. The original fresh/picker axe baseline remains unchanged until actual post-fix states qualify. Broader history and roleplay/status text findings remain separate recorded scope.

ADR required: no; existing shared presentation and semantic token contracts remain governing.
