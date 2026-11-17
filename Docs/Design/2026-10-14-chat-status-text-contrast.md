# Chat text contrast — TASK-13674

The existing actual axe matrix measures stable theme endpoints after finite CSS animations settle. Its remaining three failures are Draft saved (4.46:1 on white at11px), selected composer control text (3.17:1 on dark tinted surface at12px), and connected status text (2.69:1 on light success tint at12px). Earlier transient theme-token findings are not credited as persistent source defects.

Use existing readable text tokens for both full/compact Draft saved indicators, shared selected composer controls and the shared connection status label. Keep selected borders/background, pressed semantics, severity badge/tint and descriptive labels. Do not change the global theme palette, layout, actions, persistence or ownership. Shared producers and actual browser contrast are the scope; no mirrored class-only tests or new dependencies.

ADR assessment: existing semantic theme/component patterns; no new durable architecture rule. Master TASK-13458 owns the12-cell320/390/1280 light/dark and CSS renderer200% stress matrix. It is not native browser-toolbar zoom or physical assistive-technology evidence. Both client profiles, source types/lint, actual production gates and bounded source review must remain green.
