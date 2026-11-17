# Tracked picker discovery readiness — TASK-13681

A real runtime-rail card click reached the tracked Form callback with all entry/source/account guards current, while model usability was loading with an empty catalog. Native preflight correctly refused; the list still appeared selectable. Reuse existing parent discovery state in AssistantSelect, show an accessible waiting status and disable only tracked selection until discovery settles. Do not relax preflight, await a new request, replay a click, or retire the original requested mode when loading changes. Overlay selection and staged editor identities do not require this gate.

Full Form → ComposerToolbar and compact Form → CharacterControlsSheet forward the existing loading state. The shared picker controls card availability, aria-busy and handler admission; the overlay fallback keeps the existing busy guard. Both existing locale forms carry the same copy. Tests cover actual shared admission, settlement/mode preservation, overlays and both forwarding seams; real-server qualification retains the original one-click rail assertions.

ADR assessment: existing shared readiness and ADR-049 ownership; no new durable rule, API, dependency or state framework. Original browser failures and temporary diagnostic errors remain excluded. All emitted diagnostics and their test listeners have been restored before qualification.
