# UAT161: development indicator no longer covers Settings

The actual bottom-left Next development indicator intercepted a normal Settings click at1200×969. Keyboard navigation worked. The two-line `next.config.mjs` change uses the installed Next16.1.4 option `devIndicators.position: bottom-right`; errors remain enabled and explicit developer position preferences are preserved.

Author and root independent existing configuration suites each pass12tests/2suites. Syntax, installed schema and lint checks pass; independent review is clear. Native post-restart geometry places Settings at x7.5–39.5 and the indicator at x1146–1178. An ordinary pointer click reaches Settings. Chat composer, Send and Advanced controls also do not overlap the indicator in the captured viewport. This is bounded route/viewport acceptance, not a guarantee for every layout. Bandit cannot analyze JavaScript; its parse error is retained without a security-pass claim.

The original click failure/screenshot are the RED. `after-geometry.txt` began collecting before the edit but returned after Next automatically restarted; it is correctly labeled AFTER. No source code, browser preference or portal CSS was changed to fabricate that geometry. Evidence copies exclude credentials.
