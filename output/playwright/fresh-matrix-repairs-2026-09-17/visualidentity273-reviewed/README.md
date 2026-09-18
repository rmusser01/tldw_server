# Reviewed optional expression-pack capability gate (UAT273)

The panel loads capabilities before requesting active packs. An explicit unsupported metadata capability skips pack authoring requests and disables authoring controls while retaining expression slots and binding resolution. Supported and legacy capability responses retain authoring behavior. Original PostgreSQL Metadata expansion acceptance remains pending.

Independent review reproduced the unsupported-pack call on the baseline and reran four maintained UI tests plus sequencing controls. Review exposed two new test-mock type errors not covered by the frontend compiler entry point; the narrow mock annotation fixes those, with original failure and evidence chronology retained. Scoped lint is clean; broad frontend compilation retains its known baseline. Bandit cannot assess TypeScript. No full-matrix acceptance is claimed.
