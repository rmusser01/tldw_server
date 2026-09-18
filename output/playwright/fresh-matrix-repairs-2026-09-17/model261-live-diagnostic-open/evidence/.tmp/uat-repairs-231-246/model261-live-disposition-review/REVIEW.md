# UAT261 / TASK13260.203 — live paired-capture disposition

## Verdict

**No actionable application defect is established. UAT261 remains unqualified
because one of the two bounded, like-for-like native submissions was nonexact.**
The exact second result is a current observation, not a synthetic pass and not
a basis for retrying until green.

## What the pair establishes

The two fresh-library conversations used the same original tagged character,
source, settings, and provider seam. The wrapper recorded complete input
projections with equal ordered-message fingerprints, equal effective-settings
fingerprints, and equal provider-system fingerprints. Both streams reached a
`stop` terminal and were persisted and canonically reloaded. One retained a
55-character nonexact final; the other retained a 10-character exact final.

This rules out a difference in the application-controlled input and settings
that the instrumented boundary records. It supports output variation after
that boundary. It does not identify a product bug, nor does it prove a
provider-internal cause: sampling/runtime state outside the captured request
projection remains unavailable.

The diagnostic complied with its cap of two submissions and restored the
ordinary backend afterward. The restored ordinary process is healthy with the
same source/profile/initialization/holder evidence.

## Relation to the earlier UAT261 finding

The earlier historical outbound fingerprints cannot be reconstructed. The
current equal pair therefore cannot assign a cause to an earlier nonexact turn
or prove that the prior deviation had the same source. It does strengthen the
prior disposition: a current source-level change or provider tuning would be
unsupported by the available evidence.

## Smallest evidence-based next step

Do not make an application change, tune the provider, or send another request.
Keep TASK13260.203 open with its unchanged exact-final criterion. If its owner
needs a resolution, the next action is an acceptance-policy decision about a
literal final under an output-varying provider; it is not a technical retry.
Any later investigation must begin with a newly authorized, bounded design
that states what additional provider-side state can be safely observed. It
must not reinterpret the exact second result as satisfying the paired UAT.

## Evidence and limits

| Input | SHA-256 | Review use |
| --- | --- | --- |
| Live capture | `79650174bb9a004bf58b7f69e54cced68ae3729022c716b46976a4bdf63e046f` | Equal complete message/settings/system fingerprints for A and B |
| Capture status | `0a78a372f4886d3b7b8ab854b107dd812bc0d47a0bcb9c40c8159d6b1d5686f3` | Exactly two calls and restored binding |
| Native audit | `77e44fd58a1ac116144dad8c440e422821c44ba930b5ebb48c7c45da9ed838cf` | Nonexact/exact canonical outcomes, normal persistence/reload, and bounded protocol checks |
| Protocol | `cab8b3eafc5378b770f41e1749240f76b4123805df40b8b78db0365d8c2d61f7` | Original fixture, two-call cap, unchanged criterion/settings, and restoration rules |
| Capture wrapper | `3ff6ea2ae18c572e6cba57626a230a9c156f38a3ac524b320a296b6fb1c3cc4e` | Independent wrapper bytes used by the released launcher |
| Launcher | `f2fcf115eafb77c43f2d330526d59eb21f18a2a72e008676503f04389ab65e1f` | Released-source, module-hash, profile, and diagnostic-source gate checks |
| Backend source comparison | `9864416834b6f9ca449d18634c33e4548cc5d59590f1c1fd49bd2679dbddcc3a` | Relevant character stream source unchanged between historical and active snapshots |
| Ordinary restoration | `84c47595ba9495cd15d2194e1c623bbd91c6dc81f8e7f927011c0bad73cea859` | Healthy ordinary backend on the diagnostic source/profile binding |
| Earlier read-only disposition | `a13c0d0b0f8203ae636cb43c88b6bd1cc1164b36820fc952fca3d17503ba097b` | Historical fixture/prompt scope and unrecoverable-fingerprint limitation |

This review reads only safe metadata, hashes, source/protocol controls, and
prior safe projections. It contains no provider text, reasoning, prompt body,
headers, credentials, or raw private logs. No product, browser, runtime,
model, Git, Backlog, or tracker state was changed.
