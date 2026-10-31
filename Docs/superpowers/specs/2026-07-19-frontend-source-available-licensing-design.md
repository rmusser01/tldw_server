# Time-Delayed Source-Available Frontend Licensing Design

- **Date:** 2026-07-19
- **Backlog task:** TASK-12974
- **Status:** Approved product and engineering design; license implementation and
  counsel review remain separate gates
- **Licensor:** Robert Benjamin Jake Musser

## Purpose

Protect future tldw WebUI, admin UI, shared UI, and browser-extension releases
from commercial substitution during an approximately two-year
customer-development window. At the same time, keep the server implementation
commercially usable under GPL, preserve normal personal and internal frontend
use, and place every protected release on an irrevocable path to
`AGPL-3.0-only`.

This document records the selected product policy, repository boundary, release
mechanics, and safety gates. It is not itself a software license, a contributor
agreement, a commercial contract, or legal advice. Exact legal files govern.

Because counsel is unavailable for approximately two weeks and a major patch is
already public, the design deliberately separates a conservative initial
license cutoff from later custom permissions. Only published, effective legal
files grant rights. A description of a later grant in this design does not make
that grant effective.

## Decision Summary

Each new protected frontend release will have two coordinated legal documents:

1. The repository-authored protected material is offered immediately under the
   unmodified **PolyForm Perimeter License 1.0.1**.
2. A release-specific **PolyForm Countdown License Grant 1.0.0** makes a present,
   non-revocable grant of the full unmodified **GNU Affero General Public
   License v3.0 only** terms, beginning at 12:00 noon UTC on that release's
   second calendar anniversary.

When the Countdown start date arrives, `AGPL-3.0-only` becomes an additional
license option for that release. Perimeter does not terminate. Recipients may
then rely on either the continuing Perimeter terms or AGPL-3.0-only. This is a
rolling schedule calculated independently for each release.

The server implementation and unlisted repository-authored code will be
`GPL-3.0-only`. The canonical OpenAPI contract and published contract snapshots
will be `Apache-2.0`. Third-party material keeps its upstream terms.

The initial pre-counsel launch will not publish custom community-fork,
dedicated-customer, CLA, trademark, or commercial-license language. It will use
the standard Perimeter and Countdown texts, an exact path scope, historical
cutoff records, and third-party notices. The custom permissions described later
in this document become prospective policy only after counsel approves and the
corresponding legal files are published.

## Goals

- Delay commercial products or hosted services that use protected code to
  substitute for the official WebUI, admin UI, or browser extension.
- Keep the GPL server available for commercial use, hosting, support,
  integration, and clean-room client development.
- Permit personal use, internal commercial use, self-hosting, modification, and
  redistribution of protected releases when those activities are permitted by
  Perimeter and do not provide a competing product to others.
- After counsel review, permit genuinely free competing community forks and
  paid deployment for one named customer's isolated internal environment.
- Accept future contributions without losing the ability to publish scheduled
  AGPL grants or offer selective commercial licenses.
- Make path scope, release identity, future AGPL date, provenance, notices, and
  artifact licensing mechanically verifiable.

## Non-Goals and Accepted Limits

- This change cannot revoke or narrow GPL, ISC, or other permissions already
  granted on previously public code.
- Draft pull request
  [#2727](https://github.com/rmusser01/tldw_server/pull/2727) is already public.
  Code that appeared there under the repository's former terms remains
  available under those terms. Moving, closing, merging, or retagging the pull
  request does not claw those permissions back.
- The new policy materially protects only future repository-authored code that
  has never been published under the former terms.
- The policy does not prevent independently developed clients, clean-room
  implementations, competing ideas, or competitors that use only the public
  API contract.
- Perimeter itself prohibits both paid and free competing products. The desired
  exception for genuinely free competing community forks therefore requires a
  separate counsel-reviewed grant; it is not effective during the conservative
  initial launch.
- Once a release's Countdown grant starts, AGPL permits commercial
  redistribution and hosting subject to AGPL. The two-year window delays that
  competition; it does not prohibit it permanently.
- Source-available protected releases will not be described as open source
  before their AGPL option starts.
- Software licenses do not by themselves guarantee enforceability, market
  differentiation, trademark ownership, or protection from clean-room
  competition.

## Considered Models

### Perimeter plus release-specific Countdown (selected)

Unmodified Perimeter directly targets products that compete with the software.
The separate Countdown template creates a present grant of a known future
license for one release. Together they preserve a restricted commercial window
without relying on a later promise to relicense.

The main costs are source-available status during the restricted period, a
custom post-counsel community exception, per-release legal records, contributor
rights management, and careful separation from GPL implementation code.

### Immediate AGPL-3.0-only

AGPL would require corresponding source for modified network deployments, but
it would permit immediate paid hosting and commercial substitutes. It protects
software freedom rather than a market-entry window.

### Business Source License 1.1

BSL can express a change license and date in one document, but its Additional
Use Grant would require custom competitive-use drafting before the urgent
cutoff. Perimeter plus Countdown keeps the initial legal text standard and
defers custom permissions until counsel is available.

### Proprietary frontend with later publication

This would provide stronger exclusivity but would conflict with the desired
self-hosting, redistribution, visible-source, and contribution model.

## Authoritative Path Scope

The repository becomes an explicit multi-license monorepo. The root `LICENSE`
file is the human-readable authoritative scope map; exact license texts live
under `LICENSES/`. A nested third-party notice or license overrides the default
for the material it covers.

| Repository-authored scope | Governing terms |
| --- | --- |
| `admin-ui/**` | Perimeter 1.0.1 plus that release's Countdown grant |
| `apps/tldw-frontend/**` | Perimeter 1.0.1 plus that release's Countdown grant |
| `apps/extension/**` | Perimeter 1.0.1 plus that release's Countdown grant |
| `apps/packages/ui/**` | Perimeter 1.0.1 plus that release's Countdown grant |
| Canonical OpenAPI document and versioned complete contract snapshots | `Apache-2.0` |
| `tldw_Server_API/**` | `GPL-3.0-only` |
| All other unlisted repository-authored code, tests, scripts, and documentation | `GPL-3.0-only` |
| Legal documents and third-party material | Their stated terms |

The protected defaults cover original code, tests, build definitions, and
original assets inside the four named paths. They do not overwrite third-party
copyrights or licenses. Generated, copied, and vendored files must be classified
individually and keep required upstream notices.

Each release manifest distinguishes repository-authored protected material from
third-party components in the same source tree or artifact. Perimeter and the
Countdown grant apply only to material the identified contributors are legally
able to license. The project does not state that upstream MIT, Apache, or other
third-party components have been relicensed merely because they ship in a
protected artifact.

Markdown documentation remains GPL-3.0-only even when nested under a protected
path unless the root scope map expressly identifies a document as part of a
protected release. Legal records state their own status and are not governed by
the code default.

No broader wildcard such as `apps/**` is permitted. A shared or root-level file
that does not fall inside one of the four protected paths remains GPL-3.0-only
until an explicit, reviewed scope-map change says otherwise.

## Release Licensing Mechanics

### Synchronized frontend release identity

The four protected path families use one synchronized frontend licensing
release ID initially. A legal release record identifies the exact source commit
and the material covered at that commit. WebUI, admin UI, shared UI, and
extension artifacts built from that record share its Countdown date even when
their product version strings differ.

Splitting the components into independent licensing trains is a later design
change. It must not happen implicitly through unrelated package-version bumps.

### Release-specific Countdown grant

Every protected release receives an immutable directory under
`LICENSES/releases/<release-id>/` containing at least:

- a machine-readable release record;
- the unmodified Perimeter 1.0.1 text or an exact repository reference to it;
- a completed, release-specific Countdown 1.0.0 grant;
- the full, verbatim AGPLv3 license text in the Countdown grant's “New License
  Terms” section, not merely a URL or SPDX identifier;
- the protected source revision, release date, Countdown start date, and file
  digests; and
- applicable required notices and third-party notice references.

The Countdown start date is the same month and day in the second calendar year
after the release date, at 12:00 noon UTC. Release tooling rejects February 29
as a legal release date until a leap-day convention is explicitly approved.
This avoids silently choosing between February 28 and March 1.

The Countdown grant applies only to its named release. A later release never
inherits an earlier start date. The release record, completed Countdown grant,
source revision, and artifacts must agree exactly.

### Append-only records

Once a protected release is public, its release record and Countdown grant are
append-only legal history. They are committed, checksumed, and referenced by a
signed tag when signing infrastructure is available. Corrections use a new
release ID and a superseding record; they do not rewrite or delete the original
grant.

Repository documentation may mark that AGPL has started for discoverability,
but no transition commit is legally required. The original Countdown document
is the grant.

### Metadata

Package metadata that cannot accurately express both Perimeter and a future
AGPL option uses the ecosystem's “see included license file” form. It must not
claim `AGPL-3.0-only` before the Countdown date or use an unrecognized SPDX
identifier as though it were standard. Release-specific legal records remain
authoritative.

OCI annotations, browser-store listings, source archives, and release pages
identify the product version, licensing release ID, source revision,
source-available status, license-record location, and source URL.

## What Perimeter Permits and Restricts Initially

Perimeter 1.0.1 permits use, modification, creation of new works, and
distribution for permitted purposes. Its standard noncompete provision excludes
providing others a product that competes with the software, and its competition
definition can apply whether that competing product is paid or free and whether
it is a good or hosted service.

Accordingly, the conservative initial launch supports ordinary personal use,
internal company use, self-hosting, modification, and redistribution when they
are not part of providing a competing product to others. It does not add a
project-authored interpretation that weakens or rewrites the standard license.

Before counsel-reviewed additional grants are published, these uses remain
outside the standard public permission:

- a free or paid product offered as a substitute for an official protected
  frontend;
- a free or paid hosted service offered as such a substitute; and
- a consultant's paid deployment or management of a substituting frontend for
  a customer when that activity would amount to providing a competing product.

Ambiguous uses are directed to a licensing inquiry. The project will not make
ad hoc public exceptions in issues, email, documentation, or release notes.

## Conservative Pre-Counsel Launch

The initial license-cutoff change may publish only:

- the unmodified PolyForm Perimeter License 1.0.1;
- completed release-specific PolyForm Countdown 1.0.0 grants containing the
  unmodified AGPL-3.0-only terms;
- the root path scope map and historical cutoff record;
- third-party notices and corrected package/repository license metadata; and
- a plain statement that no trademark rights are granted.

Until counsel review is complete:

- do not merge third-party changes to the four protected path families;
- do not merge third-party changes that alter the canonical OpenAPI contract;
- do not publish protected binary artifacts until their license bundles and
  image separation pass the artifact gate;
- do not publish the Community Fork Grant, Dedicated Customer Grant, frontend
  CLA, API contract contribution grant, detailed trademark policy, or template
  commercial agreement; and
- do not enter a bespoke commercial frontend license without review.

Owner-authored work may continue under the new terms after the cutoff. Backend
contributions that neither touch protected paths nor alter the Apache-licensed
API contract may continue under the normal GPL contribution process.

## Prospective Post-Counsel Grants

These grants are desired policy, but they become effective only through
separate, versioned, counsel-reviewed legal files published prospectively. They
do not edit the Perimeter text and do not retroactively alter already released
versions unless an executed grant expressly identifies those versions.

### Community Fork Grant

The Community Fork Grant should permit a product that substitutes for an
official protected frontend only while the entire competing offering is truly
free and non-monetized. At minimum it must require:

- no purchase price, subscription, usage charge, mandatory donation, paid
  account, advertising, sponsorship placement, affiliate revenue, sale or
  monetization of user data, paid feature, paid priority, or paid access;
- no bundling whose purpose is to sell or promote a commercial product or
  service;
- no paid hosting, deployment, customization, support, warranty, or management
  attached to the competing offering;
- public availability of the complete corresponding source and practical build
  and installation materials at no charge;
- clear rebranding and no implication that the fork is official or endorsed;
  and
- continuing compliance with the applicable Perimeter, Countdown, notices, and
  third-party terms except to the exact extent the grant adds permission.

Unconditional donations and institutional grants may be accepted only when no
access, feature, support, influence, visibility, data, service level, or other
consideration is provided in exchange. The offering must remain available on
the same terms whether or not a person donates. Counsel must decide how to treat
commercial sponsorship, cross-subsidization, and grants from an operator's
customers before this permission is published.

This is the highest-risk custom public grant in the design because it must
override Perimeter's treatment of free competing products without creating an
easy monetization loophole.

After a release's Countdown date, these monetization limits cannot restrict a
recipient who chooses the independent AGPL-3.0-only option. The Community Fork
Grant governs only recipients relying on that additional permission during the
restricted period.

### Dedicated Customer Grant

The Dedicated Customer Grant should allow a consultant or service provider to
charge for deployment, customization, migration, training, support, and
management for one identified customer's internal use when:

- the customer is named in the engagement;
- the instance runs in infrastructure controlled by that customer and isolated
  for that customer's use;
- the provider supplies the customer the modified source and build materials;
- the provider does not operate a shared multi-customer control plane,
  multi-tenant service, pooled data layer, or generally available substituting
  product; and
- the provider does not use the engagement as a browser-store or public SaaS
  offering.

A provider may perform more than one engagement only when each independently
satisfies these conditions and the engagements do not become a shared or
generally offered substituting product. Counsel should define when standardized
repeat deployments cross that boundary rather than leaving it to informal
interpretation.

### Selective commercial licenses

Robert Benjamin Jake Musser may offer separate commercial licenses selectively
and case by case. There is no automatic right to purchase an exception and no
public universal price. Each agreement must identify the legal entities,
covered release IDs or source revisions, product or deployment, permitted
competitive use, term, and any trademark rights.

Support, warranty, indemnity, privacy, security, service levels, and trademark
permissions are separate contractual subjects and are never implied by the
public source license.

## Contributions and Rights Intake

### Backend contributions

Backend and other unprotected implementation contributions continue under the
project's normal `GPL-3.0-only` inbound terms, except when they alter the
canonical API contract.

### Protected frontend contributions

After counsel approval, a path-aware frontend CLA is required before merging a
third-party contribution to any protected path. Contributors retain copyright.
The CLA must give the Licensor sufficient copyright and patent permission to:

- publish the contribution under Perimeter 1.0.1;
- make each release-specific Countdown grant to AGPL-3.0-only;
- continue distributing the contribution under AGPL after the start date;
- issue the Community Fork and Dedicated Customer grants; and
- offer separate commercial licenses.

The CLA must address employer authority and must not imply a copyright
assignment if contributors are intended to retain ownership. The Countdown
template speaks in terms of each contributor licensing the release, so counsel
must specifically confirm that the CLA mechanism supports that present future
grant.

### API contract contributions

Any contribution that changes the canonical OpenAPI contract requires a narrow,
explicit `Apache-2.0` contribution grant for the resulting contract material in
addition to the GPL inbound terms for implementation code. This prevents a
later contributor from blocking publication of the shared contract under
Apache-2.0.

Before those two contribution mechanisms exist, the conservative launch pauses
the affected third-party merges. Automated dependency metadata does not expand
license scope and retains any upstream notices.

## Public API and SDK Boundary

The server implementation remains GPL-3.0-only. The complete canonical OpenAPI
document exposed by the server and immutable published snapshots are
Apache-2.0 so commercial and community clients can interoperate without copying
protected frontend implementation.

A future generic generated SDK may also be Apache-2.0, but it is not required
for the licensing cutoff. If created, it is generated from the complete public
contract and contains only generated endpoint declarations, generated data
types, and minimal generic transport primitives.

Existing frontend client and policy code remains protected, including:

- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`;
- `apps/packages/ui/src/services/tldw/request-core.ts`;
- `apps/packages/ui/src/services/tldw/server-capabilities.ts`;
- curated fallback schema definitions; and
- OpenAPI guard behavior used by the product.

Those files combine storage, caching, deployment modes, authentication UX,
capability discovery, fallbacks, and product policy. They are not migrated into
the generic SDK merely because they call a public API. Generated Apache
replacements are created independently from the complete contract if needed.

The existing voice-oriented SDKs are separate products, not the generic API
SDK. Their current license history and third-party inputs must be audited and
preserved; this change must not silently relicense them.

## Historical Cutoff and Pull Request #2727

Immediately before the license-only change, the project records the latest
commit that was publicly available under the prior repository terms and creates
an immutable historical tag or equivalent signed record for it.

As inspected on 2026-07-19, the candidate public head is
`60ce244fb6a65a79489b3f77299340afa501be24`. This is evidence, not a frozen
answer: if any additional code is pushed publicly before the cutoff, the final
record must advance to the actual latest public pre-license commit.

The historical record includes:

- the full commit hash;
- the date and repository URL;
- the license and package metadata present at that commit;
- the relationship to public draft PR #2727; and
- a statement that existing grants remain available and are not revoked.

The repository change is a separate, license-only pull request into `dev`. It
must land before further protected owner-authored work is published under the
new policy and before PR #2727 is merged. This ordering creates a clean future
cutoff, but it does not retroactively protect code already visible in #2727.

## Source and Artifact Notices

Every protected source archive, WebUI image, admin UI image, browser-extension
package, and other protected binary distribution includes:

- the unmodified Perimeter 1.0.1 terms;
- the completed release-specific Countdown grant with embedded AGPL terms;
- release ID, source revision, release date, and Countdown start date;
- a durable source repository or source-archive link;
- required copyright and third-party notices; and
- a statement that no tldw name, logo, icon, domain, signing identity, or store
  identity right is granted.

The root README, release pages, container descriptions, browser-store listing,
and product legal surfaces describe restricted-period releases as
“source-available,” not “open source.” They separately state that the server is
GPL-3.0-only and the canonical API contract is Apache-2.0.

An About/Legal surface is required before a protected binary artifact is
published, but not before the source-only license-cutoff commit. The existing
shared About component must be corrected rather than duplicated; its obsolete
repository and product wording must not ship in a protected release.

The initial notice states that no trademark rights are granted. A fuller
`TRADEMARKS` policy, including fair compatibility references and fork
rebranding, is published only after counsel review.

## Artifact and Image Separation

The GPL API image must contain neither protected frontend source nor protected
frontend build artifacts. Protected WebUI and admin UI images are built and
published separately. A Compose file or release bundle may aggregate separate
images when each image retains accurate metadata and notices.

The current production Dockerfile copies WebUI source into the API image; that
behavior must be removed before the API image can be represented as GPL-only.
WebUI and admin UI runtime stages must copy their governing legal records and
third-party notices into the final image.

Protected images and extension packages are not published automatically from
every `main` push or bare commit SHA. They are published only from an immutable
licensing release record after the release gate passes. API-only GPL image
publishing may remain independent.

## Verification and Failure Handling

### Merge gate

- Classify every changed file against the root scope map.
- Block third-party protected-path changes until the frontend CLA is active.
- Block third-party OpenAPI-changing contributions until the Apache contract
  grant is active.
- Reject imports or copies of GPL backend implementation code into protected
  frontend bundles; HTTP and WebSocket interoperability remains allowed.
- Preserve nested third-party notices and flag blanket license rewrites.
- Detect stale GPLv2, ISC, AGPL, BSL, or generic “open source” statements that
  conflict with the new path and release scope.

### Release gate

- Verify the release record, Countdown grant, AGPL text, protected paths,
  dates, source revision, and artifact digests agree.
- Generate a dependency and third-party license inventory for each artifact;
  resolve incompatible or unknown inputs instead of trusting a scanner's
  guessed classification.
- Verify the Countdown date is the second calendar anniversary at noon UTC and
  reject February 29 release dates.
- Build and inspect WebUI, admin UI, and every supported extension artifact for
  legal files, source links, notices, and accurate metadata.
- Verify the API image contains no protected source or protected build output.
- Require an About/Legal surface before publishing protected binaries.
- Require a human-reviewed release record; automation may verify but not invent
  legal dates or select a fallback license.

### Transition gate

- Periodically verify every public release record and report Countdown dates
  that are approaching or have started.
- Once AGPL starts, update descriptive indexes without modifying the original
  Countdown grant.
- Keep source for each released artifact available at the recorded revision so
  AGPL users can exercise the granted rights.

### Fail closed

If a path, contributor grant, date, release record, notice, source revision, or
artifact license is missing or inconsistent, protected publication stops. The
pipeline must not fall back to the root GPL license, prematurely claim AGPL, or
borrow another release's Countdown date.

If an incorrect protected artifact is published, preserve its legal record,
remove it from normal distribution when feasible, publish a notice, and issue a
new corrected release ID. Never silently replace an artifact under an existing
version and digest.

## Known Repository Corrections Required

Implementation must reconcile these observed inconsistencies rather than
letting them become accidental grants or misleading notices:

- `admin-ui/package.json` currently says ISC;
- `pyproject.toml` says GPL-3.0-only while server and UI surfaces still contain
  GPLv2 language;
- the extension README currently says AGPL;
- the production API Dockerfile copies WebUI source into the API image;
- the current GHCR workflow publishes API, WebUI, and admin images together on
  every `main` push;
- WebUI and admin UI runtime images do not currently bundle the new legal
  records;
- the shared About component contains an obsolete repository link and product
  wording;
- the Page Assist MIT notice and the PDF worker's Apache-2.0 notice must be
  preserved; and
- manually curated OpenAPI fallbacks and guards must remain protected until
  truly generated Apache replacements exist.

Robert Benjamin Jake Musser has stated that he is the original author of all
repository code. That supports relicensing his authored code but does not
replace the separate inventory of dependencies, generated files, copied assets,
and upstream notices. A Dependabot metadata change does not authorize
relicensing third-party material.

## Implementation Sequence

### Stage 1: License-only cutoff before counsel

**Goal:** Create an honest prospective cutoff without publishing unreviewed
custom terms.

**Success criteria:** The final public pre-license commit is recorded; root
scope and exact standard terms are present; release records are internally
consistent; metadata contradictions are corrected; third-party protected and
OpenAPI-changing contributions are paused; no protected binaries are published
without passing the artifact gate.

### Stage 2: Artifact and automation hardening

**Goal:** Make license scope survive builds and releases.

**Success criteria:** API and frontend images are separated; legal records and
notices are bundled; merge, release, and transition checks fail closed; product
legal surfaces and release descriptions are accurate.

### Stage 3: Counsel-reviewed public permissions

**Goal:** Add the desired community and customer permissions without weakening
the commercial-substitute boundary by accident.

**Success criteria:** Versioned Community Fork and Dedicated Customer grants,
frontend CLA, API contract contribution grant, trademark policy, and commercial
agreement template have been reviewed and published prospectively.

### Stage 4: Contribution and commercial operations

**Goal:** Reopen protected-path and API contract contributions and handle
exceptions consistently.

**Success criteria:** CLA and contract-grant checks are active; contributor
records are auditable; commercial exceptions identify exact releases and uses;
public documentation distinguishes standard rights from case-specific rights.

## Counsel Review Checklist

Counsel review should specifically address:

- whether the planned CLA validly supports Countdown's “each contributor”
  present future grant;
- the scope and enforceability of the Community Fork Grant's monetization
  boundary, donations, sponsorship, grants, and cross-subsidization;
- the line between a permitted dedicated customer engagement and a repeatable
  competing managed service;
- whether repository distribution creates GPL/Perimeter combination issues
  beyond the intended arms-length API and aggregate-image model;
- compatibility and notice requirements for bundled frontend dependencies,
  fonts, generated files, workers, and copied assets;
- package, container, browser-store, and interactive-UI notice sufficiency;
- trademark wording and fair compatibility references;
- patent permissions in the CLA and public additional grants;
- selective commercial-license ownership and signature authority; and
- whether any public history, contributor, or third-party material prevents the
  intended prospective grants.

If counsel recommends corrections, they apply prospectively through new
versioned documents and release records. Existing public grants are not edited,
revoked, or described as if they never existed.

## Authoritative References

- PolyForm Perimeter 1.0.1:
  <https://polyformproject.org/licenses/perimeter/1.0.1/>
- PolyForm Countdown License Grant 1.0.0:
  <https://polyformproject.org/licenses/countdown/1.0.0/>
- GNU Affero General Public License v3.0:
  <https://www.gnu.org/licenses/agpl-3.0.html>
- GNU General Public License v3.0:
  <https://www.gnu.org/licenses/gpl-3.0.html>
- Apache License 2.0:
  <https://www.apache.org/licenses/LICENSE-2.0>
- SPDX license-expression specification:
  <https://spdx.github.io/spdx-spec/v3.0.1/annexes/spdx-license-expressions/>
- npm package license metadata:
  <https://docs.npmjs.com/cli/v11/configuring-npm/package-json#license>
- OCI image annotations:
  <https://specs.opencontainers.org/image-spec/annotations/>

## Approval Record

In the design conversation on 2026-07-19, the user approved the protected path
boundary, GPL-3.0-only backend, Apache-2.0 OpenAPI contract, Perimeter 1.0.1,
release-specific Countdown grants to AGPL-3.0-only at each second anniversary,
one synchronized frontend licensing release ID, conservative pre-counsel
launch, later free-community and dedicated-customer grants, retained contributor
copyright with a frontend CLA, narrow API contract grant, selective case-by-case
commercial licensing, artifact separation, notice strategy, historical cutoff,
and fail-closed governance gates.
