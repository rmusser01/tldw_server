# Conservative Frontend Licensing Cutoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a prospective, source-only frontend licensing cutoff using
unmodified PolyForm Perimeter 1.0.1, preserve the GPL backend and Apache API
contract boundaries, and fail closed on protected contributions and artifact
publishing until the later legal and release gates exist.

**Architecture:** Keep licensing authority in a root path-scope map plus a
verbatim legal corpus under `LICENSES/`. Add small static tests and one
standard-library CI classifier instead of a licensing framework. Enforce the
freeze from a base-controlled trusted workflow whose branch-qualified statuses
are source-bound in the `main` and `dev` rulesets, so a pull request cannot
replace the workflow or classifier that evaluates it.

**Tech Stack:** Markdown and plain-text legal records, JSON, Python 3 standard
library, pytest, FastAPI/OpenAPI 3.1 metadata, GitHub Actions YAML, Docker.

## Scope Split

This plan implements only the urgent pre-counsel cutoff. It deliberately does
not publish a Community Fork Grant, Dedicated Customer Grant, frontend CLA, API
contract contribution grant, detailed trademark policy, commercial agreement,
or protected binary release.

It also does not create a completed Countdown grant for an imaginary release.
The cutoff branch is an unreleased development snapshot governed by Perimeter.
Before the first tagged protected frontend release, a separate artifact-release
plan must create the immutable release record, embed the full AGPL-3.0-only text
in its completed Countdown grant, bundle notices into every artifact, add the
About/Legal release data, and re-enable only the protected publishing paths that
pass those checks.

This split avoids assigning a false publication date or a source revision that
cannot be known until the release exists. It does not weaken the approved
rolling rule: every tagged protected release still requires its own Countdown
grant before publication.

## Global Constraints

- Set TASK-12976 to In Progress before repository edits; it is the pre-created
  execution task linked to this plan and the approved design spec.
- Execute in a worktree created from the latest `origin/dev` with
  `superpowers:using-git-worktrees`; do not reuse the current dirty worktree.
- Make the licensing pull request a license-only change and merge it before PR
  #2727. Do not mix feature code from #2727 into the cutoff PR.
- Licensor: Robert Benjamin Jake Musser.
- Protected paths are exactly `admin-ui/**`, `apps/tldw-frontend/**`,
  `apps/extension/**`, and `apps/packages/ui/**`.
- Protected repository-authored material is source-available under the
  unmodified PolyForm Perimeter License 1.0.1 during this cutoff stage.
- The server implementation and unlisted repository-authored files are
  `GPL-3.0-only` prospectively.
- The generated canonical `/openapi.json` contract is `Apache-2.0`; backend
  implementation remains `GPL-3.0-only`.
- Previously public GPL, ISC, AGPL, and other grants remain available for the
  versions on which they were granted.
- Third-party material keeps its upstream license and notice.
- Do not describe a restricted frontend version as open source.
- Do not publish WebUI, admin UI, or extension binaries in this plan.
- Do not add a new package or service. Use Python's standard library and the
  repository's existing pytest, PyYAML, FastAPI, Docker, and Actions tooling.
- Run Bandit on changed Python implementation paths before completion.
- A human must write the pull request `Change summary` in their own words.

## Implementation Stages

### Stage 1: Legal boundary and historical record

**Goal:** Replace the single-license impression with an exact prospective scope
and preserve evidence of prior public grants.

**Success Criteria:** Verbatim license texts pass fixed checksums; the scope map
names only the four protected paths; the historical record captures current
public refs and expressly preserves all earlier grants.

**Tests:** `tldw_Server_API/tests/CI/test_licensing_policy.py`.

**Status:** Complete

### Stage 2: Public metadata and API contract

**Goal:** Make source, package, contribution, UI, and OpenAPI claims agree with
the new boundary.

**Success Criteria:** Protected manifests point to local notices; public copy
says source-available; `/openapi.json` declares Apache-2.0 for the contract and
GPL-3.0-only for the implementation.

**Tests:** Licensing policy tests, About component test, and targeted OpenAPI
contract test.

**Status:** Complete

### Stage 3: Temporary enforcement and publication freeze

**Goal:** Prevent unlicensed third-party protected changes and protected binary
publishing until counsel and artifact follow-ups are complete.

**Success Criteria:** The base-controlled trusted workflow blocks external
protected or contract-boundary changes through source-bound required statuses;
rolling GHCR publishing contains backend images only; the API image contains
no protected frontend.

**Tests:** CI classifier, workflow contract, release workflow, and Dockerfile
contract tests.

**Status:** Complete

### Stage 4: Verification and handoff

**Goal:** Prove that the cutoff is internally consistent and hand off a narrowly
scoped pull request.

**Success Criteria:** Targeted tests, Bandit, workflow linting, image inspection,
and whitespace checks pass; the Backlog task records results; PR #2727 remains
unmerged until the cutoff lands.

**Tests:** Commands in Task 6.

**Status:** Complete

---

### Task 1: Establish the authoritative legal corpus and historical boundary

**Files:**

- Create: `LICENSES/README.md`
- Create: `LICENSES/GPL-3.0-only.txt`
- Create: `LICENSES/AGPL-3.0-only.txt`
- Create: `LICENSES/Apache-2.0.txt`
- Create: `LICENSES/PolyForm-Perimeter-1.0.1.txt`
- Create: `LICENSES/PolyForm-Countdown-1.0.0-template.txt`
- Create: `LICENSES/history/pre-source-available.json`
- Create: `LICENSES/releases/README.md`
- Modify: `LICENSE`
- Create: `tldw_Server_API/tests/CI/test_licensing_policy.py`

**Interfaces:**

- Consumes: Approved design
  `Docs/superpowers/specs/2026-07-19-frontend-source-available-licensing-design.md`
  and the current public GitHub refs.
- Produces: The authoritative path map, immutable upstream legal texts, and
  `pre-source-available.json` consumed by all later tasks.

- [x] **Step 1: Re-verify the public pre-cutoff refs and fail on drift**

Run:

```bash
gh api repos/rmusser01/tldw_server/git/matching-refs/heads \
  --jq '.[] | select(.ref == "refs/heads/main" or .ref == "refs/heads/dev") | [.ref, .object.sha] | @tsv'
gh pr view 2727 --repo rmusser01/tldw_server \
  --json state,isDraft,headRefOid,updatedAt,url
```

Initial observation as of 2026-07-20 (superseded by the final pre-push refresh
recorded in Step 6):

```text
refs/heads/dev	29acaca8c781213e27b12066372df13855e2e7a6
refs/heads/main	7a23be3202e360f2d8e7cfe208e13ba406cf0507
{"headRefOid":"60ce244fb6a65a79489b3f77299340afa501be24","isDraft":true,"state":"OPEN","updatedAt":"2026-07-19T16:11:18Z","url":"https://github.com/rmusser01/tldw_server/pull/2727"}
```

If any SHA differs, stop before editing, record the newly observed public refs
in both the test and JSON record, and note the drift in the execution Backlog
task. Never preserve a stale SHA merely to match this plan.

- [x] **Step 2: Write the failing legal-boundary tests**

Create `tldw_Server_API/tests/CI/test_licensing_policy.py`:

```python
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


LICENSE_DIGESTS = {
    "LICENSES/PolyForm-Perimeter-1.0.1.txt": "5c7a5ccd847fcc285dda039e511ba013693fe979dfc5faee47f6fb59c7add337",
    "LICENSES/PolyForm-Countdown-1.0.0-template.txt": "eebf8d02412aa89d3d82fabdf6c67dfef04067e79f3f42d102a770c73590f2bf",
    "LICENSES/GPL-3.0-only.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
    "LICENSES/AGPL-3.0-only.txt": "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
    "LICENSES/Apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
PROTECTED_PATHS = [
    "admin-ui/**",
    "apps/tldw-frontend/**",
    "apps/extension/**",
    "apps/packages/ui/**",
]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_verbatim_license_corpus_matches_reviewed_upstream_bytes() -> None:
    for path, expected_digest in LICENSE_DIGESTS.items():
        actual_digest = sha256(Path(path).read_bytes()).hexdigest()
        assert actual_digest == expected_digest, path


def test_root_license_maps_only_the_approved_protected_paths() -> None:
    text = _read("LICENSE")
    for path in PROTECTED_PATHS:
        assert f"`{path}`" in text
    assert "`apps/**`" not in text
    assert "PolyForm Perimeter License 1.0.1" in text
    assert "GPL-3.0-only" in text
    assert "Apache-2.0" in text
    assert "previously public" in text
    assert "No trademark rights are granted" in text


def test_historical_record_preserves_public_refs_and_prior_grants() -> None:
    record = json.loads(_read("LICENSES/history/pre-source-available.json"))
    assert record["schema_version"] == 1
    assert record["recorded_on"] == "2026-07-21"
    assert record["repository"] == "https://github.com/rmusser01/tldw_server"
    assert record["public_refs"]["refs/heads/main"] == "7a23be3202e360f2d8e7cfe208e13ba406cf0507"
    assert record["public_refs"]["refs/heads/dev"] == "29acaca8c781213e27b12066372df13855e2e7a6"
    assert record["public_refs"]["refs/pull/2727/head"] == "e8bcc4c8b705df50a5f7e6299335ba8001ff4811"
    assert record["prior_grants_preserved"] is True
    assert record["ref_snapshot_is_exhaustive"] is False


def test_countdown_template_is_not_misrepresented_as_an_active_grant() -> None:
    readme = _read("LICENSES/releases/README.md")
    assert "No protected frontend release may be published" in readme
    assert "completed release-specific Countdown grant" in readme
    assert not any(
        child.is_dir()
        for child in Path("LICENSES/releases").iterdir()
    )
```

- [x] **Step 3: Run the test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_licensing_policy.py -q
```

Expected: FAIL because `LICENSES/` and the new root scope map do not exist.

- [x] **Step 4: Add the five verbatim upstream legal texts**

Fetch the exact plain-text bodies from these primary URLs, verify the listed
SHA-256 values, and add the verified bytes without project-authored headers or
edits:

```text
https://polyformproject.org/licenses/perimeter/1.0.1.txt
https://polyformproject.org/licenses/countdown/1.0.0.txt
https://www.gnu.org/licenses/gpl-3.0.txt
https://www.gnu.org/licenses/agpl-3.0.txt
https://www.apache.org/licenses/LICENSE-2.0.txt
```

Store Countdown under the filename ending in `-template.txt`. It remains the
official uncompleted template, not an active grant. The fixed digest test is
the guard against accidental edits.

- [x] **Step 5: Replace root `LICENSE` with the exact scope map**

Use this content:

```markdown
# tldw_server Licensing

Copyright (c) 2026 Robert Benjamin Jake Musser.

This repository is a multi-license work. This file is the authoritative scope
map. Exact license texts are under `LICENSES/`. A more specific third-party or
nested notice controls the material it identifies.

Required Notice: Copyright (c) 2026 Robert Benjamin Jake Musser.

## Protected Frontend Material

Repository-authored code, tests, build definitions, and original assets under
these paths are licensed under the PolyForm Perimeter License 1.0.1:

- `admin-ui/**`
- `apps/tldw-frontend/**`
- `apps/extension/**`
- `apps/packages/ui/**`

For a tagged protected frontend release, a completed release-specific PolyForm
Countdown License Grant under `LICENSES/releases/` additionally makes
AGPL-3.0-only available from the date and time stated in that grant. An
uncompleted Countdown template is not a license grant. PolyForm Perimeter does
not terminate when an AGPL option starts.

## Server and Unlisted Repository Material

`tldw_Server_API/**` and all other unlisted repository-authored code, tests,
scripts, and documentation are licensed GPL-3.0-only.

## Public API Contract

The canonical OpenAPI document generated at `/openapi.json`, and any complete
versioned snapshots expressly identified as such, are licensed Apache-2.0.
This does not license the server implementation under Apache-2.0.

## Documentation Inside Protected Paths

Markdown documentation remains GPL-3.0-only unless a more specific notice
expressly classifies it as protected release material.

## Third-Party Material

Third-party, copied, generated, and vendored material retains its upstream
terms. See `THIRD_PARTY_NOTICES.txt` and nested license files. No upstream work
is relicensed merely because it appears under a protected path or in an
artifact.

## Prior Public Versions

Licenses already granted for previously public versions remain available and
are not revoked or narrowed by this prospective scope change. See
`LICENSES/history/pre-source-available.json`.

## Trademarks

No trademark rights are granted. Software-license permission does not grant a
right to use tldw names, logos, icons, domains, signing identities, or store
identities as branding.
```

- [x] **Step 6: Add the legal-corpus README and historical JSON**

Create `LICENSES/README.md`:

```markdown
# License Corpus

The repository root `LICENSE` is the authoritative path-scope map. The files in
this directory preserve exact legal texts used by that map.

- `GPL-3.0-only.txt`: verbatim GNU GPL version 3 text.
- `AGPL-3.0-only.txt`: verbatim GNU AGPL version 3 text.
- `Apache-2.0.txt`: verbatim Apache License 2.0 text.
- `PolyForm-Perimeter-1.0.1.txt`: verbatim PolyForm Perimeter 1.0.1 text.
- `PolyForm-Countdown-1.0.0-template.txt`: verbatim uncompleted PolyForm
  Countdown 1.0.0 template for reference.

The Countdown template is not an active license grant. Only a completed,
release-specific Countdown grant under `LICENSES/releases/` can add a future
AGPL-3.0-only option to the release it identifies.

Do not edit a verbatim license file. Add a new versioned file when a different
upstream license version is intentionally adopted.
```

Create `LICENSES/history/pre-source-available.json` with the refs refreshed
immediately before the final cutoff push:

```json
{
  "schema_version": 1,
  "record_type": "pre_source_available_history",
  "recorded_on": "2026-07-21",
  "repository": "https://github.com/rmusser01/tldw_server",
  "public_refs": {
    "refs/heads/dev": "29acaca8c781213e27b12066372df13855e2e7a6",
    "refs/heads/main": "7a23be3202e360f2d8e7cfe208e13ba406cf0507",
    "refs/pull/2727/head": "e8bcc4c8b705df50a5f7e6299335ba8001ff4811"
  },
  "pull_request": "https://github.com/rmusser01/tldw_server/pull/2727",
  "prior_grants_preserved": true,
  "ref_snapshot_is_exhaustive": false,
  "statement": "All permissions already granted on code made public before the prospective cutoff remain available. This ref snapshot is evidence, not an exhaustive list of every previously public commit or artifact."
}
```

Create `LICENSES/releases/README.md` with no subdirectories:

```markdown
# Protected Frontend Release Records

No protected frontend release may be published from this cutoff alone.

Before publication, a later artifact-release plan must add an immutable release
directory containing a concrete release ID, exact source revision, release
date, second-calendar-anniversary start date at 12:00 noon UTC, completed
release-specific Countdown grant with the full AGPL-3.0-only text, checksums,
required notices, and artifact-verification results.

An uncompleted template or this README is not a Countdown grant. Published
release records are append-only; corrections use a new release ID and a
superseding record.
```

- [x] **Step 7: Run the legal-boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_licensing_policy.py -q
```

Expected: PASS.

- [x] **Step 8: Commit the legal boundary**

```bash
git add LICENSE LICENSES tldw_Server_API/tests/CI/test_licensing_policy.py
git commit -m "chore: establish prospective frontend license boundary"
```

### Task 2: Align package, contribution, notice, and product metadata

**Files:**

- Create: `admin-ui/LICENSE`
- Create: `apps/tldw-frontend/LICENSE`
- Create: `apps/extension/LICENSE`
- Create: `apps/packages/ui/LICENSE`
- Modify: `admin-ui/package.json`
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/extension/package.json`
- Modify: `apps/packages/ui/package.json`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `THIRD_PARTY_NOTICES.txt`
- Modify: `apps/extension/README.md`
- Modify: `apps/tldw-frontend/components/landing/LandingLayout.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/about.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/about.test.tsx`
- Modify: `tldw_Server_API/tests/CI/test_licensing_policy.py`

**Interfaces:**

- Consumes: Root scope and legal paths created in Task 1.
- Produces: Consistent human- and package-visible source-available notices;
  later artifact work will reuse the four local `LICENSE` files.

- [x] **Step 1: Extend the failing policy tests for package and public copy**

Append:

```python
PROTECTED_PACKAGES = [
    "admin-ui",
    "apps/tldw-frontend",
    "apps/extension",
    "apps/packages/ui",
]


def test_protected_packages_use_local_license_notices() -> None:
    for package in PROTECTED_PACKAGES:
        manifest = json.loads(_read(f"{package}/package.json"))
        assert manifest["license"] == "SEE LICENSE IN LICENSE"
        notice = _read(f"{package}/LICENSE")
        assert "PolyForm Perimeter License 1.0.1" in notice
        assert "Required Notice:" in notice
        assert "LICENSES/releases" in notice
        assert "No trademark rights are granted" in notice


def test_public_copy_distinguishes_server_and_frontend_terms() -> None:
    root_readme = _read("README.md")
    extension_readme = _read("apps/extension/README.md")
    landing = _read("apps/tldw-frontend/components/landing/LandingLayout.tsx")
    contributing = _read("CONTRIBUTING.md")

    assert "Frontend: source-available" in root_readme
    assert "Server: GPL-3.0-only" in root_readme
    assert "source-available under PolyForm Perimeter 1.0.1" in extension_readme
    assert "Frontend source-available" in landing
    assert "Temporary licensing contribution gate" in contributing
    assert "apps/packages/ui/**" in contributing
    assert "Open source under GPL v2.0" not in landing


def test_third_party_notices_preserve_frontend_upstream_terms() -> None:
    notices = _read("THIRD_PARTY_NOTICES.txt")
    assert "Host project: multi-license; see LICENSE" in notices
    assert "apps/packages/ui/src/Licenses/Page-Assist-LICENCE" in notices
    assert "apps/extension/public/pdf.worker.min.mjs" in notices
```

- [x] **Step 2: Add a failing About/Legal component test**

Append inside the existing `describe("AboutApp")` block:

```tsx
  it("links to the current repository and frontend license terms", async () => {
    mocks.getOllamaURL.mockResolvedValue("http://127.0.0.1:8000/")
    mocks.fetcher.mockResolvedValue({
      ok: true,
      json: async () => ({ info: { version: "1.2.3" } })
    })

    renderWithQueryClient(<AboutApp />)

    const repositoryLink = await screen.findByRole("link", {
      name: "tldw_server on GitHub"
    })
    expect(repositoryLink).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server"
    )

    expect(
      screen.getByRole("link", { name: "Source-available frontend terms" })
    ).toHaveAttribute(
      "href",
      "https://github.com/rmusser01/tldw_server/blob/dev/LICENSE"
    )
  })
```

- [x] **Step 3: Run the targeted tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_licensing_policy.py -q
cd apps/packages/ui
bunx vitest run src/components/Option/Settings/__tests__/about.test.tsx
```

Expected: policy assertions fail on missing package notices and stale wording;
the About test fails because both links are absent or stale.

- [x] **Step 4: Add identical local protected-package notices and metadata**

Each local `LICENSE` must contain this text, with only the relative root path
adjusted when useful:

```text
tldw Protected Frontend Licensing Notice

Repository-authored code, tests, build definitions, and original assets in this
package are source-available under the PolyForm Perimeter License 1.0.1:
https://polyformproject.org/licenses/perimeter/1.0.1/

Markdown documentation in this package remains GPL-3.0-only unless a more
specific notice expressly classifies it as protected release material.

Required Notice: Copyright (c) 2026 Robert Benjamin Jake Musser.

A tagged release may include a completed release-specific PolyForm Countdown
License Grant under LICENSES/releases that makes AGPL-3.0-only available from
the date and time stated in that grant. An uncompleted template is not a grant.

Third-party material retains its upstream terms. See the repository root
LICENSE, LICENSES directory, THIRD_PARTY_NOTICES.txt, and nested notices.

No trademark rights are granted.
```

Set this exact field in all four package manifests:

```json
"license": "SEE LICENSE IN LICENSE"
```

Do not change either voice SDK manifest in this task.

- [x] **Step 5: Correct repository, contribution, extension, and landing copy**

Update the root README badge/intro and License section to use these exact
labels:

```text
Server: GPL-3.0-only
Frontend: source-available under PolyForm Perimeter 1.0.1
OpenAPI contract: Apache-2.0
```

Link the detailed statement to root `LICENSE`; do not call the entire
repository open source.

Add this section near the top of `CONTRIBUTING.md`:

```markdown
Temporary licensing contribution gate
---------------------------------------

Until the counsel-reviewed frontend CLA and API contract contribution grant are
published, pull requests authored by anyone other than Robert Benjamin Jake
Musser cannot modify:

- `admin-ui/**`
- `apps/tldw-frontend/**`
- `apps/extension/**`
- `apps/packages/ui/**`
- root licensing governance files; or
- the public API declaration boundary under `tldw_Server_API/app/api/v1/**`
  and `tldw_Server_API/app/main.py`.

This is a temporary, conservative intake pause. Backend and documentation
contributions outside those boundaries remain welcome under the normal GPL
process. Do not place an exception in an issue, review comment, commit message,
or pull request; only a published contributor agreement can reopen a paused
boundary.
```

Replace the extension's bare `AGPL` statement with:

```markdown
## License

Repository-authored extension code is source-available under PolyForm Perimeter
1.0.1. Tagged releases may add a release-specific, time-delayed
AGPL-3.0-only option. See [LICENSE](LICENSE), the repository root `LICENSE`, and
`LICENSES/releases/` for the terms that apply to a specific version.

Third-party material retains its upstream terms.
```

Change the landing footer heading from `Open Source` to `Source & Community`
and the bottom line to:

```tsx
<p>Server GPL-3.0-only. Frontend source-available. No telemetry. No data collection.</p>
```

- [x] **Step 6: Correct About/Legal links with the smallest UI change**

In `about.tsx`, keep the current component and query. Rename the returned
`ollama` field to `serverVersion`, change the displayed repository link to
`https://github.com/rmusser01/tldw_server` with text `tldw_server on GitHub`,
and append this `Descriptions` item:

```tsx
{
  key: 4,
  label: "License",
  children: (
    <a
      href="https://github.com/rmusser01/tldw_server/blob/dev/LICENSE"
      target="_blank"
      rel="noreferrer"
      className="text-primary">
      Source-available frontend terms
    </a>
  )
}
```

Change the fallback version label to `tldw Assistant Version`. Do not add a
new legal component or runtime configuration system; the future artifact plan
will add release-specific values.

- [x] **Step 7: Correct third-party notices without changing upstream files**

Change the host line to `Host project: multi-license; see LICENSE`. Correct the
PDF.js path to `apps/extension/public/pdf.worker.min.mjs`. Add this separate
entry, renumbering later entries as needed:

```text
Page Assist
Author/Year : Page Assist contributors
License      : MIT
Files        : apps/packages/ui/src/ and apps/extension/ portions derived from Page Assist
License file : apps/packages/ui/src/Licenses/Page-Assist-LICENCE
URL          : https://github.com/n4ze3m/page-assist
```

Do not add Perimeter or AGPL headers to either upstream file.

- [x] **Step 8: Run the metadata and UI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/CI/test_licensing_policy.py -q
cd apps/packages/ui
bunx vitest run src/components/Option/Settings/__tests__/about.test.tsx
```

Expected: PASS.

- [x] **Step 9: Commit the metadata disclosure**

```bash
git add README.md CONTRIBUTING.md THIRD_PARTY_NOTICES.txt admin-ui/LICENSE admin-ui/package.json apps/tldw-frontend/LICENSE apps/tldw-frontend/package.json apps/tldw-frontend/components/landing/LandingLayout.tsx apps/extension/LICENSE apps/extension/package.json apps/extension/README.md apps/packages/ui/LICENSE apps/packages/ui/package.json apps/packages/ui/src/components/Option/Settings/about.tsx apps/packages/ui/src/components/Option/Settings/__tests__/about.test.tsx tldw_Server_API/tests/CI/test_licensing_policy.py
git commit -m "docs: disclose source-available frontend terms"
```

### Task 3: License the generated OpenAPI contract under Apache-2.0

**Files:**

- Modify: `tldw_Server_API/app/main.py:1659`
- Modify: `tldw_Server_API/app/main.py:1998`
- Modify: `tldw_Server_API/tests/Services/test_openapi_contracts.py`

**Interfaces:**

- Consumes: FastAPI's existing `license_info` and `custom_openapi()` path.
- Produces: `/openapi.json` fields `info.license.identifier` and
  `info.x-server-code-license` used by clients and policy tests.

- [x] **Step 1: Write the failing OpenAPI license test**

Add after the `openapi_spec` fixture helpers:

```python
@pytest.mark.integration
def test_openapi_contract_declares_contract_and_code_licenses(
    openapi_spec: dict[str, Any],
) -> None:
    info = openapi_spec["info"]

    assert info["license"] == {
        "name": "Apache License 2.0 (OpenAPI contract only)",
        "identifier": "Apache-2.0",
    }
    assert info["x-server-code-license"] == "GPL-3.0-only"
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses -q
```

Expected: FAIL because the generated schema omits the app's stale GPLv2
`license_info` and has no server-code extension.

- [x] **Step 3: Update the FastAPI and custom OpenAPI metadata**

Replace the stale license block with:

```python
    license_info={
        "name": "Apache License 2.0 (OpenAPI contract only)",
        "identifier": "Apache-2.0",
    },
```

In the existing `get_openapi(...)` call, pass through the application metadata:

```python
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
```

After `_ensure_openapi_operation_tags_declared(openapi_schema)`, add:

```python
    openapi_schema.setdefault("info", {})["x-server-code-license"] = "GPL-3.0-only"
```

Also correct the adjacent repository URLs from the obsolete `cpacker` location
to `https://github.com/rmusser01/tldw_server` and its `/issues` page.

- [x] **Step 4: Run the targeted OpenAPI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses tldw_Server_API/tests/Services/test_openapi_contracts.py::test_custom_openapi_reuses_cached_schema_for_tag_declarations -q
```

Expected: PASS.

- [x] **Step 5: Commit the API contract metadata**

```bash
git add tldw_Server_API/app/main.py tldw_Server_API/tests/Services/test_openapi_contracts.py
git commit -m "fix: declare OpenAPI contract Apache license"
```

### Task 4: Enforce the temporary contribution freeze with the trusted gate

**Canonical implementation:** TASK-12977 and
`Docs/superpowers/plans/2026-07-20-base-controlled-frontend-license-gate-implementation-plan.md`.

**Files:**

- `Helper_Scripts/ci/check_frontend_license_gate.py`
- `.github/workflows/frontend-license-gate.yml`
- `.github/workflows/actionlint.yml`
- `.github/workflows/frontend-required.yml` (restored to its pre-gate behavior)
- `tldw_Server_API/tests/CI/test_frontend_license_gate.py`
- `tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py`
- `tldw_Server_API/tests/CI/test_frontend_required_workflow.py`
- `Docs/superpowers/evidence/TASK-12977/*.json`

**Interfaces:**

- Consumes GitHub-supplied immutable pull-request metadata and bounded,
  NUL-delimited changed paths from `git diff --name-only -z --no-renames`.
- Produces branch-qualified exact-head statuses
  `frontend-license-policy/trusted/main` and
  `frontend-license-policy/trusted/dev`.
- Runs only the trusted workflow and classifier from the base-controlled
  `main` revision; never checks out or executes pull-request content.

- [x] **Step 1: Reject the original PR-controlled/newline design**

  Independent review proved that a mutable `pull_request` workflow and
  line-delimited filenames were not an adequate trust boundary. The replacement
  design and rollout plan are recorded in TASK-12977.

- [x] **Step 2: Land the base-controlled workflow and NUL-safe classifier**

  Bootstrap PR #2753 placed the reviewed `pull_request_target` workflow,
  standard-library classifier, and exact contract tests on `main`. The missing
  human-written Change summary on that PR remains recorded policy
  noncompliance.

- [x] **Step 3: Prove and require source-bound branch contexts**

  Temporary PR #2754 proved `/main` and was closed unmerged with its branch
  removed. Draft licensing PR #2755 proved `/dev`. Main ruleset `5653432`
  and dev ruleset `19362594` require only their matching contexts from GitHub
  Actions App `15368`, with no bypass actors.

- [x] **Step 4: Reconcile the licensing branch**

  The trusted implementation was carried byte-for-byte from merged `main`;
  `frontend-required.yml` was restored byte-for-byte from `origin/dev`; and
  a negative regression test forbids licensing enforcement or status
  publication in that PR-controlled workflow.

- [x] **Step 5: Verify the trusted contract**

  The focused matrix passed 40/40; pinned actionlint 1.7.12, Ruff, Black,
  Bandit, deterministic owner/external cases, evidence assertions, and diff
  hygiene passed. Independent security review found no code or security
  findings. Public ruleset snapshots are stored under
  `Docs/superpowers/evidence/TASK-12977/`.

### Task 5: Remove protected code from the API image and suspend protected publishing

**Files:**

- Modify: `Dockerfiles/Dockerfile.prod`
- Modify: `.github/workflows/publish-ghcr-main.yml`
- Modify: `Dockerfiles/README.md`
- Modify: `Docs/Development/Container_Image_Lifecycle.md`
- Modify: `Docs/Development/Packaging_and_Distribution_Strategy.md`
- Modify: `Docs/Development/PyPI_Publishing.md`
- Modify: `Docs/Development/Release_Process.md`
- Modify: `Docs/Release_Checklist.md`
- Modify: `tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py`
- Modify: `tldw_Server_API/tests/CI/test_release_workflow_contracts.py`

**Interfaces:**

- Consumes: Root legal corpus from Task 1 and existing container workflows.
- Produces: A GPL backend image with bundled notices and a rolling publish matrix
  that contains only the backend `app` image.

- [x] **Step 1: Write failing API-image isolation assertions**

Append to `test_docker_quickstart_hardening.py`:

```python
def test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files():
    text = _read_text("Dockerfiles/Dockerfile.prod")

    _require(
        "apps/tldw-frontend" not in text,
        "Expected the GPL API image to exclude protected frontend source",
    )
    for required_copy in (
        "LICENSE /app/LICENSE",
        "LICENSES /app/LICENSES",
        "THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt",
    ):
        _require(required_copy in text, f"Expected API image legal copy: {required_copy}")
```

- [x] **Step 2: Change the failing rolling-publish contract test**

Replace `test_publish_ghcr_main_matrix_remains_app_webui_admin_ui` with:

```python
def test_publish_ghcr_main_matrix_is_backend_only_during_frontend_freeze() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    matrix = workflow["jobs"]["publish-ghcr-main"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == ["app"]
    assert matrix[0]["dockerfile"] == "Dockerfiles/Dockerfile.prod"
```

- [x] **Step 3: Run the container policy tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py::test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files tldw_Server_API/tests/CI/test_release_workflow_contracts.py::test_publish_ghcr_main_matrix_is_backend_only_during_frontend_freeze -q
```

Expected: FAIL because the API image copies the WebUI and the workflow publishes
all three images.

- [x] **Step 4: Isolate the API Dockerfile**

Delete:

```dockerfile
COPY --chown=appuser:appuser apps/tldw-frontend /app/apps/tldw-frontend
```

Add the legal corpus to both the builder input and runtime image. Replace the
builder metadata copy and add the license directory as follows:

```dockerfile
COPY pyproject.toml README.md LICENSE /app/
COPY LICENSES /app/LICENSES
```

The runtime copy block must contain:

```dockerfile
COPY --chown=appuser:appuser LICENSE /app/LICENSE
COPY --chown=appuser:appuser LICENSES /app/LICENSES
COPY --chown=appuser:appuser THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt
```

Do not copy any of the four protected path families into the API image.

- [x] **Step 5: Make rolling GHCR publishing backend-only**

Keep the existing matrix structure but remove its `webui` and `admin-ui`
entries. Keep the `app` entry, `main` and SHA tags, cache, and attestations
unchanged. Build-only validation in `container-build-check.yml` stays enabled
for all three images because building a PR artifact is not publishing it.

- [x] **Step 6: Update container lifecycle documentation**

Update both container docs to state:

```text
During the pre-counsel frontend licensing freeze, WebUI and Admin UI images are
build-checked on pull requests but are not pushed by publish-ghcr-main or
publish-docker. publish-ghcr-main publishes only the GPL backend app image.
Protected image publishing resumes only through a release-specific licensing
workflow after artifact notices and image separation pass their later gate.
```

In the coverage matrix, set WebUI and admin UI `publish-ghcr-main` entries to
`No (licensing freeze)`. Remove claims that their `main` or SHA tags are
currently published.

- [x] **Step 7: Run container policy and release workflow tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py -q
```

Expected: PASS.

- [x] **Step 8: Commit image isolation and publishing freeze**

```bash
git add Dockerfiles/Dockerfile.prod Dockerfiles/README.md Docs/Development/Container_Image_Lifecycle.md .github/workflows/publish-ghcr-main.yml tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py
git commit -m "ci: suspend protected frontend image publishing"
```

### Task 6: Verify the cutoff and prepare the license-only pull request

**Files:**

- Modify: `backlog/tasks/task-12976 - Implement-conservative-frontend-licensing-cutoff.md`

**Interfaces:**

- Consumes: Tasks 1–5.
- Produces: Verification evidence and a license-only PR ready for human review
  and human-authored change summary.

- [x] **Step 1: Run the complete targeted test set**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/CI/test_licensing_policy.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate.py \
  tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py \
  tldw_Server_API/tests/CI/test_frontend_required_workflow.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py::test_openapi_contract_declares_contract_and_code_licenses \
  tldw_Server_API/tests/Services/test_openapi_contracts.py::test_custom_openapi_reuses_cached_schema_for_tag_declarations \
  -q
```

Expected: PASS.

- [x] **Step 2: Run the protected About test**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Settings/__tests__/about.test.tsx
```

Expected: PASS.

- [x] **Step 3: Run security and static integrity checks**

Run:

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/ci/check_frontend_license_gate.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_frontend_license_cutoff.json
git diff --check origin/dev...HEAD
rg -n "Open source under GPL v2.0|Host project license: GNU General Public License v3.0|\"license\": \"ISC\"|^AGPL$" README.md CONTRIBUTING.md THIRD_PARTY_NOTICES.txt admin-ui/package.json apps/tldw-frontend/package.json apps/tldw-frontend/components/landing/LandingLayout.tsx apps/extension/package.json apps/extension/README.md apps/packages/ui/package.json
```

Expected: Bandit exits `0`; `git diff --check` has no output; the stale-language
scan has no matches. Do not suppress a new Bandit finding.

- [x] **Step 4: Lint the changed workflows**

Run the same pinned `actionlint` installer used by
`.github/workflows/actionlint.yml`, then run:

```bash
./actionlint -color -config-file .github/actionlint.yaml \
  .github/workflows/actionlint.yml \
  .github/workflows/frontend-license-gate.yml \
  .github/workflows/frontend-required.yml \
  .github/workflows/publish-ghcr-main.yml
```

Expected: exit `0` with no errors.

- [x] **Step 5: Build and inspect the API image**

Run:

```bash
docker build -f Dockerfiles/Dockerfile.prod -t tldw-server:license-cutoff .
docker run --rm --entrypoint sh tldw-server:license-cutoff -c 'set -eu; test ! -e /app/admin-ui; test ! -e /app/apps/tldw-frontend; test ! -e /app/apps/extension; test ! -e /app/apps/packages/ui; test -f /app/LICENSE; test -f /app/LICENSES/GPL-3.0-only.txt; test -f /app/LICENSES/AGPL-3.0-only.txt; test -f /app/LICENSES/PolyForm-Perimeter-1.0.1.txt; test -f /app/THIRD_PARTY_NOTICES.txt'
```

Expected: build succeeds and the inspection command exits `0`.

- [x] **Step 6: Review the diff for scope and legal-record integrity**

Run:

```bash
git diff --name-status origin/dev...HEAD
git log --oneline --decorate origin/dev..HEAD
```

Expected: only the files listed in Tasks 1–5, the execution Backlog task, and
TASK-12977's reviewed plan, design, task, public ruleset evidence, and append-only
progress record are present. There must be no feature implementation from PR
#2727, no active Community or Dedicated Customer grant, no frontend CLA, no
completed Countdown grant, and no protected binary artifact.

- [x] **Step 7: Finalize the execution Backlog task**

Record the exact test, Bandit, actionlint, and image-inspection results; list
modified files; link the approved spec and this plan; state that full protected
artifact publishing and custom grants are intentionally deferred. Mark the task
Done only after every required check passes.

- [x] **Step 8: Commit the finalized task record**

```bash
git add 'backlog/tasks/task-12976 - Implement-conservative-frontend-licensing-cutoff.md'
git commit -m "chore: complete frontend licensing cutoff task"
```

- [x] **Step 9: Open the license-only pull request into `dev`**

Push the worktree branch and open a draft PR targeting `dev`. The human
requester must write the required `Change summary` explaining what changed and
why. Do not merge PR #2727 or publish protected artifacts until this cutoff PR
is reviewed and merged.

After the cutoff merges, rebase PR #2727 onto the new `dev` head before its own
merge. This sequencing does not change the prior licenses of code already
public in #2727; it only establishes the governing terms for future work.

## Follow-On Plans Required

1. **Protected artifact release and Countdown automation:** immutable release
   IDs, actual publication dates, second-anniversary noon-UTC computation,
   embedded AGPL text, checksums, signed tags, source archives, About/Legal
   release data, WebUI/admin runtime license bundles, extension packages, and a
   controlled publish workflow.
2. **Counsel-reviewed contribution and exception terms:** frontend CLA, narrow
   Apache API contract grant, Community Fork Grant, Dedicated Customer Grant,
   detailed trademark policy, and commercial-license template.

Neither follow-on may silently edit an already published legal record. New
permissions and corrections use new versioned files prospectively.
