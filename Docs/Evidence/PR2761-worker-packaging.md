# PR2761 worker packaging repair

**Tracking:** TASK-13013.7. **Date:** 2026-09-10.
**Source:** release worktree at `decdf9db7756031c379d5b019b0436711b9068b1`,
plus the two worker Dockerfile changes described here. This is a bounded
packaging correction; supply-chain release admission remains incomplete.

## Defect and correction

Both release worker recipes copied `Config_Files` from the repository root,
which no longer exists. They also omitted the separately located
`mcp_unified` and `tldw_profile_core` package inputs declared by root
`pyproject.toml`, together with the license corpus and profile data files.
Previously the release container-build matrix tested the app and two frontends,
so it did not exercise either worker recipe before release publication.

Both recipes now copy the canonical `tldw_Server_API/Config_Files` directory,
the two local package trees, and the license inputs. Copying the complete
profile package directory includes its declared schemas and fixtures as well
as Python source. A post-install import check fails the image build if either
local package is unavailable. These copy paths agree with the committed
supply-chain recipes; this repair preserves the release recipes' existing
Python dependency resolution and install profiles.

| Recipe | SHA256 after repair |
| --- | --- |
| `Dockerfiles/Dockerfile.worker` | `5388f47ab58a3b6b6d82dec638089a195e12ec180ffb1c5a8d6efd25b56e7006` |
| `Dockerfiles/Dockerfile.audio_gpu_worker` | `fe9e55a5694905952d0ec28a21858d963a0b8cfb74b49d416d7c3117ac846eed` |

## Verification

The new `test_worker_container_packaging.py` interprets each recipe's local
COPY inputs and checks their existence and coverage of manifest-declared
package/data paths. All four cases failed against the original recipes for
the expected omissions. After the correction, the focused suite passes
**25 tests**, with four inherited fixture/configuration/dependency warnings.

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/CI/test_worker_container_packaging.py \
  tldw_Server_API/tests/CI/test_release_workflow_contracts.py \
  tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py \
  -q --override-ini addopts=''
```

Separate temporary contexts were populated from tracked files selected by
each recipe's COPY instructions, excluding the API tests as the Docker
context does. Each context built a wheel using the project environment and
`pip wheel --no-deps --no-build-isolation --no-index`. No dependency resolution
or network acquisition occurred. Both wheels contain the two local package
initializers, **2 profile schema files**, and **45 profile fixture files**.
Separate Python `-I` subprocesses successfully imported both packages directly
from each wheel, with assertions checking that module origins were inside
that wheel. Installed third-party dependencies came from the project
environment; this is not a clean dependency-install or worker-startup test.

| Staged context | Wheel SHA256 |
| --- | --- |
| Embedding worker | `29527be3e11fc1cf21ed89be3a90234ae97ef7ef8a2c5df689dac90a62580f30` |
| Audio worker | `bbdb61fc21beb627ac293fbde76a803b672ac9f57d263c10f016a8255ea52084` |

Temporary wheel logs and `results.json` are retained at
`/var/folders/sn/m80n2j152t9gw3w8qwk2nykh0000gn/T/pr2761-worker-packaging-_xvryr4w`.
Regression logs are `/tmp/pr2761-worker-red.log` and
`/tmp/pr2761-worker-green-final.log`. Ruff, Black and scoped `git diff --check`
pass. Full Bandit output at `/tmp/pr2761-worker-bandit.json` contains two Low
B101 findings for ordinary pytest assertions; the run excluding B101 has zero
findings and errors. No production Python was changed.

## Supply-chain blockers retained

Read-only inspection confirms PR2869 still targets remote source
`78c3f92228c6411ee4637b9d2df9aa3b50aacdc8` with nine failed checks: five
image build/scan jobs, container aggregate, source scan, source aggregate,
and security-required. The worker image was built and scanned successfully
before its [policy decision rejected it](https://github.com/rmusser01/tldw_server/actions/runs/34362968533/job/102504379152).
The [source gate](https://github.com/rmusser01/tldw_server/actions/runs/34362968663/job/102505879755)
reports `vulnerability policy failed: source-python-root`.
[Dependency review](https://github.com/rmusser01/tldw_server/actions/runs/34362968402/job/102504643584)
reports ChromaDB 1.5.9 findings GHSA-f4j7-r4q5-qw2c,
GHSA-36p7-vc44-83pf, GHSA-2wm9-hf6c-p5cr and GHSA-xph7-9rjv-w5fr.
These results describe that remote candidate, not the current release branch.

The active supply-chain worktree has advanced to `073eb4c4a3` and contains
uncommitted TASK-13013.7.40 importer/applicability work. Its retained
TASK-13013.7.39 evidence reports 323 remaining backend scanner rows after its
own existing dispositions: 160 app, 3 embedding worker, and 160 audio worker.
The reported families are FFmpeg, ACL, SQLite, libsndfile, XML and TIFF.
These are unresolved scanner/applicability decisions, not an independent
confirmation of exploitability. That branch's frontend/source evidence and
runtime-admission work also require integration and fresh-candidate checks.

No active supply-chain files, exceptions, locks or workflows were changed or
integrated here. No scanner finding was suppressed. Full worker image builds,
worker workloads, immutable dependency resolution and digest-bound current
scans remain necessary before release. These packaging tests and wheel probes
do not close TASK-13013.7 or qualify either worker as a release artifact.

## Worker CI admission follow-up

The container-build gate now includes all five recipes: app, embedding worker,
audio worker, WebUI and Admin UI. Each backend image is loaded into the runner's
local Docker store, resolved to its immutable local image ID, and executed with
an overridden Python entrypoint. Python `-I` imports both local packages with
repository/environment Python paths ignored. Missing imports propagate a
nonzero exit to the matrix aggregate. The check uses no network, a read-only
filesystem, no capabilities and no new privileges.

Both frontend builds remain validation-only. All five builds retain
`push: false`; the workflow retains read-only permissions and its existing
license-admission conditions. Frontend images are not loaded or run by the
Python check. No publication workflow was changed.

The matrix/import contracts initially failed in two cases, with seven passing,
against the earlier workflow. The updated workflow passes **40 focused tests**
across packaging, release workflow, license-admission and Docker hardening
suites. The same four inherited warnings remain. Actionlint, Ruff, Black and
diff checks pass. Bandit finds only B101 pytest assertions; excluding that
test-specific rule leaves zero findings or scan errors.

The exact isolated Docker import command also passes against retained app
diagnostic image
`sha256:e01578be89000870b86c09164ad724160654f410a755a44fb84500857977b630`.
Docker reports amd64 execution on this arm64 host; no native-platform
performance claim is made. This is evidence that the new command runs against
a real image, not a new build or verification of either worker image. Fresh
remote execution must still succeed for all five candidate builds and all
three backend import checks.

## ChromaDB patched-version investigation

All four current GitHub Reviewed advisories explicitly list **no patched
version**. The affected ranges observed on September 10, 2026 are:

| Advisory | Affected range | Patched version |
| --- | --- | --- |
| [GHSA-f4j7-r4q5-qw2c / CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) | `>=1.0.0, <=1.5.9` | None |
| [GHSA-36p7-vc44-83pf / CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf) | `>=0.4.17, <=1.5.9` | None |
| [GHSA-2wm9-hf6c-p5cr / CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | `>=0.4.17, <=1.5.9` | None |
| [GHSA-xph7-9rjv-w5fr / CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr) | `>=0.5.0, <=1.5.9` | None |

[PyPI](https://pypi.org/project/chromadb/) and the
[latest upstream release](https://github.com/chroma-core/chroma/releases/tag/1.5.9)
both identify 1.5.9, published May 5, 2026. This equals the active supply-chain
branch's `uv.lock` resolution and its `chromadb>=1.5.9` compatibility floor.
The release branch currently permits `chromadb>=0.4.0` and has no root lock.
Simply raising that floor would not remove any of these four findings.

The upstream [authorization patch PR7602](https://github.com/chroma-core/chroma/pull/7602)
is still open with `mergedAt: null`. It concerns the tenant authorization
issues; it is not a published package fixing all four advisories. A downgrade
outside an advisory's listed range is also not established as a compatible
security fix. No bounded, published patched dependency version was identified,
so this investigation changes neither dependency declarations nor exceptions.
The scanner blocker remains explicit pending actual remediation or a separately
reviewed, evidence-backed applicability decision for the supported deployment.
