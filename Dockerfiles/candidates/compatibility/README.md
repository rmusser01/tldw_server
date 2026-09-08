# Retained candidate compatibility

TASK-13013.7.9 Stage 3 only. `candidate-compatibility.yml` downloads the retained
combined candidate from successful run `34177651966`, artifact `10038017399`,
producer commit `20fbadd2dc23ff3c186153f581a9b1dbd2731b24`. It does not build,
publish, promote, install dependencies into, or otherwise modify that image.

`retained-inputs.py` reuses the assembly metadata and payload checks. Fresh
GitHub API metadata must establish repository, branch, workflow, commit, outcome
and artifact/run binding. The complete OCI archive, raw baseline, source archive
and unchanged evaluator must match independently reviewed SHA-256 pins before
loading. Caller-supplied metadata is not an attestation by itself.

The existing OCI identity helper independently verifies subject/manifest/config
metadata and native platform/user. Docker's containerd store then loads the
attested archive directly; no cached rebuild or unattested replacement is used.
The loaded Docker `Id` must equal the OCI **subject**, not the distinct image
config digest. Execution uses that subject with `--pull never`; the evaluator's
`candidate_image` field records the config digest. Retained input, OCI and Docker
inspection records establish the mapping. Producer and consumer checkout commits
are recorded separately. Layer validation remains owned by the trusted
build/export/load path, not by the metadata helper alone.

## Baseline provenance and unchanged contract

`baseline-capabilities.txt` is the byte-for-byte retained inventory from
TASK-13013.7.6's original baseline image
`e90381ffb6a7a8f57783c11d54265d5ddcd740fc763118f043088d441ffa1f54`.
Its SHA-256 is `ec383730c4906414d7dda7d92ddeb7a9ffb1934f74c3c741de79906d37026b32`.
This is the original baseline, not an inventory regenerated from the candidate.

The existing `Helper_Scripts/Supply_Chain/ffmpeg_candidate.py` remains unchanged
and hash-pinned. The only accepted removals remain its reviewed category-specific
retirements: sonic/sonicls/v308/v408/v410 encoders; sonic/v308/v408/v410 decoders;
opengl/sdl/sdl2 muxers; pp filter; and hls **input protocol only**. No other removal
is accepted, including the HLS demuxer or output protocol.

The exact FFmpeg 9.0.1 source archive is downloaded and hash-verified, then mounted
read-only as an identity input. It is not extracted, built or executed here;
source signature authentication belongs to the retained producer evidence.

The evaluator runs as UID/GID 10001, with no network, a read-only root, no Linux
capabilities, no privilege escalation, bounded CPU/memory/processes, and only
temporary/evidence writes. Host application code is not mounted over `/app`;
only the hash-pinned evaluator and data inputs are mounted read-only. It retains
all capability inventories/deltas, software media probes, logs and exit status.
Failures remain fatal and available evidence is uploaded even on failure.

A pass establishes the existing software compatibility contract, not real GPU
or device compatibility, application import provenance, vulnerability clearance
or production admission. Source-aware Syft/Trivy/Grype reports and final security
review remain separate gates. Prior successful drawtext/subtitles controls also
reported a nonfatal fontconfig cache warning under a read-only root; this workflow
neither suppresses that warning nor claims to fix it.
