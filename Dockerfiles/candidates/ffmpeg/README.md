# FFmpeg 9 candidate evaluation

This directory is an evaluation artifact for TASK-13013.7.6. It does not replace a production Dockerfile, dependency lock, admission policy, or runtime probe.

## Candidate provenance

The recipe uses the pinned production-compatible base image and authenticates every downloaded build input before extraction:

- FFmpeg `ffmpeg-9.0.1.tar.xz`: SHA-256 `cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635`. The recipe verifies the detached RSA signature in a fresh GnuPG home against fingerprint `FCF986EA15E6E293A5644F10B4322F04D67658D8`; the signature and key files are hash-pinned too.
- libplacebo `7.360.1`: signed tag object `719cc95244a1f1d648dd72459822e026e6530f22`, commit `cee9b076f2c63104ccfd497fa79c39a867293ec4`, archive SHA-256 `6f8fa218cbafd8e5f50b8a82d918e1d8bbb92f9f980820bc0b34d92e9b79e484`. GitHub's API reported the upstream maintainer tag signature valid. This is not described as local detached-signature verification of the commit archive; the downloaded archive is locally hash-verified.
- Vulkan-Headers commit `450bd2232225d6c7728a4108055ac2e37cef6475`, archive SHA-256 `26df9841c30806a994e2fdf42f7c87bcb1ced9db9a06033469123939fb3fa075`. This is libplacebo's exact gitlink and is a build input, not a runtime library.

Trixie's packaged libplacebo `7.349.0` is below FFmpeg 9.0.1's required `7.351.0`. The separately authenticated `7.360.1` build resolves that constraint. Its Meson build uses `--wrap-mode=nodownload`, system Python, explicit OpenGL/Vulkan/glslang/lcms enablement, and no demos. No submodule update or Sid/Forky binary package is used.

The FFmpeg configuration is a fixed list copied from the captured baseline build configuration, with only these upstream compatibility decisions:

- removed `--disable-omx` is omitted; OMX was already disabled, so this does not remove an enabled feature;
- removed `--enable-libglslang` is not passed; the Vulkan shaders are compiled with the installed external `glslangValidator` tool;
- the retired `pp`/libpostproc facility is the only allowed capability removal;
- libplacebo remains enabled using the isolated `/opt/tldw-ffmpeg9` prefix.

`ffmpeg`, `ffprobe`, and `ffplay` are all retained. The build fails when an enabled dependency cannot be configured or compiled.

## Build and evaluate

Build this candidate from the repository root:

```bash
docker build \
  --file Dockerfiles/candidates/ffmpeg/Dockerfile \
  --tag tldw-ffmpeg9-candidate:task-13013-7-6 \
  .
docker image inspect --format '{{.Id}}' tldw-ffmpeg9-candidate:task-13013-7-6
```

The builder derives a complete library inventory from the final ELF files, fails on unresolved libraries or old libav ABI, maps canonical library paths to Debian package owners, and writes exact `package=version` requirements. The runtime stage installs that derived set and verifies the installed versions. Evidence under `/opt/tldw-ffmpeg9/share/candidate/` keeps source identities, signature status, exact `-buildconf`, all builder package versions, required runtime package versions, installed runtime package versions, and build/runtime `ldd` results. Build packages and runtime components are deliberately recorded separately.

Run the helper inside the exact image under evaluation. Mount only the helper, baseline, and pinned source archive read-only. The container runs without network access, Linux capabilities, or root privileges and writes only to its own `/tmp`; copy the evidence out after the evaluation container stops:

```bash
(
  set -eu
  candidate_image="$(docker image inspect --format '{{.Id}}' tldw-ffmpeg9-candidate:task-13013-7-6)"
  candidate_container="$(
    docker create \
      --network none \
      --cap-drop ALL \
      --security-opt no-new-privileges:true \
      --user 65534:65534 \
      --mount type=bind,src="$PWD/Helper_Scripts/Supply_Chain/ffmpeg_candidate.py",dst=/work/ffmpeg_candidate.py,readonly \
      --mount type=bind,src=/absolute/path/to/baseline-capabilities.txt,dst=/evidence/baseline-capabilities.txt,readonly \
      --mount type=bind,src=/absolute/path/to/ffmpeg-9.0.1.tar.xz,dst=/evidence/ffmpeg-9.0.1.tar.xz,readonly \
      "$candidate_image" \
      python /work/ffmpeg_candidate.py \
        --baseline /evidence/baseline-capabilities.txt \
        --source-archive /evidence/ffmpeg-9.0.1.tar.xz \
        --candidate-image "$candidate_image" \
        --output-dir /tmp/candidate-output
  )"
  test -n "$candidate_container"

  evaluation_status=0
  docker start --attach "$candidate_container" || evaluation_status=$?
  if ! docker cp "$candidate_container:/tmp/candidate-output" /absolute/path/to/candidate-output; then
    printf 'Evidence copy failed; container retained for recovery: %s\n' "$candidate_container" >&2
    exit 1
  fi
  if ! docker rm "$candidate_container"; then
    printf 'Evidence copied, but container removal failed: %s\n' "$candidate_container" >&2
    exit 1
  fi
  exit "$evaluation_status"
)
```

The caller is responsible for obtaining the image ID with `docker image inspect` immediately before this command. `docker create` uses that immutable ID rather than the mutable tag, and its successful result supplies the only container ID used by later commands. The helper validates the image ID's syntax but cannot independently inspect the container engine from inside the network-disabled container. A failed create cannot select a pre-existing container. A failed evidence copy exits nonzero and retains the stopped container, whose ID is printed for recovery. The container is removed only after evidence extraction succeeds; the subshell then returns the evaluator's original exit status.

The recipe performs the actual GnuPG signature verification before extracting FFmpeg. The helper independently enforces the pinned source-archive hash; it deliberately does not accept a saved GnuPG status log as authentication. Such logs may be retained as build attachments, but their text alone is not cryptographic proof.

The helper preserves raw build and capability listings, parses aliases and input/output protocols separately, reports every unexplained baseline capability loss, and runs real media workflows: WAV resampling; MP3, FLAC, Opus, and AAC round trips; H.264/AAC MP4; ffprobe metadata; thumbnail extraction; and segment/concat. It checks decoded, non-silent sample data rather than relying on command exit codes alone.

These are software-only probes. They do not prove access to or behavior of a real GPU, capture device, audio device, hardware encoder, or driver.

## Promotion blockers

This candidate must not be promoted unless all of the following are attached to the immutable image ID and independently reviewed:

- no unexplained capability delta other than the retired `pp` filter;
- every synthetic probe passes and any required real GPU/device validation is completed separately;
- runtime dependency evidence contains no missing library, Debian FFmpeg package, libpostproc, or old libav ABI;
- fresh SBOM and vulnerability scans explicitly identify both source-built FFmpeg and libplacebo.

Trivy documents C/C++ package coverage through supported package metadata such as Conan lock files. A scan that does not identify these source-built components is a coverage gap and blocks promotion; it is not evidence of a clean or no-CVE result. This candidate does not invent Conan metadata or change admission policy.
