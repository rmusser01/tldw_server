# util-linux native candidate qualification

This candidate-only harness qualifies the Debian Trixie `util-linux` source `2.41.5-0+deb13u1` plus the exact upstream commit `1d14676ea70003e9f5b2a6a76af0cadb1190411a` as local version `2.41.5-0+deb13u1+tldw1`. It does not alter a production image, publish packages, or authorize adoption.

The path-filtered `Util-linux Native Candidate Qualification` workflow must run on GitHub's native `ubuntu-24.04` amd64 runner. It authenticates the fixed Debian snapshot through APT, independently checks all three downloaded source hashes, checks the exact HTTPS-fetched Git object and full patch hash, applies the unchanged patch with fuzz zero, and runs the unmodified Debian source and binary builds as UID 1000 with no network, capabilities, or new privileges.

After a passing test gate, the job rejects SONAME changes or missing baseline versioned exports for `libmount`, `libblkid`, `libuuid`, `libsmartcols`, and `liblastlog2`. It then co-installs every emitted `.deb` from the source family in a fresh offline container, including the `bsdutils` epoch and special `login` version, and records dependency, audit, version, and permitted smoke results. Containers are retained when evidence extraction or a qualification phase fails.

The Actions artifact contains the runner, Docker, image, and checked-out commit identities; signed APT indices and source metadata; source and patch provenance; complete source/binary outputs and relocatable SHA-256 manifests that verify from each downloaded artifact directory; raw build/test logs; ABI comparisons; and fresh-install results. Retention is 14 days. These records establish signed-index/hash and content-addressed provenance only; they do not claim a direct Debian uploader signature or upstream maintainer signature.

Privileged CVE-path testing and official vulnerability-scanner recognition remain unverified. A green harness unit test is not a successful package qualification; only the authentic native workflow artifact can support a candidate decision.
