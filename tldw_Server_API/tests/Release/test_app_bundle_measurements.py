"""Measurements count real transfer bytes and fail closed on corrupt data."""

import gzip
import hashlib
import json
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

RegistryFixture = tuple[Path, str, list[str], list[int], str]


@pytest.fixture
def blob_server() -> Iterator[tuple[str, bytes]]:
    payload = b"measured payload, not image metadata"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/blob", payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_download_counts_and_hashes_actual_payload(blob_server: tuple[str, bytes]) -> None:
    from Helper_Scripts.measure_app_bundle import download

    url, payload = blob_server
    count, body = download(url, hashlib.sha256(payload).hexdigest(), len(payload), capture=True)
    assert (count, body) == (36, b"measured payload, not image metadata")


@pytest.mark.parametrize("change", ["digest", "truncated", "oversized"])
def test_corrupt_or_incomplete_download_cannot_supply_measurement(blob_server: tuple[str, bytes], change: str) -> None:
    from Helper_Scripts.measure_app_bundle import download

    url, payload = blob_server
    digest = "0" * 64 if change == "digest" else hashlib.sha256(payload).hexdigest()
    expected = len(payload) + 1 if change == "truncated" else len(payload) - 1
    if change == "digest":
        expected = len(payload)
    with pytest.raises(ValueError):
        download(url, digest, expected)


@pytest.fixture
def measuring_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "docker"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "args=sys.argv[1:]\n"
        "if args[0]=='create': print('a'*64)\n"
        "if args[0]=='volume': print(os.environ.get('VOLUME_PROJECT','fixture'))\n"
        "if args[0]=='start': print(os.environ.get('DU_RESULT','8192\\t/'))\n"
        "if args[0]=='rm' and os.environ.get('FAIL_CLEANUP')=='1': sys.exit(1)\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def test_installed_footprint_comes_from_filesystem_du(measuring_docker: None) -> None:
    from Helper_Scripts.measure_app_bundle import docker_usage

    image = "localhost:5000/tldw/backend@sha256:" + "b" * 64
    assert docker_usage(image) == {"/": 8192}


@pytest.mark.parametrize("result", ["", "4096 /data", "not-a-size /", "4096 /\n8192 /"])
def test_missing_or_malformed_filesystem_measurement_is_refused(
    measuring_docker: None, monkeypatch: pytest.MonkeyPatch, result: str
) -> None:
    from Helper_Scripts.measure_app_bundle import docker_usage

    monkeypatch.setenv("DU_RESULT", result)
    with pytest.raises(ValueError):
        docker_usage("localhost:5000/tldw/backend@sha256:" + "b" * 64)


def test_failed_owned_measurement_cleanup_prevents_success(
    measuring_docker: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Helper_Scripts.measure_app_bundle import docker_usage

    monkeypatch.setenv("FAIL_CLEANUP", "1")
    with pytest.raises(RuntimeError, match="cleanup"):
        docker_usage("localhost:5000/tldw/backend@sha256:" + "b" * 64)


@pytest.fixture
def image_registry(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[RegistryFixture]:
    commit = "c" * 40
    payloads = {}
    layer = gzip.compress(b"shared public layer fixture", mtime=0)
    layer_digest = hashlib.sha256(layer).hexdigest()
    requests = []
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            body = payloads[self.path.rsplit("sha256:", 1)[1]]
            received.append(len(body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    payloads[layer_digest] = layer
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    lines = []
    for role in ("backend", "webui", "gateway", "control"):
        config = json.dumps(
            {
                "os": "linux",
                "architecture": "amd64",
                "config": {"Labels": {"org.opencontainers.image.revision": commit, "fixture.role": role}},
            }
        ).encode()
        config_digest = hashlib.sha256(config).hexdigest()
        payloads[config_digest] = config
        manifest = json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {"digest": "sha256:" + config_digest, "size": len(config)},
                "layers": [{"digest": "sha256:" + layer_digest, "size": len(layer)}],
            }
        ).encode()
        digest = hashlib.sha256(manifest).hexdigest()
        payloads[digest] = manifest
        mode = getattr(request, "param", "manifest")
        if mode != "manifest":
            descriptor = {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": "sha256:" + digest,
                "size": len(manifest),
                "platform": {"os": "linux", "architecture": "amd64"},
            }
            if mode == "missing-platform":
                descriptor["platform"]["architecture"] = "arm64"
            if mode == "corrupt-selected":
                descriptor["digest"] = "sha256:" + "e" * 64
                payloads["e" * 64] = manifest
            entries = [
                descriptor,
                {"digest": "sha256:" + "f" * 64, "platform": {"os": "unknown", "architecture": "unknown"}},
            ]
            if mode == "ambiguous-platform":
                entries.append(descriptor)
            index = json.dumps(
                {
                    "schemaVersion": 1 if mode == "invalid-index-schema" else 2,
                    "mediaType": (
                        "application/vnd.docker.distribution.manifest.list.v2+json"
                        if mode == "docker-index"
                        else "application/vnd.oci.image.index.v1+json"
                    ),
                    "manifests": entries,
                }
            ).encode()
            digest = hashlib.sha256(index).hexdigest()
            payloads[digest] = index
        lines.append(f"{role}\t127.0.0.1:{server.server_port}/tldw/{role}@sha256:{digest}\t999999999")
    images = tmp_path / "images.tsv"
    images.write_text("\n".join(lines))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield images, commit, requests, received, layer_digest
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_all_image_measurements_deduplicate_real_downloads_and_ignore_size_metadata(
    image_registry: RegistryFixture, measuring_docker: None
) -> None:
    from Helper_Scripts.measure_app_bundle import measure_images

    images, commit, requests, received, layer_digest = image_registry
    result = measure_images(images, "linux/amd64", commit)
    assert result["download_unique_image_payload_bytes"] == sum(received)
    assert len([path for path in requests if path.endswith("/blobs/sha256:" + layer_digest)]) == 1
    assert {role: item["rootfs_allocated_bytes"] for role, item in result["roles"].items()} == {
        "backend": 8192,
        "webui": 8192,
        "gateway": 8192,
        "control": 8192,
    }


def test_downloaded_image_source_cannot_be_borrowed_from_another_revision(
    image_registry: RegistryFixture, measuring_docker: None
) -> None:
    from Helper_Scripts.measure_app_bundle import measure_images

    images, _commit, _requests, _received, _digest = image_registry
    with pytest.raises(ValueError, match="source/platform"):
        measure_images(images, "linux/amd64", "d" * 40)


@pytest.mark.parametrize("image_registry", ["oci-index", "docker-index"], indirect=True)
def test_index_download_counts_root_and_selected_manifest_without_attestation_payload(
    image_registry: RegistryFixture, measuring_docker: None
) -> None:
    from Helper_Scripts.measure_app_bundle import measure_images

    images, commit, requests, received, _digest = image_registry
    report = measure_images(images, "linux/amd64", commit)
    assert report["download_unique_image_payload_bytes"] == sum(received)
    assert len([path for path in requests if "/manifests/" in path]) == 8
    assert not any("f" * 64 in path for path in requests)


@pytest.mark.parametrize(
    "image_registry",
    ["missing-platform", "ambiguous-platform", "corrupt-selected", "invalid-index-schema"],
    indirect=True,
)
def test_invalid_index_or_selected_manifest_is_refused(image_registry: RegistryFixture, measuring_docker: None) -> None:
    from Helper_Scripts.measure_app_bundle import measure_images

    images, commit, _requests, _received, _digest = image_registry
    with pytest.raises(ValueError):
        measure_images(images, "linux/amd64", commit)


def test_download_deadline_is_checked_while_response_body_is_still_open(monkeypatch: pytest.MonkeyPatch) -> None:
    from Helper_Scripts import measure_app_bundle

    release = threading.Event()
    payload = b"public slow registry payload"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload[:1])
            self.wfile.flush()
            release.wait(5)
            try:
                self.wfile.write(payload[1:])
            except BrokenPipeError:
                pass

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clock = iter((0.0, 601.0))
    monkeypatch.setattr(measure_app_bundle.time, "monotonic", lambda: next(clock, 601.0))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                measure_app_bundle.download,
                f"http://127.0.0.1:{server.server_port}/blob",
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
            try:
                with pytest.raises(ValueError, match="measurement bound"):
                    future.result(timeout=2)
            finally:
                release.set()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_fresh_state_counts_owned_volumes_and_excludes_symlink_target_contents(
    measuring_docker: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from Helper_Scripts.measure_app_bundle import measure_state

    state = tmp_path / "private-state"
    state.mkdir()
    (state / "config.env").write_bytes(b"public configuration")
    external = tmp_path / "external-file"
    external.write_bytes(b"x" * 1000)
    (state / "external-link").symlink_to(external)
    monkeypatch.setenv("DU_RESULT", "4096\t/data\n8192\t/config")
    result = measure_state("localhost:5000/tldw/control@sha256:" + "b" * 64, "fixture", state)
    assert (
        result["backend_data_allocated_bytes"],
        result["backend_config_allocated_bytes"],
        result["helper_state_logical_file_bytes"],
    ) == (4096, 8192, 20)


def test_foreign_volume_cannot_supply_persistent_storage_measurement(
    measuring_docker: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from Helper_Scripts.measure_app_bundle import measure_state

    monkeypatch.setenv("VOLUME_PROJECT", "another-project")
    with pytest.raises(ValueError, match="ownership"):
        measure_state("localhost:5000/tldw/control@sha256:" + "b" * 64, "fixture", tmp_path)
