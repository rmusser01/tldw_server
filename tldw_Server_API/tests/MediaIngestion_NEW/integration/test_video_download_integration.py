import http.server
import shutil
import socketserver
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video import Video_DL_Ingestion_Lib as video_lib


@contextmanager
def _serve_directory(directory: Path):
    """Run a temporary HTTP server rooted at ``directory``."""
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, fmt, *args):  # noqa: A003 - keep handler signature
            return

    # Bind to an ephemeral port on localhost
    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    try:
        with ReusableTCPServer(("127.0.0.1", 0), QuietHandler) as server:
            host, port = server.server_address

            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://{host}:{port}"
            finally:
                server.shutdown()
                thread.join()
    except PermissionError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Cannot bind local HTTP server for video download test: {exc}")


@pytest.mark.integration
def test_download_video_real_http(monkeypatch, tmp_path):
    """Exercise download_video end-to-end against a local HTTP server."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        pytest.skip("ffmpeg not available; skipping HTTP download integration test.")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    media_source = source_dir / "clip.mp4"

    # Generate a very small test clip (1s silent video) via ffmpeg.
    generate_cmd = [
        ffmpeg_path,
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-t",
        "1",
        "-y",
        str(media_source),
    ]
    subprocess.run(generate_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    server_dir = tmp_path / "server"
    server_dir.mkdir()
    served_clip = server_dir / "sample.mp4"
    shutil.copy(media_source, served_clip)

    with _serve_directory(server_dir) as base_url:
        media_url = f"{base_url}/sample.mp4"
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        allowed_netloc = urlparse(base_url).netloc

        def _allow_test_server(url, **_kwargs):
            parsed = urlparse(url)
            return SimpleNamespace(
                allowed=parsed.scheme == "http" and parsed.netloc == allowed_netloc,
                reason=None if parsed.netloc == allowed_netloc else "unexpected test host",
            )

        monkeypatch.setattr(video_lib, "evaluate_url_policy", _allow_test_server)

        downloaded_path = video_lib.download_video(
            media_url,
            str(download_dir),
            info_dict=None,
            download_video_flag=True,
            use_cookies=False,
            cookies=None,
        )

        downloaded_file = Path(downloaded_path)
        assert downloaded_file.exists()
        assert downloaded_file.suffix in {".mp4", ".m4v"}
        assert downloaded_file.stat().st_size == served_clip.stat().st_size
