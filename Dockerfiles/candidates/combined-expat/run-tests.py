"""Run the eight application controls with append-only, isolated pytest tooling.

Run with the candidate application interpreter. This report is not a scanner,
container-identity or parser-version gate and cannot qualify an image on its own.
"""

import argparse
import json
import os
import sys
from pathlib import Path

TESTS = (
    "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_reports_malformed_input",
    "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_sanitizes_unexpected_processing_failure",
    "tldw_Server_API/tests/MediaIngestion_NEW/unit/test_xml_ingestion.py::test_import_xml_handler_uses_managed_media_database",
    "tldw_Server_API/tests/Chunking/test_xml_allows_url_text.py::test_xml_allows_urls_in_text_nodes",
    "tldw_Server_API/tests/Chunking/test_xml_allows_url_text.py::test_xml_allows_system_in_text_nodes",
    "tldw_Server_API/tests/Chunking/test_json_xml_offsets.py::test_chunk_with_metadata_json_offsets_match_source",
    "tldw_Server_API/tests/Chunking/test_json_xml_offsets.py::test_chunk_with_metadata_xml_offsets_match_source",
    "tldw_Server_API/tests/Chunking/test_xml_tail_preservation.py::test_xml_chunk_preserves_tail_text",
)


class Evidence:
    """Retain actual collection and all three phases, including teardown errors."""

    def __init__(self) -> None:
        self.collected: list[str] = []
        self.reports: list[dict] = []

    def pytest_collection_finish(self, session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report) -> None:
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": report.outcome,
                "wasxfail": getattr(report, "wasxfail", None),
            }
        )

    def result(self, exit_code: int) -> dict:
        """Require exactly one successful setup/call/teardown per expected test."""
        expected_phases = {(nodeid, phase) for nodeid in TESTS for phase in ("setup", "call", "teardown")}
        observed = {(report["nodeid"], report["phase"]) for report in self.reports}
        passed = (
            exit_code == 0
            and sorted(self.collected) == sorted(TESTS)
            and len(self.reports) == len(expected_phases)
            and observed == expected_phases
            and all(report["outcome"] == "passed" and report["wasxfail"] is None for report in self.reports)
        )
        return {
            "scope": "application-tests-only",
            "passed": passed,
            "pytest_exit_code": int(exit_code),
            "expected": list(TESTS),
            "collected": self.collected,
            "reports": self.reports,
            "interpreter": sys.executable,
            "prefix": sys.prefix,
        }


def main() -> int:
    """Keep runtime imports ahead of test tools and preserve normal conftests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/app"))
    parser.add_argument("--tools", type=Path, default=Path("/opt/expat-test-tools"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root, tools = args.root.resolve(), args.tools.resolve()
    report_path = args.report.resolve()
    if not root.is_dir() or not tools.is_dir():
        parser.error("application root and isolated tools must exist")
    os.chdir(root)
    # The application source is present in its image, never mounted from a host.
    sys.path.insert(0, str(root))
    # Do not use PYTHONPATH or site.addsitedir: precedence and .pth execution matter.
    if any(Path(entry).resolve() == tools for entry in sys.path):
        parser.error("test tools must not already be on the import path")
    sys.path.append(str(tools))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    import pytest

    evidence = Evidence()
    exit_code = pytest.main(
        [
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "pytest_timeout",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(root),
            *TESTS,
        ],
        plugins=[evidence],
    )
    result = evidence.result(exit_code)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
