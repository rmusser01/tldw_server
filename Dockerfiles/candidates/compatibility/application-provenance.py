"""Observe candidate application imports without replacing its eight-test contract."""

import hashlib
import importlib.util
import sys
import sysconfig
import types
from pathlib import Path

CORE = "tldw_Server_API.app.core."
XML = CORE + "Ingestion_Media_Processing.XML_Ingestion_Lib"
CHUNK = CORE + "Chunking"
STRATEGY = CHUNK + ".strategies.json_xml"
LAUNCHER_SHA256 = "2d233408102f35185b3a1e7b2098e9ca2a9f50fa2d5f8898dbd0444012ac820e"
PARSER_HASHES = {
    "pyexpat": "9b2373fe4f83cca3ece2e1dc12713580b439b897138c8405f93b7542980df4ad",
    "_elementtree": "38df3969aaedc9a64a1e9cbb5dad606524eb9b58d5e41cf1118369693b827685",
}


def file_record(path: Path) -> dict:
    """Hash a regular, non-linked file with its canonical location."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked provenance file: {path}")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def snapshot(root: Path, items: list, *, candidate: bool = False, allow_processing_mock: bool = False) -> dict:
    """Inspect already-loaded objects; never import missing application/parser modules."""
    root = root.resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    extensions = Path(sysconfig.get_config_var("DESTSHARED")).resolve()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    paths = {
        XML: root / "tldw_Server_API/app/core/Ingestion_Media_Processing/XML_Ingestion_Lib.py",
        CHUNK: root / "tldw_Server_API/app/core/Chunking/__init__.py",
        CHUNK + ".chunker": root / "tldw_Server_API/app/core/Chunking/chunker.py",
        CHUNK + ".base": root / "tldw_Server_API/app/core/Chunking/base.py",
        STRATEGY: root / "tldw_Server_API/app/core/Chunking/strategies/json_xml.py",
        "defusedxml.ElementTree": site / "defusedxml/ElementTree.py",
        "defusedxml.common": site / "defusedxml/common.py",
        "xml.etree.ElementTree": stdlib / "xml/etree/ElementTree.py",
        "xml.parsers.expat": stdlib / "xml/parsers/expat.py",
        "pyexpat": extensions / ("pyexpat" + suffix),
        "_elementtree": extensions / ("_elementtree" + suffix),
    }
    # This Python facade is imported lazily by the pure-Python XMLParser.
    # Missing at collection is normal; the complete run must observe it later.
    if "xml.parsers.expat" not in sys.modules:
        del paths["xml.parsers.expat"]
    records = {}
    for name, expected in paths.items():
        module = sys.modules.get(name)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        actual = getattr(module, "__file__", None)
        if not actual or not origin or Path(actual).resolve() != expected or Path(origin).resolve() != expected:
            raise ValueError(f"missing or foreign loaded module: {name}")
        if hasattr(module, "__path__") and list(module.__path__) != [str(expected.parent)]:
            raise ValueError(f"unexpected package search path: {name}")
        records[name] = file_record(expected)

    def alias(value, expected, label):
        if value is not expected:
            raise ValueError(f"detached provenance alias: {label}")

    callables = {}

    def code(value, expected, label):
        filename = getattr(getattr(value, "__code__", None), "co_filename", None)
        if not filename or Path(filename).resolve() != expected:
            raise ValueError(f"foreign callable: {label}")
        callables[label] = str(expected)

    xml, chunk, chunker, strategy = (sys.modules[name] for name in (XML, CHUNK, CHUNK + ".chunker", STRATEGY))
    if not allow_processing_mock:
        alias(xml.improved_chunking_process, chunk.improved_chunking_process, "ingestion.improved_chunking_process")
    det, et, pyexpat, element = (
        sys.modules[name] for name in ("defusedxml.ElementTree", "xml.etree.ElementTree", "pyexpat", "_elementtree")
    )
    for value, expected, label in (
        (xml.DET, det, "ingestion.DET"),
        (sys.modules["defusedxml"].ElementTree, det, "defusedxml.ElementTree package alias"),
        (sys.modules["defusedxml"].common, sys.modules["defusedxml.common"], "defusedxml.common package alias"),
        (sys.modules["xml.etree"].ElementTree, et, "xml.etree.ElementTree package alias"),
        (chunk.Chunker, chunker.Chunker, "Chunking.Chunker"),
        (strategy.ET, et, "strategy.ET"),
        (strategy.ParseError, et.ParseError, "strategy.ParseError"),
        (det._parse, et.parse, "defusedxml._parse"),
        (det._TreeBuilder, et.TreeBuilder, "defusedxml._TreeBuilder"),
        (et.XMLParser, element.XMLParser, "ElementTree.XMLParser"),
        (det.DefusedXMLParser.__bases__[0], det._XMLParser, "defusedxml parser base"),
    ):
        alias(value, expected, label)
    if "xml.parsers.expat" in paths:
        alias(sys.modules["xml.parsers.expat"].ParserCreate, pyexpat.ParserCreate, "expat.ParserCreate")
        alias(sys.modules["xml.parsers"].expat, sys.modules["xml.parsers.expat"], "xml.parsers.expat package alias")
    factory = pyexpat.ParserCreate
    if (
        not isinstance(factory, types.BuiltinFunctionType)
        or factory.__self__ is not pyexpat
        or factory.__module__ != "pyexpat"
        or factory.__name__ != "ParserCreate"
    ):
        raise ValueError("foreign native Expat parser factory")
    for name, retained in (
        ("fromstring", {}),
        ("parse", {"_parse": det._parse}),
        ("iterparse", {"_iterparse": det._iterparse}),
    ):
        function = getattr(det, name)
        expected = {"DefusedXMLParser": det.DefusedXMLParser, "_TreeBuilder": det._TreeBuilder, **retained}
        bindings = dict(
            zip(function.__code__.co_freevars, (cell.cell_contents for cell in (function.__closure__ or ())))
        )
        if bindings.keys() != expected.keys() or any(bindings[key] is not value for key, value in expected.items()):
            raise ValueError(f"foreign captured parser binding: defusedxml.{name}")
    for value, owner, label in (
        (xml.import_xml_handler, XML, "ingestion.import_xml_handler"),
        (xml._parse_xml_string, XML, "ingestion._parse_xml_string"),
        (chunk.improved_chunking_process, CHUNK, "Chunking.improved_chunking_process"),
        (chunker.Chunker.chunk_text_with_metadata, CHUNK + ".chunker", "Chunker.chunk_text_with_metadata"),
        (strategy.XMLChunkingStrategy.chunk, STRATEGY, "XMLChunkingStrategy.chunk"),
        (strategy.XMLChunkingStrategy.chunk_with_metadata, STRATEGY, "XMLChunkingStrategy.chunk_with_metadata"),
        (det.fromstring, "defusedxml.common", "defusedxml.fromstring"),
        (det.parse, "defusedxml.common", "defusedxml.parse"),
        (det.iterparse, "defusedxml.common", "defusedxml.iterparse"),
        (det._iterparse, "xml.etree.ElementTree", "defusedxml._iterparse"),
        (det._XMLParser.__init__, "xml.etree.ElementTree", "defusedxml.ElementTree._XMLParser.__init__"),
        (det.DefusedXMLParser.__init__, "defusedxml.ElementTree", "defusedxml.DefusedXMLParser.__init__"),
    ):
        code(value, paths[owner], label)
    tests = {}
    for item in items:
        relative, function = item.nodeid.split("::")
        expected = root / relative
        module = item.module
        if Path(module.__file__).resolve() != expected or Path(module.__spec__.origin).resolve() != expected:
            raise ValueError(f"foreign collected test: {item.nodeid}")
        code(getattr(module, function), expected, item.nodeid)
        alias(item.obj, getattr(module, function), f"collected callable:{item.nodeid}")
        tests[relative] = file_record(expected)
        for name, expected_alias in (
            ("xml_lib", xml),
            ("XMLChunkingStrategy", strategy.XMLChunkingStrategy),
            ("Chunker", chunker.Chunker),
            ("ET", det),
        ):
            if hasattr(module, name):
                alias(getattr(module, name), expected_alias, f"{relative}:{name}")
    if candidate:
        if root != Path("/app") or sys.prefix != "/opt/tldw-venv" or sys.base_prefix != "/usr/local":
            raise ValueError("not the qualified candidate application interpreter")
        if pyexpat.EXPAT_VERSION != "expat_2.8.4" or any(
            records[name]["sha256"] != digest for name, digest in PARSER_HASHES.items()
        ):
            raise ValueError("unqualified candidate parser copy")
    return {"modules": records, "callables": callables, "tests": tests, "expat_version": pyexpat.EXPAT_VERSION}


def evidence_class(base, root: Path, *, candidate: bool = True):
    """Extend, never replace, the original exact-test/phase admission checks."""
    pytest = sys.modules["pytest"]  # Loaded by the original launcher after its path checks.

    class ProvenanceEvidence(base):
        def __init__(self):
            super().__init__()
            self.items = []
            self.observations = []
            self.errors = []

        def capture(self, phase, nodeid=""):
            try:
                observed = snapshot(
                    root,
                    self.items,
                    candidate=candidate,
                    allow_processing_mock=(
                        phase == "after-call"
                        and nodeid.endswith("::test_import_xml_handler_sanitizes_unexpected_processing_failure")
                    ),
                )
                self.observations.append({"phase": phase, "nodeid": nodeid, "snapshot": observed})
            except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
                # Preserve diagnostics but force the final result to fail.
                self.errors.append({"phase": phase, "nodeid": nodeid, "error": str(exc)})

        def pytest_collection_finish(self, session):
            super().pytest_collection_finish(session)
            self.items = list(session.items)
            self.capture("collection")

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtest_call(self, item):
            self.capture("before-call", item.nodeid)
            try:
                yield
            finally:
                self.capture("after-call", item.nodeid)

        def result(self, exit_code):
            result = super().result(exit_code)
            expected = {("collection", "")} | {
                (phase, nodeid) for nodeid in result["expected"] for phase in ("before-call", "after-call")
            }
            observed = {(record["phase"], record["nodeid"]) for record in self.observations}
            stable = {}
            for record in self.observations:
                for name, identity in record["snapshot"]["modules"].items():
                    if name in stable and stable[name] != identity:
                        self.errors.append({"error": f"module identity changed: {name}"})
                    stable[name] = identity
            passed = (
                result["passed"]
                and not self.errors
                and observed == expected
                and len(self.observations) == len(expected)
                and "xml.parsers.expat" in stable
            )
            result.update(
                {
                    "scope": "application-tests-with-import-provenance-not-admitted",
                    "passed": passed,
                    "provenance": {"observations": self.observations, "errors": self.errors},
                }
            )
            return result

    return ProvenanceEvidence


def run(launcher_path: Path) -> int:
    """Wrap only the hash-pinned launcher supplied by the retained image."""
    if file_record(launcher_path)["sha256"] != LAUNCHER_SHA256:
        raise ValueError("unreviewed application test launcher")
    spec = importlib.util.spec_from_file_location("candidate_application_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    base = launcher.Evidence
    launcher.Evidence = lambda: evidence_class(base, Path.cwd())()
    return launcher.main()


if __name__ == "__main__":
    raise SystemExit(run(Path("/opt/combined/run-tests.py")))
