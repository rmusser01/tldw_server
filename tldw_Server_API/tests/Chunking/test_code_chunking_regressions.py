from tldw_Server_API.app.core.Chunking import Chunker


def test_code_single_line_brace_block_does_not_merge_next():

    ck = Chunker()
    src = "function foo() {}\nfunction bar() {}\n"
    chunks = ck.chunk_text(
        src,
        method="code",
        max_size=30,
        overlap=0,
        language="javascript",
    )

    assert chunks == ["function foo() {}", "function bar() {}"]


def test_code_prefix_without_headers_does_not_duplicate_text():
    ck = Chunker()
    src = "// header comment\nconst x = 1;\nconsole.log(x)\n"

    chunks = ck.chunk_text(
        src,
        method="code",
        max_size=500,
        overlap=0,
        language="javascript",
    )

    assert chunks == ["// header comment\nconst x = 1;\nconsole.log(x)"]


def test_code_prefix_without_headers_metadata_keeps_full_file():
    ck = Chunker()
    src = "// header comment\nconst x = 1;\nconsole.log(x)\n"

    chunks = ck.chunk_text_with_metadata(
        src,
        method="code",
        max_size=500,
        overlap=0,
        language="javascript",
    )

    assert len(chunks) == 1
    assert chunks[0].text == src
    assert chunks[0].metadata.start_char == 0
    assert chunks[0].metadata.end_char == len(src)
