"""Deterministic parser for structured Q&A flashcard preview imports."""
from dataclasses import dataclass, field


QUESTION_LABELS = frozenset({"q", "question"})
ANSWER_LABELS = frozenset({"a", "answer"})
LABEL_SEPARATORS = frozenset({":", ".", "-"})


@dataclass
class StructuredQaDraft:
    front: str
    back: str
    line_start: int
    line_end: int
    notes: str | None = None
    extra: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class StructuredQaParseError:
    line: int | None
    error: str


@dataclass
class StructuredQaPreviewResult:
    drafts: list[StructuredQaDraft] = field(default_factory=list)
    errors: list[StructuredQaParseError] = field(default_factory=list)
    detected_format: str = "qa_labels"
    skipped_blocks: int = 0


def _utf8_length(value: str) -> int:
    return len(value.encode("utf-8"))


def _match_labeled_line(raw_line: str, labels: frozenset[str]) -> str | None:
    stripped = raw_line.lstrip()
    for index, char in enumerate(stripped):
        if char not in LABEL_SEPARATORS:
            continue
        label = stripped[:index].strip().lower()
        if label in labels:
            return stripped[index + 1 :].strip()
        return None
    return None


def parse_structured_qa_preview(
    content: str,
    *,
    max_lines: int | None = None,
    max_line_length: int | None = None,
    max_field_length: int | None = None,
) -> StructuredQaPreviewResult:
    """Parse labeled Q&A text into preview drafts and parse errors."""
    result = StructuredQaPreviewResult()
    lines = (content or "").splitlines()
    if max_lines is not None and max_lines < len(lines):
        lines_to_process = lines[:max_lines]
        limit_reached = True
    else:
        lines_to_process = lines
        limit_reached = False

    question_lines: list[str] = []
    answer_lines: list[str] = []
    line_start: int | None = None
    line_end: int | None = None
    in_answer = False

    def reset_block() -> None:
        nonlocal question_lines, answer_lines, line_start, line_end, in_answer
        question_lines = []
        answer_lines = []
        line_start = None
        line_end = None
        in_answer = False

    def push_error(line: int | None, error: str) -> None:
        result.errors.append(StructuredQaParseError(line=line, error=error))

    def finalize_block() -> None:
        nonlocal question_lines, answer_lines, line_start, line_end
        if line_start is None:
            return

        front = "\n".join(question_lines).strip()
        back = "\n".join(answer_lines).strip()

        if not front and not back:
            reset_block()
            return
        if not front:
            result.skipped_blocks += 1
            push_error(line_start, "Missing question before answer block")
            reset_block()
            return
        if not back:
            result.skipped_blocks += 1
            push_error(line_start, f"Missing answer for question starting on line {line_start}")
            reset_block()
            return
        if max_field_length is not None:
            if _utf8_length(front) > max_field_length:
                result.skipped_blocks += 1
                push_error(line_start, f"Field too long: Front (> {max_field_length} bytes)")
                reset_block()
                return
            if _utf8_length(back) > max_field_length:
                result.skipped_blocks += 1
                push_error(line_start, f"Field too long: Back (> {max_field_length} bytes)")
                reset_block()
                return

        result.drafts.append(
            StructuredQaDraft(
                front=front,
                back=back,
                line_start=line_start,
                line_end=line_end or line_start,
            )
        )
        reset_block()

    for index, raw_line in enumerate(lines_to_process, start=1):
        if max_line_length is not None and _utf8_length(raw_line) > max_line_length:
            if line_start is not None:
                result.skipped_blocks += 1
                reset_block()
            push_error(index, f"Line too long (> {max_line_length} bytes)")
            continue

        question_value = _match_labeled_line(raw_line, QUESTION_LABELS)
        if question_value is not None:
            finalize_block()
            question_lines = [question_value]
            answer_lines = []
            line_start = index
            line_end = index
            in_answer = False
            continue

        answer_value = _match_labeled_line(raw_line, ANSWER_LABELS)
        if answer_value is not None:
            if line_start is None:
                result.skipped_blocks += 1
                push_error(index, "Answer block found before any question label")
                continue
            answer_lines.append(answer_value)
            line_end = index
            in_answer = True
            continue

        if not raw_line.strip():
            if line_start is not None:
                if in_answer:
                    if answer_lines and answer_lines[-1] != "":
                        answer_lines.append("")
                else:
                    if question_lines and question_lines[-1] != "":
                        question_lines.append("")
                line_end = index
            continue

        if line_start is None:
            continue

        if in_answer:
            answer_lines.append(raw_line.rstrip())
        else:
            question_lines.append(raw_line.rstrip())
        line_end = index

    finalize_block()

    if limit_reached:
        push_error(None, f"Maximum preview line limit reached ({max_lines})")

    return result
