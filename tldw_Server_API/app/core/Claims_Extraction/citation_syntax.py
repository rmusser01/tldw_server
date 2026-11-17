"""Literal citation syntax shared by source verification and offline receipts."""

import re

# Join only adjacent bracket groups, optionally separated by one list delimiter.
# Sentence punctuation, words, ranges, bullets and blank lines are not separators.
CITATION_RUN = re.compile(
    r"\[[^\]\n]+\](?:[ \t]*(?:\r?\n[ \t]*)?(?:[,;][ \t]*(?:\r?\n[ \t]*)?)?\[[^\]\n]+\])*"
)
_PAGE = r"[1-9][0-9]*(?:[ \t]*[-–][ \t]*[1-9][0-9]*)?"
_ANNOTATED_NUMERIC = re.compile(
    r"(?:Source )?([1-9][0-9]*)[ \t]*,[ \t]*PDF pp?\.[ \t]*" + _PAGE
    + r"(?:[ \t]*[,;][ \t]*printed pp?\.[ \t]*" + _PAGE + r")?\Z"
)


def extract_citation_ids(text: str) -> list[str]:
    """Return literal IDs, recognizing only the bounded numeric PDF-location form.

    Unknown bracket content remains an unresolved ID; location text is ignored
    only when the entire canonical annotation matches. Citation ranges are never
    expanded and this function supplies no source or page authority.
    """
    ids = []
    for group in re.findall(r"\[([^\]\n]+)\]", text):
        annotated = _ANNOTATED_NUMERIC.fullmatch(group.strip())
        if annotated:
            ids.append(annotated[1])
        else:
            ids.extend(token.strip().removeprefix("Source ") for token in group.split(","))
    return list(dict.fromkeys(ids))
