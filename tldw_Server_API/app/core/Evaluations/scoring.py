from __future__ import annotations

"""Convert a judge's raw score on a declared scale to 0-1.

This knowledge was re-derived sixteen times across the Evaluations module with five
different answers. The two that mattered disagreed on the bottom of the range:

    (raw - 1) / 4   -> 1 maps to 0.0, 3 to 0.5, 5 to 1.0
    raw / 5.0       -> 1 maps to 0.2, 3 to 0.6, 5 to 1.0

Both agree at 5, so the divergence was invisible on happy-path fixtures and only
distorted poor results. `raw / 5.0` puts a 20% floor under every metric: a retrieval
system whose judge rates every context "1 = completely irrelevant" reported 0.2 rather
than 0.0, which also shifted the observed-range clamp in _calculate_overall_score and
every threshold comparison built on it.

DECISION: the affine mapping of the declared range wins. On a 1-5 Likert scale the
minimum observable score is 1, not 0, so 1 must map to 0.0. This is what
RAGEvaluator._normalize_score always implemented -- correctly, and with tests -- while
having zero production callers.

Scales other than 1-5 pass their own bounds rather than growing another copy.
"""

__all__ = ["normalize_likert", "parse_judge_score"]


def normalize_likert(
    raw: float,
    *,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> float:
    """Map ``raw`` on ``[scale_min, scale_max]`` onto ``[0.0, 1.0]``.

    Out-of-range input is clamped to the scale rather than rejected: judges do
    occasionally emit a 0 or a 6, and an evaluation run should degrade rather than fail.
    """
    if scale_max <= scale_min:
        raise ValueError("scale_max must be greater than scale_min")
    clamped = max(scale_min, min(scale_max, float(raw)))
    return (clamped - scale_min) / (scale_max - scale_min)


def parse_judge_score(
    text: str | float | int | None,
    *,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
    default: float = 0.0,
) -> float:
    """Parse a judge's reply and normalize it, returning ``default`` when unparseable."""
    if text is None:
        return default
    if isinstance(text, (int, float)):
        return normalize_likert(float(text), scale_min=scale_min, scale_max=scale_max)
    try:
        return normalize_likert(
            float(str(text).strip()), scale_min=scale_min, scale_max=scale_max
        )
    except (TypeError, ValueError):
        return default
