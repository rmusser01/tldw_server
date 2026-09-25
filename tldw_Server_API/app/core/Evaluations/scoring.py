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

Sites that receive a mix of raw judge scores and already-normalized values use
normalize_judge_score: a value in [0, 1) cannot be a raw score on a scale that starts
at 1, so it is taken as already normalized (0 is also run_geval's parse-failure
sentinel). Exactly 1 is on the scale and maps to 0.0. Before this, those sites split
three ways at 1: two returned 1.0 (the worst rating reported as perfect), the rest 0.2.
"""

__all__ = ["normalize_geval_metric", "normalize_judge_score", "normalize_likert", "parse_judge_score"]

# ms_g_eval.run_geval asks for fluency on 1-3 and every other metric on 1-5.
_GEVAL_FLUENCY_MAX = 3.0
_GEVAL_DEFAULT_MAX = 5.0


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


def normalize_judge_score(
    raw: float,
    *,
    scale_min: float = 1.0,
    scale_max: float = 5.0,
) -> float:
    """Like normalize_likert, but a value in [0, 1) below a 1-based scale passes through."""
    value = float(raw)
    if scale_min >= 1.0 and 0.0 <= value < 1.0:
        return value
    return normalize_likert(value, scale_min=scale_min, scale_max=scale_max)


def normalize_geval_metric(metric: str, raw: float) -> float:
    """Normalize one run_geval metric on its own scale (fluency 1-3, others 1-5)."""
    scale_max = _GEVAL_FLUENCY_MAX if metric == "fluency" else _GEVAL_DEFAULT_MAX
    return normalize_judge_score(raw, scale_max=scale_max)


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
