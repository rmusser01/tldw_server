from __future__ import annotations

import random
import re
from collections.abc import Iterable

from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.models import DiceRollResult

_DICE_EXPR_RE = re.compile(r"^(?P<count>\d{1,3})d(?P<sides>\d{1,4})(?P<modifier>[+-]\d{1,4})?$")

MAX_DICE_COUNT = 100
MAX_DIE_SIDES = 1000
MAX_ROLL_MODIFIER = 1000
MAX_FATE_DICE = 20


class DiceRoller:
    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        injected_values: Iterable[int] | None = None,
        injected_fate_values: Iterable[int] | None = None,
    ) -> None:
        self._rng = rng or random.Random()  # nosec B311
        self._injected_values = list(injected_values or [])
        self._injected_fate_values = list(injected_fate_values or [])

    def roll(self, expression: str) -> DiceRollResult:
        dice_count, sides, modifier = self._parse_expression(expression)
        values = [self._next_die_value(sides) for _ in range(dice_count)]
        total = sum(values) + modifier

        return DiceRollResult(
            expression=expression,
            values=values,
            modifier=modifier,
            total=total,
            dice_count=dice_count,
            sides=sides,
            details={},
        )

    def roll_fate(self, *, dice_count: int = 4, modifier: int = 0) -> DiceRollResult:
        if not 1 <= dice_count <= MAX_FATE_DICE:
            raise RPGValidationError(f"fate dice count must be between 1 and {MAX_FATE_DICE}")
        if abs(modifier) > MAX_ROLL_MODIFIER:
            raise RPGValidationError(f"fate modifier must be between -{MAX_ROLL_MODIFIER} and {MAX_ROLL_MODIFIER}")

        values = [self._next_fate_value() for _ in range(dice_count)]
        total = sum(values) + modifier
        expression = f"{dice_count}dF{modifier:+d}" if modifier else f"{dice_count}dF"

        return DiceRollResult(
            expression=expression,
            values=values,
            modifier=modifier,
            total=total,
            dice_count=dice_count,
            sides=None,
            details={"dice": "fate"},
        )

    def _parse_expression(self, expression: str) -> tuple[int, int, int]:
        match = _DICE_EXPR_RE.fullmatch(expression.strip())
        if match is None:
            raise RPGValidationError("dice expression must match NdM, NdM+K, or NdM-K")

        dice_count = int(match.group("count"))
        sides = int(match.group("sides"))
        modifier_text = match.group("modifier")
        modifier = int(modifier_text) if modifier_text else 0

        if not 1 <= dice_count <= MAX_DICE_COUNT:
            raise RPGValidationError(f"dice count must be between 1 and {MAX_DICE_COUNT}")
        if not 1 <= sides <= MAX_DIE_SIDES:
            raise RPGValidationError(f"die sides must be between 1 and {MAX_DIE_SIDES}")
        if abs(modifier) > MAX_ROLL_MODIFIER:
            raise RPGValidationError(f"modifier must be between -{MAX_ROLL_MODIFIER} and {MAX_ROLL_MODIFIER}")

        return dice_count, sides, modifier

    def _next_die_value(self, sides: int) -> int:
        if self._injected_values:
            value = self._injected_values.pop(0)
        else:
            value = self._rng.randint(1, sides)  # nosec B311

        if not 1 <= value <= sides:
            raise RPGValidationError(f"injected die value {value} is outside 1..{sides}")
        return value

    def _next_fate_value(self) -> int:
        if self._injected_fate_values:
            value = self._injected_fate_values.pop(0)
        else:
            value = self._rng.choice((-1, 0, 1))  # nosec B311

        if value not in {-1, 0, 1}:
            raise RPGValidationError("injected fate die values must be -1, 0, or 1")
        return value
