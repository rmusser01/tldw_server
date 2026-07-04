import pytest

from tldw_Server_API.app.core.RPG.checks import resolve_check
from tldw_Server_API.app.core.RPG.constants import RPG_ADAPTER_DND5E_SRD, RPG_ADAPTER_FATE
from tldw_Server_API.app.core.RPG.dice import DiceRoller
from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.models import CheckResult, DiceRollResult
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry

pytestmark = pytest.mark.unit


def test_dice_roller_parses_expression_with_injected_values():
    roller = DiceRoller(injected_values=[2, 6])

    result = roller.roll("2d6+3")

    assert isinstance(result, DiceRollResult)  # nosec B101
    assert result.expression == "2d6+3"  # nosec B101
    assert result.values == [2, 6]  # nosec B101
    assert result.modifier == 3  # nosec B101
    assert result.total == 11  # nosec B101


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "2d6+",
        "101d6",
        "1d1001",
        "2d6+1001",
        "2d6-1001",
        "2d6+3+4",
    ],
)
def test_dice_roller_rejects_invalid_or_unbounded_expressions(expression):
    roller = DiceRoller()

    with pytest.raises(RPGValidationError):
        roller.roll(expression)


def test_dice_roller_uses_injected_fate_values():
    roller = DiceRoller(injected_fate_values=[-1, 0, 1, 1])

    result = roller.roll_fate(modifier=2)

    assert isinstance(result, DiceRollResult)  # nosec B101
    assert result.expression == "4dF+2"  # nosec B101
    assert result.values == [-1, 0, 1, 1]  # nosec B101
    assert result.modifier == 2  # nosec B101
    assert result.total == 3  # nosec B101


def test_dnd5e_adapter_resolves_d20_check_with_injected_roll():
    registry = build_default_adapter_registry()
    adapter = registry.get(RPG_ADAPTER_DND5E_SRD)
    roller = DiceRoller(injected_values=[14])

    result = resolve_check(
        adapter,
        roller,
        {
            "check_label": "Stealth",
            "roll_expression": "1d20",
            "modifier": 5,
            "dc": 18,
        },
    )

    assert isinstance(result, CheckResult)  # nosec B101
    assert result.check_label == "Stealth"  # nosec B101
    assert result.mechanics == "d20"  # nosec B101
    assert result.roll.total == 19  # nosec B101
    assert result.target == 18  # nosec B101
    assert result.success is True  # nosec B101
    assert result.margin == 1  # nosec B101


def test_fate_adapter_resolves_ladder_check_with_injected_fate_rolls():
    registry = build_default_adapter_registry()
    adapter = registry.get(RPG_ADAPTER_FATE)
    roller = DiceRoller(injected_fate_values=[-1, 0, 1, 1])

    result = resolve_check(
        adapter,
        roller,
        {
            "check_label": "Careful defense",
            "skill_bonus": 2,
            "ladder_target": 3,
        },
    )

    assert isinstance(result, CheckResult)  # nosec B101
    assert result.check_label == "Careful defense"  # nosec B101
    assert result.mechanics == "fate"  # nosec B101
    assert result.roll.total == 3  # nosec B101
    assert result.target == 3  # nosec B101
    assert result.success is True  # nosec B101
    assert result.margin == 0  # nosec B101


def test_resolve_check_delegates_to_adapter_owned_resolution():
    class SpyAdapter:
        mechanics_tags = {"resolution_family": "unsupported"}

        def __init__(self) -> None:
            self.payload = None

        def resolve_check(self, roller, payload):
            self.payload = payload
            return CheckResult(
                check_label=payload["check_label"],
                mechanics="spy",
                roll=roller.roll("1d4"),
                target=None,
                success=None,
                margin=None,
                details={},
            )

    adapter = SpyAdapter()
    roller = DiceRoller(injected_values=[3])

    result = resolve_check(adapter, roller, {"check_label": "Adapter-owned"})

    assert adapter.payload == {"check_label": "Adapter-owned"}  # nosec B101
    assert result.mechanics == "spy"  # nosec B101
    assert result.roll.total == 3  # nosec B101
