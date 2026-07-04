"""Property-based tests for the chat-macro slash-argument parser.

Prepared ahead of the Chat_Macros merge (codex/chat-macros-v1, PR #2618):
the importorskip below makes this file collect-and-skip until
``app/core/Chat_Macros`` lands on dev, then the properties activate
automatically. They generalize the invariants of the bounds-validation
fix c1f4e6eb95 (see audits/2026-07-04-test-suite-audit-round2.md, RA4).
"""

import shlex

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

parser = pytest.importorskip(
    "tldw_Server_API.app.core.Chat_Macros.parser",
    reason="Chat_Macros not merged yet (codex/chat-macros-v1, PR #2618)",
)
macro_models = pytest.importorskip("tldw_Server_API.app.core.Chat_Macros.models")
macro_exceptions = pytest.importorskip("tldw_Server_API.app.core.Chat_Macros.exceptions")

MacroArgSpec = macro_models.MacroArgSpec
MacroValidationError = macro_exceptions.MacroValidationError

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_ARG_NAME = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)

_TRUE_TOKENS = {"1", "true", "yes", "on"}
_FALSE_TOKENS = {"0", "false", "no", "off"}


def _value_strategy(arg_type: str):
    if arg_type == "string":
        # no NUL (shlex chokes), keep printable-ish but include spaces/quotes/=
        return st.text(
            alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
            max_size=30,
        )
    if arg_type == "boolean":
        return st.booleans()
    if arg_type == "integer":
        return st.integers(min_value=-(10**9), max_value=10**9)
    return st.floats(allow_nan=False, allow_infinity=False, width=32)


@st.composite
def _specs_and_values(draw):
    """A set of distinct arg specs plus one generated value per arg."""
    names = draw(
        st.lists(_ARG_NAME, min_size=1, max_size=5, unique_by=lambda n: n.replace("_", "-"))
    )
    specs = {}
    values = {}
    for name in names:
        arg_type = draw(st.sampled_from(["string", "boolean", "integer", "number"]))
        specs[name] = MacroArgSpec(type=arg_type)
        values[name] = draw(_value_strategy(arg_type))
    return specs, values


def _render_cli(specs, values) -> str:
    """Render values back to a slash-arg command line the parser must accept."""
    tokens = []
    for name, value in values.items():
        if specs[name].type == "boolean":
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        # inline form handles values with spaces/leading-dashes uniformly
        tokens.append(shlex.quote(f"--{name}={rendered}"))
    return " ".join(tokens)


class TestParseMacroArgsProperties:
    @_COMMON
    @given(raw=st.text(max_size=60), data=st.data())
    def test_parser_is_total(self, raw, data):
        """Arbitrary input either parses or raises MacroValidationError — nothing else."""
        specs, _ = data.draw(_specs_and_values())
        try:
            result = parser.parse_macro_args(raw, specs)
        except MacroValidationError:
            return
        assert isinstance(result, dict)
        assert set(result) == set(specs)

    @_COMMON
    @given(data=st.data())
    def test_rendered_values_round_trip(self, data):
        specs, values = data.draw(_specs_and_values())
        parsed = parser.parse_macro_args(_render_cli(specs, values), specs)
        for name, expected in values.items():
            if specs[name].type == "number":
                assert parsed[name] == pytest.approx(float(expected))
            else:
                assert parsed[name] == expected

    @_COMMON
    @given(data=st.data())
    def test_empty_input_yields_defaults_without_aliasing(self, data):
        specs, _ = data.draw(_specs_and_values())
        repeated = MacroArgSpec(type="string", repeated=True, default=["seed"])
        specs = {**specs, "zz_repeated": repeated}
        parsed = parser.parse_macro_args("", specs)
        assert parsed["zz_repeated"] == ["seed"]
        parsed["zz_repeated"].append("mutated")
        # a second parse must not observe the mutation (defaults are copied)
        assert parser.parse_macro_args("", specs)["zz_repeated"] == ["seed"]

    @_COMMON
    @given(data=st.data(), unknown=_ARG_NAME)
    def test_unknown_option_always_rejected(self, data, unknown):
        specs, _ = data.draw(_specs_and_values())
        aliases = {n.replace("_", "-") for n in specs} | set(specs)
        if unknown in aliases or unknown.replace("_", "-") in aliases:
            return
        with pytest.raises(MacroValidationError):
            parser.parse_macro_args(f"--{unknown}=x", specs)

    @_COMMON
    @given(name=_ARG_NAME, value=st.text(alphabet="ab", max_size=5))
    def test_duplicate_non_repeated_rejected(self, name, value):
        specs = {name: MacroArgSpec(type="string")}
        arg = shlex.quote(f"--{name}={value}")
        with pytest.raises(MacroValidationError):
            parser.parse_macro_args(f"{arg} {arg}", specs)

    @_COMMON
    @given(count=st.integers(min_value=0, max_value=12), limit=st.integers(min_value=1, max_value=8))
    def test_max_questions_bound_is_exact(self, count, limit):
        specs = {"question": MacroArgSpec(type="string", repeated=True)}
        raw = " ".join(f"--question=q{i}" for i in range(count))
        if count > limit:
            with pytest.raises(MacroValidationError):
                parser.parse_macro_args(raw, specs, max_questions=limit)
        else:
            parsed = parser.parse_macro_args(raw, specs, max_questions=limit)
            assert parsed["question"] == [f"q{i}" for i in range(count)]

    @_COMMON
    @given(name=_ARG_NAME, data=st.data())
    def test_hyphen_alias_is_equivalent_to_canonical_name(self, name, data):
        if "_" not in name:
            return
        value = data.draw(st.text(alphabet="abc", min_size=1, max_size=5))
        specs = {name: MacroArgSpec(type="string")}
        via_name = parser.parse_macro_args(f"--{name}={value}", specs)
        via_alias = parser.parse_macro_args(f"--{name.replace('_', '-')}={value}", specs)
        assert via_name == via_alias == {name: value}


class TestCoerceBoolProperties:
    @_COMMON
    @given(raw=st.text(max_size=10))
    def test_accepts_exactly_the_documented_tokens_case_insensitively(self, raw):
        normalized = raw.lower()
        if normalized in _TRUE_TOKENS:
            assert parser._coerce_bool(raw) is True
        elif normalized in _FALSE_TOKENS:
            assert parser._coerce_bool(raw) is False
        else:
            with pytest.raises(MacroValidationError):
                parser._coerce_bool(raw)


class TestLoadMacroDefinitionProperties:
    @hyp_settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(raw=st.text(max_size=120))
    def test_loader_is_total(self, raw):
        """Arbitrary YAML text either loads or raises MacroValidationError —
        yaml.YAMLError / pydantic.ValidationError must never leak."""
        try:
            definition = parser.load_macro_definition(raw)
        except MacroValidationError:
            return
        assert definition.schema_version == 1

    @_COMMON
    @given(bad_name=st.text(max_size=20))
    def test_arg_name_bounds_enforced(self, bad_name):
        """Generalizes fix c1f4e6eb95: arg names outside ^[a-z][a-z0-9_]{0,63}$
        are always rejected; conforming names are always accepted."""
        import re

        definition = {
            "schema_version": 1,
            "name": "probe",
            "command": "probe",
            "args": {bad_name: {"type": "string"}},
        }
        conforming = re.fullmatch(r"[a-z][a-z0-9_]{0,63}", bad_name) is not None
        import yaml

        raw = yaml.safe_dump(definition, allow_unicode=True)
        if conforming:
            loaded = parser.load_macro_definition(raw)
            assert bad_name in loaded.args
        else:
            with pytest.raises(MacroValidationError):
                parser.load_macro_definition(raw)
