from __future__ import annotations

import ipaddress
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock
from tldw_Server_API.app.core.Sandbox import network_policy as policy


class Firewall:
    """Model rule insertion and packet verdicts at the subprocess boundary."""

    def __init__(self, fallback: bool = False, fail_jump: bool = False) -> None:
        self.chains: dict[str, list[list[str]]] = {
            "INPUT": [["-j", "ACCEPT"]],
            "FORWARD": [["-j", "ACCEPT"], ["-j", "DOCKER-USER"]],
            "DOCKER-USER": [["-j", "RETURN"]],
        }
        self.fallback = fallback
        self.fail_jump = fail_jump
        self.ignore_jump = False
        self.omit_drop = False
        self.fail_delete = False

    def run(self, args, *, input=None, **kwargs):
        del kwargs
        if args[0] == "iptables-restore":
            if self.fallback:
                return SimpleNamespace(returncode=1)
            for line in input.decode().splitlines():
                if line.startswith(":"):
                    self.chains[line.split()[0][1:]] = []
                elif line.startswith("-"):
                    self.run(["iptables", *shlex.split(line)])
            return SimpleNamespace(returncode=0)
        action, chain, *rule = args[1:]
        if action == "-N":
            if chain in self.chains:
                return SimpleNamespace(returncode=1)
            self.chains[chain] = []
        elif action == "-A":
            if not (self.omit_drop and rule[-1] == "DROP"):
                self.chains[chain].append(rule)
        elif action == "-I":
            if self.fail_jump:
                return SimpleNamespace(returncode=1)
            if not self.ignore_jump:
                self.chains[chain].insert(int(rule[0]) - 1, rule[1:])
        elif action == "-D":
            if self.fail_delete:
                raise subprocess.CalledProcessError(1, args)
            self.chains[chain].remove(rule)
        elif action == "-F":
            self.chains[chain].clear()
        elif action == "-X":
            del self.chains[chain]
        else:
            raise AssertionError(args)
        return SimpleNamespace(returncode=0)

    def output(self, args, **kwargs):
        del kwargs
        assert args == ["iptables", "-S"]
        declarations = [shlex.join(["-N", chain]) for chain in self.chains if chain.startswith("TLDW_SBX_")]
        return "\n".join(declarations + [
            shlex.join(["-A", chain, *rule])
            for chain, rules in self.chains.items()
            for rule in rules
        ])

    def verdict(self, chain: str, source: str, destination: str) -> str:
        for rule in self.chains[chain]:
            fields = dict(zip(rule[::2], rule[1::2]))
            if any(
                key in fields and ipaddress.ip_address(value) not in ipaddress.ip_network(fields[key])
                for key, value in (("-s", source), ("-d", destination))
            ):
                continue
            target = fields["-j"]
            if target in {"ACCEPT", "DROP", "RETURN"}:
                return target
            result = self.verdict(target, source, destination)
            if result != "RETURN":
                return result
        return "RETURN"


@pytest.fixture
def firewall(monkeypatch: pytest.MonkeyPatch) -> Firewall:
    firewall = Firewall()
    monkeypatch.setattr(policy.subprocess, "run", firewall.run)
    monkeypatch.setattr(policy.subprocess, "check_output", firewall.output)
    return firewall


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("entry", ["FORWARD", "INPUT"])
def test_existing_terminating_rules_cannot_bypass_policy(firewall: Firewall, fallback: bool, entry: str) -> None:
    firewall.fallback = fallback
    policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10/32"], "tldw-run-test")

    assert firewall.verdict(entry, "172.18.0.2", "192.0.2.10") == "ACCEPT"
    assert firewall.verdict(entry, "172.18.0.2", "192.0.2.11") == "DROP"
    assert firewall.verdict(entry, "172.18.0.3", "192.0.2.11") == "ACCEPT"


@pytest.mark.parametrize("fallback", [False, True])
def test_successful_commands_without_effective_attachment_fail_closed(firewall: Firewall, fallback: bool) -> None:
    firewall.fallback = fallback
    firewall.ignore_jump = True
    with pytest.raises(RuntimeError, match="verif"):
        policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test")

    assert set(firewall.chains) == {"INPUT", "FORWARD", "DOCKER-USER"}


def test_partial_fallback_failure_removes_only_owned_rules(firewall: Firewall) -> None:
    firewall.fallback = True
    firewall.fail_jump = True
    before = {chain: list(rules) for chain, rules in firewall.chains.items()}
    with pytest.raises(RuntimeError, match="iptables"):
        policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test")

    assert firewall.chains == before


@pytest.mark.parametrize("fallback", [False, True])
def test_cleanup_preserves_similar_labels_and_existing_rules(firewall: Firewall, fallback: bool) -> None:
    firewall.fallback = fallback
    before = {chain: list(rules) for chain, rules in firewall.chains.items()}
    policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test-extra")
    policy.apply_egress_rules_atomic("172.18.0.3", [], "tldw-run-test")

    policy.delete_rules_by_label("tldw-run-test")
    assert firewall.verdict("FORWARD", "172.18.0.2", "192.0.2.11") == "DROP"
    policy.delete_rules_by_label("tldw-run-test-extra")
    assert firewall.chains == before


@pytest.mark.parametrize(
    ("source", "targets", "label"),
    [
        ("::1", ["192.0.2.10"], "test"),
        ("172.18.0.2", ["2001:db8::1/128"], "test"),
        ("172.18.0.2", ["192.0.2.10"], "test\n-A FORWARD -j ACCEPT"),
    ],
)
def test_invalid_policy_inputs_never_mutate_firewall(firewall: Firewall, source, targets, label) -> None:
    before = {chain: list(rules) for chain, rules in firewall.chains.items()}
    with pytest.raises(ValueError):
        policy.apply_egress_rules_atomic(source, targets, label)
    assert firewall.chains == before


def test_firewall_verification_timeout_fails_closed(firewall: Firewall, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))

    monkeypatch.setattr(policy.subprocess, "check_output", timeout)
    with pytest.raises(RuntimeError, match="verif"):
        policy.apply_egress_rules_atomic("172.18.0.2", [], "tldw-run-test")


@pytest.mark.parametrize("fallback", [False, True])
def test_missing_terminal_drop_cannot_release_execution(firewall: Firewall, fallback: bool) -> None:
    firewall.fallback = fallback
    firewall.omit_drop = True
    with pytest.raises(RuntimeError, match="verif"):
        policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test")
    assert set(firewall.chains) == {"INPUT", "FORWARD", "DOCKER-USER"}


def test_cleanup_keeps_drop_when_attached_jump_cannot_be_deleted(firewall: Firewall) -> None:
    policy.apply_egress_rules_atomic("172.18.0.2", [], "tldw-run-test")
    firewall.fail_delete = True
    policy.delete_rules_by_label("tldw-run-test")
    assert firewall.verdict("FORWARD", "172.18.0.2", "192.0.2.10") == "DROP"


def test_failed_refresh_keeps_previous_policy_enforced(firewall: Firewall) -> None:
    policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test")
    firewall.fallback = True
    with pytest.raises(RuntimeError, match="iptables"):
        policy.refresh_egress_rules("172.18.0.2", ["192.0.2.11"], "tldw-run-test")
    assert firewall.verdict("FORWARD", "172.18.0.2", "192.0.2.10") == "ACCEPT"
    assert firewall.verdict("FORWARD", "172.18.0.2", "192.0.2.11") == "DROP"


def test_refresh_verification_error_does_not_remove_enforcement(firewall: Firewall, monkeypatch: pytest.MonkeyPatch) -> None:
    policy.apply_egress_rules_atomic("172.18.0.2", ["192.0.2.10"], "tldw-run-test")

    def unavailable(*args):
        raise RuntimeError("verification unavailable")

    monkeypatch.setattr(policy, "_verify_policy", unavailable)
    with pytest.raises(RuntimeError, match="verif"):
        policy.refresh_egress_rules("172.18.0.2", ["192.0.2.11"], "tldw-run-test")
    assert firewall.verdict("FORWARD", "172.18.0.2", "192.0.2.99") == "DROP"


def test_contending_firewall_writer_cannot_mutate_policy(firewall: Firewall, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    lock_path = tmp_path / "netfilter.lock"
    monkeypatch.setattr(policy, "_EGRESS_LOCK_PATH", lock_path, raising=False)
    monkeypatch.setattr(policy, "_IPTABLES_CLEANUP_TIMEOUT_SEC", 0.01)
    before = {chain: list(rules) for chain, rules in firewall.chains.items()}
    with FileLock(lock_path, timeout=0):
        with pytest.raises(RuntimeError, match="lock"):
            policy.apply_egress_rules_atomic("172.18.0.2", [], "tldw-run-test")
    assert firewall.chains == before
