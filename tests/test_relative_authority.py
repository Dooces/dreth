from __future__ import annotations

import ast
from pathlib import Path

from dreth.relative_authority import NethraNodeRef, RelativeAuthorityRecord


ROOT = Path(__file__).resolve().parents[1]


def _record(**kwargs) -> RelativeAuthorityRecord:
    return RelativeAuthorityRecord(
        node=NethraNodeRef(node_id="n0", kind="var", var=0),
        context_key="ctx",
        **kwargs,
    )


def test_authority_score_increases_with_positive_evidence() -> None:
    base = _record()
    stronger = _record(wins=2, reuse_count=3, downstream_support=4)

    assert stronger.authority_score() > base.authority_score()


def test_authority_score_decreases_with_losses_and_failures() -> None:
    base = _record(wins=5, reuse_count=2)
    weaker = _record(wins=5, reuse_count=2, losses=3, failures=2)

    assert weaker.authority_score() < base.authority_score()


def test_should_prefer_over_chooses_higher_relative_score() -> None:
    higher = _record(wins=3, reuse_count=2)
    lower = _record(wins=1, losses=1)

    assert higher.should_prefer_over(lower)
    assert not lower.should_prefer_over(higher)


def test_should_localize_failure_below_broad_failure_threshold() -> None:
    local = _record(failures=2)
    broad = _record(failures=3)

    assert local.should_localize_failure(global_failure_count_threshold=3)
    assert not broad.should_localize_failure(global_failure_count_threshold=3)


def test_relative_authority_has_no_runtime_imports() -> None:
    source = (ROOT / "dreth" / "relative_authority.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)

    forbidden = {
        "dreth.agent",
        "agent",
        "dreth.ledger",
        "ledger",
        "dreth.fit",
        "fit",
        "dreth.fit.fit_var",
        "fit.fit_var",
        "dreth.sentinels",
        "sentinels",
    }
    assert imported.isdisjoint(forbidden)
    assert "ChainedAgent" not in source
    assert "fit_var" not in source


def test_agent_does_not_import_relative_authority_yet() -> None:
    agent_source = (ROOT / "dreth" / "agent.py").read_text()

    assert "relative_authority" not in agent_source


def test_glossary_says_certificate_is_not_proof_of_truth() -> None:
    glossary = (ROOT / "docs" / "glossary.md").read_text().lower()

    assert "not proof of truth" in glossary
