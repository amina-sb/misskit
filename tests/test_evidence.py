"""The README's simulation tables must be what the evidence script prints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evidence.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evidence"] = mod
    spec.loader.exec_module(mod)
    return mod


evidence = _load()


def test_summarise_by_hand() -> None:
    est = np.array([0.7, 0.8, 0.9, 1.0])
    lo = est - 0.15
    hi = est + 0.15
    cell = evidence.summarise(est, lo, hi)
    assert cell.bias == pytest.approx(0.05)
    assert cell.rmse == pytest.approx(np.sqrt((0.01 + 0 + 0.01 + 0.04) / 4))
    assert cell.coverage == 0.75  # 1.0 - 0.15 = 0.85 misses 0.8
    assert cell.width == pytest.approx(0.3)
    assert cell.cov_lo < 0.75 < cell.cov_hi


def test_readme_blocks_parse() -> None:
    text = "x\n<!-- evidence:main:start -->\n| a |\n<!-- evidence:main:end -->\n"
    assert evidence.readme_blocks(text) == {"main": "| a |", "spread": "", "delta": ""}


def test_readme_matches_evidence_script() -> None:
    assert evidence.main(["--check"]) == 0
