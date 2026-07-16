"""CLI tests for the all-pair source-only preflight helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_preflight_pairs_script_runs_all_pairs_with_root_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    script = _load_script_module("preflight_source_only_pairs.py")
    captured: list[list[str]] = []

    def fake_run(command, check, env):
        captured.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/preflight_source_only_pairs.py",
            "--output-root",
            str(tmp_path / "preflight"),
            "--root",
            "PTBXL=/data/ptbxl",
            "--root",
            "CHAPMAN=/data/chapman",
            "--root",
            "SPH=/data/sph",
            "--root",
            "CODE15=/data/code15",
            "--root",
            "CPSC=/data/cpsc",
        ],
    )

    exit_code = script.main()

    assert exit_code == 0
    assert len(captured) == 20
    pairs = {
        (
            command[command.index("--source-dataset") + 1],
            command[command.index("--target-dataset") + 1],
        )
        for command in captured
    }
    assert len(pairs) == 20
    assert all(source != target for source, target in pairs)
    assert {source for source, _ in pairs} == {"PTBXL", "CHAPMAN", "SPH", "CODE15", "CPSC"}
    assert {target for _, target in pairs} == {"PTBXL", "CHAPMAN", "SPH", "CODE15", "CPSC"}
    assert captured[0].count("--preflight-only") == 1
    assert "--source-dataset" in captured[0]
    assert "--target-dataset" in captured[0]
    assert "PTBXL" in captured[0]
    assert "CHAPMAN" in captured[0]
    assert "--source-root" in captured[0]
    assert "/data/ptbxl" in captured[0]
    assert "--target-root" in captured[0]
    assert "/data/chapman" in captured[0]
    assert any("ptbxl_to_sph" in item for item in captured[1])
