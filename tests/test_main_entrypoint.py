from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


def test_main_entrypoint_runs_when_executed_as_script(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    entrypoint = project_root / "touchstone_viewer" / "__main__.py"

    called = {"count": 0}

    package = types.ModuleType("touchstone_viewer")
    package.__path__ = [str(project_root / "touchstone_viewer")]  # type: ignore[attr-defined]

    app_module = types.ModuleType("touchstone_viewer.app")

    def fake_main() -> int:
        called["count"] += 1
        return 0

    app_module.main = fake_main  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "touchstone_viewer", package)
    monkeypatch.setitem(sys.modules, "touchstone_viewer.app", app_module)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(entrypoint), run_name="__main__")

    assert exc_info.value.code == 0
    assert called["count"] == 1
