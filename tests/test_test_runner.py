from __future__ import annotations

import subprocess

from touchstone_viewer import test_runner


def test_test_runner_invokes_uv_with_coverage(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(test_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(test_runner.sys, "argv", ["touch-test", "-q"])

    exit_code = test_runner.main()

    assert exit_code == 0
    assert commands == [[
        "uv",
        "run",
        "--group",
        "dev",
        "pytest",
        "--cov=touchstone_viewer",
        "--cov-report=term-missing",
        "-q",
    ]]
