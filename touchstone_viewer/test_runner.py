from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    command = [
        "uv",
        "run",
        "--group",
        "dev",
        "pytest",
        "--cov=touchstone_viewer",
        "--cov-report=term-missing",
    ]
    command.extend(sys.argv[1:])

    completed = subprocess.run(command, check=False)
    return completed.returncode
