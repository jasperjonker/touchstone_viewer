from __future__ import annotations

from pathlib import Path

import pytest

from touchstone_viewer.touchstone import gamma_to_impedance, load_touchstone


def test_load_touchstone_ri_format(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.s1p"
    file_path.write_text(
        "# Hz S RI R 50\n"
        "1000000000 0.1 0.2\n"
        "2000000000 0.2 0.3\n",
        encoding="utf-8",
    )

    data = load_touchstone(file_path)

    assert data.frequencies_hz.tolist() == [1.0e9, 2.0e9]
    assert data.reference_impedance_ohms == pytest.approx(50.0)
    assert data.gamma[0] == pytest.approx(complex(0.1, 0.2))


def test_load_touchstone_db_format(tmp_path: Path) -> None:
    file_path = tmp_path / "db.s1p"
    file_path.write_text(
        "# MHz S DB R 50\n"
        "2400 -6 180\n",
        encoding="utf-8",
    )

    data = load_touchstone(file_path)

    assert data.frequencies_hz[0] == pytest.approx(2.4e9)
    assert abs(data.gamma[0]) == pytest.approx(10 ** (-6 / 20))
    assert gamma_to_impedance(complex(0.0, 0.0), 50.0) == pytest.approx(50.0 + 0.0j)
