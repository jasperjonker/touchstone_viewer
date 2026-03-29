from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from touchstone_viewer.touchstone import TouchstoneData, gamma_to_impedance, load_touchstone


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


def test_load_touchstone_sorts_data_and_interpolates(tmp_path: Path) -> None:
    file_path = tmp_path / "ma.s1p"
    file_path.write_text(
        "! comment line\n"
        "# GHz S MA R 50\n"
        "3.0 0.3 -90\n"
        "2.0 0.5 0\n"
        "2.5 0.4 180\n",
        encoding="utf-8",
    )

    data = load_touchstone(file_path)

    assert data.frequencies_hz.tolist() == [2.0e9, 2.5e9, 3.0e9]
    assert data.interpolated_gamma(2.25e9) is not None
    assert data.interpolated_gamma(1.9e9) is None


def test_touchstone_impedance_helpers_handle_open_condition() -> None:
    data = TouchstoneData(
        path=Path("dummy.s1p"),
        label="dummy",
        frequencies_hz=np.array([1.0]),
        gamma=np.array([1.0 + 0.0j]),
        reference_impedance_ohms=50.0,
    )

    impedance = data.impedance_ohms()

    assert not np.isfinite(impedance[0])
    assert not np.isfinite(gamma_to_impedance(1.0 + 0.0j, 50.0))


def test_load_touchstone_rejects_invalid_input(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.s1p"
    invalid_path = tmp_path / "invalid.s1p"
    invalid_path.write_text("# GHz S XY R 50\n2.4 0.1 0.2\n", encoding="utf-8")
    empty_path = tmp_path / "empty.s1p"
    empty_path.write_text("! only comments\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_touchstone(missing_path)

    with pytest.raises(ValueError):
        load_touchstone(invalid_path)

    with pytest.raises(ValueError):
        load_touchstone(empty_path)
