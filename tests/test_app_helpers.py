from __future__ import annotations

import pytest

from touchstone_viewer.app import _clamp_frequency_region_hz, _default_frequency_region_hz


def test_default_frequency_region_uses_center_window() -> None:
    start_hz, stop_hz = _default_frequency_region_hz((0.0, 100.0))

    assert start_hz == pytest.approx(39.0)
    assert stop_hz == pytest.approx(61.0)


def test_clamp_frequency_region_preserves_width_when_possible() -> None:
    start_hz, stop_hz = _clamp_frequency_region_hz((90.0, 130.0), (0.0, 100.0))

    assert start_hz == pytest.approx(60.0)
    assert stop_hz == pytest.approx(100.0)
