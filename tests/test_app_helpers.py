from __future__ import annotations

from pathlib import Path

import pytest

from touchstone_viewer.app import (
    _clamp_frequency_region_hz,
    _default_frequency_region_hz,
    _resolve_browser_directory,
)


def test_default_frequency_region_uses_center_window() -> None:
    start_hz, stop_hz = _default_frequency_region_hz((0.0, 100.0))

    assert start_hz == pytest.approx(39.0)
    assert stop_hz == pytest.approx(61.0)


def test_clamp_frequency_region_preserves_width_when_possible() -> None:
    start_hz, stop_hz = _clamp_frequency_region_hz((90.0, 130.0), (0.0, 100.0))

    assert start_hz == pytest.approx(60.0)
    assert stop_hz == pytest.approx(100.0)


def test_resolve_browser_directory_prefers_saved_directory(tmp_path: Path) -> None:
    saved_directory = tmp_path / "saved"
    downloads_directory = tmp_path / "downloads"
    home_directory = tmp_path / "home"
    saved_directory.mkdir()
    downloads_directory.mkdir()
    home_directory.mkdir()

    result = _resolve_browser_directory(saved_directory, downloads_directory, home_directory)

    assert result == saved_directory.resolve()


def test_resolve_browser_directory_falls_back_to_downloads(tmp_path: Path) -> None:
    downloads_directory = tmp_path / "downloads"
    home_directory = tmp_path / "home"
    downloads_directory.mkdir()
    home_directory.mkdir()

    result = _resolve_browser_directory(None, downloads_directory, home_directory)

    assert result == downloads_directory.resolve()
