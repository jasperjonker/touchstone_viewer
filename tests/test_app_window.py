from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets
import pytest

from touchstone_viewer.app import LAST_OPEN_DIRECTORY_KEY, TouchstoneViewerWindow


class _FakeMimeData:
    def __init__(self, urls: list["_FakeUrl"]) -> None:
        self._urls = urls

    def urls(self) -> list["_FakeUrl"]:
        return self._urls


class _FakeUrl:
    def __init__(self, path: Path) -> None:
        self._path = path

    def isLocalFile(self) -> bool:
        return True

    def toLocalFile(self) -> str:
        return str(self._path)


class _FakeDropEvent:
    def __init__(self, paths: list[Path]) -> None:
        self._mime_data = _FakeMimeData([_FakeUrl(path) for path in paths])
        self.accepted = False
        self.ignored = False

    def mimeData(self) -> _FakeMimeData:
        return self._mime_data

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


def _write_touchstone_file(path: Path) -> Path:
    path.write_text(
        "# GHz S MA R 50\n"
        "2.0 0.50 45\n"
        "2.5 0.25 10\n"
        "3.0 0.75 -30\n",
        encoding="utf-8",
    )
    return path


def _write_touchstone_file_with_content(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _write_touchstone_two_port_file(path: Path) -> Path:
    path.write_text(
        "# GHz S RI R 50\n"
        "2.0 0.10 0.00 0.60 0.00 0.05 0.00 0.20 0.00\n"
        "2.5 0.20 0.10 0.50 -0.10 0.10 0.05 0.30 0.10\n",
        encoding="utf-8",
    )
    return path


def test_window_initial_load_builds_plots(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "first.s1p")

    window = TouchstoneViewerWindow([file_path])

    assert len(window.traces) == 1
    assert window.marker_line is not None
    assert window.aoi_region_item is not None
    assert window.summary_label.text() == "1 trace(s) loaded, 1 visible"
    assert window.aoi_start_input.isEnabled()
    assert window.aoi_stop_input.isEnabled()
    assert window.marker_table.rowCount() == 1
    assert window.tab_widget.count() == 2
    assert window.controls_panel.isHidden()
    assert window.controls_toggle_button.arrowType() == QtCore.Qt.ArrowType.RightArrow

    window.close()


def test_controls_panel_and_overlay_toggles(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "controls.s1p")

    window = TouchstoneViewerWindow([file_path])

    assert window.controls_panel.isHidden()
    window.controls_toggle_button.setChecked(True)

    assert not window.controls_panel.isHidden()
    assert window.controls_toggle_button.arrowType() == QtCore.Qt.ArrowType.DownArrow
    assert window.aoi_region_item is not None
    assert window.aoi_region_item.isVisible()
    assert window.marker_line is not None
    assert window.marker_line.isVisible()
    assert not window.marker_table.isHidden()
    assert not window.s21_marker_table.isHidden()

    window.aoi_enabled_checkbox.setChecked(False)

    assert not window.aoi_start_input.isEnabled()
    assert not window.aoi_stop_input.isEnabled()
    assert window.aoi_region_item is not None
    assert not window.aoi_region_item.isVisible()

    window.marker_enabled_checkbox.setChecked(False)

    assert window.marker_line is not None
    assert window.s21_marker_line is not None
    assert not window.marker_line.isVisible()
    assert not window.s21_marker_line.isVisible()
    assert window.marker_table.rowCount() == 0
    assert window.s21_marker_table.rowCount() == 0
    assert window.marker_table.isHidden()
    assert window.s21_marker_table.isHidden()

    window.marker_enabled_checkbox.setChecked(True)

    assert window.marker_line.isVisible()
    assert window.s21_marker_line.isVisible()
    assert window.marker_table.rowCount() == 1
    assert window.s21_marker_table.rowCount() == 1
    assert not window.marker_table.isHidden()
    assert not window.s21_marker_table.isHidden()

    window.close()


def test_view_and_trace_controls_drive_visibility_and_reference_state(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    first_file = _write_touchstone_file(tmp_path / "first_control.s1p")
    second_file = _write_touchstone_file_with_content(
        tmp_path / "second_control.s1p",
        "# GHz S MA R 50\n"
        "2.0 0.40 0\n"
        "2.5 0.50 10\n"
        "3.0 0.60 -20\n",
    )

    window = TouchstoneViewerWindow([first_file, second_file])

    assert window.trace_visibility_list.count() == 2
    assert window.reference_trace_combo.count() == 3
    assert window.reference_trace_combo.minimumWidth() >= 300

    window.frequency_unit_combo.setCurrentText("MHz")

    assert window.frequency_scale.unit == "MHz"
    assert window.marker_table.item(0, 1).text() == "2500.000000"
    assert window.marker_frequency_input.value() == pytest.approx(2500.0)
    assert window.s11_plot.getPlotItem().getAxis("bottom").labelText == "Frequency (MHz)"
    assert window.s11_plot.getPlotItem().getAxis("bottom").labelUnits == ""

    window.marker_frequency_input.setValue(2600.0)

    assert window.marker_table.item(0, 1).text() == "2600.000000"
    assert window.marker_line is not None
    assert window.marker_line.value() == pytest.approx(2600.0)

    window.marker_frequency_input.setValue(2500.0)

    first_item = window.trace_visibility_list.item(0)
    first_item.setCheckState(QtCore.Qt.CheckState.Unchecked)

    assert window.summary_label.text() == "2 trace(s) loaded, 1 visible"
    assert window.marker_table.rowCount() == 1
    assert window.trace_visibility_list.item(0).checkState() == QtCore.Qt.CheckState.Unchecked

    window.reference_trace_combo.setCurrentIndex(2)

    assert window.reference_trace_path == second_file.resolve()
    reference_rows = [
        row
        for row in range(window.marker_table.rowCount())
        if window.marker_table.item(row, 0).text().endswith("(ref)")
    ]
    assert reference_rows == [0]
    assert window.marker_table.item(reference_rows[0], 0).font().bold()
    assert window.trace_visibility_list.item(1).font().bold()
    assert window.marker_table.item(0, 3).text() == "+0.000"
    assert window.marker_table.item(0, 5).text() == "+0.0000"

    window.trace_visibility_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)

    assert window.summary_label.text() == "2 trace(s) loaded, 2 visible"
    assert window.marker_table.rowCount() == 2
    assert window.marker_table.item(0, 3).text() == "-6.021"
    assert window.marker_table.item(0, 5).text() == "-0.2500"
    assert window.marker_table.item(1, 0).text().endswith("(ref)")
    assert window.marker_table.item(1, 3).text() == "+0.000"
    assert window.marker_table.item(1, 5).text() == "+0.0000"

    window.marker_table.sortItems(4, QtCore.Qt.SortOrder.DescendingOrder)

    assert window.marker_table.item(0, 4).text() == "0.5000"
    assert window.marker_table.item(0, 0).font().bold()
    assert window.marker_table.item(1, 4).text() == "0.2500"

    window.close()


def test_empty_plots_use_negative_db_range_and_default_threshold(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    window = TouchstoneViewerWindow([])

    s11_y_range = window.s11_plot.getPlotItem().viewRange()[1]
    s21_y_range = window.s21_plot.getPlotItem().viewRange()[1]

    assert s11_y_range == pytest.approx([-40.0, 0.0])
    assert s21_y_range == pytest.approx([-40.0, 0.0])
    assert window.s11_threshold_line is not None
    assert window.s21_threshold_line is not None
    assert window.s11_threshold_line.value() == pytest.approx(-10.0)
    assert window.s21_threshold_line.value() == pytest.approx(-10.0)
    assert not window.threshold_enabled_checkbox.isChecked()
    assert not window.threshold_input.isEnabled()
    assert not window.s11_threshold_line.isVisible()
    assert not window.s21_threshold_line.isVisible()

    window.threshold_enabled_checkbox.setChecked(True)
    window.threshold_input.setValue(6.0)

    assert window.threshold_input.isEnabled()
    assert window.s11_threshold_line.isVisible()
    assert window.s21_threshold_line.isVisible()
    assert window.s11_threshold_line.value() == pytest.approx(-6.0)
    assert window.s21_threshold_line.value() == pytest.approx(-6.0)

    window.close()


def test_open_files_dialog_uses_default_directory_and_remembers_last_folder(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    browse_directory = tmp_path / "downloads"
    browse_directory.mkdir()
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    file_path = _write_touchstone_file(nested_directory / "second.s1p")

    captured_directory = {"value": None}

    def fake_get_open_file_names(*args):
        captured_directory["value"] = args[2]
        return ([str(file_path)], "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", fake_get_open_file_names)

    window = TouchstoneViewerWindow([])
    monkeypatch.setattr(window, "_default_browser_directory", lambda: browse_directory)

    window._open_files_dialog()

    assert captured_directory["value"] == str(browse_directory)
    assert len(window.traces) == 1
    assert window.settings.value(LAST_OPEN_DIRECTORY_KEY) == str(file_path.parent)

    window.close()


def test_load_files_reports_errors_and_skips_duplicates(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    valid_file = _write_touchstone_file(tmp_path / "valid.s1p")
    invalid_file = tmp_path / "invalid.s1p"
    invalid_file.write_text("# GHz S MA R 50\n2.4 1.0\n", encoding="utf-8")

    warnings: list[tuple[str, str]] = []

    def fake_warning(_parent, title: str, message: str) -> None:
        warnings.append((title, message))

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", fake_warning)

    window = TouchstoneViewerWindow([valid_file])
    window.load_files([valid_file, invalid_file])

    assert len(window.traces) == 1
    assert warnings
    assert "invalid.s1p" in warnings[0][1]

    window.close()


def test_aoi_controls_update_region_and_clear_resets_state(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "third.s1p")

    window = TouchstoneViewerWindow([file_path])
    window.aoi_unit_combo.setCurrentText("GHz")
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)

    assert window.aoi_region_hz == (2.1e9, 2.7e9)

    window.clear_traces()

    assert len(window.traces) == 0
    assert not window.aoi_start_input.isEnabled()
    assert not window.aoi_stop_input.isEnabled()
    assert window.marker_table.rowCount() == 0

    window.close()


def test_drop_event_loads_files(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "drop.s1p")
    event = _FakeDropEvent([file_path])

    window = TouchstoneViewerWindow([])
    window.dropEvent(event)

    assert event.accepted
    assert len(window.traces) == 1

    window.close()


def test_s2p_load_populates_s21_tab(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_two_port_file(tmp_path / "amp.s2p")

    window = TouchstoneViewerWindow([file_path])

    assert len(window.traces) == 1
    assert window.traces[0].s21_curve is not None
    assert window.s21_marker_line is not None
    assert window.s21_marker_table.rowCount() == 1
    assert window.s21_marker_table.item(0, 2).text() != "not available"

    window.close()
