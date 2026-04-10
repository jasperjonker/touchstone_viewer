from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
import pytest

from touchstone_viewer.app import LAST_OPEN_DIRECTORY_KEY, TouchstoneViewerWindow
from touchstone_viewer.version import __version__


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


class _FakeMouseClickEvent:
    def __init__(
        self,
        scene_position: QtCore.QPointF,
        button: QtCore.Qt.MouseButton,
    ) -> None:
        self._scene_position = scene_position
        self._button = button

    def scenePos(self) -> QtCore.QPointF:
        return self._scene_position

    def button(self) -> QtCore.Qt.MouseButton:
        return self._button


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


def _matching_stage_state(window: TouchstoneViewerWindow) -> list[tuple[str, str, str, float, bool]]:
    return [
        (
            stage_controls.topology_combo.currentText(),
            stage_controls.component_combo.currentText(),
            stage_controls.unit_combo.currentText(),
            stage_controls.value_input.value(),
            stage_controls.enabled_checkbox.isChecked(),
        )
        for stage_controls in window.matching_stage_controls
    ]


def test_window_initial_load_builds_plots(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "first.s1p")

    window = TouchstoneViewerWindow([file_path])

    assert len(window.traces) == 1
    assert window.marker_line is not None
    assert window.aoi_region_item is None
    assert window.windowTitle() == f"Touchstone Viewer - v{__version__}"
    assert window.summary_label.text() == "1 trace(s) loaded, 1 visible"
    assert not window.aoi_enabled_checkbox.isChecked()
    assert not window.aoi_start_input.isEnabled()
    assert not window.aoi_stop_input.isEnabled()
    assert not window.clear_aoi_button.isEnabled()
    assert window.marker_table.rowCount() == 1
    assert window.tab_widget.count() == 3
    assert window.controls_panel.isHidden()
    assert window.controls_toggle_button.arrowType() == QtCore.Qt.ArrowType.RightArrow
    assert window.smith_plot.getViewBox().state["mouseEnabled"] == [True, True]
    assert window.smith_plot.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode

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
    assert not window.aoi_enabled_checkbox.isChecked()
    assert not window.aoi_start_input.isEnabled()
    assert not window.aoi_stop_input.isEnabled()

    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)

    assert not window.controls_panel.isHidden()
    assert window.controls_toggle_button.arrowType() == QtCore.Qt.ArrowType.DownArrow
    assert window.aoi_region_item is not None
    assert not window.aoi_region_item.isVisible()
    assert window.marker_line is not None
    assert window.marker_line.isVisible()
    assert not window.marker_table.isHidden()
    assert not window.s21_marker_table.isHidden()

    window.aoi_enabled_checkbox.setChecked(True)

    assert window.aoi_start_input.isEnabled()
    assert window.aoi_stop_input.isEnabled()
    assert window.aoi_region_item is not None
    assert window.aoi_region_item.isVisible()

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
    assert window.marker_frequency_input.suffix() == ""
    assert window.marker_frequency_unit_label.text() == "GHz"
    assert window.aoi_display_unit_label.text() == "GHz"

    window.frequency_unit_combo.setCurrentText("MHz")

    assert window.frequency_scale.unit == "MHz"
    assert window.marker_frequency_unit_label.text() == "MHz"
    assert window.aoi_display_unit_label.text() == "MHz"
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


def test_loaded_traces_are_sorted_naturally_in_controls(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    trace_10 = _write_touchstone_file(tmp_path / "trace_10.s1p")
    trace_2 = _write_touchstone_file(tmp_path / "trace_2.s1p")
    trace_1 = _write_touchstone_file(tmp_path / "trace_1.s1p")

    window = TouchstoneViewerWindow([trace_10, trace_2, trace_1])

    assert [trace.data.label for trace in window.traces] == ["trace_1", "trace_2", "trace_10"]
    assert [window.trace_visibility_list.item(index).text() for index in range(3)] == [
        "trace_1",
        "trace_2",
        "trace_10",
    ]
    assert [window.reference_trace_combo.itemText(index) for index in range(4)] == [
        "None",
        "trace_1",
        "trace_2",
        "trace_10",
    ]
    assert [window.match_trace_combo.itemText(index) for index in range(3)] == [
        "trace_1",
        "trace_2",
        "trace_10",
    ]

    window.close()


def test_marker_frequency_input_selects_all_on_focus(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "marker_focus.s1p")

    window = TouchstoneViewerWindow([file_path])
    focus_event = QtGui.QFocusEvent(
        QtCore.QEvent.Type.FocusIn,
        QtCore.Qt.FocusReason.TabFocusReason,
    )
    QtWidgets.QApplication.sendEvent(window.marker_frequency_input, focus_event)
    qapp.processEvents()

    line_edit = window.marker_frequency_input.lineEdit()
    assert line_edit is not None
    assert line_edit.selectedText() == window.marker_frequency_input.text()

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


def test_reset_view_restores_frequency_and_smith_ranges(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_two_port_file(tmp_path / "reset_view.s2p")

    window = TouchstoneViewerWindow([file_path])
    trace = window.traces[0]
    default_smith_x_range, default_smith_y_range = window.smith_plot.getPlotItem().viewRange()

    window.s11_plot.getPlotItem().setXRange(2.1, 2.2, padding=0.0)
    window.s11_plot.getPlotItem().setYRange(-35.0, -25.0, padding=0.0)
    window.s21_plot.getPlotItem().setXRange(2.1, 2.2, padding=0.0)
    window.s21_plot.getPlotItem().setYRange(-25.0, -15.0, padding=0.0)
    window.smith_plot.getPlotItem().setXRange(-0.2, 0.2, padding=0.0)
    window.smith_plot.getPlotItem().setYRange(-0.2, 0.2, padding=0.0)

    window.reset_view_button.click()

    assert window.s11_plot.getPlotItem().viewRange()[0] == pytest.approx([2.0, 2.5])
    assert window.s11_plot.getPlotItem().viewRange()[1] == pytest.approx(
        [
            float(trace.data.s11_db().min()),
            float(trace.data.s11_db().max()),
        ]
    )
    assert window.s21_plot.getPlotItem().viewRange()[0] == pytest.approx([2.0, 2.5])
    assert window.s21_plot.getPlotItem().viewRange()[1] == pytest.approx(
        [
            float(trace.data.s21_db().min()),
            float(trace.data.s21_db().max()),
        ]
    )
    assert window.smith_plot.getPlotItem().viewRange()[0] == pytest.approx(default_smith_x_range)
    assert window.smith_plot.getPlotItem().viewRange()[1] == pytest.approx(default_smith_y_range)

    window.close()


def test_right_click_does_not_move_marker_but_left_click_still_does(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "clicks.s1p")

    window = TouchstoneViewerWindow([file_path])
    view_box = window.s11_plot.getPlotItem().vb
    scene_position = QtCore.QPointF(10.0, 20.0)
    monkeypatch.setattr(
        view_box,
        "sceneBoundingRect",
        lambda: QtCore.QRectF(0.0, 0.0, 100.0, 100.0),
    )
    monkeypatch.setattr(
        view_box,
        "mapSceneToView",
        lambda _position: QtCore.QPointF(2.8, -10.0),
    )

    original_marker_hz = window.marker_frequency_hz
    assert original_marker_hz is not None

    right_click = _FakeMouseClickEvent(
        scene_position,
        QtCore.Qt.MouseButton.RightButton,
    )
    window._handle_plot_click(window.s11_plot, right_click)

    assert window.marker_frequency_hz == pytest.approx(original_marker_hz)

    left_click = _FakeMouseClickEvent(
        scene_position,
        QtCore.Qt.MouseButton.LeftButton,
    )
    window._handle_plot_click(window.s11_plot, left_click)

    assert window.marker_frequency_hz == pytest.approx(2.8e9)
    assert window.marker_line is not None
    assert window.marker_line.value() == pytest.approx(2.8)

    window.close()


def test_frequency_unit_change_preserves_view_marker_and_aoi(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "preserve_view.s1p")

    window = TouchstoneViewerWindow([file_path])
    window.s11_plot.getPlotItem().setXRange(2.2, 2.4, padding=0.0)
    window.s11_plot.getPlotItem().setYRange(-18.0, -4.0, padding=0.0)
    window.marker_frequency_input.setValue(2.3)
    window.aoi_start_input.setValue(2.15)
    window.aoi_stop_input.setValue(2.45)

    window.frequency_unit_combo.setCurrentText("MHz")

    assert window.frequency_scale.unit == "MHz"
    assert window.marker_frequency_hz == pytest.approx(2.3e9)
    assert window.marker_line is not None
    assert window.marker_line.value() == pytest.approx(2300.0)
    assert window.aoi_region_hz == pytest.approx((2.15e9, 2.45e9))
    assert window.aoi_start_input.value() == pytest.approx(2150.0)
    assert window.aoi_stop_input.value() == pytest.approx(2450.0)
    assert window.aoi_display_unit_label.text() == "MHz"
    assert window.s11_plot.getPlotItem().viewRange()[0] == pytest.approx([2200.0, 2400.0])
    assert window.s11_plot.getPlotItem().viewRange()[1] == pytest.approx([-18.0, -4.0])

    window.close()


def test_s11_table_shows_aoi_area_as_an_extra_column(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file_with_content(
        tmp_path / "aoi_area.s1p",
        "# GHz S DB R 50\n"
        "2.0 -10 0\n"
        "2.5 -20 0\n"
        "3.0 -30 0\n",
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_stop_input.setValue(2.9)
    window.aoi_stop_input.setValue(3.0)

    assert [window.marker_table.horizontalHeaderItem(index).text() for index in range(9)] == [
        "Trace",
        "Freq",
        "S11 (dB)",
        "ΔRef (dB)",
        "|S11| (lin)",
        "ΔRef |S11| (lin)",
        "Angle (deg)",
        "Z (ohm)",
        "AOI Area (|dB|*GHz)",
    ]
    assert [window.marker_table.item(0, index).text() for index in range(9)] == [
        "aoi_area",
        "2.500000",
        "-20.000",
        "-",
        "0.1000",
        "-",
        "0.00",
        "61.11 + j0.00",
        "20.000",
    ]

    window.close()


def test_s11_table_shows_one_area_column_per_visible_preset_band(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file_with_content(
        tmp_path / "aoi_band_columns.s1p",
        "# GHz S DB R 50\n"
        "2.0 -10 0\n"
        "2.5 -20 0\n"
        "3.0 -30 0\n",
    )
    preset_names = iter([("GPS L1", True), ("Galileo E1", True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(preset_names),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.0)
    window.aoi_stop_input.setValue(2.5)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(2.5)
    window.aoi_stop_input.setValue(3.0)
    window.save_aoi_preset_button.click()

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GPS L1"].setChecked(True)
    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["Galileo E1"].setChecked(True)

    assert [
        window.marker_table.horizontalHeaderItem(index).text()
        for index in range(window.marker_table.columnCount())
    ] == [
        "Trace",
        "Freq",
        "S11 (dB)",
        "ΔRef (dB)",
        "|S11| (lin)",
        "ΔRef |S11| (lin)",
        "Angle (deg)",
        "Z (ohm)",
        "Galileo E1 Area (|dB|*GHz)",
        "GPS L1 Area (|dB|*GHz)",
    ]
    assert [window.marker_table.item(0, index).text() for index in range(10)] == [
        "aoi_band_columns",
        "2.500000",
        "-20.000",
        "-",
        "0.1000",
        "-",
        "0.00",
        "61.11 + j0.00",
        "12.500",
        "7.500",
    ]
    assert window.marker_table.horizontalHeaderItem(8).background().color().name() == "#f59e0b"
    assert window.marker_table.horizontalHeaderItem(9).background().color().name() == "#10b981"
    assert window.marker_table.item(0, 8).background().color().name() == "#f59e0b"
    assert window.marker_table.item(0, 9).background().color().name() == "#10b981"

    window.close()


def test_match_tab_applies_enabled_network_stage(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "match_trace.s1p")

    window = TouchstoneViewerWindow([file_path])

    assert window.match_trace_combo.count() == 1
    assert window.match_target_frequency_input.value() == pytest.approx(2.5)
    assert window.match_original_s11_db is not None
    assert window.match_transformed_s11_db is not None
    assert np.allclose(window.match_original_s11_db, window.match_transformed_s11_db)
    assert _matching_stage_state(window) == [
        ("Shunt", "C", "pF", 0.0, False),
        ("Series", "R", "ohm", 0.0, True),
        ("Shunt", "C", "pF", 0.0, False),
    ]
    assert "antenna/load -> coax/feed" in window.match_order_label.text()
    assert window.match_suggestion_table.rowCount() > 0
    assert all(
        window.match_suggestion_table.item(row, 0).text().split()[1] in {"L", "C"}
        for row in range(window.match_suggestion_table.rowCount())
    )
    assert "append at the bottom" in window.match_suggestion_label.text()
    assert "reactive" in window.match_suggestion_label.text().lower()

    window.match_target_frequency_input.setValue(2.3)

    assert window.marker_frequency_hz == pytest.approx(2.3e9)
    assert window.match_summary_label.text().startswith("At 2.300000 GHz")

    window.add_matching_stage_button.click()

    assert len(window.matching_stage_controls) == 4

    stage_controls = window.matching_stage_controls[-1]
    stage_controls.component_combo.setCurrentText("R")
    stage_controls.unit_combo.setCurrentText("ohm")
    stage_controls.value_input.setValue(10.0)
    stage_controls.enabled_checkbox.setChecked(True)

    assert window.match_transformed_s11_db is not None
    assert not np.allclose(window.match_original_s11_db, window.match_transformed_s11_db)
    assert window.match_s11_marker_line is not None
    assert window.match_summary_label.text().startswith("At 2.300000 GHz")
    assert "Matched" in window.match_summary_label.text()

    window.match_suggestion_table.selectRow(0)
    selected_action = window.match_suggestion_table.item(0, 0).text()
    selected_value = window.match_suggestion_table.item(0, 1).text()
    window.apply_matching_suggestion_button.click()

    assert len(window.matching_stage_controls) == 5
    appended_stage = window.matching_stage_controls[-1]
    assert (
        f"{appended_stage.topology_combo.currentText()} {appended_stage.component_combo.currentText()}"
        == selected_action
    )
    assert appended_stage.component_combo.currentText() in {"L", "C"}
    assert (
        f"{appended_stage.value_input.value():g} {appended_stage.unit_combo.currentText()}"
        == selected_value
    )

    appended_stage.remove_button.click()

    assert len(window.matching_stage_controls) == 4

    window.reset_matching_button.click()

    assert _matching_stage_state(window) == [
        ("Shunt", "C", "pF", 0.0, False),
        ("Series", "R", "ohm", 0.0, True),
        ("Shunt", "C", "pF", 0.0, False),
    ]

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


def test_loading_new_files_refits_the_plot_view_to_all_visible_traces(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    first_file = _write_touchstone_file_with_content(
        tmp_path / "first_fit.s1p",
        "# GHz S DB R 50\n"
        "2.0 -12 0\n"
        "2.5 -10 0\n"
        "3.0 -8 0\n",
    )
    second_file = _write_touchstone_file_with_content(
        tmp_path / "second_fit.s1p",
        "# GHz S DB R 50\n"
        "1.0 -25 0\n"
        "4.0 -2 0\n",
    )

    window = TouchstoneViewerWindow([first_file])
    window.s11_plot.getPlotItem().setXRange(2.2, 2.4, padding=0.0)
    window.s11_plot.getPlotItem().setYRange(-11.0, -9.0, padding=0.0)

    window.load_files([second_file])

    assert len(window.traces) == 2
    assert window.s11_plot.getPlotItem().viewRange()[0] == pytest.approx([1.0, 4.0])
    assert window.s11_plot.getPlotItem().viewRange()[1] == pytest.approx([-25.0, -2.0])

    window.close()


def test_marker_table_can_be_exported_to_csv(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "exportable.s1p")
    export_path = tmp_path / "marker_table.csv"

    def fake_get_save_file_name(*args):
        return (str(export_path), "CSV files (*.csv)")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", fake_get_save_file_name)

    window = TouchstoneViewerWindow([file_path])
    window._export_table_to_csv(window.marker_table, suggested_name="s11_marker_table.csv")
    expected_header = ",".join(
        window.marker_table.horizontalHeaderItem(column).text()
        for column in range(window.marker_table.columnCount())
    )
    expected_row = ",".join(
        window.marker_table.item(0, column).text()
        for column in range(window.marker_table.columnCount())
    )

    assert export_path.read_text(encoding="utf-8").splitlines() == [
        expected_header,
        expected_row,
    ]

    window.close()


def test_common_settings_are_persisted_in_yaml_config(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "persisted.s1p")
    config_path = Path(os.environ["XDG_CONFIG_HOME"]) / "touchstone_viewer" / "config.yaml"

    window = TouchstoneViewerWindow([file_path])
    window.force_light_theme_checkbox.setChecked(True)
    window.frequency_unit_combo.setCurrentText("MHz")
    window.aoi_enabled_checkbox.setChecked(True)
    window.aoi_start_input.setValue(2400.0)
    window.aoi_stop_input.setValue(2500.0)
    window.marker_frequency_input.setValue(2450.0)
    window.threshold_enabled_checkbox.setChecked(True)
    window.threshold_input.setValue(6.0)
    window.close()

    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert 'frequency_unit_mode: "MHz"' in config_text
    assert "force_light_mode: true" in config_text
    assert "aoi_visible: true" in config_text
    assert "marker_frequency_hz: 2450000000.0" in config_text
    assert "aoi_start_hz: 2400000000.0" in config_text
    assert "aoi_stop_hz: 2500000000.0" in config_text

    restored_window = TouchstoneViewerWindow([file_path])

    assert restored_window.force_light_theme_checkbox.isChecked()
    assert restored_window.frequency_unit_combo.currentText() == "MHz"
    assert not restored_window.aoi_enabled_checkbox.isChecked()
    assert not restored_window.aoi_start_input.isEnabled()
    assert not restored_window.aoi_stop_input.isEnabled()
    assert restored_window.aoi_display_unit_label.text() == "MHz"
    assert restored_window.aoi_start_input.value() == pytest.approx(2400.0)
    assert restored_window.aoi_stop_input.value() == pytest.approx(2500.0)
    assert restored_window.marker_frequency_input.value() == pytest.approx(2450.0)
    assert restored_window.threshold_enabled_checkbox.isChecked()
    assert restored_window.threshold_input.value() == pytest.approx(6.0)

    restored_window.close()


def test_light_mode_toggle_applies_and_restores_palette(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "light_mode.s1p")
    window = TouchstoneViewerWindow([file_path])
    original_window_color = qapp.palette().color(QtGui.QPalette.ColorRole.Window).name()
    window.force_light_theme_checkbox.setChecked(True)

    assert qapp.palette().color(QtGui.QPalette.ColorRole.Window).name() == "#f8fafc"
    assert window.palette().color(QtGui.QPalette.ColorRole.Window).name() == "#f8fafc"

    window.force_light_theme_checkbox.setChecked(False)

    assert qapp.palette().color(QtGui.QPalette.ColorRole.Window).name() == original_window_color

    window.close()


def test_aoi_presets_can_be_saved_and_reused(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "presettable.s1p")
    config_path = Path(os.environ["XDG_CONFIG_HOME"]) / "touchstone_viewer" / "config.yaml"

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("GNSS L1", True),
    )

    window = TouchstoneViewerWindow([file_path])
    window.frequency_unit_combo.setCurrentText("MHz")
    window.aoi_start_input.setValue(2400.0)
    window.aoi_stop_input.setValue(2500.0)
    window.marker_frequency_input.setValue(2450.0)
    window.save_aoi_preset_button.click()

    assert window.aoi_preset_combo.currentText() == "GNSS L1"

    window.aoi_start_input.setValue(2600.0)
    window.aoi_stop_input.setValue(2700.0)
    window.marker_frequency_input.setValue(2650.0)
    window.aoi_preset_combo.setCurrentText("GNSS L1")

    assert window.aoi_start_input.value() == pytest.approx(2400.0)
    assert window.aoi_stop_input.value() == pytest.approx(2500.0)
    assert window.marker_frequency_input.value() == pytest.approx(2450.0)

    window.close()

    config_text = config_path.read_text(encoding="utf-8")
    assert '"GNSS L1":' in config_text
    assert "marker_frequency_hz: 2450000000.0" in config_text

    restored_window = TouchstoneViewerWindow([file_path])

    assert restored_window.frequency_unit_combo.currentText() == "MHz"
    assert "GNSS L1" in [
        restored_window.aoi_preset_combo.itemText(index)
        for index in range(restored_window.aoi_preset_combo.count())
    ]
    assert restored_window.aoi_preset_combo.currentText() == "Custom"
    assert restored_window.aoi_start_input.value() == pytest.approx(2400.0)
    assert restored_window.aoi_stop_input.value() == pytest.approx(2500.0)
    assert restored_window.marker_frequency_input.value() == pytest.approx(2450.0)

    restored_window.close()


def test_selected_preset_is_reflected_as_a_single_visible_band(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "selected_preset_band.s1p")
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("GPS L1", True),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)
    window.save_aoi_preset_button.click()

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }

    assert window.aoi_preset_combo.currentText() == "GPS L1"
    assert window.aoi_enabled_checkbox.isChecked()
    assert window.aoi_preset_bands_button.text() == "Bands (1)"
    assert band_actions["GPS L1"].isChecked()
    assert band_actions["GPS L1"].isEnabled()
    assert "GPS L1" in window.aoi_preset_region_items
    assert window.aoi_region_item is not None
    assert not window.aoi_region_item.isVisible()
    assert window.marker_table.horizontalHeaderItem(8).text() == "GPS L1 Area (|dB|*GHz)"

    window.close()


def test_preset_combo_switches_single_band_selection_and_custom_clears_it(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "preset_combo_bands.s1p")
    preset_names = iter([("GNSS L1", True), ("GNSS L5", True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(preset_names),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(2.2)
    window.aoi_stop_input.setValue(2.8)
    window.save_aoi_preset_button.click()

    def action_state() -> dict[str, bool]:
        return {
            action.text(): action.isChecked()
            for action in window.aoi_preset_bands_menu.actions()
            if action.isCheckable()
        }

    assert window.aoi_preset_combo.currentText() == "GNSS L5"
    assert window.aoi_enabled_checkbox.isChecked()
    assert action_state() == {"Custom": False, "GNSS L1": False, "GNSS L5": True}
    assert set(window.aoi_preset_region_items) == {"GNSS L5"}

    window.aoi_enabled_checkbox.setChecked(False)

    assert not window.aoi_enabled_checkbox.isChecked()
    assert set(window.aoi_preset_region_items) == {"GNSS L5"}

    window.aoi_preset_combo.setCurrentText("GNSS L1")

    assert window.aoi_enabled_checkbox.isChecked()
    assert action_state() == {"Custom": False, "GNSS L1": True, "GNSS L5": False}
    assert window.aoi_preset_bands_button.text() == "Bands (1)"

    window.aoi_preset_combo.setCurrentText("Custom")

    assert action_state() == {"Custom": True, "GNSS L1": False, "GNSS L5": False}
    assert window.aoi_preset_bands_button.text() == "Bands (1)"

    window.close()


def test_custom_band_is_included_with_visible_preset_bands(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file_with_content(
        tmp_path / "custom_and_preset_bands.s1p",
        "# GHz S DB R 50\n"
        "2.0 -10 0\n"
        "2.2 -15 0\n"
        "2.4 -20 0\n"
        "2.6 -25 0\n"
        "2.8 -30 0\n",
    )
    preset_names = iter([("GNSS L1", True), ("GNSS L5", True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(preset_names),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_enabled_checkbox.setChecked(True)
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.2)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(2.5)
    window.aoi_stop_input.setValue(2.6)
    window.save_aoi_preset_button.click()

    window.aoi_preset_combo.setCurrentText("Custom")
    window.aoi_start_input.setValue(2.3)
    window.aoi_stop_input.setValue(2.7)

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GNSS L1"].setChecked(True)
    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GNSS L5"].setChecked(True)
    qapp.processEvents()

    assert {
        action.text(): action.isChecked()
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    } == {"Custom": True, "GNSS L1": True, "GNSS L5": True}
    assert window.aoi_preset_bands_button.text() == "Bands (3)"
    assert [window.marker_table.horizontalHeaderItem(index).text() for index in range(11)] == [
        "Trace",
        "Freq",
        "S11 (dB)",
        "ΔRef (dB)",
        "|S11| (lin)",
        "ΔRef |S11| (lin)",
        "Angle (deg)",
        "Z (ohm)",
        "Custom Area (|dB|*GHz)",
        "GNSS L1 Area (|dB|*GHz)",
        "GNSS L5 Area (|dB|*GHz)",
    ]

    window.close()


def test_controls_panel_does_not_force_window_wider_for_multiple_aoi_bands(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file_with_content(
        tmp_path / "responsive_controls_panel.s1p",
        "# GHz S DB R 50\n"
        "2.0 -10 0\n"
        "2.2 -15 0\n"
        "2.4 -20 0\n"
        "2.6 -25 0\n"
        "2.8 -30 0\n",
    )
    preset_names = iter([("GNSS L1", True), ("GNSS L5", True), ("Galileo E1", True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(preset_names),
    )

    window = TouchstoneViewerWindow([file_path])
    window.controls_toggle_button.setChecked(True)
    window.resize(1200, 900)
    window.show()
    qapp.processEvents()

    assert window.width() == 1200

    window.aoi_enabled_checkbox.setChecked(True)
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.2)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(2.3)
    window.aoi_stop_input.setValue(2.4)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(2.5)
    window.aoi_stop_input.setValue(2.6)
    window.save_aoi_preset_button.click()

    window.aoi_preset_combo.setCurrentText("Custom")
    window.aoi_start_input.setValue(2.2)
    window.aoi_stop_input.setValue(2.7)

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GNSS L1"].setChecked(True)
    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GNSS L5"].setChecked(True)
    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["Galileo E1"].setChecked(True)
    qapp.processEvents()

    assert window.width() == 1200
    assert window.minimumSizeHint().width() <= 1200

    window.close()


def test_multiple_aoi_preset_bands_can_be_shown_at_once(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file_with_content(
        tmp_path / "preset_bands.s1p",
        "# GHz S MA R 50\n"
        "1.540 0.50 0\n"
        "1.580 0.35 0\n"
        "1.620 0.45 0\n",
    )
    preset_names = iter([("GPS L1", True), ("Galileo E1", True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(preset_names),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(1.574)
    window.aoi_stop_input.setValue(1.576)
    window.save_aoi_preset_button.click()
    window.aoi_start_input.setValue(1.598)
    window.aoi_stop_input.setValue(1.602)
    window.save_aoi_preset_button.click()

    window.aoi_enabled_checkbox.setChecked(False)

    assert not window.aoi_enabled_checkbox.isChecked()
    assert set(window.aoi_preset_region_items) == {"Galileo E1"}
    assert set(window.aoi_preset_label_items) == {"Galileo E1"}

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GPS L1"].setChecked(True)
    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["Galileo E1"].setChecked(True)
    qapp.processEvents()

    assert window.aoi_enabled_checkbox.isChecked()
    assert window.aoi_preset_bands_button.text() == "Bands (2)"
    assert set(window.aoi_preset_region_items) == {"GPS L1", "Galileo E1"}
    assert set(window.aoi_preset_label_items) == {"GPS L1", "Galileo E1"}
    assert window.aoi_preset_region_items["GPS L1"].getRegion() == pytest.approx([1.574, 1.576])
    assert window.aoi_preset_region_items["Galileo E1"].getRegion() == pytest.approx(
        [1.598, 1.602]
    )

    window.aoi_enabled_checkbox.setChecked(False)

    assert set(window.aoi_preset_region_items) == {"GPS L1", "Galileo E1"}
    assert set(window.aoi_preset_label_items) == {"GPS L1", "Galileo E1"}

    window.aoi_enabled_checkbox.setChecked(True)

    assert set(window.aoi_preset_region_items) == {"GPS L1", "Galileo E1"}

    window.frequency_unit_combo.setCurrentText("MHz")

    assert window.aoi_preset_region_items["GPS L1"].getRegion() == pytest.approx([1574.0, 1576.0])
    assert window.aoi_preset_region_items["Galileo E1"].getRegion() == pytest.approx(
        [1598.0, 1602.0]
    )

    window.close()


def test_disabling_custom_keeps_enabled_preset_band_visible(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "disable_custom_keeps_preset.s1p")
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("GNSS L1", True),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)
    window.save_aoi_preset_button.click()

    window.aoi_preset_combo.setCurrentText("Custom")
    window.aoi_start_input.setValue(2.2)
    window.aoi_stop_input.setValue(2.6)

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["GNSS L1"].setChecked(True)
    qapp.processEvents()

    assert set(window.aoi_preset_region_items) == {"GNSS L1"}
    assert window.aoi_region_item is not None
    assert window.aoi_region_item.isVisible()

    band_actions = {
        action.text(): action
        for action in window.aoi_preset_bands_menu.actions()
        if action.isCheckable()
    }
    band_actions["Custom"].setChecked(False)
    qapp.processEvents()

    assert not window.aoi_enabled_checkbox.isChecked()
    assert set(window.aoi_preset_region_items) == {"GNSS L1"}
    assert window.aoi_region_item is not None
    assert not window.aoi_region_item.isVisible()

    window.close()


def test_aoi_controls_update_region_and_clear_resets_state(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "third.s1p")

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)

    assert window.aoi_region_hz == (2.1e9, 2.7e9)

    window.clear_traces()

    assert len(window.traces) == 0
    assert not window.aoi_start_input.isEnabled()
    assert not window.aoi_stop_input.isEnabled()
    assert window.marker_table.rowCount() == 0

    window.close()


def test_clear_switches_to_custom_hides_aoi_and_show_restores_custom_region(
    monkeypatch,
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "clear_aoi.s1p")
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: ("GPS L1", True),
    )

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)
    window.save_aoi_preset_button.click()

    assert window.aoi_region_item is not None
    assert window.clear_aoi_button.isEnabled()
    assert "GPS L1" in window.aoi_preset_region_items

    original_region = window.aoi_region_item.getRegion()
    window.clear_aoi_button.click()

    assert window.aoi_preset_combo.currentText() == "Custom"
    assert not window.aoi_enabled_checkbox.isChecked()
    assert window.aoi_region_hz == pytest.approx((2.1e9, 2.7e9))
    assert window.aoi_region_item is not None
    assert not window.aoi_region_item.isVisible()
    assert window.clear_aoi_button.isEnabled()
    assert "GPS L1" not in window.aoi_preset_region_items
    assert window.aoi_preset_bands_button.text() == "Bands"
    assert window.marker_table.horizontalHeaderItem(8).text() == "AOI Area (|dB|*GHz)"

    window.aoi_enabled_checkbox.setChecked(True)

    assert window.aoi_region_item is not None
    assert window.aoi_region_item.isVisible()
    assert window.aoi_region_item.getRegion() == pytest.approx(original_region)

    window.close()


def test_aoi_inputs_collapse_to_the_edited_bound_when_crossed(
    qapp: QtWidgets.QApplication,
    isolated_qsettings: None,
    tmp_path: Path,
) -> None:
    file_path = _write_touchstone_file(tmp_path / "aoi_crossing.s1p")

    window = TouchstoneViewerWindow([file_path])
    window.aoi_start_input.setValue(2.1)
    window.aoi_stop_input.setValue(2.7)

    window.aoi_start_input.setValue(2.8)

    assert window.aoi_region_hz == pytest.approx((2.8e9, 2.8e9))
    assert window.aoi_start_input.value() == pytest.approx(2.8)
    assert window.aoi_stop_input.value() == pytest.approx(2.8)

    window.aoi_stop_input.setValue(2.2)

    assert window.aoi_region_hz == pytest.approx((2.2e9, 2.2e9))
    assert window.aoi_start_input.value() == pytest.approx(2.2)
    assert window.aoi_stop_input.value() == pytest.approx(2.2)

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
