from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .version import __version__
from .matching import (
    MatchingStage,
    MatchingSuggestion,
    apply_matching_network,
    component_units,
    impedance_to_gamma,
    suggest_matching_stages,
)
from .smith import add_smith_grid, reset_smith_view
from .touchstone import TouchstoneData, gamma_to_impedance, load_touchstone

TRACE_COLORS = [
    "#0f766e",
    "#c2410c",
    "#1d4ed8",
    "#b45309",
    "#7c3aed",
    "#15803d",
    "#be123c",
    "#334155",
]
AOI_PRESET_OVERLAY_COLORS = [
    "#f59e0b",
    "#10b981",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#ca8a04",
    "#4f46e5",
]

FREQUENCY_UNIT_FACTORS_HZ = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}

AOI_UNIT_FACTORS_HZ = {
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}

LAST_OPEN_DIRECTORY_KEY = "paths/last_open_directory"
DEFAULT_EMPTY_DB_RANGE = (-40.0, 0.0)
DEFAULT_THRESHOLD_DB = 10.0
DEFAULT_THRESHOLD_VISIBLE = False
DEFAULT_CONTROLS_VISIBLE = False
DEFAULT_AOI_VISIBLE = False
DEFAULT_MARKER_VISIBLE = True
DEFAULT_FORCE_LIGHT_MODE = False
DEFAULT_FREQUENCY_UNIT_MODE = "Auto"
DEFAULT_AOI_UNIT = "GHz"
CUSTOM_AOI_BAND_NAME = "Custom"
CONFIG_SAVE_DELAY_MS = 400
MATCHING_STAGE_TEMPLATES = [
    ("Series", "L", "nH"),
    ("Shunt", "C", "pF"),
    ("Series", "C", "pF"),
    ("Shunt", "L", "nH"),
]
MATCHING_DEFAULT_NETWORK = [
    ("Shunt", "C", "pF", 0.0, False),
    ("Series", "R", "ohm", 0.0, True),
    ("Shunt", "C", "pF", 0.0, False),
]
LIGHT_MODE_WINDOW_COLOR = "#f8fafc"
LIGHT_MODE_WINDOW_TEXT_COLOR = "#0f172a"
LIGHT_MODE_BASE_COLOR = "#ffffff"
LIGHT_MODE_ALTERNATE_BASE_COLOR = "#f1f5f9"
LIGHT_MODE_BUTTON_COLOR = "#ffffff"
LIGHT_MODE_HIGHLIGHT_COLOR = "#bfdbfe"
LIGHT_MODE_PLACEHOLDER_TEXT_COLOR = "#64748b"
_DEFAULT_APPLICATION_PALETTE: QtGui.QPalette | None = None
ACTIVE_AOI_COLOR = "#60a5fa"


@dataclass(frozen=True)
class FrequencyScale:
    factor_hz: float
    unit: str


@dataclass
class LoadedTrace:
    data: TouchstoneData
    color: str
    visible: bool = True
    s11_curve: pg.PlotDataItem | None = None
    smith_curve: pg.PlotDataItem | None = None
    s21_curve: pg.PlotDataItem | None = None
    s11_marker: pg.ScatterPlotItem | None = None
    smith_marker: pg.ScatterPlotItem | None = None
    s21_marker: pg.ScatterPlotItem | None = None


@dataclass
class MatchingStageControls:
    row_widget: QtWidgets.QWidget
    stage_label: QtWidgets.QLabel
    enabled_checkbox: QtWidgets.QCheckBox
    topology_combo: QtWidgets.QComboBox
    component_combo: QtWidgets.QComboBox
    value_input: QtWidgets.QDoubleSpinBox
    unit_combo: QtWidgets.QComboBox
    remove_button: QtWidgets.QPushButton


@dataclass(frozen=True)
class _DbPlotViewState:
    x_range_hz: tuple[float, float]
    y_range: tuple[float, float]


@dataclass(frozen=True)
class _PlotViewState:
    x_range: tuple[float, float]
    y_range: tuple[float, float]


@dataclass(frozen=True)
class _ViewerViewState:
    s11: _DbPlotViewState | None = None
    s21: _DbPlotViewState | None = None
    smith: _PlotViewState | None = None
    match_s11: _DbPlotViewState | None = None
    match_smith: _PlotViewState | None = None


@dataclass(frozen=True)
class AoiPreset:
    start_hz: float
    stop_hz: float
    unit: str = DEFAULT_AOI_UNIT
    marker_frequency_hz: float | None = None


@dataclass
class ViewerUserConfig:
    frequency_unit_mode: str = DEFAULT_FREQUENCY_UNIT_MODE
    force_light_mode: bool = DEFAULT_FORCE_LIGHT_MODE
    controls_visible: bool = DEFAULT_CONTROLS_VISIBLE
    aoi_visible: bool = DEFAULT_AOI_VISIBLE
    aoi_start_hz: float | None = None
    aoi_stop_hz: float | None = None
    aoi_unit: str = DEFAULT_AOI_UNIT
    marker_visible: bool = DEFAULT_MARKER_VISIBLE
    marker_frequency_hz: float | None = None
    threshold_visible: bool = DEFAULT_THRESHOLD_VISIBLE
    threshold_db: float = DEFAULT_THRESHOLD_DB
    aoi_presets: dict[str, AoiPreset] = field(default_factory=dict)
    selected_aoi_preset: str | None = None


class _SortableTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        left_value = self.data(QtCore.Qt.ItemDataRole.UserRole)
        right_value = other.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            return float(left_value) < float(right_value)
        return self.text() < other.text()


class _SelectAllDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        QtCore.QTimer.singleShot(0, self._select_editor_text)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        should_select = (
            not self.hasFocus() and event.button() == QtCore.Qt.MouseButton.LeftButton
        )
        super().mousePressEvent(event)
        if should_select:
            QtCore.QTimer.singleShot(0, self._select_editor_text)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            QtCore.QTimer.singleShot(0, self._select_editor_text)

    def _select_editor_text(self) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.selectAll()


class TouchstoneViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_paths: Sequence[Path]) -> None:
        super().__init__()
        self.setWindowTitle(f"Touchstone Viewer - v{__version__}")
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings("TouchstoneViewer", "Touch")
        self.user_config = _load_viewer_user_config()
        # Always start with the editable AOI hidden; saved bounds still restore.
        self.user_config.aoi_visible = False
        self.force_light_mode = self.user_config.force_light_mode
        _apply_application_appearance(self.force_light_mode)

        self.traces: list[LoadedTrace] = []
        self.frequency_scale = FrequencyScale(1.0e6, "MHz")
        self.frequency_unit_mode = self.user_config.frequency_unit_mode
        self.reference_trace_path: Path | None = None
        self.marker_frequency_hz: float | None = self.user_config.marker_frequency_hz
        self.marker_line: pg.InfiniteLine | None = None
        self.s21_marker_line: pg.InfiniteLine | None = None
        self.marker_plot_label: pg.TextItem | None = None
        self.s21_marker_plot_label: pg.TextItem | None = None
        self.aoi_region_hz: tuple[float, float] | None = (
            _sorted_frequency_region_hz(
                (self.user_config.aoi_start_hz, self.user_config.aoi_stop_hz)
            )
            if self.user_config.aoi_start_hz is not None and self.user_config.aoi_stop_hz is not None
            else None
        )
        self.aoi_region_item: pg.LinearRegionItem | None = None
        self.aoi_preset_region_items: dict[str, pg.LinearRegionItem] = {}
        self.aoi_preset_label_items: dict[str, pg.TextItem] = {}
        self.s11_threshold_line: pg.InfiniteLine | None = None
        self.s21_threshold_line: pg.InfiniteLine | None = None
        self.match_trace_path: Path | None = None
        self.matching_stage_controls: list[MatchingStageControls] = []
        self.matching_suggestions: list[MatchingSuggestion] = []
        self.match_marker_frequency_hz: float | None = None
        self.match_original_gamma: np.ndarray | None = None
        self.match_transformed_gamma: np.ndarray | None = None
        self.match_original_s11_db: np.ndarray | None = None
        self.match_transformed_s11_db: np.ndarray | None = None
        self.match_frequencies_hz: np.ndarray | None = None
        self.match_s11_original_marker: pg.ScatterPlotItem | None = None
        self.match_s11_transformed_marker: pg.ScatterPlotItem | None = None
        self.match_smith_original_marker: pg.ScatterPlotItem | None = None
        self.match_smith_transformed_marker: pg.ScatterPlotItem | None = None
        self.match_s11_marker_line: pg.InfiniteLine | None = None
        self.match_marker_plot_label: pg.TextItem | None = None
        self._updating_marker = False
        self._updating_marker_controls = False
        self._updating_aoi_controls = False
        self._updating_aoi_preset_controls = False
        self._updating_aoi_preset_band_controls = False
        self._updating_trace_controls = False
        self._updating_matching_controls = False
        self._updating_match_target_controls = False
        self._aoi_presets = dict(self.user_config.aoi_presets)
        self._visible_aoi_preset_names: set[str] = set()
        self._selected_aoi_preset_name: str | None = None
        self._config_save_timer = QtCore.QTimer(self)
        self._config_save_timer.setSingleShot(True)
        self._config_save_timer.setInterval(CONFIG_SAVE_DELAY_MS)
        self._config_save_timer.timeout.connect(self._write_user_config)

        self._build_ui()
        self._restore_default_matching_network()
        self._refresh_plots()

        if initial_paths:
            self.load_files(initial_paths)

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)

        open_button = QtWidgets.QPushButton("Open Files")
        open_button.clicked.connect(self._open_files_dialog)
        header.addWidget(open_button)

        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_traces)
        header.addWidget(clear_button)

        self.reset_view_button = QtWidgets.QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        header.addWidget(self.reset_view_button)

        header.addSpacing(12)

        self.summary_label = QtWidgets.QLabel(
            "Drop .s1p or .s2p files here or open them from the dialog."
        )
        header.addWidget(self.summary_label, stretch=1)

        self.force_light_theme_checkbox = QtWidgets.QCheckBox("Force Light Theme")
        self.force_light_theme_checkbox.setChecked(self.force_light_mode)
        self.force_light_theme_checkbox.toggled.connect(self._handle_light_mode_toggled)
        header.addWidget(self.force_light_theme_checkbox)

        self.controls_toggle_button = QtWidgets.QToolButton()
        self.controls_toggle_button.setText("Controls")
        self.controls_toggle_button.setCheckable(True)
        self.controls_toggle_button.setChecked(self.user_config.controls_visible)
        self.controls_toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.controls_toggle_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.controls_toggle_button.toggled.connect(self._set_controls_panel_visible)
        header.addWidget(self.controls_toggle_button)

        layout.addLayout(header)

        self.controls_panel = self._build_controls_panel()
        layout.addWidget(self.controls_panel)
        self._set_controls_panel_visible(self.user_config.controls_visible)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.addTab(self._build_s11_tab(), "S11")
        self.tab_widget.addTab(self._build_s21_tab(), "S21")
        self.tab_widget.addTab(self._build_match_tab(), "Match")
        layout.addWidget(self.tab_widget, stretch=1)

        self.setCentralWidget(central_widget)

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        primary_controls_row = QtWidgets.QHBoxLayout()
        primary_controls_row.setSpacing(10)

        view_section = QtWidgets.QWidget()
        view_layout = QtWidgets.QHBoxLayout(view_section)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(8)

        view_prefix = QtWidgets.QLabel("View")
        view_layout.addWidget(view_prefix)

        frequency_unit_prefix = QtWidgets.QLabel("Freq")
        view_layout.addWidget(frequency_unit_prefix)

        self.frequency_unit_combo = QtWidgets.QComboBox()
        self.frequency_unit_combo.addItems(
            [DEFAULT_FREQUENCY_UNIT_MODE, *FREQUENCY_UNIT_FACTORS_HZ.keys()]
        )
        self.frequency_unit_combo.setCurrentText(self.frequency_unit_mode)
        self.frequency_unit_combo.currentTextChanged.connect(self._handle_frequency_unit_changed)
        view_layout.addWidget(self.frequency_unit_combo)

        aoi_section = QtWidgets.QWidget()
        aoi_layout = QtWidgets.QHBoxLayout(aoi_section)
        aoi_layout.setContentsMargins(0, 0, 0, 0)
        aoi_layout.setSpacing(8)

        aoi_prefix = QtWidgets.QLabel("Area of Interest")
        aoi_layout.addWidget(aoi_prefix)
        self.aoi_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.aoi_enabled_checkbox.setChecked(self.user_config.aoi_visible)
        self.aoi_enabled_checkbox.toggled.connect(self._handle_aoi_visibility_changed)
        aoi_layout.addWidget(self.aoi_enabled_checkbox)

        self.aoi_start_input = self._build_aoi_spin_box()
        self.aoi_start_input.valueChanged.connect(self._handle_aoi_value_changed)
        aoi_layout.addWidget(self.aoi_start_input)

        aoi_to_label = QtWidgets.QLabel("to")
        aoi_layout.addWidget(aoi_to_label)

        self.aoi_stop_input = self._build_aoi_spin_box()
        self.aoi_stop_input.valueChanged.connect(self._handle_aoi_value_changed)
        aoi_layout.addWidget(self.aoi_stop_input)

        self.aoi_display_unit_label = QtWidgets.QLabel(self.frequency_scale.unit)
        aoi_layout.addWidget(self.aoi_display_unit_label)

        preset_prefix = QtWidgets.QLabel("Preset")
        aoi_layout.addWidget(preset_prefix)

        self.aoi_preset_combo = QtWidgets.QComboBox()
        self.aoi_preset_combo.setMinimumContentsLength(10)
        self.aoi_preset_combo.currentIndexChanged.connect(self._handle_aoi_preset_changed)
        aoi_layout.addWidget(self.aoi_preset_combo)

        self.save_aoi_preset_button = QtWidgets.QPushButton("Save")
        self.save_aoi_preset_button.clicked.connect(self._save_current_aoi_as_preset)
        aoi_layout.addWidget(self.save_aoi_preset_button)

        self.clear_aoi_button = QtWidgets.QPushButton("Clear")
        self.clear_aoi_button.clicked.connect(self._clear_active_aoi)
        aoi_layout.addWidget(self.clear_aoi_button)

        self.delete_aoi_preset_button = QtWidgets.QPushButton("Delete")
        self.delete_aoi_preset_button.clicked.connect(self._delete_selected_aoi_preset)
        aoi_layout.addWidget(self.delete_aoi_preset_button)

        self.aoi_preset_bands_button = QtWidgets.QToolButton()
        self.aoi_preset_bands_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.aoi_preset_bands_menu = QtWidgets.QMenu(self.aoi_preset_bands_button)
        self.aoi_preset_bands_button.setMenu(self.aoi_preset_bands_menu)
        aoi_layout.addWidget(self.aoi_preset_bands_button)

        primary_controls_row.addWidget(view_section)
        primary_controls_row.addWidget(self._build_panel_separator())
        primary_controls_row.addWidget(aoi_section)
        primary_controls_row.addStretch(1)

        threshold_section = QtWidgets.QWidget()
        threshold_layout = QtWidgets.QHBoxLayout(threshold_section)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)

        threshold_prefix = QtWidgets.QLabel("Threshold")
        threshold_layout.addWidget(threshold_prefix)

        self.threshold_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.threshold_enabled_checkbox.setChecked(self.user_config.threshold_visible)
        self.threshold_enabled_checkbox.toggled.connect(self._handle_threshold_visibility_changed)
        threshold_layout.addWidget(self.threshold_enabled_checkbox)

        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setDecimals(2)
        self.threshold_input.setRange(0.0, 200.0)
        self.threshold_input.setSingleStep(0.5)
        self.threshold_input.setValue(self.user_config.threshold_db)
        self.threshold_input.setSuffix(" dB")
        self.threshold_input.setMinimumWidth(95)
        self.threshold_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.threshold_input.valueChanged.connect(self._handle_threshold_changed)
        self.threshold_input.setEnabled(self.user_config.threshold_visible)
        threshold_layout.addWidget(self.threshold_input)

        secondary_controls_row = QtWidgets.QHBoxLayout()
        secondary_controls_row.setSpacing(10)
        secondary_controls_row.addWidget(threshold_section)
        secondary_controls_row.addWidget(self._build_panel_separator())

        marker_section = QtWidgets.QWidget()
        marker_layout = QtWidgets.QHBoxLayout(marker_section)
        marker_layout.setContentsMargins(0, 0, 0, 0)
        marker_layout.setSpacing(8)

        marker_prefix = QtWidgets.QLabel("Marker")
        marker_layout.addWidget(marker_prefix)

        self.marker_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.marker_enabled_checkbox.setChecked(self.user_config.marker_visible)
        self.marker_enabled_checkbox.toggled.connect(self._handle_marker_visibility_changed)
        marker_layout.addWidget(self.marker_enabled_checkbox)

        marker_at_label = QtWidgets.QLabel("At")
        marker_layout.addWidget(marker_at_label)

        self.marker_frequency_input = _SelectAllDoubleSpinBox()
        self.marker_frequency_input.setDecimals(6)
        self.marker_frequency_input.setMinimumWidth(120)
        self.marker_frequency_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.marker_frequency_input.setKeyboardTracking(False)
        self.marker_frequency_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.marker_frequency_input.valueChanged.connect(self._handle_marker_frequency_changed)
        marker_layout.addWidget(self.marker_frequency_input)

        self.marker_frequency_unit_label = QtWidgets.QLabel(self.frequency_scale.unit)
        marker_layout.addWidget(self.marker_frequency_unit_label)

        secondary_controls_row.addWidget(marker_section)
        secondary_controls_row.addStretch(1)
        layout.addLayout(primary_controls_row)
        layout.addLayout(secondary_controls_row)

        traces_section = QtWidgets.QWidget()
        traces_layout = QtWidgets.QVBoxLayout(traces_section)
        traces_layout.setContentsMargins(0, 0, 0, 0)
        traces_layout.setSpacing(8)

        traces_header = QtWidgets.QHBoxLayout()
        traces_header.setSpacing(8)

        traces_prefix = QtWidgets.QLabel("Traces")
        traces_header.addWidget(traces_prefix)

        reference_prefix = QtWidgets.QLabel("Compare To")
        traces_header.addWidget(reference_prefix)

        self.reference_trace_combo = QtWidgets.QComboBox()
        self.reference_trace_combo.setMinimumWidth(300)
        self.reference_trace_combo.setMinimumContentsLength(28)
        self.reference_trace_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.reference_trace_combo.setToolTip(
            "Select a loaded trace as the comparison baseline for delta values and highlighting."
        )
        self.reference_trace_combo.currentIndexChanged.connect(
            self._handle_reference_trace_changed
        )
        traces_header.addWidget(self.reference_trace_combo, 1)

        self.show_all_traces_button = QtWidgets.QPushButton("Show All")
        self.show_all_traces_button.clicked.connect(lambda: self._set_all_traces_visible(True))
        traces_header.addWidget(self.show_all_traces_button)

        self.hide_all_traces_button = QtWidgets.QPushButton("Hide All")
        self.hide_all_traces_button.clicked.connect(lambda: self._set_all_traces_visible(False))
        traces_header.addWidget(self.hide_all_traces_button)
        traces_header.addStretch(1)

        traces_layout.addLayout(traces_header)

        self.trace_visibility_list = QtWidgets.QListWidget()
        self.trace_visibility_list.setAlternatingRowColors(True)
        self.trace_visibility_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.trace_visibility_list.setFixedHeight(92)
        self.trace_visibility_list.itemChanged.connect(self._handle_trace_visibility_changed)
        traces_layout.addWidget(self.trace_visibility_list)

        layout.addWidget(traces_section)

        return panel

    def _build_panel_separator(self) -> QtWidgets.QFrame:
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #cbd5e1;")
        return separator

    def _visible_traces(self) -> list[LoadedTrace]:
        return [trace for trace in self.traces if trace.visible]

    def _display_name_for_trace(self, trace: LoadedTrace) -> str:
        if self.reference_trace_path == trace.data.path:
            return f"{trace.data.label} (ref)"
        return trace.data.label

    def _reference_trace(self) -> LoadedTrace | None:
        for trace in self.traces:
            if trace.data.path == self.reference_trace_path:
                return trace
        return None

    def _sync_trace_controls(self) -> None:
        if self._updating_trace_controls:
            return

        self._updating_trace_controls = True
        try:
            valid_reference_paths = {trace.data.path for trace in self.traces}
            if self.reference_trace_path not in valid_reference_paths:
                self.reference_trace_path = None

            self.reference_trace_combo.blockSignals(True)
            self.reference_trace_combo.clear()
            self.reference_trace_combo.addItem("None", None)
            for trace in self.traces:
                self.reference_trace_combo.addItem(trace.data.label, str(trace.data.path))

            if self.reference_trace_path is None:
                self.reference_trace_combo.setCurrentIndex(0)
            else:
                index = self.reference_trace_combo.findData(str(self.reference_trace_path))
                self.reference_trace_combo.setCurrentIndex(index if index >= 0 else 0)
            self.reference_trace_combo.blockSignals(False)

            self.trace_visibility_list.blockSignals(True)
            self.trace_visibility_list.clear()
            for trace in self.traces:
                item = QtWidgets.QListWidgetItem(self._display_name_for_trace(trace))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(trace.data.path))
                item.setToolTip(str(trace.data.path))
                item.setFlags(
                    item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked if trace.visible else QtCore.Qt.CheckState.Unchecked
                )
                item.setForeground(QtGui.QBrush(QtGui.QColor(trace.color)))
                if self.reference_trace_path == trace.data.path:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.trace_visibility_list.addItem(item)
            self.trace_visibility_list.blockSignals(False)

            has_traces = bool(self.traces)
            self.reference_trace_combo.setEnabled(has_traces)
            self.show_all_traces_button.setEnabled(has_traces)
            self.hide_all_traces_button.setEnabled(has_traces)
            self.trace_visibility_list.setEnabled(has_traces)
        finally:
            self._updating_trace_controls = False

    def _matching_trace(self) -> LoadedTrace | None:
        for trace in self.traces:
            if trace.data.path == self.match_trace_path:
                return trace
        return self.traces[0] if self.traces else None

    def _matching_stage_template(self, index: int) -> tuple[str, str, str]:
        return MATCHING_STAGE_TEMPLATES[index % len(MATCHING_STAGE_TEMPLATES)]

    def _sync_matching_controls(self) -> None:
        if self._updating_matching_controls:
            return

        self._updating_matching_controls = True
        try:
            valid_paths = {trace.data.path for trace in self.traces}
            if self.match_trace_path not in valid_paths:
                self.match_trace_path = self.traces[0].data.path if self.traces else None

            self.match_trace_combo.blockSignals(True)
            self.match_trace_combo.clear()
            for trace in self.traces:
                self.match_trace_combo.addItem(trace.data.label, str(trace.data.path))
            if self.match_trace_path is not None:
                index = self.match_trace_combo.findData(str(self.match_trace_path))
                self.match_trace_combo.setCurrentIndex(index if index >= 0 else 0)
            self.match_trace_combo.blockSignals(False)

            has_traces = bool(self.traces)
            self.match_trace_combo.setEnabled(has_traces)
            self.match_target_frequency_input.setEnabled(has_traces)
            self.add_matching_stage_button.setEnabled(has_traces)
            self.reset_matching_button.setEnabled(has_traces)
            self.apply_matching_suggestion_button.setEnabled(
                has_traces and bool(self.match_suggestion_table.selectedItems())
            )
            self.match_suggestion_table.setEnabled(has_traces)
            for stage_controls in self.matching_stage_controls:
                stage_controls.enabled_checkbox.setEnabled(has_traces)
                stage_controls.topology_combo.setEnabled(has_traces)
                stage_controls.component_combo.setEnabled(has_traces)
                stage_controls.value_input.setEnabled(has_traces)
                stage_controls.unit_combo.setEnabled(has_traces)
                stage_controls.remove_button.setEnabled(has_traces)
            self._sync_match_target_frequency_input()
            self._update_matching_stage_row_labels()
            self.match_empty_stages_label.setVisible(not self.matching_stage_controls)
        finally:
            self._updating_matching_controls = False

    def _handle_match_trace_changed(self, index: int) -> None:
        if self._updating_matching_controls:
            return
        path_value = self.match_trace_combo.itemData(index)
        self.match_trace_path = Path(path_value) if path_value else None
        self._refresh_matching_tab()

    def _handle_match_target_frequency_changed(self, value: float) -> None:
        if self._updating_match_target_controls:
            return

        self.marker_frequency_hz = value * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_matching_component_changed(self, stage_controls: MatchingStageControls) -> None:
        self._sync_matching_stage_units(stage_controls, stage_controls.component_combo.currentText())
        self._refresh_matching_tab()

    def _sync_matching_stage_units(
        self,
        stage_controls: MatchingStageControls,
        component: str,
        *,
        preferred_unit: str | None = None,
    ) -> None:
        units = component_units(component)
        if preferred_unit not in units:
            preferred_unit = (
                stage_controls.unit_combo.currentText()
                if stage_controls.unit_combo.currentText() in units
                else units[0]
            )
        stage_controls.unit_combo.blockSignals(True)
        stage_controls.unit_combo.clear()
        stage_controls.unit_combo.addItems(units)
        stage_controls.unit_combo.setCurrentText(preferred_unit)
        stage_controls.unit_combo.blockSignals(False)

    def _matching_target_frequency_hz(self) -> float | None:
        trace = self._matching_trace()
        if trace is None:
            return None

        minimum_hz = float(trace.data.frequencies_hz[0])
        maximum_hz = float(trace.data.frequencies_hz[-1])
        if self.marker_frequency_hz is None:
            return self._default_matching_marker_frequency(trace.data.frequencies_hz)
        return min(max(self.marker_frequency_hz, minimum_hz), maximum_hz)

    def _sync_match_target_frequency_input(self) -> None:
        trace = self._matching_trace()
        self._updating_match_target_controls = True
        try:
            self.match_target_frequency_input.setEnabled(trace is not None)
            self.match_target_frequency_input.setSuffix(f" {self.frequency_scale.unit}")
            if trace is None:
                self.match_target_frequency_input.setRange(0.0, 0.0)
                self.match_target_frequency_input.setValue(0.0)
                return

            minimum_hz = float(trace.data.frequencies_hz[0])
            maximum_hz = float(trace.data.frequencies_hz[-1])
            display_bounds = (
                minimum_hz / self.frequency_scale.factor_hz,
                maximum_hz / self.frequency_scale.factor_hz,
            )
            step_size = max((display_bounds[1] - display_bounds[0]) / 200.0, 1.0e-6)
            self.match_target_frequency_input.setRange(display_bounds[0], display_bounds[1])
            self.match_target_frequency_input.setSingleStep(step_size)
            target_frequency_hz = self._matching_target_frequency_hz()
            if target_frequency_hz is None:
                self.match_target_frequency_input.setValue(0.0)
            else:
                self.match_target_frequency_input.setValue(
                    target_frequency_hz / self.frequency_scale.factor_hz
                )
        finally:
            self._updating_match_target_controls = False

    def _handle_matching_network_changed(self, *_args: object) -> None:
        if self._updating_matching_controls:
            return
        self._refresh_matching_tab()

    def _handle_matching_suggestion_selection_changed(self) -> None:
        self.apply_matching_suggestion_button.setEnabled(
            bool(self.traces) and bool(self.match_suggestion_table.selectedItems())
        )

    def _append_selected_matching_suggestion(self) -> None:
        selection_model = self.match_suggestion_table.selectionModel()
        if selection_model is None or not selection_model.hasSelection():
            return

        row = selection_model.selectedRows()[0].row()
        if row < 0 or row >= len(self.matching_suggestions):
            return

        self._append_matching_stage(stage=self.matching_suggestions[row].stage)

    def _handle_add_matching_stage_clicked(self) -> None:
        self._append_matching_stage()

    def _append_matching_stage(self, *, stage: MatchingStage | None = None) -> MatchingStageControls:
        if stage is None:
            default_topology, default_component, default_unit = self._matching_stage_template(
                len(self.matching_stage_controls)
            )
            stage = MatchingStage(
                topology=default_topology,
                component=default_component,
                value=0.0,
                unit=default_unit,
                enabled=False,
            )

        stage_controls = self._build_matching_stage_controls(
            stage.topology,
            stage.component,
            stage.unit,
            value=stage.value,
            enabled=stage.enabled,
        )
        self.matching_stage_controls.append(stage_controls)
        self.match_stage_rows_layout.addWidget(stage_controls.row_widget)
        self._sync_matching_controls()
        self._refresh_matching_tab()
        return stage_controls

    def _remove_matching_stage(self, stage_controls: MatchingStageControls) -> None:
        if stage_controls not in self.matching_stage_controls:
            return

        self.matching_stage_controls.remove(stage_controls)
        stage_controls.row_widget.setParent(None)
        stage_controls.row_widget.deleteLater()
        self._sync_matching_controls()
        self._refresh_matching_tab()

    def _update_matching_stage_row_labels(self) -> None:
        for index, stage_controls in enumerate(self.matching_stage_controls, start=1):
            stage_controls.stage_label.setText(str(index))

    def _matching_stages(self) -> list[MatchingStage]:
        stages: list[MatchingStage] = []
        for stage_controls in self.matching_stage_controls:
            stages.append(
                MatchingStage(
                    topology=stage_controls.topology_combo.currentText(),
                    component=stage_controls.component_combo.currentText(),
                    value=float(stage_controls.value_input.value()),
                    unit=stage_controls.unit_combo.currentText(),
                    enabled=stage_controls.enabled_checkbox.isChecked(),
                )
            )
        return stages

    def _reset_matching_network(self) -> None:
        self._restore_default_matching_network()
        self.matching_suggestions = []
        self.match_suggestion_table.setRowCount(0)
        self.match_suggestion_label.setText(
            "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
        )
        self._sync_matching_controls()
        self._refresh_matching_tab()

    def _restore_default_matching_network(self) -> None:
        while self.matching_stage_controls:
            stage_controls = self.matching_stage_controls.pop()
            stage_controls.row_widget.setParent(None)
            stage_controls.row_widget.deleteLater()
        for topology, component, unit, value, enabled in MATCHING_DEFAULT_NETWORK:
            self._append_matching_stage(
                stage=MatchingStage(
                    topology=topology,
                    component=component,
                    value=value,
                    unit=unit,
                    enabled=enabled,
                )
            )

    def _build_s11_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.s11_plot = pg.PlotWidget()
        self.s11_plot.scene().sigMouseClicked.connect(
            lambda event: self._handle_plot_click(self.s11_plot, event)
        )
        self.s11_plot.getPlotItem().vb.sigRangeChanged.connect(
            self._handle_s11_plot_range_changed
        )
        splitter.addWidget(self.s11_plot)

        self.smith_plot = pg.PlotWidget()
        splitter.addWidget(self.smith_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

        self.marker_table = self._build_marker_table(self._s11_marker_table_headers())
        layout.addLayout(
            self._build_table_actions_row(
                "Marker Table",
                lambda: self._export_table_to_csv(
                    self.marker_table,
                    suggested_name="s11_marker_table.csv",
                ),
            )
        )
        layout.addWidget(self.marker_table)

        return tab

    def _build_s21_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.s21_plot = pg.PlotWidget()
        self.s21_plot.scene().sigMouseClicked.connect(
            lambda event: self._handle_plot_click(self.s21_plot, event)
        )
        self.s21_plot.getPlotItem().vb.sigRangeChanged.connect(
            self._update_s21_marker_plot_label_position
        )
        layout.addWidget(self.s21_plot, stretch=1)

        self.s21_marker_table = self._build_marker_table(
            [
                "Trace",
                "Freq",
                "S21 (dB)",
                "ΔRef (dB)",
                "|S21| (lin)",
                "ΔRef |S21| (lin)",
                "Angle (deg)",
            ]
        )
        layout.addLayout(
            self._build_table_actions_row(
                "Marker Table",
                lambda: self._export_table_to_csv(
                    self.s21_marker_table,
                    suggested_name="s21_marker_table.csv",
                ),
            )
        )
        layout.addWidget(self.s21_marker_table)

        return tab

    def _build_match_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        controls_frame = QtWidgets.QFrame()
        controls_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        controls_layout = QtWidgets.QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)

        trace_row = QtWidgets.QHBoxLayout()
        trace_row.setSpacing(8)
        trace_row.addWidget(QtWidgets.QLabel("Trace"))

        self.match_trace_combo = QtWidgets.QComboBox()
        self.match_trace_combo.setMinimumWidth(280)
        self.match_trace_combo.currentIndexChanged.connect(self._handle_match_trace_changed)
        trace_row.addWidget(self.match_trace_combo)

        trace_row.addWidget(QtWidgets.QLabel("Target"))

        self.match_target_frequency_input = QtWidgets.QDoubleSpinBox()
        self.match_target_frequency_input.setDecimals(6)
        self.match_target_frequency_input.setMinimumWidth(150)
        self.match_target_frequency_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.match_target_frequency_input.setKeyboardTracking(False)
        self.match_target_frequency_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.match_target_frequency_input.valueChanged.connect(
            self._handle_match_target_frequency_changed
        )
        trace_row.addWidget(self.match_target_frequency_input)

        self.add_matching_stage_button = QtWidgets.QPushButton("Add Stage")
        self.add_matching_stage_button.clicked.connect(self._handle_add_matching_stage_clicked)
        trace_row.addWidget(self.add_matching_stage_button)

        self.reset_matching_button = QtWidgets.QPushButton("Reset Network")
        self.reset_matching_button.clicked.connect(self._reset_matching_network)
        trace_row.addWidget(self.reset_matching_button)
        trace_row.addStretch(1)

        self.match_order_label = QtWidgets.QLabel(
            "Order is top to bottom: antenna/load -> coax/feed. The first row is closest to the antenna. Added stages and suggestions append at the bottom, closest to the feed."
        )
        self.match_order_label.setStyleSheet("color: #475569;")
        controls_layout.addLayout(trace_row)
        controls_layout.addWidget(self.match_order_label)

        headers_row = QtWidgets.QHBoxLayout()
        headers_row.setSpacing(10)
        headers = ["Stage", "Use", "Topology", "Part", "Value", "Unit", ""]
        stretches = [0, 0, 0, 0, 0, 0, 1]
        for header, stretch in zip(headers, stretches, strict=False):
            label = QtWidgets.QLabel(header)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            headers_row.addWidget(label, stretch=stretch)
        controls_layout.addLayout(headers_row)

        stages_scroll = QtWidgets.QScrollArea()
        stages_scroll.setWidgetResizable(True)
        stages_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        stages_scroll.setMaximumHeight(190)
        self.match_stage_rows_container = QtWidgets.QWidget()
        self.match_stage_rows_layout = QtWidgets.QVBoxLayout(self.match_stage_rows_container)
        self.match_stage_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.match_stage_rows_layout.setSpacing(6)

        self.match_empty_stages_label = QtWidgets.QLabel(
            "No stages added yet. Use Add Stage or append one from the suggestions below."
        )
        self.match_empty_stages_label.setStyleSheet("color: #64748b;")
        self.match_stage_rows_layout.addWidget(self.match_empty_stages_label)
        stages_scroll.setWidget(self.match_stage_rows_container)
        controls_layout.addWidget(stages_scroll)

        suggestions_header = QtWidgets.QHBoxLayout()
        suggestions_header.setSpacing(8)
        suggestions_prefix = QtWidgets.QLabel("Reactive Suggestions")
        suggestions_header.addWidget(suggestions_prefix)
        suggestions_header.addStretch(1)
        self.apply_matching_suggestion_button = QtWidgets.QPushButton("Append Suggestion")
        self.apply_matching_suggestion_button.clicked.connect(
            self._append_selected_matching_suggestion
        )
        suggestions_header.addWidget(self.apply_matching_suggestion_button)
        controls_layout.addLayout(suggestions_header)

        self.match_suggestion_table = QtWidgets.QTableWidget(0, 5)
        self.match_suggestion_table.setHorizontalHeaderLabels(
            [
                "Next Stage",
                "Value",
                "Result S11 (dB)",
                "Improve (dB)",
                "Result Z (ohm)",
            ]
        )
        self.match_suggestion_table.verticalHeader().setVisible(False)
        self.match_suggestion_table.setAlternatingRowColors(True)
        self.match_suggestion_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.match_suggestion_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.match_suggestion_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.match_suggestion_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.match_suggestion_table.setMinimumHeight(180)
        suggestion_header = self.match_suggestion_table.horizontalHeader()
        suggestion_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        suggestion_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        suggestion_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        suggestion_header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        suggestion_header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.match_suggestion_table.itemSelectionChanged.connect(
            self._handle_matching_suggestion_selection_changed
        )
        self.match_suggestion_table.itemDoubleClicked.connect(
            lambda _item: self._append_selected_matching_suggestion()
        )
        controls_layout.addWidget(self.match_suggestion_table)

        self.match_suggestion_label = QtWidgets.QLabel(
            "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
        )
        self.match_suggestion_label.setStyleSheet("color: #64748b;")
        controls_layout.addWidget(self.match_suggestion_label)
        layout.addWidget(controls_frame)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.match_s11_plot = pg.PlotWidget()
        self.match_s11_plot.scene().sigMouseClicked.connect(
            lambda event: self._handle_plot_click(self.match_s11_plot, event)
        )
        self.match_s11_plot.getPlotItem().vb.sigRangeChanged.connect(
            self._update_match_marker_plot_label_position
        )
        splitter.addWidget(self.match_s11_plot)

        self.match_smith_plot = pg.PlotWidget()
        splitter.addWidget(self.match_smith_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        self.match_summary_label = QtWidgets.QLabel("Load a trace to preview a matching network.")
        self.match_summary_label.setStyleSheet("color: #334155;")
        layout.addWidget(self.match_summary_label)

        return tab

    def _build_matching_stage_controls(
        self,
        default_topology: str,
        default_component: str,
        default_unit: str,
        *,
        value: float = 0.0,
        enabled: bool = False,
    ) -> MatchingStageControls:
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        stage_label = QtWidgets.QLabel("?")
        stage_label.setMinimumWidth(38)
        stage_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        row_layout.addWidget(stage_label)

        enabled_checkbox = QtWidgets.QCheckBox()
        row_layout.addWidget(enabled_checkbox)

        topology_combo = QtWidgets.QComboBox()
        topology_combo.addItems(["Series", "Shunt"])
        topology_combo.setCurrentText(default_topology)
        row_layout.addWidget(topology_combo)

        component_combo = QtWidgets.QComboBox()
        component_combo.addItems(["R", "L", "C"])
        component_combo.setCurrentText(default_component)
        row_layout.addWidget(component_combo)

        value_input = QtWidgets.QDoubleSpinBox()
        value_input.setDecimals(6)
        value_input.setRange(0.0, 1.0e9)
        value_input.setSingleStep(0.1)
        value_input.setMinimumWidth(110)
        value_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        value_input.setKeyboardTracking(False)
        value_input.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        value_input.setValue(value)
        row_layout.addWidget(value_input)

        unit_combo = QtWidgets.QComboBox()
        row_layout.addWidget(unit_combo)

        remove_button = QtWidgets.QPushButton("Remove")
        remove_button.setAutoDefault(False)
        row_layout.addWidget(remove_button)

        controls = MatchingStageControls(
            row_widget=row_widget,
            stage_label=stage_label,
            enabled_checkbox=enabled_checkbox,
            topology_combo=topology_combo,
            component_combo=component_combo,
            value_input=value_input,
            unit_combo=unit_combo,
            remove_button=remove_button,
        )
        self._sync_matching_stage_units(controls, default_component, preferred_unit=default_unit)
        enabled_checkbox.setChecked(enabled)

        enabled_checkbox.toggled.connect(self._handle_matching_network_changed)
        topology_combo.currentTextChanged.connect(self._handle_matching_network_changed)
        value_input.valueChanged.connect(self._handle_matching_network_changed)
        unit_combo.currentTextChanged.connect(self._handle_matching_network_changed)
        component_combo.currentTextChanged.connect(
            lambda _text, stage_controls=controls: self._handle_matching_component_changed(
                stage_controls
            )
        )
        remove_button.clicked.connect(
            lambda _checked=False, stage_controls=controls: self._remove_matching_stage(
                stage_controls
            )
        )
        return controls

    def _build_marker_table(self, headers: list[str]) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        table.setMinimumHeight(210)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
        header.setSortIndicatorShown(True)
        for column in range(1, table.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        table.setSortingEnabled(True)
        return table

    def _s11_marker_table_headers(self) -> list[str]:
        return [
            "Trace",
            "Freq",
            "S11 (dB)",
            "ΔRef (dB)",
            "|S11| (lin)",
            "ΔRef |S11| (lin)",
            "Angle (deg)",
            "Z (ohm)",
            *[header for header, _region_hz, _color in self._s11_area_column_specs()],
        ]

    def _s11_area_column_specs(
        self,
    ) -> list[tuple[str, tuple[float, float] | None, str | None]]:
        unit = self.frequency_scale.unit
        visible_specs = []
        if self._custom_aoi_band_selected():
            visible_specs.append(
                (
                    f"{CUSTOM_AOI_BAND_NAME} Area (|dB|*{unit})",
                    self.aoi_region_hz,
                    ACTIVE_AOI_COLOR if self.aoi_region_hz is not None else None,
                )
            )
        visible_specs.extend(
            [
                (
                    f"{preset_name} Area (|dB|*{unit})",
                    _sorted_frequency_region_hz((preset.start_hz, preset.stop_hz)),
                    self._aoi_preset_overlay_color(preset_name),
                )
                for preset_name in sorted(self._visible_aoi_preset_names, key=_natural_sort_key)
                for preset in [self._aoi_presets.get(preset_name)]
                if preset is not None
            ]
        )
        if visible_specs:
            return visible_specs
        return [
            (
                f"AOI Area (|dB|*{unit})",
                self.aoi_region_hz,
                ACTIVE_AOI_COLOR if self.aoi_region_hz is not None else None,
            )
        ]

    def _build_table_actions_row(
        self,
        title: str,
        export_callback: callable,
    ) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        title_label = QtWidgets.QLabel(title)
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        row.addWidget(title_label)
        row.addStretch(1)
        export_button = QtWidgets.QPushButton("Export CSV")
        export_button.clicked.connect(export_callback)
        row.addWidget(export_button)
        return row

    def _build_aoi_spin_box(self) -> QtWidgets.QDoubleSpinBox:
        spin_box = _SelectAllDoubleSpinBox()
        spin_box.setDecimals(6)
        spin_box.setRange(0.0, 0.0)
        spin_box.setValue(0.0)
        spin_box.setMinimumWidth(95)
        spin_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        spin_box.setKeyboardTracking(False)
        spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        return spin_box

    def _current_user_config(self) -> ViewerUserConfig:
        aoi_start_hz = None
        aoi_stop_hz = None
        if self.aoi_region_hz is not None:
            aoi_start_hz, aoi_stop_hz = _sorted_frequency_region_hz(self.aoi_region_hz)

        return ViewerUserConfig(
            frequency_unit_mode=self.frequency_unit_mode,
            force_light_mode=self.force_light_mode,
            controls_visible=self.controls_toggle_button.isChecked(),
            aoi_visible=self.aoi_enabled_checkbox.isChecked(),
            aoi_start_hz=aoi_start_hz,
            aoi_stop_hz=aoi_stop_hz,
            aoi_unit=self.frequency_scale.unit,
            marker_visible=self.marker_enabled_checkbox.isChecked(),
            marker_frequency_hz=self.marker_frequency_hz,
            threshold_visible=self.threshold_enabled_checkbox.isChecked(),
            threshold_db=float(self.threshold_input.value()),
            aoi_presets=dict(self._aoi_presets),
            selected_aoi_preset=self._selected_aoi_preset_name,
        )

    def _schedule_user_config_save(self) -> None:
        self._config_save_timer.start()

    def _write_user_config(self) -> None:
        self.user_config = self._current_user_config()
        _write_viewer_user_config(self.user_config)

    def _sync_aoi_preset_controls(self) -> None:
        self._updating_aoi_preset_controls = True
        try:
            current_name = self._selected_aoi_preset_name
            self.aoi_preset_combo.blockSignals(True)
            self.aoi_preset_combo.clear()
            self.aoi_preset_combo.addItem("Custom", None)
            for preset_name in sorted(self._aoi_presets, key=_natural_sort_key):
                self.aoi_preset_combo.addItem(preset_name, preset_name)
            if current_name is None:
                self.aoi_preset_combo.setCurrentIndex(0)
            else:
                index = self.aoi_preset_combo.findData(current_name)
                self.aoi_preset_combo.setCurrentIndex(index if index >= 0 else 0)
            self.aoi_preset_combo.blockSignals(False)
            has_region = self.aoi_region_hz is not None and bool(self.traces)
            self.save_aoi_preset_button.setEnabled(has_region)
            self.clear_aoi_button.setEnabled(has_region)
            self.delete_aoi_preset_button.setEnabled(current_name is not None and bool(self.traces))
        finally:
            self._updating_aoi_preset_controls = False

    def _sync_aoi_preset_band_controls(self) -> None:
        self._visible_aoi_preset_names = {
            preset_name
            for preset_name in self._visible_aoi_preset_names
            if preset_name in self._aoi_presets
        }

        self._updating_aoi_preset_band_controls = True
        try:
            self.aoi_preset_bands_menu.clear()
            has_custom_band = self._custom_aoi_band_available()
            preset_names = sorted(self._aoi_presets, key=_natural_sort_key)
            if has_custom_band:
                action = self.aoi_preset_bands_menu.addAction(CUSTOM_AOI_BAND_NAME)
                action.setCheckable(True)
                action.setChecked(self._custom_aoi_band_selected())
                action.toggled.connect(self._handle_custom_aoi_band_toggled)

            if not has_custom_band and not preset_names:
                empty_action = self.aoi_preset_bands_menu.addAction("No presets saved")
                empty_action.setEnabled(False)
            for preset_name in preset_names:
                action = self.aoi_preset_bands_menu.addAction(preset_name)
                action.setCheckable(True)
                action.setChecked(preset_name in self._visible_aoi_preset_names)
                action.toggled.connect(
                    lambda visible, name=preset_name: self._handle_aoi_preset_band_toggled(
                        name,
                        visible,
                    )
                )

            visible_count = len(self._visible_aoi_preset_names) + int(
                self._custom_aoi_band_selected()
            )
            if visible_count == 0:
                self.aoi_preset_bands_button.setText("Bands")
            else:
                self.aoi_preset_bands_button.setText(f"Bands ({visible_count})")
            self.aoi_preset_bands_button.setEnabled(has_custom_band or bool(preset_names))
        finally:
            self._updating_aoi_preset_band_controls = False

    def _custom_aoi_band_available(self) -> bool:
        return self.aoi_region_hz is not None and bool(self.traces)

    def _custom_aoi_band_selected(self) -> bool:
        return (
            self._custom_aoi_band_available()
            and self._selected_aoi_preset_name is None
            and self.aoi_enabled_checkbox.isChecked()
        )

    def _handle_custom_aoi_band_toggled(self, visible: bool) -> None:
        if self._updating_aoi_preset_band_controls or not self._custom_aoi_band_available():
            return

        if self.aoi_enabled_checkbox.isChecked() != visible:
            self.aoi_enabled_checkbox.setChecked(visible)

    def _handle_aoi_preset_band_toggled(self, preset_name: str, visible: bool) -> None:
        if self._updating_aoi_preset_band_controls:
            return

        if visible:
            if preset_name in self._aoi_presets:
                self._visible_aoi_preset_names.add(preset_name)
                if not self.aoi_enabled_checkbox.isChecked():
                    self.aoi_enabled_checkbox.setChecked(True)
        else:
            self._visible_aoi_preset_names.discard(preset_name)
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._update_aoi_region_visibility()
        self._update_marker_table()

    def _apply_aoi_preset(self, preset_name: str) -> None:
        preset = self._aoi_presets.get(preset_name)
        if preset is None:
            return

        self._selected_aoi_preset_name = preset_name
        self._visible_aoi_preset_names = {preset_name}
        self.aoi_region_hz = _sorted_frequency_region_hz((preset.start_hz, preset.stop_hz))
        if preset.marker_frequency_hz is not None:
            self.marker_frequency_hz = preset.marker_frequency_hz
        if self._visible_traces():
            self._sync_aoi_controls_to_region()
        else:
            self._reset_aoi_controls()
        if not self.aoi_enabled_checkbox.isChecked():
            self.aoi_enabled_checkbox.setChecked(True)
        self._update_aoi_region_item()
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._update_aoi_region_visibility()
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_aoi_preset_changed(self, index: int) -> None:
        if self._updating_aoi_preset_controls:
            return

        preset_name = self.aoi_preset_combo.itemData(index)
        if not preset_name:
            self._selected_aoi_preset_name = None
            self._visible_aoi_preset_names.clear()
            self._sync_aoi_preset_controls()
            self._sync_aoi_preset_band_controls()
            self._refresh_aoi_preset_regions()
            self._update_aoi_region_visibility()
            self._update_marker_table()
            self._schedule_user_config_save()
            return
        self._apply_aoi_preset(str(preset_name))

    def _save_current_aoi_as_preset(self) -> None:
        if self.aoi_region_hz is None:
            return

        preset_name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Save AOI Preset",
            "Preset name:",
            text=self._selected_aoi_preset_name or "",
        )
        if not accepted:
            return
        preset_name = preset_name.strip()
        if not preset_name:
            return

        start_hz, stop_hz = _sorted_frequency_region_hz(self.aoi_region_hz)
        self._aoi_presets[preset_name] = AoiPreset(
            start_hz=start_hz,
            stop_hz=stop_hz,
            unit=self.frequency_scale.unit,
            marker_frequency_hz=self.marker_frequency_hz,
        )
        self._selected_aoi_preset_name = preset_name
        self._visible_aoi_preset_names = {preset_name}
        if not self.aoi_enabled_checkbox.isChecked():
            self.aoi_enabled_checkbox.setChecked(True)
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._update_aoi_region_visibility()
        self._update_marker_table()
        self._schedule_user_config_save()

    def _clear_active_aoi(self) -> None:
        self._selected_aoi_preset_name = None
        self._visible_aoi_preset_names.clear()
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        if self.aoi_enabled_checkbox.isChecked():
            self.aoi_enabled_checkbox.setChecked(False)
        else:
            self._sync_control_states()
        self._update_marker_table()
        self._schedule_user_config_save()

    def _delete_selected_aoi_preset(self) -> None:
        preset_name = self._selected_aoi_preset_name
        if preset_name is None:
            return
        self._aoi_presets.pop(preset_name, None)
        self._visible_aoi_preset_names.discard(preset_name)
        self._selected_aoi_preset_name = None
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._schedule_user_config_save()

    def _set_controls_panel_visible(self, visible: bool) -> None:
        self.controls_panel.setVisible(visible)
        self.controls_toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )
        self._schedule_user_config_save()

    def _capture_view_state(self) -> _ViewerViewState:
        return _ViewerViewState(
            s11=self._capture_db_plot_view_state(self.s11_plot),
            s21=self._capture_db_plot_view_state(self.s21_plot),
            smith=self._capture_plot_view_state(self.smith_plot),
            match_s11=self._capture_db_plot_view_state(self.match_s11_plot),
            match_smith=self._capture_plot_view_state(self.match_smith_plot),
        )

    def _capture_db_plot_view_state(self, plot_widget: pg.PlotWidget) -> _DbPlotViewState:
        x_range, y_range = plot_widget.getPlotItem().viewRange()
        return _DbPlotViewState(
            x_range_hz=(
                x_range[0] * self.frequency_scale.factor_hz,
                x_range[1] * self.frequency_scale.factor_hz,
            ),
            y_range=(y_range[0], y_range[1]),
        )

    def _capture_plot_view_state(self, plot_widget: pg.PlotWidget) -> _PlotViewState:
        x_range, y_range = plot_widget.getPlotItem().viewRange()
        return _PlotViewState(
            x_range=(x_range[0], x_range[1]),
            y_range=(y_range[0], y_range[1]),
        )

    def _restore_view_state(self, view_state: _ViewerViewState) -> None:
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is not None:
            self._restore_db_plot_view_state(self.s11_plot, view_state.s11, bounds_hz)
            self._restore_db_plot_view_state(self.s21_plot, view_state.s21, bounds_hz)

        self._restore_plot_view_state(self.smith_plot, view_state.smith)

        if self.match_frequencies_hz is not None:
            self._restore_db_plot_view_state(
                self.match_s11_plot,
                view_state.match_s11,
                (float(self.match_frequencies_hz[0]), float(self.match_frequencies_hz[-1])),
            )

        self._restore_plot_view_state(self.match_smith_plot, view_state.match_smith)
        self._update_marker_plot_label_position()
        self._update_s21_marker_plot_label_position()
        self._update_match_marker_plot_label_position()

    def _restore_db_plot_view_state(
        self,
        plot_widget: pg.PlotWidget,
        view_state: _DbPlotViewState | None,
        bounds_hz: tuple[float, float],
    ) -> None:
        if view_state is None:
            return

        x_range_hz = _clamp_frequency_region_hz(view_state.x_range_hz, bounds_hz)
        plot_item = plot_widget.getPlotItem()
        plot_item.setXRange(
            x_range_hz[0] / self.frequency_scale.factor_hz,
            x_range_hz[1] / self.frequency_scale.factor_hz,
            padding=0.0,
        )
        self._restore_y_range(plot_item, view_state.y_range)

    def _restore_plot_view_state(
        self,
        plot_widget: pg.PlotWidget,
        view_state: _PlotViewState | None,
    ) -> None:
        if view_state is None:
            return

        plot_item = plot_widget.getPlotItem()
        plot_item.setXRange(view_state.x_range[0], view_state.x_range[1], padding=0.0)
        self._restore_y_range(plot_item, view_state.y_range)

    def _restore_y_range(
        self,
        plot_item: pg.PlotItem,
        y_range: tuple[float, float],
    ) -> None:
        minimum_y, maximum_y = y_range
        if not np.isfinite(minimum_y) or not np.isfinite(maximum_y):
            return
        if np.isclose(minimum_y, maximum_y):
            delta = max(abs(minimum_y) * 0.01, 1.0)
            minimum_y -= delta
            maximum_y += delta
        plot_item.setYRange(minimum_y, maximum_y, padding=0.0)

    def _refresh_plots(self, *, preserve_view_state: _ViewerViewState | None = None) -> None:
        visible_traces = self._visible_traces()
        self._choose_frequency_scale()

        self.s11_plot.clear()
        self.smith_plot.clear()
        self.s21_plot.clear()
        self.marker_plot_label = None
        self.s21_marker_plot_label = None
        self.aoi_region_item = None
        self.aoi_preset_region_items = {}
        self.aoi_preset_label_items = {}
        self.s11_threshold_line = None
        self.s21_threshold_line = None

        self._configure_s11_plot()
        self._configure_smith_plot()
        self._configure_s21_plot()

        if visible_traces:
            self._sync_aoi_controls_to_region()
            if self.aoi_region_hz is not None:
                self._add_aoi_region()

        for trace in self.traces:
            trace.s11_curve = None
            trace.smith_curve = None
            trace.s21_curve = None
            trace.s11_marker = None
            trace.smith_marker = None
            trace.s21_marker = None

        for trace in visible_traces:
            self._add_trace_items(trace)

        if visible_traces:
            if self.marker_frequency_hz is None:
                self.marker_frequency_hz = self._default_marker_frequency()

            marker_pen = pg.mkPen("#475569", width=2, style=QtCore.Qt.PenStyle.DashLine)
            self.marker_line = pg.InfiniteLine(angle=90, movable=True, pen=marker_pen)
            self.marker_line.sigPositionChanged.connect(self._handle_s11_marker_moved)
            self.s11_plot.addItem(self.marker_line, ignoreBounds=True)

            self.s21_marker_line = pg.InfiniteLine(angle=90, movable=True, pen=marker_pen)
            self.s21_marker_line.sigPositionChanged.connect(self._handle_s21_marker_moved)
            self.s21_plot.addItem(self.s21_marker_line, ignoreBounds=True)

            self._set_marker_line_values(self.marker_frequency_hz)
        else:
            self.marker_line = None
            self.s21_marker_line = None
            self._reset_aoi_controls()

        self._sync_trace_controls()
        self._sync_matching_controls()
        self._sync_control_states()
        self._refresh_matching_tab()
        self._update_marker_outputs()
        if preserve_view_state is not None:
            self._restore_view_state(preserve_view_state)
        self.summary_label.setText(
            f"{len(self.traces)} trace(s) loaded, {len(visible_traces)} visible"
        )

    def _handle_s11_plot_range_changed(self, *_args: object) -> None:
        self._update_marker_plot_label_position()
        self._update_aoi_preset_label_positions()

    def _configure_s11_plot(self) -> None:
        plot_item = self.s11_plot.getPlotItem()
        plot_item.setTitle("S11 Over Frequency")
        plot_item.setLabel("left", "S11 (dB)")
        plot_item.setLabel("bottom", f"Frequency ({self.frequency_scale.unit})")
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.s11_plot.setMouseEnabled(x=True, y=True)
        if not self._visible_traces():
            plot_item.setYRange(*DEFAULT_EMPTY_DB_RANGE, padding=0.0)

        self.marker_plot_label = self._build_marker_plot_label()
        self.s11_plot.addItem(self.marker_plot_label, ignoreBounds=True)
        self.s11_threshold_line = self._build_threshold_line()
        self.s11_plot.addItem(self.s11_threshold_line, ignoreBounds=True)
        self._update_threshold_lines()
        self._update_marker_plot_label_position()

    def _configure_smith_plot(self) -> None:
        plot_item = self.smith_plot.getPlotItem()
        plot_item.setTitle("Smith Chart")
        self.smith_plot.setMouseEnabled(x=True, y=True)
        add_smith_grid(plot_item)

    def _configure_s21_plot(self) -> None:
        plot_item = self.s21_plot.getPlotItem()
        plot_item.setTitle("S21 Over Frequency")
        plot_item.setLabel("left", "S21 (dB)")
        plot_item.setLabel("bottom", f"Frequency ({self.frequency_scale.unit})")
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.s21_plot.setMouseEnabled(x=True, y=True)
        if not self._visible_traces():
            plot_item.setYRange(*DEFAULT_EMPTY_DB_RANGE, padding=0.0)

        self.s21_marker_plot_label = self._build_marker_plot_label()
        self.s21_plot.addItem(self.s21_marker_plot_label, ignoreBounds=True)
        self.s21_threshold_line = self._build_threshold_line()
        self.s21_plot.addItem(self.s21_threshold_line, ignoreBounds=True)
        self._update_threshold_lines()
        self._update_s21_marker_plot_label_position()

    def reset_view(self) -> None:
        visible_traces = self._visible_traces()
        self._reset_db_plot_view(
            self.s11_plot,
            [trace.data.s11_db() for trace in visible_traces],
        )
        self._reset_db_plot_view(
            self.s21_plot,
            [trace.data.s21_db() for trace in visible_traces if trace.data.has_parameter(2, 1)],
        )
        reset_smith_view(self.smith_plot.getPlotItem())
        if self.match_frequencies_hz is not None:
            self._reset_db_plot_view(
                self.match_s11_plot,
                [
                    self.match_original_s11_db
                    if self.match_original_s11_db is not None
                    else np.asarray([], dtype=np.float64),
                    self.match_transformed_s11_db
                    if self.match_transformed_s11_db is not None
                    else np.asarray([], dtype=np.float64),
                ],
                frequency_span_hz=(
                    float(self.match_frequencies_hz[0]),
                    float(self.match_frequencies_hz[-1]),
                ),
            )
        reset_smith_view(self.match_smith_plot.getPlotItem())
        self._update_marker_plot_label_position()
        self._update_s21_marker_plot_label_position()
        self._update_match_marker_plot_label_position()

    def _reset_db_plot_view(
        self,
        plot_widget: pg.PlotWidget,
        y_data_sets: Sequence[np.ndarray],
        *,
        frequency_span_hz: tuple[float, float] | None = None,
    ) -> None:
        plot_item = plot_widget.getPlotItem()
        if frequency_span_hz is None:
            frequency_span_hz = self._frequency_span_hz()
        if frequency_span_hz is None:
            plot_item.setXRange(0.0, 1.0, padding=0.0)
        else:
            plot_item.setXRange(
                frequency_span_hz[0] / self.frequency_scale.factor_hz,
                frequency_span_hz[1] / self.frequency_scale.factor_hz,
                padding=0.0,
            )

        if not y_data_sets:
            plot_item.setYRange(*DEFAULT_EMPTY_DB_RANGE, padding=0.0)
            return

        min_y = min(float(np.min(values)) for values in y_data_sets)
        max_y = max(float(np.max(values)) for values in y_data_sets)
        if np.isclose(min_y, max_y):
            min_y -= 1.0
            max_y += 1.0
        plot_item.setYRange(min_y, max_y, padding=0.0)

    def _configure_match_s11_plot(self) -> None:
        plot_item = self.match_s11_plot.getPlotItem()
        plot_item.setTitle("Matched S11 Over Frequency")
        plot_item.setLabel("left", "S11 (dB)")
        plot_item.setLabel("bottom", f"Frequency ({self.frequency_scale.unit})")
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.match_s11_plot.setMouseEnabled(x=True, y=True)
        if self.match_frequencies_hz is None:
            plot_item.setYRange(*DEFAULT_EMPTY_DB_RANGE, padding=0.0)

        self.match_marker_plot_label = self._build_marker_plot_label()
        self.match_s11_plot.addItem(self.match_marker_plot_label, ignoreBounds=True)
        self._update_match_marker_plot_label_position()

    def _configure_match_smith_plot(self) -> None:
        plot_item = self.match_smith_plot.getPlotItem()
        plot_item.setTitle("Matched Smith Chart")
        self.match_smith_plot.setMouseEnabled(x=True, y=True)
        add_smith_grid(plot_item)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")

    def _refresh_matching_tab(self) -> None:
        self.match_s11_plot.clear()
        self.match_smith_plot.clear()
        self.match_marker_plot_label = None
        self.match_s11_marker_line = None
        self.match_s11_original_marker = None
        self.match_s11_transformed_marker = None
        self.match_smith_original_marker = None
        self.match_smith_transformed_marker = None
        self.match_original_gamma = None
        self.match_transformed_gamma = None
        self.match_original_s11_db = None
        self.match_transformed_s11_db = None
        self.match_frequencies_hz = None
        self.match_marker_frequency_hz = None

        self._configure_match_s11_plot()
        self._configure_match_smith_plot()

        trace = self._matching_trace()
        if trace is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Load a trace to preview a matching network.")
            return

        frequencies_hz = trace.data.frequencies_hz
        transformed_impedance = apply_matching_network(
            trace.data.impedance_ohms(),
            frequencies_hz,
            self._matching_stages(),
        )
        transformed_gamma = impedance_to_gamma(
            transformed_impedance,
            trace.data.reference_impedance_ohms,
        )
        transformed_s11_db = 20.0 * np.log10(np.maximum(np.abs(transformed_gamma), 1.0e-12))

        self.match_frequencies_hz = frequencies_hz
        self.match_original_gamma = trace.data.gamma
        self.match_transformed_gamma = transformed_gamma
        self.match_original_s11_db = trace.data.s11_db()
        self.match_transformed_s11_db = transformed_s11_db

        scaled_frequency = frequencies_hz / self.frequency_scale.factor_hz
        original_pen = pg.mkPen(trace.color, width=2.2)
        transformed_pen = pg.mkPen("#dc2626", width=2.4)

        self.match_s11_plot.plot(
            scaled_frequency,
            self.match_original_s11_db,
            pen=original_pen,
            name=f"{trace.data.label} original",
        )
        self.match_s11_plot.plot(
            scaled_frequency,
            self.match_transformed_s11_db,
            pen=transformed_pen,
            name="Matched",
        )
        self.match_smith_plot.plot(
            self.match_original_gamma.real,
            self.match_original_gamma.imag,
            pen=original_pen,
            name=f"{trace.data.label} original",
        )
        self.match_smith_plot.plot(
            self.match_transformed_gamma.real,
            self.match_transformed_gamma.imag,
            pen=transformed_pen,
            name="Matched",
        )

        self.match_s11_original_marker = self._build_marker_scatter(trace.color, size=10)
        self.match_s11_transformed_marker = self._build_marker_scatter("#dc2626", size=10)
        self.match_smith_original_marker = self._build_marker_scatter(trace.color, size=11)
        self.match_smith_transformed_marker = self._build_marker_scatter("#dc2626", size=11)
        self.match_s11_plot.addItem(self.match_s11_original_marker)
        self.match_s11_plot.addItem(self.match_s11_transformed_marker)
        self.match_smith_plot.addItem(self.match_smith_original_marker)
        self.match_smith_plot.addItem(self.match_smith_transformed_marker)

        marker_pen = pg.mkPen("#475569", width=2, style=QtCore.Qt.PenStyle.DashLine)
        self.match_s11_marker_line = pg.InfiniteLine(angle=90, movable=True, pen=marker_pen)
        self.match_s11_marker_line.sigPositionChanged.connect(self._handle_match_marker_moved)
        self.match_s11_plot.addItem(self.match_s11_marker_line, ignoreBounds=True)
        marker_frequency_hz = self._matching_target_frequency_hz()
        if marker_frequency_hz is not None:
            self.match_marker_frequency_hz = marker_frequency_hz
            self.match_s11_marker_line.setValue(marker_frequency_hz / self.frequency_scale.factor_hz)

        self._reset_db_plot_view(
            self.match_s11_plot,
            [self.match_original_s11_db, self.match_transformed_s11_db],
            frequency_span_hz=(float(frequencies_hz[0]), float(frequencies_hz[-1])),
        )
        self._update_matching_marker_outputs()

    def _default_matching_marker_frequency(self, frequencies_hz: np.ndarray) -> float:
        return 0.5 * (float(frequencies_hz[0]) + float(frequencies_hz[-1]))

    def _build_marker_plot_label(self) -> pg.TextItem:
        label = pg.TextItem(
            text="Marker: n/a",
            color="#1e293b",
            anchor=(1.0, 0.0),
            border=pg.mkPen("#93c5fd", width=1.2),
            fill=pg.mkBrush(255, 255, 255, 228),
        )
        label.setZValue(30)
        return label

    def _build_threshold_line(self) -> pg.InfiniteLine:
        return pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("#dc2626", width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            label="{value:.1f} dB",
            labelOpts={
                "position": 0.95,
                "color": "#dc2626",
                "fill": "#fff7eddd",
                "movable": False,
            },
        )

    def _choose_frequency_scale(self) -> None:
        if self.frequency_unit_mode != DEFAULT_FREQUENCY_UNIT_MODE:
            factor_hz = FREQUENCY_UNIT_FACTORS_HZ[self.frequency_unit_mode]
            self.frequency_scale = FrequencyScale(factor_hz, self.frequency_unit_mode)
            return

        visible_traces = self._visible_traces()
        if not visible_traces:
            self.frequency_scale = FrequencyScale(1.0e6, "MHz")
            return

        max_frequency = max(trace.data.frequencies_hz[-1] for trace in visible_traces)
        if max_frequency >= 1.0e9:
            self.frequency_scale = FrequencyScale(1.0e9, "GHz")
        elif max_frequency >= 1.0e6:
            self.frequency_scale = FrequencyScale(1.0e6, "MHz")
        elif max_frequency >= 1.0e3:
            self.frequency_scale = FrequencyScale(1.0e3, "kHz")
        else:
            self.frequency_scale = FrequencyScale(1.0, "Hz")

    def _add_trace_items(self, trace: LoadedTrace) -> None:
        pen = pg.mkPen(trace.color, width=3 if self.reference_trace_path == trace.data.path else 2)
        scaled_frequency = trace.data.frequencies_hz / self.frequency_scale.factor_hz

        trace.s11_curve = self.s11_plot.plot(
            scaled_frequency,
            trace.data.s11_db(),
            pen=pen,
            name=self._display_name_for_trace(trace),
        )
        trace.smith_curve = self.smith_plot.plot(
            trace.data.gamma.real,
            trace.data.gamma.imag,
            pen=pen,
        )

        trace.s11_marker = self._build_marker_scatter(trace.color, size=10)
        trace.smith_marker = self._build_marker_scatter(trace.color, size=11)
        self.s11_plot.addItem(trace.s11_marker)
        self.smith_plot.addItem(trace.smith_marker)

        if trace.data.has_parameter(2, 1):
            trace.s21_curve = self.s21_plot.plot(
                scaled_frequency,
                trace.data.s21_db(),
                pen=pen,
                name=self._display_name_for_trace(trace),
            )
            trace.s21_marker = self._build_marker_scatter(trace.color, size=10)
            self.s21_plot.addItem(trace.s21_marker)

    def _build_marker_scatter(self, color: str, size: int) -> pg.ScatterPlotItem:
        return pg.ScatterPlotItem(
            size=size,
            brush=pg.mkBrush(color),
            pen=pg.mkPen("#ffffff", width=1.5),
        )

    def _aoi_preset_overlay_color(self, preset_name: str) -> str:
        preset_names = sorted(self._aoi_presets, key=_natural_sort_key)
        try:
            color_index = preset_names.index(preset_name)
        except ValueError:
            color_index = 0
        return AOI_PRESET_OVERLAY_COLORS[color_index % len(AOI_PRESET_OVERLAY_COLORS)]

    def _active_aoi_matches_preset(self, preset_name: str) -> bool:
        preset = self._aoi_presets.get(preset_name)
        if preset is None or self.aoi_region_hz is None:
            return False

        active_region_hz = _sorted_frequency_region_hz(self.aoi_region_hz)
        preset_region_hz = _sorted_frequency_region_hz((preset.start_hz, preset.stop_hz))
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is not None:
            active_region_hz = _clamp_frequency_region_hz(active_region_hz, bounds_hz)
            visible_preset_region_hz = _intersect_frequency_region_hz(preset_region_hz, bounds_hz)
            if visible_preset_region_hz is None:
                return False
            preset_region_hz = visible_preset_region_hz

        return bool(
            np.isclose(active_region_hz[0], preset_region_hz[0])
            and np.isclose(active_region_hz[1], preset_region_hz[1])
        )

    def _clear_aoi_preset_regions(self) -> None:
        for region_item in self.aoi_preset_region_items.values():
            self.s11_plot.removeItem(region_item)
        for label_item in self.aoi_preset_label_items.values():
            self.s11_plot.removeItem(label_item)
        self.aoi_preset_region_items.clear()
        self.aoi_preset_label_items.clear()

    def _refresh_aoi_preset_regions(self) -> None:
        self._clear_aoi_preset_regions()
        if not self._aoi_overlay_enabled():
            return

        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            return

        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)

        for preset_name in sorted(self._visible_aoi_preset_names, key=_natural_sort_key):
            preset = self._aoi_presets.get(preset_name)
            if preset is None:
                continue

            visible_region_hz = _intersect_frequency_region_hz(
                (preset.start_hz, preset.stop_hz),
                bounds_hz,
            )
            if visible_region_hz is None:
                continue

            color = QtGui.QColor(self._aoi_preset_overlay_color(preset_name))
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(52)
            line_color = QtGui.QColor(color)
            line_color.setAlpha(200)

            scaled_region = [
                visible_region_hz[0] / self.frequency_scale.factor_hz,
                visible_region_hz[1] / self.frequency_scale.factor_hz,
            ]
            region_item = pg.LinearRegionItem(
                values=scaled_region,
                orientation="vertical",
                brush=pg.mkBrush(fill_color),
                pen=pg.mkPen(line_color, width=1.6),
                hoverBrush=pg.mkBrush(fill_color),
                hoverPen=pg.mkPen(line_color, width=1.6),
                movable=False,
            )
            region_item.setZValue(-7)
            self.s11_plot.addItem(region_item, ignoreBounds=True)
            self.aoi_preset_region_items[preset_name] = region_item

            label = pg.TextItem(
                text=preset_name,
                color="#0f172a",
                anchor=(0.5, 0.0),
                border=pg.mkPen(line_color, width=1.0),
                fill=pg.mkBrush(255, 255, 255, 224),
            )
            label.setFont(font)
            label.setZValue(22)
            self.s11_plot.addItem(label, ignoreBounds=True)
            self.aoi_preset_label_items[preset_name] = label

        self._update_aoi_preset_label_positions()

    def _add_aoi_region(self) -> None:
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None or self.aoi_region_hz is None:
            return

        start_hz, stop_hz = _clamp_frequency_region_hz(self.aoi_region_hz, bounds_hz)

        scaled_bounds = tuple(value / self.frequency_scale.factor_hz for value in bounds_hz)
        scaled_region = [
            start_hz / self.frequency_scale.factor_hz,
            stop_hz / self.frequency_scale.factor_hz,
        ]

        self.aoi_region_item = pg.LinearRegionItem(
            values=scaled_region,
            orientation="vertical",
            brush=pg.mkBrush(147, 197, 253, 76),
            pen=pg.mkPen(ACTIVE_AOI_COLOR, width=1.8),
            hoverBrush=pg.mkBrush(96, 165, 250, 104),
            hoverPen=pg.mkPen("#3b82f6", width=2.0),
            movable=False,
            bounds=scaled_bounds,
        )
        self.aoi_region_item.setZValue(-5)
        self.s11_plot.addItem(self.aoi_region_item, ignoreBounds=True)

        self.aoi_region_hz = (start_hz, stop_hz)
        self._refresh_aoi_preset_regions()

    def load_files(self, paths: Sequence[Path]) -> None:
        existing_paths = {trace.data.path for trace in self.traces}
        new_traces: list[LoadedTrace] = []
        errors: list[str] = []
        remembered_directory = False

        for raw_path in paths:
            path = raw_path.expanduser().resolve()
            if not remembered_directory:
                self._remember_directory(path.parent)
                remembered_directory = True
            if path in existing_paths:
                continue

            try:
                data = load_touchstone(path)
            except Exception as exc:  # pragma: no cover - UI error path
                errors.append(f"{path.name}: {exc}")
                continue

            color = TRACE_COLORS[(len(self.traces) + len(new_traces)) % len(TRACE_COLORS)]
            new_traces.append(LoadedTrace(data=data, color=color))
            existing_paths.add(path)

        if new_traces:
            self.traces.extend(new_traces)
            self.traces.sort(key=_trace_sort_key)
            if self.marker_frequency_hz is None:
                self.marker_frequency_hz = self._default_marker_frequency()
            self._refresh_plots()
            self.reset_view()

        if errors:
            QtWidgets.QMessageBox.warning(self, "Touchstone Load Errors", "\n".join(errors))

    def clear_traces(self) -> None:
        self.traces.clear()
        self._refresh_plots()

    def _open_files_dialog(self) -> None:
        selected, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Touchstone Files",
            str(self._default_browser_directory()),
            "Touchstone files (*.s1p *.S1P *.s2p *.S2P);;All files (*.*)",
        )
        if not selected:
            return
        self.load_files([Path(path) for path in selected])

    def _export_table_to_csv(
        self,
        table: QtWidgets.QTableWidget,
        *,
        suggested_name: str,
    ) -> None:
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Table to CSV",
            str(self._default_browser_directory() / suggested_name),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not selected:
            return
        self._write_table_to_csv(table, Path(selected))

    def _write_table_to_csv(
        self,
        table: QtWidgets.QTableWidget,
        path: Path,
    ) -> None:
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        table.horizontalHeaderItem(column).text()
                        if table.horizontalHeaderItem(column) is not None
                        else ""
                        for column in range(table.columnCount())
                    ]
                )
                for row in range(table.rowCount()):
                    writer.writerow(
                        [
                            table.item(row, column).text()
                            if table.item(row, column) is not None
                            else ""
                            for column in range(table.columnCount())
                        ]
                    )
        except OSError as exc:  # pragma: no cover - UI error path
            QtWidgets.QMessageBox.warning(
                self,
                "CSV Export Failed",
                f"{path.name}: {exc}",
            )

    def _default_browser_directory(self) -> Path:
        saved_directory = self.settings.value(LAST_OPEN_DIRECTORY_KEY, None, type=str)
        downloads_directory = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.DownloadLocation
        )
        home_directory = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.HomeLocation
        )
        return _resolve_browser_directory(saved_directory, downloads_directory, home_directory)

    def _remember_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        self.settings.setValue(LAST_OPEN_DIRECTORY_KEY, str(directory))
        self.settings.sync()

    def _default_marker_frequency(self) -> float:
        visible_traces = self._visible_traces()
        overlap_start = max(trace.data.frequencies_hz[0] for trace in visible_traces)
        overlap_end = min(trace.data.frequencies_hz[-1] for trace in visible_traces)
        if overlap_start < overlap_end:
            return 0.5 * (overlap_start + overlap_end)

        first_trace = visible_traces[0].data
        return 0.5 * (first_trace.frequencies_hz[0] + first_trace.frequencies_hz[-1])

    def _handle_plot_click(self, plot_widget: pg.PlotWidget, event: object) -> None:
        if not self._marker_overlay_enabled():
            return

        button = getattr(event, "button", None)
        if callable(button):
            button = button()
        if button is not None and button != QtCore.Qt.MouseButton.LeftButton:
            return

        if not hasattr(event, "scenePos"):
            return

        scene_position = event.scenePos()
        bounding_rect = plot_widget.getPlotItem().vb.sceneBoundingRect()
        if not bounding_rect.contains(scene_position):
            return

        mouse_point = plot_widget.getPlotItem().vb.mapSceneToView(scene_position)
        frequency_hz = mouse_point.x() * self.frequency_scale.factor_hz
        self.marker_frequency_hz = frequency_hz
        self._set_marker_line_values(frequency_hz)
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_s11_marker_moved(self) -> None:
        if not self.marker_line or self._updating_marker or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = self.marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_s21_marker_moved(self) -> None:
        if not self.s21_marker_line or self._updating_marker or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = self.s21_marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_match_marker_moved(self) -> None:
        if not self.match_s11_marker_line or self._updating_marker or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = self.match_s11_marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_aoi_value_changed(self, _value: float) -> None:
        if self._updating_aoi_controls or not self._visible_traces():
            return

        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            return

        factor_hz = self._aoi_unit_factor_hz()
        start_hz = self.aoi_start_input.value() * factor_hz
        stop_hz = self.aoi_stop_input.value() * factor_hz
        changed_control = self.sender()
        if changed_control is self.aoi_start_input and start_hz > stop_hz:
            stop_hz = start_hz
        elif changed_control is self.aoi_stop_input and stop_hz < start_hz:
            start_hz = stop_hz
        self.aoi_region_hz = _clamp_frequency_region_hz((start_hz, stop_hz), bounds_hz)
        self._selected_aoi_preset_name = None
        self._sync_aoi_controls_to_region()
        self._update_aoi_region_item()
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._update_marker_table()
        self._schedule_user_config_save()

    def _handle_frequency_unit_changed(self, unit: str) -> None:
        view_state = self._capture_view_state()
        self.frequency_unit_mode = unit
        self._refresh_plots(preserve_view_state=view_state)
        self._schedule_user_config_save()

    def _handle_light_mode_toggled(self, enabled: bool) -> None:
        self.force_light_mode = enabled
        self.force_light_theme_checkbox.blockSignals(True)
        self.force_light_theme_checkbox.setChecked(enabled)
        self.force_light_theme_checkbox.blockSignals(False)
        _apply_application_appearance(enabled)
        self._schedule_user_config_save()

    def _handle_marker_frequency_changed(self, value: float) -> None:
        if self._updating_marker_controls or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = value * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_aoi_visibility_changed(self, visible: bool) -> None:
        if visible and self._visible_traces():
            bounds_hz = self._frequency_span_hz()
            if bounds_hz is not None and self.aoi_region_hz is None:
                factor_hz = self._aoi_unit_factor_hz()
                self.aoi_region_hz = _clamp_frequency_region_hz(
                    (
                        self.aoi_start_input.value() * factor_hz,
                        self.aoi_stop_input.value() * factor_hz,
                    ),
                    bounds_hz,
                )
            self._update_aoi_region_item()
        self._sync_control_states()
        self._schedule_user_config_save()

    def _handle_threshold_changed(self, _value: float) -> None:
        self._update_threshold_lines()
        self._schedule_user_config_save()

    def _handle_threshold_visibility_changed(self, visible: bool) -> None:
        self.threshold_input.setEnabled(visible)
        self._update_threshold_lines()
        self._schedule_user_config_save()

    def _handle_marker_visibility_changed(self, _visible: bool) -> None:
        self._sync_control_states()
        self._update_marker_outputs()
        self._schedule_user_config_save()

    def _handle_reference_trace_changed(self, index: int) -> None:
        if self._updating_trace_controls:
            return

        path_value = self.reference_trace_combo.itemData(index)
        self.reference_trace_path = Path(path_value) if path_value else None
        self._refresh_plots()

    def _handle_trace_visibility_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if self._updating_trace_controls:
            return

        path_value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path_value:
            return

        path = Path(path_value)
        visible = item.checkState() == QtCore.Qt.CheckState.Checked
        for trace in self.traces:
            if trace.data.path == path:
                trace.visible = visible
                break

        self._refresh_plots()

    def _set_all_traces_visible(self, visible: bool) -> None:
        if not self.traces:
            return

        for trace in self.traces:
            trace.visible = visible
        self._refresh_plots()

    def _aoi_unit_factor_hz(self) -> float:
        return self.frequency_scale.factor_hz

    def _aoi_overlay_enabled(self) -> bool:
        return bool(self._visible_traces()) and self.aoi_enabled_checkbox.isChecked()

    def _marker_overlay_enabled(self) -> bool:
        return bool(self._visible_traces()) and self.marker_enabled_checkbox.isChecked()

    def _sync_control_states(self) -> None:
        has_loaded_traces = bool(self.traces)
        has_visible_traces = bool(self._visible_traces())
        self.aoi_enabled_checkbox.setEnabled(has_loaded_traces)
        self.marker_enabled_checkbox.setEnabled(has_loaded_traces)
        self.aoi_preset_combo.setEnabled(has_loaded_traces)
        self._set_aoi_controls_enabled(self._aoi_overlay_enabled())
        self._sync_marker_frequency_input()
        self._update_aoi_region_visibility()
        self._sync_aoi_preset_controls()
        self._sync_aoi_preset_band_controls()
        self._refresh_aoi_preset_regions()
        self._update_marker_visibility()
        if not has_visible_traces:
            self._reset_aoi_controls()

    def _set_aoi_controls_enabled(self, enabled: bool) -> None:
        self.aoi_start_input.setEnabled(enabled)
        self.aoi_stop_input.setEnabled(enabled)
        self.aoi_display_unit_label.setEnabled(enabled)

    def _update_aoi_region_visibility(self) -> None:
        if self.aoi_region_item is None:
            return

        self.aoi_region_item.setVisible(
            self._aoi_overlay_enabled()
            and not (
                self._selected_aoi_preset_name is not None
                and self._active_aoi_matches_preset(self._selected_aoi_preset_name)
            )
        )

    def _reset_aoi_controls(self) -> None:
        self._updating_aoi_controls = True
        try:
            self.aoi_display_unit_label.setText(self.frequency_scale.unit)
            for spin_box in (self.aoi_start_input, self.aoi_stop_input):
                spin_box.setRange(0.0, 0.0)
                spin_box.setSingleStep(0.001)
                spin_box.setValue(0.0)
        finally:
            self._updating_aoi_controls = False
        self._sync_aoi_preset_controls()

    def _sync_aoi_controls_to_region(self) -> None:
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            self._reset_aoi_controls()
            return

        factor_hz = self._aoi_unit_factor_hz()
        display_bounds = (bounds_hz[0] / factor_hz, bounds_hz[1] / factor_hz)
        step_size = max((display_bounds[1] - display_bounds[0]) / 200.0, 1.0e-6)

        self._updating_aoi_controls = True
        try:
            self.aoi_display_unit_label.setText(self.frequency_scale.unit)
            for spin_box in (self.aoi_start_input, self.aoi_stop_input):
                spin_box.setRange(display_bounds[0], display_bounds[1])
                spin_box.setSingleStep(step_size)

            if self.aoi_region_hz is None:
                self.aoi_start_input.setValue(display_bounds[0])
                self.aoi_stop_input.setValue(display_bounds[1])
            else:
                self.aoi_region_hz = _clamp_frequency_region_hz(self.aoi_region_hz, bounds_hz)
                display_region = (
                    self.aoi_region_hz[0] / factor_hz,
                    self.aoi_region_hz[1] / factor_hz,
                )
                self.aoi_start_input.setValue(display_region[0])
                self.aoi_stop_input.setValue(display_region[1])
        finally:
            self._updating_aoi_controls = False
        self._sync_aoi_preset_controls()

    def _update_aoi_region_item(self) -> None:
        if self.aoi_region_hz is None:
            if self.aoi_region_item is not None:
                self.s11_plot.removeItem(self.aoi_region_item)
                self.aoi_region_item = None
            return
        if self.aoi_region_item is None:
            self._add_aoi_region()
            self._update_aoi_region_visibility()
            return

        self.aoi_region_item.setRegion(
            [
                self.aoi_region_hz[0] / self.frequency_scale.factor_hz,
                self.aoi_region_hz[1] / self.frequency_scale.factor_hz,
            ]
        )
        self._update_aoi_region_visibility()

    def _update_threshold_lines(self) -> None:
        threshold_db = -float(self.threshold_input.value())
        visible = self.threshold_enabled_checkbox.isChecked()
        if self.s11_threshold_line is not None:
            self.s11_threshold_line.setValue(threshold_db)
            self.s11_threshold_line.setVisible(visible)
        if self.s21_threshold_line is not None:
            self.s21_threshold_line.setValue(threshold_db)
            self.s21_threshold_line.setVisible(visible)

    def _update_marker_visibility(self) -> None:
        visible = self._marker_overlay_enabled()

        if visible and self.marker_frequency_hz is None and self._visible_traces():
            self.marker_frequency_hz = self._default_marker_frequency()

        for marker_line in (self.marker_line, self.s21_marker_line):
            if marker_line is not None:
                marker_line.setVisible(visible)

        self.marker_table.setHidden(not visible)
        self.s21_marker_table.setHidden(not visible)

        if not visible:
            for trace in self.traces:
                self._set_scatter_point(trace.s11_marker, None)
                self._set_scatter_point(trace.smith_marker, None)
                self._set_scatter_point(trace.s21_marker, None)
            self._sync_marker_frequency_input()
            return

        if self.marker_frequency_hz is not None:
            self._set_marker_line_values(self.marker_frequency_hz)
        self._sync_marker_frequency_input()

    def _set_marker_line_values(self, frequency_hz: float) -> None:
        self._updating_marker = True
        try:
            if self.marker_line is not None:
                self.marker_line.setValue(frequency_hz / self.frequency_scale.factor_hz)
            if self.s21_marker_line is not None:
                self.s21_marker_line.setValue(frequency_hz / self.frequency_scale.factor_hz)
        finally:
            self._updating_marker = False

    def _update_marker_outputs(self) -> None:
        self._sync_marker_frequency_input()
        self._sync_match_target_frequency_input()
        self._update_marker_plot_label()
        self._update_s21_marker_plot_label()
        self._update_marker_table()
        self._update_s21_marker_table()
        self._update_matching_marker_outputs()

    def _sync_marker_frequency_input(self) -> None:
        bounds_hz = self._frequency_span_hz()
        visible = self._marker_overlay_enabled() and bounds_hz is not None
        self._updating_marker_controls = True
        try:
            self.marker_frequency_input.setEnabled(visible)
            self.marker_frequency_unit_label.setText(self.frequency_scale.unit)
            self.marker_frequency_unit_label.setEnabled(visible)
            if not visible:
                self.marker_frequency_input.setRange(0.0, 0.0)
                self.marker_frequency_input.setValue(0.0)
                return

            display_bounds = (
                bounds_hz[0] / self.frequency_scale.factor_hz,
                bounds_hz[1] / self.frequency_scale.factor_hz,
            )
            step_size = max((display_bounds[1] - display_bounds[0]) / 200.0, 1.0e-6)
            self.marker_frequency_input.setRange(display_bounds[0], display_bounds[1])
            self.marker_frequency_input.setSingleStep(step_size)

            marker_frequency_hz = self.marker_frequency_hz
            if marker_frequency_hz is None:
                marker_frequency_hz = self._default_marker_frequency()
            marker_frequency_hz = min(max(marker_frequency_hz, bounds_hz[0]), bounds_hz[1])
            self.marker_frequency_hz = marker_frequency_hz
            self._set_marker_line_values(marker_frequency_hz)
            self.marker_frequency_input.setValue(
                marker_frequency_hz / self.frequency_scale.factor_hz
            )
        finally:
            self._updating_marker_controls = False

    def _update_marker_plot_label(self) -> None:
        self._update_plot_marker_label(
            self.marker_plot_label,
            self.s11_plot,
            self.marker_frequency_hz,
            self._update_marker_plot_label_position,
        )

    def _update_s21_marker_plot_label(self) -> None:
        self._update_plot_marker_label(
            self.s21_marker_plot_label,
            self.s21_plot,
            self.marker_frequency_hz,
            self._update_s21_marker_plot_label_position,
        )

    def _update_plot_marker_label(
        self,
        label: pg.TextItem | None,
        _plot_widget: pg.PlotWidget,
        frequency_hz: float | None,
        position_callback: callable,
    ) -> None:
        if label is None:
            return

        visible = self._marker_overlay_enabled()
        label.setVisible(visible)
        if not visible:
            return

        if frequency_hz is None:
            label.setText("Marker: n/a")
        else:
            display_frequency = frequency_hz / self.frequency_scale.factor_hz
            label.setText(f"Marker: {display_frequency:.6f} {self.frequency_scale.unit}")

        position_callback()

    def _update_marker_plot_label_position(self, *_args: object) -> None:
        self._position_plot_label(self.marker_plot_label, self.s11_plot)

    def _update_s21_marker_plot_label_position(self, *_args: object) -> None:
        self._position_plot_label(self.s21_marker_plot_label, self.s21_plot)

    def _update_match_marker_plot_label_position(self, *_args: object) -> None:
        self._position_plot_label(self.match_marker_plot_label, self.match_s11_plot)

    def _update_aoi_preset_label_positions(self) -> None:
        if not self.aoi_preset_label_items:
            return

        x_range, y_range = self.s11_plot.getPlotItem().viewRange()
        x_margin = (x_range[1] - x_range[0]) * 0.02
        y_margin = max((y_range[1] - y_range[0]) * 0.07, 0.5)
        minimum_x = x_range[0] + x_margin
        maximum_x = x_range[1] - x_margin

        for index, preset_name in enumerate(sorted(self.aoi_preset_label_items, key=_natural_sort_key)):
            label = self.aoi_preset_label_items[preset_name]
            region_item = self.aoi_preset_region_items.get(preset_name)
            if region_item is None:
                continue

            region = region_item.getRegion()
            center_x = 0.5 * (region[0] + region[1])
            clamped_x = min(max(center_x, minimum_x), maximum_x)
            label.setPos(clamped_x, y_range[1] - y_margin * (index + 1))

    def _position_plot_label(
        self,
        label: pg.TextItem | None,
        plot_widget: pg.PlotWidget,
    ) -> None:
        if label is None:
            return

        x_range, y_range = plot_widget.getPlotItem().viewRange()
        x_margin = (x_range[1] - x_range[0]) * 0.02
        y_margin = (y_range[1] - y_range[0]) * 0.04
        label.setPos(x_range[1] - x_margin, y_range[1] - y_margin)

    def _preferred_aoi_span_hz(
        self, fallback_bounds_hz: tuple[float, float]
    ) -> tuple[float, float]:
        visible_traces = self._visible_traces()
        overlap_start = max(trace.data.frequencies_hz[0] for trace in visible_traces)
        overlap_end = min(trace.data.frequencies_hz[-1] for trace in visible_traces)
        if overlap_start < overlap_end:
            return (overlap_start, overlap_end)
        return fallback_bounds_hz

    def _frequency_span_hz(self) -> tuple[float, float] | None:
        visible_traces = self._visible_traces()
        if not visible_traces:
            return None

        minimum_hz = min(trace.data.frequencies_hz[0] for trace in visible_traces)
        maximum_hz = max(trace.data.frequencies_hz[-1] for trace in visible_traces)
        return (minimum_hz, maximum_hz)

    def _begin_table_update(
        self, table: QtWidgets.QTableWidget
    ) -> tuple[int, QtCore.Qt.SortOrder]:
        header = table.horizontalHeader()
        sort_column = header.sortIndicatorSection()
        if sort_column < 0:
            sort_column = 0
        sort_order = header.sortIndicatorOrder()
        table.setSortingEnabled(False)
        return sort_column, sort_order

    def _end_table_update(
        self,
        table: QtWidgets.QTableWidget,
        sort_column: int,
        sort_order: QtCore.Qt.SortOrder,
    ) -> None:
        table.setSortingEnabled(True)
        sort_column = min(max(sort_column, 0), table.columnCount() - 1)
        table.sortItems(sort_column, sort_order)
        table.resizeRowsToContents()

    def _set_s11_table_headers(
        self,
        headers: list[str],
        *,
        background_colors: list[str | None] | None = None,
    ) -> None:
        self.marker_table.setColumnCount(len(headers))
        self.marker_table.setHorizontalHeaderLabels(headers)
        header = self.marker_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, self.marker_table.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        for column in range(self.marker_table.columnCount()):
            header_item = self.marker_table.horizontalHeaderItem(column)
            if header_item is None:
                continue
            color_hex = None
            if background_colors is not None and column < len(background_colors):
                color_hex = background_colors[column]
            if color_hex:
                header_item.setBackground(_table_tint_brush(color_hex, alpha=88))
            else:
                header_item.setBackground(QtGui.QBrush())

    def _marker_parameter(
        self,
        trace: LoadedTrace,
        out_port: int,
        in_port: int,
    ) -> complex | None:
        if not self._marker_overlay_enabled() or self.marker_frequency_hz is None:
            return None
        return trace.data.interpolated_parameter(out_port, in_port, self.marker_frequency_hz)

    def _update_s11_marker_points(self, trace: LoadedTrace) -> complex | None:
        parameter = self._marker_parameter(trace, 1, 1)
        if parameter is None or self.marker_frequency_hz is None:
            self._set_scatter_point(trace.s11_marker, None)
            self._set_scatter_point(trace.smith_marker, None)
            return None

        display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
        self._set_scatter_point(trace.s11_marker, (display_frequency, _parameter_db(parameter)))
        self._set_scatter_point(trace.smith_marker, (parameter.real, parameter.imag))
        return parameter

    def _aoi_parameter_segment(
        self,
        trace: LoadedTrace,
        out_port: int,
        in_port: int,
        region_hz: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if region_hz is None:
            region_hz = self.aoi_region_hz
        if region_hz is None or not trace.data.has_parameter(out_port, in_port):
            return None

        start_hz, stop_hz = sorted(region_hz)
        minimum_hz = float(trace.data.frequencies_hz[0])
        maximum_hz = float(trace.data.frequencies_hz[-1])
        if start_hz < minimum_hz or stop_hz > maximum_hz:
            return None

        start_parameter = trace.data.interpolated_parameter(out_port, in_port, start_hz)
        stop_parameter = trace.data.interpolated_parameter(out_port, in_port, stop_hz)
        if start_parameter is None or stop_parameter is None:
            return None

        if np.isclose(start_hz, stop_hz):
            return (
                np.asarray([start_hz], dtype=np.float64),
                np.asarray([start_parameter], dtype=np.complex128),
            )

        mask = (trace.data.frequencies_hz > start_hz) & (trace.data.frequencies_hz < stop_hz)
        segment_frequencies_hz = np.concatenate(
            (
                np.asarray([start_hz], dtype=np.float64),
                trace.data.frequencies_hz[mask],
                np.asarray([stop_hz], dtype=np.float64),
            )
        )
        segment_parameters = np.concatenate(
            (
                np.asarray([start_parameter], dtype=np.complex128),
                trace.data.parameter(out_port, in_port)[mask],
                np.asarray([stop_parameter], dtype=np.complex128),
            )
        )
        return (segment_frequencies_hz, segment_parameters)

    def _aoi_area_value(
        self,
        trace: LoadedTrace,
        out_port: int,
        in_port: int,
        region_hz: tuple[float, float] | None = None,
    ) -> float | None:
        segment = self._aoi_parameter_segment(trace, out_port, in_port, region_hz=region_hz)
        if segment is None:
            return None
        factor_hz = self._aoi_unit_factor_hz()
        segment_s_db = 20.0 * np.log10(np.maximum(np.abs(segment[1]), 1.0e-12))
        return float(np.trapezoid(np.abs(segment_s_db), x=segment[0] / factor_hz))

    def _update_marker_table(self) -> None:
        visible_traces = self._visible_traces()
        reference_trace = self._reference_trace()
        area_column_specs = self._s11_area_column_specs()
        area_column_backgrounds = [color_hex for _header, _region_hz, color_hex in area_column_specs]
        column_backgrounds = [None] * 8 + area_column_backgrounds
        reference_parameter = None
        if reference_trace is not None and self.marker_frequency_hz is not None:
            reference_parameter = reference_trace.data.interpolated_parameter(1, 1, self.marker_frequency_hz)

        sort_column, sort_order = self._begin_table_update(self.marker_table)
        self._set_s11_table_headers(
            self._s11_marker_table_headers(),
            background_colors=column_backgrounds,
        )
        if not self._marker_overlay_enabled():
            self.marker_table.setRowCount(0)
            for trace in visible_traces:
                self._update_s11_marker_points(trace)
            self._end_table_update(self.marker_table, sort_column, sort_order)
            return

        self.marker_table.setRowCount(len(visible_traces))

        for row_index, trace in enumerate(visible_traces):
            parameter = self._update_s11_marker_points(trace)
            area_values = [
                self._aoi_area_value(trace, 1, 1, region_hz=region_hz)
                for _header, region_hz, _color in area_column_specs
            ]
            area_texts = ["-" if area_value is None else f"{area_value:.3f}" for area_value in area_values]

            if parameter is None:
                values = [
                    self._display_name_for_trace(trace),
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    *area_texts,
                ]
                sort_values = [None] * len(values)
                for column_offset, area_value in enumerate(area_values, start=8):
                    if area_value is not None:
                        sort_values[column_offset] = area_value
            else:
                display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
                s11_db = _parameter_db(parameter)
                impedance = gamma_to_impedance(parameter, trace.data.reference_impedance_ohms)
                reference_s11_db = (
                    None if reference_parameter is None else _parameter_db(reference_parameter)
                )
                reference_s11_mag = None if reference_parameter is None else abs(reference_parameter)
                delta_s11_db = _format_optional_delta(
                    s11_db,
                    reference_s11_db,
                    precision=3,
                )
                delta_s11_mag = _format_optional_delta(
                    abs(parameter),
                    reference_s11_mag,
                    precision=4,
                )
                values = [
                    self._display_name_for_trace(trace),
                    f"{display_frequency:.6f}",
                    f"{s11_db:.3f}",
                    delta_s11_db,
                    f"{abs(parameter):.4f}",
                    delta_s11_mag,
                    f"{np.degrees(np.angle(parameter)):.2f}",
                    _format_impedance(impedance),
                    *area_texts,
                ]
                sort_values = [
                    None,
                    display_frequency,
                    s11_db,
                    None if reference_s11_db is None else s11_db - reference_s11_db,
                    abs(parameter),
                    None if reference_s11_mag is None else abs(parameter) - reference_s11_mag,
                    float(np.degrees(np.angle(parameter))),
                    None,
                    *area_values,
                ]

            self._set_table_row(
                self.marker_table,
                row_index,
                trace.color,
                values,
                background_colors=column_backgrounds,
                sort_values=sort_values,
                bold=reference_trace is not None and trace.data.path == reference_trace.data.path,
            )

        self._end_table_update(self.marker_table, sort_column, sort_order)

    def _update_s21_marker_table(self) -> None:
        visible_traces = self._visible_traces()
        reference_trace = self._reference_trace()
        reference_parameter = None
        if reference_trace is not None and self.marker_frequency_hz is not None:
            reference_parameter = reference_trace.data.interpolated_parameter(2, 1, self.marker_frequency_hz)

        sort_column, sort_order = self._begin_table_update(self.s21_marker_table)
        if not self._marker_overlay_enabled():
            self.s21_marker_table.setRowCount(0)
            for trace in visible_traces:
                self._set_scatter_point(trace.s21_marker, None)
            self._end_table_update(self.s21_marker_table, sort_column, sort_order)
            return

        self.s21_marker_table.setRowCount(len(visible_traces))

        for row_index, trace in enumerate(visible_traces):
            if not trace.data.has_parameter(2, 1):
                self._set_scatter_point(trace.s21_marker, None)
                values = [
                    self._display_name_for_trace(trace),
                    "not available",
                    "not available",
                    "not available",
                    "not available",
                    "not available",
                    "not available",
                ]
                self._set_table_row(
                    self.s21_marker_table,
                    row_index,
                    trace.color,
                    values,
                    sort_values=[None] * len(values),
                    bold=reference_trace is not None and trace.data.path == reference_trace.data.path,
                )
                continue

            parameter = None
            if self.marker_frequency_hz is not None:
                parameter = trace.data.interpolated_parameter(2, 1, self.marker_frequency_hz)

            if parameter is None:
                self._set_scatter_point(trace.s21_marker, None)
                values = [
                    self._display_name_for_trace(trace),
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                ]
                sort_values = [None] * len(values)
            else:
                display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
                s21_db = _parameter_db(parameter)
                reference_s21_db = (
                    None if reference_parameter is None else _parameter_db(reference_parameter)
                )
                reference_s21_mag = None if reference_parameter is None else abs(reference_parameter)
                delta_s21_db = _format_optional_delta(
                    s21_db,
                    reference_s21_db,
                    precision=3,
                )
                delta_s21_mag = _format_optional_delta(
                    abs(parameter),
                    reference_s21_mag,
                    precision=4,
                )
                self._set_scatter_point(trace.s21_marker, (display_frequency, s21_db))
                values = [
                    self._display_name_for_trace(trace),
                    f"{display_frequency:.6f}",
                    f"{s21_db:.3f}",
                    delta_s21_db,
                    f"{abs(parameter):.4f}",
                    delta_s21_mag,
                    f"{np.degrees(np.angle(parameter)):.2f}",
                ]
                sort_values = [
                    None,
                    display_frequency,
                    s21_db,
                    None if reference_s21_db is None else s21_db - reference_s21_db,
                    abs(parameter),
                    None if reference_s21_mag is None else abs(parameter) - reference_s21_mag,
                    float(np.degrees(np.angle(parameter))),
                ]

            self._set_table_row(
                self.s21_marker_table,
                row_index,
                trace.color,
                values,
                sort_values=sort_values,
                bold=reference_trace is not None and trace.data.path == reference_trace.data.path,
            )

        self._end_table_update(self.s21_marker_table, sort_column, sort_order)

    def _update_matching_marker_outputs(self) -> None:
        if self.match_original_gamma is None or self.match_transformed_gamma is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Load a trace to preview a matching network.")
            return

        marker_visible = self._marker_overlay_enabled()
        if self.match_s11_marker_line is not None:
            self.match_s11_marker_line.setVisible(marker_visible)
        if self.match_marker_plot_label is not None:
            self.match_marker_plot_label.setVisible(marker_visible)

        trace = self._matching_trace()
        if trace is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Load a trace to preview a matching network.")
            return

        frequency_hz = self._matching_target_frequency_hz()
        if frequency_hz is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Select a trace to evaluate the matching network.")
            return

        self.match_marker_frequency_hz = frequency_hz
        if self.match_s11_marker_line is not None:
            self.match_s11_marker_line.setValue(frequency_hz / self.frequency_scale.factor_hz)

        original_gamma = trace.data.interpolated_gamma(frequency_hz)
        if original_gamma is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Target frequency is outside the selected trace range.")
            return

        original_impedance = gamma_to_impedance(original_gamma, trace.data.reference_impedance_ohms)
        transformed_impedance = self._matched_impedance_at_frequency(frequency_hz)
        if transformed_impedance is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            self.match_summary_label.setText("Target frequency is outside the selected trace range.")
            return

        transformed_gamma = impedance_to_gamma(
            np.asarray([transformed_impedance], dtype=np.complex128),
            trace.data.reference_impedance_ohms,
        )[0]
        display_frequency = frequency_hz / self.frequency_scale.factor_hz
        original_s11_db = _parameter_db(original_gamma)
        transformed_s11_db = _parameter_db(transformed_gamma)

        if marker_visible:
            self._set_scatter_point(
                self.match_s11_original_marker,
                (display_frequency, original_s11_db),
            )
            self._set_scatter_point(
                self.match_s11_transformed_marker,
                (display_frequency, transformed_s11_db),
            )
            self._set_scatter_point(
                self.match_smith_original_marker,
                (original_gamma.real, original_gamma.imag),
            )
            self._set_scatter_point(
                self.match_smith_transformed_marker,
                (transformed_gamma.real, transformed_gamma.imag),
            )
            if self.match_marker_plot_label is not None:
                self.match_marker_plot_label.setText(
                    f"Marker: {display_frequency:.6f} {self.frequency_scale.unit}"
                )
                self._update_match_marker_plot_label_position()
        else:
            self._set_scatter_point(self.match_s11_original_marker, None)
            self._set_scatter_point(self.match_s11_transformed_marker, None)
            self._set_scatter_point(self.match_smith_original_marker, None)
            self._set_scatter_point(self.match_smith_transformed_marker, None)
            if self.match_marker_plot_label is not None:
                self.match_marker_plot_label.setText("Marker: n/a")

        self.match_summary_label.setText(
            " | ".join(
                [
                    f"At {display_frequency:.6f} {self.frequency_scale.unit}",
                    f"Original {original_s11_db:.3f} dB, Z={_format_impedance(original_impedance)}",
                    f"Matched {transformed_s11_db:.3f} dB, Z={_format_impedance(transformed_impedance)}",
                    f"Delta {_format_signed(transformed_s11_db - original_s11_db, 3)} dB",
                ]
            )
        )
        self._update_matching_suggestions(
            transformed_impedance=transformed_impedance,
            transformed_s11_db=transformed_s11_db,
            target_frequency_hz=frequency_hz,
        )

    def _matched_impedance_at_frequency(self, frequency_hz: float) -> complex | None:
        trace = self._matching_trace()
        if trace is None:
            return None

        original_gamma = trace.data.interpolated_gamma(frequency_hz)
        if original_gamma is None:
            return None

        original_impedance = gamma_to_impedance(original_gamma, trace.data.reference_impedance_ohms)
        matched_impedance = apply_matching_network(
            np.asarray([original_impedance], dtype=np.complex128),
            np.asarray([frequency_hz], dtype=np.float64),
            self._matching_stages(),
        )[0]
        if not np.isfinite(matched_impedance.real) or not np.isfinite(matched_impedance.imag):
            return None
        return matched_impedance

    def _update_matching_suggestions(
        self,
        *,
        transformed_impedance: complex,
        transformed_s11_db: float,
        target_frequency_hz: float,
    ) -> None:
        trace = self._matching_trace()
        if trace is None:
            self.matching_suggestions = []
            self.match_suggestion_table.setRowCount(0)
            self.match_suggestion_label.setText(
                "Suggestions evaluate one additional reactive stage (L/C only) at the target frequency."
            )
            return

        self.matching_suggestions = suggest_matching_stages(
            transformed_impedance,
            target_frequency_hz,
            trace.data.reference_impedance_ohms,
        )
        self.match_suggestion_table.setRowCount(len(self.matching_suggestions))

        for row_index, suggestion in enumerate(self.matching_suggestions):
            resulting_s11_db = _parameter_db(suggestion.resulting_gamma)
            values = [
                f"{suggestion.stage.topology} {suggestion.stage.component}",
                f"{suggestion.stage.value:g} {suggestion.stage.unit}",
                f"{resulting_s11_db:.3f}",
                f"{transformed_s11_db - resulting_s11_db:.3f}",
                _format_impedance(suggestion.resulting_impedance_ohms),
            ]
            sort_values = [
                None,
                suggestion.stage.value_si,
                resulting_s11_db,
                transformed_s11_db - resulting_s11_db,
                None,
            ]
            self._set_table_row(
                self.match_suggestion_table,
                row_index,
                "#1e293b",
                values,
                sort_values=sort_values,
            )

        if self.matching_suggestions:
            self.match_suggestion_table.selectRow(0)

        display_frequency = target_frequency_hz / self.frequency_scale.factor_hz
        self.match_suggestion_label.setText(
            " ".join(
                [
                    f"Suggestions are single reactive next stages at {display_frequency:.6f} {self.frequency_scale.unit}.",
                    "They append at the bottom, on the coax/feed side of the current network.",
                ]
            )
        )

    def _set_scatter_point(
        self,
        scatter: pg.ScatterPlotItem | None,
        point: tuple[float, float] | None,
    ) -> None:
        if scatter is None:
            return
        if point is None:
            scatter.setData([], [])
            return
        scatter.setData([point[0]], [point[1]])

    def _set_table_row(
        self,
        table: QtWidgets.QTableWidget,
        row_index: int,
        color_hex: str,
        values: list[str],
        *,
        background_colors: list[str | None] | None = None,
        sort_values: list[float | None] | None = None,
        bold: bool = False,
    ) -> None:
        color = QtGui.QColor(color_hex)
        for column, value in enumerate(values):
            item = _SortableTableWidgetItem(value)
            item.setForeground(QtGui.QBrush(color))
            if background_colors is not None and column < len(background_colors):
                background_color = background_colors[column]
                if background_color:
                    item.setBackground(_table_tint_brush(background_color))
            if sort_values is not None and column < len(sort_values) and sort_values[column] is not None:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, float(sort_values[column]))
            if bold:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            table.setItem(row_index, column, item)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.load_files(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._config_save_timer.isActive():
            self._config_save_timer.stop()
        self._write_user_config()
        super().closeEvent(event)


def _parameter_db(parameter: complex) -> float:
    return 20.0 * np.log10(max(abs(parameter), 1.0e-12))


def _format_signed(value: float, precision: int) -> str:
    return f"{value:+.{precision}f}"


def _format_optional_delta(
    value: float,
    reference_value: float | None,
    *,
    precision: int,
) -> str:
    if reference_value is None:
        return "-"
    return _format_signed(value - reference_value, precision)


def _format_impedance(value: complex) -> str:
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        return "open"
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{value.real:.2f} {sign} j{abs(value.imag):.2f}"


def _natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def _trace_sort_key(trace: LoadedTrace) -> tuple[tuple[tuple[int, int | str], ...], str]:
    return (_natural_sort_key(trace.data.label), str(trace.data.path).casefold())


def _sorted_frequency_region_hz(region_hz: tuple[float, float]) -> tuple[float, float]:
    start_hz, stop_hz = region_hz
    return (float(min(start_hz, stop_hz)), float(max(start_hz, stop_hz)))


def _default_frequency_region_hz(bounds_hz: tuple[float, float]) -> tuple[float, float]:
    start_hz, stop_hz = sorted(bounds_hz)
    span_hz = stop_hz - start_hz
    if span_hz <= 0.0:
        return (start_hz, stop_hz)

    region_width_hz = span_hz * 0.22
    region_center_hz = 0.5 * (start_hz + stop_hz)
    return (
        region_center_hz - 0.5 * region_width_hz,
        region_center_hz + 0.5 * region_width_hz,
    )


def _clamp_frequency_region_hz(
    region_hz: tuple[float, float],
    bounds_hz: tuple[float, float],
) -> tuple[float, float]:
    bound_start_hz, bound_stop_hz = sorted(bounds_hz)
    region_start_hz, region_stop_hz = sorted(region_hz)

    available_span_hz = bound_stop_hz - bound_start_hz
    if available_span_hz <= 0.0:
        return (bound_start_hz, bound_stop_hz)

    requested_span_hz = min(region_stop_hz - region_start_hz, available_span_hz)
    clamped_start_hz = min(
        max(region_start_hz, bound_start_hz),
        bound_stop_hz - requested_span_hz,
    )
    return (clamped_start_hz, clamped_start_hz + requested_span_hz)


def _intersect_frequency_region_hz(
    region_hz: tuple[float, float],
    bounds_hz: tuple[float, float],
) -> tuple[float, float] | None:
    bound_start_hz, bound_stop_hz = sorted(bounds_hz)
    region_start_hz, region_stop_hz = sorted(region_hz)
    visible_start_hz = max(bound_start_hz, region_start_hz)
    visible_stop_hz = min(bound_stop_hz, region_stop_hz)
    if visible_stop_hz < visible_start_hz:
        return None
    return (visible_start_hz, visible_stop_hz)


def _table_tint_brush(color_hex: str, *, alpha: int = 48) -> QtGui.QBrush:
    color = QtGui.QColor(color_hex)
    color.setAlpha(alpha)
    return QtGui.QBrush(color)


def _resolve_browser_directory(
    saved_directory: str | Path | None,
    downloads_directory: str | Path | None,
    home_directory: str | Path | None,
) -> Path:
    for candidate in (saved_directory, downloads_directory, home_directory, Path.home()):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_dir():
            return path.resolve()
    return Path.cwd()


def _viewer_config_path() -> Path:
    base_directory = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    ).expanduser()
    return base_directory / "touchstone_viewer" / "config.yaml"


def _build_light_mode_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    disabled_text = QtGui.QColor("#94a3b8")

    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(LIGHT_MODE_WINDOW_COLOR))
    palette.setColor(
        QtGui.QPalette.ColorRole.WindowText,
        QtGui.QColor(LIGHT_MODE_WINDOW_TEXT_COLOR),
    )
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(LIGHT_MODE_BASE_COLOR))
    palette.setColor(
        QtGui.QPalette.ColorRole.AlternateBase,
        QtGui.QColor(LIGHT_MODE_ALTERNATE_BASE_COLOR),
    )
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(LIGHT_MODE_BASE_COLOR))
    palette.setColor(
        QtGui.QPalette.ColorRole.ToolTipText,
        QtGui.QColor(LIGHT_MODE_WINDOW_TEXT_COLOR),
    )
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(LIGHT_MODE_WINDOW_TEXT_COLOR))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(LIGHT_MODE_BUTTON_COLOR))
    palette.setColor(
        QtGui.QPalette.ColorRole.ButtonText,
        QtGui.QColor(LIGHT_MODE_WINDOW_TEXT_COLOR),
    )
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#2563eb"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(LIGHT_MODE_HIGHLIGHT_COLOR))
    palette.setColor(
        QtGui.QPalette.ColorRole.HighlightedText,
        QtGui.QColor(LIGHT_MODE_WINDOW_TEXT_COLOR),
    )
    palette.setColor(
        QtGui.QPalette.ColorRole.PlaceholderText,
        QtGui.QColor(LIGHT_MODE_PLACEHOLDER_TEXT_COLOR),
    )
    palette.setColor(QtGui.QPalette.ColorRole.Light, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#cbd5e1"))
    palette.setColor(QtGui.QPalette.ColorRole.Dark, QtGui.QColor("#94a3b8"))
    palette.setColor(QtGui.QPalette.ColorRole.Shadow, QtGui.QColor("#64748b"))

    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.WindowText,
        disabled_text,
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.Text,
        disabled_text,
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.ButtonText,
        disabled_text,
    )
    palette.setColor(
        QtGui.QPalette.ColorGroup.Disabled,
        QtGui.QPalette.ColorRole.HighlightedText,
        disabled_text,
    )
    return palette


def _apply_application_appearance(force_light_mode: bool) -> None:
    global _DEFAULT_APPLICATION_PALETTE

    app = QtWidgets.QApplication.instance()
    if app is None:
        return

    app.setStyle("Fusion")
    if _DEFAULT_APPLICATION_PALETTE is None:
        _DEFAULT_APPLICATION_PALETTE = QtGui.QPalette(app.palette())

    if force_light_mode:
        app.setPalette(_build_light_mode_palette())
    else:
        app.setPalette(QtGui.QPalette(_DEFAULT_APPLICATION_PALETTE))

    pg.setConfigOptions(
        antialias=True,
        background=LIGHT_MODE_WINDOW_COLOR,
        foreground="#24313f",
    )


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


def _parse_yaml_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith('"'):
        return json.loads(text)
    try:
        if any(character in text for character in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_yaml_key(text: str) -> str:
    parsed = _parse_yaml_scalar(text)
    return str(parsed)


def _parse_viewer_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_preset_name: str | None = None

    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line_without_comment.strip()

        if indent == 0:
            current_preset_name = None
            if stripped.endswith(":"):
                current_section = _parse_yaml_key(stripped[:-1].strip())
                if current_section == "aoi_presets":
                    data[current_section] = {}
                continue

            key_text, value_text = stripped.split(":", maxsplit=1)
            data[_parse_yaml_key(key_text.strip())] = _parse_yaml_scalar(value_text.strip())
            current_section = None
            continue

        if indent == 2 and current_section == "aoi_presets" and stripped.endswith(":"):
            current_preset_name = _parse_yaml_key(stripped[:-1].strip())
            data["aoi_presets"][current_preset_name] = {}
            continue

        if indent == 4 and current_section == "aoi_presets" and current_preset_name is not None:
            key_text, value_text = stripped.split(":", maxsplit=1)
            data["aoi_presets"][current_preset_name][
                _parse_yaml_key(key_text.strip())
            ] = _parse_yaml_scalar(value_text.strip())

    return data


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_viewer_user_config() -> ViewerUserConfig:
    config_path = _viewer_config_path()
    if not config_path.exists():
        return ViewerUserConfig()

    try:
        raw_config = _parse_viewer_yaml(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return ViewerUserConfig()

    config = ViewerUserConfig()

    frequency_unit_mode = raw_config.get("frequency_unit_mode")
    if frequency_unit_mode in {DEFAULT_FREQUENCY_UNIT_MODE, *FREQUENCY_UNIT_FACTORS_HZ.keys()}:
        config.frequency_unit_mode = str(frequency_unit_mode)

    force_light_mode = raw_config.get("force_light_mode")
    if isinstance(force_light_mode, bool):
        config.force_light_mode = force_light_mode

    controls_visible = raw_config.get("controls_visible")
    if isinstance(controls_visible, bool):
        config.controls_visible = controls_visible

    aoi_visible = raw_config.get("aoi_visible")
    if isinstance(aoi_visible, bool):
        config.aoi_visible = aoi_visible

    config.aoi_start_hz = _float_or_none(raw_config.get("aoi_start_hz"))
    config.aoi_stop_hz = _float_or_none(raw_config.get("aoi_stop_hz"))

    aoi_unit = raw_config.get("aoi_unit")
    if aoi_unit in AOI_UNIT_FACTORS_HZ:
        config.aoi_unit = str(aoi_unit)

    marker_visible = raw_config.get("marker_visible")
    if isinstance(marker_visible, bool):
        config.marker_visible = marker_visible

    config.marker_frequency_hz = _float_or_none(raw_config.get("marker_frequency_hz"))

    threshold_visible = raw_config.get("threshold_visible")
    if isinstance(threshold_visible, bool):
        config.threshold_visible = threshold_visible

    threshold_db = _float_or_none(raw_config.get("threshold_db"))
    if threshold_db is not None:
        config.threshold_db = threshold_db

    raw_presets = raw_config.get("aoi_presets")
    if isinstance(raw_presets, dict):
        for preset_name, preset_data in raw_presets.items():
            if not isinstance(preset_name, str) or not isinstance(preset_data, dict):
                continue
            start_hz = _float_or_none(preset_data.get("start_hz"))
            stop_hz = _float_or_none(preset_data.get("stop_hz"))
            if start_hz is None or stop_hz is None:
                continue
            preset_unit = preset_data.get("unit")
            if preset_unit not in AOI_UNIT_FACTORS_HZ:
                preset_unit = DEFAULT_AOI_UNIT
            preset_marker_frequency_hz = _float_or_none(preset_data.get("marker_frequency_hz"))
            config.aoi_presets[preset_name] = AoiPreset(
                start_hz=start_hz,
                stop_hz=stop_hz,
                unit=str(preset_unit),
                marker_frequency_hz=preset_marker_frequency_hz,
            )

    selected_aoi_preset = raw_config.get("selected_aoi_preset")
    if isinstance(selected_aoi_preset, str) and selected_aoi_preset in config.aoi_presets:
        config.selected_aoi_preset = selected_aoi_preset

    return config


def _write_viewer_user_config(config: ViewerUserConfig) -> None:
    config_path = _viewer_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Touchstone Viewer user configuration",
        f"frequency_unit_mode: {_yaml_scalar(config.frequency_unit_mode)}",
        f"force_light_mode: {_yaml_scalar(config.force_light_mode)}",
        f"controls_visible: {_yaml_scalar(config.controls_visible)}",
        f"aoi_visible: {_yaml_scalar(config.aoi_visible)}",
        f"aoi_start_hz: {_yaml_scalar(config.aoi_start_hz)}",
        f"aoi_stop_hz: {_yaml_scalar(config.aoi_stop_hz)}",
        f"aoi_unit: {_yaml_scalar(config.aoi_unit)}",
        f"marker_visible: {_yaml_scalar(config.marker_visible)}",
        f"marker_frequency_hz: {_yaml_scalar(config.marker_frequency_hz)}",
        f"threshold_visible: {_yaml_scalar(config.threshold_visible)}",
        f"threshold_db: {_yaml_scalar(config.threshold_db)}",
        f"selected_aoi_preset: {_yaml_scalar(config.selected_aoi_preset)}",
        "aoi_presets:",
    ]
    for preset_name in sorted(config.aoi_presets, key=_natural_sort_key):
        preset = config.aoi_presets[preset_name]
        lines.extend(
            [
                f"  {_yaml_scalar(preset_name)}:",
                f"    start_hz: {_yaml_scalar(preset.start_hz)}",
                f"    stop_hz: {_yaml_scalar(preset.stop_hz)}",
                f"    unit: {_yaml_scalar(preset.unit)}",
                f"    marker_frequency_hz: {_yaml_scalar(preset.marker_frequency_hz)}",
            ]
        )

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive S11 and S21 viewer for Touchstone .s1p and .s2p files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Touchstone .s1p or .s2p files to open",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _apply_application_appearance(DEFAULT_FORCE_LIGHT_MODE)

    window = TouchstoneViewerWindow(args.files)
    window.showMaximized()
    return app.exec()
