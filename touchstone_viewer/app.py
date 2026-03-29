from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .smith import add_smith_grid
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
DEFAULT_AOI_VISIBLE = True
DEFAULT_MARKER_VISIBLE = True
DEFAULT_FREQUENCY_UNIT_MODE = "Auto"


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


class _SortableTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other: QtWidgets.QTableWidgetItem) -> bool:
        left_value = self.data(QtCore.Qt.ItemDataRole.UserRole)
        right_value = other.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            return float(left_value) < float(right_value)
        return self.text() < other.text()


class TouchstoneViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_paths: Sequence[Path]) -> None:
        super().__init__()
        self.setWindowTitle("Touchstone Viewer")
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings("TouchstoneViewer", "Touch")

        self.traces: list[LoadedTrace] = []
        self.frequency_scale = FrequencyScale(1.0e6, "MHz")
        self.frequency_unit_mode = DEFAULT_FREQUENCY_UNIT_MODE
        self.reference_trace_path: Path | None = None
        self.marker_frequency_hz: float | None = None
        self.marker_line: pg.InfiniteLine | None = None
        self.s21_marker_line: pg.InfiniteLine | None = None
        self.marker_plot_label: pg.TextItem | None = None
        self.s21_marker_plot_label: pg.TextItem | None = None
        self.aoi_region_hz: tuple[float, float] | None = None
        self.aoi_region_item: pg.LinearRegionItem | None = None
        self.s11_threshold_line: pg.InfiniteLine | None = None
        self.s21_threshold_line: pg.InfiniteLine | None = None
        self._updating_marker = False
        self._updating_marker_controls = False
        self._updating_aoi_controls = False
        self._updating_trace_controls = False

        self._build_ui()
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

        header.addSpacing(12)

        self.summary_label = QtWidgets.QLabel(
            "Drop .s1p or .s2p files here or open them from the dialog."
        )
        header.addWidget(self.summary_label, stretch=1)

        self.controls_toggle_button = QtWidgets.QToolButton()
        self.controls_toggle_button.setText("Controls")
        self.controls_toggle_button.setCheckable(True)
        self.controls_toggle_button.setChecked(DEFAULT_CONTROLS_VISIBLE)
        self.controls_toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.controls_toggle_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.controls_toggle_button.toggled.connect(self._set_controls_panel_visible)
        header.addWidget(self.controls_toggle_button)

        layout.addLayout(header)

        self.controls_panel = self._build_controls_panel()
        layout.addWidget(self.controls_panel)
        self._set_controls_panel_visible(DEFAULT_CONTROLS_VISIBLE)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.addTab(self._build_s11_tab(), "S11")
        self.tab_widget.addTab(self._build_s21_tab(), "S21")
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

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)

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
        self.frequency_unit_combo.setCurrentText(DEFAULT_FREQUENCY_UNIT_MODE)
        self.frequency_unit_combo.currentTextChanged.connect(self._handle_frequency_unit_changed)
        view_layout.addWidget(self.frequency_unit_combo)

        aoi_section = QtWidgets.QWidget()
        aoi_layout = QtWidgets.QHBoxLayout(aoi_section)
        aoi_layout.setContentsMargins(0, 0, 0, 0)
        aoi_layout.setSpacing(8)

        aoi_prefix = QtWidgets.QLabel("Area of Interest")
        aoi_layout.addWidget(aoi_prefix)
        self.aoi_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.aoi_enabled_checkbox.setChecked(DEFAULT_AOI_VISIBLE)
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

        self.aoi_unit_combo = QtWidgets.QComboBox()
        self.aoi_unit_combo.addItems(AOI_UNIT_FACTORS_HZ.keys())
        self.aoi_unit_combo.setCurrentText("GHz")
        self.aoi_unit_combo.currentTextChanged.connect(self._handle_aoi_unit_changed)
        aoi_layout.addWidget(self.aoi_unit_combo)

        top_row.addWidget(view_section)
        top_row.addWidget(self._build_panel_separator())
        top_row.addWidget(aoi_section)
        top_row.addWidget(self._build_panel_separator())

        threshold_section = QtWidgets.QWidget()
        threshold_layout = QtWidgets.QHBoxLayout(threshold_section)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(8)

        threshold_prefix = QtWidgets.QLabel("Threshold")
        threshold_layout.addWidget(threshold_prefix)

        self.threshold_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.threshold_enabled_checkbox.setChecked(DEFAULT_THRESHOLD_VISIBLE)
        self.threshold_enabled_checkbox.toggled.connect(self._handle_threshold_visibility_changed)
        threshold_layout.addWidget(self.threshold_enabled_checkbox)

        self.threshold_input = QtWidgets.QDoubleSpinBox()
        self.threshold_input.setDecimals(2)
        self.threshold_input.setRange(0.0, 200.0)
        self.threshold_input.setSingleStep(0.5)
        self.threshold_input.setValue(DEFAULT_THRESHOLD_DB)
        self.threshold_input.setSuffix(" dB")
        self.threshold_input.setMinimumWidth(95)
        self.threshold_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.threshold_input.valueChanged.connect(self._handle_threshold_changed)
        self.threshold_input.setEnabled(DEFAULT_THRESHOLD_VISIBLE)
        threshold_layout.addWidget(self.threshold_input)

        top_row.addWidget(threshold_section)
        top_row.addWidget(self._build_panel_separator())

        marker_section = QtWidgets.QWidget()
        marker_layout = QtWidgets.QHBoxLayout(marker_section)
        marker_layout.setContentsMargins(0, 0, 0, 0)
        marker_layout.setSpacing(8)

        marker_prefix = QtWidgets.QLabel("Marker")
        marker_layout.addWidget(marker_prefix)

        self.marker_enabled_checkbox = QtWidgets.QCheckBox("Show")
        self.marker_enabled_checkbox.setChecked(DEFAULT_MARKER_VISIBLE)
        self.marker_enabled_checkbox.toggled.connect(self._handle_marker_visibility_changed)
        marker_layout.addWidget(self.marker_enabled_checkbox)

        marker_at_label = QtWidgets.QLabel("At")
        marker_layout.addWidget(marker_at_label)

        self.marker_frequency_input = QtWidgets.QDoubleSpinBox()
        self.marker_frequency_input.setDecimals(6)
        self.marker_frequency_input.setMinimumWidth(120)
        self.marker_frequency_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.marker_frequency_input.setKeyboardTracking(False)
        self.marker_frequency_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.marker_frequency_input.valueChanged.connect(self._handle_marker_frequency_changed)
        marker_layout.addWidget(self.marker_frequency_input)

        top_row.addWidget(marker_section)
        top_row.addStretch(1)
        layout.addLayout(top_row)

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
            self._update_marker_plot_label_position
        )
        splitter.addWidget(self.s11_plot)

        self.smith_plot = pg.PlotWidget()
        splitter.addWidget(self.smith_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

        self.marker_table = self._build_marker_table(
            [
                "Trace",
                "Freq",
                "S11 (dB)",
                "ΔRef (dB)",
                "|S11|",
                "ΔRef |S11|",
                "Angle (deg)",
                "Z (ohm)",
            ]
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
            ["Trace", "Freq", "S21 (dB)", "ΔRef (dB)", "|S21|", "ΔRef |S21|", "Angle (deg)"]
        )
        layout.addWidget(self.s21_marker_table)

        return tab

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

    def _build_aoi_spin_box(self) -> QtWidgets.QDoubleSpinBox:
        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setDecimals(6)
        spin_box.setRange(0.0, 0.0)
        spin_box.setValue(0.0)
        spin_box.setMinimumWidth(95)
        spin_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        spin_box.setKeyboardTracking(False)
        spin_box.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        return spin_box

    def _set_controls_panel_visible(self, visible: bool) -> None:
        self.controls_panel.setVisible(visible)
        self.controls_toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        )

    def _refresh_plots(self) -> None:
        visible_traces = self._visible_traces()
        self._choose_frequency_scale()

        self.s11_plot.clear()
        self.smith_plot.clear()
        self.s21_plot.clear()
        self.marker_plot_label = None
        self.s21_marker_plot_label = None
        self.aoi_region_item = None
        self.s11_threshold_line = None
        self.s21_threshold_line = None

        self._configure_s11_plot()
        self._configure_smith_plot()
        self._configure_s21_plot()

        if visible_traces:
            self._add_aoi_region()
            self._sync_aoi_controls_to_region()

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
            self.marker_frequency_hz = None
            self.aoi_region_hz = None
            self._reset_aoi_controls()

        self._sync_trace_controls()
        self._sync_control_states()
        self._update_marker_outputs()
        self.summary_label.setText(
            f"{len(self.traces)} trace(s) loaded, {len(visible_traces)} visible"
        )

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
        self.smith_plot.setMouseEnabled(x=False, y=False)
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

    def _add_aoi_region(self) -> None:
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            return

        if self.aoi_region_hz is None:
            start_hz, stop_hz = _default_frequency_region_hz(
                self._preferred_aoi_span_hz(bounds_hz)
            )
        else:
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
            pen=pg.mkPen("#60a5fa", width=1.8),
            hoverBrush=pg.mkBrush(96, 165, 250, 104),
            hoverPen=pg.mkPen("#3b82f6", width=2.0),
            movable=False,
            bounds=scaled_bounds,
        )
        self.aoi_region_item.setZValue(-5)
        self.s11_plot.addItem(self.aoi_region_item, ignoreBounds=True)

        self.aoi_region_hz = (start_hz, stop_hz)

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
            if self.marker_frequency_hz is None:
                self.marker_frequency_hz = self._default_marker_frequency()
            self._refresh_plots()

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

    def _handle_s11_marker_moved(self) -> None:
        if not self.marker_line or self._updating_marker or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = self.marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_s21_marker_moved(self) -> None:
        if not self.s21_marker_line or self._updating_marker or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = self.s21_marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_aoi_value_changed(self, _value: float) -> None:
        if self._updating_aoi_controls or not self._visible_traces():
            return

        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            return

        factor_hz = self._aoi_unit_factor_hz()
        start_hz = self.aoi_start_input.value() * factor_hz
        stop_hz = self.aoi_stop_input.value() * factor_hz
        self.aoi_region_hz = _clamp_frequency_region_hz((start_hz, stop_hz), bounds_hz)
        self._sync_aoi_controls_to_region()
        self._update_aoi_region_item()

    def _handle_aoi_unit_changed(self, _unit: str) -> None:
        if self._updating_aoi_controls:
            return

        if self._visible_traces():
            self._sync_aoi_controls_to_region()
        else:
            self._reset_aoi_controls()

    def _handle_frequency_unit_changed(self, unit: str) -> None:
        self.frequency_unit_mode = unit
        self._refresh_plots()

    def _handle_marker_frequency_changed(self, value: float) -> None:
        if self._updating_marker_controls or not self._marker_overlay_enabled():
            return

        self.marker_frequency_hz = value * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_aoi_visibility_changed(self, _visible: bool) -> None:
        self._sync_control_states()

    def _handle_threshold_changed(self, _value: float) -> None:
        self._update_threshold_lines()

    def _handle_threshold_visibility_changed(self, visible: bool) -> None:
        self.threshold_input.setEnabled(visible)
        self._update_threshold_lines()

    def _handle_marker_visibility_changed(self, _visible: bool) -> None:
        self._sync_control_states()
        self._update_marker_outputs()

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
        return AOI_UNIT_FACTORS_HZ[self.aoi_unit_combo.currentText()]

    def _aoi_overlay_enabled(self) -> bool:
        return bool(self._visible_traces()) and self.aoi_enabled_checkbox.isChecked()

    def _marker_overlay_enabled(self) -> bool:
        return bool(self._visible_traces()) and self.marker_enabled_checkbox.isChecked()

    def _sync_control_states(self) -> None:
        has_loaded_traces = bool(self.traces)
        has_visible_traces = bool(self._visible_traces())
        self.aoi_enabled_checkbox.setEnabled(has_loaded_traces)
        self.marker_enabled_checkbox.setEnabled(has_loaded_traces)
        self._set_aoi_controls_enabled(self._aoi_overlay_enabled())
        self._sync_marker_frequency_input()
        self._update_aoi_region_visibility()
        self._update_marker_visibility()
        if not has_visible_traces:
            self._reset_aoi_controls()

    def _set_aoi_controls_enabled(self, enabled: bool) -> None:
        self.aoi_start_input.setEnabled(enabled)
        self.aoi_stop_input.setEnabled(enabled)
        self.aoi_unit_combo.setEnabled(enabled)

    def _update_aoi_region_visibility(self) -> None:
        if self.aoi_region_item is None:
            return

        self.aoi_region_item.setVisible(self._aoi_overlay_enabled())

    def _reset_aoi_controls(self) -> None:
        self._updating_aoi_controls = True
        try:
            for spin_box in (self.aoi_start_input, self.aoi_stop_input):
                spin_box.setRange(0.0, 0.0)
                spin_box.setSingleStep(0.001)
                spin_box.setValue(0.0)
        finally:
            self._updating_aoi_controls = False

    def _sync_aoi_controls_to_region(self) -> None:
        bounds_hz = self._frequency_span_hz()
        if bounds_hz is None:
            self._reset_aoi_controls()
            return

        if self.aoi_region_hz is None:
            self.aoi_region_hz = _default_frequency_region_hz(
                self._preferred_aoi_span_hz(bounds_hz)
            )
        else:
            self.aoi_region_hz = _clamp_frequency_region_hz(self.aoi_region_hz, bounds_hz)

        factor_hz = self._aoi_unit_factor_hz()
        display_bounds = (bounds_hz[0] / factor_hz, bounds_hz[1] / factor_hz)
        display_region = (
            self.aoi_region_hz[0] / factor_hz,
            self.aoi_region_hz[1] / factor_hz,
        )

        step_size = max((display_bounds[1] - display_bounds[0]) / 200.0, 1.0e-6)

        self._updating_aoi_controls = True
        try:
            for spin_box in (self.aoi_start_input, self.aoi_stop_input):
                spin_box.setRange(display_bounds[0], display_bounds[1])
                spin_box.setSingleStep(step_size)
            self.aoi_start_input.setValue(display_region[0])
            self.aoi_stop_input.setValue(display_region[1])
        finally:
            self._updating_aoi_controls = False

    def _update_aoi_region_item(self) -> None:
        if self.aoi_region_item is None or self.aoi_region_hz is None:
            return

        self.aoi_region_item.setRegion(
            [
                self.aoi_region_hz[0] / self.frequency_scale.factor_hz,
                self.aoi_region_hz[1] / self.frequency_scale.factor_hz,
            ]
        )

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

        for marker_table in (self.marker_table, self.s21_marker_table):
            marker_table.setHidden(not visible)

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
        self._update_marker_plot_label()
        self._update_s21_marker_plot_label()
        self._update_marker_table()
        self._update_s21_marker_table()

    def _sync_marker_frequency_input(self) -> None:
        bounds_hz = self._frequency_span_hz()
        visible = self._marker_overlay_enabled() and bounds_hz is not None
        self._updating_marker_controls = True
        try:
            self.marker_frequency_input.setEnabled(visible)
            self.marker_frequency_input.setSuffix(f" {self.frequency_scale.unit}")
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
        table.sortItems(sort_column, sort_order)
        table.resizeRowsToContents()

    def _update_marker_table(self) -> None:
        visible_traces = self._visible_traces()
        reference_trace = self._reference_trace()
        reference_parameter = None
        if reference_trace is not None and self.marker_frequency_hz is not None:
            reference_parameter = reference_trace.data.interpolated_parameter(1, 1, self.marker_frequency_hz)

        sort_column, sort_order = self._begin_table_update(self.marker_table)
        if not self._marker_overlay_enabled():
            self.marker_table.setRowCount(0)
            for trace in visible_traces:
                self._set_scatter_point(trace.s11_marker, None)
                self._set_scatter_point(trace.smith_marker, None)
            self._end_table_update(self.marker_table, sort_column, sort_order)
            return

        self.marker_table.setRowCount(len(visible_traces))

        for row_index, trace in enumerate(visible_traces):
            parameter = None
            if self.marker_frequency_hz is not None:
                parameter = trace.data.interpolated_parameter(1, 1, self.marker_frequency_hz)

            if parameter is None:
                self._set_scatter_point(trace.s11_marker, None)
                self._set_scatter_point(trace.smith_marker, None)
                values = [
                    self._display_name_for_trace(trace),
                    "out of range",
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
                self._set_scatter_point(trace.s11_marker, (display_frequency, s11_db))
                self._set_scatter_point(trace.smith_marker, (parameter.real, parameter.imag))
                values = [
                    self._display_name_for_trace(trace),
                    f"{display_frequency:.6f}",
                    f"{s11_db:.3f}",
                    delta_s11_db,
                    f"{abs(parameter):.4f}",
                    delta_s11_mag,
                    f"{np.degrees(np.angle(parameter)):.2f}",
                    _format_impedance(impedance),
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
                ]

            self._set_table_row(
                self.marker_table,
                row_index,
                trace.color,
                values,
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
        sort_values: list[float | None] | None = None,
        bold: bool = False,
    ) -> None:
        color = QtGui.QColor(color_hex)
        for column, value in enumerate(values):
            item = _SortableTableWidgetItem(value)
            item.setForeground(QtGui.QBrush(color))
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

    pg.setConfigOptions(antialias=True, background="#f8fafc", foreground="#24313f")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")

    window = TouchstoneViewerWindow(args.files)
    window.showMaximized()
    return app.exec()
