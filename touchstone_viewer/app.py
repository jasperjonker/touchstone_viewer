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

AOI_UNIT_FACTORS_HZ = {
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}

LAST_OPEN_DIRECTORY_KEY = "paths/last_open_directory"


@dataclass(frozen=True)
class FrequencyScale:
    factor_hz: float
    unit: str


@dataclass
class LoadedTrace:
    data: TouchstoneData
    color: str
    s11_curve: pg.PlotDataItem | None = None
    smith_curve: pg.PlotDataItem | None = None
    s21_curve: pg.PlotDataItem | None = None
    s11_marker: pg.ScatterPlotItem | None = None
    smith_marker: pg.ScatterPlotItem | None = None
    s21_marker: pg.ScatterPlotItem | None = None


class TouchstoneViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, initial_paths: Sequence[Path]) -> None:
        super().__init__()
        self.setWindowTitle("Touchstone Viewer")
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self.settings = QtCore.QSettings("TouchstoneViewer", "Touch")

        self.traces: list[LoadedTrace] = []
        self.frequency_scale = FrequencyScale(1.0e6, "MHz")
        self.marker_frequency_hz: float | None = None
        self.marker_line: pg.InfiniteLine | None = None
        self.s21_marker_line: pg.InfiniteLine | None = None
        self.marker_plot_label: pg.TextItem | None = None
        self.s21_marker_plot_label: pg.TextItem | None = None
        self.aoi_region_hz: tuple[float, float] | None = None
        self.aoi_region_item: pg.LinearRegionItem | None = None
        self._updating_marker = False
        self._updating_aoi_controls = False

        self._build_ui()
        self._refresh_plots()

        if initial_paths:
            self.load_files(initial_paths)

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        open_button = QtWidgets.QPushButton("Open Files")
        open_button.clicked.connect(self._open_files_dialog)
        controls.addWidget(open_button)

        clear_button = QtWidgets.QPushButton("Clear")
        clear_button.clicked.connect(self.clear_traces)
        controls.addWidget(clear_button)

        controls.addSpacing(12)

        self.summary_label = QtWidgets.QLabel(
            "Drop .s1p or .s2p files here or open them from the dialog."
        )
        controls.addWidget(self.summary_label, stretch=1)

        aoi_prefix = QtWidgets.QLabel("Area of Interest")
        controls.addWidget(aoi_prefix)

        self.aoi_start_input = self._build_aoi_spin_box()
        self.aoi_start_input.valueChanged.connect(self._handle_aoi_value_changed)
        controls.addWidget(self.aoi_start_input)

        aoi_to_label = QtWidgets.QLabel("to")
        controls.addWidget(aoi_to_label)

        self.aoi_stop_input = self._build_aoi_spin_box()
        self.aoi_stop_input.valueChanged.connect(self._handle_aoi_value_changed)
        controls.addWidget(self.aoi_stop_input)

        self.aoi_unit_combo = QtWidgets.QComboBox()
        self.aoi_unit_combo.addItems(AOI_UNIT_FACTORS_HZ.keys())
        self.aoi_unit_combo.setCurrentText("GHz")
        self.aoi_unit_combo.currentTextChanged.connect(self._handle_aoi_unit_changed)
        controls.addWidget(self.aoi_unit_combo)

        self._set_aoi_controls_enabled(False)

        layout.addLayout(controls)

        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.addTab(self._build_s11_tab(), "S11")
        self.tab_widget.addTab(self._build_s21_tab(), "S21")
        layout.addWidget(self.tab_widget, stretch=1)

        self.setCentralWidget(central_widget)

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
            ["Trace", "Freq", "S11 (dB)", "|S11|", "Angle (deg)", "Z (ohm)"]
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
            ["Trace", "Freq", "S21 (dB)", "|S21|", "Angle (deg)"]
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
        for column in range(1, table.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

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

    def _refresh_plots(self) -> None:
        self._choose_frequency_scale()

        self.s11_plot.clear()
        self.smith_plot.clear()
        self.s21_plot.clear()
        self.marker_plot_label = None
        self.s21_marker_plot_label = None
        self.aoi_region_item = None

        self._configure_s11_plot()
        self._configure_smith_plot()
        self._configure_s21_plot()

        if self.traces:
            self._add_aoi_region()
            self._sync_aoi_controls_to_region()
            self._set_aoi_controls_enabled(True)

        for trace in self.traces:
            self._add_trace_items(trace)

        if self.traces:
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
            self._update_marker_outputs()
        else:
            self.marker_line = None
            self.s21_marker_line = None
            self.marker_frequency_hz = None
            self.aoi_region_hz = None
            self._update_marker_table()
            self._update_s21_marker_table()
            self._update_marker_plot_label()
            self._update_s21_marker_plot_label()
            self._reset_aoi_controls()
            self._set_aoi_controls_enabled(False)

        self.summary_label.setText(f"{len(self.traces)} trace(s) loaded")

    def _configure_s11_plot(self) -> None:
        plot_item = self.s11_plot.getPlotItem()
        plot_item.setTitle("S11 Over Frequency")
        plot_item.setLabel("left", "S11", units="dB")
        plot_item.setLabel("bottom", "Frequency", units=self.frequency_scale.unit)
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.s11_plot.setMouseEnabled(x=True, y=True)

        self.marker_plot_label = self._build_marker_plot_label()
        self.s11_plot.addItem(self.marker_plot_label, ignoreBounds=True)
        self._update_marker_plot_label_position()

    def _configure_smith_plot(self) -> None:
        plot_item = self.smith_plot.getPlotItem()
        plot_item.setTitle("Smith Chart")
        self.smith_plot.setMouseEnabled(x=False, y=False)
        add_smith_grid(plot_item)

    def _configure_s21_plot(self) -> None:
        plot_item = self.s21_plot.getPlotItem()
        plot_item.setTitle("S21 Over Frequency")
        plot_item.setLabel("left", "S21", units="dB")
        plot_item.setLabel("bottom", "Frequency", units=self.frequency_scale.unit)
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.s21_plot.setMouseEnabled(x=True, y=True)

        self.s21_marker_plot_label = self._build_marker_plot_label()
        self.s21_plot.addItem(self.s21_marker_plot_label, ignoreBounds=True)
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

    def _choose_frequency_scale(self) -> None:
        if not self.traces:
            self.frequency_scale = FrequencyScale(1.0e6, "MHz")
            return

        max_frequency = max(trace.data.frequencies_hz[-1] for trace in self.traces)
        if max_frequency >= 1.0e9:
            self.frequency_scale = FrequencyScale(1.0e9, "GHz")
        elif max_frequency >= 1.0e6:
            self.frequency_scale = FrequencyScale(1.0e6, "MHz")
        elif max_frequency >= 1.0e3:
            self.frequency_scale = FrequencyScale(1.0e3, "kHz")
        else:
            self.frequency_scale = FrequencyScale(1.0, "Hz")

    def _add_trace_items(self, trace: LoadedTrace) -> None:
        pen = pg.mkPen(trace.color, width=2)
        scaled_frequency = trace.data.frequencies_hz / self.frequency_scale.factor_hz

        trace.s11_curve = self.s11_plot.plot(
            scaled_frequency,
            trace.data.s11_db(),
            pen=pen,
            name=trace.data.label,
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
                name=trace.data.label,
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
        overlap_start = max(trace.data.frequencies_hz[0] for trace in self.traces)
        overlap_end = min(trace.data.frequencies_hz[-1] for trace in self.traces)
        if overlap_start < overlap_end:
            return 0.5 * (overlap_start + overlap_end)

        first_trace = self.traces[0].data
        return 0.5 * (first_trace.frequencies_hz[0] + first_trace.frequencies_hz[-1])

    def _handle_plot_click(self, plot_widget: pg.PlotWidget, event: object) -> None:
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
        if not self.marker_line or self._updating_marker:
            return

        self.marker_frequency_hz = self.marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_s21_marker_moved(self) -> None:
        if not self.s21_marker_line or self._updating_marker:
            return

        self.marker_frequency_hz = self.s21_marker_line.value() * self.frequency_scale.factor_hz
        self._set_marker_line_values(self.marker_frequency_hz)
        self._update_marker_outputs()

    def _handle_aoi_value_changed(self, _value: float) -> None:
        if self._updating_aoi_controls or not self.traces:
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

        if self.traces:
            self._sync_aoi_controls_to_region()
        else:
            self._reset_aoi_controls()

    def _aoi_unit_factor_hz(self) -> float:
        return AOI_UNIT_FACTORS_HZ[self.aoi_unit_combo.currentText()]

    def _set_aoi_controls_enabled(self, enabled: bool) -> None:
        self.aoi_start_input.setEnabled(enabled)
        self.aoi_stop_input.setEnabled(enabled)
        self.aoi_unit_combo.setEnabled(enabled)

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
        self._update_marker_plot_label()
        self._update_s21_marker_plot_label()
        self._update_marker_table()
        self._update_s21_marker_table()

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
        overlap_start = max(trace.data.frequencies_hz[0] for trace in self.traces)
        overlap_end = min(trace.data.frequencies_hz[-1] for trace in self.traces)
        if overlap_start < overlap_end:
            return (overlap_start, overlap_end)
        return fallback_bounds_hz

    def _frequency_span_hz(self) -> tuple[float, float] | None:
        if not self.traces:
            return None

        minimum_hz = min(trace.data.frequencies_hz[0] for trace in self.traces)
        maximum_hz = max(trace.data.frequencies_hz[-1] for trace in self.traces)
        return (minimum_hz, maximum_hz)

    def _update_marker_table(self) -> None:
        self.marker_table.setRowCount(len(self.traces))

        for row_index, trace in enumerate(self.traces):
            parameter = None
            if self.marker_frequency_hz is not None:
                parameter = trace.data.interpolated_parameter(1, 1, self.marker_frequency_hz)

            if parameter is None:
                self._set_scatter_point(trace.s11_marker, None)
                self._set_scatter_point(trace.smith_marker, None)
                values = [
                    trace.data.label,
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                ]
            else:
                display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
                s11_db = _parameter_db(parameter)
                impedance = gamma_to_impedance(parameter, trace.data.reference_impedance_ohms)
                self._set_scatter_point(trace.s11_marker, (display_frequency, s11_db))
                self._set_scatter_point(trace.smith_marker, (parameter.real, parameter.imag))
                values = [
                    trace.data.label,
                    f"{display_frequency:.6f}",
                    f"{s11_db:.3f}",
                    f"{abs(parameter):.4f}",
                    f"{np.degrees(np.angle(parameter)):.2f}",
                    _format_impedance(impedance),
                ]

            self._set_table_row(self.marker_table, row_index, trace.color, values)

        self.marker_table.resizeRowsToContents()

    def _update_s21_marker_table(self) -> None:
        self.s21_marker_table.setRowCount(len(self.traces))

        for row_index, trace in enumerate(self.traces):
            if not trace.data.has_parameter(2, 1):
                self._set_scatter_point(trace.s21_marker, None)
                values = [
                    trace.data.label,
                    "not available",
                    "not available",
                    "not available",
                    "not available",
                ]
                self._set_table_row(self.s21_marker_table, row_index, trace.color, values)
                continue

            parameter = None
            if self.marker_frequency_hz is not None:
                parameter = trace.data.interpolated_parameter(2, 1, self.marker_frequency_hz)

            if parameter is None:
                self._set_scatter_point(trace.s21_marker, None)
                values = [
                    trace.data.label,
                    "out of range",
                    "out of range",
                    "out of range",
                    "out of range",
                ]
            else:
                display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
                s21_db = _parameter_db(parameter)
                self._set_scatter_point(trace.s21_marker, (display_frequency, s21_db))
                values = [
                    trace.data.label,
                    f"{display_frequency:.6f}",
                    f"{s21_db:.3f}",
                    f"{abs(parameter):.4f}",
                    f"{np.degrees(np.angle(parameter)):.2f}",
                ]

            self._set_table_row(self.s21_marker_table, row_index, trace.color, values)

        self.s21_marker_table.resizeRowsToContents()

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
    ) -> None:
        color = QtGui.QColor(color_hex)
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            item.setForeground(QtGui.QBrush(color))
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
