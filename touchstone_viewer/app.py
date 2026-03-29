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
    s11_marker: pg.ScatterPlotItem | None = None
    smith_marker: pg.ScatterPlotItem | None = None


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
        self.marker_plot_label: pg.TextItem | None = None
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

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.s11_plot = pg.PlotWidget()
        self.s11_plot.scene().sigMouseClicked.connect(self._handle_plot_click)
        self.s11_plot.getPlotItem().vb.sigRangeChanged.connect(
            self._update_marker_plot_label_position
        )
        splitter.addWidget(self.s11_plot)

        self.smith_plot = pg.PlotWidget()
        splitter.addWidget(self.smith_plot)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.summary_label = QtWidgets.QLabel("Drop .s1p files here or open them from the dialog.")
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

        layout.addWidget(splitter, stretch=1)

        self.marker_table = QtWidgets.QTableWidget(0, 6)
        self.marker_table.setHorizontalHeaderLabels(
            ["Trace", "Freq", "S11 (dB)", "|Gamma|", "Angle (deg)", "Z (ohm)"]
        )
        self.marker_table.verticalHeader().setVisible(False)
        self.marker_table.setAlternatingRowColors(True)
        self.marker_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.marker_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.marker_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.marker_table.setMinimumHeight(210)

        header = self.marker_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, self.marker_table.columnCount()):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.marker_table)

        self.setCentralWidget(central_widget)

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

    def _configure_s11_plot(self) -> None:
        plot_item = self.s11_plot.getPlotItem()
        plot_item.setTitle("S11 Over Frequency")
        plot_item.setLabel("left", "S11", units="dB")
        plot_item.setLabel("bottom", "Frequency", units=self.frequency_scale.unit)
        plot_item.showGrid(x=True, y=True, alpha=0.18)
        plot_item.addLegend(labelTextColor="#24313f", brush="#ffffffdd")
        self.s11_plot.setMouseEnabled(x=True, y=True)

        self.marker_plot_label = pg.TextItem(
            text="Marker: n/a",
            color="#1e293b",
            anchor=(1.0, 0.0),
            border=pg.mkPen("#93c5fd", width=1.2),
            fill=pg.mkBrush(255, 255, 255, 228),
        )
        self.marker_plot_label.setZValue(30)
        self.s11_plot.addItem(self.marker_plot_label, ignoreBounds=True)
        self._update_marker_plot_label_position()

    def _configure_smith_plot(self) -> None:
        plot_item = self.smith_plot.getPlotItem()
        plot_item.setTitle("Smith Chart")
        self.smith_plot.setMouseEnabled(x=False, y=False)
        add_smith_grid(plot_item)

    def _refresh_plots(self) -> None:
        self._choose_frequency_scale()

        self.s11_plot.clear()
        self.smith_plot.clear()
        self.marker_plot_label = None
        self.aoi_region_item = None
        self._configure_s11_plot()
        self._configure_smith_plot()

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
            self.marker_line.sigPositionChanged.connect(self._handle_marker_moved)
            self.s11_plot.addItem(self.marker_line, ignoreBounds=True)
            self._set_marker_line_value(self.marker_frequency_hz)
            self._update_marker_outputs()
        else:
            self.marker_line = None
            self.marker_frequency_hz = None
            self.aoi_region_hz = None
            self._update_marker_table()
            self._update_marker_plot_label()
            self._reset_aoi_controls()
            self._set_aoi_controls_enabled(False)

        self.summary_label.setText(f"{len(self.traces)} trace(s) loaded")

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

        trace.s11_marker = pg.ScatterPlotItem(
            size=10,
            brush=pg.mkBrush(trace.color),
            pen=pg.mkPen("#ffffff", width=1.5),
        )
        trace.smith_marker = pg.ScatterPlotItem(
            size=11,
            brush=pg.mkBrush(trace.color),
            pen=pg.mkPen("#ffffff", width=1.5),
        )

        self.s11_plot.addItem(trace.s11_marker)
        self.smith_plot.addItem(trace.smith_marker)

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
            "Touchstone files (*.s1p *.S1P);;All files (*.*)",
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

    def _handle_plot_click(self, event: object) -> None:
        if not self.marker_line or not hasattr(event, "scenePos"):
            return

        scene_position = event.scenePos()
        bounding_rect = self.s11_plot.getPlotItem().vb.sceneBoundingRect()
        if not bounding_rect.contains(scene_position):
            return

        mouse_point = self.s11_plot.getPlotItem().vb.mapSceneToView(scene_position)
        frequency_hz = mouse_point.x() * self.frequency_scale.factor_hz
        self.marker_frequency_hz = frequency_hz
        self._set_marker_line_value(frequency_hz)
        self._update_marker_outputs()

    def _handle_marker_moved(self) -> None:
        if not self.marker_line or self._updating_marker:
            return

        self.marker_frequency_hz = self.marker_line.value() * self.frequency_scale.factor_hz
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

    def _set_marker_line_value(self, frequency_hz: float) -> None:
        if not self.marker_line:
            return

        self._updating_marker = True
        try:
            self.marker_line.setValue(frequency_hz / self.frequency_scale.factor_hz)
        finally:
            self._updating_marker = False

    def _update_marker_outputs(self) -> None:
        if self.marker_frequency_hz is None:
            self._update_marker_plot_label()
            self._update_marker_table()
            return

        self._update_marker_plot_label()
        self._update_marker_table()

    def _update_marker_plot_label(self) -> None:
        if self.marker_plot_label is None:
            return

        if self.marker_frequency_hz is None:
            self.marker_plot_label.setText("Marker: n/a")
        else:
            display_frequency = self.marker_frequency_hz / self.frequency_scale.factor_hz
            self.marker_plot_label.setText(
                f"Marker: {display_frequency:.6f} {self.frequency_scale.unit}"
            )

        self._update_marker_plot_label_position()

    def _update_marker_plot_label_position(self, *_args: object) -> None:
        if self.marker_plot_label is None:
            return

        x_range, y_range = self.s11_plot.getPlotItem().viewRange()
        x_margin = (x_range[1] - x_range[0]) * 0.02
        y_margin = (y_range[1] - y_range[0]) * 0.04
        self.marker_plot_label.setPos(x_range[1] - x_margin, y_range[1] - y_margin)

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
            color = QtGui.QColor(trace.color)
            gamma = None
            if self.marker_frequency_hz is not None:
                gamma = trace.data.interpolated_gamma(self.marker_frequency_hz)

            if gamma is None:
                if trace.s11_marker is not None:
                    trace.s11_marker.setData([], [])
                if trace.smith_marker is not None:
                    trace.smith_marker.setData([], [])
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
                s11_db = 20.0 * np.log10(max(abs(gamma), 1.0e-12))
                impedance = gamma_to_impedance(gamma, trace.data.reference_impedance_ohms)

                if trace.s11_marker is not None:
                    trace.s11_marker.setData([display_frequency], [s11_db])
                if trace.smith_marker is not None:
                    trace.smith_marker.setData([gamma.real], [gamma.imag])

                values = [
                    trace.data.label,
                    f"{display_frequency:.6f}",
                    f"{s11_db:.3f}",
                    f"{abs(gamma):.4f}",
                    f"{np.degrees(np.angle(gamma)):.2f}",
                    _format_impedance(impedance),
                ]

            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setForeground(QtGui.QBrush(color))
                self.marker_table.setItem(row_index, column, item)

        self.marker_table.resizeRowsToContents()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.load_files(paths)
            event.acceptProposedAction()
            return
        event.ignore()


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
        description="Interactive S11 and Smith chart viewer for Touchstone .s1p files."
    )
    parser.add_argument("files", nargs="*", type=Path, help="Touchstone .s1p files to open")
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
