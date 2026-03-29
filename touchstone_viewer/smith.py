from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

_RESISTANCE_VALUES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
_REACTANCE_VALUES = [0.2, 0.5, 1.0, 2.0, 5.0]
_SMITH_DEFAULT_BOUNDARY = 1.05
_SMITH_LIMIT = 1.25
_SMITH_MIN_RANGE = 0.08


def add_smith_grid(plot_item: pg.PlotItem) -> None:
    view_box = plot_item.getViewBox()
    plot_item.setAspectLocked(True)
    plot_item.enableAutoRange(False)
    view_box.setMouseMode(pg.ViewBox.RectMode)
    view_box.setMouseEnabled(x=True, y=True)
    view_box.setLimits(
        xMin=-_SMITH_LIMIT,
        xMax=_SMITH_LIMIT,
        yMin=-_SMITH_LIMIT,
        yMax=_SMITH_LIMIT,
        minXRange=_SMITH_MIN_RANGE,
        maxXRange=2.0 * _SMITH_LIMIT,
        minYRange=_SMITH_MIN_RANGE,
        maxYRange=2.0 * _SMITH_LIMIT,
    )
    plot_item.setXRange(-_SMITH_DEFAULT_BOUNDARY, _SMITH_DEFAULT_BOUNDARY, padding=0.0)
    plot_item.setYRange(-_SMITH_DEFAULT_BOUNDARY, _SMITH_DEFAULT_BOUNDARY, padding=0.0)
    plot_item.setLabel("bottom", "Re(Gamma)")
    plot_item.setLabel("left", "Im(Gamma)")
    plot_item.showGrid(x=False, y=False)

    boundary_pen = pg.mkPen("#0f172a", width=2.2)
    axis_pen = pg.mkPen("#64748b", width=1.1)
    grid_pen = pg.mkPen("#94a3b8", width=1, style=QtCore.Qt.PenStyle.DashLine)

    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    unit_circle = np.exp(1j * theta)
    plot_item.addItem(
        pg.PlotDataItem(unit_circle.real, unit_circle.imag, pen=boundary_pen)
    )
    plot_item.addItem(pg.PlotDataItem([-1.0, 1.0], [0.0, 0.0], pen=axis_pen))
    plot_item.addItem(pg.PlotDataItem([0.0, 0.0], [-1.0, 1.0], pen=axis_pen))

    reactance_span = np.linspace(-25.0, 25.0, 1200)
    resistance_span = np.linspace(0.0, 25.0, 1200)

    for resistance in _RESISTANCE_VALUES:
        gamma = normalized_impedance_to_gamma(resistance + 1j * reactance_span)
        plot_item.addItem(pg.PlotDataItem(gamma.real, gamma.imag, pen=grid_pen))

    for reactance in _REACTANCE_VALUES:
        for sign in (-1.0, 1.0):
            gamma = normalized_impedance_to_gamma(resistance_span + 1j * sign * reactance)
            plot_item.addItem(pg.PlotDataItem(gamma.real, gamma.imag, pen=grid_pen))

    _add_grid_labels(plot_item)


def normalized_impedance_to_gamma(z: np.ndarray | complex) -> np.ndarray:
    z_array = np.asarray(z, dtype=np.complex128)
    return (z_array - 1.0) / (z_array + 1.0)


def _add_grid_labels(plot_item: pg.PlotItem) -> None:
    font = QtGui.QFont()
    font.setPointSize(8)

    label_color = "#52606d"

    for resistance in _RESISTANCE_VALUES:
        gamma = normalized_impedance_to_gamma(complex(resistance, 0.0))
        label = pg.TextItem(_format_grid_value(resistance), color=label_color, anchor=(0.5, 0.0))
        label.setFont(font)
        label.setPos(float(np.real(gamma)), 0.02)
        plot_item.addItem(label)

    for reactance in _REACTANCE_VALUES:
        gamma_upper = normalized_impedance_to_gamma(complex(0.15, reactance))
        upper = pg.TextItem(
            f"+j{_format_grid_value(reactance)}",
            color=label_color,
            anchor=(0.0, 1.0),
        )
        upper.setFont(font)
        upper.setPos(float(np.real(gamma_upper)), float(np.imag(gamma_upper)))
        plot_item.addItem(upper)

        gamma_lower = normalized_impedance_to_gamma(complex(0.15, -reactance))
        lower = pg.TextItem(
            f"-j{_format_grid_value(reactance)}",
            color=label_color,
            anchor=(0.0, 0.0),
        )
        lower.setFont(font)
        lower.setPos(float(np.real(gamma_lower)), float(np.imag(gamma_lower)))
        plot_item.addItem(lower)


def _format_grid_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"
