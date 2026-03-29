from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pytest

from touchstone_viewer.smith import (
    add_smith_grid,
    normalized_impedance_to_gamma,
    reset_smith_view,
)


def test_normalized_impedance_to_gamma_matches_expected_values() -> None:
    gamma = normalized_impedance_to_gamma(np.asarray([1.0 + 0.0j, 0.0 + 0.0j]))

    assert gamma[0] == 0.0 + 0.0j
    assert gamma[1] == -1.0 + 0.0j


def test_add_smith_grid_adds_plot_items(qapp) -> None:
    widget = pg.PlotWidget()

    add_smith_grid(widget.getPlotItem())

    assert len(widget.getPlotItem().items) > 5
    assert widget.getViewBox().state["mouseEnabled"] == [True, True]
    assert widget.getViewBox().state["mouseMode"] == pg.ViewBox.RectMode
    assert widget.getViewBox().state["limits"]["xRange"][0] == 0.08
    assert widget.getViewBox().state["limits"]["yRange"][0] == 0.08
    widget.close()


def test_reset_smith_view_restores_default_bounds(qapp) -> None:
    widget = pg.PlotWidget()
    plot_item = widget.getPlotItem()
    add_smith_grid(plot_item)
    default_x_range, default_y_range = plot_item.viewRange()
    plot_item.setXRange(-0.2, 0.2, padding=0.0)
    plot_item.setYRange(-0.1, 0.1, padding=0.0)

    reset_smith_view(plot_item)

    assert plot_item.viewRange()[0] == pytest.approx(default_x_range)
    assert plot_item.viewRange()[1] == pytest.approx(default_y_range)
    widget.close()
