from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from touchstone_viewer.smith import add_smith_grid, normalized_impedance_to_gamma


def test_normalized_impedance_to_gamma_matches_expected_values() -> None:
    gamma = normalized_impedance_to_gamma(np.asarray([1.0 + 0.0j, 0.0 + 0.0j]))

    assert gamma[0] == 0.0 + 0.0j
    assert gamma[1] == -1.0 + 0.0j


def test_add_smith_grid_adds_plot_items(qapp) -> None:
    widget = pg.PlotWidget()

    add_smith_grid(widget.getPlotItem())

    assert len(widget.getPlotItem().items) > 5
    widget.close()
