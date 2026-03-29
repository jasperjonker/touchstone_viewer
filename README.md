# Touchstone Viewer

Desktop viewer for Touchstone `.s1p` files, built with Python, PySide6, and pyqtgraph.

Primary CLI command: `touch`

It is designed for quick antenna and matching-network inspection with:

- `S11 (dB)` over frequency
- Smith chart plotting
- multiple traces open at the same time for comparison
- an interactive frequency marker with per-trace readout
- a shaded area of interest on the S11 plot

## Features

- Open one or more `.s1p` files from the command line, file dialog, or drag and drop.
- Compare multiple traces with clear coloring and plot legends.
- Inspect a specific frequency using the movable marker.
- Define an area of interest with start/stop fields and a shared frequency unit selector.
- View the same marker position on both the S11 plot and Smith chart.

## Requirements

- Python 3.10 or newer

## Quick Start With `uv`

Run the app directly:

```bash
uv run touch
```

Open files immediately on startup:

```bash
uv run touch path/to/file1.s1p path/to/file2.s1p
```

You can also run it as a module:

```bash
uv run python -m touchstone_viewer
```

## Quick Start With A Standard Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

Run the application:

```bash
touch
```

Or:

```bash
python -m touchstone_viewer
```

## Usage

- `Open Files` appends traces to the current session.
- `Clear` removes all loaded traces.
- Drag `.s1p` files into the window to load them.
- Drag the vertical marker on the S11 plot, or click on the plot, to update the marker position.
- Enter the area of interest start and stop values, then choose `kHz`, `MHz`, or `GHz`.

## Supported Touchstone Formats

The parser currently supports common 1-port Touchstone option formats:

- `RI`
- `MA`
- `DB`

## Example

If this repository contains sample files, you can launch the viewer like this:

```bash
uv run touch s11_pc11_default.s1p s11_pc11_balun_200mm.s1p
```

## Command Names

- `touch` is the preferred short command.
- `touchstone-viewer` is also available

## Roadmap

- Matching and tuning assistance on top of the Smith chart
- Component what-if analysis such as series inductors or shunt capacitors at a target frequency
