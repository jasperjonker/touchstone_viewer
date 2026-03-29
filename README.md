# Touchstone Viewer

Desktop viewer for Touchstone `.s1p` and `.s2p` files, built with Python, PySide6, and pyqtgraph.

<img width="2255" height="1453" alt="image" src="https://github.com/user-attachments/assets/f79e4b16-2b7f-4d28-bc97-41c508008349" />


It is designed for quick antenna and matching-network inspection with:

- `S11 (dB)` over frequency
- `S21 (dB)` over frequency for 2-port files
- Smith chart plotting
- multiple traces open at the same time for comparison
- an interactive frequency marker with per-trace readout
- dedicated `S11` and `S21` tabs
- a shaded area of interest on the S11 plot

## Features

- Open one or more `.s1p` or `.s2p` files from the command line, file dialog, or drag and drop.
- Compare multiple traces with clear coloring and plot legends.
- Inspect a specific frequency using the movable marker.
- Define an area of interest with start/stop fields and a shared frequency unit selector.
- View the same marker position on the `S11` plot, Smith chart, and `S21` plot.
- Remember the last opened directory in the file dialog, with Downloads as the first-run default.

## Requirements

- Python 3.10 or newer

## Quick Start With `uv`

Run the app directly:

```bash
uv run touch
```

Open files immediately on startup:

```bash
uv run touch path/to/file1.s1p path/to/file2.s2p
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

## Running Tests

Run the full test suite with the `uv` shortcut:

```bash
uv run touch-test
```

Run plain pytest directly from the project environment:

```bash
uv run --group dev pytest
```

With a standard virtual environment:

```bash
pip install -e . pytest pytest-cov
pytest --cov=touchstone_viewer --cov-report=term-missing
```

The project currently enforces a minimum total coverage of `80%`.

## Usage

- `Open Files` appends traces to the current session.
- `Clear` removes all loaded traces.
- Drag `.s1p` or `.s2p` files into the window to load them.
- Use the `S11` tab for return loss and Smith chart inspection.
- Use the `S21` tab for transmission comparison on 2-port files.
- Drag the vertical marker on the `S11` or `S21` plot, or click on either plot, to update the marker position.
- Enter the area of interest start and stop values, then choose `kHz`, `MHz`, or `GHz`.
- The file dialog reopens in the last directory you used. On first launch, it defaults to your Downloads folder.

## Supported Touchstone Formats

The parser currently supports:

- `.s1p`
- `.s2p`

Supported data encodings:

- `RI`
- `MA`
- `DB`

## Command Names

- `touch` is the preferred short command.
- `touchstone-viewer` is also available

## Roadmap

- Matching and tuning assistance on top of the Smith chart
- Component what-if analysis such as series inductors or shunt capacitors at a target frequency
