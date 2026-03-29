from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FREQUENCY_UNITS = {
    "HZ": 1.0,
    "KHZ": 1.0e3,
    "MHZ": 1.0e6,
    "GHZ": 1.0e9,
}

_SUPPORTED_DATA_FORMATS = {"RI", "MA", "DB"}


@dataclass(frozen=True)
class TouchstoneOptions:
    frequency_scale_hz: float = 1.0e9
    parameter: str = "S"
    data_format: str = "MA"
    reference_impedance_ohms: float = 50.0


@dataclass(frozen=True)
class TouchstoneData:
    path: Path
    label: str
    frequencies_hz: NDArray[np.float64]
    gamma: NDArray[np.complex128]
    reference_impedance_ohms: float

    def s11_db(self) -> NDArray[np.float64]:
        magnitude = np.maximum(np.abs(self.gamma), 1.0e-12)
        return 20.0 * np.log10(magnitude)

    def impedance_ohms(self) -> NDArray[np.complex128]:
        denominator = 1.0 - self.gamma
        safe_denominator = np.where(
            np.abs(denominator) < 1.0e-12,
            np.nan + 0.0j,
            denominator,
        )
        return self.reference_impedance_ohms * (1.0 + self.gamma) / safe_denominator

    def interpolated_gamma(self, frequency_hz: float) -> complex | None:
        if frequency_hz < self.frequencies_hz[0] or frequency_hz > self.frequencies_hz[-1]:
            return None

        real = np.interp(frequency_hz, self.frequencies_hz, self.gamma.real)
        imag = np.interp(frequency_hz, self.frequencies_hz, self.gamma.imag)
        return complex(real, imag)


def load_touchstone(path: str | Path) -> TouchstoneData:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Touchstone file not found: {file_path}")

    options = TouchstoneOptions()
    frequencies_hz: list[float] = []
    gamma_values: list[complex] = []

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("!", maxsplit=1)[0].strip()
            if not line:
                continue

            if line.startswith("#"):
                options = _parse_options(line)
                continue

            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{file_path.name}:{line_number} is not valid 1-port Touchstone data: {raw_line.rstrip()}"
                )

            frequency = _parse_float(fields[0]) * options.frequency_scale_hz
            value_a = _parse_float(fields[1])
            value_b = _parse_float(fields[2])
            gamma = _convert_to_gamma(options.data_format, value_a, value_b)

            frequencies_hz.append(frequency)
            gamma_values.append(gamma)

    if not frequencies_hz:
        raise ValueError(f"{file_path.name} does not contain any data rows")

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    gamma = np.asarray(gamma_values, dtype=np.complex128)

    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    gamma = gamma[order]

    return TouchstoneData(
        path=file_path,
        label=file_path.stem,
        frequencies_hz=frequencies,
        gamma=gamma,
        reference_impedance_ohms=options.reference_impedance_ohms,
    )


def gamma_to_impedance(gamma: complex, reference_impedance_ohms: float) -> complex:
    denominator = 1.0 - gamma
    if abs(denominator) < 1.0e-12:
        return complex(float("nan"), float("nan"))
    return reference_impedance_ohms * (1.0 + gamma) / denominator


def _parse_options(line: str) -> TouchstoneOptions:
    defaults = ["GHZ", "S", "MA", "R", "50"]
    tokens = line[1:].split()
    if len(tokens) > len(defaults):
        raise ValueError(f"Unsupported Touchstone option line: {line}")

    for index, token in enumerate(tokens):
        defaults[index] = token

    frequency_unit = defaults[0].upper()
    parameter = defaults[1].upper()
    data_format = defaults[2].upper()
    reference_token = defaults[3].upper()
    reference_impedance = _parse_float(defaults[4])

    if frequency_unit not in _FREQUENCY_UNITS:
        raise ValueError(f"Unsupported frequency unit in option line: {line}")
    if parameter != "S":
        raise ValueError(f"Only S-parameter Touchstone files are supported: {line}")
    if data_format not in _SUPPORTED_DATA_FORMATS:
        raise ValueError(f"Unsupported data format in option line: {line}")
    if reference_token != "R":
        raise ValueError(f"Unsupported reference syntax in option line: {line}")

    return TouchstoneOptions(
        frequency_scale_hz=_FREQUENCY_UNITS[frequency_unit],
        parameter=parameter,
        data_format=data_format,
        reference_impedance_ohms=reference_impedance,
    )


def _convert_to_gamma(data_format: str, value_a: float, value_b: float) -> complex:
    if data_format == "RI":
        return complex(value_a, value_b)
    if data_format == "MA":
        return value_a * np.exp(1j * np.deg2rad(value_b))
    if data_format == "DB":
        magnitude = 10.0 ** (value_a / 20.0)
        return magnitude * np.exp(1j * np.deg2rad(value_b))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def _parse_float(raw_value: str) -> float:
    return float(raw_value.replace("D", "E").replace("d", "e"))

