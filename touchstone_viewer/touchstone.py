from __future__ import annotations

import re
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
_SUPPORTED_PORT_COUNTS = {1, 2}


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
    s_parameters: NDArray[np.complex128]
    reference_impedance_ohms: float

    @property
    def port_count(self) -> int:
        return int(self.s_parameters.shape[1])

    @property
    def gamma(self) -> NDArray[np.complex128]:
        return self.parameter(1, 1)

    def has_parameter(self, out_port: int, in_port: int) -> bool:
        return 1 <= out_port <= self.port_count and 1 <= in_port <= self.port_count

    def parameter(self, out_port: int, in_port: int) -> NDArray[np.complex128]:
        if not self.has_parameter(out_port, in_port):
            raise ValueError(
                f"{self.label} does not contain S{out_port}{in_port} data"
            )
        return self.s_parameters[:, out_port - 1, in_port - 1]

    def parameter_db(self, out_port: int, in_port: int) -> NDArray[np.float64]:
        magnitude = np.maximum(np.abs(self.parameter(out_port, in_port)), 1.0e-12)
        return 20.0 * np.log10(magnitude)

    def s11_db(self) -> NDArray[np.float64]:
        return self.parameter_db(1, 1)

    def s21_db(self) -> NDArray[np.float64]:
        return self.parameter_db(2, 1)

    def impedance_ohms(self) -> NDArray[np.complex128]:
        denominator = 1.0 - self.gamma
        safe_denominator = np.where(
            np.abs(denominator) < 1.0e-12,
            np.nan + 0.0j,
            denominator,
        )
        numerator = self.reference_impedance_ohms * (1.0 + self.gamma)
        impedance = np.full(self.gamma.shape, np.nan + 0.0j, dtype=np.complex128)
        np.divide(
            numerator,
            safe_denominator,
            out=impedance,
            where=np.isfinite(safe_denominator),
        )
        return impedance

    def interpolated_parameter(
        self,
        out_port: int,
        in_port: int,
        frequency_hz: float,
    ) -> complex | None:
        if not self.has_parameter(out_port, in_port):
            return None
        if frequency_hz < self.frequencies_hz[0] or frequency_hz > self.frequencies_hz[-1]:
            return None

        parameter = self.parameter(out_port, in_port)
        real = np.interp(frequency_hz, self.frequencies_hz, parameter.real)
        imag = np.interp(frequency_hz, self.frequencies_hz, parameter.imag)
        return complex(real, imag)

    def interpolated_gamma(self, frequency_hz: float) -> complex | None:
        return self.interpolated_parameter(1, 1, frequency_hz)


def load_touchstone(path: str | Path) -> TouchstoneData:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Touchstone file not found: {file_path}")

    port_count = _infer_port_count(file_path)
    options = TouchstoneOptions()
    frequencies_hz: list[float] = []
    parameter_rows: list[NDArray[np.complex128]] = []

    expected_row_tokens = 1 + 2 * port_count * port_count
    pending_fields: list[str] = []
    pending_line_number: int | None = None

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("!", maxsplit=1)[0].strip()
            if not line:
                continue

            if line.startswith("#"):
                options = _parse_options(line)
                continue

            fields = line.split()
            if not pending_fields:
                pending_line_number = line_number
            pending_fields.extend(fields)

            while len(pending_fields) >= expected_row_tokens:
                row_fields = pending_fields[:expected_row_tokens]
                pending_fields = pending_fields[expected_row_tokens:]

                frequency = _parse_float(row_fields[0]) * options.frequency_scale_hz
                parameters = [
                    _convert_to_gamma(
                        options.data_format,
                        _parse_float(row_fields[index]),
                        _parse_float(row_fields[index + 1]),
                    )
                    for index in range(1, expected_row_tokens, 2)
                ]

                frequencies_hz.append(frequency)
                parameter_rows.append(_reshape_parameter_row(parameters, port_count))
                pending_line_number = line_number if pending_fields else None

    if pending_fields:
        line_number = pending_line_number or 1
        raise ValueError(
            f"{file_path.name}:{line_number} has incomplete Touchstone data for a {port_count}-port file"
        )

    if not frequencies_hz:
        raise ValueError(f"{file_path.name} does not contain any data rows")

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    s_parameters = np.asarray(parameter_rows, dtype=np.complex128)

    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    s_parameters = s_parameters[order]

    return TouchstoneData(
        path=file_path,
        label=file_path.stem,
        frequencies_hz=frequencies,
        s_parameters=s_parameters,
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


def _infer_port_count(path: Path) -> int:
    match = re.search(r"\.s(\d+)p$", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"{path.name} does not use a supported Touchstone extension like .s1p or .s2p"
        )

    port_count = int(match.group(1))
    if port_count not in _SUPPORTED_PORT_COUNTS:
        raise ValueError(
            f"{path.name} uses {port_count} ports, but only .s1p and .s2p are supported"
        )
    return port_count


def _reshape_parameter_row(
    flat_parameters: list[complex],
    port_count: int,
) -> NDArray[np.complex128]:
    matrix = np.empty((port_count, port_count), dtype=np.complex128)
    index = 0
    for in_port in range(port_count):
        for out_port in range(port_count):
            matrix[out_port, in_port] = flat_parameters[index]
            index += 1
    return matrix


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
