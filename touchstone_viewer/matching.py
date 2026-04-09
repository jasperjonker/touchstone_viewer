from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

MATCHING_COMPONENT_UNITS = {
    "R": {"ohm": 1.0, "kOhm": 1.0e3},
    "L": {"pH": 1.0e-12, "nH": 1.0e-9, "uH": 1.0e-6},
    "C": {"pF": 1.0e-12, "nF": 1.0e-9, "uF": 1.0e-6},
}


def _stepped_candidates(unit: str, start: float, stop: float, step: float) -> list[tuple[str, float]]:
    count = int(round((stop - start) / step)) + 1
    return [
        (unit, round(start + index * step, 10))
        for index in range(count)
    ]


_SUGGESTION_VALUES = {
    "L": [
        *_stepped_candidates("nH", 1.0, 10.0, 0.1),
        ("pH", 50.0),
        ("pH", 100.0),
        ("pH", 200.0),
        ("pH", 500.0),
        ("nH", 25.0),
        ("nH", 50.0),
        ("nH", 100.0),
        ("uH", 0.2),
        ("uH", 0.5),
        ("uH", 1.0),
    ],
    "C": [
        *_stepped_candidates("pF", 1.0, 10.0, 0.1),
        ("pF", 0.1),
        ("pF", 0.2),
        ("pF", 0.5),
        ("pF", 25.0),
        ("pF", 50.0),
        ("pF", 100.0),
        ("nF", 0.2),
        ("nF", 0.5),
        ("nF", 1.0),
    ],
}


@dataclass(frozen=True)
class MatchingStage:
    topology: str
    component: str
    value: float
    unit: str
    enabled: bool = True

    @property
    def value_si(self) -> float:
        return self.value * MATCHING_COMPONENT_UNITS[self.component][self.unit]


@dataclass(frozen=True)
class MatchingSuggestion:
    stage: MatchingStage
    resulting_impedance_ohms: complex
    resulting_gamma: complex
    score: float


def component_units(component: str) -> tuple[str, ...]:
    return tuple(MATCHING_COMPONENT_UNITS[component].keys())


def apply_matching_network(
    load_impedance_ohms: NDArray[np.complex128],
    frequencies_hz: NDArray[np.float64],
    stages: Sequence[MatchingStage],
) -> NDArray[np.complex128]:
    matched_impedance = np.asarray(load_impedance_ohms, dtype=np.complex128).copy()
    for stage in stages:
        if not stage.enabled or stage.value <= 0.0:
            continue
        matched_impedance = apply_matching_stage(matched_impedance, frequencies_hz, stage)
    return matched_impedance


def apply_matching_stage(
    load_impedance_ohms: NDArray[np.complex128],
    frequencies_hz: NDArray[np.float64],
    stage: MatchingStage,
) -> NDArray[np.complex128]:
    element_impedance = element_impedance_ohms(
        frequencies_hz,
        component=stage.component,
        value_si=stage.value_si,
    )
    if stage.topology == "Series":
        return load_impedance_ohms + element_impedance
    if stage.topology == "Shunt":
        return _parallel_impedance(load_impedance_ohms, element_impedance)
    raise ValueError(f"Unsupported matching topology: {stage.topology}")


def element_impedance_ohms(
    frequencies_hz: NDArray[np.float64],
    *,
    component: str,
    value_si: float,
) -> NDArray[np.complex128]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if component == "R":
        return np.full(frequencies.shape, complex(value_si, 0.0), dtype=np.complex128)

    omega = 2.0 * np.pi * frequencies
    if component == "L":
        return 1j * omega * value_si
    if component == "C":
        impedance = np.full(frequencies.shape, np.inf + 0.0j, dtype=np.complex128)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(
                -1j,
                omega * value_si,
                out=impedance,
                where=np.abs(omega * value_si) >= 1.0e-30,
            )
        return impedance
    raise ValueError(f"Unsupported matching component: {component}")


def impedance_to_gamma(
    impedance_ohms: NDArray[np.complex128],
    reference_impedance_ohms: float,
) -> NDArray[np.complex128]:
    impedance = np.asarray(impedance_ohms, dtype=np.complex128)
    denominator = impedance + reference_impedance_ohms
    gamma = np.full(impedance.shape, np.nan + 0.0j, dtype=np.complex128)
    np.divide(
        impedance - reference_impedance_ohms,
        denominator,
        out=gamma,
        where=np.abs(denominator) >= 1.0e-12,
    )
    return gamma


def suggest_matching_stages(
    load_impedance_ohms: complex,
    frequency_hz: float,
    reference_impedance_ohms: float,
    *,
    max_results: int = 8,
) -> list[MatchingSuggestion]:
    load = np.asarray([load_impedance_ohms], dtype=np.complex128)
    frequency = np.asarray([frequency_hz], dtype=np.float64)
    suggestions: list[MatchingSuggestion] = []

    for topology in ("Series", "Shunt"):
        for component, candidates in _SUGGESTION_VALUES.items():
            for unit, value in candidates:
                stage = MatchingStage(
                    topology=topology,
                    component=component,
                    value=value,
                    unit=unit,
                    enabled=True,
                )
                resulting_impedance = apply_matching_stage(load, frequency, stage)[0]
                if not np.isfinite(resulting_impedance.real) or not np.isfinite(resulting_impedance.imag):
                    continue

                resulting_gamma = impedance_to_gamma(
                    np.asarray([resulting_impedance], dtype=np.complex128),
                    reference_impedance_ohms,
                )[0]
                if not np.isfinite(resulting_gamma.real) or not np.isfinite(resulting_gamma.imag):
                    continue

                suggestions.append(
                    MatchingSuggestion(
                        stage=stage,
                        resulting_impedance_ohms=resulting_impedance,
                        resulting_gamma=resulting_gamma,
                        score=float(abs(resulting_gamma)),
                    )
                )

    suggestions.sort(
        key=lambda suggestion: (
            suggestion.score,
            abs(suggestion.resulting_impedance_ohms.imag),
            abs(suggestion.resulting_impedance_ohms.real - reference_impedance_ohms),
        )
    )
    return suggestions[:max_results]


def _parallel_impedance(
    impedance_a_ohms: NDArray[np.complex128],
    impedance_b_ohms: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    with np.errstate(divide="ignore", invalid="ignore"):
        admittance = 1.0 / impedance_a_ohms + 1.0 / impedance_b_ohms
        return 1.0 / admittance
