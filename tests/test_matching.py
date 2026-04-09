from __future__ import annotations

import numpy as np
import pytest

from touchstone_viewer.matching import (
    MatchingStage,
    apply_matching_network,
    component_units,
    element_impedance_ohms,
    impedance_to_gamma,
    suggest_matching_stages,
)


def test_component_units_follow_component_type() -> None:
    assert component_units("R") == ("ohm", "kOhm")
    assert component_units("L") == ("pH", "nH", "uH")
    assert component_units("C") == ("pF", "nF", "uF")


def test_element_impedance_ohms_handles_inductors_and_capacitors() -> None:
    frequencies_hz = np.asarray([1.0e9], dtype=np.float64)

    inductor = element_impedance_ohms(frequencies_hz, component="L", value_si=1.0e-9)
    capacitor = element_impedance_ohms(frequencies_hz, component="C", value_si=1.0e-12)

    assert inductor[0] == pytest.approx(1j * 2.0 * np.pi)
    assert capacitor[0] == pytest.approx(-1j / (2.0 * np.pi * 1.0e9 * 1.0e-12))


def test_apply_matching_network_combines_series_and_shunt_stages() -> None:
    frequencies_hz = np.asarray([1.0e9], dtype=np.float64)
    load_impedance = np.asarray([50.0 + 0.0j], dtype=np.complex128)
    stages = [
        MatchingStage("Series", "R", 25.0, "ohm"),
        MatchingStage("Shunt", "R", 100.0, "ohm"),
    ]

    matched = apply_matching_network(load_impedance, frequencies_hz, stages)

    assert matched[0] == pytest.approx(42.857142857142854 + 0.0j)


def test_impedance_to_gamma_maps_reference_impedance_to_zero() -> None:
    impedance = np.asarray([50.0 + 0.0j, 25.0 + 0.0j], dtype=np.complex128)

    gamma = impedance_to_gamma(impedance, 50.0)

    assert gamma[0] == pytest.approx(0.0 + 0.0j)
    assert gamma[1] == pytest.approx((-1.0 / 3.0) + 0.0j)


def test_suggest_matching_stages_returns_useful_single_stage_match() -> None:
    load_impedance = 25.0 + 25.0j
    base_score = abs(
        impedance_to_gamma(
            np.asarray([load_impedance], dtype=np.complex128),
            50.0,
        )[0]
    )
    suggestions = suggest_matching_stages(load_impedance, 1.0e9, 50.0, max_results=5)

    assert suggestions
    assert all(suggestion.stage.component in {"L", "C"} for suggestion in suggestions)
    assert suggestions[0].score < base_score


def test_suggest_matching_stages_scans_fine_pf_and_nh_steps() -> None:
    suggestions = suggest_matching_stages(25.0 + 25.0j, 1.0e9, 50.0, max_results=500)

    capacitor_values = {
        suggestion.stage.value
        for suggestion in suggestions
        if suggestion.stage.component == "C" and suggestion.stage.unit == "pF"
    }
    inductor_values = {
        suggestion.stage.value
        for suggestion in suggestions
        if suggestion.stage.component == "L" and suggestion.stage.unit == "nH"
    }

    assert 1.0 in capacitor_values
    assert 1.1 in capacitor_values
    assert 10.0 in capacitor_values
    assert 1.0 in inductor_values
    assert 1.1 in inductor_values
    assert 10.0 in inductor_values
