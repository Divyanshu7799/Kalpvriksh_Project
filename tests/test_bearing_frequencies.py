import pytest

from machineguard.features.bearing_frequencies import defect_frequencies_hz


def test_defect_frequencies_at_1797_rpm():
    frequencies = defect_frequencies_hz(1797)

    assert frequencies["BPFO"] == pytest.approx(107.4, abs=0.1)
    assert frequencies["BPFI"] == pytest.approx(162.2, abs=0.1)
    assert frequencies["FTF"] == pytest.approx(11.9, abs=0.1)


def test_frequencies_scale_with_shaft_speed():
    slow = defect_frequencies_hz(900)
    fast = defect_frequencies_hz(1800)

    assert fast["BPFO"] == pytest.approx(2 * slow["BPFO"])


def test_twice_ball_spin_frequency_at_1797_rpm():
    assert defect_frequencies_hz(1797)["2xBSF"] == pytest.approx(141.17, abs=0.01)
