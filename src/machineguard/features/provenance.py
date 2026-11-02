"""Where each candidate feature comes from: its definition, its hypothesis, and its source.

source_status says how far the source has been checked:
  "read"        the source was opened and the feature is in it
  "exists"      the source exists, but its content was not checked for this feature
  "none yet"    no source found; the feature stays a candidate until one is found, or is dropped
Update source_status yourself as you read the papers. Never write "read" for something you have not read.
"""

from collections import namedtuple

Provenance = namedtuple("Provenance", ["family", "definition", "hypothesis", "source", "source_status"])

_STATISTICS_SOURCE = (
    "Pattern Recognition of Bearing Faults using Smoother Statistical Features, arXiv 1503.04444, Table 2"
)
_ENVELOPE_SOURCE = "Randall & Antoni (2011), Rolling element bearing diagnostics - a tutorial, MSSP 25(2)"
_CWRU_FREQUENCY_SOURCE = (
    "CWRU defect-frequency multipliers: Table 2 of arXiv 1511.03174 (3.585, 5.415, 2 x 2.357, 0.3983); "
    "also Smith & Randall (2015), CWRU benchmark study, MSSP 64-65"
)

PROVENANCE = {
    "rms": Provenance(
        "time", "sqrt(mean(x^2))", "A fault adds vibration energy; larger faults add more.", _STATISTICS_SOURCE, "read"
    ),
    "std": Provenance(
        "time",
        "standard deviation of x",
        "Spread of amplitude; near RMS when the mean is about 0.",
        _STATISTICS_SOURCE,
        "read",
    ),
    "kurtosis": Provenance(
        "time", "excess kurtosis of x", "Impacts make the signal spiky.", _STATISTICS_SOURCE, "read"
    ),
    "skewness": Provenance(
        "time", "skewness of x", "A fault may make the amplitude distribution asymmetric.", _STATISTICS_SOURCE, "read"
    ),
    "crest_factor": Provenance(
        "time", "max|x| / RMS", "Impacts raise the peak relative to the average level.", _STATISTICS_SOURCE, "read"
    ),
    "impulse_factor": Provenance(
        "time",
        "max|x| / mean|x|",
        "Like crest factor, relative to the mean absolute level.",
        _STATISTICS_SOURCE,
        "read",
    ),
    "shape_factor": Provenance(
        "time", "RMS / mean|x|", "Changes with the shape of the amplitude distribution.", _STATISTICS_SOURCE, "read"
    ),
    "peak_to_peak": Provenance(
        "time",
        "max(x) - min(x)",
        "Total swing grows with fault severity (the paper calls it Range).",
        _STATISTICS_SOURCE,
        "read",
    ),
    "peak": Provenance(
        "time",
        "max|x|",
        "Largest excursion; component of the ratio factors above.",
        "Component of crest and impulse factors in " + _STATISTICS_SOURCE,
        "exists",
    ),
    "mean_abs": Provenance(
        "time",
        "mean|x|",
        "Average absolute level; component of shape and impulse factors.",
        "Component of shape and impulse factors in " + _STATISTICS_SOURCE,
        "exists",
    ),
    "clearance_factor": Provenance(
        "time",
        "max|x| / (mean sqrt|x|)^2",
        "Peak relative to a low-order mean; sensitive to small clearance changes.",
        "Not in the paper above; find a bearing-diagnosis source",
        "none yet",
    ),
    "band_energy_ratio": Provenance(
        "frequency",
        "share of spectral energy between 2.5 and 5.5 kHz",
        "Impacts excite structural resonances, moving energy into the resonance band.",
        _ENVELOPE_SOURCE,
        "exists",
    ),
    "env_bpfo": Provenance(
        "frequency",
        "envelope-spectrum peak near BPFO / median background (20-400 Hz)",
        "An outer-race defect makes bursts at BPFO.",
        _ENVELOPE_SOURCE + "; " + _CWRU_FREQUENCY_SOURCE,
        "exists",
    ),
    "env_bpfi": Provenance(
        "frequency",
        "envelope-spectrum peak near BPFI / median background (20-400 Hz)",
        "An inner-race defect makes bursts at BPFI.",
        _ENVELOPE_SOURCE + "; " + _CWRU_FREQUENCY_SOURCE,
        "exists",
    ),
    "env_2xbsf": Provenance(
        "frequency",
        "envelope-spectrum peak near 2 x BSF / median background (20-400 Hz)",
        "A ball defect strikes both races, giving bursts at 2 x BSF.",
        _ENVELOPE_SOURCE + "; " + _CWRU_FREQUENCY_SOURCE,
        "exists",
    ),
}
