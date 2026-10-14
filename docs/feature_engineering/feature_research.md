# MachineGuard — Feature Engineering Research

## 1. Purpose

This document records the reasoning and evidence used to select features for bearing fault diagnosis.

The objective is to investigate which signal characteristics help distinguish bearing conditions and whether those features remain useful when the model is evaluated on recordings from a different bearing or dataset.

No feature is considered necessary merely because it is commonly used in other projects.

## 2. Research Questions

1. Which time-domain, frequency-domain and time-frequency features are suitable for vibration-based bearing diagnosis?
2. What physical or statistical information does each feature represent?
3. Which features provide overlapping or redundant information?
4. Which features distinguish healthy and faulty recordings?
5. Which features distinguish ball, inner-race and outer-race faults?
6. How sensitive are the features to sampling rate, RPM, load, sensor placement and measurement units?
7. Does a feature improve performance on unseen recordings?
8. Does the selected feature set generalize to another dataset?

## 3. Candidate Feature Families

### 3.1 Time-Domain Statistical Features

Candidates:

- Mean
- Minimum and maximum
- Standard deviation
- Variance
- Root mean square (RMS)
- Peak amplitude
- Peak-to-peak amplitude
- Skewness
- Kurtosis
- Crest factor
- Impulse factor
- Clearance factor
- Shape factor
- Zero-crossing rate

Investigation:

- Define each formula and its assumptions.
- Identify mathematical dependencies.
- Compare class-wise distributions.
- Examine sensitivity to outliers and signal offsets.
- Test predictive contribution on held-out recordings.

### 3.2 Frequency-Domain Features

Candidates:

- Dominant frequency
- Spectral centroid
- Spectral bandwidth
- Spectral entropy
- Frequency-band power
- Frequency-band energy ratios
- Spectral peak amplitudes

Investigation:

- Verify the sampling rate and frequency resolution.
- Define the spectrum and normalization convention.
- Determine whether the feature depends on signal length or amplitude.
- Investigate whether observed peaks are physically meaningful.

### 3.3 Bearing-Physics-Based Features

Candidates:

- Ball Pass Frequency Outer Race (BPFO)
- Ball Pass Frequency Inner Race (BPFI)
- Ball Spin Frequency (BSF)
- Fundamental Train Frequency (FTF)
- Harmonic amplitudes and selected sideband features
- Envelope-spectrum features

Investigation:

- Obtain the required bearing geometry.
- Verify shaft speed and units.
- Document the theoretical frequency formula.
- Determine whether envelope analysis is appropriate.
- Avoid assigning fault meaning to a peak without supporting evidence.

### 3.4 Time-Frequency Features

Candidates:

- Short-Time Fourier Transform (STFT) band energies
- Wavelet coefficients
- Wavelet packet energies
- Energy ratios across selected bands

Investigation:

- Identify the temporal patterns these features may capture.
- Document window, wavelet and decomposition choices.
- Compare their additional complexity with simpler alternatives.

### 3.5 Periodicity and Temporal Features

Candidates:

- Autocorrelation peaks
- Selected lag-based statistics
- Periodicity measures
- Rolling RMS
- Window-to-window variation
- Rate of change of selected features

Investigation:

- Distinguish features computed within one window from features calculated across consecutive windows.
- Determine whether the task is fault classification or degradation monitoring.
- Prevent temporal leakage between training and test periods.

## 4. Measurement and Data Requirements

Before comparing features, verify:

- Dataset and source recording
- Sensor location and measurement axis
- Measurement quantity and units
- Sampling rate
- Shaft speed (RPM)
- Operating load
- Bearing geometry, where required
- Fault category and label source
- Window length and overlap
- Any missing or uncertain metadata

Unknown values must remain explicitly unknown until verified. They must not be silently replaced with assumed values.

## 5. Evaluation Method

### Stage A — Signal inspection

Inspect representative raw recordings, plot waveforms and spectra where appropriate, and document observations.

### Stage B — Feature calculation

Calculate candidate features using documented formulas and consistent preprocessing.

### Stage C — Statistical analysis

Compare feature distributions across labels. Examine effect sizes, outliers, correlations and mathematical redundancy.

### Stage D — Baseline evaluation

Establish a reproducible baseline using a suitable recording-level data split.

### Stage E — Feature ablation

Compare feature sets while keeping the data split, model settings and evaluation metrics consistent.

### Stage F — Cross-dataset evaluation

After validating the baseline, test whether the same feature definitions and training-fitted preprocessing work on a compatible external dataset.

## 6. Important Evaluation Constraints

- Overlapping windows from one recording must not be randomly distributed across training and test sets.
- Feature selection must not use held-out test labels.
- Scaling and other fitted preprocessing must be learned from training data only.
- Accuracy alone is insufficient; per-class performance and a confusion matrix should also be reported.
- Differences between recordings must not be mistaken for fault-related patterns.
- Cross-dataset results must be interpreted in the context of sensor and operating-condition differences.

## 7. Feature Decision Register

For every candidate feature, record:

| Field | Description |
|---|---|
| Feature | Name and mathematical definition |
| Family | Time, frequency, physics-based, time-frequency or temporal |
| Motivation | Why it may be useful |
| Requirements | Metadata and preprocessing needed |
| Limitations | Sensitivity, assumptions and failure cases |
| Statistical evidence | Distribution and redundancy findings |
| Predictive evidence | Contribution on held-out recordings |
| Decision | Retain, reject, or investigate further |
| Reason | Evidence supporting the decision |

## 8. Current Status

Research planning only.

Candidate features have not yet been approved for the final model. Their usefulness must be established through data inspection, documented reasoning and controlled experiments.

## 9. Decision Principle

The final feature set should be justified by physical interpretation, data quality, statistical evidence, predictive evaluation and the intended cross-machine use case.

The goal is not to maximize the number of features. The goal is to select a defensible, reproducible feature set.