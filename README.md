# MachineGuard — Cross-Machine Bearing Fault Diagnosis

## 1. Overview

MachineGuard is a machine learning project that investigates whether a bearing fault diagnosis model trained on one machine's vibration data can generalize to data collected from another machine.

The project uses publicly available bearing datasets to study vibration features, fault classification, and cross-dataset model evaluation.

The main idea is to keep the feature extraction process consistent, train a model using labeled data from one dataset, and evaluate it on a separate dataset without retraining the model.

## 2. Objectives

- Understand and analyze bearing vibration signals.
- Extract meaningful time-domain and frequency-domain features.
- Train a supervised machine learning model to classify bearing conditions.
- Investigate whether the trained model generalizes to a different machine or test rig.
- Evaluate the effects of differences in sensors, sampling rates, operating conditions, and fault characteristics.
- Document experiments, limitations, and results with evidence.

## 3. Datasets

### CWRU Bearing Dataset

The Case Western Reserve University (CWRU) Bearing Data Center provides bearing vibration recordings collected under different operating conditions.

The initial project will use CWRU data to develop and evaluate a supervised classification model.

Planned classes:

- Normal
- Ball Fault
- Inner Race Fault
- Outer Race Fault

Dataset: https://engineering.case.edu/bearingdatacenter

### Paderborn University Bearing Dataset

The Paderborn University Bearing Dataset provides bearing condition measurements under multiple operating conditions, including healthy and damaged bearings.

It will be investigated as a potential external dataset for cross-dataset evaluation.

Dataset: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download

### NASA IMS Bearing Dataset

The NASA IMS Bearing Dataset contains time-series vibration measurements collected during bearing experiments.

It will be investigated separately for anomaly detection, degradation monitoring, and analysis of vibration changes over time.

Dataset information: https://data.nasa.gov/

**Note:** Dataset compatibility, label mappings, measurement units, sensor specifications, and sampling rates must be verified before performing cross-dataset evaluation.

## 4. Project Workflow

The planned workflow is:

1. Collect and inspect the raw datasets.
2. Verify sensor information, sampling rates, operating conditions, and available labels.
3. Clean and segment vibration signals.
4. Extract relevant vibration features.
5. Analyze feature distributions, correlations, and redundancy.
6. Train a baseline classifier using CWRU data.
7. Evaluate the classifier on held-out CWRU recordings.
8. Apply the same feature definitions and trained model to a compatible external dataset.
9. Compare the results and investigate possible domain shifts.
10. Document findings, limitations, and future improvements.

## 5. Feature Engineering

Candidate features include:

| Feature | Purpose |
|---|---|
| RMS | Measures signal magnitude and vibration energy level |
| Standard deviation | Measures signal variation |
| Peak amplitude | Captures the largest absolute amplitude |
| Peak-to-peak | Measures the difference between maximum and minimum signal values |
| Crest factor | Relates peak amplitude to RMS |
| Kurtosis | Captures the prominence of extreme values |
| Dominant frequency | Identifies the strongest non-DC frequency component |

Additional frequency-domain or bearing-geometry-based features may be investigated when the required metadata is available.

Features will be selected based on physical interpretation, statistical analysis, redundancy checks, and controlled experiments rather than adding features without justification.

## 6. Cross-Dataset Evaluation

The central experiment is to train a model on CWRU and evaluate it on a separate dataset, such as Paderborn, where compatible fault labels are available.

The external dataset will be processed using consistent feature definitions and the preprocessing objects fitted during training.

The model will remain unchanged during the direct-transfer experiment.

Evaluation will consider:

- Accuracy and balanced accuracy
- Per-class precision and recall
- Confusion matrix
- Differences between training and target feature distributions
- Sensor and operating-condition differences

If the target dataset does not support the same fault classes or measurement assumptions, the evaluation protocol will be adjusted and its limitations documented.

## 7. Tools and Technologies

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- Git and GitHub

Additional libraries will be introduced only when required by the implementation.

## 8. Project Structure

The repository will be organized into separate areas for raw data, processed data, reusable source code, experiments, trained models, tests, and reports.

The structure will evolve as the implementation develops. Each module will have a clear responsibility, and experiments will be kept reproducible and understandable.

## 9. Research Questions

1. Which vibration features provide useful information for bearing fault classification?
2. How well does the classifier perform on recordings excluded from training?
3. Can a model trained on CWRU generalize to a different bearing test rig?
4. How do sampling rates, sensor characteristics, loads, and machine conditions affect performance?
5. Can time-series vibration features help identify abnormal degradation in run-to-failure data?

## 10. Current Status

**Stage:** Repository initialization and project planning.

The initial implementation will focus on understanding the data, establishing a reliable baseline, and documenting the reasoning behind preprocessing and feature selection.

Cross-dataset performance and generalization have not yet been established.

## 11. Guiding Principle

MachineGuard emphasizes understanding the complete machine learning process—not only achieving a high accuracy score.

Every major decision involving data preparation, feature selection, model choice, and evaluation should be supported by experiments, documented reasoning, and reproducible results.

## 12. License and Data Attribution

Dataset licenses and usage conditions will be reviewed before redistribution. Original datasets remain subject to their respective terms and attribution requirements.