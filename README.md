# An Intelligent Federated Deep Learning Framework for Multi-Class IoT Intrusion Detection

A research-oriented **IoT network intrusion detection system** that explores federated deep learning for classifying network traffic into normal activity and multiple attack categories. The project includes multi-client training experiments, CNN–BiGRU–Multi-Head Attention models, evaluation reports, and SHAP-based explainability.

> **Research note:** This repository contains multiple experimental variants. Results depend on the dataset preparation, model version, sampling strategy, and evaluation split. Use the report that matches the exact model and experiment you are discussing; do not treat metrics from different runs as directly interchangeable.

## Contents

- [Project overview](#project-overview)
- [Key features](#key-features)
- [Datasets and federated clients](#datasets-and-federated-clients)
- [Model architecture](#model-architecture)
- [Repository structure](#repository-structure)
- [Environment setup](#environment-setup)
- [Prepare data and run experiments](#prepare-data-and-run-experiments)
- [Run the Streamlit applications](#run-the-streamlit-applications)
- [Evaluation and explainability](#evaluation-and-explainability)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [References](#references)

## Project overview

Intrusion Detection Systems (IDS) help identify suspicious network activity. IoT environments are especially challenging because devices, traffic patterns, and attack types can vary widely. A model trained on one dataset may not generalize equally well to other environments, and centralized training may require raw traffic from multiple organizations to be pooled.

This project investigates a **federated learning workflow** in which participating clients train locally and contribute model updates to a shared global model. It also explores multi-class prediction and model explanations to help analysts understand which traffic features influence a prediction.

The implementation is an experimental research project, not a production network security appliance. Federated learning can reduce the need to centralize raw training data, but it does not by itself guarantee privacy, security, or robustness.

## Key features

- **Multi-class intrusion detection:** supports a common 13-class label set in the three-dataset workflow.
- **Federated training experiments:** scripts for client-based training and global-model aggregation.
- **Deep learning:** CNN, bidirectional GRU (BiGRU), and Multi-Head Attention model variants.
- **Multiple data workflows:** UNSW-NB15, CICIoT2023, and Edge-IIoTset experiments, including a harmonized three-dataset pipeline.
- **Evaluation artifacts:** training metrics, classification reports, confusion matrices, and model-comparison outputs.
- **Explainable AI:** SHAP-related scripts and saved explanations for selected predictions.
- **Interactive demos:** Streamlit applications for the full-feature and three-dataset workflows, provided the matching local model and preprocessing artifacts are available.

## Datasets and federated clients

The three-dataset workflow uses these datasets as separate client domains:

1. **UNSW-NB15**
2. **CICIoT2023**
3. **Edge-IIoTset**

The three-dataset pipeline harmonizes traffic into a shared feature representation and a common set of attack labels. The 16-feature representation used by the relevant three-dataset scripts includes:

`duration`, `packet_rate`, `src_packets`, `dst_packets`, `src_bytes`, `dst_bytes`, `packet_size_mean`, `packet_size_min`, `packet_size_max`, `packet_size_std`, `ttl`, `syn_count`, `ack_count`, `fin_count`, `rst_count`, and `iat`.

The 13-class label set used in this workflow is:

`Normal`, `DDoS`, `DoS`, `Reconnaissance`, `Backdoor`, `Injection`, `BruteForce`, `MITM`, `Malware`, `Exploits`, `Fuzzers`, `Generic`, and `OtherAttack`.

**Dataset access and preparation:** dataset files are not included in this repository. Obtain them from their official sources and follow the preparation scripts and expected paths in the code. Check each dataset's license and terms before use or redistribution.

- [UNSW-NB15 — UNSW research page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [CICIoT2023 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- [Edge-IIoTset — dataset publication and resources](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)

## Model architecture

The principal three-dataset experiments use a hybrid **CNN–BiGRU–Multi-Head Attention** architecture.

1. **Input and preprocessing:** load prepared features and apply the shared training-derived scaling/preprocessing artifacts.
2. **CNN layers:** learn local patterns in traffic features.
3. **BiGRU layers:** learn relationships across the feature sequence in both directions.
4. **Multi-Head Attention:** model relationships among learned representations.
5. **Classification layer:** output scores for the supported traffic classes.
6. **Federated aggregation:** combine client model updates into a global model according to the experiment implementation.

The repository contains several experiment versions, including a V6 three-client, 16-feature, 13-class workflow. Different scripts may use different data folders, model filenames, sampling methods, or training settings; keep each run paired with its own preprocessing artifacts and report.

## Repository structure

```text
.
├── app.py                              # Streamlit demo for the full-feature workflow
├── app_3dataset.py                     # Streamlit demo for the three-dataset workflow
├── src/
│   ├── federated.py                    # Federated experiment
│   ├── federated_20percent_bigru_mha.py # Earlier 20% data experiment
│   ├── federated_20percent_bigru_mha_v6.py
│   ├── federated_3dataset_bigru_mha.py # Three-dataset model workflow
│   ├── federated_cnn_bigru_mha.py
│   ├── federated_cnn_bigru_mha_focal.py
│   ├── federated_hybrid_full.py
│   ├── harmonize_3_datasets.py          # Dataset harmonization
│   ├── prepare_3dataset_20percent.py    # Prepare a subset for experiments
│   ├── prepare_3dataset_fl.py           # Prepare federated client data
│   ├── prepare_unseen.py                # Prepare unseen evaluation data
│   ├── evaluate_unseen.py               # Evaluate on unseen data
│   ├── explain*.py                      # SHAP/explainability workflows
│   └── predict*.py                      # Prediction utilities
├── models/                              # Local model and preprocessing artifacts
├── report/                              # Metrics, reports, plots, and explanations
├── data/                                # Local datasets and prepared arrays (not committed)
├── requirements.txt
└── README.md
```

The source directory also contains data inspection, validation, balancing, scaler-building, plotting, and diagnostic utilities. See the scripts' configuration sections for their exact input and output paths.

## Environment setup

Python **3.12.10** was used in the development environment for this project. TensorFlow and other compiled dependencies can be sensitive to Python and operating-system versions, so use a clean virtual environment and compatible package versions.

### Windows PowerShell

```powershell
git clone https://github.com/g-vijaya123/An-Intelligent-Federated-Deep-Learning-Framework-for-Multi-Class-IoT-Intrusion-Detection.git
cd An-Intelligent-Federated-Deep-Learning-Framework-for-Multi-Class-IoT-Intrusion-Detection

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow streamlit shap joblib
```

The extra install command includes packages imported by the application and experiment scripts that may not all be listed in the current `requirements.txt`. For a repeatable research setup, pin and test the dependency versions used in your environment before sharing results.

Verify the environment:

```powershell
python --version
python -m pip check
```

## Prepare data and run experiments

The repository contains multiple pipelines, so there is no single command that safely prepares every experiment. Follow the sequence and file paths expected by the script you choose.

### Three-dataset federated workflow

A typical workflow is:

1. Download the datasets from their official sources.
2. Inspect and harmonize the source schemas and labels.
3. Prepare the federated client datasets.
4. Build or load the training-only scaler required by that experiment.
5. Run the selected federated training script.
6. Review the matching report and confusion matrix in `report/`.

Relevant scripts include:

```text
src/inspect_3_datasets.py
src/harmonize_3_datasets.py
src/validate_harmonized_data.py
src/prepare_3dataset_fl.py
src/prepare_3dataset_20percent.py
src/build_global_train_scaler.py
src/federated_3dataset_bigru_mha.py
src/federated_20percent_bigru_mha_v6.py
```

Before executing a script, review its configuration constants and expected data paths. Some scripts expect prepared arrays or a saved scaler from an earlier step and are not standalone download-and-train commands.

### Full-feature workflow

The full-feature Streamlit application expects local artifacts at these paths:

- `models/federated_cnn_bigru_mha_full.keras`
- `models/preprocessing_full_features.joblib`
- `data/processed_full/X_test.npy`
- `data/processed_full/y_test.npy`

Generate or place the corresponding artifacts using the matching preparation and training workflow before launching the application.

## Run the Streamlit applications

From the repository root, with the virtual environment activated:

### Full-feature application

```powershell
streamlit run app.py
```

This application expects the full-feature model, preprocessing artifact, and prepared test arrays listed above.

### Three-dataset application

```powershell
streamlit run app_3dataset.py
```

This application expects the three-dataset model at `models/federated_20percent_cnn_bigru_mha_corrected.keras` and the corresponding scaler/data artifacts under `data/fl_3datasets_20percent/`.

If the application reports a missing file, verify that the artifact belongs to the same experiment version and that its path matches the constants in the application. Do not rename an unrelated model or scaler just to bypass a missing-file error.

## Evaluation and explainability

The `report/` directory contains generated artifacts from different experiment versions, including:

- Federated training and round metrics
- Per-class metrics and final text reports
- Confusion matrices in CSV and image formats
- Model comparison tables
- Unseen-data prediction checks
- SHAP explanation tables and visualizations

Useful examples include:

- `report/federated_20percent_v6_final_report.txt`
- `report/federated_20percent_v6_per_class_metrics.csv`
- `report/federated_20percent_v6_confusion_matrix.csv`
- `report/federated_3dataset_round_metrics.csv`
- `report/final_model_comparison.csv`
- `report/final_3dataset_shap_explanation.csv`

Use the exact report associated with the model checkpoint and data split. For multiclass IDS evaluation, report accuracy together with per-class precision, recall, F1-score, support, and the confusion matrix. Overall accuracy alone can hide weak detection of minority attack classes.

SHAP scripts and outputs provide feature-attribution explanations for selected predictions. These explanations are useful for analysis, but they should not be interpreted as proof of causality or as a substitute for security validation.

## Reproducibility and limitations

- **Data is not included:** raw datasets and large prepared arrays are excluded from version control. You must obtain and prepare them locally.
- **Model artifacts may be local-only:** the application may require trained model files, scalers, and prepared test arrays that are not committed.
- **Experiment variants differ:** do not compare metrics across runs unless their datasets, splits, preprocessing, class mapping, and evaluation procedures are aligned.
- **Avoid data leakage:** fit preprocessing/scaling on training data only and keep test data out of model selection and training.
- **Federated learning is not automatically private:** this code explores federated training; it does not establish formal differential privacy or guarantee protection against model-update leakage.
- **Research use:** validate the pipeline in your target environment before considering operational deployment.

## Future improvements

- Add a single documented configuration/CLI entry point for each experiment.
- Pin dependency versions and add automated smoke tests for data preparation and inference.
- Provide a clean experiment manifest linking every model checkpoint to its dataset split, preprocessing artifacts, seed, and report.
- Evaluate robustness and cross-dataset generalization under consistent protocols.
- Investigate privacy-preserving aggregation and adversarial robustness.

## References

1. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network intrusion detection systems*. Military Communications and Information Systems Conference (MilCIS).
2. [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
3. [CICIoT2023 dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
4. [Edge-IIoTset dataset resources](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications).

## Author

Developed as a final-year research project on federated deep learning and multi-class IoT intrusion detection.

---

If you use or adapt this repository in academic work, cite the datasets and the research publications relevant to your experiment.
