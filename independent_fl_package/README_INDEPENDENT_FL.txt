SECURE IoT IDS - THREE SEPARATE DATASET EXPERIMENTS
==================================================

IMPORTANT DESIGN
----------------
This workflow does NOT merge UNSW-NB15, CICIoT2023 and Edge-IIoTset into one
feature space or one classifier.

It prepares and trains three separate experiments:
  1. UNSW_NB15  -> its own features, labels, scaler, model and metrics
  2. CICIoT2023 -> its own features, labels, scaler, model and metrics
  3. EdgeIIoTset -> its own features, labels, scaler, model and metrics

Each experiment has three simulated FL clients belonging only to that dataset.
FedAvg occurs only among those three clients. There is NO cross-dataset model
aggregation.

20 PERCENT DATA POLICY
----------------------
The preparation script samples approximately 20% of each dataset (per class,
using a deterministic hash). That selected 20% is then divided into:
  - 70% training
  - 15% validation
  - 15% final test

Therefore, relative to all valid labelled source rows, this is approximately:
  - 14% used for training
  -  3% used for validation
  -  3% used for final testing

The held-out validation and test rows are not used to fit the scaler. If the
requirement instead means 20% for training PLUS separate validation/test data,
the split policy must be changed deliberately.

EXPECTED RAW DATA PATHS
-----------------------
Place raw data under the project's data folder:

UNSW-NB15:
  data/UNSW-NB15_1.csv
  data/UNSW-NB15_2.csv
  data/UNSW-NB15_3.csv
  data/UNSW-NB15_4.csv

CICIoT2023:
  data/CICIoT2023/MERGED_CSV/*.csv

Edge-IIoTset:
  data/EdgeIIoTset/archive (1)/Edge-IIoTset dataset/
    Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv

RUN ON WINDOWS POWERSHELL
-------------------------
Open PowerShell in the project root:

  .\run_independent_fl.ps1 -Mode Prepare
  .\run_independent_fl.ps1 -Mode Train -Dataset all

Or run both in sequence:

  .\run_independent_fl.ps1 -Mode All

Train one dataset only:

  .\run_independent_fl.ps1 -Mode Train -Dataset UNSW_NB15
  .\run_independent_fl.ps1 -Mode Train -Dataset CICIoT2023
  .\run_independent_fl.ps1 -Mode Train -Dataset EdgeIIoTset

Example lighter test run (not final research settings):

  .\run_independent_fl.ps1 -Mode Train -Dataset UNSW_NB15 -Rounds 1 -LocalSteps 5 -BatchSize 64

Rebuild only the new independent data output folder:

  .\run_independent_fl.ps1 -Mode Prepare -RebuildData

The preparation script deletes ONLY data/independent_fl_v1 when -RebuildData
is supplied. It does not delete the existing V5 data folder or models.

OUTPUTS
-------
Prepared data:
  data/independent_fl_v1/<dataset>/

Models:
  models/independent_fl_v1/<dataset>/federated_cnn_bigru_mha.keras

Metrics and reports:
  results/independent_fl_v1/<dataset>/
    evaluation_metrics.json
    classification_report.csv
    confusion_matrix.csv
    round_history.csv
    experiment_config.json

Overall result index:
  results/independent_fl_v1/all_dataset_results.json

RESEARCH VALIDITY NOTES
----------------------
1. Accuracy is not guaranteed to reach 90%; it must be measured per dataset.
2. Compare accuracy with macro-F1, macro-recall and per-class recall.
3. The clients are currently created by deterministic random assignment within
   each dataset. They are simulated clients, not evidence of real devices.
4. The current Edge-IIoTset preparation converts selected feature columns to
   numeric and replaces non-numeric values with zero. Some categorical/text
   fields may therefore lose information. Inspect the actual Edge data schema
   and consider train-only categorical encoding before treating this as final.
5. The current trainer uses fixed local_steps per client but sample-count
   weighting for FedAvg. This aggregation/training-workload pairing should be
   reviewed before making a formal FedAvg claim.
6. Syntax and source inspection do not replace running the experiment on the
   actual datasets and validating the output metrics.
