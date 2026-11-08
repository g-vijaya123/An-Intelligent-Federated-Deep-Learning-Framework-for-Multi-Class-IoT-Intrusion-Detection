# Three Independent Federated IoT Intrusion Detection Experiments

This implementation follows Option 1: run three separate federated-learning experiments. It does **not** combine the datasets into one shared feature space or one shared output layer.

## Architecture

Each experiment uses:

**CNN + BiGRU + Multi-Head Self-Attention**

Within each dataset, the training split is divided into three simulated clients. The clients train locally and their model weights are combined using sample-count-weighted FedAvg. There is no aggregation between UNSW-NB15, CICIoT2023 and Edge-IIoTset.

## Dataset-native input schemas

- **UNSW-NB15:** numeric native features from the original 49-column schema; `attack_cat` and binary `label` are targets, not inputs. `Backdoor` and `Backdoors` are normalized to one class. Normal records are identified using binary label 0.
- **CICIoT2023:** the 39 numeric columns from the original 40-column schema; `Label` is the target. The 9 rows with missing labels are excluded and counted in metadata.
- **Edge-IIoTset:** the 50 numeric feature columns from the original 63-column schema; `Attack_type` is the multiclass target and `Attack_label` is excluded from inputs.

The scripts exclude non-numeric IP/text/payload columns rather than inventing numeric encodings for them. Numeric feature values that are missing or non-finite are replaced with zero.

## Data selection and split

For each complete original dataset independently:

1. Deterministically select approximately 20% of each class.
2. Split the selected subset into 70% training, 15% validation and 15% testing.
3. Divide only the training portion among three simulated clients.
4. Fit the dataset's scaler using training-client files only.

The hash-based sampling is reproducible and class-aware. The resulting percentages are approximate because each row is selected independently; actual counts are written to each dataset's `metadata.json`.

## Important safety

- Existing `data/fl_3datasets`, `data/fl_3datasets_20percent`, V5/V6 models and previous results are not modified.
- New prepared data goes to `data/independent_fl_v1`.
- New models go to `models/independent_fl_v1`.
- New results go to `results/independent_fl_v1`.
- The preparation script refuses to overwrite an existing new output folder unless `--overwrite` is explicitly supplied. That option deletes only `data/independent_fl_v1`.

## Files

- `prepare_independent_fl_data.py`: reads the original raw datasets, selects 20%, creates 70/15/15 splits and assigns training rows to simulated clients.
- `train_independent_fl.py`: fits one training-only scaler per dataset, trains the local client models, performs within-dataset FedAvg, selects the best round using validation loss and evaluates on the held-out test split.

## Run in Windows PowerShell

Open PowerShell in the project root:

```powershell
cd "C:\Users\gvija\Desktop\final project"
```

### 1. Prepare the independent datasets

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\prepare_independent_fl_data.py
```

This reads the original raw CSVs. It may take a long time and requires additional disk space. Do not run the old `prepare_3dataset_20percent.py` for this new experiment because it deletes and rebuilds the old V5 output directory.

If you intentionally want to rebuild only the new independent output folder:

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\prepare_independent_fl_data.py --overwrite
```

### 2. Install/check TensorFlow

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow, pandas, numpy; print('TensorFlow:', tensorflow.__version__); print('Pandas:', pandas.__version__); print('NumPy:', numpy.__version__)"
```

If TensorFlow is missing, install a TensorFlow version compatible with your Python version and hardware before training.

### 3. Run a small smoke test first

Start with UNSW only, fewer rounds and steps:

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset UNSW_NB15 --rounds 2 --local-steps 10 --batch-size 64
```

This confirms the data files, scaler, model, client training and evaluation pipeline work. These smoke-test metrics are not final project results.

### 4. Train each complete experiment

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset UNSW_NB15
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset CICIoT2023
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset EdgeIIoTset
```

Or train all three sequentially:

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset all
```

Default training configuration: 10 communication rounds, 100 local steps per client per round, batch size 128, learning rate 0.0001. You can change these using `--rounds`, `--local-steps`, `--batch-size` and `--learning-rate`.

For example:

```powershell
.\.venv\Scripts\python.exe .\src\independent_fl\train_independent_fl.py --dataset CICIoT2023 --rounds 12 --local-steps 120 --batch-size 128
```

## Output layout

```text
data/
  independent_fl_v1/
    UNSW_NB15/
      train_client_01.csv
      train_client_02.csv
      train_client_03.csv
      validation.csv
      test.csv
      metadata.json
      scaler.npz
      scaler_metadata.json
    CICIoT2023/
      ...
    EdgeIIoTset/
      ...
    global_summary.json

models/
  independent_fl_v1/
    UNSW_NB15/federated_cnn_bigru_mha.keras
    CICIoT2023/federated_cnn_bigru_mha.keras
    EdgeIIoTset/federated_cnn_bigru_mha.keras

results/
  independent_fl_v1/
    UNSW_NB15/
      round_history.csv
      evaluation_metrics.json
      classification_report.csv
      confusion_matrix.csv
      experiment_config.json
    CICIoT2023/
      ...
    EdgeIIoTset/
      ...
    all_dataset_results.json
```

## Important interpretation

These are three independent FL experiments with simulated clients inside each dataset. They are not a single cross-dataset federated model. Each model's softmax probabilities refer only to that dataset's own classes. Report the three results separately and do not compare accuracy alone; include macro-F1, weighted-F1, per-class recall and confusion matrices.

The scripts have been syntax-checked, but they have not been executed against your local raw CSV files or TensorFlow installation. Run the small smoke test before starting full training.
