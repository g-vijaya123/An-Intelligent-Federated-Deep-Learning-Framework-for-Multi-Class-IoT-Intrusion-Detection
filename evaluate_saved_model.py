import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.independent_fl.train_independent_fl import (
    make_evaluation_dataset,
    evaluate_dataset_once,
    metrics_from_confusion_matrix,
)

DATASET = "CICIoT2023"
DATA_DIR = ROOT / "data" / "independent_fl_v1" / DATASET
MODEL_PATH = ROOT / "models" / "independent_fl_v1" / DATASET / "federated_cnn_bigru_mha.keras"
RESULT_DIR = ROOT / "results" / "independent_fl_v1" / DATASET

BATCH_SIZE = 32

def main():
    print("=" * 75)
    print("SAVED MODEL EVALUATION - CICIoT2023")
    print("=" * 75)

    metadata_path = DATA_DIR / "metadata.json"
    scaler_path = DATA_DIR / "scaler.npz"
    test_path = DATA_DIR / "test.csv"

    for path in [metadata_path, scaler_path, test_path, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["feature_columns"]
    classes = metadata["class_names"]

    scaler = np.load(scaler_path)
    mean = scaler["mean"]
    std = scaler["std"]
    scaler_features = scaler["features"].tolist()

    if features != scaler_features:
        raise ValueError("Feature order mismatch between metadata and scaler.")

    print(f"Dataset       : {DATASET}")
    print(f"Features      : {len(features)}")
    print(f"Classes       : {len(classes)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Test records  : {metadata['test_records']:,}")
    print("Loading saved model...", flush=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    test_ds = make_evaluation_dataset(
        test_path,
        features,
        mean,
        std,
        BATCH_SIZE,
    )

    print("\nStarting final test evaluation...", flush=True)

    test_loss, test_accuracy, confusion = evaluate_dataset_once(
        model,
        test_ds,
        len(classes),
    )

    metrics, report_rows = metrics_from_confusion_matrix(
        confusion,
        classes,
    )

    metrics.update({
        "dataset": DATASET,
        "test_records": int(confusion.sum()),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "model": "CNN + BiGRU + Multi-Head Attention",
        "evaluation": "Saved checkpoint; untouched test split",
    })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    (RESULT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        confusion,
        index=classes,
        columns=classes,
    ).to_csv(RESULT_DIR / "confusion_matrix.csv", index=True)

    pd.DataFrame(report_rows).to_csv(
        RESULT_DIR / "classification_report.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINAL TEST RESULTS")
    print("=" * 75)
    print(json.dumps(metrics, indent=2))
    print("\nSaved files:")
    print(RESULT_DIR / "evaluation_metrics.json")
    print(RESULT_DIR / "confusion_matrix.csv")
    print(RESULT_DIR / "classification_report.csv")
    print("\nEVALUATION COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
