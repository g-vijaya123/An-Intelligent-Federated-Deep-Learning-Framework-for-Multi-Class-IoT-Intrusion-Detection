from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "fl_3datasets"
MODEL_FILE = PROJECT_ROOT / "models" / "federated_3dataset_round_1.keras"
SCALER_FILE = DATA_DIR / "global_train_scaler.npz"

FEATURES = [
    "duration",
    "packet_rate",
    "src_packets",
    "dst_packets",
    "src_bytes",
    "dst_bytes",
    "packet_size_mean",
    "packet_size_min",
    "packet_size_max",
    "packet_size_std",
    "ttl",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "iat",
]

CLASSES = [
    "Normal",
    "DDoS",
    "DoS",
    "Reconnaissance",
    "Backdoor",
    "Injection",
    "BruteForce",
    "MITM",
    "Malware",
    "Exploits",
    "Fuzzers",
    "Generic",
    "OtherAttack",
]

CLIENTS = [
    "UNSW",
    "CICIoT2023",
    "EdgeIIoTset",
]

# Only a small held-out sample.
# No training happens here.
SAMPLE_SIZE = 50_000
BATCH_SIZE = 1024
SEED = 42


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 80)
print("ROUND-1 FEDERATED MODEL DIAGNOSTIC")
print("=" * 80)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False,
)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# LOAD GLOBAL SCALER
# ============================================================

print("\nLoading global scaler...")

scaler = np.load(
    SCALER_FILE,
    allow_pickle=True,
)

mean = scaler["mean"].astype(np.float32)
std = scaler["std"].astype(np.float32)

std = np.where(
    std < 1e-12,
    1.0,
    std,
).astype(np.float32)

print("Scaler loaded successfully.")
print("Features:", len(mean))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_sample(client):

    test_file = (
        DATA_DIR
        / client
        / "test_raw.csv"
    )

    print(f"\nReading sample from {client}...")

    # Read a larger candidate block first.
    candidate_size = SAMPLE_SIZE * 4

    df = pd.read_csv(
        test_file,
        usecols=FEATURES + ["label_id"],
        nrows=candidate_size,
        low_memory=False,
    )

    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=SEED,
        )

    return df


def preprocess(df):

    X = (
        df[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    y = (
        df["label_id"]
        .to_numpy(dtype=np.int32)
    )

    X = (X - mean) / std

    X = np.clip(
        X,
        -20.0,
        20.0,
    )

    X = X[..., np.newaxis]

    return X, y


def print_distribution(
    title,
    values,
):

    counts = np.bincount(
        values,
        minlength=len(CLASSES),
    )

    total = len(values)

    print(f"\n{title}")
    print("-" * 60)

    for i, class_name in enumerate(CLASSES):

        percentage = (
            counts[i] / total * 100
            if total > 0
            else 0
        )

        print(
            f"{i:>2} "
            f"{class_name:<18} "
            f"{counts[i]:>8,} "
            f"({percentage:>6.2f}%)"
        )

    return counts


# ============================================================
# DIAGNOSE EACH CLIENT
# ============================================================

all_true = []
all_pred = []

for client in CLIENTS:

    print("\n")
    print("=" * 80)
    print(f"CLIENT: {client}")
    print("=" * 80)

    df = load_sample(client)

    print(
        f"Sample records: {len(df):,}"
    )

    X, y_true = preprocess(df)

    print("Running prediction...")

    probabilities = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    confidence = np.max(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"\n{client} sample accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{np.mean(confidence) * 100:.2f}%"
    )

    print_distribution(
        "ACTUAL CLASS DISTRIBUTION",
        y_true,
    )

    print_distribution(
        "PREDICTED CLASS DISTRIBUTION",
        y_pred,
    )

    print("\nClassification Report")
    print("-" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
            digits=4,
        )
    )

    all_true.append(y_true)
    all_pred.append(y_pred)


# ============================================================
# COMBINED DIAGNOSTIC
# ============================================================

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

print("\n")
print("=" * 80)
print("COMBINED 3-DATASET DIAGNOSTIC")
print("=" * 80)

combined_accuracy = accuracy_score(
    all_true,
    all_pred,
)

print(
    f"\nCombined sample accuracy: "
    f"{combined_accuracy * 100:.2f}%"
)

actual_counts = print_distribution(
    "COMBINED ACTUAL DISTRIBUTION",
    all_true,
)

predicted_counts = print_distribution(
    "COMBINED PREDICTED DISTRIBUTION",
    all_pred,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(CLASSES))),
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{x}"
        for x in CLASSES
    ],
    columns=[
        f"Pred_{x}"
        for x in CLASSES
    ],
)

output_file = (
    PROJECT_ROOT
    / "report"
    / "round1_diagnostic_confusion_matrix.csv"
)

cm_df.to_csv(output_file)

print(
    "\nConfusion matrix saved to:"
)

print(output_file)


# ============================================================
# COLLAPSE CHECK
# ============================================================

print("\n")
print("=" * 80)
print("MODEL COLLAPSE CHECK")
print("=" * 80)

total_predictions = len(all_pred)

dominant_id = int(
    np.argmax(predicted_counts)
)

dominant_count = int(
    predicted_counts[dominant_id]
)

dominant_percentage = (
    dominant_count
    / total_predictions
    * 100
)

print(
    f"\nMost predicted class: "
    f"{CLASSES[dominant_id]}"
)

print(
    f"Predicted records   : "
    f"{dominant_count:,} / "
    f"{total_predictions:,}"
)

print(
    f"Prediction share    : "
    f"{dominant_percentage:.2f}%"
)

if dominant_percentage >= 80:

    print(
        "\nWARNING:"
        "\nThe global model is strongly collapsed "
        "toward one class."
    )

elif dominant_percentage >= 60:

    print(
        "\nWARNING:"
        "\nThe global model shows substantial "
        "prediction concentration."
    )

else:

    print(
        "\nNo extreme single-class collapse "
        "detected by this simple check."
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(
    "\nIMPORTANT:"
    "\nThis script did NOT train the model."
    "\nIt only evaluated the saved Round-1 checkpoint."
)