from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "fl_3datasets_20percent"
)

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

CLIENTS = [
    "UNSW",
    "CICIoT2023",
    "EdgeIIoTset",
]

CHUNK_SIZE = 200_000

SCALER_FILE = (
    DATA_DIR
    / "global_train_scaler.npz"
)

METADATA_FILE = (
    DATA_DIR
    / "global_train_scaler_metadata.json"
)


# ============================================================
# PRINT HELPER
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# START
# ============================================================

section("BUILD GLOBAL TRAIN-ONLY SCALER — 20% DATASET")

print(f"Data directory : {DATA_DIR}")
print(f"Features       : {len(FEATURES)}")
print(f"Clients        : {len(CLIENTS)}")
print("Test data used : NO")


if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATA_DIR}"
    )


# ============================================================
# RUNNING STATISTICS
# ============================================================

# Welford-style batch combination.
# This avoids loading all ~8 million training rows into RAM.

global_count = 0

global_mean = np.zeros(
    len(FEATURES),
    dtype=np.float64,
)

global_M2 = np.zeros(
    len(FEATURES),
    dtype=np.float64,
)

client_counts = {}


# ============================================================
# PROCESS CLIENT TRAINING FILES
# ============================================================

for client in CLIENTS:

    section(f"PROCESSING TRAINING DATA: {client}")

    train_file = (
        DATA_DIR
        / client
        / "train_raw.csv"
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{train_file}"
        )

    client_count = 0
    chunk_number = 0

    for chunk in pd.read_csv(
        train_file,
        usecols=FEATURES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk_number += 1

        # ----------------------------------------------------
        # Convert safely to numeric
        # ----------------------------------------------------

        for feature in FEATURES:
            chunk[feature] = pd.to_numeric(
                chunk[feature],
                errors="coerce",
            )

        chunk = chunk.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Harmonized data was already validated,
        # but keep this safety check.
        if chunk.isna().any().any():

            bad_count = int(
                chunk.isna().sum().sum()
            )

            raise ValueError(
                f"{client}: found {bad_count} "
                f"NaN/Infinity values in training data."
            )

        X = chunk.to_numpy(
            dtype=np.float64
        )

        n = X.shape[0]

        if n == 0:
            continue

        # ----------------------------------------------------
        # Batch statistics
        # ----------------------------------------------------

        batch_mean = np.mean(
            X,
            axis=0,
        )

        batch_M2 = np.sum(
            (X - batch_mean) ** 2,
            axis=0,
        )

        # ----------------------------------------------------
        # Merge batch stats into global stats
        # ----------------------------------------------------

        if global_count == 0:

            global_mean = batch_mean.copy()
            global_M2 = batch_M2.copy()
            global_count = n

        else:

            delta = (
                batch_mean
                - global_mean
            )

            new_count = (
                global_count
                + n
            )

            global_mean = (
                global_mean
                + delta
                * n
                / new_count
            )

            global_M2 = (
                global_M2
                + batch_M2
                + (delta ** 2)
                * global_count
                * n
                / new_count
            )

            global_count = new_count

        client_count += n

        print(
            f"Chunk {chunk_number:>3} | "
            f"Rows={n:>9,} | "
            f"Client total={client_count:>10,} | "
            f"Global total={global_count:>10,}"
        )

    client_counts[client] = int(
        client_count
    )

    print(
        f"\n{client} training records used: "
        f"{client_count:,}"
    )


# ============================================================
# CALCULATE STANDARD DEVIATION
# ============================================================

section("CALCULATING GLOBAL SCALER")

if global_count <= 1:
    raise ValueError(
        "Not enough training records "
        "to calculate scaler."
    )

variance = (
    global_M2
    / global_count
)

variance = np.maximum(
    variance,
    0.0,
)

global_std = np.sqrt(
    variance
)

constant_mask = (
    global_std < 1e-12
)

constant_features = [
    FEATURES[i]
    for i in range(len(FEATURES))
    if constant_mask[i]
]

# Prevent divide-by-zero during transform.
safe_std = global_std.copy()

safe_std[
    constant_mask
] = 1.0


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nTraining records used: "
    f"{global_count:,}"
)

print(
    "Testing records used : 0"
)

print(
    f"Number of features   : "
    f"{len(FEATURES)}"
)

print(
    f"Constant features    : "
    f"{len(constant_features)}"
)

print("\nGLOBAL TRAIN FEATURE STATISTICS")
print("-" * 80)

for i, feature in enumerate(FEATURES):

    print(
        f"{feature:<20} "
        f"mean={global_mean[i]:>15.6f} "
        f"std={safe_std[i]:>15.6f}"
    )


# ============================================================
# VALIDATE EXPECTED TRAINING COUNT
# ============================================================

metadata_path = (
    DATA_DIR
    / "global_metadata.json"
)

expected_train_count = None

if metadata_path.exists():

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        dataset_metadata = json.load(f)

    expected_train_count = int(
        dataset_metadata.get(
            "global_train_records",
            0,
        )
    )

    print("\nROW COUNT VALIDATION")
    print("-" * 80)

    print(
        f"Expected training records: "
        f"{expected_train_count:,}"
    )

    print(
        f"Processed training records: "
        f"{global_count:,}"
    )

    if expected_train_count != global_count:

        raise ValueError(
            "Training row-count mismatch."
        )

    print("Row count: PASS")


# ============================================================
# SAVE SCALER
# ============================================================

np.savez(
    SCALER_FILE,
    mean=global_mean.astype(
        np.float32
    ),
    std=safe_std.astype(
        np.float32
    ),
    features=np.array(
        FEATURES,
        dtype=object,
    ),
    count=np.array(
        [global_count],
        dtype=np.int64,
    ),
)


# ============================================================
# SAVE METADATA
# ============================================================

scaler_metadata = {
    "description": (
        "Global StandardScaler statistics "
        "computed only from the training "
        "portion of the 20-percent "
        "three-dataset subset."
    ),
    "training_records_used": int(
        global_count
    ),
    "testing_records_used": 0,
    "features": FEATURES,
    "feature_count": len(
        FEATURES
    ),
    "client_training_counts": (
        client_counts
    ),
    "mean": {
        FEATURES[i]: float(
            global_mean[i]
        )
        for i in range(
            len(FEATURES)
        )
    },
    "std": {
        FEATURES[i]: float(
            safe_std[i]
        )
        for i in range(
            len(FEATURES)
        )
    },
    "constant_features": (
        constant_features
    ),
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        scaler_metadata,
        f,
        indent=2,
    )


# ============================================================
# FINAL CHECK
# ============================================================

section("SCALER BUILD COMPLETE")

print(
    f"Scaler saved:\n{SCALER_FILE}"
)

print(
    f"\nMetadata saved:\n{METADATA_FILE}"
)

print(
    "\nIMPORTANT:"
    "\nScaler was fitted using TRAINING DATA ONLY."
    "\nNo test records were used."
)

print("\nSTATUS: PASS")