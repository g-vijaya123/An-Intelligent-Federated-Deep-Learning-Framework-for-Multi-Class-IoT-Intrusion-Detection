from pathlib import Path
from collections import Counter
import json
import hashlib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "harmonized_3datasets"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "fl_3datasets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_RATIO = 0.80
SEED = 42
READ_CHUNK_SIZE = 200_000


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


LABEL_TO_ID = {
    label: index
    for index, label in enumerate(CLASSES)
}


DATASETS = {
    "UNSW": {
        "pattern": "UNSW_chunk_*.csv",
        "client_id": 1,
    },

    "CICIoT2023": {
        "pattern": "CICIoT2023_chunk_*.csv",
        "client_id": 2,
    },

    "EdgeIIoTset": {
        "pattern": "EdgeIIoTset_chunk_*.csv",
        "client_id": 3,
    },
}


# ============================================================
# HELPERS
# ============================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def reset_output_file(path):

    if path.exists():
        path.unlink()


def append_csv(df, path):

    header = not path.exists()

    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
    )


# ============================================================
# DETERMINISTIC STRATIFIED SPLIT
# ============================================================
#
# We do not load the full 49.78M rows into memory.
#
# Each row receives a deterministic pseudo-random number based
# on:
#
#     dataset + source file + row number + class label + seed
#
# The hash is stable across runs.
#
# Because the same 80/20 rule is applied independently to
# every label, this approximates a stratified split while
# remaining streaming/chunk-friendly.
#
# ============================================================

def stable_train_mask(
    dataset_name,
    source_file,
    start_row,
    labels,
):

    mask = np.empty(
        len(labels),
        dtype=bool,
    )

    threshold = int(
        TRAIN_RATIO * 1_000_000
    )

    for i, label in enumerate(labels):

        row_number = start_row + i

        key = (
            f"{SEED}|"
            f"{dataset_name}|"
            f"{source_file}|"
            f"{row_number}|"
            f"{label}"
        )

        digest = hashlib.blake2b(
            key.encode("utf-8"),
            digest_size=8,
        ).digest()

        value = int.from_bytes(
            digest,
            byteorder="little",
            signed=False,
        )

        bucket = value % 1_000_000

        mask[i] = bucket < threshold

    return mask


# ============================================================
# RUNNING TRAIN STATISTICS
# ============================================================
#
# We calculate scaling statistics ONLY from training data.
#
# Uses:
#     count
#     sum
#     sum of squares
#
# Then:
#
# mean = sum / count
#
# variance =
#     (sum_of_squares / count) - mean^2
#
# std = sqrt(variance)
#
# ============================================================

class RunningStats:

    def __init__(self, n_features):

        self.n_features = n_features

        self.count = 0

        self.sum = np.zeros(
            n_features,
            dtype=np.float64,
        )

        self.sum_sq = np.zeros(
            n_features,
            dtype=np.float64,
        )


    def update(self, X):

        if len(X) == 0:
            return

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        self.count += X.shape[0]

        self.sum += np.sum(
            X,
            axis=0,
        )

        self.sum_sq += np.sum(
            np.square(X),
            axis=0,
        )


    def finalize(self):

        if self.count == 0:

            raise ValueError(
                "No training records were found."
            )

        mean = self.sum / self.count

        variance = (
            self.sum_sq / self.count
        ) - np.square(mean)

        variance = np.maximum(
            variance,
            0.0,
        )

        std = np.sqrt(variance)

        # Prevent division by zero.
        std = np.where(
            std < 1e-12,
            1.0,
            std,
        )

        return mean, std


# ============================================================
# PASS 1
# SPLIT DATA + CALCULATE TRAIN STATISTICS
# ============================================================

def split_dataset(
    dataset_name,
    config,
):

    section(
        f"CLIENT {config['client_id']} "
        f"- {dataset_name}"
    )

    files = sorted(
        INPUT_DIR.glob(
            config["pattern"]
        )
    )

    if not files:

        raise FileNotFoundError(
            f"No files found for {dataset_name}"
        )

    print(
        f"Source chunks: {len(files)}"
    )

    dataset_output = (
        OUTPUT_DIR
        / dataset_name
    )

    dataset_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_raw_file = (
        dataset_output
        / "train_raw.csv"
    )

    test_raw_file = (
        dataset_output
        / "test_raw.csv"
    )

    reset_output_file(
        train_raw_file
    )

    reset_output_file(
        test_raw_file
    )

    stats = RunningStats(
        len(FEATURES)
    )

    total_rows = 0
    train_rows = 0
    test_rows = 0

    train_labels = Counter()
    test_labels = Counter()

    processed_files = 0

    required_columns = (
        FEATURES
        + ["label", "dataset"]
    )

    for path in files:

        file_row_offset = 0

        reader = pd.read_csv(
            path,
            chunksize=READ_CHUNK_SIZE,
            low_memory=False,
        )

        for df in reader:

            missing = [
                col
                for col in required_columns
                if col not in df.columns
            ]

            if missing:

                raise ValueError(
                    f"\nMissing columns in "
                    f"{path.name}:\n"
                    f"{missing}"
                )

            # ----------------------------------------
            # Validate dataset identity
            # ----------------------------------------

            dataset_values = (
                df["dataset"]
                .astype(str)
                .unique()
            )

            if (
                len(dataset_values) != 1
                or dataset_values[0]
                != dataset_name
            ):

                raise ValueError(
                    f"Dataset identity mismatch "
                    f"in {path.name}"
                )

            # ----------------------------------------
            # Validate labels
            # ----------------------------------------

            labels = (
                df["label"]
                .astype(str)
                .to_numpy()
            )

            invalid_labels = set(
                labels
            ) - set(
                CLASSES
            )

            if invalid_labels:

                raise ValueError(
                    f"Unexpected labels found:\n"
                    f"{invalid_labels}"
                )

            # ----------------------------------------
            # Numeric features
            # ----------------------------------------

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
                .to_numpy(
                    dtype=np.float32
                )
            )

            # ----------------------------------------
            # Deterministic 80/20 split
            # ----------------------------------------

            train_mask = stable_train_mask(
                dataset_name,
                path.name,
                file_row_offset,
                labels,
            )

            test_mask = ~train_mask

            train_df = df.loc[
                train_mask,
                FEATURES + ["label"]
            ].copy()

            test_df = df.loc[
                test_mask,
                FEATURES + ["label"]
            ].copy()

            # Add numeric target.
            train_df["label_id"] = (
                train_df["label"]
                .map(LABEL_TO_ID)
                .astype(np.int16)
            )

            test_df["label_id"] = (
                test_df["label"]
                .map(LABEL_TO_ID)
                .astype(np.int16)
            )

            # ----------------------------------------
            # Save raw split
            # ----------------------------------------

            append_csv(
                train_df,
                train_raw_file,
            )

            append_csv(
                test_df,
                test_raw_file,
            )

            # ----------------------------------------
            # Training-only scaling statistics
            # ----------------------------------------

            stats.update(
                X[train_mask]
            )

            # ----------------------------------------
            # Counters
            # ----------------------------------------

            current_train_labels = Counter(
                labels[train_mask]
            )

            current_test_labels = Counter(
                labels[test_mask]
            )

            train_labels.update(
                current_train_labels
            )

            test_labels.update(
                current_test_labels
            )

            chunk_rows = len(df)

            total_rows += chunk_rows

            current_train_rows = int(
                np.sum(train_mask)
            )

            current_test_rows = int(
                np.sum(test_mask)
            )

            train_rows += (
                current_train_rows
            )

            test_rows += (
                current_test_rows
            )

            file_row_offset += (
                chunk_rows
            )

        processed_files += 1

        print(
            f"{dataset_name:<12} | "
            f"{processed_files:>3}/"
            f"{len(files):<3} chunks | "
            f"Total={total_rows:,} | "
            f"Train={train_rows:,} | "
            f"Test={test_rows:,}"
        )

    mean, std = stats.finalize()

    # ========================================================
    # SAVE SCALER
    # ========================================================

    scaler_file = (
        dataset_output
        / "scaler.npz"
    )

    np.savez(
        scaler_file,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        features=np.array(
            FEATURES,
            dtype=str,
        ),
    )

    # ========================================================
    # SAVE CLIENT METADATA
    # ========================================================

    metadata = {
        "dataset": dataset_name,
        "client_id": config["client_id"],
        "features": FEATURES,
        "classes": CLASSES,
        "label_to_id": LABEL_TO_ID,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_ratio_actual": (
            train_rows / total_rows
        ),
        "test_ratio_actual": (
            test_rows / total_rows
        ),
        "train_label_counts": dict(
            train_labels
        ),
        "test_label_counts": dict(
            test_labels
        ),
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "split_seed": SEED,
        "split_method": (
            "Deterministic streaming "
            "label-aware hash split"
        ),
        "important_note": (
            "Scaler statistics were calculated "
            "only from this client's training "
            "partition. Test records were not "
            "used to calculate scaling parameters."
        ),
    }

    metadata_file = (
        dataset_output
        / "client_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print("\nSplit complete:")

    print(
        f"Total : {total_rows:,}"
    )

    print(
        f"Train : {train_rows:,} "
        f"({train_rows / total_rows * 100:.2f}%)"
    )

    print(
        f"Test  : {test_rows:,} "
        f"({test_rows / total_rows * 100:.2f}%)"
    )

    print("\nScaler saved:")
    print(scaler_file)

    print("\nMetadata saved:")
    print(metadata_file)

    return metadata


# ============================================================
# VALIDATE CLASS COVERAGE
# ============================================================

def validate_class_coverage(
    metadata_list,
):

    section(
        "CLASS COVERAGE VALIDATION"
    )

    for metadata in metadata_list:

        dataset = metadata[
            "dataset"
        ]

        print(
            f"\n{dataset}"
        )

        print(
            "-" * 60
        )

        train_counts = metadata[
            "train_label_counts"
        ]

        test_counts = metadata[
            "test_label_counts"
        ]

        dataset_classes = sorted(
            set(train_counts.keys())
            | set(test_counts.keys())
        )

        for label in dataset_classes:

            train_count = int(
                train_counts.get(
                    label,
                    0,
                )
            )

            test_count = int(
                test_counts.get(
                    label,
                    0,
                )
            )

            status = (
                "PASS"
                if (
                    train_count > 0
                    and test_count > 0
                )
                else "CHECK"
            )

            print(
                f"{label:<20} "
                f"Train={train_count:>10,} | "
                f"Test={test_count:>9,} | "
                f"{status}"
            )


# ============================================================
# GLOBAL METADATA
# ============================================================

def save_global_metadata(
    metadata_list,
):

    section(
        "GLOBAL FEDERATED DATA SUMMARY"
    )

    total_rows = sum(
        m["total_rows"]
        for m in metadata_list
    )

    total_train = sum(
        m["train_rows"]
        for m in metadata_list
    )

    total_test = sum(
        m["test_rows"]
        for m in metadata_list
    )

    print(
        f"Total records : "
        f"{total_rows:,}"
    )

    print(
        f"Train records : "
        f"{total_train:,}"
    )

    print(
        f"Test records  : "
        f"{total_test:,}"
    )

    print("\nFederated clients:")

    for metadata in metadata_list:

        print(
            f"Client "
            f"{metadata['client_id']} -> "
            f"{metadata['dataset']} | "
            f"Train="
            f"{metadata['train_rows']:,} | "
            f"Test="
            f"{metadata['test_rows']:,}"
        )

    global_metadata = {
        "federated_setup": (
            "3-client cross-dataset "
            "non-IID simulated federated learning"
        ),
        "number_of_clients": 3,
        "features": FEATURES,
        "number_of_features": len(
            FEATURES
        ),
        "classes": CLASSES,
        "number_of_classes": len(
            CLASSES
        ),
        "label_to_id": LABEL_TO_ID,
        "total_records": total_rows,
        "total_train_records": total_train,
        "total_test_records": total_test,
        "clients": [
            {
                "client_id": m[
                    "client_id"
                ],
                "dataset": m[
                    "dataset"
                ],
                "total_rows": m[
                    "total_rows"
                ],
                "train_rows": m[
                    "train_rows"
                ],
                "test_rows": m[
                    "test_rows"
                ],
            }
            for m in metadata_list
        ],
        "methodology_note": (
            "The three datasets are retained "
            "as separate heterogeneous federated "
            "clients rather than concatenated into "
            "one centralized training dataset."
        ),
    }

    path = (
        OUTPUT_DIR
        / "global_metadata.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            global_metadata,
            f,
            indent=4,
        )

    print("\nGlobal metadata saved:")
    print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "PREPARE 3-DATASET FEDERATED LEARNING DATA"
    )

    print(
        f"Input directory:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print(
        "\nSplit: 80% train / "
        "20% held-out test"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Each dataset remains a separate "
        "federated client."
    )

    print(
        "Scaling statistics are calculated "
        "from training data only."
    )

    metadata_list = []

    for dataset_name, config in (
        DATASETS.items()
    ):

        metadata = split_dataset(
            dataset_name,
            config,
        )

        metadata_list.append(
            metadata
        )

    validate_class_coverage(
        metadata_list
    )

    save_global_metadata(
        metadata_list
    )

    section(
        "PREPARATION COMPLETED"
    )

    print(
        "Client 1 = UNSW-NB15"
    )

    print(
        "Client 2 = CICIoT2023"
    )

    print(
        "Client 3 = Edge-IIoTset"
    )

    print(
        "\nNext stage:"
    )

    print(
        "Build streaming TensorFlow datasets "
        "and train the federated "
        "CNN-BiGRU-Multi-Head Attention model."
    )


if __name__ == "__main__":
    main()