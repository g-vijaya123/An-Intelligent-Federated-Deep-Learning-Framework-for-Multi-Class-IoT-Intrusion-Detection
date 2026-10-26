from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "fl_3datasets"
)

OUTPUT_FILE = (
    DATA_DIR
    / "global_train_scaler.npz"
)

METADATA_FILE = (
    DATA_DIR
    / "global_train_scaler_metadata.json"
)

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


CLIENTS = [
    "UNSW",
    "CICIoT2023",
    "EdgeIIoTset",
]


# ============================================================
# DISPLAY HELPER
# ============================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# NUMERICALLY STABLE STREAMING STATISTICS
# ============================================================
#
# We use Welford-style batch combination.
#
# This is safer than:
#
#     variance = E[x^2] - E[x]^2
#
# especially for features with very large values.
#
# ============================================================

class StreamingStats:

    def __init__(self, n_features):

        self.count = 0

        self.mean = np.zeros(
            n_features,
            dtype=np.float64,
        )

        self.M2 = np.zeros(
            n_features,
            dtype=np.float64,
        )


    def update(self, X):

        if X.size == 0:
            return

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        batch_count = X.shape[0]

        batch_mean = np.mean(
            X,
            axis=0,
            dtype=np.float64,
        )

        centered = (
            X - batch_mean
        )

        batch_M2 = np.sum(
            centered * centered,
            axis=0,
            dtype=np.float64,
        )

        if self.count == 0:

            self.count = batch_count

            self.mean = batch_mean

            self.M2 = batch_M2

            return

        total_count = (
            self.count
            + batch_count
        )

        delta = (
            batch_mean
            - self.mean
        )

        self.mean = (
            self.mean
            + delta
            * batch_count
            / total_count
        )

        self.M2 = (
            self.M2
            + batch_M2
            + (
                delta * delta
                * self.count
                * batch_count
                / total_count
            )
        )

        self.count = total_count


    def finalize(self):

        if self.count == 0:

            raise ValueError(
                "No training records were processed."
            )

        variance = (
            self.M2
            / self.count
        )

        variance = np.maximum(
            variance,
            0.0,
        )

        std = np.sqrt(
            variance
        )

        constant_mask = (
            std < 1e-12
        )

        std[constant_mask] = 1.0

        return (
            self.mean.copy(),
            std,
            variance,
            constant_mask,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "BUILD GLOBAL TRAIN-ONLY SCALER"
    )

    print(
        "This scaler uses ONLY training records."
    )

    print(
        "Held-out test records are NOT used."
    )

    stats = StreamingStats(
        len(FEATURES)
    )

    client_counts = {}

    total_processed = 0

    # ========================================================
    # PROCESS EACH FEDERATED CLIENT
    # ========================================================

    for client in CLIENTS:

        section(
            f"PROCESSING CLIENT: {client}"
        )

        train_file = (
            DATA_DIR
            / client
            / "train_raw.csv"
        )

        if not train_file.exists():

            raise FileNotFoundError(
                f"Training file not found:\n"
                f"{train_file}"
            )

        client_rows = 0
        chunk_number = 0

        reader = pd.read_csv(
            train_file,
            usecols=FEATURES,
            chunksize=READ_CHUNK_SIZE,
            low_memory=False,
        )

        for df in reader:

            chunk_number += 1

            # --------------------------------------------
            # Convert safely to numeric
            # --------------------------------------------

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
                    dtype=np.float64
                )
            )

            # --------------------------------------------
            # Validate
            # --------------------------------------------

            if not np.all(
                np.isfinite(X)
            ):

                raise ValueError(
                    f"Non-finite values detected "
                    f"in {client}, chunk "
                    f"{chunk_number}"
                )

            # --------------------------------------------
            # Update global training statistics
            # --------------------------------------------

            stats.update(X)

            rows = len(X)

            client_rows += rows
            total_processed += rows

            if (
                chunk_number % 25 == 0
            ):

                print(
                    f"{client:<12} | "
                    f"Chunks={chunk_number:>4} | "
                    f"Client rows="
                    f"{client_rows:,} | "
                    f"Global rows="
                    f"{total_processed:,}"
                )

        client_counts[
            client
        ] = client_rows

        print(
            f"\n{client} completed:"
        )

        print(
            f"Training records = "
            f"{client_rows:,}"
        )

    # ========================================================
    # FINALIZE SCALER
    # ========================================================

    section(
        "FINALIZING GLOBAL SCALER"
    )

    (
        mean,
        std,
        variance,
        constant_mask,
    ) = stats.finalize()

    # ========================================================
    # VALIDATE EXPECTED TRAIN COUNT
    # ========================================================

    global_metadata_file = (
        DATA_DIR
        / "global_metadata.json"
    )

    expected_train_rows = None

    if global_metadata_file.exists():

        with open(
            global_metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            global_metadata = json.load(f)

        expected_train_rows = (
            global_metadata.get(
                "total_train_records"
            )
        )

    print(
        f"Processed training records: "
        f"{stats.count:,}"
    )

    if expected_train_rows is not None:

        print(
            f"Expected training records : "
            f"{expected_train_rows:,}"
        )

        if (
            stats.count
            != expected_train_rows
        ):

            raise ValueError(
                "Training record count mismatch."
            )

        print(
            "Training row count        : PASS"
        )

    # ========================================================
    # SAVE GLOBAL SCALER
    # ========================================================

    np.savez(
        OUTPUT_FILE,
        mean=mean.astype(
            np.float32
        ),
        std=std.astype(
            np.float32
        ),
        variance=variance.astype(
            np.float32
        ),
        features=np.array(
            FEATURES,
            dtype=str,
        ),
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {
        "description": (
            "Shared StandardScaler-style "
            "statistics calculated only from "
            "training partitions of UNSW-NB15, "
            "CICIoT2023 and Edge-IIoTset."
        ),

        "features": FEATURES,

        "number_of_features": len(
            FEATURES
        ),

        "training_records_used": int(
            stats.count
        ),

        "client_training_records": {
            key: int(value)
            for key, value
            in client_counts.items()
        },

        "mean": [
            float(x)
            for x in mean
        ],

        "std": [
            float(x)
            for x in std
        ],

        "variance": [
            float(x)
            for x in variance
        ],

        "constant_features": [
            FEATURES[i]
            for i, flag
            in enumerate(
                constant_mask
            )
            if flag
        ],

        "test_data_used": False,

        "scaling_formula": (
            "scaled_x = "
            "(x - training_mean) "
            "/ training_std"
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    # ========================================================
    # DISPLAY FEATURE STATISTICS
    # ========================================================

    section(
        "GLOBAL TRAINING FEATURE STATISTICS"
    )

    print(
        f"{'Feature':<22}"
        f"{'Mean':>18}"
        f"{'Std':>18}"
    )

    print(
        "-" * 58
    )

    for feature, m, s in zip(
        FEATURES,
        mean,
        std,
    ):

        print(
            f"{feature:<22}"
            f"{m:>18.6f}"
            f"{s:>18.6f}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    section(
        "GLOBAL SCALER COMPLETED"
    )

    print(
        f"Training records used : "
        f"{stats.count:,}"
    )

    print(
        "Test records used     : 0"
    )

    print(
        f"Features              : "
        f"{len(FEATURES)}"
    )

    print(
        f"Constant features     : "
        f"{int(np.sum(constant_mask))}"
    )

    print(
        "\nScaler saved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_FILE
    )

    print(
        "\nSTATUS: PASS"
    )

    print(
        "\nNext:"
    )

    print(
        "Train the 3-client federated "
        "CNN-BiGRU-Multi-Head Attention model."
    )


if __name__ == "__main__":
    main()