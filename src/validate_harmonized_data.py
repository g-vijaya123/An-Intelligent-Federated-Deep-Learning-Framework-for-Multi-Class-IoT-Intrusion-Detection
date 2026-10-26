from pathlib import Path
from collections import Counter
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
    / "harmonized_3datasets"
)

METADATA_FILE = (
    DATA_DIR
    / "harmonization_metadata.json"
)

CHUNK_SIZE = 200_000


EXPECTED_FEATURES = [
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


EXPECTED_CLASSES = [
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


EXPECTED_DATASETS = {
    "UNSW": 2_540_047,
    "CICIoT2023": 45_019_243,
    "EdgeIIoTset": 2_219_201,
}


# ============================================================
# HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_dataset_from_filename(path):

    name = path.name

    if name.startswith("UNSW_chunk_"):
        return "UNSW"

    if name.startswith("CICIoT2023_chunk_"):
        return "CICIoT2023"

    if name.startswith("EdgeIIoTset_chunk_"):
        return "EdgeIIoTset"

    return "UNKNOWN"


# ============================================================
# LOAD METADATA
# ============================================================

def validate_metadata():

    print_section("1. METADATA VALIDATION")

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    print("Metadata file found:")
    print(METADATA_FILE)

    metadata_features = metadata.get(
        "common_features",
        []
    )

    metadata_classes = metadata.get(
        "common_classes",
        []
    )

    print(
        f"\nFeatures in metadata : "
        f"{len(metadata_features)}"
    )

    print(
        f"Classes in metadata  : "
        f"{len(metadata_classes)}"
    )

    if metadata_features == EXPECTED_FEATURES:
        print("Feature list         : PASS")
    else:
        print("Feature list         : FAIL")
        print("Expected:")
        print(EXPECTED_FEATURES)
        print("Found:")
        print(metadata_features)

    if metadata_classes == EXPECTED_CLASSES:
        print("Class list           : PASS")
    else:
        print("Class list           : FAIL")
        print("Expected:")
        print(EXPECTED_CLASSES)
        print("Found:")
        print(metadata_classes)

    return metadata


# ============================================================
# VALIDATE ALL GENERATED CSV FILES
# ============================================================

def validate_files():

    print_section("2. GENERATED FILE VALIDATION")

    csv_files = sorted(
        DATA_DIR.glob("*_chunk_*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No harmonized chunk files found in:\n"
            f"{DATA_DIR}"
        )

    print(
        f"Total generated CSV chunks: "
        f"{len(csv_files)}"
    )

    file_counts = Counter()

    for path in csv_files:
        dataset = get_dataset_from_filename(path)
        file_counts[dataset] += 1

    print("\nChunk count by dataset:")

    for dataset, count in file_counts.items():
        print(
            f"{dataset:<15}: {count}"
        )

    return csv_files


# ============================================================
# FULL DATA VALIDATION
# ============================================================

def validate_data(csv_files):

    print_section("3. FULL DATA VALIDATION")

    total_rows = 0

    dataset_rows = Counter()
    label_counts = Counter()

    nan_count = Counter()
    inf_count = Counter()
    zero_count = Counter()

    feature_min = {
        feature: np.inf
        for feature in EXPECTED_FEATURES
    }

    feature_max = {
        feature: -np.inf
        for feature in EXPECTED_FEATURES
    }

    unexpected_labels = Counter()

    schema_errors = []

    processed_files = 0

    expected_columns = (
        EXPECTED_FEATURES
        + ["label", "dataset"]
    )

    for file_number, path in enumerate(
        csv_files,
        start=1,
    ):

        dataset_name = get_dataset_from_filename(
            path
        )

        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        file_rows = 0

        for df in reader:

            current_columns = list(df.columns)

            if current_columns != expected_columns:

                schema_errors.append(
                    {
                        "file": path.name,
                        "expected": expected_columns,
                        "found": current_columns,
                    }
                )

                continue

            rows = len(df)

            file_rows += rows
            total_rows += rows

            dataset_rows[dataset_name] += rows

            # ----------------------------------------
            # LABEL VALIDATION
            # ----------------------------------------

            current_labels = (
                df["label"]
                .astype(str)
                .value_counts()
            )

            label_counts.update(
                current_labels.to_dict()
            )

            for label, count in current_labels.items():

                if label not in EXPECTED_CLASSES:

                    unexpected_labels[label] += int(
                        count
                    )

            # ----------------------------------------
            # DATASET COLUMN VALIDATION
            # ----------------------------------------

            unique_dataset_values = (
                df["dataset"]
                .dropna()
                .astype(str)
                .unique()
            )

            expected_dataset_value = {
                "UNSW": "UNSW",
                "CICIoT2023": "CICIoT2023",
                "EdgeIIoTset": "EdgeIIoTset",
            }.get(dataset_name)

            if (
                len(unique_dataset_values) != 1
                or unique_dataset_values[0]
                != expected_dataset_value
            ):

                print(
                    "\nWARNING: dataset column mismatch"
                )

                print(
                    f"File: {path.name}"
                )

                print(
                    f"Values: "
                    f"{unique_dataset_values}"
                )

            # ----------------------------------------
            # FEATURE VALIDATION
            # ----------------------------------------

            for feature in EXPECTED_FEATURES:

                values = pd.to_numeric(
                    df[feature],
                    errors="coerce",
                ).to_numpy(
                    dtype=np.float64
                )

                nan_mask = np.isnan(values)
                inf_mask = np.isinf(values)

                nan_count[feature] += int(
                    nan_mask.sum()
                )

                inf_count[feature] += int(
                    inf_mask.sum()
                )

                finite_mask = np.isfinite(values)

                finite_values = values[
                    finite_mask
                ]

                if finite_values.size > 0:

                    zero_count[feature] += int(
                        np.count_nonzero(
                            finite_values == 0
                        )
                    )

                    local_min = float(
                        finite_values.min()
                    )

                    local_max = float(
                        finite_values.max()
                    )

                    if (
                        local_min
                        < feature_min[feature]
                    ):
                        feature_min[
                            feature
                        ] = local_min

                    if (
                        local_max
                        > feature_max[feature]
                    ):
                        feature_max[
                            feature
                        ] = local_max

        processed_files += 1

        if (
            processed_files % 25 == 0
            or processed_files
            == len(csv_files)
        ):

            print(
                f"Checked "
                f"{processed_files:>3}/"
                f"{len(csv_files)} files | "
                f"Rows checked: "
                f"{total_rows:,}"
            )

    return {
        "total_rows": total_rows,
        "dataset_rows": dataset_rows,
        "label_counts": label_counts,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_count": zero_count,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "unexpected_labels": unexpected_labels,
        "schema_errors": schema_errors,
    }


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_results(results, metadata):

    print_section("4. ROW COUNT VALIDATION")

    all_row_counts_pass = True

    for dataset, expected in (
        EXPECTED_DATASETS.items()
    ):

        actual = results[
            "dataset_rows"
        ].get(
            dataset,
            0,
        )

        status = (
            "PASS"
            if actual == expected
            else "FAIL"
        )

        if actual != expected:
            all_row_counts_pass = False

        print(
            f"{dataset:<15} "
            f"Expected={expected:>12,} | "
            f"Actual={actual:>12,} | "
            f"{status}"
        )

    expected_total = sum(
        EXPECTED_DATASETS.values()
    )

    actual_total = results[
        "total_rows"
    ]

    total_status = (
        "PASS"
        if actual_total == expected_total
        else "FAIL"
    )

    print(
        f"\nTOTAL           "
        f"Expected={expected_total:>12,} | "
        f"Actual={actual_total:>12,} | "
        f"{total_status}"
    )

    # ========================================================

    print_section("5. LABEL VALIDATION")

    total = results["total_rows"]

    for label in EXPECTED_CLASSES:

        count = results[
            "label_counts"
        ].get(
            label,
            0,
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{label:<20} "
            f"{count:>12,} "
            f"({percentage:>8.4f}%)"
        )

    if results["unexpected_labels"]:

        print("\nUnexpected labels found:")

        for label, count in (
            results[
                "unexpected_labels"
            ].items()
        ):

            print(
                f"{label:<30} "
                f"{count:,}"
            )

    else:
        print(
            "\nUnexpected labels: NONE"
        )

    # ========================================================

    print_section(
        "6. NaN / INFINITY VALIDATION"
    )

    total_nan = sum(
        results["nan_count"].values()
    )

    total_inf = sum(
        results["inf_count"].values()
    )

    for feature in EXPECTED_FEATURES:

        print(
            f"{feature:<22} "
            f"NaN={results['nan_count'][feature]:>8,} | "
            f"Inf={results['inf_count'][feature]:>8,}"
        )

    print(
        f"\nTotal NaN values     : "
        f"{total_nan:,}"
    )

    print(
        f"Total Infinity values: "
        f"{total_inf:,}"
    )

    # ========================================================

    print_section(
        "7. FEATURE RANGE + ZERO ANALYSIS"
    )

    for feature in EXPECTED_FEATURES:

        zeros = results[
            "zero_count"
        ][feature]

        zero_pct = (
            zeros / total * 100
            if total
            else 0
        )

        minimum = results[
            "feature_min"
        ][feature]

        maximum = results[
            "feature_max"
        ][feature]

        print(
            f"{feature:<22} "
            f"Min={minimum:>12.4f} | "
            f"Max={maximum:>14.4f} | "
            f"Zero={zero_pct:>7.2f}%"
        )

    # ========================================================

    print_section("8. SCHEMA VALIDATION")

    if results["schema_errors"]:

        print(
            f"Schema errors found: "
            f"{len(results['schema_errors'])}"
        )

        for error in (
            results[
                "schema_errors"
            ][:5]
        ):

            print(
                "\nFile:",
                error["file"],
            )

            print(
                "Found:",
                error["found"],
            )

    else:

        print(
            "All chunk files have the expected "
            "18 columns:"
        )

        print(
            "16 features + label + dataset"
        )

    # ========================================================

    print_section("9. FINAL VALIDATION STATUS")

    metadata_total = metadata.get(
        "total_records"
    )

    metadata_total_pass = (
        metadata_total
        == results["total_rows"]
    )

    schema_pass = (
        len(
            results[
                "schema_errors"
            ]
        )
        == 0
    )

    label_pass = (
        len(
            results[
                "unexpected_labels"
            ]
        )
        == 0
    )

    numeric_pass = (
        total_nan == 0
        and total_inf == 0
    )

    total_pass = (
        results["total_rows"]
        == expected_total
    )

    checks = {
        "Row counts": (
            all_row_counts_pass
            and total_pass
        ),
        "Metadata total": metadata_total_pass,
        "Schema": schema_pass,
        "Labels": label_pass,
        "NaN/Infinity": numeric_pass,
    }

    for name, passed in checks.items():

        print(
            f"{name:<20}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if all(checks.values()):

        print(
            "\nVALIDATION RESULT: PASS"
        )

        print(
            "Harmonized dataset is structurally "
            "ready for the train/test preparation stage."
        )

    else:

        print(
            "\nVALIDATION RESULT: FAIL"
        )

        print(
            "Do not start model training until "
            "the failed checks are corrected."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "3-DATASET HARMONIZED DATA VALIDATION"
    )

    print("Dataset directory:")
    print(DATA_DIR)

    metadata = validate_metadata()

    csv_files = validate_files()

    results = validate_data(
        csv_files
    )

    print_results(
        results,
        metadata,
    )


if __name__ == "__main__":
    main()