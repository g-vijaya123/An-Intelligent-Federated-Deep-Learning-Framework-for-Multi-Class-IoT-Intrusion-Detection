from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "harmonized_3datasets"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "fl_3datasets_20percent"
)

SAMPLE_FRACTION = 0.20
TRAIN_FRACTION = 0.80

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

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASSES)
}

DATASETS = {
    "UNSW": "UNSW_chunk_*.csv",
    "CICIoT2023": "CICIoT2023_chunk_*.csv",
    "EdgeIIoTset": "EdgeIIoTset_chunk_*.csv",
}

OUTPUT_COLUMNS = (
    FEATURES
    + [
        "label",
        "label_id",
    ]
)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def stable_hash_fraction(
    dataset_name,
    source_file,
    row_number,
    label,
    stage,
):
    """
    Deterministic pseudo-random number in [0, 1).

    Including dataset, source file, row number and label makes
    selection reproducible without loading the full dataset
    into memory.
    """

    text = (
        f"{SEED}|"
        f"{stage}|"
        f"{dataset_name}|"
        f"{source_file}|"
        f"{row_number}|"
        f"{label}"
    )

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
    )

    return value / float(2**64)


def append_dataframe(
    dataframe,
    output_file,
    write_header,
):
    if dataframe.empty:
        return write_header

    dataframe.to_csv(
        output_file,
        mode="a",
        index=False,
        header=write_header,
    )

    return False


# ============================================================
# VALIDATE INPUT
# ============================================================

section("20% STRATIFIED 3-DATASET PREPARATION")

print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Sample       : {SAMPLE_FRACTION * 100:.0f}% per class")
print(
    f"Within sample: "
    f"{TRAIN_FRACTION * 100:.0f}% train / "
    f"{(1 - TRAIN_FRACTION) * 100:.0f}% test"
)

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Harmonized input directory not found:\n{INPUT_DIR}"
    )


# ============================================================
# CLEAN/CREATE OUTPUT
# ============================================================

if OUTPUT_DIR.exists():

    print(
        "\nExisting 20% output directory found."
    )

    print(
        "Removing it so this run starts clean..."
    )

    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

global_original = 0
global_selected = 0
global_train = 0
global_test = 0

all_metadata = {
    "seed": SEED,
    "sample_fraction": SAMPLE_FRACTION,
    "train_fraction_within_sample": TRAIN_FRACTION,
    "features": FEATURES,
    "classes": CLASSES,
    "class_to_id": CLASS_TO_ID,
    "clients": {},
}


# ============================================================
# PROCESS EACH DATASET / FL CLIENT
# ============================================================

for dataset_name, pattern in DATASETS.items():

    section(f"PROCESSING {dataset_name}")

    files = sorted(
        INPUT_DIR.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No files found for "
            f"{dataset_name} using pattern:\n{pattern}"
        )

    print(
        f"Input chunks found: {len(files)}"
    )

    client_dir = (
        OUTPUT_DIR
        / dataset_name
    )

    client_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = (
        client_dir
        / "train_raw.csv"
    )

    test_file = (
        client_dir
        / "test_raw.csv"
    )

    selected_file = (
        client_dir
        / "selected_20percent.csv"
    )

    train_header = True
    test_header = True
    selected_header = True

    original_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    selected_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    train_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    test_counts = {
        class_name: 0
        for class_name in CLASSES
    }

    dataset_original = 0
    dataset_selected = 0
    dataset_train = 0
    dataset_test = 0

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for file_index, input_file in enumerate(
        files,
        start=1,
    ):

        print(
            f"\n[{file_index}/{len(files)}] "
            f"{input_file.name}"
        )

        file_row_offset = 0

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                input_file,
                chunksize=READ_CHUNK_SIZE,
                low_memory=False,
            ),
            start=1,
        ):

            required_columns = (
                FEATURES
                + [
                    "label",
                    "dataset",
                ]
            )

            missing = [
                column
                for column in required_columns
                if column not in chunk.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing columns in "
                    f"{input_file.name}: {missing}"
                )

            # -----------------------------------------------
            # Keep only expected labels
            # -----------------------------------------------

            chunk["label"] = (
                chunk["label"]
                .astype(str)
                .str.strip()
            )

            unexpected = sorted(
                set(chunk["label"].unique())
                - set(CLASSES)
            )

            if unexpected:
                raise ValueError(
                    f"Unexpected labels found in "
                    f"{input_file.name}: "
                    f"{unexpected}"
                )

            current_rows = len(chunk)

            dataset_original += current_rows

            # -----------------------------------------------
            # Original class counts
            # -----------------------------------------------

            original_value_counts = (
                chunk["label"]
                .value_counts()
            )

            for class_name, count in (
                original_value_counts.items()
            ):
                original_counts[class_name] += int(
                    count
                )

            # -----------------------------------------------
            # Deterministic 20% class-aware sampling
            # -----------------------------------------------

            selected_mask = np.zeros(
                current_rows,
                dtype=bool,
            )

            labels = (
                chunk["label"]
                .to_numpy()
            )

            for local_index in range(
                current_rows
            ):

                label = labels[local_index]

                global_row_number = (
                    file_row_offset
                    + local_index
                )

                value = stable_hash_fraction(
                    dataset_name=dataset_name,
                    source_file=input_file.name,
                    row_number=global_row_number,
                    label=label,
                    stage="sample20",
                )

                if value < SAMPLE_FRACTION:
                    selected_mask[
                        local_index
                    ] = True

            selected = (
                chunk.loc[
                    selected_mask,
                    FEATURES + ["label"]
                ]
                .copy()
            )

            file_row_offset += current_rows

            if selected.empty:

                print(
                    f"  chunk {chunk_number}: "
                    f"{current_rows:,} read, "
                    f"0 selected"
                )

                continue

            # -----------------------------------------------
            # Label IDs
            # -----------------------------------------------

            selected["label_id"] = (
                selected["label"]
                .map(CLASS_TO_ID)
                .astype(np.int32)
            )

            selected = selected[
                OUTPUT_COLUMNS
            ]

            # -----------------------------------------------
            # Save complete selected 20%
            # -----------------------------------------------

            selected_header = append_dataframe(
                selected,
                selected_file,
                selected_header,
            )

            selected_value_counts = (
                selected["label"]
                .value_counts()
            )

            for class_name, count in (
                selected_value_counts.items()
            ):
                selected_counts[
                    class_name
                ] += int(count)

            dataset_selected += len(
                selected
            )

            # -----------------------------------------------
            # 80/20 train/test split WITHIN selected subset
            # -----------------------------------------------

            train_mask = np.zeros(
                len(selected),
                dtype=bool,
            )

            selected_labels = (
                selected["label"]
                .to_numpy()
            )

            selected_indices = (
                selected.index
                .to_numpy()
            )

            for position in range(
                len(selected)
            ):

                label = (
                    selected_labels[
                        position
                    ]
                )

                original_index = int(
                    selected_indices[
                        position
                    ]
                )

                value = stable_hash_fraction(
                    dataset_name=dataset_name,
                    source_file=input_file.name,
                    row_number=original_index,
                    label=label,
                    stage="train_test",
                )

                if value < TRAIN_FRACTION:
                    train_mask[
                        position
                    ] = True

            train_part = (
                selected.iloc[
                    train_mask
                ]
                .copy()
            )

            test_part = (
                selected.iloc[
                    ~train_mask
                ]
                .copy()
            )

            # -----------------------------------------------
            # Save train
            # -----------------------------------------------

            if not train_part.empty:

                train_header = append_dataframe(
                    train_part,
                    train_file,
                    train_header,
                )

                values = (
                    train_part["label"]
                    .value_counts()
                )

                for class_name, count in (
                    values.items()
                ):
                    train_counts[
                        class_name
                    ] += int(count)

                dataset_train += len(
                    train_part
                )

            # -----------------------------------------------
            # Save test
            # -----------------------------------------------

            if not test_part.empty:

                test_header = append_dataframe(
                    test_part,
                    test_file,
                    test_header,
                )

                values = (
                    test_part["label"]
                    .value_counts()
                )

                for class_name, count in (
                    values.items()
                ):
                    test_counts[
                        class_name
                    ] += int(count)

                dataset_test += len(
                    test_part
                )

            print(
                f"  chunk {chunk_number}: "
                f"read={current_rows:,}, "
                f"selected={len(selected):,}, "
                f"train={len(train_part):,}, "
                f"test={len(test_part):,}"
            )

    # ========================================================
    # CLIENT VALIDATION
    # ========================================================

    section(
        f"{dataset_name} VALIDATION"
    )

    print(
        f"Original records : "
        f"{dataset_original:,}"
    )

    print(
        f"Selected records : "
        f"{dataset_selected:,}"
    )

    actual_percentage = (
        dataset_selected
        / dataset_original
        * 100
        if dataset_original
        else 0
    )

    print(
        f"Actual selected %: "
        f"{actual_percentage:.4f}%"
    )

    print(
        f"Train records    : "
        f"{dataset_train:,}"
    )

    print(
        f"Test records     : "
        f"{dataset_test:,}"
    )

    if (
        dataset_train
        + dataset_test
        != dataset_selected
    ):
        raise ValueError(
            f"{dataset_name}: "
            f"train + test != selected"
        )

    print("\nClass coverage")
    print("-" * 80)

    for class_name in CLASSES:

        original = (
            original_counts[
                class_name
            ]
        )

        selected_count = (
            selected_counts[
                class_name
            ]
        )

        train_count = (
            train_counts[
                class_name
            ]
        )

        test_count = (
            test_counts[
                class_name
            ]
        )

        if original > 0:

            coverage_status = (
                "PASS"
                if selected_count > 0
                else "FAIL"
            )

            print(
                f"{class_name:<18} "
                f"Original={original:>10,} | "
                f"Selected={selected_count:>9,} | "
                f"Train={train_count:>9,} | "
                f"Test={test_count:>9,} | "
                f"{coverage_status}"
            )

            if selected_count == 0:
                raise ValueError(
                    f"{dataset_name}: "
                    f"class {class_name} disappeared "
                    f"during 20% sampling."
                )

            # For datasets this large, every selected
            # class should normally appear in both splits.
            if (
                selected_count >= 2
                and (
                    train_count == 0
                    or test_count == 0
                )
            ):
                raise ValueError(
                    f"{dataset_name}: "
                    f"class {class_name} is missing "
                    f"from train or test."
                )

    # ========================================================
    # CLIENT METADATA
    # ========================================================

    client_metadata = {
        "dataset": dataset_name,
        "sample_fraction": SAMPLE_FRACTION,
        "train_fraction_within_sample": TRAIN_FRACTION,
        "original_records": dataset_original,
        "selected_records": dataset_selected,
        "train_records": dataset_train,
        "test_records": dataset_test,
        "original_class_counts": original_counts,
        "selected_class_counts": selected_counts,
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "features": FEATURES,
        "classes": CLASSES,
    }

    metadata_file = (
        client_dir
        / "client_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            client_metadata,
            f,
            indent=2,
        )

    all_metadata["clients"][
        dataset_name
    ] = client_metadata

    global_original += (
        dataset_original
    )

    global_selected += (
        dataset_selected
    )

    global_train += (
        dataset_train
    )

    global_test += (
        dataset_test
    )


# ============================================================
# GLOBAL VALIDATION
# ============================================================

section("GLOBAL 20% DATASET SUMMARY")

print(
    f"Original records : "
    f"{global_original:,}"
)

print(
    f"Selected records : "
    f"{global_selected:,}"
)

print(
    f"Training records : "
    f"{global_train:,}"
)

print(
    f"Testing records  : "
    f"{global_test:,}"
)

selected_percentage = (
    global_selected
    / global_original
    * 100
)

print(
    f"Selected percent : "
    f"{selected_percentage:.4f}%"
)

train_percentage = (
    global_train
    / global_selected
    * 100
)

print(
    f"Train of selected: "
    f"{train_percentage:.4f}%"
)

print(
    f"Test of selected : "
    f"{100 - train_percentage:.4f}%"
)

if (
    global_train
    + global_test
    != global_selected
):
    raise ValueError(
        "Global train + test does not "
        "equal selected records."
    )

all_metadata[
    "global_original_records"
] = global_original

all_metadata[
    "global_selected_records"
] = global_selected

all_metadata[
    "global_train_records"
] = global_train

all_metadata[
    "global_test_records"
] = global_test


# ============================================================
# SAVE GLOBAL METADATA
# ============================================================

global_metadata_file = (
    OUTPUT_DIR
    / "global_metadata.json"
)

with open(
    global_metadata_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_metadata,
        f,
        indent=2,
    )


# ============================================================
# FINAL STATUS
# ============================================================

section("20% STRATIFIED PREPARATION COMPLETED")

print(
    "All available dataset classes "
    "were checked for representation."
)

print(
    "All 16 harmonized features were retained."
)

print(
    "The selected subset was split into "
    "train and test partitions."
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nMetadata:\n{global_metadata_file}"
)

print("\nSTATUS: PASS")