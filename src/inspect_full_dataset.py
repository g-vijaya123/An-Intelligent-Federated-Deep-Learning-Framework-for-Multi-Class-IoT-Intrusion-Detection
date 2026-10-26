import os
import pandas as pd
from collections import Counter

DATA_DIR = "data"

RAW_FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

# 49 raw UNSW-NB15 columns
RAW_COLUMNS = [
    "srcip",
    "sport",
    "dstip",
    "dsport",
    "proto",
    "state",
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sloss",
    "dloss",
    "service",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "swin",
    "dwin",
    "stcpb",
    "dtcpb",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "sjit",
    "djit",
    "stime",
    "ltime",
    "sinpkt",
    "dinpkt",
    "tcprtt",
    "synack",
    "ackdat",
    "is_sm_ips_ports",
    "ct_state_ttl",
    "ct_flw_http_mthd",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "attack_cat",
    "label",
]


def clean_attack_category(value, binary_label):
    """
    Convert raw attack-category values into consistent class names.
    """

    try:
        binary_label = int(float(binary_label))
    except (ValueError, TypeError):
        binary_label = None

    # Binary label 0 = Normal
    if binary_label == 0:
        return "Normal"

    value = str(value).strip()

    # Remove quotes / spaces
    value = value.replace('"', "")
    value = value.replace("'", "")
    value = value.strip()

    if value.lower() in ["", "nan", "none", "normal"]:
        return "Normal"

    # Dataset may contain Backdoors in some raw records.
    if value.lower() == "backdoors":
        return "Backdoor"

    return value


def main():

    print("\n" + "=" * 70)
    print("UNSW-NB15 FULL DATASET INSPECTION")
    print("=" * 70)

    # ----------------------------------------------------
    # 1. Check all four files
    # ----------------------------------------------------

    print("\n[1] CHECKING RAW FILES\n")

    available_files = []

    for filename in RAW_FILES:

        path = os.path.join(DATA_DIR, filename)

        if os.path.exists(path):

            size_mb = os.path.getsize(path) / (1024 * 1024)

            print(
                f"FOUND : {filename:<20} "
                f"{size_mb:>10.2f} MB"
            )

            available_files.append(path)

        else:

            print(f"MISSING: {filename}")

    if len(available_files) != 4:

        print("\nERROR:")
        print("All four UNSW-NB15 raw files are required.")
        return

    print("\nAll 4 raw dataset files found.")

    # ----------------------------------------------------
    # 2. Inspect number of columns
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] RAW DATASET SCHEMA")
    print("=" * 70)

    sample = pd.read_csv(
        available_files[0],
        header=None,
        nrows=5,
        low_memory=False
    )

    print("\nDetected raw columns:", sample.shape[1])
    print("Expected raw columns:", len(RAW_COLUMNS))

    if sample.shape[1] != len(RAW_COLUMNS):

        print("\nWARNING:")
        print(
            "Raw dataset column count does not match "
            "the expected 49-column schema."
        )
        return

    print("\nColumn schema is valid.")

    # ----------------------------------------------------
    # 3. Display all raw columns
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] ALL RAW COLUMNS")
    print("=" * 70)

    for number, column in enumerate(RAW_COLUMNS, start=1):

        print(
            f"{number:02d}. {column}"
        )

    # ----------------------------------------------------
    # 4. Scan complete dataset
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] SCANNING ALL FOUR FILES")
    print("=" * 70)

    total_records = 0

    class_counter = Counter()

    protocol_counter = Counter()
    service_counter = Counter()
    state_counter = Counter()

    # Chunk reading prevents huge RAM usage.
    chunk_size = 100000

    for file_number, path in enumerate(
        available_files,
        start=1
    ):

        filename = os.path.basename(path)

        print(
            f"\nReading file {file_number}/4: "
            f"{filename}"
        )

        file_records = 0

        reader = pd.read_csv(
            path,
            header=None,
            names=RAW_COLUMNS,
            chunksize=chunk_size,
            low_memory=False
        )

        for chunk_number, chunk in enumerate(
            reader,
            start=1
        ):

            total_records += len(chunk)
            file_records += len(chunk)

            # -----------------------------
            # Attack classes
            # -----------------------------

            cleaned_classes = [
                clean_attack_category(
                    attack,
                    label
                )
                for attack, label
                in zip(
                    chunk["attack_cat"],
                    chunk["label"]
                )
            ]

            class_counter.update(cleaned_classes)

            # -----------------------------
            # Categorical feature values
            # -----------------------------

            protocol_counter.update(
                chunk["proto"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            service_counter.update(
                chunk["service"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            state_counter.update(
                chunk["state"]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            print(
                f"  chunk {chunk_number:<3} "
                f"| records scanned: "
                f"{total_records:,}"
            )

        print(
            f"Completed {filename}: "
            f"{file_records:,} records"
        )

    # ----------------------------------------------------
    # 5. Results
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] COMPLETE DATASET RESULT")
    print("=" * 70)

    print(
        "\nTotal records:",
        f"{total_records:,}"
    )

    print(
        "Total raw columns:",
        len(RAW_COLUMNS)
    )

    # ----------------------------------------------------
    # 6. Classes
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] ALL DETECTED CLASSES")
    print("=" * 70)

    classes = sorted(
        class_counter.keys()
    )

    print(
        "\nTotal classes:",
        len(classes)
    )

    print()

    for index, class_name in enumerate(
        classes,
        start=1
    ):

        count = class_counter[class_name]

        percentage = (
            count / total_records
        ) * 100

        print(
            f"{index:02d}. "
            f"{class_name:<20} "
            f"{count:>10,} "
            f"({percentage:>6.3f}%)"
        )

    # ----------------------------------------------------
    # 7. Categorical values
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] CATEGORICAL FEATURE INFORMATION")
    print("=" * 70)

    print(
        "\nUnique protocols:",
        len(protocol_counter)
    )

    print(
        "Unique services:",
        len(service_counter)
    )

    print(
        "Unique states:",
        len(state_counter)
    )

    print("\nStates:")

    for value in sorted(
        state_counter.keys()
    ):

        print(
            f"  {value:<10} "
            f"{state_counter[value]:,}"
        )

    # ----------------------------------------------------
    # 8. Feature plan
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[8] NEW FULL-FEATURE MODEL PLAN")
    print("=" * 70)

    target_columns = [
        "attack_cat",
        "label"
    ]

    identifier_columns = [
        "srcip",
        "dstip"
    ]

    candidate_features = [
        column
        for column in RAW_COLUMNS
        if column not in target_columns
    ]

    recommended_features = [
        column
        for column in RAW_COLUMNS
        if column not in (
            target_columns
            + identifier_columns
        )
    ]

    print(
        "\nRaw columns:",
        len(RAW_COLUMNS)
    )

    print(
        "Target columns:",
        len(target_columns)
    )

    print(
        "Non-target candidate columns:",
        len(candidate_features)
    )

    print(
        "Recommended model features:",
        len(recommended_features)
    )

    print("\nTARGET / LABEL COLUMNS (NOT INPUT):")

    for feature in target_columns:
        print("  -", feature)

    print("\nIDENTIFIER COLUMNS EXCLUDED:")

    for feature in identifier_columns:
        print("  -", feature)

    print("\nRECOMMENDED MODEL FEATURES:")

    for index, feature in enumerate(
        recommended_features,
        start=1
    ):

        print(
            f"{index:02d}. {feature}"
        )

    # ----------------------------------------------------
    # 9. Important comparison
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("[9] OLD MODEL VS NEW MODEL")
    print("=" * 70)

    print("""
OLD MODEL
---------
Standard train/test dataset
42 preprocessing features
SelectKBest
32 selected features
10 classes
CNN-LSTM
5 simulated federated clients
FedAvg


NEW MODEL PLAN
--------------
All 4 raw UNSW-NB15 files
Full raw dataset
All detected attack categories
All usable traffic features
NO SelectKBest
NO 32-feature restriction
CNN-LSTM
5 simulated federated clients
FedAvg
SHAP explanations
Streamlit prediction
""")

    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()