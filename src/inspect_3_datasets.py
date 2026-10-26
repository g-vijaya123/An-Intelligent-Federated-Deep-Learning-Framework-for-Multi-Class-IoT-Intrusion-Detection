from pathlib import Path
from collections import Counter
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# UNSW-NB15 raw files
UNSW_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]

# CICIoT2023 - 63 merged CSV files
CIC_DIR = DATA_DIR / "CICIoT2023" / "MERGED_CSV"

# Edge-IIoTset
EDGE_FILE = (
    DATA_DIR
    / "EdgeIIoTset"
    / "archive (1)"
    / "Edge-IIoTset dataset"
    / "Selected dataset for ML and DL"
    / "DNN-EdgeIIoT-dataset.csv"
)


# ============================================================
# UNSW-NB15 COLUMN NAMES
# ============================================================

UNSW_COLUMNS = [
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_label(value):
    """
    Convert labels to a clean string.
    """
    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip()

    if value == "":
        return "UNKNOWN"

    return value


def print_counter(counter):
    """
    Print class counts sorted from highest to lowest.
    """
    if not counter:
        print("No labels found.")
        return

    total = sum(counter.values())

    for label, count in counter.most_common():
        percentage = (count / total) * 100 if total else 0

        print(
            f"{label:<35} "
            f"{count:>12,} "
            f"({percentage:>7.3f}%)"
        )


# ============================================================
# INSPECT UNSW-NB15
# ============================================================

def inspect_unsw():
    print("\n")
    print("=" * 90)
    print("DATASET 1: UNSW-NB15")
    print("=" * 90)

    existing_files = [p for p in UNSW_FILES if p.exists()]

    print(f"\nExpected raw files : {len(UNSW_FILES)}")
    print(f"Files found        : {len(existing_files)}")

    for path in UNSW_FILES:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"  [{status}] {path.name}")

    if not existing_files:
        print("\nERROR: UNSW raw files were not found.")
        return None

    print(f"\nRaw columns        : {len(UNSW_COLUMNS)}")
    print("\nColumns:")
    print(UNSW_COLUMNS)

    class_counts = Counter()
    total_rows = 0

    print("\nScanning UNSW-NB15 labels...")

    # Only load the two target columns while scanning.
    # This avoids loading all ~2.54M rows x 49 columns into RAM.
    for path in existing_files:

        file_rows = 0

        try:
            reader = pd.read_csv(
                path,
                header=None,
                names=UNSW_COLUMNS,
                usecols=["attack_cat", "label"],
                chunksize=200_000,
                low_memory=False,
            )

            for chunk in reader:

                attack = chunk["attack_cat"].apply(clean_label)

                binary = pd.to_numeric(
                    chunk["label"],
                    errors="coerce"
                ).fillna(0).astype(int)

                # Binary label 0 means Normal.
                attack.loc[binary == 0] = "Normal"

                # Normalize known naming variation.
                attack = attack.replace(
                    {
                        "Backdoors": "Backdoor",
                        "Backdoor": "Backdoor",
                    }
                )

                counts = attack.value_counts()

                class_counts.update(counts.to_dict())

                rows = len(chunk)

                file_rows += rows
                total_rows += rows

            print(
                f"  {path.name:<22} "
                f"{file_rows:>12,} rows"
            )

        except Exception as e:
            print(f"\nERROR reading {path.name}:")
            print(e)

    print("\nUNSW-NB15 total rows:")
    print(f"{total_rows:,}")

    print("\nUNSW-NB15 classes:")
    print_counter(class_counts)

    print(f"\nNumber of classes: {len(class_counts)}")

    return {
        "dataset": "UNSW-NB15",
        "files": len(existing_files),
        "rows": total_rows,
        "columns": len(UNSW_COLUMNS),
        "classes": dict(class_counts),
    }


# ============================================================
# INSPECT CICIoT2023
# ============================================================

def inspect_ciciot():
    print("\n")
    print("=" * 90)
    print("DATASET 2: CICIoT2023")
    print("=" * 90)

    if not CIC_DIR.exists():
        print(f"\nERROR: Folder not found:")
        print(CIC_DIR)
        return None

    files = sorted(CIC_DIR.glob("*.csv"))

    print(f"\nCSV files found: {len(files)}")

    if not files:
        print("\nERROR: No CICIoT2023 CSV files found.")
        return None

    try:
        sample = pd.read_csv(files[0], nrows=5)

    except Exception as e:
        print("\nERROR reading CICIoT2023 sample:")
        print(e)
        return None

    print(f"\nFirst file:")
    print(files[0].name)

    print(f"\nColumns: {len(sample.columns)}")

    print("\nColumn names:")
    print(sample.columns.tolist())

    if "Label" not in sample.columns:
        print("\nERROR: 'Label' column not found.")
        return None

    class_counts = Counter()
    total_rows = 0

    print("\nScanning CICIoT2023 labels...")

    for index, path in enumerate(files, start=1):

        file_rows = 0

        try:
            reader = pd.read_csv(
                path,
                usecols=["Label"],
                chunksize=250_000,
                low_memory=False,
            )

            for chunk in reader:

                labels = chunk["Label"].apply(clean_label)

                counts = labels.value_counts()

                class_counts.update(counts.to_dict())

                rows = len(chunk)

                file_rows += rows
                total_rows += rows

            print(
                f"  [{index:02d}/{len(files):02d}] "
                f"{path.name:<18} "
                f"{file_rows:>12,} rows"
            )

        except Exception as e:
            print(f"\nERROR reading {path.name}:")
            print(e)

    print("\nCICIoT2023 total rows:")
    print(f"{total_rows:,}")

    print("\nCICIoT2023 original labels:")
    print_counter(class_counts)

    print(f"\nNumber of original labels: {len(class_counts)}")

    return {
        "dataset": "CICIoT2023",
        "files": len(files),
        "rows": total_rows,
        "columns": len(sample.columns),
        "classes": dict(class_counts),
        "column_names": sample.columns.tolist(),
    }


# ============================================================
# INSPECT EDGE-IIoTset
# ============================================================

def inspect_edge():
    print("\n")
    print("=" * 90)
    print("DATASET 3: Edge-IIoTset")
    print("=" * 90)

    print("\nDataset file:")
    print(EDGE_FILE)

    if not EDGE_FILE.exists():
        print("\nERROR: Edge-IIoTset file not found.")
        return None

    try:
        sample = pd.read_csv(
            EDGE_FILE,
            nrows=5,
            low_memory=False,
        )

    except Exception as e:
        print("\nERROR reading Edge-IIoTset sample:")
        print(e)
        return None

    print(f"\nColumns: {len(sample.columns)}")

    print("\nColumn names:")
    print(sample.columns.tolist())

    if "Attack_type" not in sample.columns:
        print("\nERROR: 'Attack_type' column not found.")
        return None

    class_counts = Counter()
    binary_counts = Counter()

    total_rows = 0

    print("\nScanning Edge-IIoTset labels...")

    try:
        reader = pd.read_csv(
            EDGE_FILE,
            usecols=["Attack_label", "Attack_type"],
            chunksize=250_000,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(reader, start=1):

            attack_types = chunk["Attack_type"].apply(clean_label)

            attack_labels = chunk["Attack_label"].apply(clean_label)

            class_counts.update(
                attack_types.value_counts().to_dict()
            )

            binary_counts.update(
                attack_labels.value_counts().to_dict()
            )

            rows = len(chunk)
            total_rows += rows

            print(
                f"  Chunk {chunk_number:<3} "
                f"Total processed: {total_rows:>12,}"
            )

    except Exception as e:
        print("\nERROR scanning Edge-IIoTset:")
        print(e)

    print("\nEdge-IIoTset total rows:")
    print(f"{total_rows:,}")

    print("\nEdge-IIoTset Attack_type classes:")
    print_counter(class_counts)

    print(f"\nNumber of Attack_type classes: {len(class_counts)}")

    print("\nEdge-IIoTset binary Attack_label values:")
    print_counter(binary_counts)

    return {
        "dataset": "Edge-IIoTset",
        "files": 1,
        "rows": total_rows,
        "columns": len(sample.columns),
        "classes": dict(class_counts),
        "column_names": sample.columns.tolist(),
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

def print_final_summary(results):
    print("\n")
    print("=" * 90)
    print("FINAL 3-DATASET SUMMARY")
    print("=" * 90)

    valid_results = [
        result
        for result in results
        if result is not None
    ]

    if not valid_results:
        print("\nNo dataset could be inspected.")
        return

    total_records = sum(
        result["rows"]
        for result in valid_results
    )

    print("\nDataset overview:\n")

    for result in valid_results:

        print(
            f"{result['dataset']:<20} "
            f"Files={result['files']:<4} "
            f"Rows={result['rows']:<12,} "
            f"Columns={result['columns']:<4} "
            f"Classes={len(result['classes'])}"
        )

    print("\nCombined records across available datasets:")
    print(f"{total_records:,}")

    print("\nIMPORTANT:")
    print(
        "These datasets have different feature schemas and different "
        "attack taxonomies."
    )

    print(
        "They must NOT be directly concatenated before feature and "
        "label harmonization."
    )

    print(
        "The next preprocessing stage will build a defensible common "
        "representation for the three datasets."
    )

    print(
        "After harmonization, the three datasets can be used as "
        "heterogeneous federated data sources/clients for one global "
        "multi-class IDS model."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("THREE-DATASET IDS INSPECTION")
    print("UNSW-NB15 + CICIoT2023 + Edge-IIoTset")
    print("=" * 90)

    unsw_result = inspect_unsw()

    cic_result = inspect_ciciot()

    edge_result = inspect_edge()

    print_final_summary(
        [
            unsw_result,
            cic_result,
            edge_result,
        ]
    )

    print("\n")
    print("=" * 90)
    print("INSPECTION COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()