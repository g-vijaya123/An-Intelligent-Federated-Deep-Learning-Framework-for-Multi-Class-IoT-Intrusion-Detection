from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

OUTPUT = ROOT / "raw_dataset_schema_report.txt"
CHUNK_SIZE = 200_000

UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "service", "sload", "dload", "spkts",
    "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean",
    "dmean", "trans_depth", "response_body_len", "sjit",
    "djit", "stime", "ltime", "sinpkt", "dinpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

report = []


def log(value=""):
    report.append(str(value))
    print(value)


def inspect_schema(name, path, header="infer", names=None):
    log("\n" + "=" * 80)
    log(name + " - ORIGINAL FEATURE COLUMNS")
    log("=" * 80)

    if not path.exists():
        log(f"FILE NOT FOUND: {path}")
        return

    sample = pd.read_csv(
        path,
        header=header,
        names=names,
        nrows=1000,
        low_memory=False
    )

    log(f"File: {path}")
    log(f"Number of columns: {len(sample.columns)}")
    log("\nColumn names and detected data types:")

    for column in sample.columns:
        log(f"{column:<45} {sample[column].dtype}")


def scan_labels(paths, label_columns, header="infer", names=None):
    counters = {
        column: Counter()
        for column in label_columns
    }

    for path in paths:
        log(f"Scanning labels: {path.name}")

        reader = pd.read_csv(
            path,
            header=header,
            names=names,
            usecols=label_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )

        for chunk in reader:
            for column in label_columns:
                values = (
                    chunk[column]
                    .astype("string")
                    .fillna("<MISSING>")
                    .str.strip()
                )

                counters[column].update(
                    values.value_counts().to_dict()
                )

    for column, counts in counters.items():
        log(f"\nLabel distribution: {column}")

        for label, count in counts.most_common():
            log(f"{label:<45} {count:,}")


# ============================================================
# UNSW-NB15
# ============================================================

unsw_files = [
    DATA / f"UNSW-NB15_{i}.csv"
    for i in range(1, 5)
]

unsw_existing = [
    path for path in unsw_files
    if path.exists()
]

if unsw_existing:
    inspect_schema(
        "UNSW-NB15",
        unsw_existing[0],
        header=None,
        names=UNSW_COLUMNS
    )

    log("\nUNSW-NB15 ORIGINAL LABELS")

    scan_labels(
        unsw_existing,
        ["attack_cat", "label"],
        header=None,
        names=UNSW_COLUMNS
    )


# ============================================================
# CICIoT2023
# ============================================================

cic_dir = DATA / "CICIoT2023" / "MERGED_CSV"

cic_files = sorted(cic_dir.glob("*.csv"))

if cic_files:
    inspect_schema(
        "CICIoT2023",
        cic_files[0]
    )

    log("\nCICIoT2023 ORIGINAL LABELS")

    scan_labels(
        cic_files,
        ["Label"]
    )


# ============================================================
# EDGE-IIoTset
# ============================================================

edge_file = (
    DATA
    / "EdgeIIoTset"
    / "archive (1)"
    / "Edge-IIoTset dataset"
    / "Selected dataset for ML and DL"
    / "DNN-EdgeIIoT-dataset.csv"
)

if edge_file.exists():
    inspect_schema(
        "Edge-IIoTset",
        edge_file
    )

    log("\nEDGE-IIoTset ORIGINAL LABELS")

    scan_labels(
        [edge_file],
        ["Attack_label", "Attack_type"]
    )


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("\n" + "=" * 80)
print("SCHEMA INSPECTION COMPLETED")
print("=" * 80)
print(f"Report saved at: {OUTPUT}")