from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = DATA_DIR / "harmonized_3datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNSW_FILES = [
    DATA_DIR / "UNSW-NB15_1.csv",
    DATA_DIR / "UNSW-NB15_2.csv",
    DATA_DIR / "UNSW-NB15_3.csv",
    DATA_DIR / "UNSW-NB15_4.csv",
]

CIC_DIR = DATA_DIR / "CICIoT2023" / "MERGED_CSV"

EDGE_FILE = (
    DATA_DIR
    / "EdgeIIoTset"
    / "archive (1)"
    / "Edge-IIoTset dataset"
    / "Selected dataset for ML and DL"
    / "DNN-EdgeIIoT-dataset.csv"
)

CHUNK_SIZE = 200_000


# ============================================================
# COMMON OUTPUT LABELS
# ============================================================

COMMON_CLASSES = [
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


# ============================================================
# COMMON FEATURES
# ============================================================
#
# These are not assumed to be identical raw fields.
# They represent harmonized network-behaviour concepts.
#
# Missing concepts in a dataset are represented by 0.
# Dataset_ID is intentionally included so the model can
# distinguish heterogeneous measurement domains.
#
# ============================================================

COMMON_FEATURES = [
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


# ============================================================
# UNSW COLUMN NAMES
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
# BASIC HELPERS
# ============================================================

def numeric(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def safe_series(df, column):
    if column in df.columns:
        return numeric(df[column])

    return pd.Series(
        np.zeros(len(df), dtype=np.float32),
        index=df.index,
    )


def zero_series(df):
    return pd.Series(
        np.zeros(len(df), dtype=np.float32),
        index=df.index,
    )


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# LABEL HARMONIZATION
# ============================================================

def map_unsw_label(attack_cat, binary_label):

    try:
        binary_label = int(float(binary_label))
    except Exception:
        binary_label = 1

    if binary_label == 0:
        return "Normal"

    label = clean_text(attack_cat).lower()

    if label == "normal":
        return "Normal"

    if label == "dos":
        return "DoS"

    if label == "reconnaissance":
        return "Reconnaissance"

    if label in {"backdoor", "backdoors"}:
        return "Backdoor"

    if label == "exploits":
        return "Exploits"

    if label == "fuzzers":
        return "Fuzzers"

    if label == "generic":
        return "Generic"

    # UNSW Analysis, Shellcode and Worms do not have a clean
    # one-to-one equivalent across all three datasets.
    return "OtherAttack"


def map_cic_label(value):

    label = clean_text(value).upper()

    if label == "BENIGN":
        return "Normal"

    if label.startswith("DDOS"):
        return "DDoS"

    if label.startswith("DOS"):
        return "DoS"

    if label.startswith("RECON"):
        return "Reconnaissance"

    if label == "VULNERABILITYSCAN":
        return "Reconnaissance"

    if label == "BACKDOOR_MALWARE":
        return "Backdoor"

    if label in {
        "SQLINJECTION",
        "COMMANDINJECTION",
        "XSS",
    }:
        return "Injection"

    if label == "DICTIONARYBRUTEFORCE":
        return "BruteForce"

    if label in {
        "MITM-ARPSPOOFING",
        "DNS_SPOOFING",
    }:
        return "MITM"

    if label.startswith("MIRAI"):
        return "Malware"

    if label in {
        "BROWSERHIJACKING",
        "UPLOADING_ATTACK",
    }:
        return "OtherAttack"

    return "OtherAttack"


def map_edge_label(value, binary_label=None):

    label = clean_text(value).lower()

    if binary_label is not None:
        try:
            if int(float(binary_label)) == 0:
                return "Normal"
        except Exception:
            pass

    if label == "normal":
        return "Normal"

    if label.startswith("ddos"):
        return "DDoS"

    if label in {
        "port_scanning",
        "fingerprinting",
        "vulnerability_scanner",
    }:
        return "Reconnaissance"

    if label == "backdoor":
        return "Backdoor"

    if label in {
        "sql_injection",
        "xss",
    }:
        return "Injection"

    if label == "password":
        return "BruteForce"

    if label == "mitm":
        return "MITM"

    if label == "ransomware":
        return "Malware"

    if label == "uploading":
        return "OtherAttack"

    return "OtherAttack"


# ============================================================
# UNSW FEATURE HARMONIZATION
# ============================================================

def harmonize_unsw_features(df):

    out = pd.DataFrame(index=df.index)

    out["duration"] = safe_series(df, "dur")

    total_packets = (
        safe_series(df, "spkts")
        + safe_series(df, "dpkts")
    )

    duration = safe_series(df, "dur")

    calculated_rate = np.where(
        duration.to_numpy() > 0,
        np.maximum(
            total_packets.to_numpy() - 1.0,
            0.0,
        )
        / duration.to_numpy(),
        0.0,
    )

    out["packet_rate"] = calculated_rate.astype(np.float32)

    out["src_packets"] = safe_series(df, "spkts")
    out["dst_packets"] = safe_series(df, "dpkts")

    out["src_bytes"] = safe_series(df, "sbytes")
    out["dst_bytes"] = safe_series(df, "dbytes")

    smean = safe_series(df, "smean")
    dmean = safe_series(df, "dmean")

    out["packet_size_mean"] = (
        (smean + dmean) / 2.0
    ).astype(np.float32)

    # Exact min/max/std packet size are not directly represented
    # in this raw UNSW schema.
    out["packet_size_min"] = zero_series(df)
    out["packet_size_max"] = zero_series(df)
    out["packet_size_std"] = zero_series(df)

    sttl = safe_series(df, "sttl")
    dttl = safe_series(df, "dttl")

    out["ttl"] = (
        (sttl + dttl) / 2.0
    ).astype(np.float32)

    # UNSW does not expose directly comparable per-flow TCP
    # flag-count fields in this representation.
    out["syn_count"] = zero_series(df)
    out["ack_count"] = zero_series(df)
    out["fin_count"] = zero_series(df)
    out["rst_count"] = zero_series(df)

    sinpkt = safe_series(df, "sinpkt")
    dinpkt = safe_series(df, "dinpkt")

    out["iat"] = (
        (sinpkt + dinpkt) / 2.0
    ).astype(np.float32)

    return out[COMMON_FEATURES]


# ============================================================
# CICIoT2023 FEATURE HARMONIZATION
# ============================================================

def harmonize_cic_features(df):

    out = pd.DataFrame(index=df.index)

    # CICIoT2023 merged features do not expose a directly
    # comparable flow duration field here.
    out["duration"] = zero_series(df)

    out["packet_rate"] = safe_series(df, "Rate")

    # No directly equivalent directional packet counts.
    out["src_packets"] = zero_series(df)
    out["dst_packets"] = zero_series(df)

    # No directly equivalent directional byte totals.
    out["src_bytes"] = zero_series(df)
    out["dst_bytes"] = zero_series(df)

    out["packet_size_mean"] = safe_series(df, "AVG")
    out["packet_size_min"] = safe_series(df, "Min")
    out["packet_size_max"] = safe_series(df, "Max")
    out["packet_size_std"] = safe_series(df, "Std")

    out["ttl"] = safe_series(df, "Time_To_Live")

    out["syn_count"] = safe_series(df, "syn_count")
    out["ack_count"] = safe_series(df, "ack_count")
    out["fin_count"] = safe_series(df, "fin_count")
    out["rst_count"] = safe_series(df, "rst_count")

    out["iat"] = safe_series(df, "IAT")

    return out[COMMON_FEATURES]


# ============================================================
# EDGE-IIoTset FEATURE HARMONIZATION
# ============================================================

def harmonize_edge_features(df):

    out = pd.DataFrame(index=df.index)

    # Edge-IIoTset DNN table is packet/protocol-oriented.
    # Many flow statistics used by UNSW/CIC are not directly
    # represented, so unsupported concepts remain zero.
    out["duration"] = zero_series(df)

    udp_delta = safe_series(df, "udp.time_delta")

    packet_rate = np.where(
        udp_delta.to_numpy() > 0,
        1.0 / udp_delta.to_numpy(),
        0.0,
    )

    # Avoid extreme reciprocal values.
    packet_rate = np.clip(
        packet_rate,
        0.0,
        1_000_000.0,
    )

    out["packet_rate"] = packet_rate.astype(np.float32)

    out["src_packets"] = zero_series(df)
    out["dst_packets"] = zero_series(df)

    out["src_bytes"] = zero_series(df)
    out["dst_bytes"] = zero_series(df)

    # tcp.len is the closest directly available packet/payload
    # length indicator in the selected DNN table.
    tcp_len = safe_series(df, "tcp.len")

    out["packet_size_mean"] = tcp_len
    out["packet_size_min"] = tcp_len
    out["packet_size_max"] = tcp_len
    out["packet_size_std"] = zero_series(df)

    out["ttl"] = zero_series(df)

    out["syn_count"] = safe_series(
        df,
        "tcp.connection.syn",
    )

    out["ack_count"] = safe_series(
        df,
        "tcp.flags.ack",
    )

    out["fin_count"] = safe_series(
        df,
        "tcp.connection.fin",
    )

    out["rst_count"] = safe_series(
        df,
        "tcp.connection.rst",
    )

    out["iat"] = udp_delta

    return out[COMMON_FEATURES]


# ============================================================
# SAVE CHUNK
# ============================================================

def save_chunk(
    features,
    labels,
    dataset_name,
    chunk_number,
):

    features = features.copy()

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    features.fillna(0.0, inplace=True)

    for column in COMMON_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        ).fillna(0.0).astype(np.float32)

    output = features.copy()

    output["label"] = labels.values
    output["dataset"] = dataset_name

    output_file = (
        OUTPUT_DIR
        / f"{dataset_name}_chunk_{chunk_number:04d}.csv"
    )

    output.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# PROCESS UNSW
# ============================================================

def process_unsw():

    print("\n" + "=" * 80)
    print("PROCESSING UNSW-NB15")
    print("=" * 80)

    label_counts = Counter()
    total_rows = 0
    chunk_number = 0

    usecols = [
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "spkts",
        "dpkts",
        "smean",
        "dmean",
        "sinpkt",
        "dinpkt",
        "attack_cat",
        "label",
    ]

    for path in UNSW_FILES:

        if not path.exists():
            print(f"MISSING: {path}")
            continue

        reader = pd.read_csv(
            path,
            header=None,
            names=UNSW_COLUMNS,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for df in reader:

            chunk_number += 1

            labels = pd.Series(
                [
                    map_unsw_label(a, b)
                    for a, b in zip(
                        df["attack_cat"],
                        df["label"],
                    )
                ],
                index=df.index,
                dtype="object",
            )

            features = harmonize_unsw_features(df)

            output_file = save_chunk(
                features,
                labels,
                "UNSW",
                chunk_number,
            )

            label_counts.update(
                labels.value_counts().to_dict()
            )

            total_rows += len(df)

            print(
                f"UNSW chunk {chunk_number:03d} | "
                f"{len(df):,} rows | "
                f"Total {total_rows:,} | "
                f"{output_file.name}"
            )

    return {
        "dataset": "UNSW-NB15",
        "rows": total_rows,
        "labels": dict(label_counts),
        "chunks": chunk_number,
    }


# ============================================================
# PROCESS CICIoT2023
# ============================================================

def process_cic():

    print("\n" + "=" * 80)
    print("PROCESSING CICIoT2023")
    print("=" * 80)

    files = sorted(CIC_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CICIoT2023 CSV files found in {CIC_DIR}"
        )

    required_columns = [
        "Rate",
        "Time_To_Live",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "AVG",
        "Min",
        "Max",
        "Std",
        "IAT",
        "Label",
    ]

    label_counts = Counter()
    total_rows = 0
    chunk_number = 0

    for file_index, path in enumerate(files, start=1):

        reader = pd.read_csv(
            path,
            usecols=required_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for df in reader:

            chunk_number += 1

            labels = df["Label"].apply(
                map_cic_label
            )

            features = harmonize_cic_features(df)

            output_file = save_chunk(
                features,
                labels,
                "CICIoT2023",
                chunk_number,
            )

            label_counts.update(
                labels.value_counts().to_dict()
            )

            total_rows += len(df)

            print(
                f"CIC file {file_index:02d}/{len(files):02d} | "
                f"chunk {chunk_number:04d} | "
                f"Total {total_rows:,} | "
                f"{output_file.name}"
            )

    return {
        "dataset": "CICIoT2023",
        "rows": total_rows,
        "labels": dict(label_counts),
        "chunks": chunk_number,
    }


# ============================================================
# PROCESS EDGE-IIoTset
# ============================================================

def process_edge():

    print("\n" + "=" * 80)
    print("PROCESSING EDGE-IIoTset")
    print("=" * 80)

    if not EDGE_FILE.exists():
        raise FileNotFoundError(
            f"Edge-IIoTset file not found: {EDGE_FILE}"
        )

    required_columns = [
        "udp.time_delta",
        "tcp.len",
        "tcp.connection.syn",
        "tcp.flags.ack",
        "tcp.connection.fin",
        "tcp.connection.rst",
        "Attack_label",
        "Attack_type",
    ]

    label_counts = Counter()
    total_rows = 0
    chunk_number = 0

    reader = pd.read_csv(
        EDGE_FILE,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for df in reader:

        chunk_number += 1

        labels = pd.Series(
            [
                map_edge_label(a, b)
                for a, b in zip(
                    df["Attack_type"],
                    df["Attack_label"],
                )
            ],
            index=df.index,
            dtype="object",
        )

        features = harmonize_edge_features(df)

        output_file = save_chunk(
            features,
            labels,
            "EdgeIIoTset",
            chunk_number,
        )

        label_counts.update(
            labels.value_counts().to_dict()
        )

        total_rows += len(df)

        print(
            f"Edge chunk {chunk_number:03d} | "
            f"{len(df):,} rows | "
            f"Total {total_rows:,} | "
            f"{output_file.name}"
        )

    return {
        "dataset": "Edge-IIoTset",
        "rows": total_rows,
        "labels": dict(label_counts),
        "chunks": chunk_number,
    }


# ============================================================
# PRINT LABEL COUNTS
# ============================================================

def print_label_counts(result):

    print("\n" + "-" * 80)
    print(result["dataset"])
    print("-" * 80)

    print(f"Rows   : {result['rows']:,}")
    print(f"Chunks : {result['chunks']}")

    total = sum(result["labels"].values())

    for label, count in sorted(
        result["labels"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        pct = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{label:<20} "
            f"{count:>12,} "
            f"({pct:>7.3f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("3-DATASET FEATURE + LABEL HARMONIZATION")
    print("=" * 80)

    print("\nCommon feature representation:")
    for i, feature in enumerate(COMMON_FEATURES, start=1):
        print(f"{i:02d}. {feature}")

    print("\nCommon attack taxonomy:")
    for i, label in enumerate(COMMON_CLASSES, start=1):
        print(f"{i:02d}. {label}")

    results = []

    results.append(
        process_unsw()
    )

    results.append(
        process_cic()
    )

    results.append(
        process_edge()
    )

    print("\n")
    print("=" * 80)
    print("HARMONIZATION SUMMARY")
    print("=" * 80)

    for result in results:
        print_label_counts(result)

    total_rows = sum(
        result["rows"]
        for result in results
    )

    combined_counts = Counter()

    for result in results:
        combined_counts.update(
            result["labels"]
        )

    print("\n" + "=" * 80)
    print("COMBINED 3-DATASET LABEL DISTRIBUTION")
    print("=" * 80)

    for label, count in combined_counts.most_common():

        pct = count / total_rows * 100

        print(
            f"{label:<20} "
            f"{count:>12,} "
            f"({pct:>7.3f}%)"
        )

    print("\nTotal harmonized records:")
    print(f"{total_rows:,}")

    metadata = {
        "common_features": COMMON_FEATURES,
        "common_classes": COMMON_CLASSES,
        "total_records": total_rows,
        "datasets": results,
        "combined_label_counts": dict(combined_counts),
        "important_note": (
            "Features are semantic harmonizations. "
            "Unsupported concepts are represented as zero. "
            "The three datasets remain identifiable as separate "
            "federated domains and are not treated as identical "
            "measurement schemas."
        ),
    }

    metadata_file = (
        OUTPUT_DIR
        / "harmonization_metadata.json"
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

    print("\nMetadata saved:")
    print(metadata_file)

    print("\nHarmonized chunks saved in:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("HARMONIZATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()