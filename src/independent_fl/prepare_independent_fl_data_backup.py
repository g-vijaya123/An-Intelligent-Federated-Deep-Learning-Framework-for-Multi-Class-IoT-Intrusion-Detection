from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# Project layout:
# project/
#   data/...
#   src/independent_fl/prepare_independent_fl_data.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_ROOT = DATA_DIR / "independent_fl_v1"

SEED = 42
SAMPLE_FRACTION = 0.20
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
NUM_CLIENTS = 3
READ_CHUNK_SIZE = 100_000

UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smean", "dmean", "trans_depth", "response_body_len",
    "sjit", "djit", "stime", "ltime", "sinpkt", "dinpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

UNSW_FEATURES = [
    "sport", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss",
    "dloss", "sload", "dload", "spkts", "dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smean", "dmean", "trans_depth",
    "response_body_len", "sjit", "djit", "stime", "ltime",
    "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

UNSW_CLASSES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]

UNSW_LABEL_MAP = {
    "generic": "Generic",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",
    "shellcode": "Shellcode",
    "worms": "Worms",
}

CIC_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number",
    "cwr_flag_number", "ack_count", "syn_count", "fin_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH",
    "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv",
    "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Variance",
]

CIC_CLASSES = [
    "BENIGN",
    "DDOS-ICMP_FLOOD",
    "DDOS-UDP_FLOOD",
    "DDOS-TCP_FLOOD",
    "DDOS-PSHACK_FLOOD",
    "DDOS-SYN_FLOOD",
    "DDOS-RSTFINFLOOD",
    "DDOS-SYNONYMOUSIP_FLOOD",
    "DOS-UDP_FLOOD",
    "DOS-TCP_FLOOD",
    "DOS-SYN_FLOOD",
    "MIRAI-GREETH_FLOOD",
    "MIRAI-UDPPLAIN",
    "MIRAI-GREIP_FLOOD",
    "DDOS-ICMP_FRAGMENTATION",
    "VULNERABILITYSCAN",
    "MITM-ARPSPOOFING",
    "DDOS-UDP_FRAGMENTATION",
    "DDOS-ACK_FRAGMENTATION",
    "DNS_SPOOFING",
    "RECON-HOSTDISCOVERY",
    "RECON-OSSCAN",
    "RECON-PORTSCAN",
    "DOS-HTTP_FLOOD",
    "DDOS-HTTP_FLOOD",
    "DDOS-SLOWLORIS",
    "DICTIONARYBRUTEFORCE",
    "BROWSERHIJACKING",
    "COMMANDINJECTION",
    "SQLINJECTION",
    "XSS",
    "BACKDOOR_MALWARE",
    "RECON-PINGSWEEP",
    "UPLOADING_ATTACK",
]

EDGE_FEATURES = [
    "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "icmp.unused", "http.file_data",
    "http.content_length", "http.request.uri.query",
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "http.response", "http.tls_port",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin",
    "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.dstport", "tcp.flags",
    "tcp.flags.ack", "tcp.len", "tcp.seq", "tcp.srcport",
    "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name",
    "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request",
    "dns.retransmit_request_in", "mqtt.conflag.cleansess",
    "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msg_decoded_as", "mqtt.msgtype", "mqtt.proto_len",
    "mqtt.topic_len", "mqtt.ver", "mbtcp.len", "mbtcp.trans_id",
    "mbtcp.unit_id",
]

EDGE_CLASSES = [
    "Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_injection", "Password",
    "Vulnerability_scanner", "DDoS_TCP", "DDoS_HTTP", "Uploading",
    "Backdoor", "Port_Scanning", "XSS", "Ransomware", "MITM",
    "Fingerprinting",
]

DATASETS = {
    "UNSW_NB15": {
        "features": UNSW_FEATURES,
        "classes": UNSW_CLASSES,
        "kind": "unsw",
        "paths": [
            DATA_DIR / f"UNSW-NB15_{i}.csv" for i in range(1, 5)
        ],
    },
    "CICIoT2023": {
        "features": CIC_FEATURES,
        "classes": CIC_CLASSES,
        "kind": "cic",
        "paths": sorted((DATA_DIR / "CICIoT2023" / "MERGED_CSV").glob("*.csv")),
    },
    "EdgeIIoTset": {
        "features": EDGE_FEATURES,
        "classes": EDGE_CLASSES,
        "kind": "edge",
        "paths": [
            DATA_DIR / "EdgeIIoTset" / "archive (1)" /
            "Edge-IIoTset dataset" / "Selected dataset for ML and DL" /
            "DNN-EdgeIIoT-dataset.csv"
        ],
    },
}


def stable_seed(dataset_name: str, file_name: str, label: str, stage: str) -> int:
    raw = f"{SEED}|{dataset_name}|{file_name}|{label}|{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def stable_uniform_by_label(
    dataset_name: str,
    file_name: str,
    row_numbers: np.ndarray,
    labels: np.ndarray,
    stage: str,
) -> np.ndarray:
    """Vectorized deterministic pseudo-random values in [0, 1), per class."""
    output = np.empty(len(labels), dtype=np.float64)
    row_numbers = np.asarray(row_numbers, dtype=np.uint64)

    for label in np.unique(labels):
        mask = labels == label
        seed = np.uint64(stable_seed(dataset_name, file_name, str(label), stage))
        with np.errstate(over="ignore"):
            x = row_numbers[mask] + seed + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
        output[mask] = (x >> np.uint64(11)).astype(np.float64) / 9007199254740992.0

    return output


def read_dataset_chunks(dataset_name: str, config: dict):
    kind = config["kind"]
    paths = config["paths"]

    existing = [p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No input files found for {dataset_name}. Checked: {paths}"
        )

    if kind == "unsw":
        usecols = config["features"] + ["attack_cat", "label"]
        for path in existing:
            reader = pd.read_csv(
                path,
                header=None,
                names=UNSW_COLUMNS,
                usecols=usecols,
                chunksize=READ_CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                yield path, chunk

    elif kind == "cic":
        usecols = config["features"] + ["Label"]
        for path in existing:
            reader = pd.read_csv(
                path,
                usecols=usecols,
                chunksize=READ_CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                yield path, chunk

    elif kind == "edge":
        usecols = config["features"] + ["Attack_label", "Attack_type"]
        for path in existing:
            reader = pd.read_csv(
                path,
                usecols=usecols,
                chunksize=READ_CHUNK_SIZE,
                low_memory=False,
            )
            for chunk in reader:
                yield path, chunk

    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")


def extract_labels(kind: str, chunk: pd.DataFrame) -> np.ndarray:
    if kind == "unsw":
        binary = pd.to_numeric(chunk["label"], errors="coerce").fillna(1).to_numpy()
        attack = chunk["attack_cat"].astype("string").fillna("").str.strip()
        normalized = attack.str.lower().map(UNSW_LABEL_MAP)
        labels = np.empty(len(chunk), dtype=object)
        normal_mask = binary == 0
        labels[normal_mask] = "Normal"
        attack_mask = ~normal_mask
        labels[attack_mask] = normalized.to_numpy(dtype=object)[attack_mask]
        return labels

    if kind == "cic":
        raw = chunk["Label"].astype("string").str.strip().fillna("")
        labels = raw.to_numpy(dtype=object)
        labels[labels == ""] = None
        return labels

    if kind == "edge":
        binary = pd.to_numeric(chunk["Attack_label"], errors="coerce").fillna(-1).to_numpy()
        raw = chunk["Attack_type"].astype("string").str.strip().fillna("")
        labels = raw.to_numpy(dtype=object)
        labels[labels == ""] = None
        labels[binary == 0] = "Normal"
        return labels

    raise ValueError(f"Unsupported dataset kind: {kind}")


def append_csv(frame: pd.DataFrame, path: Path, header_state: dict) -> None:
    if frame.empty:
        return
    frame.to_csv(
        path,
        mode="a",
        index=False,
        header=not header_state.get(path, False),
    )
    header_state[path] = True


def make_empty_csv(path: Path, columns: list[str]) -> None:
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def process_dataset(dataset_name: str, config: dict, output_root: Path) -> dict:
    print("\n" + "=" * 88)
    print(f"PREPARING DATASET: {dataset_name}")
    print("=" * 88)

    features = config["features"]
    classes = config["classes"]
    label_to_id = {label: index for index, label in enumerate(classes)}
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    output_columns = features + ["label", "label_id"]
    client_paths = {
        client_id: dataset_dir / f"train_client_{client_id:02d}.csv"
        for client_id in range(1, NUM_CLIENTS + 1)
    }
    validation_path = dataset_dir / "validation.csv"
    test_path = dataset_dir / "test.csv"

    # Fresh output folder is guaranteed by the main() safety check.
    headers = {}
    original_counts = Counter()
    selected_counts = Counter()
    train_counts = Counter()
    validation_counts = Counter()
    test_counts = Counter()
    client_counts = Counter()

    original_rows = 0
    valid_label_rows = 0
    dropped_missing_labels = 0
    selected_rows = 0
    train_rows = 0
    validation_rows = 0
    test_rows = 0

    current_file = None
    file_row_offset = 0

    for path, chunk in read_dataset_chunks(dataset_name, config):
        if current_file != path:
            current_file = path
            file_row_offset = 0

        chunk_size = len(chunk)
        row_numbers = np.arange(
            file_row_offset,
            file_row_offset + chunk_size,
            dtype=np.uint64,
        )
        file_row_offset += chunk_size
        original_rows += chunk_size

        labels_all = extract_labels(config["kind"], chunk)
        valid_mask = np.array(
            [label is not None and str(label).strip() != "" for label in labels_all],
            dtype=bool,
        )
        dropped_missing_labels += int((~valid_mask).sum())

        if not valid_mask.any():
            continue

        valid_chunk = chunk.loc[valid_mask]
        labels = np.asarray(labels_all[valid_mask], dtype=object)
        valid_row_numbers = row_numbers[valid_mask]

        unknown = sorted(set(labels.tolist()) - set(classes))
        if unknown:
            raise ValueError(
                f"{dataset_name}: unexpected labels in {path.name}: {unknown}"
            )

        valid_label_rows += len(labels)
        original_counts.update(Counter(labels.tolist()))

        # Numeric conversion is dataset-specific: no common feature mapping is applied.
        X_frame = valid_chunk[features].apply(pd.to_numeric, errors="coerce")
        X_frame = X_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X_frame.to_numpy(dtype=np.float32, copy=True)

        sample_random = stable_uniform_by_label(
            dataset_name, path.name, valid_row_numbers, labels, "sample_20_percent"
        )
        sample_mask = sample_random < SAMPLE_FRACTION

        if not sample_mask.any():
            continue

        sampled_X = X[sample_mask]
        sampled_labels = labels[sample_mask]
        sampled_rows = valid_row_numbers[sample_mask]
        selected_counts.update(Counter(sampled_labels.tolist()))
        selected_rows += len(sampled_labels)

        split_random = stable_uniform_by_label(
            dataset_name, path.name, sampled_rows, sampled_labels, "train_val_test_split"
        )
        train_mask = split_random < TRAIN_FRACTION
        validation_mask = (
            (split_random >= TRAIN_FRACTION) &
            (split_random < TRAIN_FRACTION + VALIDATION_FRACTION)
        )
        test_mask = split_random >= (TRAIN_FRACTION + VALIDATION_FRACTION)

        # Train records are distributed only among clients belonging to this dataset.
        if train_mask.any():
            train_X = sampled_X[train_mask]
            train_y = sampled_labels[train_mask]
            train_rows_ids = sampled_rows[train_mask]
            client_random = stable_uniform_by_label(
                dataset_name, path.name, train_rows_ids, train_y, "simulated_client_assignment"
            )
            client_ids = np.minimum(
                (client_random * NUM_CLIENTS).astype(np.int32) + 1,
                NUM_CLIENTS,
            )

            for client_id in range(1, NUM_CLIENTS + 1):
                client_mask = client_ids == client_id
                if not client_mask.any():
                    continue
                part = pd.DataFrame(train_X[client_mask], columns=features)
                part["label"] = train_y[client_mask]
                part["label_id"] = np.fromiter(
                    (label_to_id[label] for label in train_y[client_mask]),
                    dtype=np.int32,
                    count=int(client_mask.sum()),
                )
                append_csv(part[output_columns], client_paths[client_id], headers)
                count = int(client_mask.sum())
                client_counts[str(client_id)] += count
                train_rows += count
                train_counts.update(Counter(train_y[client_mask].tolist()))

        if validation_mask.any():
            part = pd.DataFrame(sampled_X[validation_mask], columns=features)
            part["label"] = sampled_labels[validation_mask]
            part["label_id"] = np.fromiter(
                (label_to_id[label] for label in sampled_labels[validation_mask]),
                dtype=np.int32,
                count=int(validation_mask.sum()),
            )
            append_csv(part[output_columns], validation_path, headers)
            count = int(validation_mask.sum())
            validation_rows += count
            validation_counts.update(Counter(sampled_labels[validation_mask].tolist()))

        if test_mask.any():
            part = pd.DataFrame(sampled_X[test_mask], columns=features)
            part["label"] = sampled_labels[test_mask]
            part["label_id"] = np.fromiter(
                (label_to_id[label] for label in sampled_labels[test_mask]),
                dtype=np.int32,
                count=int(test_mask.sum()),
            )
            append_csv(part[output_columns], test_path, headers)
            count = int(test_mask.sum())
            test_rows += count
            test_counts.update(Counter(sampled_labels[test_mask].tolist()))

        print(
            f"{path.name}: read={chunk_size:,}, valid={len(labels):,}, "
            f"selected={int(sample_mask.sum()):,}, "
            f"train={int(train_mask.sum()):,}, "
            f"validation={int(validation_mask.sum()):,}, "
            f"test={int(test_mask.sum()):,}"
        )

    for client_id, path in client_paths.items():
        make_empty_csv(path, output_columns)
    make_empty_csv(validation_path, output_columns)
    make_empty_csv(test_path, output_columns)

    if selected_rows == 0 or train_rows == 0 or validation_rows == 0 or test_rows == 0:
        raise RuntimeError(
            f"{dataset_name}: one or more data splits are empty. "
            "Check the raw data and label mapping."
        )

    if train_rows + validation_rows + test_rows != selected_rows:
        raise RuntimeError(f"{dataset_name}: split row counts do not add up.")

    missing_classes = [
        label for label, count in original_counts.items() if count > 0
        and selected_counts.get(label, 0) == 0
    ]
    if missing_classes:
        raise RuntimeError(
            f"{dataset_name}: classes disappeared during 20% sampling: {missing_classes}"
        )

    metadata = {
        "dataset": dataset_name,
        "seed": SEED,
        "sample_fraction_target": SAMPLE_FRACTION,
        "train_fraction_of_selected": TRAIN_FRACTION,
        "validation_fraction_of_selected": VALIDATION_FRACTION,
        "test_fraction_of_selected": TEST_FRACTION,
        "number_of_simulated_clients": NUM_CLIENTS,
        "feature_columns": features,
        "number_of_features": len(features),
        "class_names": classes,
        "number_of_classes": len(classes),
        "label_to_id": label_to_id,
        "original_records_read": int(original_rows),
        "records_with_valid_labels": int(valid_label_rows),
        "records_dropped_for_missing_labels": int(dropped_missing_labels),
        "selected_20_percent_records": int(selected_rows),
        "actual_selected_fraction_of_valid_labeled_records": (
            float(selected_rows / valid_label_rows) if valid_label_rows else 0.0
        ),
        "train_records": int(train_rows),
        "validation_records": int(validation_rows),
        "test_records": int(test_rows),
        "actual_train_fraction": float(train_rows / selected_rows),
        "actual_validation_fraction": float(validation_rows / selected_rows),
        "actual_test_fraction": float(test_rows / selected_rows),
        "original_class_counts": {k: int(v) for k, v in original_counts.items()},
        "selected_class_counts": {k: int(v) for k, v in selected_counts.items()},
        "train_class_counts": {k: int(train_counts.get(k, 0)) for k in classes},
        "validation_class_counts": {k: int(validation_counts.get(k, 0)) for k in classes},
        "test_class_counts": {k: int(test_counts.get(k, 0)) for k in classes},
        "train_client_counts": {
            str(i): int(client_counts.get(str(i), 0))
            for i in range(1, NUM_CLIENTS + 1)
        },
        "client_train_files": {
            str(i): client_paths[i].name for i in range(1, NUM_CLIENTS + 1)
        },
        "validation_file": validation_path.name,
        "test_file": test_path.name,
        "sampling_method": (
            "Deterministic class-aware hash sampling; approximately 20% per class. "
            "The selected subset is then split by class-aware hash into 70/15/15."
        ),
        "client_method": (
            "Training records only are assigned to three simulated clients within "
            "this dataset. Validation and test records are not assigned to clients."
        ),
        "feature_handling": (
            "Dataset-native numeric columns are retained. Target columns and raw "
            "text/IP/payload columns are excluded. Non-numeric or non-finite feature "
            "values are converted to zero."
        ),
    }

    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nDATASET SUMMARY")
    print(f"Original rows read       : {original_rows:,}")
    print(f"Valid labeled rows       : {valid_label_rows:,}")
    print(f"Rows dropped (no label)  : {dropped_missing_labels:,}")
    print(f"Selected 20% rows        : {selected_rows:,}")
    print(f"Train rows across clients: {train_rows:,}")
    print(f"Validation rows          : {validation_rows:,}")
    print(f"Test rows                : {test_rows:,}")
    print(f"Feature count            : {len(features)}")
    print(f"Class count              : {len(classes)}")
    print(f"Client counts            : {dict(client_counts)}")
    print(f"Metadata                 : {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare independent 20% dataset subsets for three separate FL experiments."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate only data/independent_fl_v1. Existing V5 data is not touched.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("INDEPENDENT DATASET PREPARATION FOR THREE FEDERATED EXPERIMENTS")
    print("=" * 88)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {OUTPUT_ROOT}")
    print("Sampling     : 20% per class (deterministic, approximate)")
    print("Split        : 70% train / 15% validation / 15% test")
    print(f"FL clients   : {NUM_CLIENTS} simulated clients per dataset")
    print("No cross-dataset feature harmonization or model aggregation.")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"\nOutput already exists: {OUTPUT_ROOT}\n"
                "To intentionally rebuild this NEW output folder, rerun with --overwrite.\n"
                "This script never deletes data/fl_3datasets_20percent or V5 model files."
            )
        print(f"\nRemoving only the new output folder: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for dataset_name, config in DATASETS.items():
        summaries[dataset_name] = process_dataset(
            dataset_name,
            config,
            OUTPUT_ROOT,
        )

    global_summary = {
        "experiment": "Three independent federated learning experiments",
        "datasets": summaries,
        "cross_dataset_aggregation": False,
        "number_of_simulated_clients_per_dataset": NUM_CLIENTS,
        "sampling_fraction": SAMPLE_FRACTION,
        "split": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
    }
    summary_path = OUTPUT_ROOT / "global_summary.json"
    summary_path.write_text(json.dumps(global_summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("PREPARATION COMPLETED")
    print("=" * 88)
    print(f"Prepared datasets: {', '.join(summaries.keys())}")
    print(f"Summary          : {summary_path}")
    print("Next: run train_independent_fl.py.")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
