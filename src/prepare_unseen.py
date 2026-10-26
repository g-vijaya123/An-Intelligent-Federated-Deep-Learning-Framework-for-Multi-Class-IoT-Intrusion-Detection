import os
import pandas as pd
import joblib


# ============================================================
# FILE PATHS
# ============================================================

RAW_FILE = "data/UNSW-NB15_1.csv"

OUTPUT_FILE = "data/unseen_candidates_with_labels.csv"

ARTIFACT_FILE = (
    "models/preprocessing_multiclass_stratified.joblib"
)


# ============================================================
# RAW UNSW-NB15 COLUMN NAMES
# ============================================================

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
    "label"
]


print("=" * 70)
print("PREPARING LABELED RAW UNSW-NB15 CANDIDATES")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.isfile(RAW_FILE):
    raise FileNotFoundError(
        f"Raw dataset not found: {RAW_FILE}"
    )

if not os.path.isfile(ARTIFACT_FILE):
    raise FileNotFoundError(
        f"Preprocessing artifact not found: {ARTIFACT_FILE}"
    )


# ============================================================
# LOAD MODEL PREPROCESSING INFORMATION
# ============================================================

artifacts = joblib.load(
    ARTIFACT_FILE
)

required_features = list(
    artifacts["all_feature_names"]
)

model_classes = list(
    artifacts["class_names"]
)


print(
    f"Model features : {len(required_features)}"
)

print(
    f"Model classes  : {len(model_classes)}"
)


# ============================================================
# LOAD RAW UNSW-NB15 DATA
# ============================================================

raw = pd.read_csv(
    RAW_FILE,
    header=None,
    names=RAW_COLUMNS,
    low_memory=False
)


print(
    f"Raw records loaded : {len(raw)}"
)


if raw.shape[1] != len(RAW_COLUMNS):

    raise ValueError(
        f"Expected {len(RAW_COLUMNS)} raw columns, "
        f"but found {raw.shape[1]}."
    )


# ============================================================
# CONVERT IMPORTANT NUMERIC COLUMNS
# ============================================================

duration = pd.to_numeric(
    raw["dur"],
    errors="coerce"
).fillna(0)

spkts = pd.to_numeric(
    raw["spkts"],
    errors="coerce"
).fillna(0)

dpkts = pd.to_numeric(
    raw["dpkts"],
    errors="coerce"
).fillna(0)


# ============================================================
# CREATE RATE FEATURE
# ============================================================
# UNSW-NB15 rate:
# rate = (spkts + dpkts - 1) / dur
# If duration is 0, rate remains 0.
# ============================================================

raw["rate"] = 0.0

valid_duration = duration > 0

raw.loc[
    valid_duration,
    "rate"
] = (
    (
        spkts[valid_duration]
        + dpkts[valid_duration]
        - 1
    )
    / duration[valid_duration]
)

# Prevent negative values in unusual/invalid records
raw["rate"] = raw["rate"].clip(lower=0)
# ============================================================
# CLEAN BINARY LABEL
# ============================================================

raw["label"] = pd.to_numeric(
    raw["label"],
    errors="coerce"
).fillna(0).astype(int)


# ============================================================
# CLEAN ATTACK CATEGORY
# ============================================================

raw["attack_cat"] = (
    raw["attack_cat"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# NORMAL TRAFFIC
# ============================================================
# Raw normal records generally have label = 0.
# Standard model class name is "Normal".
# ============================================================

raw.loc[
    raw["label"] == 0,
    "attack_cat"
] = "Normal"


# ============================================================
# NORMALIZE CATEGORY NAMES
# ============================================================
# Raw UNSW file contains "Backdoors"
# but trained model uses "Backdoor".
# Therefore both names must match before sampling.
# ============================================================

raw["attack_cat"] = raw["attack_cat"].replace(
    {
        "Backdoors": "Backdoor"
    }
)


# Remove any remaining spaces
raw["attack_cat"] = (
    raw["attack_cat"]
    .astype(str)
    .str.strip()
)


# ============================================================
# SHOW AVAILABLE CATEGORIES
# ============================================================

print()
print("=" * 70)
print("AVAILABLE RAW CATEGORIES AFTER CLEANING")
print("=" * 70)

category_counts = (
    raw["attack_cat"]
    .value_counts()
)

print(
    category_counts.to_string()
)


# ============================================================
# SHOW MODEL CLASSES
# ============================================================

print()
print("=" * 70)
print("MODEL CLASSES")
print("=" * 70)

for class_name in model_classes:
    print(f" - {class_name}")


# ============================================================
# SELECT 20 RECORDS FROM EACH MODEL CLASS
# ============================================================

selected_parts = []

print()
print("=" * 70)
print("SELECTING CANDIDATE RECORDS")
print("=" * 70)


for category in model_classes:

    category_rows = raw[
        raw["attack_cat"] == category
    ].copy()

    available_count = len(
        category_rows
    )

    if available_count == 0:

        print(
            f"WARNING: No raw records found "
            f"for {category}"
        )

        continue


    sample_size = min(
        20,
        available_count
    )


    sampled = category_rows.sample(
        n=sample_size,
        random_state=42
    )


    selected_parts.append(
        sampled
    )


    print(
        f"Selected {sample_size:2d} records "
        f"from {category} "
        f"(available: {available_count})"
    )


# ============================================================
# CHECK WHETHER RECORDS WERE SELECTED
# ============================================================

if not selected_parts:

    raise ValueError(
        "No suitable candidate records were found."
    )


selected = pd.concat(
    selected_parts,
    ignore_index=True
)


# ============================================================
# CHECK REQUIRED MODEL FEATURES
# ============================================================

missing_features = [
    feature
    for feature in required_features
    if feature not in selected.columns
]


if missing_features:

    print()
    print("ERROR: Missing required model features:")

    for feature in missing_features:
        print(
            f" - {feature}"
        )

    raise ValueError(
        "Raw dataset could not be converted "
        "to the required model feature format."
    )


# ============================================================
# KEEP MODEL FEATURES + ACTUAL LABELS
# ============================================================

output_columns = (
    required_features
    + [
        "attack_cat",
        "label"
    ]
)


final_df = selected[
    output_columns
].copy()


# ============================================================
# RENAME GROUND-TRUTH COLUMNS
# ============================================================

final_df = final_df.rename(
    columns={
        "attack_cat":
            "Actual_Category",

        "label":
            "Actual_Binary_Label"
    }
)


# ============================================================
# VERIFY FINAL CLASS NAMES
# ============================================================

invalid_categories = sorted(
    set(
        final_df["Actual_Category"]
    )
    - set(model_classes)
)


if invalid_categories:

    print()
    print(
        "WARNING: Categories not supported "
        "by the trained model:"
    )

    for category in invalid_categories:
        print(
            f" - {category}"
        )


# ============================================================
# SAVE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("LABELED CANDIDATE DATA CREATED")
print("=" * 70)


print(
    f"Total candidate records : "
    f"{len(final_df)}"
)

print(
    f"Feature columns         : "
    f"{len(required_features)}"
)

print(
    f"Ground-truth columns    : 2"
)

print(
    f"Saved to                : "
    f"{OUTPUT_FILE}"
)


print()
print("Selected class distribution:")
print("-" * 70)

print(
    final_df[
        "Actual_Category"
    ]
    .value_counts()
    .reindex(
        model_classes,
        fill_value=0
    )
    .to_string()
)


# ============================================================
# EXPECTED RESULT
# ============================================================

expected_total = (
    len(model_classes) * 20
)


print()
print("=" * 70)

if len(final_df) == expected_total:

    print(
        f"SUCCESS: {len(model_classes)} classes × "
        f"20 records = {expected_total} candidates created."
    )

else:

    print(
        f"NOTE: Created {len(final_df)} candidates. "
        f"Expected maximum = {expected_total}."
    )

print("=" * 70)

print()
print(
    "NEXT STEP: Remove records that overlap "
    "with the existing train/test partitions."
)