import pandas as pd
import joblib


CANDIDATE_FILE = "data/unseen_candidates_with_labels.csv"
TRAIN_FILE = "data/UNSW_NB15_training-set.csv"
TEST_FILE = "data/UNSW_NB15_testing-set.csv"

OUTPUT_FILE = "data/verified_unseen_with_labels.csv"

ARTIFACT_FILE = (
    "models/preprocessing_multiclass_stratified.joblib"
)


print("=" * 70)
print("REMOVING TRAIN / TEST OVERLAPPING RECORDS")
print("=" * 70)


# ============================================================
# LOAD MODEL FEATURE NAMES
# ============================================================

artifacts = joblib.load(
    ARTIFACT_FILE
)

feature_names = list(
    artifacts["all_feature_names"]
)


# ============================================================
# LOAD DATA
# ============================================================

candidates = pd.read_csv(
    CANDIDATE_FILE
)

train = pd.read_csv(
    TRAIN_FILE
)

test = pd.read_csv(
    TEST_FILE
)


print(
    f"Candidate records : {len(candidates)}"
)

print(
    f"Training records  : {len(train)}"
)

print(
    f"Testing records   : {len(test)}"
)


# ============================================================
# KEEP ONLY 42 FEATURES FOR COMPARISON
# ============================================================

candidate_features = candidates[
    feature_names
].copy()

train_features = train[
    feature_names
].copy()

test_features = test[
    feature_names
].copy()


reference = pd.concat(
    [
        train_features,
        test_features
    ],
    ignore_index=True
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(df):

    result = df.copy()

    for column in result.columns:

        if (
            result[column].dtype == "object"
            or str(result[column].dtype).startswith("string")
        ):

            result[column] = (
                result[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        else:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            ).round(6)

    return result


# ============================================================
# NORMALIZE
# ============================================================

candidate_normalized = normalize(
    candidate_features
)

reference_normalized = normalize(
    reference
)


# ============================================================
# CREATE ROW HASHES
# ============================================================

reference_hashes = set(
    pd.util.hash_pandas_object(
        reference_normalized,
        index=False
    )
)


candidate_hashes = (
    pd.util.hash_pandas_object(
        candidate_normalized,
        index=False
    )
)


# ============================================================
# FIND OVERLAPS
# ============================================================

is_overlap = candidate_hashes.isin(
    reference_hashes
)


overlapping = candidates[
    is_overlap
].copy()

verified_unseen = candidates[
    ~is_overlap
].copy()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("OVERLAP RESULTS")
print("=" * 70)

print(
    f"Total candidates     : {len(candidates)}"
)

print(
    f"Overlapping records  : {len(overlapping)}"
)

print(
    f"Non-overlapping      : {len(verified_unseen)}"
)


# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print()
print("NON-OVERLAPPING RECORDS BY CLASS")
print("-" * 70)

class_counts = (
    verified_unseen[
        "Actual_Category"
    ]
    .value_counts()
)

for class_name in artifacts["class_names"]:

    count = int(
        class_counts.get(
            class_name,
            0
        )
    )

    print(
        f"{class_name:<18} : {count}"
    )


# ============================================================
# SAVE VERIFIED RECORDS
# ============================================================

verified_unseen.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Verified records saved to: {OUTPUT_FILE}"
)

print()
print(
    "These records had no exact 42-feature match "
    "in the standard train/test partitions."
)