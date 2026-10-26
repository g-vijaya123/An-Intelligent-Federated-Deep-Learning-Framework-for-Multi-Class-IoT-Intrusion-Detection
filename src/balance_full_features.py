import os
import gc
import numpy as np
from collections import Counter
from sklearn.utils import resample

DATA_DIR = os.path.join("data", "processed_full")
OUTPUT_DIR = os.path.join("data", "processed_full_balanced")

RANDOM_STATE = 42

# Controlled target per class.
# We do NOT make every class as large as the original Normal class.
TARGET_PER_CLASS = 100_000


def print_distribution(title, y):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    counts = Counter(y.tolist())

    for cls in sorted(counts):
        print(
            f"Class {cls}: "
            f"{counts[cls]:,}"
        )

    print(
        "\nTotal:",
        f"{len(y):,}"
    )


def main():

    print("=" * 70)
    print("FULL-FEATURE CONTROLLED CLASS BALANCING")
    print("=" * 70)

    # --------------------------------------------------
    # Load processed training data
    # --------------------------------------------------

    x_path = os.path.join(
        DATA_DIR,
        "X_train.npy"
    )

    y_path = os.path.join(
        DATA_DIR,
        "y_train.npy"
    )

    if not os.path.exists(x_path):
        raise FileNotFoundError(x_path)

    if not os.path.exists(y_path):
        raise FileNotFoundError(y_path)

    print("\nLoading processed training data...")

    # mmap avoids immediately duplicating the full array in RAM.
    X = np.load(
        x_path,
        mmap_mode="r"
    )

    y = np.load(
        y_path
    )

    print(
        "X_train shape:",
        X.shape
    )

    print(
        "y_train shape:",
        y.shape
    )

    print_distribution(
        "BEFORE BALANCING",
        y
    )

    # --------------------------------------------------
    # Controlled resampling
    # --------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    X_parts = []
    y_parts = []

    classes = np.unique(y)

    print("\nBalancing classes...")

    for cls in classes:

        indices = np.where(
            y == cls
        )[0]

        original_count = len(indices)

        if original_count > TARGET_PER_CLASS:

            # Controlled undersampling
            selected_indices = rng.choice(
                indices,
                size=TARGET_PER_CLASS,
                replace=False
            )

            action = "UNDERSAMPLED"

        elif original_count < TARGET_PER_CLASS:

            # Controlled random oversampling.
            # This duplicates existing minority examples rather than
            # generating synthetic interpolated categorical values.
            selected_indices = rng.choice(
                indices,
                size=TARGET_PER_CLASS,
                replace=True
            )

            action = "OVERSAMPLED"

        else:

            selected_indices = indices
            action = "UNCHANGED"

        X_class = np.asarray(
            X[selected_indices],
            dtype=np.float32
        )

        y_class = np.full(
            TARGET_PER_CLASS,
            cls,
            dtype=np.int16
        )

        X_parts.append(
            X_class
        )

        y_parts.append(
            y_class
        )

        print(
            f"Class {cls}: "
            f"{original_count:,} -> "
            f"{TARGET_PER_CLASS:,} "
            f"[{action}]"
        )

    # --------------------------------------------------
    # Combine
    # --------------------------------------------------

    print("\nCombining balanced classes...")

    X_balanced = np.concatenate(
        X_parts,
        axis=0
    )

    y_balanced = np.concatenate(
        y_parts,
        axis=0
    )

    del X_parts
    del y_parts

    gc.collect()

    # --------------------------------------------------
    # Shuffle
    # --------------------------------------------------

    print("Shuffling balanced dataset...")

    permutation = rng.permutation(
        len(y_balanced)
    )

    X_balanced = X_balanced[
        permutation
    ]

    y_balanced = y_balanced[
        permutation
    ]

    print_distribution(
        "AFTER BALANCING",
        y_balanced
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    x_output = os.path.join(
        OUTPUT_DIR,
        "X_train_balanced.npy"
    )

    y_output = os.path.join(
        OUTPUT_DIR,
        "y_train_balanced.npy"
    )

    print("\nSaving balanced training data...")

    np.save(
        x_output,
        X_balanced.astype(np.float32)
    )

    np.save(
        y_output,
        y_balanced.astype(np.int16)
    )

    print("\n" + "=" * 70)
    print("BALANCING COMPLETE")
    print("=" * 70)

    print(
        "X balanced:",
        X_balanced.shape
    )

    print(
        "y balanced:",
        y_balanced.shape
    )

    print(
        "Features:",
        X_balanced.shape[1]
    )

    print(
        "Classes:",
        len(np.unique(y_balanced))
    )

    print(
        "\nBalanced X saved:",
        x_output
    )

    print(
        "Balanced y saved:",
        y_output
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Original test set was NOT changed."
    )

    print(
        "No SMOTE interpolation was used."
    )

    print(
        "Rare classes were randomly oversampled."
    )

    print(
        "Large classes were randomly undersampled."
    )


if __name__ == "__main__":
    main()