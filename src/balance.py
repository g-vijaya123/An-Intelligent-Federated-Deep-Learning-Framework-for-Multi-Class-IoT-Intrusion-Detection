import numpy as np
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def apply_smote_tomek(X_train, y_train, random_state=42):
    print("\n" + "=" * 60)
    print("CONTROLLED SMOTE-TOMEK CLASS BALANCING")
    print("=" * 60)

    n_samples, n_features, channels = X_train.shape

    # CNN-LSTM 3D -> SMOTE 2D
    X_2d = X_train.reshape(n_samples, n_features)

    # One-hot -> integer labels
    y_labels = np.argmax(y_train, axis=1)

    before = Counter(y_labels)

    print("\nClass distribution BEFORE balancing:")
    for cls, count in sorted(before.items()):
        print(f"Class {cls}: {count}")

    # Only heavily underrepresented classes are increased.
    # Existing classes above these values remain unchanged.
    target_counts = {
        0: 8000,    # Analysis
        1: 8000,    # Backdoor
        2: 15000,   # DoS
        7: 15000,   # Reconnaissance
        8: 8000,    # Shellcode
        9: 5000     # Worms
    }

    sampling_strategy = {
        cls: target
        for cls, target in target_counts.items()
        if before[cls] < target
    }

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5
    )

    sampler = SMOTETomek(
        smote=smote,
        random_state=random_state
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_2d,
        y_labels
    )

    after = Counter(y_resampled)

    print("\nClass distribution AFTER balancing:")
    for cls, count in sorted(after.items()):
        print(f"Class {cls}: {count}")

    # Back to CNN-LSTM input
    X_resampled = X_resampled.reshape(
        -1,
        n_features,
        channels
    )

    # Integer -> one-hot
    n_classes = y_train.shape[1]

    y_resampled_onehot = np.eye(
        n_classes,
        dtype=np.float32
    )[y_resampled]

    print("\nOriginal training samples :", len(X_train))
    print("Balanced training samples :", len(X_resampled))
    print("Balanced X shape          :", X_resampled.shape)
    print("Balanced y shape          :", y_resampled_onehot.shape)
    print("=" * 60)

    return X_resampled, y_resampled_onehot