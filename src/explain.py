"""
SHAP-based explainability for the Federated Multiclass IDS.

Purpose:
    Explain WHY the global CNN-LSTM predicted a particular
    UNSW-NB15 attack category.

Example:
    Prediction : DoS
    Important features:
        sttl
        ct_state_ttl
        rate
        sbytes
        dbytes
"""

import argparse
import os
import sys

import numpy as np
import shap
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess
from attack_info import get_attack_info


def get_shap_explanation(
    model,
    X_train,
    X_test,
    feature_names,
    class_names,
    sample_index=0,
    background_size=50,
    top_k=5,
):
    """Explain one prediction using SHAP."""

    if sample_index < 0 or sample_index >= len(X_test):
        raise IndexError("sample_index is outside the test dataset.")

    # Small background set keeps SHAP computation manageable.
    background_size = min(background_size, len(X_train))

    rng = np.random.default_rng(42)

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False,
    )

    background = X_train[background_indices]

    sample = X_test[sample_index:sample_index + 1]

    # Model prediction
    probabilities = model.predict(sample, verbose=0)

    predicted_class = int(np.argmax(probabilities[0]))

    predicted_name = class_names[predicted_class]

    confidence = float(probabilities[0][predicted_class])

    # GradientExplainer works with differentiable TensorFlow models.
    explainer = shap.GradientExplainer(
        model,
        background,
    )

    shap_values = explainer.shap_values(sample)

    # SHAP output format can differ slightly between versions.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[predicted_class])[0]
    else:
        values = np.asarray(shap_values)

        if values.ndim == 4:
            values = values[0, :, :, predicted_class]
        elif values.ndim == 3:
            values = values[0]

    values = np.squeeze(values)

    # CNN-LSTM input shape is (features, 1).
    values = values.reshape(-1)

    if len(values) != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match feature_names. "
            f"SHAP={len(values)}, features={len(feature_names)}"
        )

    importance = np.abs(values)

    top_indices = np.argsort(importance)[::-1][:top_k]

    important_features = []

    for idx in top_indices:
        important_features.append(
            {
                "feature": feature_names[idx],
                "shap_value": float(values[idx]),
                "importance": float(importance[idx]),
            }
        )

    return {
        "predicted_class": predicted_name,
        "confidence": confidence,
        "important_features": important_features,
    }


def main():

    parser = argparse.ArgumentParser(
        description="Explain Federated IDS prediction using SHAP"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Index of test record to explain",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of important features to display",
    )

    parser.add_argument(
        "--model-path",
        default="models/federated_cnn_lstm_multiclass.keras",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--split",
        choices=["official", "stratified"],
        default="stratified",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Federated model not found: {args.model_path}\n"
            "Train federated.py first after adding the dataset."
        )

    print("=" * 60)
    print("FEDERATED IDS - SHAP EXPLANATION")
    print("=" * 60)

    X_train, y_train, X_test, y_test, class_names, feature_names = preprocess(
        mode="multiclass",
        k=args.k,
        split=args.split,
        verbose=False,
    )

    model = load_model(args.model_path)

    result = get_shap_explanation(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        class_names=class_names,
        sample_index=args.sample,
        top_k=args.top_k,
    )

    attack_name = result["predicted_class"]

    attack = get_attack_info(attack_name)

    print(f"\nPredicted Attack : {attack_name}")
    print(f"Confidence       : {result['confidence'] * 100:.2f}%")
    print(f"Attack Type      : {attack['type']}")
    print(f"Description      : {attack['description']}")

    print("\nTop features influencing this prediction:")

    for rank, item in enumerate(
        result["important_features"],
        start=1,
    ):
        direction = (
            "supports prediction"
            if item["shap_value"] > 0
            else "pushes against prediction"
        )

        print(
            f"{rank}. {item['feature']:<20} "
            f"SHAP={item['shap_value']:+.6f} "
            f"({direction})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()