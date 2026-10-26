"""
Final prediction pipeline for Federated Multiclass IDS.

Output:
- Predicted attack category
- Confidence
- Attack description
- SHAP important features
"""

import argparse
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess
from attack_info import get_attack_info
from explain import get_shap_explanation


def main():
    parser = argparse.ArgumentParser(
        description="Predict and explain an IDS test sample"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Test sample index"
    )

    parser.add_argument(
        "--model-path",
        default="models/federated_cnn_lstm_multiclass.keras"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=32
    )

    parser.add_argument(
        "--split",
        choices=["official", "stratified"],
        default="stratified"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    # Check model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Train federated.py first."
        )

    # Load and preprocess dataset
    X_train, y_train, X_test, y_test, class_names, feature_names = preprocess(
        mode="multiclass",
        k=args.k,
        split=args.split,
        verbose=False
    )

    if args.sample < 0 or args.sample >= len(X_test):
        raise IndexError("Invalid test sample index.")

    # Load federated global model
    model = load_model(args.model_path)

    sample = X_test[args.sample:args.sample + 1]

    # Prediction
    probabilities = model.predict(sample, verbose=0)

    predicted_index = int(np.argmax(probabilities[0]))
    predicted_attack = class_names[predicted_index]
    confidence = float(probabilities[0][predicted_index])

    true_index = int(np.argmax(y_test[args.sample]))
    true_attack = class_names[true_index]

    # General attack information
    info = get_attack_info(predicted_attack)

    print("\n" + "=" * 65)
    print("FEDERATED MULTICLASS INTRUSION DETECTION")
    print("=" * 65)

    print(f"Sample Number     : {args.sample}")
    print(f"Actual Category   : {true_attack}")
    print(f"Predicted Category: {predicted_attack}")
    print(f"Confidence        : {confidence * 100:.2f}%")

    print("\nATTACK INFORMATION")
    print("-" * 65)

    print(f"Attack Type       : {info['type']}")
    print(f"Description       : {info['description']}")

    # SHAP explanation
    print("\nGenerating SHAP explanation...")

    explanation = get_shap_explanation(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        class_names=class_names,
        sample_index=args.sample,
        top_k=args.top_k,
    )

    print("\nTOP FEATURES INFLUENCING THE PREDICTION")
    print("-" * 65)

    for rank, item in enumerate(
        explanation["important_features"],
        start=1
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

    print("=" * 65)


if __name__ == "__main__":
    main()