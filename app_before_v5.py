
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Secure IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Secure IoT Intrusion Detection System")
st.caption(
    "Three-Dataset Federated Learning | CNN-BiGRU-MHA | SHAP Explainability"
)
st.divider()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "federated_20percent_cnn_bigru_mha_v6.keras"
)

DATA_DIR = (
    BASE_DIR
    / "data"
    / "fl_3datasets_20percent"
)

SCALER_FILE = (
    DATA_DIR
    / "v6_effective_train_scaler.npz"
)

# Keep the preview responsive on a normal laptop.
MAX_PREVIEW_ROWS = 100_000


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "UNSW-NB15": "UNSW",
    "CICIoT2023": "CICIoT2023",
    "Edge-IIoTset": "EdgeIIoTset",
}

FEATURES = [
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

CLASS_NAMES = [
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
# REPORTED V6 TEST RESULTS
# ============================================================
# These are the previously reported V6 evaluation results.
# They are not recalculated from the dashboard preview.

V6_RESULTS = {
    "UNSW-NB15": {
        "Accuracy": 95.77,
        "Weighted F1": 96.36,
        "Macro Recall": 44.30,
        "Macro F1": 37.84,
    },
    "CICIoT2023": {
        "Accuracy": 67.89,
        "Weighted F1": 71.73,
        "Macro Recall": 36.78,
        "Macro F1": 28.23,
    },
    "Edge-IIoTset": {
        "Accuracy": 67.71,
        "Weighted F1": 69.96,
        "Macro Recall": 33.21,
        "Macro F1": 31.46,
    },
}


# ============================================================
# LOAD MODEL AND TRAIN-ONLY SCALER
# ============================================================

@st.cache_resource
def load_resources():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"V6 model not found:\n{MODEL_FILE}"
        )

    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"V6 scaler not found:\n{SCALER_FILE}"
        )

    model = load_model(
        str(MODEL_FILE),
        compile=False,
    )

    with np.load(
        SCALER_FILE,
        allow_pickle=True,
    ) as scaler_data:

        if "mean" not in scaler_data or "std" not in scaler_data:
            raise ValueError(
                "The scaler NPZ must contain 'mean' and 'std'. "
                f"Available keys: {list(scaler_data.files)}"
            )

        scaler_mean = np.asarray(
            scaler_data["mean"],
            dtype=np.float32,
        )

        scaler_std = np.asarray(
            scaler_data["std"],
            dtype=np.float32,
        )

    if len(scaler_mean) != len(FEATURES):
        raise ValueError(
            f"Expected {len(FEATURES)} scaler means, "
            f"found {len(scaler_mean)}."
        )

    if len(scaler_std) != len(FEATURES):
        raise ValueError(
            f"Expected {len(FEATURES)} scaler standard deviations, "
            f"found {len(scaler_std)}."
        )

    # Prevent division by zero.
    scaler_std = np.where(
        np.abs(scaler_std) < 1e-8,
        1.0,
        scaler_std,
    )

    return model, scaler_mean, scaler_std


try:
    model, scaler_mean, scaler_std = load_resources()

except Exception as error:
    st.error(f"Model/scaler loading failed: {error}")
    st.info(
        "Check that the V6 model and scaler files exist "
        "at the paths configured in app.py."
    )
    st.stop()


# ============================================================
# THREE-DATASET PERFORMANCE OVERVIEW
# ============================================================

st.header("📊 Federated Model — Three-Dataset Results")

metric_columns = st.columns(3)

for column, (dataset_name, results) in zip(
    metric_columns,
    V6_RESULTS.items(),
):

    with column:
        st.subheader(dataset_name)

        st.metric(
            "Accuracy",
            f"{results['Accuracy']:.2f}%",
        )

        st.metric(
            "Weighted F1",
            f"{results['Weighted F1']:.2f}%",
        )

        st.caption(
            f"Macro Recall: {results['Macro Recall']:.2f}%"
        )

        st.caption(
            f"Macro F1: {results['Macro F1']:.2f}%"
        )


results_df = pd.DataFrame.from_dict(
    V6_RESULTS,
    orient="index",
)

results_df.index.name = "Dataset"

st.subheader("Dataset Performance Comparison")

st.dataframe(
    results_df.style.format("{:.2f}%"),
    use_container_width=True,
)

st.caption(
    "Values above are the previously reported V6 test metrics. "
    "The record preview below is a separate interactive demonstration."
)

st.divider()


# ============================================================
# SIDEBAR — DATASET SELECTION
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASETS.keys()),
)

dataset_folder = DATASETS[selected_dataset]

TEST_FILE = (
    DATA_DIR
    / dataset_folder
    / "test_raw.csv"
)

st.sidebar.write("### Federated Model")
st.sidebar.write("CNN-BiGRU-MHA — V6")

st.sidebar.write("### Federated Clients")
st.sidebar.write("3")

st.sidebar.write("### Output Classes")
st.sidebar.write(str(len(CLASS_NAMES)))

st.sidebar.info(
    f"Selected dataset: {selected_dataset}"
)


# ============================================================
# LOAD DATASET TEST PREVIEW
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset_preview(csv_path):

    return pd.read_csv(
        csv_path,
        nrows=MAX_PREVIEW_ROWS,
        low_memory=False,
    )


if not TEST_FILE.exists():

    st.error(
        f"Test file not found for {selected_dataset}:\n"
        f"{TEST_FILE}"
    )

    st.info(
        "Expected location: "
        f"data/fl_3datasets_20percent/{dataset_folder}/test_raw.csv"
    )

    st.stop()


try:

    with st.spinner(
        f"Loading {selected_dataset} test records..."
    ):

        df = load_dataset_preview(
            str(TEST_FILE)
        )

except Exception as error:

    st.error(
        f"Dataset loading failed: {error}"
    )

    st.stop()


required_columns = FEATURES + ["label"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"{selected_dataset} is missing required columns: "
        f"{missing_columns}"
    )

    st.stop()


if df.empty:
    st.error("The selected dataset contains no records.")
    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header(f"📡 Network Traffic — {selected_dataset}")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Records Loaded for Preview",
    f"{len(df):,}",
)

info2.metric(
    "Input Features",
    len(FEATURES),
)

info3.metric(
    "Attack Categories",
    len(CLASS_NAMES),
)

st.info(
    f"For responsiveness, this dashboard loads up to "
    f"{MAX_PREVIEW_ROWS:,} rows from the selected test CSV. "
    "The V6 evaluation metrics above refer to the reported "
    "full test evaluation, not just this preview."
)


# ============================================================
# RECORD SELECTION
# ============================================================

st.sidebar.divider()

record_number = st.sidebar.number_input(
    "Select Record Number",
    min_value=1,
    max_value=len(df),
    value=min(49, len(df)),
    step=1,
)

row_index = int(record_number) - 1
record = df.iloc[row_index]

actual_category = str(record["label"])


# ============================================================
# DISPLAY SELECTED RECORD
# ============================================================

st.subheader(
    f"Selected Record: {record_number}"
)

record_df = pd.DataFrame({
    "Feature": FEATURES,
    "Value": pd.to_numeric(
        record[FEATURES],
        errors="coerce",
    ).fillna(0).to_numpy(),
})

st.dataframe(
    record_df,
    use_container_width=True,
    hide_index=True,
)

st.write(
    f"**Actual Category:** {actual_category}"
)


# ============================================================
# PREPROCESS SELECTED RECORD
# ============================================================

raw_features = pd.to_numeric(
    record[FEATURES],
    errors="coerce",
).fillna(0).to_numpy(dtype=np.float32)

scaled_features = (
    raw_features - scaler_mean
) / scaler_std

scaled_features = np.nan_to_num(
    scaled_features,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

model_input = scaled_features.reshape(
    1,
    len(FEATURES),
    1,
)


# ============================================================
# DETECTION
# ============================================================

st.divider()

if st.button(
    "🔍 Detect Attack",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner("Analyzing network traffic..."):

            prediction = model.predict(
                model_input,
                verbose=0,
            )[0]

        if len(prediction) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(prediction)} outputs, "
                f"but {len(CLASS_NAMES)} classes are configured."
            )

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_category = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            prediction[predicted_index] * 100
        )

        st.session_state["detection_result"] = {
            "record_key": (
                selected_dataset,
                int(record_number),
            ),
            "prediction": prediction,
            "predicted_index": predicted_index,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "actual_category": actual_category,
        }

    except Exception as error:

        st.error(f"Prediction failed: {error}")


# ============================================================
# DISPLAY PERSISTENT DETECTION RESULT
# ============================================================

result = st.session_state.get(
    "detection_result"
)

current_record_key = (
    selected_dataset,
    int(record_number),
)

if (
    result is not None
    and result["record_key"] == current_record_key
):

    prediction = result["prediction"]
    predicted_index = result["predicted_index"]
    predicted_category = result["predicted_category"]
    confidence = result["confidence"]

    st.header("🚨 Detection Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Actual Category",
        result["actual_category"],
    )

    col2.metric(
        "Predicted Category",
        predicted_category,
    )

    col3.metric(
        "Confidence",
        f"{confidence:.2f}%",
    )

    if predicted_category == result["actual_category"]:
        st.success("Correct prediction for this record.")
    else:
        st.error(
            "Misclassified record: "
            f"actual **{result['actual_category']}**, "
            f"predicted **{predicted_category}**."
        )

    if predicted_category.lower() == "normal":
        st.success("🟢 NETWORK STATUS: NORMAL TRAFFIC")
    else:
        st.warning(
            f"🔴 NETWORK STATUS: "
            f"{predicted_category.upper()} ATTACK DETECTED"
        )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.divider()
    st.subheader("📊 Attack Class Probabilities")

    probability_df = pd.DataFrame({
        "Attack Category": CLASS_NAMES,
        "Probability (%)": prediction * 100,
    }).sort_values(
        "Probability (%)",
        ascending=False,
    ).reset_index(drop=True)

    probability_display = probability_df.copy()

    probability_display["Probability (%)"] = (
        probability_display["Probability (%)"]
        .map(lambda value: f"{value:.2f}%")
    )

    st.dataframe(
        probability_display,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        probability_df.set_index("Attack Category")
    )


    # ========================================================
    # OPTIONAL SHAP EXPLAINABILITY
    # ========================================================

    st.divider()
    st.subheader("🔍 SHAP Explainability")

    st.write(
        "Generate a feature-attribution explanation for "
        "the current prediction."
    )

    if st.button("Generate SHAP Explanation"):

        try:

            import shap

            background_df = df.drop(
                index=row_index
            )

            if background_df.empty:
                raise ValueError(
                    "Not enough records for a SHAP background."
                )

            background_df = background_df.sample(
                n=min(32, len(background_df)),
                random_state=42,
            )

            background_features = (
                background_df[FEATURES]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )

            background_scaled = (
                background_features - scaler_mean
            ) / scaler_std

            background_scaled = np.nan_to_num(
                background_scaled,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            background_input = background_scaled.reshape(
                len(background_scaled),
                len(FEATURES),
                1,
            )

            with st.spinner(
                "Generating SHAP explanation..."
            ):

                explainer = shap.GradientExplainer(
                    model,
                    background_input,
                )

                shap_values = explainer.shap_values(
                    model_input
                )

            if isinstance(shap_values, list):

                shap_vector = np.asarray(
                    shap_values[predicted_index]
                ).reshape(-1, len(FEATURES))[0]

            else:

                shap_array = np.asarray(shap_values)

                if shap_array.ndim == 4:
                    shap_vector = shap_array[
                        0, :, 0, predicted_index
                    ]

                elif shap_array.ndim == 3:
                    if shap_array.shape[-1] == len(CLASS_NAMES):
                        shap_vector = shap_array[
                            0, :, predicted_index
                        ]
                    else:
                        shap_vector = shap_array[
                            0, :, 0
                        ]

                elif shap_array.ndim == 2:
                    shap_vector = shap_array[0]

                else:
                    raise ValueError(
                        f"Unexpected SHAP shape: {shap_array.shape}"
                    )

            shap_vector = np.asarray(
                shap_vector
            ).reshape(-1)

            if len(shap_vector) != len(FEATURES):
                raise ValueError(
                    "SHAP output does not match the feature count."
                )

            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": shap_vector,
            })

            shap_df["Absolute SHAP"] = np.abs(
                shap_df["SHAP Value"]
            )

            shap_df = shap_df.sort_values(
                "Absolute SHAP",
                ascending=False,
            ).head(10)

            st.dataframe(
                shap_df[
                    ["Feature", "SHAP Value"]
                ].style.format({
                    "SHAP Value": "{:+.6f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                shap_df.set_index("Feature")[["SHAP Value"]]
            )

            st.caption(
                "Positive SHAP values push the selected class output "
                "up relative to the explainer's background; negative "
                "values push it down. These are feature attributions, "
                "not proof of causation."
            )

        except Exception as error:
            st.error(
                f"SHAP explanation failed: {error}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()
st.subheader("🤖 Model Information")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Architecture", "CNN-BiGRU-MHA")
m2.metric("Federated Clients", "3")
m3.metric("Input Features", len(FEATURES))
m4.metric("Output Classes", len(CLASS_NAMES))

st.caption(
    "Secure IoT Intrusion Detection System | "
    "UNSW-NB15 + CICIoT2023 + Edge-IIoTset | "
    "Federated Learning + Multi-Class Detection + SHAP"
)
