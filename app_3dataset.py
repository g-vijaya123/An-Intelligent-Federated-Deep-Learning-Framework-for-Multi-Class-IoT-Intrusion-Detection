from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import shap

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Federated Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "fl_3datasets_20percent"

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "federated_20percent_cnn_bigru_mha_corrected.keras"
)

SCALER_FILE = DATA_DIR / "global_train_scaler.npz"

CLIENTS = [
    "UNSW",
    "CICIoT2023",
    "EdgeIIoTset",
]

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

CLASSES = [
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

STANDARDIZED_CLIP = 20.0

FEATURE_HELP = {
    "duration": "Duration of the network flow.",
    "packet_rate": "Approximate number of packets transmitted per unit time.",
    "src_packets": "Number of packets sent by the source.",
    "dst_packets": "Number of packets sent by the destination.",
    "src_bytes": "Number of bytes transmitted by the source.",
    "dst_bytes": "Number of bytes transmitted by the destination.",
    "packet_size_mean": "Average packet size.",
    "packet_size_min": "Minimum packet size.",
    "packet_size_max": "Maximum packet size.",
    "packet_size_std": "Standard deviation of packet sizes.",
    "ttl": "Time To Live related network feature.",
    "syn_count": "SYN flag/count related feature.",
    "ack_count": "ACK flag/count related feature.",
    "fin_count": "FIN flag/count related feature.",
    "rst_count": "RST flag/count related feature.",
    "iat": "Inter-arrival time related feature.",
}

# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .normal-box {
        padding: 22px;
        border-radius: 14px;
        border: 2px solid #2e7d32;
        background: rgba(46,125,50,0.08);
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .attack-box {
        padding: 22px;
        border-radius: 14px;
        border: 2px solid #c62828;
        background: rgba(198,40,40,0.08);
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .info-box {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.35);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .small-note {
        color: #888;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL / SCALER
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    return tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )


@st.cache_resource
def load_scaler():
    if not SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_FILE}"
        )

    scaler = np.load(
        SCALER_FILE,
        allow_pickle=True,
    )

    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    ).astype(np.float32)

    return mean, std


try:
    model = load_model()
    scaler_mean, scaler_std = load_scaler()

except Exception as error:
    st.error("Unable to load the final IDS model.")
    st.exception(error)
    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_dataframe(df):

    x = df[FEATURES].copy()

    for feature in FEATURES:
        x[feature] = pd.to_numeric(
            x[feature],
            errors="coerce",
        )

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if x.isna().any().any():
        bad_columns = (
            x.columns[
                x.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Invalid values found in: "
            + ", ".join(bad_columns)
        )

    x = x.to_numpy(
        dtype=np.float32
    )

    x = (
        x - scaler_mean
    ) / scaler_std

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=STANDARDIZED_CLIP,
        neginf=-STANDARDIZED_CLIP,
    )

    x = np.clip(
        x,
        -STANDARDIZED_CLIP,
        STANDARDIZED_CLIP,
    )

    return x.reshape(
        len(x),
        len(FEATURES),
        1,
    ).astype(np.float32)


# ============================================================
# PREDICTION
# ============================================================

def predict_dataframe(df):

    x = preprocess_dataframe(df)

    probabilities = model.predict(
        x,
        verbose=0,
    )

    pred_ids = np.argmax(
        probabilities,
        axis=1,
    )

    return (
        x,
        probabilities,
        pred_ids,
    )


# ============================================================
# SHAP BACKGROUND
# ============================================================

@st.cache_resource
def create_shap_explainer():

    background_parts = []

    for client in CLIENTS:

        train_file = (
            DATA_DIR
            / client
            / "train_raw.csv"
        )

        if not train_file.exists():
            continue

        df = pd.read_csv(
            train_file,
            nrows=3000,
            usecols=FEATURES,
        )

        sample_size = min(
            12,
            len(df),
        )

        sampled = df.sample(
            n=sample_size,
            random_state=42,
        )

        background_parts.append(
            preprocess_dataframe(sampled)
        )

    if not background_parts:
        raise RuntimeError(
            "Unable to build SHAP background."
        )

    background = np.concatenate(
        background_parts,
        axis=0,
    )

    return shap.GradientExplainer(
        model,
        background,
    )


# ============================================================
# SHAP CALCULATION
# ============================================================

def calculate_shap(x_record, predicted_id):

    explainer = create_shap_explainer()

    shap_values = explainer.shap_values(
        x_record
    )

    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[predicted_id]
        )

        values = np.squeeze(
            values[0]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if (
            shap_array.ndim == 4
            and shap_array.shape[-1]
            == len(CLASSES)
        ):

            values = shap_array[
                0,
                :,
                0,
                predicted_id,
            ]

        elif (
            shap_array.ndim == 3
            and shap_array.shape[-1]
            == len(CLASSES)
        ):

            values = shap_array[
                0,
                :,
                predicted_id,
            ]

        elif shap_array.ndim == 3:

            values = np.squeeze(
                shap_array[0]
            )

        elif shap_array.ndim == 2:

            values = shap_array[0]

        else:
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )

    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if len(values) != len(FEATURES):
        raise ValueError(
            "SHAP feature count mismatch."
        )

    return values


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(
    raw_df,
    x_record,
    probabilities,
    predicted_id,
    actual_label=None,
):

    predicted_class = CLASSES[
        predicted_id
    ]

    confidence = float(
        probabilities[predicted_id]
    )

    if predicted_class == "Normal":

        st.markdown(
            f"""
            <div class="normal-box">
                <h2>✅ NORMAL TRAFFIC</h2>
                <h3>Predicted Class: Normal</h3>
                <p>
                    Model Confidence:
                    <b>{confidence * 100:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="attack-box">
                <h2>🚨 ATTACK DETECTED</h2>
                <h3>
                    Predicted Attack:
                    {predicted_class}
                </h3>
                <p>
                    Model Confidence:
                    <b>{confidence * 100:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if actual_label is not None:

        st.subheader("Ground Truth Verification")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Actual Class",
            actual_label,
        )

        col2.metric(
            "Predicted Class",
            predicted_class,
        )

        if actual_label == predicted_class:

            col3.metric(
                "Prediction",
                "CORRECT",
            )

            st.success(
                "The prediction matches the "
                "known dataset label."
            )

        else:

            col3.metric(
                "Prediction",
                "INCORRECT",
            )

            st.error(
                "The prediction does not match "
                "the known dataset label."
            )

    else:

        st.info(
            "Ground Truth: Not Available. "
            "This is a model prediction for "
            "user-entered traffic and cannot "
            "be independently verified by the app."
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    st.subheader("Class Probabilities")

    probability_df = pd.DataFrame({
        "Attack Class": CLASSES,
        "Probability (%)":
            probabilities * 100,
    })

    probability_df = (
        probability_df
        .sort_values(
            "Probability (%)",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        probability_df.style.format({
            "Probability (%)": "{:.3f}"
        }),
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        probability_df.set_index(
            "Attack Class"
        )["Probability (%)"]
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.subheader(
        f"Why the Model Predicted "
        f"{predicted_class}"
    )

    with st.spinner(
        "Calculating SHAP explanation..."
    ):

        shap_values = calculate_shap(
            x_record,
            predicted_id,
        )

    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "Input Value": [
            float(raw_df.iloc[0][feature])
            for feature in FEATURES
        ],
        "SHAP Value": shap_values,
    })

    shap_df[
        "Absolute Influence"
    ] = np.abs(
        shap_df["SHAP Value"]
    )

    shap_df[
        "Effect"
    ] = np.where(
        shap_df["SHAP Value"] >= 0,
        f"Supports {predicted_class}",
        f"Opposes {predicted_class}",
    )

    shap_df = (
        shap_df
        .sort_values(
            "Absolute Influence",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    top_shap = shap_df.head(10)

    st.dataframe(
        top_shap.style.format({
            "Input Value": "{:.5f}",
            "SHAP Value": "{:+.6f}",
            "Absolute Influence": "{:.6f}",
        }),
        width="stretch",
        hide_index=True,
    )

    chart_data = (
        top_shap[
            ["Feature", "SHAP Value"]
        ]
        .set_index("Feature")
    )

    st.bar_chart(
        chart_data
    )

    st.caption(
        "SHAP indicates which features influenced "
        "the model prediction. It does not prove "
        "that those features caused the attack."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🛡️ Federated Intrusion Detection System
    </div>

    <div class="subtitle">
        Multi-Dataset CNN-BiGRU-Multi-Head Attention
        Intrusion Detection with SHAP Explainability
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "IDS Navigation"
)

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Dataset Test Record",
        "Own Network Traffic",
        "Model Information",
    ],
)

st.sidebar.divider()

st.sidebar.write(
    "**Datasets used**"
)

st.sidebar.write(
    "• UNSW-NB15"
)

st.sidebar.write(
    "• CICIoT2023"
)

st.sidebar.write(
    "• Edge-IIoTset"
)

st.sidebar.write(
    "\n**Final Model**"
)

st.sidebar.write(
    "CNN + BiGRU + Multi-Head Attention"
)

# ============================================================
# MODE 1 - DATASET TEST RECORD
# ============================================================

if mode == "Dataset Test Record":

    st.header(
        "Dataset Test Record Prediction"
    )

    st.write(
        "Select a held-out test record. "
        "Because this record has a known label, "
        "the prediction can be compared with "
        "the dataset ground truth."
    )

    selected_client = st.selectbox(
        "Select Dataset",
        CLIENTS,
    )

    test_file = (
        DATA_DIR
        / selected_client
        / "test_raw.csv"
    )

    max_preview = 5000

    @st.cache_data
    def load_test_preview(
        file_path,
        nrows,
    ):
        return pd.read_csv(
            file_path,
            nrows=nrows,
        )

    test_df = load_test_preview(
        str(test_file),
        max_preview,
    )

    record_number = st.number_input(
        "Record Number",
        min_value=1,
        max_value=len(test_df),
        value=1,
        step=1,
    )

    index = int(
        record_number - 1
    )

    selected_row = (
        test_df
        .iloc[[index]]
        .copy()
    )

    actual_label = str(
        selected_row.iloc[0]["label"]
    )

    st.subheader(
        "Selected Network Record"
    )

    st.dataframe(
        selected_row[
            FEATURES + ["label"]
        ],
        width="stretch",
        hide_index=True,
    )

    if st.button(
        "Analyze Selected Record",
        type="primary",
    ):

        try:

            x, probabilities, pred_ids = (
                predict_dataframe(
                    selected_row
                )
            )

            display_prediction(
                selected_row,
                x[0:1],
                probabilities[0],
                int(pred_ids[0]),
                actual_label=actual_label,
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


# ============================================================
# MODE 2 - OWN NETWORK TRAFFIC
# ============================================================

elif mode == "Own Network Traffic":

    st.header(
        "Analyze New Network Traffic"
    )

    st.write(
        "Enter the 16 harmonized network features "
        "used by the final global model."
    )

    st.warning(
        "For manually entered traffic, the true "
        "attack label is unknown. The app can show "
        "the model prediction and confidence, but "
        "cannot verify whether that prediction is "
        "actually correct."
    )

    default_values = {
        "duration": 0.5,
        "packet_rate": 50.0,
        "src_packets": 10.0,
        "dst_packets": 10.0,
        "src_bytes": 1000.0,
        "dst_bytes": 1000.0,
        "packet_size_mean": 100.0,
        "packet_size_min": 50.0,
        "packet_size_max": 150.0,
        "packet_size_std": 20.0,
        "ttl": 64.0,
        "syn_count": 0.0,
        "ack_count": 0.0,
        "fin_count": 0.0,
        "rst_count": 0.0,
        "iat": 0.01,
    }

    values = {}

    with st.form(
        "network_input_form"
    ):

        col1, col2 = st.columns(2)

        for i, feature in enumerate(
            FEATURES
        ):

            target_col = (
                col1
                if i % 2 == 0
                else col2
            )

            with target_col:

                values[feature] = (
                    st.number_input(
                        feature,
                        value=float(
                            default_values[
                                feature
                            ]
                        ),
                        format="%.6f",
                        help=FEATURE_HELP[
                            feature
                        ],
                    )
                )

        submitted = st.form_submit_button(
            "Analyze Network Traffic",
            type="primary",
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [
                    {
                        feature:
                            values[feature]
                        for feature
                        in FEATURES
                    }
                ]
            )

            x, probabilities, pred_ids = (
                predict_dataframe(
                    input_df
                )
            )

            display_prediction(
                input_df,
                x[0:1],
                probabilities[0],
                int(pred_ids[0]),
                actual_label=None,
            )

        except Exception as error:

            st.error(
                "Unable to analyze "
                "the entered traffic."
            )

            st.exception(error)


# ============================================================
# MODE 3 - MODEL INFORMATION
# ============================================================

elif mode == "Model Information":

    st.header(
        "Final Model Information"
    )

    st.subheader(
        "Project Workflow"
    )

    st.code(
        """
UNSW-NB15
CICIoT2023
Edge-IIoTset
      ↓
Approximately 20% selected from each dataset
      ↓
Common 16-feature representation
      ↓
Common 13-class attack taxonomy
      ↓
80% Training / 20% Held-out Testing
      ↓
Shared Train-only Standardization
      ↓
Three Heterogeneous Federated Clients
      ↓
CNN + BiGRU + Multi-Head Attention
      ↓
Equal-client Federated Averaging
      ↓
One Global Intrusion Detection Model
      ↓
Multiclass Attack Prediction
      ↓
SHAP Explainability
        """
    )

    st.subheader(
        "Final Held-Out Results"
    )

    results = pd.DataFrame({
        "Evaluation": [
            "UNSW-NB15",
            "CICIoT2023",
            "Edge-IIoTset",
            "Pooled Test Set",
        ],
        "Accuracy (%)": [
            95.21,
            73.62,
            73.58,
            74.71,
        ],
        "Weighted F1 (%)": [
            94.84,
            70.74,
            63.10,
            70.15,
        ],
        "Macro F1 (%)": [
            25.75,
            18.21,
            12.93,
            30.85,
        ],
    })

    st.dataframe(
        results,
        width="stretch",
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Pooled Test Records",
        "1,993,287",
    )

    col2.metric(
        "Pooled Accuracy",
        "74.71%",
    )

    col3.metric(
        "Pooled Macro-F1",
        "30.85%",
    )

    st.subheader(
        "Input Features"
    )

    feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            FEATURE_HELP[f]
            for f in FEATURES
        ],
    })

    st.dataframe(
        feature_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Output Classes"
    )

    st.write(
        ", ".join(CLASSES)
    )

    st.subheader(
        "Important Interpretation"
    )

    st.info(
        "The pooled held-out accuracy is 74.71%, "
        "but macro-F1 is 30.85%. Therefore the "
        "model performs unevenly across the attack "
        "classes, particularly minority classes. "
        "Accuracy alone should not be used to claim "
        "uniform performance across all attacks."
    )

    st.warning(
        "The three source datasets have different "
        "network schemas. They were mapped into a "
        "shared 16-feature representation and a "
        "common attack taxonomy. Some semantic "
        "features are unavailable in particular "
        "datasets and are represented with default "
        "values during harmonization. This creates "
        "cross-domain limitations."
    )

    st.caption(
        "This project uses simulated federated "
        "learning with the three datasets acting "
        "as heterogeneous clients on one machine. "
        "Equal-client parameter averaging is used; "
        "it should not be described as canonical "
        "sample-weighted FedAvg."
    )