import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import shap


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Federated Explainable Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = os.path.join(
    "models",
    "federated_cnn_bigru_mha_full.keras"
)

ARTIFACT_FILE = os.path.join(
    "models",
    "preprocessing_full_features.joblib"
)

DATA_DIR = os.path.join(
    "data",
    "processed_full"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test.npy"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test.npy"
)


# ============================================================
# 43 FINAL FEATURES
# ============================================================

FEATURE_NAMES = [
    "sport",
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
    "ct_dst_src_ltm"
]


CATEGORICAL_FEATURES = [
    "proto",
    "state",
    "service"
]


# ============================================================
# ATTACK DESCRIPTIONS
# ============================================================

ATTACK_DESCRIPTIONS = {

    "Normal":
        "This record is classified as normal network traffic. "
        "The model did not identify a known attack pattern.",

    "Analysis":
        "Analysis attacks involve attempts to inspect systems, "
        "services or network behaviour to obtain useful information.",

    "Backdoor":
        "Backdoor attacks attempt to obtain unauthorized access "
        "through hidden or previously established access paths.",

    "DoS":
        "Denial-of-Service attacks attempt to make network services "
        "or resources unavailable by overwhelming or disrupting them.",

    "Exploits":
        "Exploit attacks attempt to take advantage of vulnerabilities "
        "in software, services or operating systems.",

    "Fuzzers":
        "Fuzzer attacks send unusual, malformed or unexpected inputs "
        "to discover vulnerabilities or trigger abnormal behaviour.",

    "Generic":
        "Generic attacks represent broad attack techniques, including "
        "techniques that may target cryptographic or network mechanisms.",

    "Reconnaissance":
        "Reconnaissance attacks gather information about systems, "
        "services, ports and network structure before further attacks.",

    "Shellcode":
        "Shellcode attacks attempt to execute malicious low-level "
        "instructions or payloads on a target system.",

    "Worms":
        "Worm attacks involve self-propagating malicious software "
        "capable of spreading between vulnerable systems."
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .normal-box {
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #21c55d;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .attack-box {
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #ef4444;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #888;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )


# ============================================================
# LOAD PREPROCESSING ARTIFACT
# ============================================================

@st.cache_resource
def load_artifact():

    if not os.path.exists(ARTIFACT_FILE):
        raise FileNotFoundError(
            f"Preprocessing artifact not found: "
            f"{ARTIFACT_FILE}"
        )

    return joblib.load(
        ARTIFACT_FILE
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

@st.cache_resource
def load_test_data():

    if not os.path.exists(X_TEST_FILE):
        raise FileNotFoundError(
            f"X_test not found: {X_TEST_FILE}"
        )

    if not os.path.exists(Y_TEST_FILE):
        raise FileNotFoundError(
            f"y_test not found: {Y_TEST_FILE}"
        )

    X = np.load(
        X_TEST_FILE,
        mmap_mode="r"
    )

    y = np.load(
        Y_TEST_FILE,
        mmap_mode="r"
    )

    return X, y


# ============================================================
# INITIALIZE APPLICATION
# ============================================================

try:

    model = load_model()

    artifact = load_artifact()

    X_test, y_test = load_test_data()

except Exception as e:

    st.error(
        "Application initialization failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = np.asarray(
    artifact["class_names"]
)


# ============================================================
# VALIDATION
# ============================================================

if len(FEATURE_NAMES) != 43:

    st.error(
        "Internal configuration error: "
        "43 features are required."
    )

    st.stop()


if X_test.shape[1] != 43:

    st.error(
        f"X_test contains {X_test.shape[1]} features. "
        "The final model requires 43."
    )

    st.stop()


if len(CLASS_NAMES) != 10:

    st.error(
        f"Expected 10 classes but found "
        f"{len(CLASS_NAMES)}."
    )

    st.stop()


# ============================================================
# ARTIFACT HELPERS
# ============================================================

def get_categorical_encoder():

    possible_keys = [
        "categorical_encoder",
        "encoder",
        "ordinal_encoder"
    ]

    for key in possible_keys:

        if key in artifact:

            return artifact[key]

    return None


def get_scaler():

    possible_keys = [
        "scaler",
        "standard_scaler"
    ]

    for key in possible_keys:

        if key in artifact:

            return artifact[key]

    return None


categorical_encoder = get_categorical_encoder()

scaler = get_scaler()


# ============================================================
# PREDICTION
# ============================================================

def predict_processed_record(
    processed_record
):

    processed_record = np.asarray(
        processed_record,
        dtype=np.float32
    ).reshape(1, 43)

    model_input = np.expand_dims(
        processed_record,
        axis=-1
    )

    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = str(
        CLASS_NAMES[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "processed_record": processed_record
    }


# ============================================================
# PROBABILITY TABLE
# ============================================================

def create_probability_table(
    probabilities
):

    df = pd.DataFrame({
        "Attack Class": CLASS_NAMES,
        "Probability (%)":
            np.asarray(probabilities) * 100
    })

    return df.sort_values(
        "Probability (%)",
        ascending=False
    ).reset_index(
        drop=True
    )


# ============================================================
# SHAP
# ============================================================

def calculate_shap(
    processed_record,
    predicted_index
):

    rng = np.random.default_rng(42)

    background_size = min(
        64,
        len(X_test)
    )

    background_indices = rng.choice(
        len(X_test),
        size=background_size,
        replace=False
    )

    background = np.asarray(
        X_test[background_indices],
        dtype=np.float32
    )

    background = np.expand_dims(
        background,
        axis=-1
    )

    sample = np.asarray(
        processed_record,
        dtype=np.float32
    ).reshape(
        1,
        43,
        1
    )

    explainer = shap.GradientExplainer(
        model,
        background
    )

    shap_values = explainer.shap_values(
        sample
    )


    if isinstance(
        shap_values,
        list
    ):

        values = np.asarray(
            shap_values[
                predicted_index
            ]
        )[0]

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 4:

            values = shap_array[
                0,
                :,
                :,
                predicted_index
            ]

        elif (
            shap_array.ndim == 3
            and
            shap_array.shape[-1]
            == len(CLASS_NAMES)
        ):

            values = shap_array[
                0,
                :,
                predicted_index
            ]

        elif shap_array.ndim == 3:

            values = shap_array[0]

        elif shap_array.ndim == 2:

            values = shap_array[0]

        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )


    values = np.asarray(
        values
    ).reshape(-1)


    if len(values) != 43:

        raise ValueError(
            f"Expected 43 SHAP values "
            f"but received {len(values)}."
        )


    return values


# ============================================================
# SHAP TABLE
# ============================================================

def create_shap_table(
    shap_values
):

    df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "SHAP Contribution":
            shap_values,
        "Importance":
            np.abs(shap_values)
    })

    df = df.sort_values(
        "Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result,
    actual_class=None,
    show_shap=True
):

    predicted_class = result[
        "predicted_class"
    ]

    confidence = result[
        "confidence"
    ]

    probabilities = result[
        "probabilities"
    ]


    st.markdown("---")

    st.subheader(
        "Detection Result"
    )


    if predicted_class == "Normal":

        st.markdown(
            """
            <div class="normal-box">
            ✅ NORMAL TRAFFIC
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="attack-box">
            🚨 ATTACK DETECTED — {predicted_class}
            </div>
            """,
            unsafe_allow_html=True
        )


    col1, col2, col3 = st.columns(
        3
    )


    col1.metric(
        "Predicted Class",
        predicted_class
    )


    col2.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%"
    )


    if actual_class is not None:

        is_correct = (
            predicted_class
            == actual_class
        )

        col3.metric(
            "Verification",
            "CORRECT"
            if is_correct
            else "INCORRECT"
        )


        if is_correct:

            st.success(
                f"Ground Truth: {actual_class} | "
                "Prediction matches the known label."
            )

        else:

            st.error(
                f"Ground Truth: {actual_class} | "
                f"Model predicted: {predicted_class}"
            )

    else:

        col3.metric(
            "Verification",
            "UNKNOWN"
        )

        st.info(
            "This is user-provided traffic. "
            "No external ground-truth label is available, "
            "so the application cannot determine whether "
            "the prediction is correct or incorrect."
        )


    # ========================================================
    # ATTACK DESCRIPTION
    # ========================================================

    st.subheader(
        "Prediction Information"
    )

    st.write(
        ATTACK_DESCRIPTIONS.get(
            predicted_class,
            "Description unavailable."
        )
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "Class Probabilities"
    )

    probability_df = (
        create_probability_table(
            probabilities
        )
    )

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )


    chart_df = (
        probability_df
        .set_index(
            "Attack Class"
        )
    )

    st.bar_chart(
        chart_df[
            "Probability (%)"
        ]
    )


    # ========================================================
    # SHAP
    # ========================================================

    if show_shap:

        st.subheader(
            "Explainable AI — SHAP"
        )

        st.write(
            f"Why did the model predict "
            f"**{predicted_class}**?"
        )


        with st.spinner(
            "Calculating SHAP explanation..."
        ):

            try:

                shap_values = (
                    calculate_shap(
                        result[
                            "processed_record"
                        ],
                        result[
                            "predicted_index"
                        ]
                    )
                )


                shap_df = (
                    create_shap_table(
                        shap_values
                    )
                )


                top10 = shap_df.head(
                    10
                ).copy()


                st.dataframe(
                    top10[
                        [
                            "Feature",
                            "SHAP Contribution",
                            "Importance"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )


                importance_chart = (
                    top10[
                        [
                            "Feature",
                            "Importance"
                        ]
                    ]
                    .set_index(
                        "Feature"
                    )
                )


                st.bar_chart(
                    importance_chart
                )


                top5 = (
                    shap_df
                    .head(5)[
                        "Feature"
                    ]
                    .tolist()
                )


                st.success(
                    "Most influential features: "
                    + ", ".join(top5)
                )


                st.caption(
                    "SHAP explains which input features "
                    "influenced the model's prediction. "
                    "It does not prove that these features "
                    "caused the attack."
                )


            except Exception as e:

                st.warning(
                    "Prediction completed successfully, "
                    "but the SHAP explanation could not "
                    f"be generated: {e}"
                )


# ============================================================
# PREPROCESS USER INPUT
# ============================================================

def preprocess_manual_input(
    raw_values
):

    df = pd.DataFrame(
        [raw_values],
        columns=FEATURE_NAMES
    )


    # --------------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------------

    if categorical_encoder is None:

        raise ValueError(
            "Categorical encoder was not found "
            "inside preprocessing artifact."
        )


    categorical_data = df[
        CATEGORICAL_FEATURES
    ].astype(str)


    encoded_categories = (
        categorical_encoder.transform(
            categorical_data
        )
    )


    encoded_df = df.copy()


    encoded_df[
        CATEGORICAL_FEATURES
    ] = encoded_categories


    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in FEATURE_NAMES:

        encoded_df[column] = (
            pd.to_numeric(
                encoded_df[column],
                errors="coerce"
            )
        )


    encoded_df = encoded_df.fillna(
        0
    )


    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    if scaler is None:

        raise ValueError(
            "StandardScaler was not found "
            "inside preprocessing artifact."
        )


    processed = scaler.transform(
        encoded_df[
            FEATURE_NAMES
        ]
    )


    return np.asarray(
        processed,
        dtype=np.float32
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
    🛡️ Federated Explainable Intrusion Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    CNN + BiGRU + Multi-Head Attention |
    Federated Learning |
    Multi-Class IDS |
    SHAP Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ IDS Control Panel"
)

st.sidebar.markdown(
    """
    **Final Model**

    CNN → BiGRU → Multi-Head Attention

    **Learning**

    Simulated Federated Learning + FedAvg

    **Dataset**

    UNSW-NB15

    **Input Features**

    43

    **Output Classes**

    10
    """
)


mode = st.sidebar.radio(
    "Choose Prediction Mode",
    [
        "Dataset Test Record",
        "Own Network Traffic",
        "Model Information"
    ]
)


st.sidebar.markdown("---")

st.sidebar.caption(
    "Final-year Federated Explainable "
    "Intrusion Detection System"
)


# ============================================================
# MODE 1
# DATASET TEST RECORD
# ============================================================

if mode == "Dataset Test Record":

    st.header(
        "Dataset Test Record Prediction"
    )

    st.write(
        "Select a record from the untouched "
        "UNSW-NB15 test partition. Because the "
        "ground-truth class is available, the "
        "prediction can be verified."
    )


    max_index = (
        len(X_test) - 1
    )


    record_input = st.text_input(
        "Enter Test Record Index",
        value="100",
        help=(
            f"Enter an integer between "
            f"0 and {max_index:,}"
        )
    )


    show_shap = st.checkbox(
        "Generate SHAP explanation",
        value=True,
        key="dataset_shap"
    )


    if st.button(
        "Analyze Dataset Record",
        type="primary"
    ):

        try:

            record_index = int(
                record_input.strip()
            )


            if (
                record_index < 0
                or
                record_index > max_index
            ):

                st.error(
                    f"Enter a record index between "
                    f"0 and {max_index:,}."
                )

                st.stop()


            processed_record = np.asarray(
                X_test[
                    record_index
                ],
                dtype=np.float32
            )


            actual_index = int(
                y_test[
                    record_index
                ]
            )


            actual_class = str(
                CLASS_NAMES[
                    actual_index
                ]
            )


            st.subheader(
                "Selected Record"
            )


            st.write(
                "**Record Index:**",
                record_index
            )


            st.write(
                "**Known Ground Truth:**",
                actual_class
            )


            result = (
                predict_processed_record(
                    processed_record
                )
            )


            display_result(
                result,
                actual_class=actual_class,
                show_shap=show_shap
            )


        except ValueError:

            st.error(
                "Record index must be "
                "a valid integer."
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# MODE 2
# OWN NETWORK TRAFFIC
# ============================================================

elif mode == "Own Network Traffic":

    st.header(
        "Own Network Traffic Prediction"
    )


    st.warning(
        "This mode accepts a manually entered network "
        "traffic record. Because there is no external "
        "ground-truth label, the result is a model "
        "prediction only."
    )


    st.info(
        "Enter raw network-flow values. "
        "Categorical fields are encoded using the "
        "training preprocessing artifact and numeric "
        "features are standardized using the saved scaler."
    )


    with st.form(
        "manual_traffic_form"
    ):

        st.subheader(
            "Protocol and Connection"
        )


        c1, c2, c3 = st.columns(
            3
        )


        sport = c1.number_input(
            "Source Port (sport)",
            min_value=0,
            max_value=65535,
            value=12345
        )


        dsport = c2.number_input(
            "Destination Port (dsport)",
            min_value=0,
            max_value=65535,
            value=80
        )


        proto = c3.text_input(
            "Protocol (proto)",
            value="tcp",
            help="Examples: tcp, udp, arp, icmp"
        )


        c1, c2, c3 = st.columns(
            3
        )


        state = c1.text_input(
            "Connection State (state)",
            value="FIN",
            help="Example: FIN, INT, CON, REQ"
        )


        service = c2.text_input(
            "Service",
            value="-",
            help="Examples: -, http, ftp, smtp, dns"
        )


        dur = c3.number_input(
            "Duration (dur)",
            min_value=0.0,
            value=1.0,
            format="%.6f"
        )


        st.subheader(
            "Traffic Volume"
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        sbytes = c1.number_input(
            "Source Bytes",
            min_value=0.0,
            value=500.0
        )


        dbytes = c2.number_input(
            "Destination Bytes",
            min_value=0.0,
            value=500.0
        )


        spkts = c3.number_input(
            "Source Packets",
            min_value=0.0,
            value=10.0
        )


        dpkts = c4.number_input(
            "Destination Packets",
            min_value=0.0,
            value=10.0
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        sload = c1.number_input(
            "Source Load",
            min_value=0.0,
            value=1000.0
        )


        dload = c2.number_input(
            "Destination Load",
            min_value=0.0,
            value=1000.0
        )


        sloss = c3.number_input(
            "Source Loss",
            min_value=0.0,
            value=0.0
        )


        dloss = c4.number_input(
            "Destination Loss",
            min_value=0.0,
            value=0.0
        )


        st.subheader(
            "TTL and TCP Information"
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        sttl = c1.number_input(
            "Source TTL",
            min_value=0.0,
            value=64.0
        )


        dttl = c2.number_input(
            "Destination TTL",
            min_value=0.0,
            value=64.0
        )


        swin = c3.number_input(
            "Source TCP Window",
            min_value=0.0,
            value=255.0
        )


        dwin = c4.number_input(
            "Destination TCP Window",
            min_value=0.0,
            value=255.0
        )


        c1, c2 = st.columns(
            2
        )


        stcpb = c1.number_input(
            "Source TCP Base Sequence",
            min_value=0.0,
            value=0.0
        )


        dtcpb = c2.number_input(
            "Destination TCP Base Sequence",
            min_value=0.0,
            value=0.0
        )


        st.subheader(
            "Packet Statistics"
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        smean = c1.number_input(
            "Source Mean Packet Size",
            min_value=0.0,
            value=50.0
        )


        dmean = c2.number_input(
            "Destination Mean Packet Size",
            min_value=0.0,
            value=50.0
        )


        sjit = c3.number_input(
            "Source Jitter",
            min_value=0.0,
            value=0.0
        )


        djit = c4.number_input(
            "Destination Jitter",
            min_value=0.0,
            value=0.0
        )


        c1, c2 = st.columns(
            2
        )


        sinpkt = c1.number_input(
            "Source Inter-Packet Time",
            min_value=0.0,
            value=0.0
        )


        dinpkt = c2.number_input(
            "Destination Inter-Packet Time",
            min_value=0.0,
            value=0.0
        )


        st.subheader(
            "TCP Timing"
        )


        c1, c2, c3 = st.columns(
            3
        )


        tcprtt = c1.number_input(
            "TCP RTT",
            min_value=0.0,
            value=0.0
        )


        synack = c2.number_input(
            "SYN-ACK Time",
            min_value=0.0,
            value=0.0
        )


        ackdat = c3.number_input(
            "ACK-DATA Time",
            min_value=0.0,
            value=0.0
        )


        st.subheader(
            "HTTP / FTP / Connection Features"
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        trans_depth = c1.number_input(
            "Transaction Depth",
            min_value=0.0,
            value=0.0
        )


        response_body_len = c2.number_input(
            "Response Body Length",
            min_value=0.0,
            value=0.0
        )


        is_sm_ips_ports = c3.number_input(
            "Same IP/Port Indicator",
            min_value=0,
            max_value=1,
            value=0
        )


        ct_state_ttl = c4.number_input(
            "State-TTL Count",
            min_value=0.0,
            value=0.0
        )


        c1, c2, c3 = st.columns(
            3
        )


        ct_flw_http_mthd = c1.number_input(
            "HTTP Method Count",
            min_value=0.0,
            value=0.0
        )


        is_ftp_login = c2.number_input(
            "FTP Login Indicator",
            min_value=0,
            max_value=1,
            value=0
        )


        ct_ftp_cmd = c3.number_input(
            "FTP Command Count",
            min_value=0.0,
            value=0.0
        )


        st.subheader(
            "Connection Count Features"
        )


        c1, c2, c3 = st.columns(
            3
        )


        ct_srv_src = c1.number_input(
            "ct_srv_src",
            min_value=0.0,
            value=1.0
        )


        ct_srv_dst = c2.number_input(
            "ct_srv_dst",
            min_value=0.0,
            value=1.0
        )


        ct_dst_ltm = c3.number_input(
            "ct_dst_ltm",
            min_value=0.0,
            value=1.0
        )


        c1, c2, c3, c4 = st.columns(
            4
        )


        ct_src_ltm = c1.number_input(
            "ct_src_ltm",
            min_value=0.0,
            value=1.0
        )


        ct_src_dport_ltm = c2.number_input(
            "ct_src_dport_ltm",
            min_value=0.0,
            value=1.0
        )


        ct_dst_sport_ltm = c3.number_input(
            "ct_dst_sport_ltm",
            min_value=0.0,
            value=1.0
        )


        ct_dst_src_ltm = c4.number_input(
            "ct_dst_src_ltm",
            min_value=0.0,
            value=1.0
        )


        show_manual_shap = st.checkbox(
            "Generate SHAP explanation",
            value=True
        )


        submitted = st.form_submit_button(
            "🔍 Analyze Network Traffic",
            type="primary",
            use_container_width=True
        )


    if submitted:

        raw_values = {

            "sport": sport,
            "dsport": dsport,
            "proto": proto.strip(),
            "state": state.strip(),
            "dur": dur,

            "sbytes": sbytes,
            "dbytes": dbytes,

            "sttl": sttl,
            "dttl": dttl,

            "sloss": sloss,
            "dloss": dloss,

            "service": service.strip(),

            "sload": sload,
            "dload": dload,

            "spkts": spkts,
            "dpkts": dpkts,

            "swin": swin,
            "dwin": dwin,

            "stcpb": stcpb,
            "dtcpb": dtcpb,

            "smean": smean,
            "dmean": dmean,

            "trans_depth":
                trans_depth,

            "response_body_len":
                response_body_len,

            "sjit": sjit,
            "djit": djit,

            "sinpkt": sinpkt,
            "dinpkt": dinpkt,

            "tcprtt": tcprtt,
            "synack": synack,
            "ackdat": ackdat,

            "is_sm_ips_ports":
                is_sm_ips_ports,

            "ct_state_ttl":
                ct_state_ttl,

            "ct_flw_http_mthd":
                ct_flw_http_mthd,

            "is_ftp_login":
                is_ftp_login,

            "ct_ftp_cmd":
                ct_ftp_cmd,

            "ct_srv_src":
                ct_srv_src,

            "ct_srv_dst":
                ct_srv_dst,

            "ct_dst_ltm":
                ct_dst_ltm,

            "ct_src_ltm":
                ct_src_ltm,

            "ct_src_dport_ltm":
                ct_src_dport_ltm,

            "ct_dst_sport_ltm":
                ct_dst_sport_ltm,

            "ct_dst_src_ltm":
                ct_dst_src_ltm
        }


        try:

            processed = (
                preprocess_manual_input(
                    raw_values
                )
            )


            result = (
                predict_processed_record(
                    processed[0]
                )
            )


            display_result(
                result,
                actual_class=None,
                show_shap=show_manual_shap
            )


        except Exception as e:

            st.error(
                "Unable to process the manually "
                "entered traffic."
            )

            st.exception(e)


# ============================================================
# MODE 3
# MODEL INFORMATION
# ============================================================

elif mode == "Model Information":

    st.header(
        "Final IDS Model Information"
    )


    st.subheader(
        "Architecture"
    )


    st.code(
        """
UNSW-NB15 Network Traffic
          ↓
43 Usable Features
          ↓
5 Simulated Federated Clients
          ↓
1D CNN
          ↓
Bidirectional GRU
          ↓
Multi-Head Self-Attention
          ↓
Dense Classification Layers
          ↓
10-Class Softmax
          ↓
Weighted FedAvg
          ↓
Global Intrusion Detection Model
          ↓
SHAP Explainable AI
        """,
        language="text"
    )


    st.subheader(
        "Model Components"
    )


    architecture_df = pd.DataFrame({

        "Component": [
            "1D CNN",
            "BiGRU",
            "Multi-Head Attention",
            "FedAvg",
            "Softmax",
            "SHAP"
        ],

        "Purpose": [
            "Extracts local patterns from network features.",
            "Learns bidirectional relationships in the feature representation.",
            "Focuses on multiple important feature interactions.",
            "Aggregates simulated client model updates into a global model.",
            "Produces probabilities for the 10 traffic classes.",
            "Explains which features influenced individual predictions."
        ]
    })


    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Dataset Configuration"
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Training Records",
        "2,032,037"
    )


    c2.metric(
        "Test Records",
        "508,010"
    )


    c3.metric(
        "Features",
        "43"
    )


    c4.metric(
        "Classes",
        "10"
    )


    st.subheader(
        "Final V2 Evaluation"
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(
        "Accuracy",
        "97.00%"
    )


    c2.metric(
        "Weighted F1",
        "97.37%"
    )


    c3.metric(
        "Macro F1",
        "49.87%"
    )


    st.info(
        "Overall accuracy is high, but class-level "
        "performance is not uniform because UNSW-NB15 "
        "is highly imbalanced. Backdoor and Worms were "
        "particularly difficult for this experiment. "
        "Therefore, overall accuracy should not be used "
        "alone to describe model performance."
    )


    st.subheader(
        "Output Classes"
    )


    class_df = pd.DataFrame({

        "Class": CLASS_NAMES,

        "Description": [
            ATTACK_DESCRIPTIONS.get(
                str(name),
                ""
            )
            for name in CLASS_NAMES
        ]
    })


    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Important Experimental Scope"
    )


    st.write(
        """
        This project uses simulated federated learning.
        The five clients are created from partitions of the
        UNSW-NB15 training data and their locally trained model
        parameters are combined using weighted FedAvg.

        This demonstration should therefore not be described as
        a production deployment across five independent real-world
        organizations.

        SHAP explanations indicate features that influenced a
        prediction. They should not be interpreted as proof that
        a particular network feature caused an attack.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Federated Explainable Intrusion Detection System | "
    "CNN-BiGRU-Multi-Head Attention | UNSW-NB15 | SHAP"
)