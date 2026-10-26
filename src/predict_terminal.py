import os
import sys
import joblib
import shap
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "federated_cnn_lstm_multiclass_smote.keras"
)

ARTIFACT_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "preprocessing_multiclass_stratified.joblib"
)

RAW_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "UNSW-NB15_1.csv"
)

VERIFIED_DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "verified_unseen_with_labels.csv"
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


# ============================================================
# LOAD MODEL + PREPROCESSING ARTIFACTS
# ============================================================

print("\n" + "=" * 72)
print("SECURE IoT INTRUSION DETECTION SYSTEM")
print("=" * 72)

print("\nLoading Federated CNN-LSTM model...")

try:

    model = load_model(
        MODEL_FILE,
        compile=False
    )

    artifacts = joblib.load(
        ARTIFACT_FILE
    )

except Exception as error:

    print("\nERROR loading model/artifacts:")
    print(error)

    sys.exit(1)


feature_names = list(
    artifacts["all_feature_names"]
)

class_names = list(
    artifacts["class_names"]
)

categorical_encoders = artifacts[
    "categorical_encoders"
]

scaler = artifacts[
    "scaler"
]

selector = artifacts[
    "selector"
]


if "selected_feature_names" in artifacts:

    selected_feature_names = list(
        artifacts["selected_feature_names"]
    )

else:

    selected_feature_names = list(
        np.array(feature_names)[
            selector.get_support()
        ]
    )


print("Model loaded successfully.")

print(
    f"Original Features : {len(feature_names)}"
)

print(
    f"Selected Features : {len(selected_feature_names)}"
)

print(
    f"Attack Classes    : {len(class_names)}"
)


# ============================================================
# CLEAN ATTACK CATEGORY
# ============================================================

def clean_attack_category(
    value,
    binary_label=None
):

    # Raw UNSW-NB15 Normal records may have
    # empty/NaN attack category.
    # label = 0 means Normal.

    if binary_label is not None:

        try:

            if int(float(binary_label)) == 0:
                return "Normal"

        except (ValueError, TypeError):
            pass

    if pd.isna(value):
        return "Normal"

    value = str(value).strip()

    value = value.replace(
        '"',
        ""
    )

    value = value.replace(
        "'",
        ""
    )

    value = value.strip()

    if value.lower() in [
        "",
        "nan",
        "none",
        "normal"
    ]:

        return "Normal"

    category_mapping = {

        "analysis": "Analysis",

        "backdoor": "Backdoor",

        "backdoors": "Backdoor",

        "dos": "DoS",

        "exploits": "Exploits",

        "fuzzers": "Fuzzers",

        "generic": "Generic",

        "reconnaissance":
            "Reconnaissance",

        "shellcode": "Shellcode",

        "worms": "Worms"
    }

    return category_mapping.get(
        value.lower(),
        value
    )


# ============================================================
# USER INPUT HELPERS
# ============================================================

def get_number(
    prompt,
    default=0.0
):

    while True:

        value = input(
            prompt
        ).strip()

        if value == "":

            return float(
                default
            )

        try:

            return float(
                value
            )

        except ValueError:

            print(
                "Please enter a valid numeric value."
            )


def get_text(
    prompt,
    default="-"
):

    value = input(
        prompt
    ).strip()

    if value == "":
        return default

    return value


# ============================================================
# PREPROCESS ONE RECORD
# ============================================================

def preprocess_record(
    record,
    show_warnings=True
):

    missing_features = [

        feature

        for feature in feature_names

        if feature not in record.index
    ]

    if missing_features:

        raise ValueError(

            "Missing required features: "

            + ", ".join(
                missing_features
            )
        )

    input_df = pd.DataFrame(
        [record[feature_names]]
    ).copy()


    # ========================================================
    # CATEGORICAL ENCODING
    # ========================================================

    for (
        column_name,
        encoder
    ) in categorical_encoders.items():

        if (
            column_name
            not in input_df.columns
        ):

            continue

        mapping = {

            str(value)
            .strip()
            .lower(): index

            for index, value

            in enumerate(
                encoder.classes_
            )
        }

        current_value = str(
            input_df.iloc[0][
                column_name
            ]
        ).strip().lower()

        if (
            current_value
            not in mapping
            and show_warnings
        ):

            print(

                f"\nWARNING: Unknown "

                f"{column_name} value "

                f"'{current_value}'."

            )

            print(

                "Using encoded value 0."
            )

        input_df[
            column_name
        ] = mapping.get(
            current_value,
            0
        )


    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    input_df = input_df.apply(

        pd.to_numeric,

        errors="coerce"

    ).fillna(0)


    # ========================================================
    # SAME STANDARD SCALER USED DURING TRAINING
    # ========================================================

    scaled_data = scaler.transform(
        input_df
    )


    # ========================================================
    # SAME SELECTKBEST USED DURING TRAINING
    # ========================================================

    selected_data = selector.transform(
        scaled_data
    )


    # ========================================================
    # CNN-LSTM INPUT
    # Shape = (1, 32, 1)
    # ========================================================

    model_input = selected_data.reshape(

        1,

        selected_data.shape[1],

        1
    )

    return model_input


# ============================================================
# PREPARE SHAP BACKGROUND
# ============================================================

def prepare_shap_background():

    try:

        background_df = pd.read_csv(
            VERIFIED_DATA_FILE
        )

        if len(
            background_df
        ) > 50:

            background_df = (
                background_df.sample(
                    n=50,
                    random_state=42
                )
            )

        processed_records = []

        for (
            _,
            row
        ) in background_df.iterrows():

            try:

                processed = (
                    preprocess_record(
                        row,
                        show_warnings=False
                    )
                )

                processed_records.append(
                    processed[0]
                )

            except Exception:

                continue

        if (
            len(
                processed_records
            )
            == 0
        ):

            return None

        return np.asarray(

            processed_records,

            dtype=np.float32
        )

    except Exception as error:

        print(

            "\nWARNING: Could not prepare "

            "SHAP background data."
        )

        print(error)

        return None


# ============================================================
# CALCULATE SHAP
# ============================================================

def calculate_shap(

    model_input,

    predicted_index,

    background

):

    if background is None:

        return None

    try:

        print(
            "\nCalculating SHAP explanation..."
        )

        explainer = shap.GradientExplainer(

            model,

            background
        )

        shap_values = (
            explainer.shap_values(
                model_input
            )
        )


        # ====================================================
        # Different SHAP versions may return different shapes
        # ====================================================

        if isinstance(
            shap_values,
            list
        ):

            values = np.asarray(

                shap_values[
                    predicted_index
                ]
            )

            values = np.squeeze(
                values
            )

        else:

            shap_array = np.asarray(
                shap_values
            )

            # Expected shape:
            # (1, 32, 1, 10)

            if (

                shap_array.ndim == 4

                and

                shap_array.shape[-1]
                == len(class_names)

            ):

                values = shap_array[
                    0,
                    :,
                    0,
                    predicted_index
                ]

            # Possible shape:
            # (1, 32, 10)

            elif (

                shap_array.ndim == 3

                and

                shap_array.shape[-1]
                == len(class_names)

            ):

                values = shap_array[
                    0,
                    :,
                    predicted_index
                ]

            else:

                values = np.squeeze(
                    shap_array
                )


        values = np.asarray(
            values
        ).reshape(-1)


        if (

            len(values)

            !=

            len(
                selected_feature_names
            )

        ):

            print(
                "\nSHAP feature-size mismatch."
            )

            print(
                f"SHAP values : "
                f"{len(values)}"
            )

            print(
                f"Features    : "
                f"{len(selected_feature_names)}"
            )

            return None

        return values

    except Exception as error:

        print(
            "\nSHAP calculation failed:"
        )

        print(error)

        return None


# ============================================================
# DISPLAY SHAP + HUMAN-READABLE EXPLANATION
# ============================================================

def display_shap(

    shap_values,

    predicted_class,

    confidence

):

    if shap_values is None:

        print(
            "\nSHAP explanation unavailable."
        )

        return


    feature_contributions = list(

        zip(

            selected_feature_names,

            shap_values
        )
    )


    feature_contributions.sort(

        key=lambda item:
            abs(item[1]),

        reverse=True
    )


    top_features = (
        feature_contributions[:10]
    )


    print("\n" + "=" * 72)

    print(
        "SHAP EXPLANATION"
    )

    print("=" * 72)


    print(

        f"Explaining Prediction : "

        f"{predicted_class}"
    )


    print(
        "\nTop Influencing Features"
    )

    print("-" * 72)


    for (
        index,
        (
            feature,
            value
        )
    ) in enumerate(
        top_features,
        start=1
    ):

        if value > 0:

            influence = (

                f"Supports "

                f"{predicted_class}"
            )

        elif value < 0:

            influence = (

                f"Opposes "

                f"{predicted_class}"
            )

        else:

            influence = (
                "Neutral"
            )

        print(

            f"{index:>2}. "

            f"{feature:<22} "

            f"{value:+.6f}   "

            f"{influence}"
        )


    # ========================================================
    # INDIVIDUAL FEATURE EXPLANATIONS
    # ========================================================

    print(
        "\n" + "-" * 72
    )

    print(
        "FEATURE EXPLANATION"
    )

    print("-" * 72)


    for (
        index,
        (
            feature,
            value
        )
    ) in enumerate(
        top_features[:5],
        start=1
    ):

        print(
            f"\n{index}. {feature}"
        )

        print(

            f"   SHAP Value : "

            f"{value:+.6f}"
        )

        if value > 0:

            print(

                "   Explanation: "

                "This feature supported "

                "the model's "

                f"{predicted_class} "

                "prediction."
            )

        elif value < 0:

            print(

                "   Explanation: "

                "This feature opposed "

                "the model's "

                f"{predicted_class} "

                "prediction."
            )

        else:

            print(

                "   Explanation: "

                "This feature had almost "

                "no influence on the "

                "prediction."
            )


    # ========================================================
    # FINAL EXPLANATION
    # ========================================================

    supporting_features = [

        feature

        for (
            feature,
            value
        ) in top_features

        if value > 0

    ][:5]


    opposing_features = [

        feature

        for (
            feature,
            value
        ) in top_features

        if value < 0

    ][:5]


    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL EXPLANATION"
    )

    print("=" * 72)


    print(

        "The Federated CNN-LSTM model "

        "classified this network traffic "

        f"as {predicted_class} "

        f"with {confidence:.2f}% confidence."
    )


    if supporting_features:

        print(

            "\nThe main features supporting "

            "this prediction were:"
        )

        print(

            ", ".join(
                supporting_features
            )
        )

        print(

            "\nThese features increased "

            "the model's output for the "

            f"{predicted_class} class."
        )


    if opposing_features:

        print(

            "\nThe main features opposing "

            "this prediction were:"
        )

        print(

            ", ".join(
                opposing_features
            )
        )

        print(

            "\nThese features reduced "

            "the model's output for the "

            f"{predicted_class} class."
        )


    print(

        "\nImportant: SHAP explains how "

        "the trained model used these "

        "features for its prediction. "

        "It does not prove that a feature "

        "caused the attack."
    )


    print("=" * 72)


# ============================================================
# COMMON MODEL PREDICTION
# ============================================================

def predict_record(

    record,

    actual_category=None,

    verification_available=True

):

    try:

        model_input = preprocess_record(
            record
        )

    except Exception as error:

        print(
            "\nERROR during preprocessing:"
        )

        print(error)

        return


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict(

        model_input,

        verbose=0

    )[0]


    ranked_indices = np.argsort(
        probabilities
    )[::-1]


    predicted_index = int(
        ranked_indices[0]
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = float(

        probabilities[
            predicted_index
        ]

        * 100
    )


    # ========================================================
    # SECOND MOST LIKELY CLASS
    # ========================================================

    second_index = int(
        ranked_indices[1]
    )


    second_class = class_names[
        second_index
    ]


    second_confidence = float(

        probabilities[
            second_index
        ]

        * 100
    )


    # ========================================================
    # RESULT
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "INTRUSION DETECTION RESULT"
    )

    print("=" * 72)


    if predicted_class == "Normal":

        print(
            "Traffic Status       : NORMAL TRAFFIC"
        )

    else:

        print(
            "Traffic Status       : ATTACK DETECTED"
        )


    print(

        f"Predicted Attack     : "

        f"{predicted_class}"
    )


    print(

        f"Confidence           : "

        f"{confidence:.2f}%"
    )


    # ========================================================
    # KNOWN GROUND TRUTH
    # OPTION 1 AND OPTION 2
    # ========================================================

    if (

        verification_available

        and

        actual_category is not None

    ):

        print(

            f"Actual Attack        : "

            f"{actual_category}"
        )


        actual_normalized = str(
            actual_category
        ).strip().lower()


        predicted_normalized = str(
            predicted_class
        ).strip().lower()


        if (
            actual_normalized
            ==
            predicted_normalized
        ):

            print(
                "Prediction Status    : CORRECT"
            )

            print(

                f"Correct Attack       : "

                f"{actual_category}"
            )

        else:

            print(
                "Prediction Status    : INCORRECT"
            )

            print(

                f"Correct Attack       : "

                f"{actual_category}"
            )

            print(

                f"Model Predicted      : "

                f"{predicted_class}"
            )

            print(

                "\nVerification Explanation: "

                "The model prediction does not "

                "match the known ground-truth "

                "label stored in the dataset."
            )


    # ========================================================
    # UNKNOWN USER TRAFFIC
    # OPTION 3
    # ========================================================

    else:

        print(
            "Prediction Status    : UNKNOWN"
        )

        print(
            "Correct Attack       : NOT AVAILABLE"
        )

        print(

            "Verification         : "

            "This is new user-provided traffic, "

            "so no ground-truth label is available."
        )


        print(

            f"Next Likely Class    : "

            f"{second_class}"
        )


        print(

            f"Next Class Confidence: "

            f"{second_confidence:.2f}%"
        )


        # Confidence description
        if confidence >= 80:

            certainty = "HIGH"

        elif confidence >= 50:

            certainty = "MODERATE"

        else:

            certainty = "LOW"


        print(

            f"Model Certainty      : "

            f"{certainty}"
        )


        print(

            "\nImportant: The Next Likely Class "

            "is only the model's second-highest "

            "prediction. It is NOT a verified "

            "correct answer."
        )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    print(
        "\n" + "-" * 72
    )

    print(
        "CLASS PROBABILITIES"
    )

    print("-" * 72)


    probability_pairs = list(

        zip(

            class_names,

            probabilities
        )
    )


    probability_pairs.sort(

        key=lambda item:
            item[1],

        reverse=True
    )


    for (
        category,
        probability
    ) in probability_pairs:

        print(

            f"{category:<20} : "

            f"{probability * 100:6.2f}%"
        )


    # ========================================================
    # SHAP
    # ========================================================

    print(

        "\nPreparing Explainable AI analysis..."
    )


    background = prepare_shap_background()


    shap_values = calculate_shap(

        model_input,

        predicted_index,

        background
    )


    display_shap(

        shap_values,

        predicted_class,

        confidence
    )


# ============================================================
# OPTION 1
# FULL RAW UNSW-NB15 DATASET
# ============================================================

def predict_from_full_dataset():

    print(

        "\nLoading full UNSW-NB15 raw dataset..."
    )


    try:

        raw_df = pd.read_csv(

            RAW_DATA_FILE,

            header=None,

            names=RAW_COLUMNS,

            low_memory=False
        )

    except Exception as error:

        print(
            "\nCould not load raw dataset:"
        )

        print(error)

        return


    total_records = len(
        raw_df
    )


    print(

        f"\nTotal records available: "

        f"{total_records}"
    )


    try:

        record_number = int(

            input(

                f"\nEnter record number "

                f"(1-{total_records}): "
            )
        )

    except ValueError:

        print(
            "\nPlease enter a valid record number."
        )

        return


    if (

        record_number < 1

        or

        record_number > total_records

    ):

        print(
            "\nRecord number is outside dataset range."
        )

        return


    record = raw_df.iloc[
        record_number - 1
    ].copy()


    # ========================================================
    # CALCULATE RATE
    # ========================================================

    duration = pd.to_numeric(

        record["dur"],

        errors="coerce"
    )


    spkts = pd.to_numeric(

        record["spkts"],

        errors="coerce"
    )


    dpkts = pd.to_numeric(

        record["dpkts"],

        errors="coerce"
    )


    if (

        pd.isna(duration)

        or duration <= 0

        or pd.isna(spkts)

        or pd.isna(dpkts)

    ):

        rate = 0.0

    else:

        rate = max(

            0.0,

            (
                float(spkts)

                + float(dpkts)

                - 1.0
            )

            / float(duration)
        )


    record["rate"] = rate


    # ========================================================
    # GET GROUND-TRUTH LABEL AUTOMATICALLY
    # ========================================================

    actual_category = clean_attack_category(

        record["attack_cat"],

        record["label"]
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "SELECTED FULL-DATASET RECORD"
    )

    print("=" * 72)


    print(
        f"Record Number       : {record_number}"
    )

    print(
        f"Protocol            : {record['proto']}"
    )

    print(
        f"Service             : {record['service']}"
    )

    print(
        f"State               : {record['state']}"
    )

    print(
        f"Duration            : {record['dur']}"
    )

    print(
        f"Source Packets      : {record['spkts']}"
    )

    print(
        f"Destination Packets : {record['dpkts']}"
    )

    print(
        f"Derived Rate        : {record['rate']}"
    )


    predict_record(

        record,

        actual_category=actual_category,

        verification_available=True
    )


# ============================================================
# OPTION 2
# VERIFIED 151-RECORD SUBSET
# ============================================================

def predict_from_verified_dataset():

    print(

        "\nLoading verified non-overlapping "

        "evaluation subset..."
    )


    try:

        verified_df = pd.read_csv(
            VERIFIED_DATA_FILE
        )

    except Exception as error:

        print(
            "\nCould not load verified dataset:"
        )

        print(error)

        return


    total_records = len(
        verified_df
    )


    print(

        f"\nTotal verified records: "

        f"{total_records}"
    )


    try:

        record_number = int(

            input(

                f"\nEnter record number "

                f"(1-{total_records}): "
            )
        )

    except ValueError:

        print(
            "\nPlease enter a valid record number."
        )

        return


    if (

        record_number < 1

        or

        record_number > total_records

    ):

        print(
            "\nRecord number is outside dataset range."
        )

        return


    record = verified_df.iloc[
        record_number - 1
    ].copy()


    actual_category = (
        clean_attack_category(
            record[
                "Actual_Category"
            ]
        )
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "SELECTED VERIFIED RECORD"
    )

    print("=" * 72)


    print(
        f"Record Number       : {record_number}"
    )

    print(
        f"Protocol            : {record['proto']}"
    )

    print(
        f"Service             : {record['service']}"
    )

    print(
        f"State               : {record['state']}"
    )

    print(
        f"Duration            : {record['dur']}"
    )

    print(
        f"Rate                : {record['rate']}"
    )


    predict_record(

        record,

        actual_category=actual_category,

        verification_available=True
    )


# ============================================================
# OPTION 3
# ENTER COMPLETELY NEW / UNKNOWN TRAFFIC
# ============================================================

def predict_custom_traffic():

    print(
        "\n" + "=" * 72
    )

    print(
        "ENTER YOUR OWN UNKNOWN NETWORK TRAFFIC"
    )

    print("=" * 72)


    print(

        "\nYou do NOT need to enter "

        "the attack name or actual label."
    )


    print(

        "The trained model will automatically "

        "predict Normal or one of the attack classes."
    )


    print(

        "\nFor a numeric feature that you "

        "do not know, press Enter to use 0."
    )


    record = {}


    # ========================================================
    # 1. CONNECTION INFORMATION
    # ========================================================

    print(
        "\n--- 1. CONNECTION INFORMATION ---"
    )


    record["proto"] = get_text(

        "Protocol (tcp/udp/etc.): ",

        "tcp"
    )


    record["service"] = get_text(

        "Service (http/dns/- etc.): ",

        "-"
    )


    record["state"] = get_text(

        "State (FIN/CON/INT etc.): ",

        "FIN"
    )


    # ========================================================
    # 2. BASIC TRAFFIC
    # ========================================================

    print(
        "\n--- 2. BASIC TRAFFIC ---"
    )


    record["dur"] = get_number(
        "Duration (seconds): "
    )


    record["spkts"] = get_number(
        "Source packets (spkts): "
    )


    record["dpkts"] = get_number(
        "Destination packets (dpkts): "
    )


    record["sbytes"] = get_number(
        "Source bytes (sbytes): "
    )


    record["dbytes"] = get_number(
        "Destination bytes (dbytes): "
    )


    # ========================================================
    # 3. TTL / LOSS
    # ========================================================

    print(
        "\n--- 3. TTL / PACKET LOSS ---"
    )


    record["sttl"] = get_number(
        "Source TTL (sttl): "
    )


    record["dttl"] = get_number(
        "Destination TTL (dttl): "
    )


    record["sloss"] = get_number(
        "Source packet loss (sloss): "
    )


    record["dloss"] = get_number(
        "Destination packet loss (dloss): "
    )


    # ========================================================
    # 4. NETWORK LOAD
    # ========================================================

    print(
        "\n--- 4. NETWORK LOAD ---"
    )


    record["sload"] = get_number(
        "Source load (sload): "
    )


    record["dload"] = get_number(
        "Destination load (dload): "
    )


    # ========================================================
    # 5. PACKET TIMING
    # ========================================================

    print(
        "\n--- 5. PACKET TIMING ---"
    )


    record["sinpkt"] = get_number(
        "Source inter-packet time (sinpkt): "
    )


    record["dinpkt"] = get_number(
        "Destination inter-packet time (dinpkt): "
    )


    record["sjit"] = get_number(
        "Source jitter (sjit): "
    )


    record["djit"] = get_number(
        "Destination jitter (djit): "
    )


    # ========================================================
    # 6. TCP INFORMATION
    # ========================================================

    print(
        "\n--- 6. TCP INFORMATION ---"
    )


    record["swin"] = get_number(
        "Source TCP window (swin): "
    )


    record["dwin"] = get_number(
        "Destination TCP window (dwin): "
    )


    record["stcpb"] = get_number(
        "Source TCP base sequence (stcpb): "
    )


    record["dtcpb"] = get_number(
        "Destination TCP base sequence (dtcpb): "
    )


    record["tcprtt"] = get_number(
        "TCP round-trip time (tcprtt): "
    )


    record["synack"] = get_number(
        "SYN-ACK time (synack): "
    )


    record["ackdat"] = get_number(
        "ACK data time (ackdat): "
    )


    # ========================================================
    # 7. PACKET SIZE
    # ========================================================

    print(
        "\n--- 7. PACKET SIZE INFORMATION ---"
    )


    record["smean"] = get_number(
        "Mean source packet size (smean): "
    )


    record["dmean"] = get_number(
        "Mean destination packet size (dmean): "
    )


    # ========================================================
    # 8. HTTP / FTP
    # ========================================================

    print(
        "\n--- 8. HTTP / FTP INFORMATION ---"
    )


    record["trans_depth"] = get_number(
        "HTTP transaction depth (trans_depth): "
    )


    record["response_body_len"] = get_number(
        "HTTP response body length: "
    )


    record["is_ftp_login"] = get_number(
        "FTP login flag (0/1): "
    )


    record["ct_ftp_cmd"] = get_number(
        "FTP command count: "
    )


    record["ct_flw_http_mthd"] = get_number(
        "HTTP method count: "
    )


    # ========================================================
    # 9. CONNECTION COUNTS
    # ========================================================

    print(
        "\n--- 9. CONNECTION COUNT FEATURES ---"
    )


    record["ct_srv_src"] = get_number(
        "ct_srv_src: "
    )


    record["ct_state_ttl"] = get_number(
        "ct_state_ttl: "
    )


    record["ct_dst_ltm"] = get_number(
        "ct_dst_ltm: "
    )


    record["ct_src_dport_ltm"] = get_number(
        "ct_src_dport_ltm: "
    )


    record["ct_dst_sport_ltm"] = get_number(
        "ct_dst_sport_ltm: "
    )


    record["ct_dst_src_ltm"] = get_number(
        "ct_dst_src_ltm: "
    )


    record["ct_src_ltm"] = get_number(
        "ct_src_ltm: "
    )


    record["ct_srv_dst"] = get_number(
        "ct_srv_dst: "
    )


    record["is_sm_ips_ports"] = get_number(
        "Same IP/port flag (0/1): "
    )


    # ========================================================
    # AUTOMATICALLY CALCULATE RATE
    # ========================================================

    duration = record["dur"]

    spkts = record["spkts"]

    dpkts = record["dpkts"]


    if duration > 0:

        record["rate"] = max(

            0.0,

            (
                spkts
                + dpkts
                - 1.0
            )

            / duration
        )

    else:

        record["rate"] = 0.0


    # ========================================================
    # FILL OTHER ORIGINAL MODEL FEATURES
    # ========================================================

    for feature in feature_names:

        if feature not in record:

            if feature in [
                "proto",
                "service",
                "state"
            ]:

                continue

            record[
                feature
            ] = 0.0


    record = pd.Series(
        record
    )


    # ========================================================
    # SHOW ENTERED DATA SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "ENTERED TRAFFIC SUMMARY"
    )

    print("=" * 72)


    print(
        f"Protocol            : {record['proto']}"
    )


    print(
        f"Service             : {record['service']}"
    )


    print(
        f"State               : {record['state']}"
    )


    print(
        f"Duration            : {record['dur']}"
    )


    print(
        f"Source Packets      : {record['spkts']}"
    )


    print(
        f"Destination Packets : {record['dpkts']}"
    )


    print(
        f"Source Bytes        : {record['sbytes']}"
    )


    print(
        f"Destination Bytes   : {record['dbytes']}"
    )


    print(
        f"Calculated Rate     : {record['rate']:.6f}"
    )


    # ========================================================
    # UNKNOWN DATA:
    # No actual label is requested from the user.
    # ========================================================

    predict_record(

        record,

        actual_category=None,

        verification_available=False
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    while True:

        print("\n")

        print("=" * 72)

        print(
            "SECURE IoT INTRUSION DETECTION SYSTEM"
        )

        print("=" * 72)


        print(

            "Federated CNN-LSTM | "

            "Multi-Class IDS | "

            "SHAP Explainable AI"
        )


        print("-" * 72)


        print(

            "1. Predict from Full "

            "UNSW-NB15 Dataset"
        )


        print(

            "2. Predict from Verified "

            "151-Record Evaluation Subset"
        )


        print(

            "3. Enter My Own Unknown "

            "Network Traffic"
        )


        print(
            "4. Exit"
        )


        print("-" * 72)


        choice = input(

            "Enter your choice (1/2/3/4): "

        ).strip()


        if choice == "1":

            predict_from_full_dataset()


        elif choice == "2":

            predict_from_verified_dataset()


        elif choice == "3":

            predict_custom_traffic()


        elif choice == "4":

            print(

                "\nIntrusion Detection System closed."
            )

            break


        else:

            print(

                "\nInvalid choice. "

                "Please enter 1, 2, 3 or 4."
            )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()