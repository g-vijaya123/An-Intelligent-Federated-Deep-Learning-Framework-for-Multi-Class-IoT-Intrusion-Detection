import os

import joblib

import shap

import numpy as np

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt



from tensorflow.keras.models import load_model





# ============================================================

# PAGE CONFIG

# ============================================================



st.set_page_config(

    page_title="Secure IoT IDS",

    page_icon="🛡️",

    layout="wide"

)



st.title("🛡️ Secure IoT Intrusion Detection System")

st.caption(

    "Federated CNN-LSTM | Multi-Class IDS | SHAP Explainable AI"

)





# ============================================================

# PATHS

# ============================================================



PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))



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

# RAW DATA COLUMNS

# ============================================================



RAW_COLUMNS = [

    "srcip", "sport", "dstip", "dsport",

    "proto", "state", "dur", "sbytes", "dbytes",

    "sttl", "dttl", "sloss", "dloss",

    "service", "sload", "dload",

    "spkts", "dpkts", "swin", "dwin",

    "stcpb", "dtcpb", "smean", "dmean",

    "trans_depth", "response_body_len",

    "sjit", "djit", "stime", "ltime",

    "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",

    "is_sm_ips_ports", "ct_state_ttl",

    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",

    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",

    "ct_src_ltm", "ct_src_dport_ltm",

    "ct_dst_sport_ltm", "ct_dst_src_ltm",

    "attack_cat", "label"

]





# ============================================================

# LOAD MODEL

# ============================================================



@st.cache_resource

def load_system():



    model = load_model(

        MODEL_FILE,

        compile=False

    )



    artifacts = joblib.load(

        ARTIFACT_FILE

    )



    return model, artifacts





try:

    model, artifacts = load_system()



except Exception as e:



    st.error(f"Model loading failed: {e}")

    st.stop()





feature_names = list(

    artifacts["all_feature_names"]

)



class_names = list(

    artifacts["class_names"]

)



categorical_encoders = artifacts[

    "categorical_encoders"

]



scaler = artifacts["scaler"]

selector = artifacts["selector"]



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





# ============================================================

# LOAD DATA

# ============================================================



@st.cache_data

def load_raw_data():



    return pd.read_csv(

        RAW_DATA_FILE,

        header=None,

        names=RAW_COLUMNS,

        low_memory=False

    )





@st.cache_data

def load_verified_data():



    return pd.read_csv(

        VERIFIED_DATA_FILE

    )





# ============================================================

# CLEAN ATTACK CATEGORY

# ============================================================



def clean_attack_category(

    value,

    binary_label=None

):



    if binary_label is not None:



        try:

            if int(float(binary_label)) == 0:

                return "Normal"

        except:

            pass



    if pd.isna(value):

        return "Normal"



    value = str(value).strip()



    value = value.replace('"', "")

    value = value.replace("'", "")



    mapping = {

        "analysis": "Analysis",

        "backdoor": "Backdoor",

        "backdoors": "Backdoor",

        "dos": "DoS",

        "exploits": "Exploits",

        "fuzzers": "Fuzzers",

        "generic": "Generic",

        "normal": "Normal",

        "reconnaissance": "Reconnaissance",

        "shellcode": "Shellcode",

        "worms": "Worms"

    }



    if value.lower() in [

        "",

        "nan",

        "none"

    ]:

        return "Normal"



    return mapping.get(

        value.lower(),

        value

    )





# ============================================================

# PREPROCESS

# ============================================================



def preprocess_record(record):



    record = record.copy()



    for feature in feature_names:



        if feature not in record.index:



            if feature in [

                "proto",

                "service",

                "state"

            ]:

                record[feature] = "-"

            else:

                record[feature] = 0.0



    input_df = pd.DataFrame(

        [record[feature_names]]

    ).copy()



    for (

        column_name,

        encoder

    ) in categorical_encoders.items():



        mapping = {

            str(value).strip().lower(): index

            for index, value

            in enumerate(encoder.classes_)

        }



        current = str(

            input_df.iloc[0][column_name]

        ).strip().lower()



        input_df[column_name] = mapping.get(

            current,

            0

        )



    input_df = input_df.apply(

        pd.to_numeric,

        errors="coerce"

    ).fillna(0)



    scaled = scaler.transform(

        input_df

    )



    selected = selector.transform(

        scaled

    )



    return selected.reshape(

        1,

        selected.shape[1],

        1

    )





# ============================================================

# PREDICTION

# ============================================================



def make_prediction(record):



    model_input = preprocess_record(

        record

    )



    probabilities = model.predict(

        model_input,

        verbose=0

    )[0]



    ranked = np.argsort(

        probabilities

    )[::-1]



    predicted_index = int(

        ranked[0]

    )



    second_index = int(

        ranked[1]

    )



    return {

        "input": model_input,



        "probabilities": probabilities,



        "predicted_index":

            predicted_index,



        "predicted_class":

            class_names[predicted_index],



        "confidence":

            float(

                probabilities[

                    predicted_index

                ] * 100

            ),



        "second_class":

            class_names[second_index],



        "second_confidence":

            float(

                probabilities[

                    second_index

                ] * 100

            )

    }





# ============================================================

# SHAP BACKGROUND

# ============================================================



@st.cache_resource

def prepare_background():



    df = load_verified_data()



    df = df.sample(

        n=min(50, len(df)),

        random_state=42

    )



    background = []



    for _, row in df.iterrows():



        try:

            x = preprocess_record(row)

            background.append(x[0])

        except:

            pass



    return np.asarray(

        background,

        dtype=np.float32

    )





# ============================================================

# SHAP

# ============================================================



def calculate_shap(

    model_input,

    predicted_index

):



    background = prepare_background()



    explainer = shap.GradientExplainer(

        model,

        background

    )



    shap_values = explainer.shap_values(

        model_input

    )



    if isinstance(shap_values, list):



        values = np.asarray(

            shap_values[

                predicted_index

            ]

        )



        values = np.squeeze(values)



    else:



        shap_array = np.asarray(

            shap_values

        )



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



    return np.asarray(

        values

    ).reshape(-1)





# ============================================================

# DISPLAY RESULTS

# ============================================================



def display_result(

    result,

    actual=None

):



    predicted = result[

        "predicted_class"

    ]



    confidence = result[

        "confidence"

    ]



    probabilities = result[

        "probabilities"

    ]



    st.divider()



    st.subheader(

        "🔍 Intrusion Detection Result"

    )



    col1, col2, col3 = st.columns(3)



    if predicted == "Normal":
        col1.success("✅ NORMAL TRAFFIC")
        col2.metric("Predicted Result", "Normal")
    else:
        col1.error(f"🚨 ATTACK DETECTED — {predicted}")
        col2.metric("Predicted Attack", predicted)

    col3.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    # For user-entered unknown traffic, make the model's prediction explicit.
    if actual is None:
        if predicted == "Normal":
            st.success(
                f"🛡️ MODEL PREDICTION: This traffic is classified as "
                f"**NORMAL** with **{confidence:.2f}% confidence**."
            )
        else:
            st.error(
                f"🚨 MODEL PREDICTION: **{predicted} ATTACK** detected "
                f"with **{confidence:.2f}% confidence**."
            )





    # ========================================================

    # KNOWN GROUND TRUTH

    # ========================================================



    if actual is not None:



        st.write(

            f"**Actual Attack:** {actual}"

        )



        if (

            actual.lower()

            ==

            predicted.lower()

        ):



            st.success(

                "✅ Prediction Status: CORRECT"

            )



            st.write(

                f"**Correct Attack:** {actual}"

            )



        else:



            st.error(

                "❌ Prediction Status: INCORRECT"

            )



            st.write(

                f"**Model Predicted:** {predicted}"

            )



            st.success(

                f"Correct Attack: {actual}"

            )





    # ========================================================

    # UNKNOWN TRAFFIC

    # ========================================================



    else:



        st.warning(

            "⚠️ Prediction Verification: UNKNOWN"

        )



        st.info(

            "Ground-truth label is not available "

            "for this new user-provided traffic."

        )



        col1, col2 = st.columns(2)



        col1.metric(

            "Next Likely Class",

            result["second_class"]

        )



        col2.metric(

            "Next Class Confidence",

            f"{result['second_confidence']:.2f}%"

        )



        st.caption(

            "The next likely class is the model's "

            "second-highest prediction. It is not "

            "a verified correct answer."

        )





    # ========================================================

    # PROBABILITIES

    # ========================================================



    st.subheader(

        "📊 Class Probabilities"

    )



    probability_df = pd.DataFrame({

        "Attack Class": class_names,

        "Probability (%)":

            probabilities * 100

    })



    probability_df = (

        probability_df.sort_values(

            "Probability (%)",

            ascending=False

        )

    )



    st.dataframe(

        probability_df,

        use_container_width=True,

        hide_index=True

    )



    st.bar_chart(

        probability_df.set_index(

            "Attack Class"

        )

    )





    # ========================================================

    # SHAP

    # ========================================================



    st.subheader(

        "🧠 SHAP Explainable AI"

    )



    with st.spinner(

        "Calculating SHAP explanation..."

    ):



        try:



            values = calculate_shap(

                result["input"],

                result[

                    "predicted_index"

                ]

            )



            shap_df = pd.DataFrame({

                "Feature":

                    selected_feature_names,



                "SHAP Value":

                    values

            })



            shap_df[

                "Absolute Importance"

            ] = abs(

                shap_df["SHAP Value"]

            )



            shap_df = (

                shap_df.sort_values(

                    "Absolute Importance",

                    ascending=False

                )

                .head(10)

            )



            shap_df[

                "Influence"

            ] = shap_df[

                "SHAP Value"

            ].apply(

                lambda x:

                f"Supports {predicted}"

                if x > 0

                else

                f"Opposes {predicted}"

            )



            st.dataframe(

                shap_df[

                    [

                        "Feature",

                        "SHAP Value",

                        "Influence"

                    ]

                ],

                use_container_width=True,

                hide_index=True

            )





            # =================================================

            # SHAP CHART

            # =================================================



            chart_df = (

                shap_df[

                    [

                        "Feature",

                        "SHAP Value"

                    ]

                ]

                .set_index("Feature")

            )



            st.bar_chart(

                chart_df

            )





            # =================================================

            # HUMAN READABLE EXPLANATION

            # =================================================



            supporting = (

                shap_df[

                    shap_df[

                        "SHAP Value"

                    ] > 0

                ]["Feature"]

                .head(5)

                .tolist()

            )



            opposing = (

                shap_df[

                    shap_df[

                        "SHAP Value"

                    ] < 0

                ]["Feature"]

                .head(5)

                .tolist()

            )





            if actual is None and predicted != "Normal":
                st.subheader(f"🧠 Why the model predicted {predicted}")
            elif actual is None:
                st.subheader("🧠 Why the model predicted Normal traffic")

            st.subheader(
                "📝 Final Explanation"
            )



            st.write(

                f"The Federated CNN-LSTM model "

                f"classified this network traffic "

                f"as **{predicted}** with "

                f"**{confidence:.2f}% confidence**."

            )



            if supporting:



                st.write(

                    "**Main features supporting "

                    "the prediction:**"

                )



                st.write(

                    ", ".join(

                        supporting

                    )

                )



            if opposing:



                st.write(

                    "**Main features opposing "

                    "the prediction:**"

                )



                st.write(

                    ", ".join(

                        opposing

                    )

                )



            st.info(

                "SHAP explains how the trained model "

                "used these features. It does not "

                "prove that a feature caused an attack."

            )



        except Exception as e:



            st.warning(

                f"SHAP explanation could not "

                f"be generated: {e}"

            )





# ============================================================

# SIDEBAR

# ============================================================



st.sidebar.title(

    "🛡️ IDS Navigation"

)



option = st.sidebar.radio(

    "Select Prediction Mode",

    [

        "Full UNSW-NB15 Dataset",

        "Verified 151 Records",

        "Enter Own Network Traffic"

    ]

)



st.sidebar.divider()



st.sidebar.write(

    f"**Original Features:** "

    f"{len(feature_names)}"

)



st.sidebar.write(

    f"**Selected Features:** "

    f"{len(selected_feature_names)}"

)



st.sidebar.write(

    f"**Attack Classes:** "

    f"{len(class_names)}"

)





# ============================================================

# OPTION 1

# ============================================================



if option == "Full UNSW-NB15 Dataset":



    st.header(

        "📁 Full UNSW-NB15 Dataset"

    )



    raw_df = load_raw_data()



    total = len(raw_df)



    st.write(

        f"Total Records: **{total:,}**"

    )



    record_number = st.number_input(

        "Enter Record Number",

        min_value=1,

        max_value=total,

        value=1,

        step=1

    )



    if st.button(

        "🔍 Predict Selected Record",

        type="primary"

    ):



        record = raw_df.iloc[

            int(record_number) - 1

        ].copy()



        dur = pd.to_numeric(

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

            pd.isna(dur)

            or dur <= 0

            or pd.isna(spkts)

            or pd.isna(dpkts)

        ):



            record["rate"] = 0.0



        else:



            record["rate"] = max(

                0.0,

                (

                    float(spkts)

                    + float(dpkts)

                    - 1

                ) / float(dur)

            )



        actual = clean_attack_category(

            record["attack_cat"],

            record["label"]

        )



        st.subheader(

            "Selected Traffic Record"

        )



        info1, info2, info3 = st.columns(3)



        info1.metric(

            "Protocol",

            str(record["proto"])

        )



        info2.metric(

            "Service",

            str(record["service"])

        )



        info3.metric(

            "State",

            str(record["state"])

        )



        result = make_prediction(

            record

        )



        display_result(

            result,

            actual

        )





# ============================================================
# OPTION 2
# VERIFIED 151-RECORD SUBSET
# ============================================================

elif option == "Verified 151 Records":

    st.header("✅ Verified 151-Record Evaluation Subset")

    verified_df = load_verified_data()
    total = len(verified_df)

    st.write(f"Total Records: **{total}**")
    st.info(f"Enter a record number between 1 and {total}.")

    record_text = st.text_input(
        "Enter Verified Record Number",
        placeholder=f"1 - {total}",
        key="verified_record_text"
    )

    predict_button = st.button(
        "🔍 Predict Verified Record",
        type="primary",
        key="verified_predict_button"
    )

    if predict_button:
        if record_text.strip() == "":
            st.error("❌ Please enter a record number.")
        else:
            try:
                record_number = int(record_text.strip())
            except ValueError:
                st.error("❌ Invalid Record Number")
                st.warning(f"Please enter a whole number between 1 and {total}.")
            else:
                if record_number < 1 or record_number > total:
                    st.error("❌ Invalid Record Number")
                    st.warning(f"The verified dataset contains only {total} records.")
                    st.info(f"Please enter a record number between 1 and {total}.")
                else:
                    record = verified_df.iloc[record_number - 1].copy()
                    actual = clean_attack_category(record["Actual_Category"])

                    st.success(f"✅ Record {record_number} selected.")
                    st.subheader("Selected Traffic Record")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Protocol", str(record["proto"]))
                    col2.metric("Service", str(record["service"]))
                    col3.metric("State", str(record["state"]))

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Duration", str(record["dur"]))
                    col2.metric("Rate", f"{float(record['rate']):.4f}")
                    col3.metric("Actual Attack", actual)

                    with st.spinner("Analyzing network traffic..."):
                        result = make_prediction(record)

                    display_result(result, actual)


# ============================================================

# OPTION 3

# ============================================================



elif option == "Enter Own Network Traffic":



    st.header(

        "⌨️ Enter Own Unknown Network Traffic"

    )



    st.info(

        "You do not need to know or enter "

        "the attack label. The trained model "

        "will make the prediction."

    )



    with st.form(

        "custom_traffic_form"

    ):



        st.subheader(

            "Connection Information"

        )



        c1, c2, c3 = st.columns(3)



        proto = c1.text_input(

            "Protocol",

            "tcp"

        )



        service = c2.text_input(

            "Service",

            "http"

        )



        state = c3.text_input(

            "State",

            "FIN"

        )





        st.subheader(

            "Basic Traffic"

        )



        c1, c2, c3 = st.columns(3)



        dur = c1.number_input(

            "Duration",

            min_value=0.0,

            value=0.3

        )



        spkts = c2.number_input(

            "Source Packets",

            min_value=0.0,

            value=6.0

        )



        dpkts = c3.number_input(

            "Destination Packets",

            min_value=0.0,

            value=7.0

        )



        c1, c2 = st.columns(2)



        sbytes = c1.number_input(

            "Source Bytes",

            min_value=0.0,

            value=8.0

        )



        dbytes = c2.number_input(

            "Destination Bytes",

            min_value=0.0,

            value=7.0

        )





        with st.expander(

            "Advanced Network Features",

            expanded=True

        ):



            sttl = st.number_input(

                "Source TTL",

                min_value=0.0,

                value=9.0

            )



            dttl = st.number_input(

                "Destination TTL",

                min_value=0.0,

                value=9.0

            )



            sloss = st.number_input(

                "Source Loss",

                min_value=0.0,

                value=0.0

            )



            dloss = st.number_input(

                "Destination Loss",

                min_value=0.0,

                value=0.0

            )



            sload = st.number_input(

                "Source Load",

                min_value=0.0,

                value=0.0

            )



            dload = st.number_input(

                "Destination Load",

                min_value=0.0,

                value=0.0

            )



            sinpkt = st.number_input(

                "Source Inter-Packet Time",

                min_value=0.0,

                value=0.0

            )



            dinpkt = st.number_input(

                "Destination Inter-Packet Time",

                min_value=0.0,

                value=0.0

            )



            sjit = st.number_input(

                "Source Jitter",

                min_value=0.0,

                value=0.0

            )



            djit = st.number_input(

                "Destination Jitter",

                min_value=0.0,

                value=0.0

            )



            swin = st.number_input(

                "Source TCP Window",

                min_value=0.0,

                value=0.0

            )



            dwin = st.number_input(

                "Destination TCP Window",

                min_value=0.0,

                value=0.0

            )



            stcpb = st.number_input(

                "Source TCP Base",

                min_value=0.0,

                value=0.0

            )



            dtcpb = st.number_input(

                "Destination TCP Base",

                min_value=0.0,

                value=0.0

            )



            tcprtt = st.number_input(

                "TCP RTT",

                min_value=0.0,

                value=0.0

            )



            synack = st.number_input(

                "SYN-ACK Time",

                min_value=0.0,

                value=0.0

            )



            ackdat = st.number_input(

                "ACK Data Time",

                min_value=0.0,

                value=0.0

            )



            smean = st.number_input(

                "Mean Source Packet Size",

                min_value=0.0,

                value=0.0

            )



            dmean = st.number_input(

                "Mean Destination Packet Size",

                min_value=0.0,

                value=0.0

            )



            trans_depth = st.number_input(

                "Transaction Depth",

                min_value=0.0,

                value=0.0

            )



            response_body_len = st.number_input(

                "Response Body Length",

                min_value=0.0,

                value=0.0

            )



            is_ftp_login = st.selectbox(

                "FTP Login",

                [0, 1]

            )



            ct_ftp_cmd = st.number_input(

                "FTP Command Count",

                min_value=0.0,

                value=0.0

            )



            ct_flw_http_mthd = st.number_input(

                "HTTP Method Count",

                min_value=0.0,

                value=0.0

            )



            ct_srv_src = st.number_input(

                "ct_srv_src",

                min_value=0.0,

                value=0.0

            )



            ct_state_ttl = st.number_input(

                "ct_state_ttl",

                min_value=0.0,

                value=0.0

            )



            ct_dst_ltm = st.number_input(

                "ct_dst_ltm",

                min_value=0.0,

                value=0.0

            )



            ct_src_dport_ltm = st.number_input(

                "ct_src_dport_ltm",

                min_value=0.0,

                value=0.0

            )



            ct_dst_sport_ltm = st.number_input(

                "ct_dst_sport_ltm",

                min_value=0.0,

                value=0.0

            )



            ct_dst_src_ltm = st.number_input(

                "ct_dst_src_ltm",

                min_value=0.0,

                value=0.0

            )



            ct_src_ltm = st.number_input(

                "ct_src_ltm",

                min_value=0.0,

                value=0.0

            )



            ct_srv_dst = st.number_input(

                "ct_srv_dst",

                min_value=0.0,

                value=0.0

            )



            is_sm_ips_ports = st.selectbox(

                "Same IP/Port Flag",

                [0, 1]

            )





        submitted = st.form_submit_button(

            "🛡️ Analyze Traffic",

            type="primary"

        )





    if submitted:



        if dur > 0:



            rate = max(

                0.0,

                (

                    spkts

                    + dpkts

                    - 1

                ) / dur

            )



        else:



            rate = 0.0





        custom_record = {

            "proto": proto,

            "service": service,

            "state": state,



            "dur": dur,

            "spkts": spkts,

            "dpkts": dpkts,

            "sbytes": sbytes,

            "dbytes": dbytes,



            "rate": rate,



            "sttl": sttl,

            "dttl": dttl,

            "sloss": sloss,

            "dloss": dloss,



            "sload": sload,

            "dload": dload,



            "sinpkt": sinpkt,

            "dinpkt": dinpkt,



            "sjit": sjit,

            "djit": djit,



            "swin": swin,

            "dwin": dwin,



            "stcpb": stcpb,

            "dtcpb": dtcpb,



            "tcprtt": tcprtt,

            "synack": synack,

            "ackdat": ackdat,



            "smean": smean,

            "dmean": dmean,



            "trans_depth": trans_depth,

            "response_body_len":

                response_body_len,



            "is_ftp_login":

                is_ftp_login,



            "ct_ftp_cmd":

                ct_ftp_cmd,



            "ct_flw_http_mthd":

                ct_flw_http_mthd,



            "ct_srv_src":

                ct_srv_src,



            "ct_state_ttl":

                ct_state_ttl,



            "ct_dst_ltm":

                ct_dst_ltm,



            "ct_src_dport_ltm":

                ct_src_dport_ltm,



            "ct_dst_sport_ltm":

                ct_dst_sport_ltm,



            "ct_dst_src_ltm":

                ct_dst_src_ltm,



            "ct_src_ltm":

                ct_src_ltm,



            "ct_srv_dst":

                ct_srv_dst,



            "is_sm_ips_ports":

                is_sm_ips_ports

        }





        # Fill any other training feature

        # not explicitly entered above.



        for feature in feature_names:



            if feature not in custom_record:



                if feature in [

                    "proto",

                    "service",

                    "state"

                ]:

                    custom_record[

                        feature

                    ] = "-"

                else:

                    custom_record[

                        feature

                    ] = 0.0





        custom_record = pd.Series(

            custom_record

        )





        st.write(

            f"**Automatically calculated rate:** "

            f"{rate:.6f}"

        )





        result = make_prediction(

            custom_record

        )





        display_result(

            result,

            actual=None

        )
