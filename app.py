
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================
# LOAD MODEL COMPONENTS
# ==========================================

model = joblib.load("predictive_maintenance_rf.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
best_threshold = joblib.load("threshold.pkl")
numeric_features = joblib.load("numeric_features.pkl")


# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0b1726 50%,
        #07111f 100%
    );
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    font-size: 46px !important;
    font-weight: 800 !important;
}

.subtitle {
    font-size: 18px;
    opacity: 0.7;
    margin-bottom: 25px;
}

.custom-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.small-label {
    font-size: 12px;
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    padding: 18px;
    border-radius: 16px;
}

div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.title("⚙️ MachineGuard AI")

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Predictive Maintenance • Predict. Prevent. Maintain.'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyse machine operating conditions and estimate failure risk "
    "before critical problems occur."
)

st.markdown("---")


# ==========================================
# PAGE LAYOUT
# ==========================================

left_col, right_col = st.columns(
    [1, 1.35],
    gap="large"
)


# ==========================================
# LEFT SIDE - INPUTS
# ==========================================

with left_col:

    st.markdown("""
    <div class="custom-card">
        <div class="small-label">Machine Configuration</div>
        <h3>Operating Parameters</h3>
        <p style="opacity:0.7;">
            Enter the current machine sensor readings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=4000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    analyse_button = st.button(
        "⚡ Analyse Machine"
    )


# ==========================================
# RIGHT SIDE - MODEL RESULT
# ==========================================

with right_col:

    st.markdown("""
    <div class="custom-card">
        <div class="small-label">AI Analysis</div>
        <h3>Machine Health Assessment</h3>
        <p style="opacity:0.7;">
            The trained Random Forest model evaluates the
            operating conditions and estimates failure risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if analyse_button:

        # ----------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------

        temp_diff = process_temp - air_temp

        power = (
            torque
            * rotational_speed
            * 2
            * np.pi
            / 60
        )

        tool_stress = (
            tool_wear
            * torque
        )


        # ----------------------------------
        # NUMERIC FEATURES
        # ----------------------------------

        numeric_input = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Temperature difference [K]": temp_diff,
            "Power [W]": power,
            "Tool stress": tool_stress
        }])

        numeric_input = numeric_input[numeric_features]

        numeric_scaled = scaler.transform(
            numeric_input
        )


        # ----------------------------------
        # TYPE ENCODING
        # ----------------------------------

        type_input = pd.DataFrame({
            "Type": [machine_type]
        })

        type_encoded = encoder.transform(
            type_input
        )

        if hasattr(type_encoded, "toarray"):
            type_encoded = type_encoded.toarray()


        # ----------------------------------
        # COMBINE MODEL INPUTS
        # ----------------------------------

        final_input = np.hstack([
            numeric_scaled,
            type_encoded
        ])


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        failure_probability = model.predict_proba(
            final_input
        )[:, 1][0]

        prediction = int(
            failure_probability >= best_threshold
        )

        risk_percent = (
            failure_probability * 100
        )


        # ----------------------------------
        # RESULT
        # ----------------------------------

        if prediction == 1:
            st.error(
                "⚠️ POTENTIAL MACHINE FAILURE DETECTED"
            )

            st.write(
                "The machine should be inspected. "
                "Preventive maintenance may be required."
            )

        else:
            st.success(
                "✅ MACHINE OPERATING NORMALLY"
            )

            st.write(
                "Current operating conditions do not "
                "indicate a high failure risk."
            )


        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Failure Risk",
                f"{risk_percent:.1f}%"
            )

        with metric2:
            st.metric(
                "Decision Threshold",
                f"{best_threshold * 100:.0f}%"
            )


        st.progress(
            min(
                float(failure_probability),
                1.0
            )
        )


        # ----------------------------------
        # ENGINEERED FEATURE INSIGHTS
        # ----------------------------------

        st.markdown("### ⚙️ Operational Insights")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Temperature Difference",
                f"{temp_diff:.2f} K"
            )

        with c2:
            st.metric(
                "Mechanical Power",
                f"{power / 1000:.2f} kW"
            )

        with c3:
            st.metric(
                "Tool Stress",
                f"{tool_stress:.0f}"
            )


        st.markdown("### 🧠 Model Information")

        st.write(
            "The final model uses a tuned Random Forest classifier "
            "with a 45% decision threshold."
        )

        st.write(
            "Key predictive signals include rotational speed, torque, "
            "mechanical power, tool wear and tool stress."
        )


    else:

        st.info(
            "👈 Enter the machine operating parameters and click "
            "'Analyse Machine' to generate an AI health assessment."
        )
