import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Mini Project", layout="centered")

MODEL_PATHS = {
    "ICU admission": "./Models/LR_ICU_model.joblib",
    "Intubation": "./Models/LR_INT_model.joblib",
    "Death": "./Models/LR_DEATH_model.joblib",
}

GENDER_MAP = {"Female": 0, "Male": 1}
YES, NO = 1, 0


FEATURE_ORDER = [
    "Age", "Has_Pneumonia", "Has_Diabetes", "Has_COPD", "Has_Asthma",
    "Is_Immunosuppressed", "Has_Hypertension", "Has_Cardiovascular",
    "Is_Smoker", "Gender", "Is_Obese",
]

# Same cutoff that model.predict() uses for logistic regression.
THRESHOLD = 0.5


@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


try:
    models = load_models()
except FileNotFoundError as e:
    st.error(f"Could not load a model file: {e}. Run the app from the folder that contains `Models/`.")
    st.stop()


# UI
st.title("ML Mini Project: COVID-19 Risk Estimation")
st.caption(
    "Logistic regression models that estimate the risk of ICU admission. "
    "Based on the Paper: https://cs229.stanford.edu/proj2020spr/report/Zhan_Li.pdf."
    " Intubation and death from basic patient information."
)
st.caption("Team: Neel Chandhrakar, Neeraj R Rugi")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
    with col2:
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))

    st.markdown("**Conditions**")
    c1, c2, c3 = st.columns(3)
    with c1:
        pneumonia = st.checkbox("Pneumonia")
        diabetes = st.checkbox("Diabetes")
        copd = st.checkbox("COPD")
        asthma = st.checkbox("Asthma")
    with c2:
        immuno = st.checkbox("Immunosuppressed")
        hypertension = st.checkbox("Hypertension")
        cardio = st.checkbox("Cardiovascular disease")
    with c3:
        smoker = st.checkbox("Smoker")
        obese = st.checkbox("Obese")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    flag = lambda x: YES if x else NO
    row = {
        "Age": age,
        "Has_Pneumonia": flag(pneumonia),
        "Has_Diabetes": flag(diabetes),
        "Has_COPD": flag(copd),
        "Has_Asthma": flag(asthma),
        "Is_Immunosuppressed": flag(immuno),
        "Has_Hypertension": flag(hypertension),
        "Has_Cardiovascular": flag(cardio),
        "Is_Smoker": flag(smoker),
        "Gender": GENDER_MAP[gender],
        "Is_Obese": flag(obese),
    }
    X = pd.DataFrame([row])[FEATURE_ORDER]

    st.subheader("Results")
    cols = st.columns(len(models))
    for col, (name, model) in zip(cols, models.items()):
        score = float(model.predict_proba(X)[0, 1])
        flagged = score >= THRESHOLD
        with col:
            st.metric(label=f"{name} risk score", value=f"{score:.2%}")
            st.text(f"{name}")
            if flagged:
                st.error(f"Flagged: higher risk")
            else:
                st.success("Not flagged")

    with st.expander("Input sent to the models"):
        st.dataframe(X, hide_index=True)

st.divider()
