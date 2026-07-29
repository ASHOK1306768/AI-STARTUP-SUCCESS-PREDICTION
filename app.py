import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    model = joblib.load("model.pkl")
    meta = joblib.load("feature_meta.pkl")
    return scaler, encoder, model, meta

scaler, encoder, model, meta = load_artifacts()

st.title("🚀 AI Startup Success Prediction")
st.write("Estimate the likelihood a startup will remain operating / get acquired / IPO, "
         "versus shutting down, based on funding and company profile.")

col1, col2 = st.columns(2)

with col1:
    funding_total_usd = st.number_input(
        "Total funding raised (USD)", min_value=0, value=1_000_000, step=10_000
    )
    funding_rounds = st.number_input(
        "Number of funding rounds", min_value=1, max_value=20, value=1
    )
    founded_year = st.number_input(
        "Year founded", min_value=1950, max_value=2017, value=2012
    )
    funding_gap_years = st.number_input(
        "Years between first and last funding round", min_value=0.0, max_value=30.0, value=1.0, step=0.1
    )

with col2:
    country_code = st.selectbox("Country", meta["country_options"])
    primary_category = st.selectbox("Primary category", meta["category_options"])

if st.button("Predict success", type="primary"):
    X_num = np.array([[funding_total_usd, funding_rounds, founded_year, funding_gap_years]])
    X_num_scaled = scaler.transform(X_num)

    X_cat = np.array([[country_code, primary_category]])
    X_cat_encoded = encoder.transform(X_cat)

    X = np.hstack([X_num_scaled, X_cat_encoded])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.subheader("Result")
    if pred == 1:
        st.success(f"Predicted outcome: **Likely to succeed** (operating / acquired / IPO)")
    else:
        st.error(f"Predicted outcome: **At risk of shutting down**")

    st.metric("Estimated probability of success", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))

st.caption(
    "Model: Random Forest trained on historical Crunchbase-style startup data. "
    "Predictions are statistical estimates, not guarantees — use as one input among many."
)
