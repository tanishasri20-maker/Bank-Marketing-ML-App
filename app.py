import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# Page Configuration

st.set_page_config(
    page_title="Bank Marketing Prediction System",
    layout="centered"
)


# Navy Blue Theme Styling

st.markdown("""
<style>
.stApp { background-color: #0a1f44; }

.block-container { padding-top: 2rem; }

html, body, [class*="css"]  { 
    color: #ffffff !important; 
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #cbd5e1;
    text-align: center;
    margin-bottom: 20px;
}

.logo {
    font-size: 80px;
    text-align: center;
}

.card {
    background-color: #112b5e;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    margin-bottom: 25px;
}

/* Creative Button */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 0.8em 2.5em;
    border-radius: 40px;
    border: none;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 0 15px rgba(0,114,255,0.6);
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,198,255,0.9);
}

label { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# Load Dataset

df = pd.read_csv("data/bank.csv")
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("deposit", axis=1)
feature_columns = X.columns


# Load Best Model

results_df = pd.read_csv("model_results.csv")
best_model_name = results_df.loc[results_df["Recall"].idxmax(), "Model"]
model = joblib.load(best_model_name + ".pkl")


# Logo + Title

st.markdown("<div class='logo'>🏦</div>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Bank Marketing Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Term Deposit Investment Predictor</div>", unsafe_allow_html=True)

st.markdown("---")


# Input Section

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Customer's Information")

age = st.slider("Age", 18, 100, 30)
duration = st.slider("Call Duration (seconds)", 0, 1000, 200)
campaign = st.slider("Number of Campaign Contacts", 0, 20, 1)
euribor3m = st.slider("Interest Rate", 0.0, 6.0, 4.0)

predict_btn = st.button(" PREDICT ")
st.markdown("</div>", unsafe_allow_html=True)


# Prediction Section


# Prediction Section

if predict_btn:

    input_dict = {col: 0 for col in feature_columns}

    for col in feature_columns:
        if col.lower() == "age":
            input_dict[col] = age
        elif col.lower() == "duration":
            input_dict[col] = duration
        elif col.lower() == "campaign":
            input_dict[col] = campaign
        elif col.lower() == "euribor3m":
            input_dict[col] = euribor3m

    input_data = pd.DataFrame([input_dict])
    probability = model.predict_proba(input_data)[0][1]

    threshold = 0.30
    prediction = 1 if probability >= threshold else 0

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Results")

    st.write(f"Selected Model: {best_model_name}")
    st.write(f"Decision Threshold: {threshold}")
    st.write(f"Predicted Investment Probability: {probability*100:.2f}%")

    # Gauge Color Logic (Dark Green if Likely)
   
    if prediction == 1:
        gauge_color = "#006400"   # Dark Green
    else:
        gauge_color = "#8B0000"   # Dark Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 40], 'color': '#330000'},
                {'range': [40, 70], 'color': '#333300'},
                {'range': [70, 100], 'color': '#003300'}
            ],
        }
    ))

    fig.update_layout(
        height=350,
        paper_bgcolor="#112b5e",
        font={'color': "white"}
    )

    st.plotly_chart(fig, use_container_width=True)

   
    # Text Result
    
    if prediction == 1:
        st.success("Customer is Likely to Invest.")
    else:
        st.error("Customer is Unlikely to Invest.")

    st.markdown("</div>", unsafe_allow_html=True)
