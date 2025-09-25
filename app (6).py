import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import os
from openai import OpenAI

# -------------------------------
# OpenAI Setup
# -------------------------------
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Health & Lifestyle App", layout="wide")
st.title("🩺 Health & Lifestyle Predictor + AI Assistant")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", [
    "🏃 Lifestyle Risk Prediction",
    "📊 Upload CSV for Batch Predictions",
    "📈 Weekly Progress",
    "📋 Analyze PKL Models",
    "💬 Chat with AI"
])

# -------------------------------
# Upload PKL Models
# -------------------------------
st.sidebar.subheader("Upload PKL Models (Batch Prediction / Analysis Only)")
heart_model_file = st.sidebar.file_uploader("Heart Risk Model", type="pkl")
cardio_model_file = st.sidebar.file_uploader("Cardio Risk Model", type="pkl")
activity_model_file = st.sidebar.file_uploader("Activity Model", type="pkl")

models_loaded = False
if heart_model_file and cardio_model_file and activity_model_file:
    heart_model = joblib.load(heart_model_file)
    cardio_model = joblib.load(cardio_model_file)
    activity_model = joblib.load(activity_model_file)
    models_loaded = True
    st.sidebar.success("✅ Models loaded successfully!")

# -------------------------------
# PAGE 1: Lifestyle Risk Prediction (Manual)
# -------------------------------
if page == "🏃 Lifestyle Risk Prediction":
    st.header("Enter Lifestyle Data for Heart Risk Prediction")
    
    # Lifestyle inputs
    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    if st.button("Predict Heart Risk"):
        # Rules-based scoring
        risk_score = 0
        if steps_per_day < 5000:
            risk_score += 2
        elif steps_per_day < 10000:
            risk_score += 1
        if sedentary_hours > 10:
            risk_score += 2
        elif sedentary_hours > 6:
            risk_score += 1
        smoke_map = {"Never":0,"Used to":1,"Occasionally":2,"Regularly":3}
        risk_score += smoke_map[smoke]
        alco_map = {"Never":0,"Occasionally":1,"Regularly":2}
        risk_score += alco_map[alco]
        active_map = {"Sedentary":3,"Lightly Active":2,"Moderately Active":1,"Very Active":0}
        risk_score += active_map[active]

        if risk_score >=6:
            st.subheader("Prediction Result: High Risk ⚠️")
            st.markdown("""
            **Tips:**  
            - Increase steps, reduce sedentary time  
            - Exercise regularly  
            - Avoid smoking & limit alcohol  
            - Balanced diet
            """)
        elif risk_score >=3:
            st.subheader("Prediction Result: Moderate Risk ⚠️")
            st.markdown("Improve activity and reduce sedentary time.")
        else:
            st.subheader("Prediction Result: Low Risk ✅")
            st.markdown("Great! Keep it up 💪")

# -------------------------------
# PAGE 2: CSV Upload (Batch PKL Prediction)
# -------------------------------
elif page == "📊 Upload CSV for Batch Predictions":
    st.header("Upload CSV for batch predictions")
    if not models_loaded:
        st.warning("Upload all PKL models first!")
    else:
        uploaded_file = st.file_uploader("CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            try:
                X = df.select_dtypes(include=np.number)
                df['Heart_Risk'] = heart_model.predict(X)
                df['Cardio_Risk'] = cardio_model.predict(X)
                df['Activity_Risk'] = activity_model.predict(X)
                st.subheader("Prediction Distribution")
                fig = px.bar(
                    x=['Low Heart','High Heart','Low Cardio','High Cardio','Good Activity','Low Activity'],
                    y=[
                        (df['Heart_Risk']==0).sum(), (df['Heart_Risk']==1).sum(),
                        (df['Cardio_Risk']==0).sum(), (df['Cardio_Risk']==1).sum(),
                        (df['Activity_Risk']==1).sum(), (df['Activity_Risk']==0).sum()
                    ],
                    labels={'x':'Category','y':'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Could not predict: {e}")

# -------------------------------
# PAGE 3: Weekly Progress
# -------------------------------
elif page == "📈 Weekly Progress":
    st.header("📅 Weekly Goal & Progress Tracker")
    st.info("Enter your steps and sedentary hours each day to track weekly progress.")
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    steps = [st.number_input(f"Steps {day}", 0, 50000, 0, key=f"steps_{day}") for day in days]
    sedentary = [st.number_input(f"Sedentary hours {day}", 0, 24, 0, key=f"sed_{day}") for day in days]
    if st.button("Show Weekly Progress"):
        df_progress = pd.DataFrame({'Day': days, 'Steps': steps, 'Sedentary Hours': sedentary})
        st.dataframe(df_progress)
        fig = px.bar(df_progress, x='Day', y='Steps', title="Weekly Steps Progress", text='Steps')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PAGE 4: Analyze PKL Models
# -------------------------------
elif page == "📋 Analyze PKL Models":
    st.header("🔍 Analyze Uploaded PKL Models")
    if not models_loaded:
        st.warning("Upload all PKL models first!")
    else:
        st.subheader("Feature Importance (Heart Model)")
        heart_imp = pd.DataFrame({
            "Feature": heart_model.feature_names_in_,
            "Importance": heart_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(heart_imp)
        st.subheader("Feature Importance (Cardio Model)")
        cardio_imp = pd.DataFrame({
            "Feature": cardio_model.feature_names_in_,
            "Importance": cardio_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(cardio_imp)
        st.subheader("Feature Importance (Activity Model)")
        activity_imp = pd.DataFrame({
            "Feature": activity_model.feature_names_in_,
            "Importance": activity_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(activity_imp)

# -------------------------------
# PAGE 5: Chatbot
# -------------------------------
elif page == "💬 Chat with AI":
    st.header("💬 Ask Health & Lifestyle Questions")
    if client is None:
        st.warning("OpenAI API key not found in Streamlit secrets.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        user_input = st.chat_input("Type your question...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":"You are a helpful health & lifestyle assistant."}] +
                             [{"role":r,"content":m} for r,m in st.session_state.chat_history]
                )
                ai_reply = response.choices[0].message.content
                st.session_state.chat_history.append(("assistant", ai_reply))
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
            except Exception as e:
                st.error(f"⚠️ OpenAI error: {e}")

