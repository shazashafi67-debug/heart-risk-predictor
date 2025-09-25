import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
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
    "📋 Analyze PKL Models",
    "📈 Weekly Progress",
    "💬 Chat with AI"
])

# -------------------------------
# Upload PKL Models
# -------------------------------
st.sidebar.subheader("Upload PKL Models")
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
    
    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    if st.button("Predict Heart Risk"):
        risk_score = 0
        risk_score += 2 if steps_per_day < 5000 else 1 if steps_per_day < 10000 else 0
        risk_score += 2 if sedentary_hours > 10 else 1 if sedentary_hours > 6 else 0
        smoke_map = {"Never":0,"Used to":1,"Occasionally":2,"Regularly":3}
        risk_score += smoke_map[smoke]
        alco_map = {"Never":0,"Occasionally":1,"Regularly":2}
        risk_score += alco_map[alco]
        active_map = {"Sedentary":3,"Lightly Active":2,"Moderately Active":1,"Very Active":0}
        risk_score += active_map[active]

        if risk_score >=6:
            st.subheader("High Risk ⚠️")
            st.markdown("- Increase steps and reduce sedentary time\n- Exercise regularly\n- Avoid smoking & limit alcohol\n- Balanced diet")
        elif risk_score >=3:
            st.subheader("Moderate Risk ⚠️")
            st.markdown("Try to improve activity and reduce sedentary time.")
        else:
            st.subheader("Low Risk ✅")
            st.markdown("Great! Keep maintaining your healthy lifestyle 💪")

# -------------------------------
# PAGE 2: Analyze PKL Models
# -------------------------------
elif page == "📋 Analyze PKL Models":
    st.header("🔍 Analyze Uploaded PKL Models")
    if not models_loaded:
        st.warning("Upload all PKL models first!")
    else:
        # Function to summarize predictions
        def model_summary(model, name):
            st.subheader(f"{name} Model Summary")
            try:
                # Feature importance
                fi = pd.DataFrame({
                    "Feature": model.feature_names_in_,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.markdown("**Feature Importance:**")
                st.dataframe(fi)

                # Dummy prediction summary
                if hasattr(model, "n_estimators"):
                    st.markdown(f"**Number of Trees:** {model.n_estimators}")
                if hasattr(model, "classes_"):
                    st.markdown(f"**Target Classes:** {model.classes_}")
            except Exception as e:
                st.error(f"Error summarizing {name} model: {e}")

        model_summary(heart_model, "Heart Risk")
        model_summary(cardio_model, "Cardio Risk")
        model_summary(activity_model, "Activity")

# -------------------------------
# PAGE 3: Weekly Progress
# -------------------------------
elif page == "📈 Weekly Progress":
    st.header("📅 Weekly Goal & Progress Tracker")
    st.info("Enter steps and sedentary hours each day.")
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    steps = [st.number_input(f"Steps {day}", 0, 50000, 0, key=f"steps_{day}") for day in days]
    sedentary = [st.number_input(f"Sedentary hours {day}", 0, 24, 0, key=f"sed_{day}") for day in days]
    if st.button("Show Weekly Progress"):
        df_progress = pd.DataFrame({'Day': days, 'Steps': steps, 'Sedentary Hours': sedentary})
        st.dataframe(df_progress)
        fig = px.bar(df_progress, x='Day', y='Steps', title="Weekly Steps Progress", text='Steps')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PAGE 4: Chatbot
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


