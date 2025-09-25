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
# Make sure to add your OpenAI API key in Streamlit secrets:
# [general]
# OPENAI_API_KEY = "sk-xxxx"

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
    "🏃 Manual Lifestyle Input",
    "📊 Upload CSV for Batch Predictions",
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
# PAGE 1: Manual Input
# -------------------------------
if page == "🏃 Manual Lifestyle Input":
    st.header("Enter Lifestyle & Health Data")
    if not models_loaded:
        st.warning("Upload all 3 PKL models first!")
    else:
        # Inputs
        age = st.number_input("Age", 20, 100, 50)
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
        max_hr = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        steps_per_day = st.number_input("Steps per day", 0, 50000, 8000)
        sedentary_hours = st.number_input("Sedentary hours/day", 0, 24, 8)

        sex = st.selectbox("Sex", ["Male", "Female"])
        smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
        alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
        active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

        # Mapping
        sex = 0 if sex=="Male" else 1
        smoke_map = {"Never":0, "Used to":1, "Occasionally":2, "Regularly":3}
        alco_map = {"Never":0, "Occasionally":1, "Regularly":2}
        active_map = {"Sedentary":0, "Lightly Active":1, "Moderately Active":2, "Very Active":3}
        smoke = smoke_map[smoke]
        alco = alco_map[alco]
        active = active_map[active]

        input_data = pd.DataFrame({
            'age':[age],'sex':[sex],'resting_bp':[resting_bp],'cholesterol':[cholesterol],
            'max_hr':[max_hr],'steps_per_day':[steps_per_day],'sedentary_hours':[sedentary_hours],
            'smoke':[smoke],'alco':[alco],'active':[active]
        })

        if st.button("Predict"):
            heart_pred = heart_model.predict(input_data)[0]
            cardio_pred = cardio_model.predict(input_data)[0]
            activity_pred = activity_model.predict(input_data)[0]

            st.subheader("Predictions")
            st.write(f"❤️ Heart Risk: {'High ⚠️' if heart_pred==1 else 'Low ✅'}")
            st.write(f"💓 Cardio Risk: {'High ⚠️' if cardio_pred==1 else 'Low ✅'}")
            st.write(f"🏃 Activity Risk: {'Low ⚠️' if activity_pred==1 else 'Good ✅'}")

            # Tips
            tips = []
            if heart_pred==1:
                tips.append("- Increase physical activity (10k steps/day)")
                tips.append("- Reduce sedentary hours")
                tips.append("- Eat a balanced diet")
            if cardio_pred==1:
                tips.append("- Include cardio exercises 3-5 times/week")
            if activity_pred==0:
                tips.append("- Stay active: break long sitting periods")
            if not tips:
                tips.append("Keep up your healthy lifestyle! 💪")
            st.subheader("Lifestyle Tips")
            for t in tips:
                st.markdown(t)

# -------------------------------
# PAGE 2: CSV Upload
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
    st.info("Enter your steps and activity each day to track weekly progress.")

    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    steps = [st.number_input(f"Steps {day}", 0, 50000, 0, key=f"steps_{day}") for day in days]
    sedentary = [st.number_input(f"Sedentary hours {day}", 0, 24, 0, key=f"sed_{day}") for day in days]

    if st.button("Show Weekly Progress"):
        df_progress = pd.DataFrame({
            'Day': days,
            'Steps': steps,
            'Sedentary Hours': sedentary
        })
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


       
