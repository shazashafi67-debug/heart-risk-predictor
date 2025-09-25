import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import datetime
# Optional: OpenAI for chatbot
from openai import OpenAI
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Lifestyle & Heart Risk Predictor + AI", layout="wide")
st.title("🩺 Lifestyle & Heart Risk Predictor + 🤖 AI Assistant")
st.sidebar.title("Navigation")

# -------------------------------
# Sidebar: Upload PKL Models
# -------------------------------
st.sidebar.header("Upload Models")
heart_model_file = st.sidebar.file_uploader("Upload Heart Model (.pkl)", type="pkl")
cardio_model_file = st.sidebar.file_uploader("Upload Cardio Model (.pkl)", type="pkl")
activity_model_file = st.sidebar.file_uploader("Upload Activity Model (.pkl)", type="pkl")

models = {}
if heart_model_file:
    models['Heart'] = joblib.load(heart_model_file)
if cardio_model_file:
    models['Cardio'] = joblib.load(cardio_model_file)
if activity_model_file:
    models['Activity'] = joblib.load(activity_model_file)

# -------------------------------
# Navigation Tabs
# -------------------------------
page = st.sidebar.radio("Go to:", [
    "🏃 Manual Input & Prediction",
    "📊 CSV Upload Batch Predictions",
    "📅 Goals & Weekly Tracker",
    "💬 Chatbot AI"
])

# -------------------------------
# TAB 1: Manual Input & Prediction
# -------------------------------
if page == "🏃 Manual Input & Prediction":
    st.header("Manual Lifestyle Input")
    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    steps = st.number_input("Steps Per Day", 0, 50000, 8000)

    sex = st.selectbox("Sex", ["Male", "Female"])
    active_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])

    sex_num = 0 if sex == "Male" else 1
    active_map = {"Sedentary":0, "Lightly Active":1, "Moderately Active":2, "Very Active":3}
    smoke_map = {"Never":0, "Used to":1, "Occasionally":2, "Regularly":3}

    input_df = pd.DataFrame({
        'age':[age],
        'sex':[sex_num],
        'resting_bp':[resting_bp],
        'cholesterol':[cholesterol],
        'max_hr':[max_hr],
        'steps':[steps],
        'active':[active_map[active_level]],
        'smoke':[smoke_map[smoke]]
    })

    if st.button("Predict"):
        if not models:
            st.warning("Upload at least one PKL model in the sidebar!")
        else:
            st.subheader("Predictions:")
            for name, model in models.items():
                try:
                    pred = model.predict(input_df)[0]
                    st.write(f"{name} Model Prediction: {pred}")
                    if pred == 1:
                        st.markdown("- ⚠️ High Risk: Consider improving lifestyle")
                    else:
                        st.markdown("- ✅ Low Risk: Keep healthy habits")
                except Exception as e:
                    st.error(f"Error with {name} model: {e}")

# -------------------------------
# TAB 2: CSV Upload Batch Predictions
# -------------------------------
elif page == "📊 CSV Upload Batch Predictions":
    st.header("Upload CSV for Batch Predictions")
    uploaded_csvs = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded_csvs:
        for file in uploaded_csvs:
            st.subheader(f"Preview: {file.name}")
            df = pd.read_csv(file)
            st.dataframe(df.head())

            for name, model in models.items():
                try:
                    X = df.select_dtypes(include=np.number)
                    df[f'Prediction_{name}'] = model.predict(X)
                    st.write(f"Predictions by {name} Model:")
                    st.dataframe(df.head())
                    # Plot distribution
                    fig = px.bar(
                        x=['Low Risk','High Risk'],
                        y=[(df[f'Prediction_{name}']==0).sum(), (df[f'Prediction_{name}']==1).sum()],
                        labels={'x':'Risk Category','y':'Count'},
                        title=f"{name} Model Prediction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Cannot predict {name} on `{file.name}`: {e}")

# -------------------------------
# TAB 3: Goals & Weekly Tracker
# -------------------------------
elif page == "📅 Goals & Weekly Tracker":
    st.header("Set Goals and Track Your Weekly Progress")
    goal_steps = st.number_input("Weekly Steps Goal", 0, 100000, 35000)
    goal_exercise = st.number_input("Weekly Exercise Goal (minutes)", 0, 1000, 150)

    # Session state to store weekly progress
    if "weekly_log" not in st.session_state:
        st.session_state.weekly_log = []

    with st.form("log_form"):
        log_date = st.date_input("Date", datetime.date.today())
        steps_done = st.number_input("Steps Today", 0, 50000, 8000)
        exercise_done = st.number_input("Exercise Minutes Today", 0, 180, 30)
        submitted = st.form_submit_button("Log Progress")
        if submitted:
            st.session_state.weekly_log.append({
                "date": log_date,
                "steps": steps_done,
                "exercise": exercise_done
            })
            st.success("Logged!")

    if st.session_state.weekly_log:
        df_log = pd.DataFrame(st.session_state.weekly_log)
        st.subheader("Weekly Progress")
        st.dataframe(df_log)
        # Plot calendar-style summary
        fig = px.bar(df_log, x='date', y=['steps','exercise'], barmode='group', title="Weekly Steps & Exercise")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4: Chatbot AI
# -------------------------------
elif page == "💬 Chatbot AI":
    st.header("💬 Ask Health & Lifestyle Questions!")
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

        # Only if OpenAI key provided in secrets
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful health & lifestyle AI."},
                    *[{"role": role, "content": msg} for role, msg in st.session_state.chat_history]
                ]
            )
            ai_reply = response.choices[0].message.content
            st.session_state.chat_history.append(("assistant", ai_reply))
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")








