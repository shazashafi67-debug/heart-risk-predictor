import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from openai import OpenAI

# ------------------------------
# OpenAI Setup
# ------------------------------
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="Health & Lifestyle App", layout="wide")
st.title("🩺 AI-Based Health Risk Prediction with Gradient Boosting + Chatbot")

# ------------------------------
# Load Gradient Boosting Model (trained in Colab)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")  # <-- export this from Colab
model = load_model()

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "🏃 Lifestyle Risk Prediction",
    "📋 Model Analysis",
    "📈 Weekly Progress",
    "💬 Chat with AI"
])

# ------------------------------
# PAGE 1: Lifestyle Risk Prediction
# ------------------------------
if page == "🏃 Lifestyle Risk Prediction":
    st.header("Enter Lifestyle Data for Heart Risk Prediction")

    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    if st.button("Predict Heart Risk"):
        # Same mapping as training
        smoke_map = {"Never":0,"Used to":1,"Occasionally":2,"Regularly":3}
        alco_map = {"Never":0,"Occasionally":1,"Regularly":2}
        active_map = {"Sedentary":3,"Lightly Active":2,"Moderately Active":1,"Very Active":0}

        input_data = pd.DataFrame([[
            steps_per_day,
            sedentary_hours,
            smoke_map[smoke],
            alco_map[alco],
            active_map[active]
        ]], columns=["steps", "sedentary_hours", "smoke", "alco", "active"])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.subheader(f"⚠️ High Risk (Probability: {proba:.2f})")
        else:
            st.subheader(f"✅ Low Risk (Probability: {proba:.2f})")

# ------------------------------
# PAGE 2: Model Analysis
# ------------------------------
elif page == "📋 Model Analysis":
    st.header("🔍 Gradient Boosting Model Feature Importance")
    try:
        fi = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(fi.set_index("Feature"))
    except Exception as e:
        st.error(f"Error showing feature importances: {e}")

# ------------------------------
# PAGE 3: Weekly Progress
# ------------------------------
elif page == "📈 Weekly Progress":
    st.header("📅 Weekly Goal & Progress Tracker")
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    steps = [st.number_input(f"Steps {day}", 0, 50000, 0, key=f"steps_{day}") for day in days]
    sedentary = [st.number_input(f"Sedentary hours {day}", 0, 24, 0, key=f"sed_{day}") for day in days]
    if st.button("Show Weekly Progress"):
        df_progress = pd.DataFrame({'Day': days, 'Steps': steps, 'Sedentary Hours': sedentary})
        st.dataframe(df_progress)
        fig = px.bar(df_progress, x='Day', y='Steps', title="Weekly Steps Progress", text='Steps')
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# PAGE 4: Chatbot
# ------------------------------
elif page == "💬 Chat with AI":
    st.header("💬 Ask Health & Lifestyle Questions")
    if client is None:
        st.warning("⚠️ OpenAI API key not found in Streamlit secrets.")
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
                    model="gpt-4o-mini",  # <-- better than gpt-3.5
                    messages=[{"role":"system","content":"You are a helpful health & lifestyle assistant."}] +
                             [{"role":r,"content":m} for r,m in st.session_state.chat_history]
                )
                ai_reply = response.choices[0].message.content
                st.session_state.chat_history.append(("assistant", ai_reply))
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
            except Exception as e:
                st.error(f"⚠️ OpenAI error: {e}")



