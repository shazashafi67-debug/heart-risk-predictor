import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from huggingface_hub import InferenceClient

# -------------------------------
# HuggingFace Client (Free model)
# -------------------------------
client = InferenceClient(repo_id="google/flan-t5-small")  # public free model

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Lifestyle & Heart Risk Predictor + AI", layout="wide")
st.title("🩺 Lifestyle & Heart Risk Predictor + 🤖 AI Assistant")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", [
    "🏃 Manual Lifestyle Input",
    "📊 CSV Upload Predictions",
    "💬 Chat with AI"
])

# -------------------------------
# PAGE 1: Manual Lifestyle Input
# -------------------------------
if page == "🏃 Manual Lifestyle Input":
    st.header("Enter your lifestyle & health data for Heart Risk Prediction")

    # Numeric inputs
    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)

    # Lifestyle categorical inputs
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    # Convert categorical to numeric
    sex = 0 if sex == "Male" else 1
    smoke_map = {"Never": 0, "Used to": 1, "Occasionally": 2, "Regularly": 3}
    alco_map = {"Never": 0, "Occasionally": 1, "Regularly": 2}
    active_map = {"Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3}
    smoke = smoke_map[smoke]
    alco = alco_map[alco]
    active = active_map[active]

    if st.button("Predict Heart Risk"):
        input_data = pd.DataFrame({
            'age':[age],
            'sex':[sex],
            'resting_bp':[resting_bp],
            'cholesterol':[cholesterol],
            'max_hr':[max_hr],
            'steps_per_day':[steps_per_day],
            'sedentary_hours':[sedentary_hours],
            'smoke':[smoke],
            'alco':[alco],
            'active':[active]
        })

        # Dummy model for demonstration
        X_demo = input_data.copy()
        y_demo = [1]  # assume high risk
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_demo, y_demo)
        pred = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write("High Risk ⚠️" if pred==1 else "Low Risk ✅")

        # Tips
        st.subheader("Lifestyle Tips")
        if pred == 1:
            st.markdown("""
            - Increase physical activity: aim for at least 10k steps/day  
            - Reduce sedentary hours: take breaks every hour  
            - Eat a balanced diet low in saturated fat and sugar  
            - Avoid smoking & limit alcohol intake  
            - Regular check-ups with your doctor
            """)
        else:
            st.markdown("Keep maintaining your healthy lifestyle! 💪")

# -------------------------------
# PAGE 2: CSV Upload Predictions
# -------------------------------
elif page == "📊 CSV Upload Predictions":
    st.header("Upload CSV files for batch heart risk predictions")
    uploaded_files = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Preview: {file.name}")
            df = pd.read_csv(file)
            st.dataframe(df.head())

            try:
                # Take only numeric columns for prediction
                X = df.select_dtypes(include=np.number)
                y_dummy = np.random.randint(0,2,len(df))  # dummy target
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y_dummy)
                df['Prediction'] = model.predict(X)

                st.subheader("Prediction Distribution")
                fig = px.bar(
                    x=['Low Risk','High Risk'],
                    y=[(df['Prediction']==0).sum(), (df['Prediction']==1).sum()],
                    labels={'x':'Risk Category','y':'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Could not predict for `{file.name}`: {e}")

# -------------------------------
# PAGE 3: Chatbot
# -------------------------------
elif page == "💬 Chat with AI":
    st.header("💬 Ask the AI any health or lifestyle questions!")

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
            response = client.text_generation(
                user_input,
                max_new_tokens=200
            )
            ai_reply = response.generated_text
            st.session_state.chat_history.append(("assistant", ai_reply))
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
        except Exception as e:
            st.error(f"⚠️ HuggingFace error: {e}")

