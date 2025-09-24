import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import openai
import os

#  Set API key from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

# 
# APP CONFIG
st.set_page_config(page_title="Health Risk Predictor + AI Chatbot", layout="wide")

st.title("🩺 Health Risk Prediction Dashboard with 🤖 AI Assistant")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Predict Health Risks", "💬 Chat with AI"])


# PAGE 1: CSV Upload + Prediction

if page == "📊 Predict Health Risks":
    st.header("📁 Upload Your CSV Files")

    # Upload multiple CSVs
    uploaded_files = st.file_uploader("Upload up to 3 CSV files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"📄 Preview of {file.name}")
            df = pd.read_csv(file)
            st.dataframe(df.head())

            # Attempt automatic model training and prediction
            try:
                # Drop non-numeric columns
                X = df.select_dtypes(include=np.number)

                # Check if 'cardio' or 'target' exists, else create dummy labels
                if 'cardio' in df.columns:
                    y = df['cardio']
                elif 'target' in df.columns:
                    y = df['target']
                else:
                    y = np.random.randint(0, 2, size=len(df))  # Dummy labels

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a basic model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                st.success(f"✅ Model trained on `{file.name}` with accuracy: {acc:.2f}")

                # Show prediction distribution
                fig = px.bar(x=['0', '1'], y=[(y_pred == 0).sum(), (y_pred == 1).sum()],
                             labels={'x': 'Predicted Class', 'y': 'Count'},
                             title='Prediction Distribution')
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Could not train model on `{file.name}`: {e}")


# PAGE 2: AI Chatbot

elif page == "💬 Chat with AI":
    st.header("💡 Ask the AI Anything!")
    st.write("You can ask about health, fitness, predictions, or any general topic.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # User input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call OpenAI GPT model
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI health assistant."},
                    *[{"role": role, "content": msg} for role, msg in st.session_state.chat_history]
                ]
            )

            ai_reply = response["choices"][0]["message"]["content"]
            st.session_state.chat_history.append(("assistant", ai_reply))

            with st.chat_message("assistant"):
                st.markdown(ai_reply)

        except Exception as e:
            st.error(f"⚠️ OpenAI error: {e}")

