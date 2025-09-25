# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from datetime import datetime, timedelta

# OpenAI client (new style). Only created if secret exists.
client = None
try:
    from openai import OpenAI
    if "OPENAI_API_KEY" in st.secrets:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None

# ---------------- App config ----------------
st.set_page_config(page_title="Health Predictor + PKL models", layout="wide")
st.title("🩺 Health & Lifestyle Dashboard — PKL Model Uploads")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "🏃 Manual Lifestyle Prediction",
    "📁 Upload PKL Models (heart / cardio / activity)",
    "📅 Weekly Goal & Progress Tracker",
    "💬 AI Health Chatbot (optional)"
])

# Common manual input used for predictions
def get_manual_inputs():
    st.subheader("Manual input (used for model test predictions)")
    age = st.number_input("Age", 18, 120, 45)
    sex_sel = st.selectbox("Sex", ["Male", "Female"])
    sex = 0 if sex_sel == "Male" else 1
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
    max_hr = st.number_input("Maximum Heart Rate Achieved", 40, 220, 150)
    steps_per_day = st.number_input("Average Steps Per Day", 0, 100000, 8000)
    sedentary_hours = st.number_input("Sedentary Hours Per Day", 0.0, 24.0, 8.0, step=0.5)

    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

    smoke_map = {"Never": 0, "Used to": 1, "Occasionally": 2, "Regularly": 3}
    alco_map = {"Never": 0, "Occasionally": 1, "Regularly": 2}
    active_map = {"Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3}

    return {
        "age": age,
        "sex": sex,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "max_hr": max_hr,
        "steps_per_day": steps_per_day,
        "sedentary_hours": sedentary_hours,
        "smoke": smoke_map[smoke],
        "alco": alco_map[alco],
        "active": active_map[active]
    }

# Utility: safe model predict
def safe_predict(model, input_df):
    """Try to prepare input_df to model expected shape and predict.
    Returns (success, message or prediction)."""
    # If scikit-learn model with feature_names_in_ use it
    try:
        # if model expects certain feature names:
        feature_names = getattr(model, "feature_names_in_", None)
        n_req = getattr(model, "n_features_in_", None)
        # If feature_names available, attempt to select them from input_df
        if feature_names is not None:
            missing = [f for f in feature_names if f not in input_df.columns]
            if missing:
                return False, f"Model expects features {list(feature_names)} but your input is missing: {missing}."
            X = input_df[list(feature_names)]
        else:
            # fallback: if model specifies n_features_in_ try to match that many from input_df by order
            if n_req is not None:
                if input_df.shape[1] < n_req:
                    return False, f"Model expects {n_req} features but input has {input_df.shape[1]}."
                X = input_df.iloc[:, :n_req]
            else:
                X = input_df  # try as-is
        # Ensure correct dtype
        X = X.astype(float)
        pred = model.predict(X)
        return True, pred
    except Exception as e:
        return False, f"Prediction failed: {str(e)}"

# ---------------- Page 1: Manual Lifestyle Prediction ----------------
if page == "🏃 Manual Lifestyle Prediction":
    st.header("Manual Lifestyle Prediction")
    inputs = get_manual_inputs()
    st.write("Preview of input (used if you press Predict):")
    st.json(inputs)

    if st.button("🔍 Quick demo predict (dummy model)"):
        demo_df = pd.DataFrame([[
            inputs["age"], inputs["sex"], inputs["resting_bp"], inputs["cholesterol"],
            inputs["max_hr"], inputs["steps_per_day"], inputs["sedentary_hours"],
            inputs["smoke"], inputs["alco"], inputs["active"]
        ]], columns=['age','sex','resting_bp','cholesterol','max_hr','steps_per_day',
                     'sedentary_hours','smoke','alco','active'])
        # simple demo model (not trained) just to show result; treat threshold on steps & bp
        score = 0
        score += (demo_df['age'].iloc[0] - 40) / 40
        score += max(0, (demo_df['resting_bp'].iloc[0] - 120) / 40)
        score += max(0, (demo_df['cholesterol'].iloc[0] - 200) / 100)
        score -= min(1, demo_df['steps_per_day'].iloc[0] / 10000)
        score = float(score)
        risk = "High Risk ⚠️" if score > 0.5 else "Low Risk ✅"
        st.subheader("Result")
        st.write(risk)
        st.caption("This is a simple rule-based demo prediction. Upload your trained PKL models to use real models.")

# ---------------- Page 2: Upload PKL Models ----------------
elif page == "📁 Upload PKL Models (heart / cardio / activity)":
    st.header("Upload your trained .pkl models (joblib / sklearn)")

    st.markdown("Upload each model separately. After loading, test it with the manual input above.")
    col1, col2, col3 = st.columns(3)

    # Keep models in session_state
    if "loaded_models" not in st.session_state:
        st.session_state.loaded_models = {"heart": None, "cardio": None, "activity": None}
        st.session_state.model_names = {"heart": None, "cardio": None, "activity": None}

    with col1:
        heart_file = st.file_uploader("Upload heart_model.pkl", type=["pkl"], key="heart_uploader")
        if heart_file is not None:
            try:
                heart_model = joblib.load(io.BytesIO(heart_file.read()))
                st.session_state.loaded_models["heart"] = heart_model
                st.session_state.model_names["heart"] = heart_file.name
                st.success("Loaded heart model.")
            except Exception as e:
                st.error(f"Failed to load heart_model.pkl: {e}")

    with col2:
        cardio_file = st.file_uploader("Upload cardio_model.pkl", type=["pkl"], key="cardio_uploader")
        if cardio_file is not None:
            try:
                cardio_model = joblib.load(io.BytesIO(cardio_file.read()))
                st.session_state.loaded_models["cardio"] = cardio_model
                st.session_state.model_names["cardio"] = cardio_file.name
                st.success("Loaded cardio model.")
            except Exception as e:
                st.error(f"Failed to load cardio_model.pkl: {e}")

    with col3:
        activity_file = st.file_uploader("Upload activity_model.pkl", type=["pkl"], key="activity_uploader")
        if activity_file is not None:
            try:
                activity_model = joblib.load(io.BytesIO(activity_file.read()))
                st.session_state.loaded_models["activity"] = activity_model
                st.session_state.model_names["activity"] = activity_file.name
                st.success("Loaded activity model.")
            except Exception as e:
                st.error(f"Failed to load activity_model.pkl: {e}")

    st.markdown("---")
    st.subheader("Test loaded models with manual input")

    # Use the same manual inputs to test loaded models
    inputs = get_manual_inputs()
    test_df_full = pd.DataFrame([[
        inputs["age"], inputs["sex"], inputs["resting_bp"], inputs["cholesterol"],
        inputs["max_hr"], inputs["steps_per_day"], inputs["sedentary_hours"],
        inputs["smoke"], inputs["alco"], inputs["active"]
    ]], columns=['age','sex','resting_bp','cholesterol','max_hr','steps_per_day',
                 'sedentary_hours','smoke','alco','active'])

    # For each model, attempt prediction
    for key, model in st.session_state.loaded_models.items():
        if model is None:
            st.info(f"No `{key}` model loaded yet.")
            continue

        st.write(f"### Model: {key} ({st.session_state.model_names.get(key)})")
        # Try to prepare features
        success, result = safe_predict(model, test_df_full)
        if success:
            # result may be array
            try:
                pred = int(np.asarray(result).ravel()[0])
                st.success(f"Prediction: {pred} — {'High Risk ⚠️' if pred==1 else 'Low Risk ✅'}")
            except Exception:
                st.success(f"Prediction array: {result}")
        else:
            st.error(result)
            # Offer user ability to provide a custom CSV with exact feature names
            with st.expander("If model expects different features, upload a CSV with the exact columns"):
                csv_file = st.file_uploader(f"Upload a CSV with columns matching the `{key}` model", type=["csv"], key=f"csv_for_{key}")
                if csv_file is not None:
                    try:
                        df_for_model = pd.read_csv(csv_file)
                        if df_for_model.empty:
                            st.error("CSV is empty.")
                        else:
                            s2, r2 = safe_predict(model, df_for_model)
                            if s2:
                                df_for_model["Prediction"] = r2
                                st.dataframe(df_for_model.head())
                                st.success("Predictions done on uploaded CSV.")
                                st.download_button("Download predictions CSV", df_for_model.to_csv(index=False), file_name=f"{key}_predictions.csv")
                            else:
                                st.error(r2)
                    except Exception as e:
                        st.error(f"Could not read uploaded CSV: {e}")

# ---------------- Page 3: Weekly Goal & Progress Tracker ----------------
elif page == "📅 Weekly Goal & Progress Tracker":
    st.header("Weekly Goal & Progress Tracker")

    steps_goal = st.number_input("Daily Steps Goal", 1000, 100000, 10000)
    exercise_goal = st.number_input("Daily Exercise Minutes Goal", 0, 300, 30)
    water_goal = st.number_input("Daily Water Intake (ml)", 500, 5000, 2000)

    if "weekly_log" not in st.session_state:
        st.session_state.weekly_log = pd.DataFrame(columns=["Date", "Steps", "Exercise", "Water Intake"])

    st.subheader("Log today's progress")
    steps_done = st.number_input("Steps today", 0, 100000, 0, key="log_steps")
    exercise_done = st.number_input("Exercise minutes today", 0, 300, 0, key="log_ex")
    water_done = st.number_input("Water intake today (ml)", 0, 10000, 0, key="log_water")
    if st.button("Add today's log"):
        today = datetime.today().strftime("%Y-%m-%d")
        new_row = {"Date": today, "Steps": steps_done, "Exercise": exercise_done, "Water Intake": water_done}
        st.session_state.weekly_log = pd.concat([st.session_state.weekly_log, pd.DataFrame([new_row])], ignore_index=True)
        st.success(f"Added {today} ✔️")

    # Show last 7 days as calendar-like table + grouped bar chart
    if not st.session_state.weekly_log.empty:
        st.subheader("Last 7 days")
        last_7 = datetime.today() - timedelta(days=6)
        df7 = st.session_state.weekly_log[pd.to_datetime(st.session_state.weekly_log["Date"]) >= last_7.strftime("%Y-%m-%d")]
        if df7.empty:
            st.info("No entries in the last 7 days.")
        else:
            df7_display = df7.copy().set_index("Date")
            st.dataframe(df7_display)

            df_pct = df7.copy()
            df_pct["Steps %"] = (df_pct["Steps"] / steps_goal * 100).clip(0,100)
            df_pct["Exercise %"] = (df_pct["Exercise"] / exercise_goal * 100).clip(0,100)
            df_pct["Water %"] = (df_pct["Water Intake"] / water_goal * 100).clip(0,100)

            fig = px.bar(
                df_pct.melt(id_vars="Date", value_vars=["Steps %", "Exercise %", "Water %"]),
                x="Date", y="value", color="variable", barmode="group",
                labels={"value":"Progress %", "variable":"Metric"}
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Page 4: AI Health Chatbot ----------------
elif page == "💬 AI Health Chatbot (optional)":
    st.header("AI Health Chatbot (OpenAI)")

    if client is None:
        st.warning("OpenAI key not found in Streamlit secrets — add OPENAI_API_KEY to use the chatbot.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        user_input = st.chat_input("Ask a health/lifestyle question...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                # Use new OpenAI client API
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":"You are a helpful health and lifestyle assistant."}] +
                             [{"role": r, "content": m} for r, m in st.session_state.chat_history]
                )
                ai_reply = response.choices[0].message.content
                st.session_state.chat_history.append(("assistant", ai_reply))
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)
            except Exception as e:
                st.error(f"OpenAI error: {e}")








