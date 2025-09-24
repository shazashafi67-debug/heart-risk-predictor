import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from datetime import datetime, timedelta


# App Configuration

st.set_page_config(page_title="Lifestyle & Heart Risk Predictor", layout="wide")
st.title("🩺 Lifestyle & Heart Risk Predictor")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to:", [
    "🏃 Manual Lifestyle Input",
    "📊 CSV Upload Predictions",
    "🎯 Weekly Goal & Progress Tracker"
])


# PAGE 1: Manual Lifestyle Input

if page == "🏃 Manual Lifestyle Input":
    st.header("Enter your lifestyle & health data for Heart Risk Prediction")

    age = st.number_input("Age", 20, 100, 50)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 400, 200)
    max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    steps_per_day = st.number_input("Average Steps Per Day", 0, 50000, 8000)
    sedentary_hours = st.number_input("Hours Sedentary Per Day", 0, 24, 8)

    sex = st.selectbox("Sex", ["Male", "Female"])
    smoke = st.selectbox("Smoking Habit", ["Never", "Used to", "Occasionally", "Regularly"])
    alco = st.selectbox("Alcohol Intake", ["Never", "Occasionally", "Regularly"])
    active = st.selectbox("Exercise Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])

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

        X_demo = input_data.copy()
        y_demo = [1]  # assume high risk
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_demo, y_demo)
        pred = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.write("High Risk ⚠️" if pred==1 else "Low Risk ✅")

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


# PAGE 2: CSV Upload Predictions

elif page == "📊 CSV Upload Predictions":
    st.header("Upload CSV files for batch heart risk predictions")
    uploaded_files = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Preview: {file.name}")
            df = pd.read_csv(file)
            st.dataframe(df.head())

            try:
                X = df.select_dtypes(include=np.number)
                y_dummy = np.random.randint(0,2,len(df))
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


# PAGE 3: Weekly Goal & Progress Tracker

elif page == "🎯 Weekly Goal & Progress Tracker":
    st.header("Set your weekly health goals and log daily progress!")

    # Weekly goals
    steps_goal = st.number_input("Daily Steps Goal", 1000, 50000, 10000)
    exercise_goal = st.number_input("Daily Exercise Minutes Goal", 0, 180, 30)
    water_goal = st.number_input("Daily Water Intake (ml)", 500, 5000, 2000)

    # Initialize session state for weekly log
    if "weekly_log" not in st.session_state:
        st.session_state.weekly_log = pd.DataFrame(
            columns=["Date", "Steps", "Exercise", "Water Intake"]
        )

    # Log today's data
    st.subheader("Log Today's Progress")
    steps_done = st.number_input("Steps Today", 0, 50000, 0)
    exercise_done = st.number_input("Exercise Minutes Today", 0, 180, 0)
    water_done = st.number_input("Water Intake Today (ml)", 0, 5000, 0)
    if st.button("Add Today's Log"):
        today = datetime.today().strftime("%Y-%m-%d")
        st.session_state.weekly_log = pd.concat([
            st.session_state.weekly_log,
            pd.DataFrame([[today, steps_done, exercise_done, water_done]], columns=["Date", "Steps", "Exercise", "Water Intake"])
        ], ignore_index=True)
        st.success(f"✅ Progress for {today} added!")

    # Display weekly calendar table
    if not st.session_state.weekly_log.empty:
        st.subheader("Weekly Progress Calendar")
        # Show last 7 days
        last_7_days = datetime.today() - timedelta(days=6)
        weekly_data = st.session_state.weekly_log[
            pd.to_datetime(st.session_state.weekly_log["Date"]) >= last_7_days
        ].copy()
        weekly_data.set_index("Date", inplace=True)
        st.dataframe(weekly_data)

        # Show progress percentages
        weekly_data_pct = weekly_data.copy()
        weekly_data_pct["Steps %"] = (weekly_data_pct["Steps"] / steps_goal * 100).clip(0,100)
        weekly_data_pct["Exercise %"] = (weekly_data_pct["Exercise"] / exercise_goal * 100).clip(0,100)
        weekly_data_pct["Water %"] = (weekly_data_pct["Water Intake"] / water_goal * 100).clip(0,100)

        fig = px.bar(
            weekly_data_pct.reset_index().melt(id_vars="Date", value_vars=["Steps %", "Exercise %", "Water %"]),
            x="Date", y="value", color="variable", barmode="group",
            labels={"value":"Progress %", "variable":"Metric"}
        )
        st.plotly_chart(fig, use_container_width=True)






